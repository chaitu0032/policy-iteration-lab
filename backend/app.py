"""Stateless FastAPI backend (runs locally with uvicorn and on Vercel serverless functions).

The browser owns all run state: it drives policy iteration one step at a time (/api/pi/step),
stores every policy + value function in IndexedDB, and sends a policy along with each
inference / benchmark request.
"""

from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .rl.environments import (
    SPECS,
    TabularModel,
    describe_layout,
    extract_model,
    get_spec,
    normalize_options,
    option_schema_with_sources,
)
from .rl.inference import MAX_BENCHMARK_EPISODES, benchmark_policies, episode_to_dict, run_episode
from .rl.policy_iteration import PIConfig, initial_policy, policy_iteration, policy_iteration_step
from .rl.serialize import env_header, iteration_to_dict

logger = logging.getLogger("pi_lab")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_STEPS_LIMIT = 1000
EVAL_TIME_BUDGET_S = 8.0  # keep each request well inside serverless time limits
MODEL_CACHE_SIZE = 16

app = FastAPI(title="Policy Iteration Lab", version="2.0.0")

EnvKey = Literal["FrozenLake", "CliffWalking", "Taxi"]


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class EnvRequest(BaseModel):
    env_key: EnvKey
    options: dict[str, Any] = Field(default_factory=dict)


class PIParams(BaseModel):
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    theta: float = Field(1e-8, gt=0.0, le=1.0)
    max_iterations: int = Field(100, ge=1, le=500)
    max_eval_sweeps: int = Field(10_000, ge=1, le=100_000)
    eval_mode: Literal["iterative", "exact"] = "iterative"
    init_policy: Literal["zeros", "random"] = "zeros"
    seed: int = Field(0, ge=0, le=2**31 - 1)


class TrainRequest(EnvRequest, PIParams):
    pass


class StepRequest(EnvRequest):
    config: PIParams = Field(default_factory=PIParams)
    index: int = Field(0, ge=0, le=500)
    policy: list[int] | None = None  # pi_k; None -> initial policy from config
    previous_policy: list[int] | None = None  # pi_{k-1}, to report which states changed


class InferRequest(EnvRequest):
    policy: list[int]
    gamma: float = Field(1.0, ge=0.0, le=1.0)
    start_state: int | None = Field(None, ge=0)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    max_steps: int | None = Field(None, ge=1, le=MAX_STEPS_LIMIT)


class BenchmarkRequest(EnvRequest):
    policies: list[list[int]] = Field(min_length=1, max_length=50)
    gamma: float = Field(1.0, ge=0.0, le=1.0)
    episodes: int = Field(50, ge=1, le=MAX_BENCHMARK_EPISODES)
    start_state: int | None = Field(None, ge=0)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    max_steps: int | None = Field(None, ge=1, le=MAX_STEPS_LIMIT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _options(req: EnvRequest) -> dict[str, Any]:
    try:
        return normalize_options(req.env_key, req.options)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _config(params: PIParams) -> PIConfig:
    try:
        return PIConfig(**params.model_dump(include=set(PIParams.model_fields)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@lru_cache(maxsize=MODEL_CACHE_SIZE)
def _cached(env_key: str, options_json: str) -> tuple[TabularModel, dict[str, Any]]:
    options = json.loads(options_json)
    return extract_model(env_key, options), describe_layout(env_key, options)


def _model_and_layout(env_key: str, options: dict[str, Any]) -> tuple[TabularModel, dict[str, Any]]:
    return _cached(env_key, json.dumps(options, sort_keys=True))


def _policy_array(policy: list[int], model: TabularModel, name: str = "policy") -> np.ndarray:
    arr = np.asarray(policy, dtype=np.int64)
    if arr.shape != (model.n_states,):
        raise HTTPException(status_code=422, detail=f"{name} must have {model.n_states} entries")
    if arr.min() < 0 or arr.max() >= model.n_actions:
        raise HTTPException(status_code=422, detail=f"{name} actions must be in [0, {model.n_actions - 1}]")
    return arr


def _check_start_state(layout: dict[str, Any], start_state: int | None) -> None:
    if start_state is not None and start_state not in set(layout["start_states"]):
        raise HTTPException(status_code=422,
                            detail=f"start_state {start_state} is not a valid start (terminal, cliff or out of range)")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/envs")
def list_envs() -> list[dict[str, Any]]:
    return [
        {
            "key": s.key, "gym_id": s.gym_id, "title": s.title, "description": s.description,
            "action_names": list(s.action_names), "option_schema": option_schema_with_sources(s.key),
            "default_max_steps": s.default_max_steps,
        }
        for s in SPECS.values()
    ]


@app.post("/api/layout")
def layout(req: EnvRequest) -> dict[str, Any]:
    options = _options(req)
    model, lay = _model_and_layout(req.env_key, options)
    return {"env": env_header(req.env_key, options, model.n_states, model.n_actions), "layout": lay}


@app.post("/api/pi/step")
def pi_step(req: StepRequest) -> dict[str, Any]:
    """Evaluate pi_k and greedily improve it. The client loops until ``stable`` (or the user stops)."""
    options = _options(req)
    cfg = _config(req.config)
    model, lay = _model_and_layout(req.env_key, options)
    policy = (initial_policy(model, cfg.init_policy, cfg.seed) if req.policy is None
              else _policy_array(req.policy, model))
    previous = None if req.previous_policy is None else _policy_array(req.previous_policy, model, "previous_policy")
    deadline = time.monotonic() + EVAL_TIME_BUDGET_S
    started = time.perf_counter()
    record, new_policy, stable = policy_iteration_step(model, cfg, policy, req.index, previous,
                                                       lambda: time.monotonic() > deadline)
    ev = record.evaluation
    return {
        "iteration": iteration_to_dict(record, lay),
        "next_policy": new_policy.astype(int).tolist(),
        "stable": stable,
        "time_budget_hit": not ev.converged and ev.sweeps < cfg.max_eval_sweeps,
        "seconds": time.perf_counter() - started,
    }


@app.post("/api/train")
def train(req: TrainRequest) -> dict[str, Any]:
    """One-shot training (all iterations in a single request) — handy for scripts and tests."""
    options = _options(req)
    cfg = _config(req)
    model, lay = _model_and_layout(req.env_key, options)
    started = time.perf_counter()
    result = policy_iteration(model, cfg)
    return {
        "env": env_header(req.env_key, options, model.n_states, model.n_actions),
        "layout": lay,
        "config": req.model_dump(include=set(PIParams.model_fields)),
        "converged": result.converged,
        "optimal_index": result.optimal.index,
        "train_seconds": time.perf_counter() - started,
        "iterations": [iteration_to_dict(r, lay) for r in result.iterations],
    }


@app.post("/api/infer")
def infer(req: InferRequest) -> dict[str, Any]:
    options = _options(req)
    model, lay = _model_and_layout(req.env_key, options)
    policy = _policy_array(req.policy, model)
    _check_start_state(lay, req.start_state)
    episode = run_episode(req.env_key, options, policy, start_state=req.start_state, seed=req.seed,
                          max_steps=req.max_steps, gamma=req.gamma)
    return episode_to_dict(req.env_key, options, episode, get_spec(req.env_key).action_names)


@app.post("/api/benchmark")
def benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    options = _options(req)
    model, lay = _model_and_layout(req.env_key, options)
    policies = [_policy_array(p, model) for p in req.policies]
    _check_start_state(lay, req.start_state)
    rows = benchmark_policies(req.env_key, options, policies, episodes=req.episodes,
                              start_state=req.start_state, seed=req.seed, max_steps=req.max_steps,
                              gamma=req.gamma)
    return {"episodes": req.episodes, "start_state": req.start_state, "rows": rows}


# Local dev: serve the frontend from the same origin (on Vercel the CDN serves it).
if FRONTEND_DIR.exists():
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
