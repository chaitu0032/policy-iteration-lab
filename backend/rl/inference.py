"""Inference engine: roll out any stored policy in the real Gymnasium env from a chosen start state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .environments import decode_state, default_max_steps, is_success, make_env

MAX_BENCHMARK_EPISODES = 1000


@dataclass(frozen=True)
class Step:
    t: int
    state: int
    action: int
    reward: float
    next_state: int
    prob: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class Episode:
    start_state: int
    steps: tuple[Step, ...]
    total_return: float
    discounted_return: float
    success: bool
    terminated: bool
    truncated: bool


def _place_agent(env, start_state: int | None, seed: int | None) -> int:
    """Reset through gymnasium (seeds the RNG / wrappers), then teleport to ``start_state``."""
    state, _ = env.reset(seed=seed)
    if start_state is None:
        return int(state)
    if not 0 <= start_state < env.observation_space.n:
        raise ValueError(f"start_state {start_state} out of range [0, {env.observation_space.n})")
    env.unwrapped.s = int(start_state)
    env.unwrapped.lastaction = None
    return int(start_state)


def run_episode(key: str, options: dict[str, Any], policy: Sequence[int], *,
                start_state: int | None = None, seed: int | None = None,
                max_steps: int | None = None, gamma: float = 1.0, env=None) -> Episode:
    """Play one greedy episode with a deterministic ``policy`` (state -> action)."""
    owns_env = env is None
    env = env or make_env(key, options, max_steps or default_max_steps(key, options))
    try:
        state = _place_agent(env, start_state, seed)
        start = state
        steps: list[Step] = []
        total, discounted, success = 0.0, 0.0, False
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(policy[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)
            discounted += (gamma ** len(steps)) * reward
            total += reward
            steps.append(Step(len(steps), state, action, reward, int(next_state),
                              float(info.get("prob", 1.0)), bool(terminated), bool(truncated)))
            success = success or is_success(key, env, int(next_state), terminated)
            state = int(next_state)
        return Episode(start, tuple(steps), total, discounted, success, terminated, truncated)
    finally:
        if owns_env:
            env.close()


def episode_to_dict(key: str, options: dict[str, Any], episode: Episode,
                    action_names: Sequence[str]) -> dict[str, Any]:
    env = make_env(key, options)
    try:
        return {
            "start_state": episode.start_state,
            "start_decoded": decode_state(key, env, episode.start_state),
            "total_return": episode.total_return,
            "discounted_return": episode.discounted_return,
            "success": episode.success,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
            "length": len(episode.steps),
            "steps": [
                {
                    "t": s.t,
                    "state": s.state,
                    "action": s.action,
                    "action_name": action_names[s.action],
                    "reward": s.reward,
                    "next_state": s.next_state,
                    "decoded": decode_state(key, env, s.next_state),
                    "prob": s.prob,
                    "terminated": s.terminated,
                    "truncated": s.truncated,
                }
                for s in episode.steps
            ],
        }
    finally:
        env.close()


def benchmark_policies(key: str, options: dict[str, Any], policies: Sequence[Sequence[int]], *,
                       episodes: int, start_state: int | None, seed: int,
                       max_steps: int | None, gamma: float) -> list[dict[str, float]]:
    """Monte-Carlo estimate of each policy's performance with common random seeds."""
    if not 1 <= episodes <= MAX_BENCHMARK_EPISODES:
        raise ValueError(f"episodes must be in [1, {MAX_BENCHMARK_EPISODES}]")
    env = make_env(key, options, max_steps or default_max_steps(key, options))
    try:
        results = []
        for index, policy in enumerate(policies):
            runs = [
                run_episode(key, options, policy, start_state=start_state, seed=seed + i,
                            gamma=gamma, env=env)
                for i in range(episodes)
            ]
            returns = np.array([r.total_return for r in runs])
            results.append({
                "iteration": index,
                "mean_return": float(returns.mean()),
                "std_return": float(returns.std()),
                "mean_discounted_return": float(np.mean([r.discounted_return for r in runs])),
                "success_rate": float(np.mean([r.success for r in runs])),
                "mean_length": float(np.mean([len(r.steps) for r in runs])),
                "truncation_rate": float(np.mean([r.truncated for r in runs])),
            })
        return results
    finally:
        env.close()
