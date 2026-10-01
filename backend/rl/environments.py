"""Builds Gymnasium toy-text envs and extracts their tabular MDP model + UI layout.

  * ``make_env(key, options)``         -> gymnasium env (with custom rewards patched into ``P``)
  * ``extract_model(key, options)``    -> TabularModel (T, R, C, terminal mask)
  * ``describe_layout(key, options)``  -> JSON-friendly layout for the JS renderers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from .env_specs import (  # noqa: F401  (re-exported for callers)
    CLIFF_WALKING,
    FROZEN_LAKE,
    SPECS,
    TAXI,
    EnvSpec,
    get_spec,
    gym_kwargs,
    normalize_options,
    option_schema_with_sources,
    with_custom_rewards,
)

TAXI_LOC_NAMES = ("R", "G", "Y", "B")
TAXI_IN_CAR = 4


@dataclass(frozen=True)
class TabularModel:
    """Sparse tabular MDP. For every (s, a) the K possible successors are stored as
    ``next_states[s, a, k]`` with probability ``probs[s, a, k]``; ``cont_probs`` keeps only the
    probability of *non-terminating* transitions (used for bootstrapping). ``R[s, a]`` is the
    expected immediate reward. Padding entries have probability 0."""

    n_states: int
    n_actions: int
    next_states: np.ndarray
    probs: np.ndarray
    cont_probs: np.ndarray
    R: np.ndarray
    terminal: np.ndarray

    def _dense(self, weights: np.ndarray) -> np.ndarray:
        out = np.zeros((self.n_states, self.n_actions, self.n_states))
        s_idx, a_idx, _ = np.indices(self.next_states.shape)
        np.add.at(out, (s_idx, a_idx, self.next_states), weights)
        return out

    @property
    def T(self) -> np.ndarray:
        """Dense transition tensor T[s, a, s'] (for tests / inspection)."""
        return self._dense(self.probs)

    @property
    def C(self) -> np.ndarray:
        """Dense continuation tensor C[s, a, s'] = P(s', not terminated | s, a)."""
        return self._dense(self.cont_probs)


def default_max_steps(key: str, options: dict[str, Any] | None) -> int:
    """Episode step limit used when the caller does not pass one (same for every world)."""
    normalize_options(key, options)  # validate even though the limit does not depend on them
    return get_spec(key).default_max_steps


def make_env(key: str, options: dict[str, Any] | None = None, max_steps: int | None = None) -> gym.Env:
    spec = get_spec(key)
    opts = normalize_options(key, options)
    steps = max_steps or default_max_steps(key, opts)
    env = gym.make(spec.gym_id, max_episode_steps=steps, **gym_kwargs(key, opts))
    # step() samples from P, so replacing P keeps the planner model and the real env consistent.
    env.unwrapped.P = with_custom_rewards(key, env.unwrapped.P, opts)
    return env


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------

def _frozen_lake_terminals(env: gym.Env) -> np.ndarray:
    desc = env.unwrapped.desc.flatten()
    return np.array([c in (b"H", b"G") for c in desc], dtype=bool)


def _cliff_terminals(env: gym.Env) -> np.ndarray:
    n = int(env.observation_space.n)
    term = np.zeros(n, dtype=bool)
    term[n - 1] = True  # bottom-right goal
    return term


def _taxi_terminals(env: gym.Env) -> np.ndarray:
    u = env.unwrapped
    n = int(env.observation_space.n)
    decoded = (tuple(u.decode(s)) for s in range(n))
    return np.array([pass_loc == dest for _, _, pass_loc, dest in decoded], dtype=bool)


_TERMINALS: dict[str, Callable[[gym.Env], np.ndarray]] = {
    FROZEN_LAKE: _frozen_lake_terminals,
    CLIFF_WALKING: _cliff_terminals,
    TAXI: _taxi_terminals,
}


def extract_model(key: str, options: dict[str, Any] | None = None) -> TabularModel:
    """Read ``env.unwrapped.P`` into padded sparse arrays. Terminal states become absorbing with 0 reward."""
    env = make_env(key, options)
    try:
        P = env.unwrapped.P
        n_s, n_a = int(env.observation_space.n), int(env.action_space.n)
        terminal = _TERMINALS[key](env)
        K = max(len(P[s][a]) for s in range(n_s) for a in range(n_a))
        next_states = np.tile(np.arange(n_s)[:, None, None], (1, n_a, K))  # pad: self-loop, prob 0
        probs = np.zeros((n_s, n_a, K))
        cont = np.zeros((n_s, n_a, K))
        R = np.zeros((n_s, n_a))
        for s in range(n_s):
            if terminal[s]:
                probs[s, :, 0] = 1.0
                continue
            for a in range(n_a):
                for k, (prob, s_next, reward, done) in enumerate(P[s][a]):
                    next_states[s, a, k] = s_next
                    probs[s, a, k] = prob
                    cont[s, a, k] = 0.0 if done else prob
                    R[s, a] += prob * reward
        return TabularModel(n_s, n_a, next_states, probs, cont, R, terminal)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Layouts + start states (consumed by the JS renderers)
# ---------------------------------------------------------------------------

def _grid_layout(key: str, env: gym.Env) -> dict[str, Any]:
    if key == FROZEN_LAKE:
        cells = [[c.decode() for c in row] for row in env.unwrapped.desc]
    else:
        rows, cols = env.unwrapped.shape
        cells = [["." for _ in range(cols)] for _ in range(rows)]
        cells[rows - 1][0] = "S"
        cells[rows - 1][cols - 1] = "G"
        for c in range(1, cols - 1):
            cells[rows - 1][c] = "C"
    return {"kind": "grid", "rows": len(cells), "cols": len(cells[0]), "cells": cells}


def _taxi_layout(env: gym.Env) -> dict[str, Any]:
    desc = env.unwrapped.desc
    # desc row r+1 holds grid row r; char at column 2*c+2 is the separator to the right of cell c.
    walls = [[r, c] for r in range(5) for c in range(4) if desc[r + 1][2 * c + 2] == b"|"]
    return {
        "kind": "taxi", "rows": 5, "cols": 5, "walls": walls,
        "desc": [row.tobytes().decode() for row in desc],  # 7x11 map, drawn tile-by-tile like gymnasium
        "locs": [list(loc) for loc in env.unwrapped.locs], "loc_names": list(TAXI_LOC_NAMES),
    }


def describe_layout(key: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    env = make_env(key, options)
    try:
        layout = _taxi_layout(env) if key == TAXI else _grid_layout(key, env)
        terminal = _TERMINALS[key](env)
        layout["terminal_states"] = [int(s) for s in np.flatnonzero(terminal)]
        layout["start_states"] = valid_start_states(key, env, terminal)
        layout["default_start"] = _default_start(key, env)
        return layout
    finally:
        env.close()


def valid_start_states(key: str, env: gym.Env, terminal: np.ndarray) -> list[int]:
    """States an episode may legally begin from (non-terminal, and not on the cliff)."""
    states = [s for s in range(int(env.observation_space.n)) if not terminal[s]]
    if key == CLIFF_WALKING:
        rows, cols = env.unwrapped.shape
        cliff = {(rows - 1) * cols + c for c in range(1, cols - 1)}
        states = [s for s in states if s not in cliff]
    return [int(s) for s in states]


def _default_start(key: str, env: gym.Env) -> int:
    if key == CLIFF_WALKING:
        return int(env.unwrapped.start_state_index)
    if key == FROZEN_LAKE:
        flat = env.unwrapped.desc.flatten()
        return int(np.flatnonzero(flat == b"S")[0])
    return int(env.unwrapped.encode(0, 0, 0, 1))


def is_success(key: str, env: gym.Env, next_state: int, terminated: bool) -> bool:
    """Whether a terminating transition reached the goal (independent of the reward settings)."""
    if not terminated:
        return False
    if key == FROZEN_LAKE:
        return env.unwrapped.desc.flatten()[next_state] == b"G"
    return True  # CliffWalking only terminates at the goal; Taxi only on a correct dropoff.


def decode_state(key: str, env: gym.Env, state: int) -> dict[str, int]:
    if key == TAXI:
        row, col, pass_loc, dest = env.unwrapped.decode(state)
        return {"row": int(row), "col": int(col), "passenger": int(pass_loc), "destination": int(dest)}
    cols = env.unwrapped.ncol if key == FROZEN_LAKE else env.unwrapped.shape[1]
    return {"row": state // cols, "col": state % cols}
