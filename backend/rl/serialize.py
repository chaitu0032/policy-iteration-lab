"""JSON serialisation of policy-iteration records (policy, V, Q, evaluation trace) and env headers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .environments import default_max_steps, get_spec
from .policy_iteration import IterationRecord

VALUE_DECIMALS = 6


def _round(values: np.ndarray) -> list:
    return np.round(values.astype(float), VALUE_DECIMALS).tolist()


def iteration_to_dict(rec: IterationRecord, layout: dict[str, Any]) -> dict[str, Any]:
    ev = rec.evaluation
    startable = layout["start_states"]
    return {
        "index": rec.index,
        "policy": rec.policy.astype(int).tolist(),
        "V": _round(rec.V),
        "Q": _round(rec.Q),
        "changed_states": list(rec.changed_states),
        "eval_sweeps": ev.sweeps,
        "eval_converged": ev.converged,
        "eval_deltas": [float(d) for d in ev.deltas],
        "eval_snapshots": [{"sweep": i, "V": _round(v)} for i, v in ev.snapshots],
        "stats": {
            "v_start": float(rec.V[layout["default_start"]]),
            "v_mean": float(np.mean(rec.V[startable])),
            "v_max": float(np.max(rec.V)),
            "v_min": float(np.min(rec.V)),
            "n_changed": len(rec.changed_states),
        },
    }


def env_header(env_key: str, options: dict[str, Any], n_states: int, n_actions: int) -> dict[str, Any]:
    spec = get_spec(env_key)
    return {
        "key": spec.key,
        "gym_id": spec.gym_id,
        "title": spec.title,
        "options": options,
        "action_names": list(spec.action_names),
        "action_vectors": [list(v) if v else None for v in spec.action_vectors],
        "default_max_steps": default_max_steps(env_key, options),
        "n_states": n_states,
        "n_actions": n_actions,
    }
