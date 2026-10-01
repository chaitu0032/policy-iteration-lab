"""Environment specs + every configurable parameter (Gymnasium constructor args and reward overrides).

Option schema entries: {type: bool|int|float|select|map, default, label, group, min/max/step/choices,
visible_if: {other_option: value}, hint}. ``group`` is "dynamics" or "rewards" (UI fieldsets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

FROZEN_LAKE = "FrozenLake"
CLIFF_WALKING = "CliffWalking"
TAXI = "Taxi"

REWARD_LIMIT = 1000.0
MAX_CUSTOM_MAP_SIDE = 16
DEFAULT_CUSTOM_MAP = "SFFF\nFHFH\nFFFH\nHFFG"
DEFAULT_MAX_STEPS = 1000  # episode step limit (gymnasium TimeLimit) for inference / benchmark

# gymnasium's built-in reward constants that we remap for envs without a reward argument
CLIFF_NATIVE_FALL = -100
TAXI_NATIVE_DROPOFF = 20
TAXI_NATIVE_ILLEGAL = -10


@dataclass(frozen=True)
class EnvSpec:
    key: str
    gym_id: str
    title: str
    description: str
    action_names: tuple[str, ...]
    # (d_row, d_col) per action, None for non-movement actions (Taxi pickup/dropoff).
    action_vectors: tuple[tuple[int, int] | None, ...]
    option_schema: dict[str, dict[str, Any]]
    default_max_steps: int


def _reward(label: str, default: float, hint: str = "") -> dict[str, Any]:
    return {"type": "float", "min": -REWARD_LIMIT, "max": REWARD_LIMIT, "step": 0.1, "default": default,
            "label": label, "group": "rewards", "hint": hint}


def _taxi_gym_id() -> str:
    return "Taxi-v4" if "Taxi-v4" in gym.registry else "Taxi-v3"


_FROZEN_RANDOM = {"map_name": "random"}

SPECS: dict[str, EnvSpec] = {
    FROZEN_LAKE: EnvSpec(
        key=FROZEN_LAKE, gym_id="FrozenLake-v1", title="Frozen Lake",
        description="Cross the frozen lake from S to G without falling into holes (H).",
        action_names=("Left", "Down", "Right", "Up"),
        action_vectors=((0, -1), (1, 0), (0, 1), (-1, 0)),
        option_schema={
            "map_name": {"type": "select", "choices": ["4x4", "8x8", "random", "custom"], "default": "8x8",
                         "label": "Map", "group": "dynamics"},
            "custom_map": {"type": "map", "default": DEFAULT_CUSTOM_MAP, "label": "Custom map (desc)",
                           "group": "dynamics", "visible_if": {"map_name": "custom"},
                           "hint": "Rows of S (start), F (frozen), H (hole), G (goal). One S, at least one G."},
            "size": {"type": "int", "min": 3, "max": MAX_CUSTOM_MAP_SIDE, "default": 6, "label": "Random map size",
                     "group": "dynamics", "visible_if": _FROZEN_RANDOM},
            "frozen_prob": {"type": "float", "min": 0.5, "max": 0.98, "step": 0.01, "default": 0.8,
                            "label": "Frozen-tile probability", "group": "dynamics", "visible_if": _FROZEN_RANDOM},
            "map_seed": {"type": "int", "min": 0, "max": 1_000_000, "default": 42, "label": "Map seed",
                         "group": "dynamics", "visible_if": _FROZEN_RANDOM},
            # Default: deterministic ice, so with gamma < 1 the optimal policy is the shortest safe path.
            "is_slippery": {"type": "bool", "default": False, "label": "Slippery ice", "group": "dynamics"},
            "success_rate": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "default": round(1 / 3, 4),
                             "label": "Success rate", "group": "dynamics", "visible_if": {"is_slippery": True},
                             "hint": "P(intended move); the rest is split between the two perpendicular moves."},
            "reward_goal": _reward("Reward: reach goal", 1.0),
            "reward_hole": _reward("Reward: fall in hole", 0.0),
            "reward_frozen": _reward("Reward: frozen step", 0.0, "Negative = step penalty (encourages short paths)"),
        },
        default_max_steps=DEFAULT_MAX_STEPS,
    ),
    CLIFF_WALKING: EnvSpec(
        key=CLIFF_WALKING, gym_id="CliffWalking-v1", title="Cliff Walking",
        description="Walk from bottom-left to bottom-right. Falling off the cliff is penalised and resets to start.",
        action_names=("Up", "Right", "Down", "Left"),
        action_vectors=((-1, 0), (0, 1), (1, 0), (0, -1)),
        option_schema={
            "is_slippery": {"type": "bool", "default": False, "label": "Slippery", "group": "dynamics",
                            "hint": "gymnasium: 1/3 intended move, 1/3 each perpendicular move"},
            "reward_step": _reward("Reward: step", -1.0),
            "reward_cliff": _reward("Reward: fall off cliff", -100.0),
            "reward_goal": _reward("Reward: step into goal", -1.0),
        },
        default_max_steps=DEFAULT_MAX_STEPS,
    ),
    TAXI: EnvSpec(
        key=TAXI, gym_id=_taxi_gym_id(), title="Taxi",
        description="Pick up the passenger and drop them at the destination (R, G, Y, B).",
        action_names=("South", "North", "East", "West", "Pickup", "Dropoff"),
        action_vectors=((1, 0), (-1, 0), (0, 1), (0, -1), None, None),
        option_schema={
            "is_rainy": {"type": "bool", "default": False, "label": "Rainy (stochastic moves)", "group": "dynamics"},
            "rainy_probability": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.8,
                                  "label": "Rain: P(intended move)", "group": "dynamics",
                                  "visible_if": {"is_rainy": True}},
            "fickle_passenger": {"type": "bool", "default": False, "label": "Fickle passenger", "group": "dynamics",
                                 "hint": "Applied by gymnasium's step(), not in P: the planner cannot see it."},
            "fickle_probability": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.3,
                                   "label": "Fickle probability", "group": "dynamics",
                                   "visible_if": {"fickle_passenger": True}},
            "reward_step": _reward("Reward: step", -1.0),
            "reward_illegal": _reward("Reward: illegal pickup/dropoff", -10.0),
            "reward_dropoff": _reward("Reward: successful dropoff", 20.0),
        },
        default_max_steps=DEFAULT_MAX_STEPS,
    ),
}


# Which gymnasium constructor argument each UI option feeds. None = lab extension (not a gym argument).
GYM_ARG_OF_OPTION: dict[str, dict[str, str | None]] = {
    FROZEN_LAKE: {
        "map_name": "map_name", "custom_map": "desc", "size": "desc (generate_random_map)",
        "frozen_prob": "desc (generate_random_map)", "map_seed": "desc (generate_random_map)",
        "is_slippery": "is_slippery", "success_rate": "success_rate",
        "reward_goal": "reward_schedule[0]", "reward_hole": "reward_schedule[1]",
        "reward_frozen": "reward_schedule[2]",
    },
    CLIFF_WALKING: {"is_slippery": "is_slippery", "reward_step": None, "reward_cliff": None, "reward_goal": None},
    TAXI: {
        "is_rainy": "is_rainy", "rainy_probability": "rainy_probability",
        "fickle_passenger": "fickle_passenger", "fickle_probability": "fickle_probability",
        "reward_step": None, "reward_illegal": None, "reward_dropoff": None,
    },
}


def option_schema_with_sources(key: str) -> dict[str, dict[str, Any]]:
    """Schema annotated with ``gym_arg`` / ``source`` so the UI can show what gymnasium natively allows."""
    mapping = GYM_ARG_OF_OPTION[key]
    return {
        name: {**schema, "gym_arg": mapping[name], "source": "gymnasium" if mapping[name] else "lab"}
        for name, schema in get_spec(key).option_schema.items()
    }


def get_spec(key: str) -> EnvSpec:
    if key not in SPECS:
        raise ValueError(f"Unknown environment '{key}'. Choose one of: {', '.join(SPECS)}")
    return SPECS[key]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def parse_custom_map(text: str) -> list[str]:
    rows = [r.strip().upper() for r in str(text).replace(",", "\n").splitlines() if r.strip()]
    if not rows:
        raise ValueError("Custom map is empty")
    if len(rows) > MAX_CUSTOM_MAP_SIDE or any(len(r) > MAX_CUSTOM_MAP_SIDE for r in rows):
        raise ValueError(f"Custom map must be at most {MAX_CUSTOM_MAP_SIDE}x{MAX_CUSTOM_MAP_SIDE}")
    if len({len(r) for r in rows}) != 1:
        raise ValueError("Custom map rows must all have the same length")
    joined = "".join(rows)
    if set(joined) - set("SFHG"):
        raise ValueError("Custom map may only contain S, F, H, G")
    if joined.count("S") != 1:
        raise ValueError("Custom map needs exactly one S")
    if "G" not in joined:
        raise ValueError("Custom map needs at least one G")
    return rows


def _validate_option(name: str, value: Any, schema: dict[str, Any]) -> Any:
    kind = schema["type"]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"Option '{name}' must be a boolean")
        return value
    if kind == "select":
        if value not in schema["choices"]:
            raise ValueError(f"Option '{name}' must be one of {schema['choices']}")
        return value
    if kind == "map":
        if not isinstance(value, str):
            raise ValueError(f"Option '{name}' must be a string")
        return "\n".join(parse_custom_map(value))
    is_int = kind == "int"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (is_int and int(value) != value):
        raise ValueError(f"Option '{name}' must be {'an integer' if is_int else 'a number'}")
    if not schema["min"] <= value <= schema["max"]:
        raise ValueError(f"Option '{name}' must be in [{schema['min']}, {schema['max']}]")
    return int(value) if is_int else float(value)


def _is_visible(schema: dict[str, Any], values: dict[str, Any]) -> bool:
    return all(values.get(k) == v for k, v in schema.get("visible_if", {}).items())


def normalize_options(key: str, options: dict[str, Any] | None) -> dict[str, Any]:
    """Validate options against the schema and fill defaults. Hidden (inactive) options are dropped."""
    spec = get_spec(key)
    raw = dict(options or {})
    unknown = set(raw) - set(spec.option_schema)
    if unknown:
        raise ValueError(f"Unknown option(s) for {key}: {', '.join(sorted(unknown))}")
    controllers = {n: raw.get(n, s["default"]) for n, s in spec.option_schema.items() if "visible_if" not in s}
    normalized = {
        name: _validate_option(name, raw.get(name, schema["default"]), schema)
        for name, schema in spec.option_schema.items()
        if _is_visible(schema, controllers)
    }
    if key == TAXI and spec.gym_id == "Taxi-v3" and (normalized["is_rainy"] or normalized["fickle_passenger"]):
        raise ValueError("Rainy / fickle taxi requires gymnasium >= 1.2 (Taxi-v4)")
    return normalized


# ---------------------------------------------------------------------------
# Translation to gymnasium
# ---------------------------------------------------------------------------

def frozen_lake_desc(opts: dict[str, Any]) -> list[str] | None:
    if opts["map_name"] == "random":
        return generate_random_map(size=opts["size"], p=opts["frozen_prob"], seed=opts["map_seed"])
    if opts["map_name"] == "custom":
        return parse_custom_map(opts["custom_map"])
    return None


def gym_kwargs(key: str, opts: dict[str, Any]) -> dict[str, Any]:
    """UI options -> gymnasium.make keyword arguments (native parameters only)."""
    if key == FROZEN_LAKE:
        kwargs: dict[str, Any] = {
            "is_slippery": opts["is_slippery"],
            "reward_schedule": (opts["reward_goal"], opts["reward_hole"], opts["reward_frozen"]),
        }
        if opts["is_slippery"]:
            kwargs["success_rate"] = opts["success_rate"]
        desc = frozen_lake_desc(opts)
        if desc is None:
            kwargs["map_name"] = opts["map_name"]
        else:
            kwargs["desc"] = desc
        return kwargs
    if key == CLIFF_WALKING:
        return {"is_slippery": opts["is_slippery"]}
    kwargs = {"is_rainy": opts["is_rainy"], "fickle_passenger": opts["fickle_passenger"]}
    if opts["is_rainy"]:
        kwargs["rainy_probability"] = opts["rainy_probability"]
    if opts["fickle_passenger"]:
        kwargs["fickle_probability"] = opts["fickle_probability"]
    return {k: v for k, v in kwargs.items() if v is not False} if _taxi_gym_id() == "Taxi-v3" else kwargs


def _cliff_reward(opts: dict[str, Any], reward: float, done: bool) -> float:
    if reward == CLIFF_NATIVE_FALL:
        return opts["reward_cliff"]
    return opts["reward_goal"] if done else opts["reward_step"]


def _taxi_reward(opts: dict[str, Any], reward: float, done: bool) -> float:
    if reward == TAXI_NATIVE_DROPOFF:
        return opts["reward_dropoff"]
    if reward == TAXI_NATIVE_ILLEGAL:
        return opts["reward_illegal"]
    return opts["reward_step"]


def with_custom_rewards(key: str, P: dict, opts: dict[str, Any]) -> dict:
    """Return a new transition table with rewards remapped (CliffWalking / Taxi have no reward args)."""
    if key == FROZEN_LAKE:
        return P
    remap = _cliff_reward if key == CLIFF_WALKING else _taxi_reward
    return {
        s: {a: [(p, s2, remap(opts, r, d), d) for p, s2, r, d in transitions] for a, transitions in actions.items()}
        for s, actions in P.items()
    }
