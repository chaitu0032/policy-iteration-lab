import inspect

import gymnasium as gym
import pytest

from backend.rl.env_specs import (
    CLIFF_WALKING,
    FROZEN_LAKE,
    GYM_ARG_OF_OPTION,
    SPECS,
    TAXI,
    normalize_options,
    option_schema_with_sources,
    parse_custom_map,
)
from backend.rl.environments import describe_layout, extract_model, make_env
from backend.rl.inference import run_episode
from backend.rl.policy_iteration import PIConfig, policy_iteration

IGNORED_CTOR_ARGS = {"self", "render_mode"}


@pytest.mark.parametrize("key", list(SPECS))
def test_every_native_gym_argument_is_exposed(key):
    """Fails if gymnasium adds a constructor argument that the UI does not expose."""
    env_cls = type(gym.make(SPECS[key].gym_id).unwrapped)
    ctor = set(inspect.signature(env_cls.__init__).parameters) - IGNORED_CTOR_ARGS
    exposed = {arg.split("[")[0].split(" ")[0] for arg in GYM_ARG_OF_OPTION[key].values() if arg}
    assert ctor <= exposed, f"not exposed: {ctor - exposed}"
    assert set(GYM_ARG_OF_OPTION[key]) == set(SPECS[key].option_schema)


def test_schema_sources():
    cliff = option_schema_with_sources(CLIFF_WALKING)
    assert cliff["is_slippery"]["source"] == "gymnasium"
    assert cliff["reward_cliff"]["source"] == "lab"
    assert option_schema_with_sources(FROZEN_LAKE)["reward_goal"]["gym_arg"] == "reward_schedule[0]"


def test_hidden_options_are_dropped():
    opts = normalize_options(FROZEN_LAKE, {"map_name": "4x4", "size": 7})
    assert "size" not in opts and "custom_map" not in opts
    assert normalize_options(TAXI, {})["is_rainy"] is False and "rainy_probability" not in normalize_options(TAXI, {})


@pytest.mark.parametrize("text, error", [("", "empty"), ("SF\nFFF", "same length"), ("SX\nFG", "only contain"),
                                         ("FF\nFG", "exactly one S"), ("SF\nFF", "at least one G")])
def test_custom_map_validation(text, error):
    with pytest.raises(ValueError, match=error):
        parse_custom_map(text)


def test_custom_map_layout_and_start():
    opts = {"map_name": "custom", "custom_map": "FFS\nFHF\nGFF", "is_slippery": False}
    layout = describe_layout(FROZEN_LAKE, opts)
    assert layout["rows"] == 3 and layout["default_start"] == 2


def test_frozen_lake_reward_schedule_and_success_rate():
    opts = {"map_name": "4x4", "is_slippery": True, "success_rate": 0.9, "reward_goal": 10, "reward_hole": -5,
            "reward_frozen": -0.1}
    P = make_env(FROZEN_LAKE, opts).unwrapped.P
    probs = sorted(p for p, *_ in P[14][2])
    assert probs[-1] == pytest.approx(0.9)
    assert {r for _, s2, r, _ in P[14][2] if s2 == 15} == {10}
    assert {r for _, s2, r, _ in P[0][1]} <= {-0.1, -5}


def test_cliff_custom_rewards_change_optimal_return():
    opts = {"reward_step": -2.0, "reward_cliff": -50.0, "reward_goal": 100.0}
    P = make_env(CLIFF_WALKING, opts).unwrapped.P
    assert P[36][1][0][2] == -50.0 and P[35][2][0][2] == 100.0 and P[0][0][0][2] == -2.0
    policy = policy_iteration(extract_model(CLIFF_WALKING, opts), PIConfig(gamma=1.0, eval_mode="exact",
                                                                            max_eval_sweeps=300)).optimal.policy
    ep = run_episode(CLIFF_WALKING, opts, policy, start_state=36, seed=0)
    assert ep.success and ep.total_return == pytest.approx(12 * -2.0 + 100.0)


def test_taxi_custom_rewards_and_fickle_passenger():
    opts = {"reward_step": -0.5, "reward_illegal": -3.0, "reward_dropoff": 7.0,
            "fickle_passenger": True, "fickle_probability": 1.0}
    rewards = {r for acts in make_env(TAXI, opts).unwrapped.P.values() for ts in acts.values() for _, _, r, _ in ts}
    assert rewards == {-0.5, -3.0, 7.0}
    policy = policy_iteration(extract_model(TAXI, opts), PIConfig(eval_mode="exact")).optimal.policy
    ep = run_episode(TAXI, opts, policy, start_state=328, seed=0, max_steps=200)
    assert len(ep.steps) > 0  # fickle dynamics are outside P; episode must still run cleanly


def test_frozen_lake_default_is_8x8_shortest_path():
    """Default config (8x8, deterministic, gamma=0.99) yields the 14-step shortest path."""
    policy = policy_iteration(extract_model(FROZEN_LAKE, {}), PIConfig(gamma=0.99)).optimal.policy
    ep = run_episode(FROZEN_LAKE, {}, policy, start_state=0, seed=0)
    assert ep.success and len(ep.steps) == 14
