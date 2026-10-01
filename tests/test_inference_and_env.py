import pytest

from backend.rl.environments import (
    CLIFF_WALKING,
    FROZEN_LAKE,
    TAXI,
    describe_layout,
    extract_model,
    normalize_options,
)
from backend.rl.inference import benchmark_policies, episode_to_dict, run_episode
from backend.rl.policy_iteration import PIConfig, policy_iteration

DET_LAKE = {"map_name": "4x4", "is_slippery": False}


def _optimal(key, options, gamma=0.99):
    return policy_iteration(extract_model(key, options), PIConfig(gamma=gamma, eval_mode="exact")).optimal.policy


def test_normalize_options_fills_defaults_and_rejects_bad_values():
    defaults = normalize_options(FROZEN_LAKE, {})
    assert defaults["map_name"] == "4x4" and defaults["is_slippery"] is True
    with pytest.raises(ValueError):
        normalize_options(FROZEN_LAKE, {"map_name": "random", "size": 2.5})
    with pytest.raises(ValueError):
        normalize_options(TAXI, {"is_rainy": True, "rainy_probability": True})
    with pytest.raises(ValueError):
        normalize_options(FROZEN_LAKE, {"map_name": "9x9"})
    with pytest.raises(ValueError):
        normalize_options(FROZEN_LAKE, {"nope": 1})
    with pytest.raises(ValueError):
        normalize_options(FROZEN_LAKE, {"is_slippery": "yes"})
    with pytest.raises(ValueError):
        normalize_options("Pong", {})


def test_layouts():
    lake = describe_layout(FROZEN_LAKE, DET_LAKE)
    assert lake["kind"] == "grid" and lake["rows"] == 4 and lake["cells"][0][0] == "S"
    assert 5 not in lake["start_states"]  # hole
    cliff = describe_layout(CLIFF_WALKING, {})
    assert cliff["default_start"] == 36 and 40 not in cliff["start_states"]
    taxi = describe_layout(TAXI, {})
    assert taxi["kind"] == "taxi" and len(taxi["start_states"]) == 400
    assert [0, 1] in taxi["walls"]  # wall to the right of cell (0, 1)
    assert taxi["locs"][0] == [0, 0]


def test_frozen_lake_episode_from_custom_start_reaches_goal():
    policy = _optimal(FROZEN_LAKE, DET_LAKE)
    ep = run_episode(FROZEN_LAKE, DET_LAKE, policy, start_state=4, seed=1)
    assert ep.start_state == 4 and ep.steps[0].state == 4
    assert ep.success and ep.terminated and ep.total_return == 1.0
    assert ep.steps[-1].next_state == 15


def test_cliff_episode_takes_13_steps():
    policy = _optimal(CLIFF_WALKING, {}, gamma=1.0)
    ep = run_episode(CLIFF_WALKING, {}, policy, start_state=36, seed=0)
    assert len(ep.steps) == 13 and ep.total_return == -13 and ep.success


def test_taxi_episode_delivers_passenger():
    policy = _optimal(TAXI, {})
    ep = run_episode(TAXI, {}, policy, start_state=328, seed=0)
    assert ep.success and ep.steps[-1].reward == 20


def test_truncation_when_policy_never_finishes():
    ep = run_episode(CLIFF_WALKING, {}, [0] * 48, start_state=36, seed=0, max_steps=7)
    assert ep.truncated and not ep.success and len(ep.steps) == 7


def test_episode_rejects_out_of_range_start():
    with pytest.raises(ValueError):
        run_episode(FROZEN_LAKE, DET_LAKE, [0] * 16, start_state=99)


def test_episode_to_dict_has_decoded_states():
    policy = _optimal(TAXI, {})
    ep = run_episode(TAXI, {}, policy, start_state=328, seed=0)
    d = episode_to_dict(TAXI, {}, ep, ["S", "N", "E", "W", "P", "D"])
    assert d["start_decoded"] == {"row": 3, "col": 1, "passenger": 2, "destination": 0}
    assert d["length"] == len(d["steps"]) and "action_name" in d["steps"][0]


def test_benchmark_optimal_beats_initial():
    model = extract_model(FROZEN_LAKE, {"is_slippery": True})
    result = policy_iteration(model, PIConfig(gamma=0.99))
    policies = [r.policy for r in result.iterations]
    rows = benchmark_policies(FROZEN_LAKE, {"is_slippery": True}, [policies[0], policies[-1]],
                              episodes=200, start_state=0, seed=0, max_steps=100, gamma=0.99)
    assert rows[1]["success_rate"] > rows[0]["success_rate"]
    assert rows[1]["success_rate"] > 0.6
    with pytest.raises(ValueError):
        benchmark_policies(FROZEN_LAKE, {}, policies, episodes=0, start_state=None, seed=0,
                           max_steps=None, gamma=0.9)
