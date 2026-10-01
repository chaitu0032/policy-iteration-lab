import pytest
from fastapi.testclient import TestClient

from backend.app import app

DET_LAKE = {"env_key": "FrozenLake", "options": {"map_name": "4x4", "is_slippery": False,
                                                    "reward_hole": 0.0, "reward_frozen": 0.0}}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _drive_steps(client, env, config, max_steps=100):
    """Mimic the browser: call /api/pi/step until the policy is stable."""
    iterations, policy, previous = [], None, None
    for k in range(max_steps):
        res = client.post("/api/pi/step", json={**env, "config": config, "index": k,
                                                "policy": policy, "previous_policy": previous})
        assert res.status_code == 200, res.text
        body = res.json()
        iterations.append(body["iteration"])
        if body["stable"]:
            return iterations
        previous, policy = body["iteration"]["policy"], body["next_policy"]
    raise AssertionError("did not converge")


def test_health_envs_and_frontend(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    envs = client.get("/api/envs").json()
    assert [e["key"] for e in envs] == ["FrozenLake", "CliffWalking", "Taxi"]
    assert envs[1]["option_schema"]["reward_cliff"]["source"] == "lab"
    assert client.get("/").status_code == 200


def test_layout_endpoint(client):
    res = client.post("/api/layout", json={"env_key": "FrozenLake", "options": {"map_name": "8x8"}}).json()
    assert res["layout"]["rows"] == 8 and res["env"]["n_states"] == 64
    bad = client.post("/api/layout", json={"env_key": "FrozenLake", "options": {"map_name": "2x2"}})
    assert bad.status_code == 422


def test_stepwise_training_matches_one_shot(client):
    config = {"gamma": 0.9}
    steps = _drive_steps(client, DET_LAKE, config)
    full = client.post("/api/train", json={**DET_LAKE, **config}).json()
    assert len(steps) == len(full["iterations"])
    assert steps[-1]["policy"] == full["iterations"][-1]["policy"]
    assert steps[-1]["V"] == full["iterations"][-1]["V"]
    assert steps[0]["changed_states"] == [] and steps[1]["changed_states"]
    it = steps[0]
    assert {"policy", "V", "Q", "eval_snapshots", "eval_deltas", "stats"} <= set(it)


@pytest.mark.parametrize("env", [
    {"env_key": "CliffWalking", "options": {"reward_cliff": -30.0}},
    {"env_key": "Taxi", "options": {"is_rainy": True, "rainy_probability": 0.9}},
    {"env_key": "FrozenLake", "options": {"map_name": "custom", "custom_map": "SFH\nFFF\nHFG"}},
])
def test_all_envs_train_stepwise(client, env):
    steps = _drive_steps(client, env, {"gamma": 0.95, "eval_mode": "exact"})
    assert len(steps) >= 2


def test_step_validation(client):
    bad_len = client.post("/api/pi/step", json={**DET_LAKE, "policy": [0, 1]})
    assert bad_len.status_code == 422
    bad_action = client.post("/api/pi/step", json={**DET_LAKE, "policy": [9] * 16})
    assert bad_action.status_code == 422
    assert client.post("/api/pi/step", json={**DET_LAKE, "config": {"gamma": 2}}).status_code == 422
    assert client.post("/api/train", json={"env_key": "Pong"}).status_code == 422
    assert client.post("/api/train", json={"env_key": "FrozenLake", "options": {"bad": 1}}).status_code == 422


def test_infer_with_any_policy_and_start(client):
    full = client.post("/api/train", json={**DET_LAKE, "gamma": 0.9}).json()
    for it in full["iterations"]:
        res = client.post("/api/infer", json={**DET_LAKE, "policy": it["policy"], "start_state": 0, "seed": 1})
        assert res.status_code == 200
    best = client.post("/api/infer", json={**DET_LAKE, "policy": full["iterations"][-1]["policy"],
                                           "start_state": 8, "gamma": 0.9}).json()
    assert best["success"] and best["steps"][0]["state"] == 8
    hole = client.post("/api/infer", json={**DET_LAKE, "policy": [0] * 16, "start_state": 5})
    assert hole.status_code == 422


def test_benchmark(client):
    full = client.post("/api/train", json={**DET_LAKE, "gamma": 0.9}).json()
    policies = [it["policy"] for it in full["iterations"]]
    res = client.post("/api/benchmark", json={**DET_LAKE, "policies": policies, "episodes": 5,
                                              "start_state": 0}).json()
    assert len(res["rows"]) == len(policies) and res["rows"][-1]["success_rate"] == 1.0
    too_many = client.post("/api/benchmark", json={**DET_LAKE, "policies": policies, "episodes": 5000})
    assert too_many.status_code == 422


@pytest.mark.parametrize("env, state", [(DET_LAKE, 5), ({"env_key": "CliffWalking"}, 36), ({"env_key": "Taxi"}, 328)])
def test_render_real_gym_frame(client, env, state):
    pytest.importorskip("pygame")
    res = client.post("/api/render", json={**env, "state": state, "last_action": 2})
    assert res.status_code == 200, res.text
    assert res.json()["image"].startswith("data:image/png;base64,")
    assert client.post("/api/render", json={**env, "state": 99999}).status_code == 422


def test_render_without_pygame_returns_501(client, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygame":
            raise ImportError("no pygame")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert client.post("/api/render", json={**DET_LAKE, "state": 0}).status_code == 501


def test_default_episode_step_limit_is_1000(client):
    lay = client.post("/api/layout", json={"env_key": "Taxi"}).json()
    assert lay["env"]["default_max_steps"] == 1000
    ok = client.post("/api/infer", json={**DET_LAKE, "policy": [0] * 16, "start_state": 0, "max_steps": 5000})
    assert ok.status_code == 200 and ok.json()["length"] == 5000  # Left forever on 4x4 = truncated at the limit
    assert client.post("/api/infer", json={**DET_LAKE, "policy": [0] * 16, "max_steps": 5001}).status_code == 422
