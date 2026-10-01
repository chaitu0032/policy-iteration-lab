import numpy as np
import pytest

from backend.rl.environments import CLIFF_WALKING, FROZEN_LAKE, TAXI, extract_model
from backend.rl.policy_iteration import (
    PIConfig,
    evaluate_exact,
    evaluate_iterative,
    improve_policy,
    policy_iteration,
    q_from_v,
)


@pytest.fixture(scope="module")
def lake_det():
    return extract_model(FROZEN_LAKE, {"map_name": "4x4", "is_slippery": False})


def test_model_is_stochastic_matrix():
    model = extract_model(FROZEN_LAKE, {"map_name": "4x4", "is_slippery": True})
    assert np.allclose(model.T.sum(axis=2), 1.0)
    assert model.terminal.sum() == 5  # 4 holes + goal
    assert np.all(model.C <= model.T + 1e-12)


def test_deterministic_frozen_lake_optimal_value(lake_det):
    gamma = 0.9
    result = policy_iteration(lake_det, PIConfig(gamma=gamma))
    assert result.converged
    # Shortest path from S to G is 6 steps; reward 1 arrives on the 6th -> gamma^5.
    assert result.optimal.V[0] == pytest.approx(gamma ** 5, abs=1e-6)


def test_every_iteration_stores_policy_value_and_q(lake_det):
    result = policy_iteration(lake_det, PIConfig(gamma=0.9))
    assert len(result.iterations) >= 2
    for k, rec in enumerate(result.iterations):
        assert rec.index == k
        assert rec.policy.shape == (16,)
        assert rec.V.shape == (16,)
        assert rec.Q.shape == (16, 4)
        assert rec.evaluation.snapshots, "evaluation snapshots must be stored"
        assert rec.evaluation.snapshots[-1][0] == rec.evaluation.sweeps
    assert result.iterations[0].changed_states == ()
    assert all(r.changed_states for r in result.iterations[1:])


def test_policy_improvement_is_monotone():
    model = extract_model(FROZEN_LAKE, {"map_name": "8x8", "is_slippery": True})
    result = policy_iteration(model, PIConfig(gamma=0.99, eval_mode="exact"))
    for prev, cur in zip(result.iterations, result.iterations[1:]):
        assert np.all(cur.V >= prev.V - 1e-8)


def test_exact_and_iterative_evaluation_agree(lake_det):
    cfg = PIConfig(gamma=0.9, theta=1e-12)
    policy = np.full(16, 2)
    a = evaluate_iterative(lake_det, policy, cfg).V
    b = evaluate_exact(lake_det, policy, cfg).V
    assert np.allclose(a, b, atol=1e-8)


def test_improve_policy_keeps_action_on_ties():
    Q = np.array([[1.0, 1.0], [0.0, 2.0]])
    new = improve_policy(Q, np.array([1, 0]), np.array([False, False]))
    assert new.tolist() == [1, 1]


def test_q_from_v_matches_bellman(lake_det):
    V = np.arange(16, dtype=float)
    Q = q_from_v(lake_det, V, 0.5)
    s, a = 4, 1
    expected = lake_det.R[s, a] + 0.5 * lake_det.C[s, a] @ V
    assert Q[s, a] == pytest.approx(expected)


def test_cliff_walking_optimal_is_shortest_edge_path():
    model = extract_model(CLIFF_WALKING, {"is_slippery": False})
    result = policy_iteration(model, PIConfig(gamma=1.0, eval_mode="iterative", max_eval_sweeps=200))
    assert result.converged
    assert result.optimal.V[36] == pytest.approx(-13.0)


def test_taxi_converges_quickly():
    model = extract_model(TAXI, {})
    result = policy_iteration(model, PIConfig(gamma=0.99, eval_mode="exact"))
    assert result.converged
    assert len(result.iterations) < 30
    assert np.all(result.optimal.V[~model.terminal] > 0)


def test_random_initial_policy_is_seeded(lake_det):
    a = policy_iteration(lake_det, PIConfig(init_policy="random", seed=3)).iterations[0].policy
    b = policy_iteration(lake_det, PIConfig(init_policy="random", seed=3)).iterations[0].policy
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kwargs", [{"gamma": 1.5}, {"theta": 0}, {"max_iterations": 0},
                                    {"eval_mode": "bogus"}, {"init_policy": "x"}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        PIConfig(**kwargs)
