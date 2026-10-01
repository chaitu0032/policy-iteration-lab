"""Custom policy iteration (Howard, 1960) on a dense tabular model.

Loop:  pi_0 -> evaluate V^{pi_k} -> greedy improve -> pi_{k+1}  ... until pi_{k+1} == pi_k.

Every policy pi_k is recorded together with its value function V^{pi_k}, its action-values
Q^{pi_k}, the per-sweep evaluation residuals and a few intermediate value snapshots so the
UI can replay how evaluation converged. All records are immutable dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal

import numpy as np

from .environments import TabularModel

EvalMode = Literal["iterative", "exact"]
StopFn = Callable[[], bool]
InitPolicy = Literal["zeros", "random"]

TIE_TOLERANCE = 1e-9
MAX_SNAPSHOTS_PER_ITERATION = 24


@dataclass(frozen=True)
class PIConfig:
    gamma: float = 0.99
    theta: float = 1e-8
    max_iterations: int = 100
    max_eval_sweeps: int = 10_000
    eval_mode: EvalMode = "iterative"
    init_policy: InitPolicy = "zeros"
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.theta <= 0:
            raise ValueError("theta must be > 0")
        if self.max_iterations < 1 or self.max_eval_sweeps < 1:
            raise ValueError("max_iterations and max_eval_sweeps must be >= 1")
        if self.eval_mode not in ("iterative", "exact"):
            raise ValueError("eval_mode must be 'iterative' or 'exact'")
        if self.init_policy not in ("zeros", "random"):
            raise ValueError("init_policy must be 'zeros' or 'random'")


@dataclass(frozen=True)
class EvaluationResult:
    V: np.ndarray
    sweeps: int
    deltas: tuple[float, ...]
    snapshots: tuple[tuple[int, np.ndarray], ...]  # (sweep index, V at that sweep)
    converged: bool


@dataclass(frozen=True)
class IterationRecord:
    index: int
    policy: np.ndarray
    V: np.ndarray
    Q: np.ndarray
    evaluation: EvaluationResult
    changed_states: tuple[int, ...]  # states whose action differs from pi_{k-1}


@dataclass(frozen=True)
class PIResult:
    iterations: tuple[IterationRecord, ...]
    converged: bool
    config: PIConfig = field(default_factory=PIConfig)

    @property
    def optimal(self) -> IterationRecord:
        return self.iterations[-1]


def initial_policy(model: TabularModel, kind: InitPolicy, seed: int) -> np.ndarray:
    if kind == "random":
        rng = np.random.default_rng(seed)
        return rng.integers(0, model.n_actions, size=model.n_states)
    return np.zeros(model.n_states, dtype=np.int64)


def policy_arrays(model: TabularModel, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_pi, successors, continuation probs) of ``policy`` — each row has K entries."""
    states = np.arange(model.n_states)
    return model.R[states, policy], model.next_states[states, policy], model.cont_probs[states, policy]


def bellman_backup(R_pi: np.ndarray, succ: np.ndarray, cont: np.ndarray, V: np.ndarray, gamma: float) -> np.ndarray:
    """V'(s) = R_pi(s) + gamma * sum_k cont[s, k] * V[succ[s, k]]  (sparse gather, no BLAS needed)."""
    return R_pi + gamma * np.einsum("sk,sk->s", cont, V[succ])


def _snapshot_schedule(total: int) -> set[int]:
    if total <= MAX_SNAPSHOTS_PER_ITERATION:
        return set(range(1, total + 1))
    # Geometric spacing: early sweeps change most, so sample them densely.
    picks = np.unique(np.geomspace(1, total, MAX_SNAPSHOTS_PER_ITERATION).astype(int))
    return set(int(p) for p in picks) | {total}


def _never() -> bool:
    return False


def evaluate_iterative(model: TabularModel, policy: np.ndarray, cfg: PIConfig,
                       should_stop: StopFn = _never) -> EvaluationResult:
    """Synchronous (Jacobi) iterative policy evaluation until max |dV| < theta (or stopped)."""
    R_pi, succ, cont = policy_arrays(model, policy)
    V = np.zeros(model.n_states)
    keep = _snapshot_schedule(cfg.max_eval_sweeps)
    snapshots: list[tuple[int, np.ndarray]] = []
    deltas: list[float] = []
    converged = False
    for sweep in range(1, cfg.max_eval_sweeps + 1):
        V_new = bellman_backup(R_pi, succ, cont, V, cfg.gamma)
        delta = float(np.max(np.abs(V_new - V)))
        V = V_new
        deltas.append(delta)
        if sweep in keep:
            snapshots.append((sweep, V))
        if delta < cfg.theta:
            converged = True
            break
        if should_stop():
            break
    if not snapshots or snapshots[-1][0] != len(deltas):
        snapshots.append((len(deltas), V))
    return EvaluationResult(V, len(deltas), tuple(deltas), tuple(snapshots), converged)


def evaluate_exact(model: TabularModel, policy: np.ndarray, cfg: PIConfig) -> EvaluationResult:
    """Solve (I - gamma * C_pi) V = R_pi directly; falls back to iterative if singular."""
    R_pi, succ, cont = policy_arrays(model, policy)
    C_pi = np.zeros((model.n_states, model.n_states))
    np.add.at(C_pi, (np.arange(model.n_states)[:, None], succ), cont)
    A = np.eye(model.n_states) - cfg.gamma * C_pi
    try:
        V = np.linalg.solve(A, R_pi)
    except np.linalg.LinAlgError:
        return evaluate_iterative(model, policy, cfg)
    if not np.all(np.isfinite(V)):
        return evaluate_iterative(model, policy, cfg)
    return EvaluationResult(V, 1, (0.0,), ((1, V),), True)


def evaluate_policy(model: TabularModel, policy: np.ndarray, cfg: PIConfig,
                    should_stop: StopFn = _never) -> EvaluationResult:
    if cfg.eval_mode == "exact":
        return evaluate_exact(model, policy, cfg)
    return evaluate_iterative(model, policy, cfg, should_stop)


def q_from_v(model: TabularModel, V: np.ndarray, gamma: float) -> np.ndarray:
    """Q[s, a] = R[s, a] + gamma * sum_s' C[s, a, s'] V[s']  (sparse gather over successors)."""
    return model.R + gamma * np.einsum("sak,sak->sa", model.cont_probs, V[model.next_states])


def improve_policy(Q: np.ndarray, policy: np.ndarray, terminal: np.ndarray) -> np.ndarray:
    """Greedy improvement. Keeps the current action on ties so the loop cannot oscillate."""
    best = np.argmax(Q, axis=1)
    rows = np.arange(len(policy))
    current_is_best = Q[rows, policy] >= Q[rows, best] - TIE_TOLERANCE
    new_policy = np.where(current_is_best, policy, best)
    return np.where(terminal, policy, new_policy)


def policy_iteration_step(model: TabularModel, cfg: PIConfig, policy: np.ndarray, index: int,
                          previous: np.ndarray | None = None,
                          should_stop: StopFn = _never) -> tuple[IterationRecord, np.ndarray, bool]:
    """One PI step: evaluate ``policy`` (pi_k), then improve it. Returns (record, pi_{k+1}, is_stable)."""
    # Each policy is evaluated from V=0 so the stored snapshots show the full convergence.
    evaluation = evaluate_policy(model, policy, cfg, should_stop)
    Q = q_from_v(model, evaluation.V, cfg.gamma)
    changed = () if previous is None else tuple(int(s) for s in np.flatnonzero(previous != policy))
    record = IterationRecord(index, policy.copy(), evaluation.V.copy(), Q, evaluation, changed)
    new_policy = improve_policy(Q, policy, model.terminal)
    return record, new_policy, bool(np.array_equal(new_policy, policy))


def policy_iteration_steps(model: TabularModel, cfg: PIConfig,
                           should_stop: StopFn = _never) -> Iterator[tuple[IterationRecord, bool]]:
    """Yield (record of pi_k, is_stable) one policy at a time. Stops early when ``should_stop()``."""
    policy = initial_policy(model, cfg.init_policy, cfg.seed)
    previous: np.ndarray | None = None
    for k in range(cfg.max_iterations):
        record, new_policy, stable = policy_iteration_step(model, cfg, policy, k, previous, should_stop)
        yield record, stable
        if stable or should_stop():
            return
        previous, policy = policy, new_policy


def policy_iteration(model: TabularModel, cfg: PIConfig | None = None) -> PIResult:
    cfg = cfg or PIConfig()
    records: list[IterationRecord] = []
    converged = False
    for record, stable in policy_iteration_steps(model, cfg):
        records.append(record)
        converged = stable
    return PIResult(tuple(records), converged, cfg)
