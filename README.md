# Policy Iteration Lab

Interactive, from-scratch **policy iteration** on Gymnasium toy-text environments (**FrozenLake → CliffWalking → Taxi**), with a JavaScript UI to watch training, inspect every policy and value function, and run an **inference engine** from any start state.

- Custom policy iteration (no RL library): iterative *or* exact policy evaluation, greedy improvement with tie-keeping, stores **π, V and Q for every iteration** plus the evaluation-sweep trace.
- **Live training** with Start / Pause / Resume / Stop and an adjustable per-iteration delay.
- **Every Gymnasium parameter configurable** (custom/random FrozenLake maps, slipperiness and success rate, rainy/fickle Taxi, …) plus **custom rewards** for all three envs. Each option shows whether it is a native Gymnasium argument (`gym`) or an app extension (`lab`).
- Views: training replay (V heatmap and policy arrows, eval-sweep scrubber, residual charts, state inspector), value-function gallery, policy table, inference (play π₀…π★ step by step), and a Monte-Carlo benchmark of all policies.
- Short theory notes with equations on every view.

## How it works

| Step | Equation |
|---|---|
| Policy evaluation | $V_{j+1}(s) \leftarrow \sum_{s'} P(s'\mid s,\pi(s))\,[r + \gamma V_j(s')]$ until $\max_s\lvert\Delta V\rvert<\theta$, or exactly $V^\pi=(I-\gamma P_\pi)^{-1}R_\pi$ |
| Action values | $Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\,[r+\gamma V^\pi(s')]$ |
| Improvement | $\pi_{k+1}(s)=\arg\max_a Q^{\pi_k}(s,a)$; stop when $\pi_{k+1}=\pi_k$ |

The MDP model is read from `env.unwrapped.P`. Terminal transitions don't bootstrap. CliffWalking and Taxi have no reward arguments in Gymnasium, so custom rewards are applied by remapping rewards inside `env.unwrapped.P`. `step()` samples from `P`, so the planner and the real env stay consistent.

### Configurable parameters

| Env | Native Gymnasium args | App extensions |
|---|---|---|
| FrozenLake-v1 | `map_name`, `desc` (custom or `generate_random_map`), `is_slippery`, `success_rate`, `reward_schedule` | — |
| CliffWalking-v1 | `is_slippery` | step / cliff / goal rewards |
| Taxi-v4 | `is_rainy`, `rainy_probability`, `fickle_passenger`, `fickle_probability` | step / illegal-action / dropoff rewards |

The fickle passenger is applied inside Taxi's `step()`, not in `P`, so the planner can't see it. That makes it a good test of how the policy copes with a model mismatch: compare the benchmark against the model line.

## Architecture

```
api/index.py          Vercel serverless entry (exposes the FastAPI app)
backend/app.py        Stateless FastAPI: /api/envs, /api/layout, /api/pi/step, /api/train, /api/infer, /api/benchmark
backend/rl/
  env_specs.py        env specs, every option + validation, reward remapping
  environments.py     gym env factory, sparse tabular model, layouts / start states
  policy_iteration.py custom policy iteration (evaluation, Q, improvement, step + full loop)
  inference.py        rollouts from arbitrary start states, Monte-Carlo benchmark
  serialize.py        JSON for each iteration (policy, V, Q, eval trace)
frontend/             vanilla JS (ES modules) + canvas renderers + SVG charts, KaTeX for equations
tests/                pytest (unit + API)
```

The backend is **stateless**. The browser runs policy iteration one `/api/pi/step` call at a time, which is what makes Start/Pause/Stop real controls, and stores runs in **IndexedDB**, with JSON export/import. Inference and benchmark requests send the policy along.

## Run locally

```bash
pip install -r requirements-dev.txt
uvicorn backend.app:app --reload --port 8000
# open http://localhost:8000
pytest --cov=backend      # 47 tests
```

## Deploy on Vercel

1. Push this repo to GitHub.
2. In Vercel: **Add New → Project → Import** the repo. No build settings are needed; `vercel.json` serves `frontend/` as static files and routes `/api/*` to the Python function `api/index.py`.
3. Deploy.

Or with the CLI: `npm i -g vercel && vercel --prod`.
