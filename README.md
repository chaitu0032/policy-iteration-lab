# Policy Iteration Lab

Interactive, from-scratch **policy iteration** on Gymnasium toy-text environments (**FrozenLake → CliffWalking → Taxi**), with a JavaScript UI to watch training, inspect every policy and value function, and run an **inference engine** from any start state.

- Custom policy iteration (no RL library): iterative *or* exact policy evaluation, greedy improvement with tie-keeping, stores **π, V and Q for every iteration** plus the evaluation-sweep trace.
- **Live training** with Start / Pause / Resume / Stop and an adjustable per-iteration delay.
- **Every Gymnasium parameter configurable** (custom/random FrozenLake maps, slipperiness and success rate, rainy/fickle Taxi, …) plus **custom rewards** for all three envs. Each option shows whether it is a native Gymnasium argument (`gym`) or an app extension (`lab`).
- **Gymnasium render view** in Inference: Analysis / Gymnasium / Side-by-side. The Gymnasium view redraws the episode with Gymnasium's own sprites (a canvas replica of `env.render()`), animating the elf or cab. Tick **Real env.render()** to fetch the exact pygame frame from the server via `/api/render`.
- Each world tints the UI with a colour from its sprites: ice blue, meadow green, cab yellow.
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
  render.py           real gymnasium env.render() frames (rgb_array -> PNG), needs pygame
frontend/             vanilla JS (ES modules) + canvas renderers + SVG charts, KaTeX for equations
frontend/js/world/    gymnasium-sprite renderers (FrozenLake, CliffWalking, Taxi) mirroring _render_gui
frontend/assets/gym/  sprites copied from Gymnasium (MIT, see NOTICE.md)
tests/                pytest (unit + API)
```

The backend is **stateless**. The browser runs policy iteration one `/api/pi/step` call at a time, which is what makes Start/Pause/Stop real controls, and stores runs in **IndexedDB**, with JSON export/import. Inference and benchmark requests send the policy along.

## Run locally

```bash
pip install -r requirements-dev.txt
./run.sh                  # frees port 8765 if needed, then serves http://localhost:8765
pytest --cov=backend      # unit + API tests
python3 -m pytest e2e     # browser stress tests (Playwright, starts its own servers)
```

## Accounts, sessions and storage

- **Accounts:** `python3 scripts/create_users.py` creates `user1`…`user5` with random 16-character passwords. You can pass your own names instead (`... alice bob`). The script writes:
  - `.env.local`: `PI_LAB_USERS` (salted PBKDF2 hashes only) and `PI_LAB_SECRET`. The local server loads this file.
  - `credentials.local.txt`: the plain passwords, for handing out.
  Both files are git-ignored and readable only by you.
- **Sessions:** a signed (HMAC-SHA256) HttpOnly, SameSite=Lax cookie that lasts 7 days.
  - No session state is kept on the server, so any serverless instance can verify any request.
  - The same user can be signed in from several browsers at once.
  - Signing out clears the cookie in that browser. Changing `PI_LAB_SECRET` signs everyone out.
- **Protection:**
  - Every `/api` route except health, session and login requires a session.
  - Write requests must carry `X-Requested-With: pi-lab` (CSRF).
  - Failed logins are rate-limited per username: 5 per 10 minutes, counted separately on each instance.
  - Error messages never reveal whether a username exists.
- **Runs per user:** each run is a gzip-compressed JSON file. The server validates it, computes its summary itself and caps each user at 100 runs.
  - Locally they go in `data/runs/<user>/`; on Vercel, in a **private Vercel Blob** store.
  - Your runs follow you to any browser. Runs made before accounts existed show a **Move to my account** button.
- **Local dev without accounts:** if `.env.local` is missing, auth is off and you are the `local` user. On Vercel the app refuses requests without accounts (it fails closed).

### Serverless notes

- Training is driven by the browser one policy-iteration step per request, with an 8-second evaluation budget and 30-second function limit, so many users can train at once without blocking each other.
- Runs are moved gzip-compressed because Vercel function bodies are limited to 4.5 MB.
- The tabular model is cached per warm instance. The first request after an idle period takes about 1–3 s to load numpy, gymnasium and pygame.

## Deploy on Vercel

1. Push this repo to GitHub, then in Vercel choose **Add New → Project → Import**. No build settings are needed: `vercel.json` serves `frontend/` statically and routes `/api/*` to `api/index.py`.
2. **Storage → Create → Blob**, choose **Private** access, and connect it to the project. This adds `BLOB_READ_WRITE_TOKEN`.
3. **Settings → Environment Variables:** add `PI_LAB_USERS` and `PI_LAB_SECRET`, copying the values from your `.env.local`.
4. Deploy, and sign in with an account from `credentials.local.txt`.

Or with the CLI: `npm i -g vercel && vercel --prod`.

### Defaults

FrozenLake opens on the **8×8 map with slippery ice off, γ = 0.99, and rewards goal +1 / hole −1 / frozen step −0.01** (Gymnasium's own default is 1 / 0 / 0). Every move is then deterministic and the goal reward is discounted per step, so π★ is the 14-step shortest safe path. Turn slippery ice on (success rate 1/3) to see the optimal policy switch to long, cautious detours.
