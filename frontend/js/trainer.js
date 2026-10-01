// Client-driven policy iteration: one /api/pi/step request per policy, with Start / Pause / Resume / Stop.
// Each finished step is appended to the run immediately, so the UI shows pi_k, V^pi_k and Q^pi_k live.

import { api } from "./api.js";
import { newRunId } from "./runsDb.js";

const PI_KEYS = ["gamma", "theta", "max_iterations", "max_eval_sweeps", "eval_mode", "init_policy", "seed"];

export function createTrainer({ onUpdate, onFinish, onError }) {
  let job = null;

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  async function waitWhilePaused() {
    while (job && job.paused && !job.stopRequested) await sleep(100);
  }

  async function loop(current) {
    let policy = null;
    let previous = null;
    try {
      for (let k = 0; k < current.config.max_iterations; k += 1) {
        await waitWhilePaused();
        if (current.stopRequested) return finish(current, "stopped");
        const body = await api.step({
          env_key: current.run.env.key, options: current.run.env.options, config: current.config,
          index: k, policy, previous_policy: previous,
        });
        if (job !== current) return undefined; // superseded
        const iterations = [...current.run.iterations, body.iteration];
        current.run = { ...current.run, iterations, optimal_index: iterations.length - 1,
                        train_seconds: current.run.train_seconds + body.seconds };
        current.warnings = body.time_budget_hit
          ? [...current.warnings, `π${k}: evaluation hit the per-request time budget (not fully converged)`]
          : current.warnings;
        onUpdate(snapshot(current));
        if (body.stable) return finish(current, "converged");
        previous = body.iteration.policy;
        policy = body.next_policy;
        if (current.delayMs) await sleep(current.delayMs);
      }
      return finish(current, "max_iterations");
    } catch (err) {
      if (current.run.iterations.length) finish(current, "stopped");
      else job = null;
      onError(err);
      return undefined;
    }
  }

  function finish(current, status) {
    current.run = { ...current.run, status };
    current.status = status;
    if (job === current) job = null;
    onFinish(snapshot(current));
  }

  function snapshot(current) {
    return {
      status: current.status, paused: current.paused, run: current.run,
      n_iterations: current.run.iterations.length, warnings: current.warnings,
    };
  }

  return {
    async start(params, envHeader, layout) {
      if (job) throw new Error("Training already running");
      const config = Object.fromEntries(PI_KEYS.map((k) => [k, params[k]]));
      const run = {
        id: newRunId(), created_at: Date.now() / 1000, env: envHeader, layout, config,
        status: "running", optimal_index: -1, train_seconds: 0, iterations: [],
      };
      job = { run, config, delayMs: params.delay_ms || 0, paused: false, stopRequested: false,
              status: "running", warnings: [] };
      onUpdate(snapshot(job));
      loop(job);
    },
    pause() { if (job) { job.paused = true; onUpdate(snapshot(job)); } },
    resume() { if (job) { job.paused = false; onUpdate(snapshot(job)); } },
    stop() { if (job) { job.stopRequested = true; job.paused = false; } },
    setDelay(ms) { if (job) job.delayMs = ms; },
    isRunning: () => !!job,
  };
}
