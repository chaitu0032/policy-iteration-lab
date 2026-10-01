// App wiring: env tabs, training form + client-driven trainer, run storage, and the five views.

import { api } from "./api.js";
import { createStore } from "./store.js";
import { createTrainer } from "./trainer.js";
import { downloadJson, validateImportedRun } from "./runsDb.js";
import { browserRunCount, migrateBrowserRuns, runs, setStorageMode, storageMode } from "./runsRepo.js";
import { createLogin } from "./components/login.js";
import { decodeTaxi } from "./taxi.js";
import { createWorld, ensureSprites } from "./world/index.js";
import { createTrainForm } from "./components/trainForm.js";
import { renderRunList } from "./components/runList.js";
import { createTrainingView } from "./components/trainingView.js";
import { createValueGallery } from "./components/valueGallery.js";
import { createPolicyTable } from "./components/policyTable.js";
import { createInferenceView } from "./components/inferenceView.js";
import { createBenchmarkView } from "./components/benchmarkView.js";
import { escapeHtml } from "./components/tooltip.js";
import { envNoteHtml, explainerHtml, renderPrimer, typesetExplainers } from "./explainers.js";

const $ = (sel) => document.querySelector(sel);
const PREVIEW_DEBOUNCE_MS = 300;

const store = createStore({
  envs: [], envKey: "FrozenLake", runs: [], run: null, job: null, view: "training",
  selectedIter: 0, followLive: true, sweepIdx: null, inspectState: null, replayMs: 700,
  display: { values: true, arrows: true, changes: true, scale: "global" },
  taxiSlice: { passenger: 0, destination: 1 }, galleryArrows: true,
  tableMode: "actions", tableScope: "slice", tableChangedOnly: false,
  inferIter: null, inferStart: 0, inferSeed: 0, inferMaxSteps: 1000, inferSpeedMs: 260,
  inferShowValues: true, inferShowArrows: true, inferBoard: "both",
  benchParams: { episodes: 50, seed: 0, max_steps: 1000, start_state: null }, trainWarnings: [],
});

// ---------------------------------------------------------------- status

let statusTimer = null;
function status(message, kind = "") {
  const el = $("#status");
  el.textContent = message;
  el.className = `status ${kind}`;
  clearTimeout(statusTimer);
  if (kind !== "error") statusTimer = setTimeout(() => { el.textContent = ""; }, 5000);
}
const ui = { status };

// ---------------------------------------------------------------- runs

async function refreshRuns() {
  try {
    store.set({ runs: await runs.list() });
  } catch (err) {
    status(`Could not read saved runs: ${err.message}`, "error");
  }
}

function setRun(run) {
  const best = run.optimal_index;
  const start = run.layout.default_start;
  const slice = run.layout.kind === "taxi"
    ? { passenger: decodeTaxi(start).passenger, destination: decodeTaxi(start).destination }
    : store.get().taxiSlice;
  store.set({
    run, selectedIter: Math.max(0, best), inferIter: Math.max(0, best), inferStart: start, inspectState: null,
    sweepIdx: null, taxiSlice: slice, inferMaxSteps: run.env.default_max_steps,
    benchParams: { ...store.get().benchParams, max_steps: run.env.default_max_steps, start_state: start },
  });
}

async function openRun(id) {
  if (trainer.isRunning()) {
    status("Stop the running training before opening another run.", "error");
    return;
  }
  const run = await runs.get(id);
  if (!run) {
    status("Run not found (it may have been deleted).", "error");
    await refreshRuns();
    return;
  }
  if (run.env.key !== store.get().envKey) selectEnv(run.env.key, false);
  const env = store.get().envs.find((e) => e.key === run.env.key);
  trainForm.setEnv(env, { env: run.env.options, pi: run.config });
  setRun(run);
}

// ---------------------------------------------------------------- trainer

const trainer = createTrainer({
  onUpdate(snap) {
    const s = store.get();
    const last = snap.run.iterations.length - 1;
    const first = !s.run || s.run.id !== snap.run.id;
    store.set({
      run: snap.run, job: snap, trainWarnings: snap.warnings,
      selectedIter: first ? 0 : s.followLive ? Math.max(0, last) : s.selectedIter,
      sweepIdx: s.followLive ? null : s.sweepIdx,
      ...(first ? { inspectState: null, inferStart: snap.run.layout.default_start,
                    inferMaxSteps: snap.run.env.default_max_steps } : {}),
    });
    trainForm.setJob(snap);
  },
  async onFinish(snap) {
    trainForm.setJob(null);
    const run = snap.run;
    store.set({ job: null, run, trainWarnings: snap.warnings });
    setRun(run);
    try {
      await runs.save(run);
    } catch (err) {
      status(`Training finished but the run could not be saved: ${err.message}`, "error");
    }
    await refreshRuns();
    const msg = { converged: "Converged: π★ found", stopped: "Training stopped", max_iterations: "Reached max iterations" }[run.status];
    status(`${msg} after ${run.iterations.length} policies (${run.train_seconds.toFixed(2)}s compute).`, run.status === "converged" ? "ok" : "");
  },
  onError(err) {
    trainForm.setJob(null);
    // A run with no evaluated policy is useless: drop it so the page doesn't sit on "Waiting for π₀".
    const run = store.get().run;
    store.set({ job: null, ...(run && !run.iterations.length ? { run: null } : {}) });
    status(`Training failed: ${err.message}`, "error");
  },
});

const trainForm = createTrainForm($("#train-card"), {
  async onStart(params) {
    try {
      status("Building the tabular model from env.P…");
      const { env, layout } = await api.layout(params.env_key, params.options);
      store.set({ view: "training", followLive: true });
      await trainer.start(params, env, layout);
      status("Training started.", "ok");
    } catch (err) {
      status(err.message, "error");
    }
  },
  onPause: () => trainer.pause(),
  onResume: () => trainer.resume(),
  onStop: () => { trainer.stop(); status("Stopping after the current step…"); },
  onLiveChange: (live) => trainer.setDelay(live.delay_ms),
  onOptionsChange: (options) => {
    // The open run no longer matches the form: show the new configuration (the run stays in Saved runs).
    if (store.get().run && !trainer.isRunning()) store.set({ run: null });
    schedulePreview(options);
  },
  onError: (msg) => status(msg, "error"),
});

// ---------------------------------------------------------------- env tabs + preview

function renderEnvTabs() {
  const { envs, envKey } = store.get();
  $("#env-tabs").innerHTML = envs.map((e) =>
    `<button data-env="${e.key}" class="${e.key === envKey ? "active" : ""}" title="${escapeHtml(e.gym_id)}">${escapeHtml(e.title)}</button>`).join("");
}

function selectEnv(key, clearRun = true) {
  if (trainer.isRunning()) {
    status("Stop the running training before switching environments.", "error");
    return;
  }
  const env = store.get().envs.find((e) => e.key === key);
  if (!env) return;
  document.body.dataset.env = key; // world accent colour
  store.set({ envKey: key, ...(clearRun ? { run: null } : {}) });
  trainForm.setEnv(env);
  $("#empty-title").textContent = `Train a policy · ${env.title}`;
  $("#empty-desc").textContent = env.description;
  renderPrimer($("#primer"), key);
  $("#env-note-body").innerHTML = `${envNoteHtml(key)}<p class="muted">See the landing page primer for the full algorithm: evaluate → Q → greedy improve → repeat until stable.</p>`;
  delete $("#env-note-body").dataset.typeset;
  typesetExplainers(document);
  schedulePreview(trainForm.envOptions());
}

let previewTimer = null;
let previewToken = 0;
function schedulePreview(options) {
  clearTimeout(previewTimer);
  const token = ++previewToken;
  previewTimer = setTimeout(async () => {
    try {
      const [{ env, layout }] = await Promise.all([api.layout(store.get().envKey, options), ensureSprites()]);
      if (token !== previewToken) return; // a newer env/option change superseded this preview
      const world = createWorld($("#preview-canvas"), { env, layout });
      world.resize(460, 420);
      const start = layout.default_start;
      world.draw({ agent: { from: start, to: start, t: 1 } });
    } catch (err) {
      if (token === previewToken) status(err.message, "error");
    }
  }, PREVIEW_DEBOUNCE_MS);
}

// ---------------------------------------------------------------- views

const actions = {
  playPolicy(k) {
    if (store.get().job) {
      status("Inference is available once training finishes or is stopped.", "error");
      return;
    }
    store.set({ inferIter: k, view: "inference" });
    views.inference.autoStart();
  },
  openIteration(k) {
    store.set({ selectedIter: k, sweepIdx: null, view: "training" });
  },
};

// Each panel = static explainer + a body the view component owns.
const panel = (name) => document.querySelector(`[data-panel="${name}"] > .panel-body`);
document.querySelectorAll("[data-panel]").forEach((p) => {
  p.innerHTML = `${explainerHtml(p.dataset.panel)}<div class="panel-body"></div>`;
});
const views = {
  training: createTrainingView(panel("training"), store, actions),
  values: createValueGallery(panel("values"), store, actions),
  policy: createPolicyTable(panel("policy"), store, actions),
  inference: createInferenceView(panel("inference"), store, ui),
  benchmark: createBenchmarkView(panel("benchmark"), store, ui),
};

function renderHeader(s) {
  const { run, job } = s;
  const opts = Object.entries(run.env.options).filter(([k]) => k !== "custom_map")
    .map(([k, v]) => `<span class="chip">${escapeHtml(k)} <b>${escapeHtml(typeof v === "number" ? +v.toFixed(4) : v)}</b></span>`).join("");
  const cfg = run.config;
  $("#run-header").innerHTML = `
    <h1>${escapeHtml(run.env.title)}</h1>
    <span class="badge ${job ? "running" : run.status}">${job ? (job.paused ? "paused" : "training") : escapeHtml(run.status)}</span>
    <span class="chip">${escapeHtml(run.env.gym_id)}</span>
    <span class="chip">|S| <b>${run.env.n_states}</b> · |A| <b>${run.env.n_actions}</b></span>
    <span class="chip">γ <b>${cfg.gamma}</b></span><span class="chip">θ <b>${cfg.theta}</b></span>
    <span class="chip">eval <b>${escapeHtml(cfg.eval_mode)}</b></span><span class="chip">π₀ <b>${escapeHtml(cfg.init_policy)}</b></span>
    ${opts}
    <span class="chip">policies <b>${run.iterations.length}</b></span>
    <span class="chip">compute <b>${run.train_seconds.toFixed(2)}s</b></span>
    ${job ? "" : '<button class="btn sm" id="export-run">⤓ Export JSON</button>'}`;
  $("#export-run")?.addEventListener("click", () => downloadJson(`pi-${run.env.key}-${run.id}.json`, run));
}

function renderMain(s) {
  const hasRun = !!s.run;
  $("#empty-state").hidden = hasRun;
  $("#run-view").hidden = !hasRun;
  if (!hasRun) return;
  renderHeader(s);
  document.querySelectorAll("#view-tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === s.view);
    if (b.dataset.view === "inference" || b.dataset.view === "benchmark") b.disabled = !!s.job;
  });
  Object.keys(views).forEach((name) => { document.querySelector(`[data-panel="${name}"]`).hidden = name !== s.view; });
  if (s.job && (s.view === "inference" || s.view === "benchmark")) {
    store.set({ view: "training" });
    return;
  }
  views[s.view].update(s);
}

let lastView = null;
store.subscribe((s, changed) => {
  if (changed.has("envs") || changed.has("envKey")) renderEnvTabs();
  if (changed.has("runs") || changed.has("run") || changed.has("envKey")) {
    renderRunList($("#run-list"), {
      runs: s.runs, envKey: s.envKey, activeId: s.run?.id,
      onOpen: openRun,
      onExport: async (id) => { const run = await runs.get(id); if (run) downloadJson(`pi-${run.env.key}-${id}.json`, run); },
      onDelete: async (id) => {
        await runs.remove(id);
        if (store.get().run?.id === id) store.set({ run: null });
        await refreshRuns();
        status("Run deleted.");
      },
    });
  }
  if (lastView && lastView !== s.view) {
    views[lastView].deactivate?.();
    views.training.stop();
  }
  lastView = s.view;
  try {
    renderMain(s);
  } catch (err) {
    // A drawing bug must never abort training or playback; surface it instead.
    console.error("[render]", err);
    status(`Display error: ${err.message}`, "error");
  }
});

// ---------------------------------------------------------------- DOM events

$("#env-tabs").addEventListener("click", (e) => {
  const key = e.target.closest("[data-env]")?.dataset.env;
  if (key && key !== store.get().envKey) selectEnv(key);
});
$("#view-tabs").addEventListener("click", (e) => {
  const view = e.target.closest("[data-view]")?.dataset.view;
  if (view) store.set({ view });
});
$("#refresh-runs").addEventListener("click", refreshRuns);
$("#import-run").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  try {
    const run = validateImportedRun(JSON.parse(await file.text()));
    await runs.save(run);
    await refreshRuns();
    await openRun(run.id);
    status("Run imported.", "ok");
  } catch (err) {
    status(`Import failed: ${err.message}`, "error");
  }
});
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { views.training.refit(); renderMain(store.get()); }, 200);
});

// ---------------------------------------------------------------- boot

// ---------------------------------------------------------------- accounts + boot

const login = createLogin($("#login"));
let sessionInfo = { auth: false, user: "local", storage: "browser" };

function showAccount() {
  $("#account").hidden = !sessionInfo.auth;
  $("#account-name").textContent = sessionInfo.user || "";
}

async function offerMigration() {
  const box = $("#migrate");
  const n = storageMode() === "server" ? await browserRunCount() : 0;
  box.hidden = n === 0;
  if (!n) return;
  box.innerHTML = `<span>${n} run${n === 1 ? " is" : "s are"} saved only in this browser.</span>
    <button class="btn sm" type="button" id="migrate-btn">Move to my account</button>`;
  $("#migrate-btn").addEventListener("click", async () => {
    try {
      const moved = await migrateBrowserRuns((i, total) => status(`Moving runs… ${i}/${total}`));
      status(`Moved ${moved} run${moved === 1 ? "" : "s"} to your account.`, "ok");
      box.hidden = true;
      await refreshRuns();
    } catch (err) {
      status(`Could not move runs: ${err.message}`, "error");
    }
  });
}

async function ensureSignedIn() {
  sessionInfo = await api.session();
  if (sessionInfo.auth && !sessionInfo.user) {
    sessionInfo = { ...sessionInfo, user: await login.prompt() };
  }
  setStorageMode(sessionInfo.storage);
  showAccount();
}

$("#logout").addEventListener("click", async () => {
  if (trainer.isRunning()) trainer.stop();
  try {
    await api.logout();
  } finally {
    location.reload();
  }
});

let reauthing = false;
window.addEventListener("pi:auth-required", async () => {
  if (reauthing || !sessionInfo.auth) return;
  reauthing = true;
  status("Your session ended. Sign in again to continue.", "error");
  sessionInfo = { ...sessionInfo, user: await login.prompt() };
  showAccount();
  reauthing = false;
  status("Signed in again.", "ok");
});

async function boot() {
  // KaTeX loads with `defer`; typeset static explainers once it is ready.
  window.addEventListener("load", () => {
    renderPrimer($("#primer"), store.get().envKey);
    typesetExplainers(document);
  });
  try {
    await ensureSignedIn();
    const envs = await api.envs();
    store.set({ envs });
    selectEnv(store.get().envKey);
    await refreshRuns();
    await offerMigration();
  } catch (err) {
    status(err.message, "error");
  }
}

boot();
