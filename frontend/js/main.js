// App wiring: env tabs, training form + client-driven trainer, run storage, and the five views.

import { api } from "./api.js";
import { createStore } from "./store.js";
import { createTrainer } from "./trainer.js";
import { downloadJson, runsDb, validateImportedRun } from "./runsDb.js";
import { decodeTaxi } from "./taxi.js";
import { createRenderer } from "./renderers/board.js";
import { cellSizeFor } from "./renderers/draw.js";
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
  inferIter: null, inferStart: 0, inferSeed: 0, inferMaxSteps: 100, inferSpeedMs: 260,
  inferShowValues: true, inferShowArrows: true,
  benchParams: { episodes: 50, seed: 0, max_steps: 100, start_state: null }, trainWarnings: [],
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
    store.set({ runs: await runsDb.list() });
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
  const run = await runsDb.get(id);
  if (!run) {
    status("Run not found (it may have been deleted).", "error");
    await refreshRuns();
    return;
  }
  if (run.env.key !== store.get().envKey) selectEnv(run.env.key, false);
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
      await runsDb.save(run);
    } catch (err) {
      status(`Training finished but the run could not be saved: ${err.message}`, "error");
    }
    await refreshRuns();
    const msg = { converged: "Converged: π★ found", stopped: "Training stopped", max_iterations: "Reached max iterations" }[run.status];
    status(`${msg} after ${run.iterations.length} policies (${run.train_seconds.toFixed(2)}s compute).`, run.status === "converged" ? "ok" : "");
  },
  onError(err) {
    trainForm.setJob(null);
    store.set({ job: null });
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
  onOptionsChange: (options) => schedulePreview(options),
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
function schedulePreview(options) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try {
      const { env, layout } = await api.layout(store.get().envKey, options);
      const canvas = $("#preview-canvas");
      const renderer = createRenderer(canvas, { env, layout });
      renderer.resize(cellSizeFor(layout.cols, layout.rows, 420, 420));
      renderer.draw({ agent: layout.default_start, slice: null, mode: "play" });
    } catch (err) {
      status(err.message, "error");
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
      onExport: async (id) => { const run = await runsDb.get(id); if (run) downloadJson(`pi-${run.env.key}-${id}.json`, run); },
      onDelete: async (id) => {
        await runsDb.remove(id);
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
  renderMain(s);
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
    await runsDb.save(run);
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

async function boot() {
  // KaTeX loads with `defer`; typeset static explainers once it is ready.
  window.addEventListener("load", () => {
    renderPrimer($("#primer"), store.get().envKey);
    typesetExplainers(document);
  });
  try {
    const envs = await api.envs();
    store.set({ envs });
    selectEnv(store.get().envKey);
    await refreshRuns();
  } catch (err) {
    status(err.message, "error");
  }
}

boot();
