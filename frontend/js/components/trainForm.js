// Training form: env options (from backend schema), every policy-iteration hyper-parameter,
// live-replay delay, and Start / Pause-Resume / Stop controls.

import { escapeHtml } from "./tooltip.js";

const PI_FIELDS = [
  { name: "gamma", label: "Discount γ", type: "float", min: 0, max: 1, step: 0.01, default: 0.99, range: true },
  { name: "theta", label: "Eval threshold θ", type: "float", min: 1e-12, max: 1, step: "any", default: 1e-8,
    hint: "Stop evaluation sweeps when max |ΔV| < θ" },
  { name: "eval_mode", label: "Evaluation", type: "select", choices: ["iterative", "exact"], default: "iterative",
    hint: "iterative = Bellman sweeps · exact = solve (I − γP)V = R" },
  { name: "max_eval_sweeps", label: "Max eval sweeps", type: "int", min: 1, max: 100000, default: 10000 },
  { name: "max_iterations", label: "Max PI iterations", type: "int", min: 1, max: 500, default: 100 },
  { name: "init_policy", label: "Initial policy π₀", type: "select", choices: ["zeros", "random"], default: "zeros" },
  { name: "seed", label: "Init seed", type: "int", min: 0, max: 2147483647, default: 0, visibleIf: { init_policy: "random" } },
];
const LIVE_FIELDS = [
  { name: "delay_ms", label: "Delay / iteration (ms)", type: "int", min: 0, max: 5000, step: 50, default: 400, range: true,
    hint: "Pause between policy-iteration steps so you can watch each π_k appear. Adjustable while training." },
];

// Bump the version whenever defaults change so stale saved values do not override them.
const STORAGE_KEY = "pi-lab-form-v2";

function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    return {};
  }
}

function persist(all) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    /* storage unavailable: form still works, values just are not remembered */
  }
}

function schemaToFields(schema) {
  return Object.entries(schema).map(([name, s]) => ({
    name, label: s.label, type: s.type, choices: s.choices, min: s.min, max: s.max, step: s.step,
    default: s.default, visibleIf: s.visible_if, range: s.type === "float" && !s.group?.startsWith("reward"),
    group: s.group, hint: s.hint, source: s.source, gymArg: s.gym_arg,
  }));
}

function fieldHtml(f, value, group) {
  const id = `f-${group}-${f.name}`;
  const data = `data-group="${group}" data-name="${f.name}" data-type="${f.type}"`;
  let control;
  if (f.type === "map") {
    control = `<textarea id="${id}" ${data} rows="5" spellcheck="false" class="map-input">${escapeHtml(value)}</textarea>`;
  } else if (f.type === "bool") {
    control = `<input type="checkbox" id="${id}" ${data} ${value ? "checked" : ""}/>`;
  } else if (f.type === "select") {
    control = `<select id="${id}" ${data}>${f.choices.map((c) => `<option ${c === value ? "selected" : ""}>${escapeHtml(c)}</option>`).join("")}</select>`;
  } else if (f.range) {
    control = `<div class="range-row"><input type="range" id="${id}" ${data} min="${f.min}" max="${f.max}" step="${f.step ?? 1}" value="${value}"/><output>${value}</output></div>`;
  } else {
    control = `<input type="number" id="${id}" ${data} min="${f.min}" max="${f.max}" step="${f.step ?? 1}" value="${value}"/>`;
  }
  const badge = f.source
    ? `<span class="src ${f.source}" title="${f.gymArg ? `gymnasium.make(…, ${escapeHtml(f.gymArg)}=…)` : "Not a gymnasium argument: applied by this app by remapping rewards in env.P"}">${f.source === "gymnasium" ? "gym" : "lab"}</span>`
    : "";
  return `<div class="field ${f.range ? "range-field" : ""} ${f.type === "map" ? "map-field" : ""}" data-field="${group}.${f.name}">
      <label for="${id}">${escapeHtml(f.label)} ${badge}</label>${control}
      ${f.hint ? `<div class="hint">${escapeHtml(f.hint)}</div>` : ""}</div>`;
}

function readValue(input) {
  const t = input.dataset.type;
  if (t === "bool") return input.checked;
  if (t === "int") return Number.parseInt(input.value, 10);
  if (t === "float") return Number.parseFloat(input.value);
  return input.value;
}

function validate(fields, values) {
  for (const f of fields) {
    const v = values[f.name];
    if (f.type === "int" || f.type === "float") {
      if (!Number.isFinite(v)) return `${f.label} must be a number`;
      if (f.type === "int" && !Number.isInteger(v)) return `${f.label} must be an integer`;
      if (v < f.min || v > f.max) return `${f.label} must be in [${f.min}, ${f.max}]`;
    }
  }
  return null;
}

export function createTrainForm(root, { onStart, onPause, onResume, onStop, onOptionsChange, onLiveChange, onError }) {
  let env = null;
  let saved = loadSaved();
  let values = { env: {}, pi: {}, live: {} };

  function fieldsFor(group) {
    if (group === "env") return schemaToFields(env.option_schema);
    return group === "pi" ? PI_FIELDS : LIVE_FIELDS;
  }

  function initialValues(group) {
    const prev = saved[env.key]?.[group] || {};
    return Object.fromEntries(fieldsFor(group).map((f) => [f.name, prev[f.name] ?? f.default]));
  }

  function applyVisibility() {
    ["env", "pi", "live"].forEach((group) => fieldsFor(group).forEach((f) => {
      const node = root.querySelector(`[data-field="${group}.${f.name}"]`);
      if (!node || !f.visibleIf) return;
      const visible = Object.entries(f.visibleIf).every(([k, v]) => values[group][k] === v);
      node.hidden = !visible;
    }));
  }

  function render() {
    root.innerHTML = `
      <div class="card-head"><h2>Train · ${escapeHtml(env.title)}</h2><span class="mono muted" style="font-size:11px">${escapeHtml(env.gym_id)}</span></div>
      <form novalidate>
        <fieldset><legend>Environment dynamics</legend>${fieldsFor("env").filter((f) => f.group !== "rewards").map((f) => fieldHtml(f, values.env[f.name], "env")).join("")}</fieldset>
        <fieldset><legend>Rewards</legend>${fieldsFor("env").filter((f) => f.group === "rewards").map((f) => fieldHtml(f, values.env[f.name], "env")).join("")}</fieldset>
        <fieldset><legend>Policy iteration</legend>${PI_FIELDS.map((f) => fieldHtml(f, values.pi[f.name], "pi")).join("")}</fieldset>
        <fieldset><legend>Live training</legend>${LIVE_FIELDS.map((f) => fieldHtml(f, values.live[f.name], "live")).join("")}</fieldset>
        <div class="btn-row grow">
          <button type="submit" class="btn primary" data-act="start">▶ Start training</button>
        </div>
        <div class="btn-row grow" style="margin-top:6px">
          <button type="button" class="btn" data-act="pause" disabled>⏸ Pause</button>
          <button type="button" class="btn danger" data-act="stop" disabled>■ Stop</button>
          <button type="button" class="btn ghost" data-act="reset" title="Restore defaults">↺ Defaults</button>
        </div>
        <div class="note" data-role="job-note" style="margin-top:8px"></div>
      </form>`;
    applyVisibility();
  }

  root.addEventListener("input", (e) => {
    const input = e.target;
    if (!input.dataset?.group) return;
    const { group, name } = input.dataset;
    values = { ...values, [group]: { ...values[group], [name]: readValue(input) } };
    const out = input.parentElement.querySelector("output");
    if (out) out.textContent = input.value;
    saved = { ...saved, [env.key]: values };
    persist(saved);
    applyVisibility();
    if (group === "env") onOptionsChange?.(envOptions());
    if (group === "live") onLiveChange?.(values.live);
  });

  root.addEventListener("submit", (e) => {
    e.preventDefault();
    const error = ["env", "pi", "live"].map((g) => validate(visibleFields(g), values[g])).find(Boolean);
    if (error) {
      onError?.(error);
      return;
    }
    onStart({ env_key: env.key, options: envOptions(), ...values.pi, ...values.live });
  });

  root.addEventListener("click", (e) => {
    const act = e.target.closest("button")?.dataset.act;
    if (act === "pause") (e.target.dataset.paused === "1" ? onResume : onPause)();
    if (act === "stop") onStop();
    if (act === "reset") {
      saved = { ...saved, [env.key]: undefined };
      persist(saved);
      setEnv(env);
      onOptionsChange?.(envOptions());
    }
  });

  function visibleFields(group) {
    return fieldsFor(group).filter((f) => !f.visibleIf || Object.entries(f.visibleIf).every(([k, v]) => values[group][k] === v));
  }

  function envOptions() {
    return Object.fromEntries(visibleFields("env").map((f) => [f.name, values.env[f.name]]));
  }

  function setEnv(nextEnv) {
    env = nextEnv;
    values = { env: initialValues("env"), pi: initialValues("pi"), live: initialValues("live") };
    render();
  }

  function setJob(job) {
    const running = job && (job.status === "running" || job.status === "pending");
    const start = root.querySelector('[data-act="start"]');
    const pause = root.querySelector('[data-act="pause"]');
    const stop = root.querySelector('[data-act="stop"]');
    const note = root.querySelector('[data-role="job-note"]');
    if (!start) return;
    start.disabled = !!running;
    pause.disabled = !running;
    stop.disabled = !running;
    pause.dataset.paused = job?.paused ? "1" : "0";
    pause.textContent = job?.paused ? "▶ Resume" : "⏸ Pause";
    root.querySelectorAll("input, select, textarea").forEach((i) => {
      i.disabled = !!running && i.dataset.group !== "live";
    });
    note.innerHTML = running
      ? `<span class="spinner"></span> ${job.paused ? "Paused" : "Training"} · ${job.n_iterations} policies evaluated`
      : "";
  }

  return { setEnv, setJob, envOptions };
}
