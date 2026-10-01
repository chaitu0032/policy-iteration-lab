// Inference engine UI: play any pi_k (or pi*) in the real gymnasium env from any start state.

import { api } from "../api.js";
import { formatValue } from "../colors.js";
import { mountBoard } from "../renderers/board.js";
import { decodeTaxi, describeState, encodeTaxi, PASSENGER_LABELS } from "../taxi.js";
import { sharedRange } from "./trainingView.js";
import { createGymView } from "./gymView.js";
import { escapeHtml, hideTooltip, showStateTooltip } from "./tooltip.js";

const MAX_STEPS_LIMIT = 5000;

export function createInferenceView(root, store, ui) {
  let board = null;
  let boardRunId = null;
  let gym = null;
  let local = freshLocal();
  let timer = null;
  let compareToken = 0;

  function freshLocal() {
    return { episode: null, frame: 0, playing: false, loading: false, compare: null };
  }

  const ctx = () => {
    const s = store.get();
    const run = s.run;
    const k = Math.min(s.inferIter ?? run.optimal_index, run.iterations.length - 1);
    return { s, run, k, it: run.iterations[k] };
  };

  function agentState() {
    const { s } = ctx();
    if (!local.episode) return s.inferStart;
    return local.frame === 0 ? local.episode.start_state : local.episode.steps[local.frame - 1].next_state;
  }

  function boardView() {
    const { s, run, it } = ctx();
    const ep = local.episode;
    const trail = ep ? [ep.start_state, ...ep.steps.slice(0, local.frame).map((st) => st.next_state)] : [];
    const lastStep = ep && local.frame > 0 ? ep.steps[local.frame - 1] : null;
    const finished = ep && local.frame === ep.steps.length;
    return {
      values: it.V, range: sharedRange(run, it, "global"), policy: it.policy,
      showValues: s.inferShowValues, showArrows: s.inferShowArrows, mode: "play",
      agent: agentState(), trail, startState: s.inferStart,
      agentStatus: finished ? (ep.success ? "success" : lastStep?.terminated ? "fail" : null) : null,
      slice: run.layout.kind === "taxi" ? sliceOfStart(s) : null,
    };
  }

  function sliceOfStart(s) {
    const d = decodeTaxi(s.inferStart ?? 0);
    return { passenger: d.passenger, destination: d.destination };
  }

  function setStart(state) {
    const { run } = ctx();
    if (!run.layout.start_states.includes(state)) {
      ui.status("That state cannot be a start state (terminal or cliff).", "error");
      return;
    }
    stopPlayback();
    local = freshLocal();
    store.set({ inferStart: state });
  }

  // ---------------- episode control ----------------

  async function startEpisode() {
    const { s, run, it } = ctx();
    stopPlayback();
    local = { ...freshLocal(), loading: true };
    render();
    try {
      const episode = await api.infer({
        env_key: run.env.key, options: run.env.options, policy: it.policy, gamma: run.config.gamma,
        start_state: s.inferStart, seed: s.inferSeed, max_steps: s.inferMaxSteps,
      });
      local = { ...freshLocal(), episode };
      play();
    } catch (err) {
      local = freshLocal();
      ui.status(err.message, "error");
      render();
    }
  }

  function play() {
    if (!local.episode) return;
    if (local.frame >= local.episode.steps.length) local = { ...local, frame: 0 };
    local = { ...local, playing: true };
    const tick = () => {
      if (!local.playing) return;
      if (local.frame >= local.episode.steps.length) {
        local = { ...local, playing: false };
        render();
        return;
      }
      local = { ...local, frame: local.frame + 1 };
      render();
      timer = setTimeout(tick, store.get().inferSpeedMs);
    };
    render();
    timer = setTimeout(tick, store.get().inferSpeedMs);
  }

  function stopPlayback() {
    clearTimeout(timer);
    timer = null;
    local = { ...local, playing: false };
  }

  function step(delta) {
    if (!local.episode) return;
    stopPlayback();
    local = { ...local, frame: Math.max(0, Math.min(local.episode.steps.length, local.frame + delta)) };
    render();
  }

  async function compareAll() {
    const { s, run } = ctx();
    const token = ++compareToken;
    local = { ...local, compare: { rows: [], running: true, total: run.iterations.length } };
    render();
    for (const it of run.iterations) {
      if (token !== compareToken) return;
      try {
        const ep = await api.infer({
          env_key: run.env.key, options: run.env.options, policy: it.policy, gamma: run.config.gamma,
          start_state: s.inferStart, seed: s.inferSeed, max_steps: s.inferMaxSteps,
        });
        if (token !== compareToken) return;
        const row = { k: it.index, ret: ep.total_return, disc: ep.discounted_return, len: ep.length, success: ep.success, expected: it.V[ep.start_state] };
        local = { ...local, compare: { ...local.compare, rows: [...local.compare.rows, row] } };
      } catch (err) {
        ui.status(err.message, "error");
        break;
      }
      render();
    }
    local = { ...local, compare: { ...local.compare, running: false } };
    render();
  }

  // ---------------- rendering ----------------

  function shell(run) {
    root.innerHTML = `
      <div class="grid-2">
        <div class="card board-card">
          <div class="card-head" style="width:100%">
            <h3 data-role="title"></h3>
            <div class="row" style="display:flex;gap:10px;align-items:center">
              <span data-role="outcome" class="outcome"></span>
              <div class="seg" role="group" aria-label="Board view">
                <button type="button" data-boardmode="analysis">Analysis</button>
                <button type="button" data-boardmode="gym">Gymnasium</button>
                <button type="button" data-boardmode="both">Side by side</button>
              </div>
            </div>
          </div>
          <div class="boards">
            <div class="board-wrap" data-role="board"></div>
            <div class="gym-wrap" data-role="gym"></div>
          </div>
          <div class="toolbar" style="margin:0">
            <div class="btn-row">
              <button class="btn primary" data-act="start">▶ Start episode</button>
              <button class="btn" data-act="pause">⏸ Pause</button>
              <button class="btn danger" data-act="stop">■ Stop</button>
              <button class="btn sm" data-act="back" title="Step back">⏪</button>
              <button class="btn sm" data-act="fwd" title="Step forward">⏩</button>
              <button class="btn sm ghost" data-act="reset" title="Reset">↺</button>
            </div>
            <label class="speed">speed <input type="range" min="40" max="1500" step="20" data-role="speed"/></label>
          </div>
          <input type="range" min="0" value="0" data-role="scrub" style="width:100%"/>
          <div class="note">Click a cell to choose the start state${run.layout.kind === "taxi" ? " (taxi position; passenger/destination below)" : ""}. Green outline = start.</div>
        </div>
        <div class="side-col">
          <div class="card">
            <h3>Inference settings</h3>
            <div class="field"><label>Policy</label><select data-role="policy"></select></div>
            <div class="field"><label>Start state</label><select data-role="start"></select></div>
            ${run.layout.kind === "taxi" ? `
              <div class="field"><label>Passenger at</label><select data-role="t-pass">${PASSENGER_LABELS.map((l, i) => `<option value="${i}">${l}</option>`).join("")}</select></div>
              <div class="field"><label>Destination</label><select data-role="t-dest">${run.layout.loc_names.map((l, i) => `<option value="${i}">${l}</option>`).join("")}</select></div>` : ""}
            <div class="field"><label>Env seed</label><div class="btn-row" style="flex-wrap:nowrap"><input type="number" min="0" data-role="seed"/><button class="btn sm" data-act="dice" title="Random seed">🎲</button></div></div>
            <div class="field"><label>Max steps</label><input type="number" min="1" max="${MAX_STEPS_LIMIT}" data-role="maxsteps"/></div>
            <div class="btn-row" style="margin-bottom:8px">
              <button class="btn sm" data-act="default-start">Default start</button>
              <button class="btn sm" data-act="random-start">Random start</button>
              <label class="toggle"><input type="checkbox" data-role="show-values"/>Values</label>
              <label class="toggle"><input type="checkbox" data-role="show-arrows"/>Arrows</label>
            </div>
          </div>
          <div class="card"><div class="stats" data-role="stats"></div></div>
          <div class="card"><h3>Step log</h3><div class="log" data-role="log"></div></div>
          <div class="card">
            <div class="card-head"><h3>Every policy from this start</h3>
              <div class="btn-row"><button class="btn sm" data-act="compare">▶ Run all</button><button class="btn sm danger" data-act="compare-stop">■</button></div></div>
            <div data-role="compare" class="note">Plays π₀ … π★ from the same start state and seed.</div>
          </div>
        </div>
      </div>`;
    board = mountBoard(root.querySelector('[data-role="board"]'), run, {
      getView: boardView,
      getMaxWidth: () => gymWidth(),
      onHover: (state, e) => showStateTooltip(e, run, ctx().it, state),
      onClick: (state) => {
        if (run.layout.kind === "taxi") {
          const d = decodeTaxi(state);
          const cur = decodeTaxi(store.get().inferStart);
          setStart(encodeTaxi(d.row, d.col, cur.passenger, cur.destination));
        } else {
          setStart(state);
        }
      },
    });
    root.querySelector(".board-wrap").addEventListener("mouseleave", hideTooltip);
    boardRunId = run.id;
    gym = createGymView(root.querySelector('[data-role="gym"]'), ui);
    gym.mount(run, gymWidth()).then(() => render());
    wire();
  }

  // Delegated once: survives re-rendering of the shell when another run is opened.
  root.addEventListener("click", (e) => {
    const mode = e.target.closest("[data-boardmode]")?.dataset.boardmode;
    if (mode) {
      store.set({ inferBoard: mode });
      gym?.resize(gymWidth(mode));
      board?.refit();
      return;
    }
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (!act || !store.get().run) return;
    const { run } = ctx();
    const handlers = {
      start: startEpisode,
      pause: () => (local.playing ? (stopPlayback(), render()) : play()),
      stop: () => { stopPlayback(); local = { ...local, frame: 0 }; render(); },
      back: () => step(-1),
      fwd: () => step(1),
      reset: () => { stopPlayback(); local = freshLocal(); render(); },
      dice: () => store.set({ inferSeed: Math.floor(Math.random() * 1e6) }),
      "default-start": () => setStart(run.layout.default_start),
      "random-start": () => setStart(run.layout.start_states[Math.floor(Math.random() * run.layout.start_states.length)]),
      compare: compareAll,
      "compare-stop": () => { compareToken += 1; local = { ...local, compare: local.compare && { ...local.compare, running: false } }; render(); },
    };
    handlers[act]?.();
  });

  function wire() {
    const q = (sel) => root.querySelector(sel);
    q('[data-role="policy"]').addEventListener("change", (e) => { stopPlayback(); local = freshLocal(); store.set({ inferIter: +e.target.value }); });
    q('[data-role="start"]').addEventListener("change", (e) => setStart(+e.target.value));
    q('[data-role="seed"]').addEventListener("change", (e) => store.set({ inferSeed: Math.max(0, +e.target.value || 0) }));
    q('[data-role="maxsteps"]').addEventListener("change", (e) => {
      const v = Math.min(MAX_STEPS_LIMIT, Math.max(1, Math.round(+e.target.value || 1)));
      store.set({ inferMaxSteps: v });
    });
    q('[data-role="speed"]').addEventListener("input", (e) => store.set({ inferSpeedMs: 1540 - +e.target.value }));
    q('[data-role="scrub"]').addEventListener("input", (e) => { stopPlayback(); local = { ...local, frame: +e.target.value }; render(); });
    q('[data-role="show-values"]').addEventListener("change", (e) => store.set({ inferShowValues: e.target.checked }));
    q('[data-role="show-arrows"]').addEventListener("change", (e) => store.set({ inferShowArrows: e.target.checked }));
    const taxiSel = (role, field) => q(`[data-role="${role}"]`)?.addEventListener("change", (e) => {
      const d = decodeTaxi(store.get().inferStart);
      const next = { ...d, [field]: +e.target.value };
      if (next.passenger === next.destination) {
        ui.status("Passenger is already at that destination — pick a different one.", "error");
        render();
        return;
      }
      setStart(encodeTaxi(next.row, next.col, next.passenger, next.destination));
    });
    taxiSel("t-pass", "passenger");
    taxiSel("t-dest", "destination");
  }

  function gymWidth(mode = store.get().inferBoard) {
    const avail = root.querySelector(".boards")?.clientWidth || 700;
    return mode === "both" ? Math.max(240, Math.floor(avail / 2) - 28) : Math.min(760, avail);
  }

  function applyBoardMode(mode) {
    root.querySelector('[data-role="board"]').hidden = mode === "gym";
    root.querySelector('[data-role="gym"]').hidden = mode === "analysis";
    root.querySelectorAll("[data-boardmode]").forEach((b) => {
      b.classList.toggle("active", b.dataset.boardmode === mode);
      b.setAttribute("aria-pressed", String(b.dataset.boardmode === mode));
    });
  }

  function render() {
    const { s, run, k, it } = ctx();
    if (boardRunId !== run.id || !root.querySelector('[data-role="board"]')) shell(run);
    const q = (sel) => root.querySelector(sel);
    const done = run.status === "converged";
    q('[data-role="policy"]').innerHTML = run.iterations.map((x, i) =>
      `<option value="${i}" ${i === k ? "selected" : ""}>${done && i === run.optimal_index ? `π★ optimal (π${i})` : `π${i}`} · V(s₀)=${formatValue(x.stats.v_start)}</option>`).join("");
    if (q('[data-role="start"]').options.length !== run.layout.start_states.length) {
      q('[data-role="start"]').innerHTML = run.layout.start_states.map((st) =>
        `<option value="${st}">${escapeHtml(describeState(run.env.key, run.layout, st))}</option>`).join("");
    }
    q('[data-role="start"]').value = String(s.inferStart);
    if (run.layout.kind === "taxi") {
      const d = decodeTaxi(s.inferStart);
      q('[data-role="t-pass"]').value = String(d.passenger);
      q('[data-role="t-dest"]').value = String(d.destination);
    }
    q('[data-role="seed"]').value = String(s.inferSeed);
    q('[data-role="maxsteps"]').value = String(s.inferMaxSteps);
    q('[data-role="speed"]').value = String(1540 - s.inferSpeedMs);
    q('[data-role="show-values"]').checked = s.inferShowValues;
    q('[data-role="show-arrows"]').checked = s.inferShowArrows;
    q('[data-role="title"]').innerHTML = `${done && k === run.optimal_index ? "π★ optimal" : `π<sub>${k}</sub>`} in ${escapeHtml(run.env.gym_id)}`;
    q('[data-act="pause"]').textContent = local.playing ? "⏸ Pause" : "▶ Resume";
    q('[data-act="pause"]').disabled = !local.episode;
    q('[data-act="stop"]').disabled = !local.episode;
    q('[data-act="start"]').innerHTML = local.loading ? '<span class="spinner"></span> Running…' : "▶ Start episode";
    const scrub = q('[data-role="scrub"]');
    scrub.max = String(local.episode?.steps.length ?? 0);
    scrub.value = String(local.frame);
    applyBoardMode(s.inferBoard);
    board.redraw();
    gym?.update({ episode: local.episode, frame: local.frame, startState: s.inferStart,
                  playing: local.playing, speedMs: s.inferSpeedMs });
    renderStats(run, it);
    renderLog(run);
    renderCompare(run);
  }

  function renderStats(run, it) {
    const ep = local.episode;
    const steps = ep ? ep.steps.slice(0, local.frame) : [];
    const ret = steps.reduce((a, st) => a + st.reward, 0);
    const finished = ep && local.frame === ep.steps.length;
    const outcome = !ep ? "" : !finished ? "running…" : ep.success ? "✓ reached goal" : ep.truncated ? "⏱ truncated (max steps)" : "✗ failed";
    const cls = !finished ? "" : ep.success ? "good" : ep.truncated ? "warn" : "bad";
    const outEl = root.querySelector('[data-role="outcome"]');
    outEl.textContent = outcome;
    outEl.className = `outcome ${cls}`;
    const start = ep ? ep.start_state : store.get().inferStart;
    const stat = (l, v, c = "") => `<div class="stat"><div class="l">${l}</div><div class="v ${c}">${v}</div></div>`;
    root.querySelector('[data-role="stats"]').innerHTML = [
      stat("Step", ep ? `${local.frame} / ${ep.steps.length}` : "–"),
      stat("Return so far", ep ? formatValue(ret) : "–"),
      stat("Episode return", finished ? formatValue(ep.total_return) : "–", cls),
      stat("Discounted G", finished ? formatValue(ep.discounted_return) : "–"),
      stat("Expected V(s₀)", formatValue(it.V[start]), "star"),
      stat("Start", `s${start}`),
    ].join("");
  }

  function renderLog(run) {
    const ep = local.episode;
    const log = root.querySelector('[data-role="log"]');
    if (!ep) {
      log.innerHTML = '<div class="note" style="padding:8px">Press Start episode to roll out the policy in gymnasium.</div>';
      return;
    }
    let cum = 0;
    log.innerHTML = `<table><thead><tr><th>t</th><th>s</th><th>action</th><th>r</th><th>G</th><th>s′</th><th>p</th></tr></thead><tbody>${
      ep.steps.map((st, i) => {
        cum += st.reward;
        return `<tr data-frame="${i + 1}" class="${i + 1 === local.frame ? "current" : ""}"><td>${st.t}</td><td>${st.state}</td><td>${escapeHtml(st.action_name)}</td><td>${formatValue(st.reward)}</td><td>${formatValue(cum)}</td><td>${st.next_state}${st.terminated ? " ⏹" : ""}</td><td>${st.prob.toFixed(2)}</td></tr>`;
      }).join("")}</tbody></table>`;
    log.querySelectorAll("[data-frame]").forEach((tr) => tr.addEventListener("click", () => {
      stopPlayback();
      local = { ...local, frame: +tr.dataset.frame };
      render();
    }));
    log.querySelector(".current")?.scrollIntoView({ block: "nearest" });
  }

  function renderCompare(run) {
    const c = local.compare;
    const el = root.querySelector('[data-role="compare"]');
    if (!c) return;
    el.className = "";
    el.innerHTML = `
      ${c.running ? `<div class="progress" style="margin-bottom:6px"><div style="width:${(100 * c.rows.length) / c.total}%"></div></div>` : ""}
      <div class="table-wrap" style="max-height:260px"><table><thead><tr><th>policy</th><th>return</th><th>disc. G</th><th>E[V(s₀)]</th><th>steps</th><th>goal</th><th></th></tr></thead><tbody>${
        c.rows.map((r) => `<tr><td>π${r.k}</td><td>${formatValue(r.ret)}</td><td>${formatValue(r.disc)}</td><td>${formatValue(r.expected)}</td><td>${r.len}</td><td>${r.success ? "✓" : "✗"}</td><td><button class="btn sm" data-replay="${r.k}">▶</button></td></tr>`).join("")
      }</tbody></table></div>`;
    el.querySelectorAll("[data-replay]").forEach((b) => b.addEventListener("click", () => {
      store.set({ inferIter: +b.dataset.replay });
      startEpisode();
    }));
  }

  return {
    update(s) {
      if (!s.run?.iterations.length) return;
      render();
    },
    deactivate() {
      stopPlayback();
      compareToken += 1;
    },
    autoStart() {
      startEpisode();
    },
  };
}
