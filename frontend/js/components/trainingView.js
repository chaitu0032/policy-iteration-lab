// Training replay: iteration strip, playback, value/policy board with eval-sweep scrubber, charts, inspector.

import { formatValue, gradientCss, valueRange } from "../colors.js";
import { mountBoard } from "../renderers/board.js";
import { PASSENGER_LABELS } from "../taxi.js";
import { renderChart } from "./chart.js";
import { SERIES } from "../theme.js";
import { renderInspector } from "./stateInspector.js";
import { escapeHtml, hideTooltip, showStateTooltip } from "./tooltip.js";

export function sharedRange(run, iteration, scale) {
  if (!iteration) return [0, 1];
  if (scale === "iteration") return valueRange([iteration.V]);
  return valueRange(run.iterations.map((it) => it.V));
}

export function sliceSelectorHtml(layout, slice) {
  const opts = (labels, sel) => labels.map((l, i) => `<option value="${i}" ${i === sel ? "selected" : ""}>${escapeHtml(l)}</option>`).join("");
  return `<span class="slice">Taxi slice · passenger
      <select data-slice="passenger">${opts(PASSENGER_LABELS, slice.passenger)}</select>
      destination <select data-slice="destination">${opts(layout.loc_names, slice.destination)}</select></span>`;
}

export function createTrainingView(root, store, actions) {
  let board = null;
  let boardRunId = null;
  let playTimer = null;
  let sweepTimer = null;

  const current = () => {
    const s = store.get();
    return { s, run: s.run, it: s.run?.iterations[s.selectedIter] };
  };

  function shell(run) {
    const isTaxi = run.layout.kind === "taxi";
    root.innerHTML = `
      <div class="strip" data-role="strip"></div>
      <div class="toolbar">
        <div class="btn-row">
          <button class="btn sm" data-act="first" title="First policy">⏮</button>
          <button class="btn sm" data-act="prev" title="Previous policy">◀</button>
          <button class="btn sm primary" data-act="play">▶ Replay</button>
          <button class="btn sm" data-act="next" title="Next policy">▶</button>
          <button class="btn sm star" data-act="last" title="Optimal policy">★</button>
        </div>
        <label class="speed">speed <input type="range" min="150" max="2500" step="50" value="${store.get().replayMs}" data-role="speed"/></label>
        <span class="sep"></span>
        <label class="toggle"><input type="checkbox" data-toggle="values"/>Values</label>
        <label class="toggle"><input type="checkbox" data-toggle="arrows"/>Policy arrows</label>
        <label class="toggle"><input type="checkbox" data-toggle="changes"/>Changed actions</label>
        <label class="toggle"><input type="checkbox" data-toggle="follow"/>Follow live</label>
        <select data-role="scale" style="width:auto" title="Colour scale">
          <option value="global">scale: all policies</option><option value="iteration">scale: this policy</option>
        </select>
        ${isTaxi ? sliceSelectorHtml(run.layout, store.get().taxiSlice) : ""}
      </div>
      <div class="grid-2">
        <div class="card board-card">
          <div class="card-head" style="width:100%"><h3 data-role="board-title"></h3>
            <div class="btn-row">
              <button class="btn sm" data-act="infer">▶ Play this policy</button>
            </div></div>
          <div class="sweep-row" style="width:100%">
            <button class="btn sm" data-act="sweep-play" title="Animate policy evaluation">▶ eval</button>
            <input type="range" min="0" value="0" data-role="sweep"/>
            <span class="mono muted" data-role="sweep-label" style="font-size:12px"></span>
          </div>
          <div class="board-wrap" data-role="board"></div>
          <div class="legend"><span data-role="lo"></span><span class="bar" data-role="bar"></span><span data-role="hi"></span></div>
          <div class="note">Hover a cell for V and Q(s,·); click it to inspect it across all policies. Amber outline = action changed from π<sub>k−1</sub>.</div>
        </div>
        <div class="side-col">
          <div class="card"><div class="stats" data-role="stats"></div><ul class="warn-list" data-role="warnings"></ul></div>
          <div class="card"><h3>Value of start state and mean V across policies</h3><div data-role="chart-v"></div></div>
          <div class="card"><h3>States whose action changed (π<sub>k</sub> vs π<sub>k−1</sub>)</h3><div data-role="chart-c"></div></div>
          <div class="card"><h3>Bellman sweeps needed to evaluate each policy</h3><div data-role="chart-s"></div></div>
          <div class="card"><h3 data-role="resid-title">Policy-evaluation residual max|ΔV| (log)</h3><div data-role="chart-r"></div></div>
          <div class="card" data-role="inspector"></div>
        </div>
      </div>`;
    wireShell(run);
  }

  function wireShell(run) {
    const q = (sel) => root.querySelector(sel);
    board = mountBoard(q('[data-role="board"]'), run, {
      getView: boardView,
      onHover: (state, e) => {
        const { it } = current();
        showStateTooltip(e, run, it, state, boardView().values);
      },
      onClick: (state) => store.set({ inspectState: state }),
    });
    boardRunId = run.id;
    root.addEventListener("click", onClick);
    q('[data-role="speed"]').addEventListener("input", (e) => store.set({ replayMs: +e.target.value }));
    q('[data-role="scale"]').addEventListener("change", (e) => store.set({ display: { ...store.get().display, scale: e.target.value } }));
    q('[data-role="sweep"]').addEventListener("input", (e) => store.set({ sweepIdx: +e.target.value }));
    root.querySelectorAll("[data-toggle]").forEach((cb) => cb.addEventListener("change", () => {
      store.set({ display: { ...store.get().display, [cb.dataset.toggle]: cb.checked } });
    }));
    root.querySelectorAll("[data-slice]").forEach((sel) => sel.addEventListener("change", () => {
      store.set({ taxiSlice: { ...store.get().taxiSlice, [sel.dataset.slice]: +sel.value } });
    }));
    q(".board-wrap").addEventListener("mouseleave", hideTooltip);
  }

  function onClick(e) {
    const act = e.target.closest("[data-act]")?.dataset.act;
    const chip = e.target.closest("[data-iter]");
    const { s, run } = current();
    if (chip) return selectIter(+chip.dataset.iter);
    if (!act || !run) return undefined;
    const last = run.iterations.length - 1;
    const moves = { first: 0, prev: s.selectedIter - 1, next: s.selectedIter + 1, last };
    if (act in moves) return selectIter(Math.max(0, Math.min(last, moves[act])));
    if (act === "play") return togglePlay();
    if (act === "sweep-play") return toggleSweepPlay();
    if (act === "infer") return actions.playPolicy(s.selectedIter);
    return undefined;
  }

  function selectIter(k) {
    store.set({ selectedIter: k, sweepIdx: null });
  }

  function togglePlay() {
    if (playTimer) return stopPlay();
    const { run, s } = current();
    if (s.selectedIter >= run.iterations.length - 1) selectIter(0);
    const tick = () => {
      const { run: r, s: st } = current();
      if (!r || st.selectedIter >= r.iterations.length - 1) return stopPlay();
      selectIter(st.selectedIter + 1);
      playTimer = setTimeout(tick, store.get().replayMs);
      return undefined;
    };
    playTimer = setTimeout(tick, store.get().replayMs);
    setPlayLabel();
    return undefined;
  }

  function stopPlay() {
    clearTimeout(playTimer);
    playTimer = null;
    setPlayLabel();
  }

  function setPlayLabel() {
    const btn = root.querySelector('[data-act="play"]');
    if (btn) btn.textContent = playTimer ? "■ Stop replay" : "▶ Replay";
  }

  function toggleSweepPlay() {
    const btn = root.querySelector('[data-act="sweep-play"]');
    if (sweepTimer) {
      clearInterval(sweepTimer);
      sweepTimer = null;
      btn.textContent = "▶ eval";
      return;
    }
    const { it } = current();
    if (!it) return;
    store.set({ sweepIdx: 0 });
    btn.textContent = "■ eval";
    sweepTimer = setInterval(() => {
      const { s, it: cur } = current();
      const next = (s.sweepIdx ?? 0) + 1;
      if (!cur || next >= cur.eval_snapshots.length) {
        clearInterval(sweepTimer);
        sweepTimer = null;
        btn.textContent = "▶ eval";
        store.set({ sweepIdx: null });
        return;
      }
      store.set({ sweepIdx: next });
    }, Math.max(60, store.get().replayMs / 4));
  }

  function boardView() {
    const { s, run, it } = current();
    if (!it) return {};
    const snap = s.sweepIdx != null ? it.eval_snapshots[s.sweepIdx] : null;
    const values = snap ? snap.V : it.V;
    return {
      values, range: sharedRange(run, it, s.display.scale), policy: it.policy,
      changed: new Set(it.changed_states), showValues: s.display.values, showArrows: s.display.arrows,
      showChanges: s.display.changes, selected: s.inspectState, slice: s.taxiSlice, mode: "analysis",
    };
  }

  function renderStrip(run, s) {
    const strip = root.querySelector('[data-role="strip"]');
    const done = run.status !== "running";
    strip.innerHTML = run.iterations.map((it, k) => {
      const optimal = done && k === run.optimal_index && run.status === "converged";
      return `<button class="it-chip ${k === s.selectedIter ? "active" : ""} ${optimal ? "optimal" : ""}" data-iter="${k}">
        <div class="k">${optimal ? "π★" : `π${k}`}</div>
        <div class="d">Δ${it.stats.n_changed} · ${it.eval_sweeps}sw</div></button>`;
    }).join("") + (run.status === "running" ? '<div class="it-chip"><span class="spinner"></span></div>' : "");
    strip.querySelector(".active")?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }

  function renderStats(run, it, s) {
    const optimal = run.status === "converged" && it.index === run.optimal_index;
    const stat = (l, v, cls = "") => `<div class="stat"><div class="l">${l}</div><div class="v ${cls}">${v}</div></div>`;
    root.querySelector('[data-role="stats"]').innerHTML = [
      stat("Policy", optimal ? "π★ optimal" : `π${it.index}`, optimal ? "star" : ""),
      stat("V(s₀)", formatValue(it.stats.v_start)),
      stat("mean V", formatValue(it.stats.v_mean)),
      stat("States changed", it.stats.n_changed),
      stat("Eval sweeps", it.eval_sweeps, it.eval_converged ? "" : "bad"),
      stat("Run status", escapeHtml(run.status), run.status === "converged" ? "good" : ""),
    ].join("");
    root.querySelector('[data-role="warnings"]').innerHTML = (s.trainWarnings || []).map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  }

  function renderCharts(run, it, s) {
    const its = run.iterations;
    const select = (i) => selectIter(i);
    const xf = (i) => `π${i}`;
    renderChart(root.querySelector('[data-role="chart-v"]'), {
      series: [
        { label: "V(s₀)", color: SERIES.primary, values: its.map((x) => x.stats.v_start) },
        { label: "mean V over start states", color: SERIES.secondary, values: its.map((x) => x.stats.v_mean), dashed: true },
      ], selected: s.selectedIter, onSelect: select, xFormat: xf,
    });
    renderChart(root.querySelector('[data-role="chart-c"]'), {
      series: [{ label: "# states changed", color: SERIES.warn, type: "bar", values: its.map((x) => x.stats.n_changed) }],
      selected: s.selectedIter, onSelect: select, xFormat: xf, height: 130,
    });
    renderChart(root.querySelector('[data-role="chart-s"]'), {
      series: [{ label: `eval sweeps (${run.config.eval_mode})`, color: SERIES.good, type: "bar", values: its.map((x) => x.eval_sweeps) }],
      selected: s.selectedIter, onSelect: select, xFormat: xf, height: 130,
    });
    root.querySelector('[data-role="resid-title"]').textContent = `Policy-evaluation residual max|ΔV| for π${it.index} (log scale)`;
    renderChart(root.querySelector('[data-role="chart-r"]'), {
      series: [{ label: `max|ΔV| per sweep · θ=${run.config.theta}`, color: SERIES.bad, values: it.eval_deltas.map((d) => Math.max(d, 1e-16)) }],
      log: true, height: 150, xFormat: (i) => i + 1,
    });
  }

  function renderBoardMeta(run, it, s) {
    const snaps = it.eval_snapshots;
    const sweep = root.querySelector('[data-role="sweep"]');
    sweep.max = String(Math.max(0, snaps.length - 1));
    sweep.value = String(s.sweepIdx ?? snaps.length - 1);
    const shown = snaps[s.sweepIdx ?? snaps.length - 1];
    root.querySelector('[data-role="sweep-label"]').textContent =
      `${s.sweepIdx == null ? "converged V" : "sweep"} ${shown?.sweep ?? "–"} / ${it.eval_sweeps}`;
    const optimal = run.status === "converged" && it.index === run.optimal_index;
    root.querySelector('[data-role="board-title"]').innerHTML =
      `${optimal ? "π★ (optimal)" : `π<sub>${it.index}</sub>`} · V<sup>π${it.index}</sup>${s.sweepIdx != null ? " (during evaluation)" : ""}`;
    const range = sharedRange(run, it, s.display.scale);
    root.querySelector('[data-role="bar"]').style.background = gradientCss(range);
    root.querySelector('[data-role="lo"]').textContent = formatValue(range[0]);
    root.querySelector('[data-role="hi"]').textContent = formatValue(range[1]);
    root.querySelectorAll("[data-toggle]").forEach((cb) => {
      cb.checked = cb.dataset.toggle === "follow" ? s.followLive : !!s.display[cb.dataset.toggle];
    });
    root.querySelector('[data-role="scale"]').value = s.display.scale;
    root.querySelectorAll("[data-slice]").forEach((sel) => { sel.value = String(s.taxiSlice[sel.dataset.slice]); });
  }

  return {
    update(s) {
      const { run } = s;
      if (!run || !run.iterations.length) {
        root.innerHTML = '<div class="note"><span class="spinner"></span> Waiting for π₀ to be evaluated…</div>';
        boardRunId = null;
        return;
      }
      if (boardRunId !== run.id || !root.querySelector('[data-role="board"]')) {
        root.removeEventListener("click", onClick);
        shell(run);
      }
      const it = run.iterations[s.selectedIter] || run.iterations[run.iterations.length - 1];
      renderStrip(run, s);
      renderStats(run, it, s);
      renderBoardMeta(run, it, s);
      board.redraw();
      renderCharts(run, it, s);
      renderInspector(root.querySelector('[data-role="inspector"]'), run, s.inspectState, s.selectedIter, (k) => selectIter(k));
    },
    stop() { stopPlay(); },
    refit() { board?.refit(); },
  };
}
