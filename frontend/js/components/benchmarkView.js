// Benchmark: Monte-Carlo evaluation of every policy (one request per policy -> progress + Stop).

import { api } from "../api.js";
import { formatValue } from "../colors.js";
import { describeState } from "../taxi.js";
import { renderChart } from "./chart.js";
import { SERIES } from "../theme.js";
import { escapeHtml } from "./tooltip.js";

export function createBenchmarkView(root, store, ui) {
  let token = 0;
  let result = null; // { runId, rows, running, total, params }

  root.addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "run") run();
    if (act === "stop") {
      token += 1;
      result = result && { ...result, running: false };
      render(store.get());
    }
  });

  function readParams() {
    const q = (r) => root.querySelector(`[data-role="${r}"]`);
    const startRaw = q("b-start").value;
    return {
      episodes: Math.min(1000, Math.max(1, Math.round(+q("b-episodes").value || 1))),
      seed: Math.max(0, Math.round(+q("b-seed").value || 0)),
      max_steps: Math.min(5000, Math.max(1, Math.round(+q("b-max").value || 1))),
      start_state: startRaw === "env" ? null : +startRaw,
    };
  }

  async function run() {
    const s = store.get();
    const r = s.run;
    const params = readParams();
    store.set({ benchParams: params });
    const my = ++token;
    result = { runId: r.id, rows: [], running: true, total: r.iterations.length, params };
    render(store.get());
    for (const it of r.iterations) {
      if (my !== token) return;
      try {
        const res = await api.benchmark({
          env_key: r.env.key, options: r.env.options, policies: [it.policy], gamma: r.config.gamma, ...params,
        });
        if (my !== token) return;
        const row = { ...res.rows[0], iteration: it.index,
                      expected_value: params.start_state == null ? null : it.V[params.start_state] };
        result = { ...result, rows: [...result.rows, row] };
      } catch (err) {
        ui.status(err.message, "error");
        break;
      }
      render(store.get());
    }
    result = { ...result, running: false };
    render(store.get());
  }

  function render(s) {
    const r = s.run;
    if (!r?.iterations.length) {
      root.innerHTML = "";
      return;
    }
    const p = s.benchParams;
    const current = result?.runId === r.id ? result : null;
    const startOptions = [`<option value="env">gymnasium reset() distribution</option>`,
      ...r.layout.start_states.map((st) => `<option value="${st}" ${p.start_state === st ? "selected" : ""}>${escapeHtml(describeState(r.env.key, r.layout, st))}</option>`)].join("");
    root.innerHTML = `
      <div class="card" style="margin-bottom:12px">
        <div class="toolbar" style="margin:0">
          <div class="field" style="grid-template-columns:auto 90px;margin:0"><label>Episodes / policy</label><input type="number" min="1" max="1000" data-role="b-episodes" value="${p.episodes}"/></div>
          <div class="field" style="grid-template-columns:auto 90px;margin:0"><label>Seed</label><input type="number" min="0" data-role="b-seed" value="${p.seed}"/></div>
          <div class="field" style="grid-template-columns:auto 90px;margin:0"><label>Max steps</label><input type="number" min="1" max="5000" data-role="b-max" value="${p.max_steps}"/></div>
          <div class="field" style="grid-template-columns:auto 260px;margin:0"><label>Start</label><select data-role="b-start">${startOptions}</select></div>
          <button class="btn primary" data-act="run" ${current?.running ? "disabled" : ""}>▶ Start benchmark</button>
          <button class="btn danger" data-act="stop" ${current?.running ? "" : "disabled"}>■ Stop</button>
        </div>
        ${current?.running ? `<div class="progress" style="margin-top:10px"><div style="width:${(100 * current.rows.length) / current.total}%"></div></div>` : ""}
        <div class="note" style="margin-top:6px">Every policy is rolled out in the real gymnasium env with the same seeds (common random numbers), so the policies are compared fairly.</div>
      </div>
      ${current ? resultsHtml(current) : '<div class="note">Press Start benchmark to evaluate π₀ … π★.</div>'}`;
    if (p.start_state == null) root.querySelector('[data-role="b-start"]').value = "env";
    if (current?.rows.length) drawCharts(current);
  }

  function resultsHtml(c) {
    return `
      <div class="grid-2" style="grid-template-columns:minmax(0,1fr) minmax(0,1fr)">
        <div class="card"><h3>Success rate per policy</h3><div data-role="c-success"></div></div>
        <div class="card"><h3>Mean return (±std) vs. expected V(s₀)</h3><div data-role="c-return"></div></div>
      </div>
      <div class="table-wrap" style="margin-top:12px"><table><thead><tr>
        <th>policy</th><th>success</th><th>mean return</th><th>std</th><th>mean disc. G</th><th>E[V(s₀)]</th><th>mean length</th><th>truncated</th></tr></thead>
        <tbody>${c.rows.map((r) => `<tr><td>π${r.iteration}</td><td>${(100 * r.success_rate).toFixed(1)}%</td><td>${formatValue(r.mean_return)}</td><td>${formatValue(r.std_return)}</td><td>${formatValue(r.mean_discounted_return)}</td><td>${formatValue(r.expected_value)}</td><td>${r.mean_length.toFixed(1)}</td><td>${(100 * r.truncation_rate).toFixed(0)}%</td></tr>`).join("")}</tbody>
      </table></div>`;
  }

  function drawCharts(c) {
    const xf = (i) => `π${c.rows[i]?.iteration ?? i}`;
    renderChart(root.querySelector('[data-role="c-success"]'), {
      series: [{ label: "success rate", color: SERIES.good, type: "bar", values: c.rows.map((r) => r.success_rate) }],
      xFormat: xf, yMin: 0, yMax: 1,
    });
    const hasExpected = c.rows.some((r) => r.expected_value != null);
    renderChart(root.querySelector('[data-role="c-return"]'), {
      series: [
        { label: "mean return", color: SERIES.primary, values: c.rows.map((r) => r.mean_return) },
        { label: "mean discounted G", color: SERIES.secondary, values: c.rows.map((r) => r.mean_discounted_return) },
        ...(hasExpected ? [{ label: "V^π(s₀) (model)", color: SERIES.star, dashed: true, values: c.rows.map((r) => r.expected_value) }] : []),
      ],
      xFormat: xf,
    });
  }

  return {
    update: render,
    deactivate() { /* keep running in background; results stay attached to the run */ },
  };
}
