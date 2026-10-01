// State inspector: how V(s), Q(s,·) and π(s) of one state evolved across all policy iterations.

import { formatValue } from "../colors.js";
import { describeState } from "../taxi.js";
import { renderChart } from "./chart.js";
import { escapeHtml } from "./tooltip.js";

const Q_COLORS = ["#5ad1e6", "#8b7bff", "#4fd18b", "#ffb547", "#ff6b7a", "#f6c544"];

export function renderInspector(root, run, state, selectedIter, onSelect) {
  if (state == null) {
    root.innerHTML = '<h3>State inspector</h3><div class="note">Click any cell on the board to follow that state across every policy.</div>';
    return;
  }
  const its = run.iterations;
  const names = run.env.action_names;
  const terminal = run.layout.terminal_states.includes(state);
  root.innerHTML = `
    <div class="card-head"><h3>State inspector · ${escapeHtml(describeState(run.env.key, run.layout, state))}</h3></div>
    ${terminal ? '<div class="note">Terminal state: V = 0 for every policy.</div>' : ""}
    <div data-role="v-chart"></div>
    <div class="table-wrap" style="margin-top:8px;max-height:none">
      <table><thead><tr><th>policy</th>${its.map((it) => `<th class="${it.index === selectedIter ? "sel" : ""}">π${it.index}</th>`).join("")}</tr></thead>
      <tbody>
        <tr><th>π(s)</th>${its.map((it, k) => `<td class="${k > 0 && its[k - 1].policy[state] !== it.policy[state] ? "changed" : ""} ${k === selectedIter ? "sel" : ""}">${terminal ? "–" : escapeHtml(names[it.policy[state]])}</td>`).join("")}</tr>
        <tr><th>V(s)</th>${its.map((it, k) => `<td class="${k === selectedIter ? "sel" : ""}">${formatValue(it.V[state])}</td>`).join("")}</tr>
        ${names.map((n, a) => `<tr><th>Q(s,${escapeHtml(n)})</th>${its.map((it, k) => `<td class="${k === selectedIter ? "sel" : ""}">${formatValue(it.Q[state][a])}</td>`).join("")}</tr>`).join("")}
      </tbody></table>
    </div>`;
  renderChart(root.querySelector('[data-role="v-chart"]'), {
    series: [
      { label: "V(s)", color: "#e7ecf7", values: its.map((it) => it.V[state]) },
      ...names.map((n, a) => ({ label: `Q(s,${n})`, color: Q_COLORS[a % Q_COLORS.length], values: its.map((it) => it.Q[state][a]), dashed: true })),
    ],
    selected: selectedIter, onSelect, xFormat: (i) => `π${i}`, height: 160,
  });
}
