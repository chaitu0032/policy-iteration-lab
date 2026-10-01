// State tooltip: shows V(s), the policy action and the Q(s, ·) bar chart for the shown iteration.

import { formatValue } from "../colors.js";
import { describeState } from "../taxi.js";

const tip = () => document.getElementById("tooltip");

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function showStateTooltip(event, run, iteration, state, values) {
  const el = tip();
  if (state == null || !iteration) {
    el.hidden = true;
    return;
  }
  const terminal = run.layout.terminal_states.includes(state);
  const q = iteration.Q[state];
  const action = iteration.policy[state];
  const lo = Math.min(...q);
  const hi = Math.max(...q);
  const span = hi - lo || 1;
  const rows = terminal
    ? '<div class="muted">terminal state</div>'
    : q.map((qv, a) => `
      <div class="q-row ${a === action ? "best" : ""}">
        <span>${a === action ? "▸ " : ""}${escapeHtml(run.env.action_names[a])}</span>
        <span class="q-bar" style="width:${(8 + 92 * (qv - lo) / span).toFixed(0)}%"></span>
        <span>${formatValue(qv)}</span>
      </div>`).join("");
  el.innerHTML = `
    <div class="t-head">${escapeHtml(describeState(run.env.key, run.layout, state))}</div>
    <div>V = <b>${formatValue((values || iteration.V)[state])}</b> · π${iteration.index}(s) = <b>${terminal ? "–" : escapeHtml(run.env.action_names[action])}</b></div>
    <div class="muted" style="margin:4px 0 2px">Q<sup>π${iteration.index}</sup>(s, a)</div>
    ${rows}`;
  el.hidden = false;
  const pad = 14;
  const { innerWidth: vw, innerHeight: vh } = window;
  const rect = el.getBoundingClientRect();
  el.style.left = `${Math.min(vw - rect.width - 8, event.clientX + pad)}px`;
  el.style.top = `${Math.min(vh - rect.height - 8, event.clientY + pad)}px`;
}

export function hideTooltip() {
  tip().hidden = true;
}

export { escapeHtml };
