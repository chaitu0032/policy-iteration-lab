// Policy table: actions (or values) of every state under every policy pi_0 .. pi_K.

import { formatValue, textColorFor, valueColor } from "../colors.js";
import { decodeTaxi } from "../taxi.js";
import { sharedRange, sliceSelectorHtml } from "./trainingView.js";
import { escapeHtml } from "./tooltip.js";

const GLYPHS = {
  Left: "←", Right: "→", Up: "↑", Down: "↓", North: "↑", South: "↓", East: "→", West: "←", Pickup: "P", Dropoff: "D",
};

function stateLabel(run, s) {
  if (run.layout.kind === "taxi") {
    const d = decodeTaxi(s);
    return `s${s} (${d.row},${d.col})`;
  }
  return `s${s} (${Math.floor(s / run.layout.cols)},${s % run.layout.cols})`;
}

function visibleStates(run, s) {
  // Only states an agent can occupy and act in (no terminal / cliff states).
  let states = [...run.layout.start_states];
  if (run.layout.kind === "taxi" && s.tableScope === "slice") {
    states = states.filter((x) => {
      const d = decodeTaxi(x);
      return d.passenger === s.taxiSlice.passenger && d.destination === s.taxiSlice.destination;
    });
  }
  if (s.tableChangedOnly) {
    states = states.filter((x) => run.iterations.some((it) => it.changed_states.includes(x)));
  }
  return states;
}

export function createPolicyTable(root, store, actions) {
  root.addEventListener("change", (e) => {
    const t = e.target;
    if (t.dataset.slice) store.set({ taxiSlice: { ...store.get().taxiSlice, [t.dataset.slice]: +t.value } });
    if (t.dataset.role === "mode") store.set({ tableMode: t.value });
    if (t.dataset.role === "scope") store.set({ tableScope: t.value });
    if (t.dataset.role === "changed") store.set({ tableChangedOnly: t.checked });
  });
  root.addEventListener("click", (e) => {
    const th = e.target.closest("[data-iter]");
    if (th) actions.openIteration(+th.dataset.iter);
    const row = e.target.closest("[data-state]");
    if (row) store.set({ inspectState: +row.dataset.state });
  });

  return {
    update(s) {
      const { run } = s;
      if (!run?.iterations.length) {
        root.innerHTML = "";
        return;
      }
      const states = visibleStates(run, s);
      const range = sharedRange(run, run.iterations[0], "global");
      const names = run.env.action_names;
      const done = run.status === "converged";
      const cell = (it, k, st) => {
        const changed = k > 0 && run.iterations[k - 1].policy[st] !== it.policy[st];
        const sel = k === s.selectedIter ? "sel" : "";
        if (s.tableMode === "values") {
          const v = it.V[st];
          return `<td class="${changed ? "changed" : ""}" style="background:${valueColor(v, range, 0.85)};color:${textColorFor(v, range)}">${formatValue(v)}</td>`;
        }
        const name = names[it.policy[st]];
        return `<td class="act ${changed ? "changed" : ""} ${sel}" title="${escapeHtml(name)}">${GLYPHS[name] || escapeHtml(name)}</td>`;
      };
      root.innerHTML = `
        <div class="toolbar">
          <select data-role="mode" style="width:auto">
            <option value="actions" ${s.tableMode === "actions" ? "selected" : ""}>Show actions π(s)</option>
            <option value="values" ${s.tableMode === "values" ? "selected" : ""}>Show values V(s)</option>
          </select>
          <label class="toggle"><input type="checkbox" data-role="changed" ${s.tableChangedOnly ? "checked" : ""}/>Only states whose action changed</label>
          ${run.layout.kind === "taxi" ? `<select data-role="scope" style="width:auto">
              <option value="slice" ${s.tableScope === "slice" ? "selected" : ""}>Current slice</option>
              <option value="all" ${s.tableScope === "all" ? "selected" : ""}>All 500 states</option></select>
              ${sliceSelectorHtml(run.layout, s.taxiSlice)}` : ""}
          <span class="note">${states.length} states · amber = action changed vs previous policy · click a header to open that policy</span>
        </div>
        <div class="table-wrap"><table>
          <thead><tr><th>state</th>${run.iterations.map((it, k) => {
            const optimal = done && k === run.optimal_index;
            return `<th data-iter="${k}" class="${optimal ? "optimal" : ""} ${k === s.selectedIter ? "sel" : ""}" style="cursor:pointer">${optimal ? "π★" : `π${k}`}</th>`;
          }).join("")}</tr></thead>
          <tbody>${states.map((st) => `<tr data-state="${st}" class="${st === s.inspectState ? "current" : ""}"><th>${stateLabel(run, st)}</th>${run.iterations.map((it, k) => cell(it, k, st)).join("")}</tr>`).join("")}</tbody>
        </table></div>`;
    },
  };
}
