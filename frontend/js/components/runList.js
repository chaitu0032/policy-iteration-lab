// Saved runs (IndexedDB) for the current environment: open, export, delete, import.

import { escapeHtml } from "./tooltip.js";
import { formatValue } from "../colors.js";

function optionSummary(options) {
  return Object.entries(options)
    .filter(([k]) => k !== "custom_map")
    .map(([k, v]) => `${k}=${typeof v === "number" ? +v.toFixed(3) : v}`)
    .join(" ");
}

export function renderRunList(root, { runs, envKey, activeId, onOpen, onDelete, onExport }) {
  const mine = runs.filter((r) => r.env_key === envKey);
  if (!mine.length) {
    root.innerHTML = '<div class="note">No runs yet for this environment. Runs are stored in this browser.</div>';
    return;
  }
  root.innerHTML = mine.map((r) => `
    <div class="run-item ${r.id === activeId ? "active" : ""}" data-id="${r.id}" title="${escapeHtml(optionSummary(r.options))}">
      <div class="title">${escapeHtml(r.title)} <span class="badge ${r.status}">${escapeHtml(r.status)}</span></div>
      <div class="btn-row">
        <button class="btn ghost sm" data-act="export" title="Download JSON">⤓</button>
        <button class="btn ghost sm" data-act="delete" title="Delete run">✕</button>
      </div>
      <div class="meta">γ=${r.gamma} · ${r.eval_mode} · ${r.n_iterations} policies · V(s₀)=${formatValue(r.v_start)}</div>
      <div class="meta">${new Date(r.created_at * 1000).toLocaleString()}</div>
      <div class="meta" style="grid-column:1/-1">${escapeHtml(optionSummary(r.options))}</div>
    </div>`).join("");

  root.querySelectorAll(".run-item").forEach((node) => {
    node.addEventListener("click", (e) => {
      const act = e.target.closest("button")?.dataset.act;
      const id = node.dataset.id;
      if (act === "delete") onDelete(id);
      else if (act === "export") onExport(id);
      else onOpen(id);
    });
  });
}
