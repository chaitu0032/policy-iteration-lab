// Small multiples: V^{pi_k} heatmap + pi_k arrows for every stored policy, on one shared colour scale.

import { formatValue, gradientCss } from "../colors.js";
import { createRenderer } from "../renderers/board.js";
import { sharedRange, sliceSelectorHtml } from "./trainingView.js";

const MINI_WIDTH = 230;

export function createValueGallery(root, store, actions) {
  root.addEventListener("click", (e) => {
    const play = e.target.closest("[data-play]");
    if (play) {
      actions.playPolicy(+play.dataset.play);
      return;
    }
    const card = e.target.closest("[data-iter]");
    if (card) actions.openIteration(+card.dataset.iter);
  });
  root.addEventListener("change", (e) => {
    const sel = e.target.closest("[data-slice]");
    if (sel) store.set({ taxiSlice: { ...store.get().taxiSlice, [sel.dataset.slice]: +sel.value } });
    if (e.target.dataset.role === "g-arrows") store.set({ galleryArrows: e.target.checked });
  });

  return {
    update(s) {
      const { run } = s;
      if (!run?.iterations.length) {
        root.innerHTML = "";
        return;
      }
      const range = sharedRange(run, run.iterations[0], "global");
      const done = run.status === "converged";
      root.innerHTML = `
        <div class="toolbar">
          <span class="note">Every policy π<sub>k</sub> with its value function V<sup>π<sub>k</sub></sup>. Shared colour scale:</span>
          <span class="legend" style="width:260px"><span>${formatValue(range[0])}</span><span class="bar" style="background:${gradientCss(range)}"></span><span>${formatValue(range[1])}</span></span>
          <label class="toggle"><input type="checkbox" data-role="g-arrows" ${s.galleryArrows ? "checked" : ""}/>Arrows</label>
          ${run.layout.kind === "taxi" ? sliceSelectorHtml(run.layout, s.taxiSlice) : ""}
        </div>
        <div class="gallery">${run.iterations.map((it, k) => {
          const optimal = done && k === run.optimal_index;
          return `<div class="g-card ${optimal ? "optimal" : ""} ${k === s.selectedIter ? "active" : ""}" data-iter="${k}">
            <div class="g-head"><span class="g-title">${optimal ? "π★ optimal" : `π${k}`}</span>
              <button class="btn sm" data-play="${k}" title="Run inference with this policy">▶ play</button></div>
            <canvas data-canvas="${k}"></canvas>
            <div class="g-foot"><span>V(s₀) ${formatValue(it.stats.v_start)}</span><span>Δ ${it.stats.n_changed}</span><span>${it.eval_sweeps} sweeps</span></div>
          </div>`;
        }).join("")}</div>`;
      run.iterations.forEach((it, k) => {
        const canvas = root.querySelector(`[data-canvas="${k}"]`);
        const renderer = createRenderer(canvas, run);
        const cell = Math.max(8, Math.floor(MINI_WIDTH / renderer.dims.cols));
        renderer.resize(cell);
        renderer.draw({
          values: it.V, range, policy: it.policy, changed: new Set(it.changed_states),
          showValues: true, showArrows: s.galleryArrows && cell >= 16, showChanges: true,
          slice: s.taxiSlice, mode: "analysis",
        });
      });
    },
  };
}
