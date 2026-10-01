// Board factory + interactive wrapper (hover tooltip, click handling, responsive sizing).

import { cellSizeFor } from "./draw.js";
import { createGridBoard } from "./gridBoard.js";
import { createTaxiBoard } from "./taxiBoard.js";

export function createRenderer(canvas, run) {
  return run.layout.kind === "taxi" ? createTaxiBoard(canvas, run) : createGridBoard(canvas, run);
}

/**
 * Mount an interactive board inside ``container``.
 * getView() -> current view object; onHover(state|null, event); onClick(state, event).
 */
export function mountBoard(container, run, { getView, onHover, onClick, maxWidth, maxHeight } = {}) {
  container.innerHTML = "";
  const canvas = document.createElement("canvas");
  canvas.className = "board";
  container.appendChild(canvas);
  const renderer = createRenderer(canvas, run);
  let hover = null;

  const fit = () => {
    const avail = maxWidth || Math.max(240, container.parentElement?.clientWidth || 600);
    renderer.resize(cellSizeFor(renderer.dims.cols, renderer.dims.rows, Math.min(avail, 760), maxHeight));
  };

  const redraw = () => renderer.draw({ ...getView(), hover });
  const localXY = (e) => {
    const rect = canvas.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  canvas.addEventListener("mousemove", (e) => {
    const s = renderer.stateAt(...localXY(e), getView().slice);
    if (s !== hover) {
      hover = s;
      redraw();
    }
    onHover?.(s, e);
  });
  canvas.addEventListener("mouseleave", (e) => {
    hover = null;
    redraw();
    onHover?.(null, e);
  });
  canvas.addEventListener("click", (e) => {
    const s = renderer.stateAt(...localXY(e), getView().slice);
    if (s != null) onClick?.(s, e);
  });

  fit();
  redraw();
  return {
    redraw,
    refit() {
      fit();
      redraw();
    },
    canvas,
  };
}
