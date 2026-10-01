// FrozenLake and CliffWalking drawn the way gymnasium's _render_gui draws them, plus analysis layers.
//
// view = { values, range, policy, layers: {values, arrows, changed, path}, changed:Set, agent:{from,to,t,action,fell},
//          path:[states], start, selected, hover }

import { arrow, blit, footprints, lerp, outline, setupCanvas, valueLabel, valueTint } from "./overlay.js";
import { sprites } from "./sprites.js";

const FROZEN_ELF = ["elf_left", "elf_down", "elf_right", "elf_up"]; // FrozenLake action order
const CLIFF_ELF = ["elf_up", "elf_right", "elf_down", "elf_left"]; // CliffWalking action order

function drawFrozenTile(ctx, img, type, x, y, size) {
  blit(ctx, img.ice, x, y, size, size);
  if (type === "H") blit(ctx, img.hole, x, y, size, size);
  else if (type === "G") blit(ctx, img.goal, x, y, size, size);
  else if (type === "S") blit(ctx, img.stool, x, y, size, size);
  ctx.strokeStyle = "rgb(180,200,230)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
}

function drawCliffTile(ctx, img, cells, r, c, x, y, size) {
  const checker = (r % 2) ^ (c % 2);
  blit(ctx, img[`mountain_bg${checker + 1}`], x, y, size, size);
  const type = cells[r][c];
  if (type === "C") blit(ctx, img.mountain_cliff, x, y, size, size);
  if (r < cells.length - 1 && cells[r + 1][c] === "C") blit(ctx, img[`mountain_near-cliff${checker + 1}`], x, y, size, size);
  if (type === "S") blit(ctx, img.stool, x, y, size, size);
  if (type === "G") blit(ctx, img.cookie, x, y, size, size);
}

export function createGridWorld(canvas, run) {
  const { layout, env } = run;
  const { rows, cols, cells } = layout;
  const isFrozen = env.key === "FrozenLake";
  const blocked = new Set(layout.terminal_states);
  if (!isFrozen) cells.forEach((row, r) => row.forEach((t, c) => { if (t === "C") blocked.add(r * cols + c); }));
  let size = 64;

  const rc = (s) => [Math.floor(s / cols), s % cols];
  const center = (s) => { const [r, c] = rc(s); return [c * size + size / 2, r * size + size / 2]; };

  function drawAgent(ctx, img, agent) {
    const [r0, c0] = rc(agent.from);
    const [r1, c1] = rc(agent.to);
    const t = agent.t ?? 1;
    const x = lerp(c0, c1, t) * size;
    const y = lerp(r0, r1, t) * size;
    if (isFrozen) {
      if (agent.fell && t >= 1) blit(ctx, img.cracked_hole, x, y, size, size);
      else blit(ctx, img[FROZEN_ELF[agent.action ?? 1]], x, y, size, size);
    } else {
      blit(ctx, img[CLIFF_ELF[agent.action ?? 2]], x, y - 0.1 * size, size, size);
    }
  }

  function draw(view) {
    const img = sprites();
    const W = cols * size;
    const H = rows * size;
    const ctx = setupCanvas(canvas, W, H);
    const L = view.layers || {};
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const x = c * size;
        const y = r * size;
        if (isFrozen) drawFrozenTile(ctx, img, cells[r][c], x, y, size);
        else drawCliffTile(ctx, img, cells, r, c, x, y, size);
      }
    }
    for (let s = 0; s < rows * cols; s += 1) {
      if (blocked.has(s)) continue;
      const [r, c] = rc(s);
      const x = c * size;
      const y = r * size;
      if (L.values && view.values) valueTint(ctx, x, y, size, size, view.values[s], view.range, view.tintAlpha ?? 0.5);
      if (L.arrows && view.policy) {
        const vec = env.action_vectors[view.policy[s]];
        if (vec) arrow(ctx, x + size / 2, y + size / 2 - (L.values && size >= 40 ? size * 0.1 : 0), vec[0], vec[1], size);
      }
      if (L.values && view.values && size >= 40) valueLabel(ctx, view.values[s], x + size / 2, y + size * 0.78, size);
      if (L.changed && view.changed?.has(s)) outline(ctx, x, y, size, size, "#f2a20c", 3);
    }
    if (L.path && view.path?.length > 1) footprints(ctx, view.path.map(center), size, "#7a3ff2");
    if (view.start != null) outline(ctx, ...rc(view.start).reverse().map((v) => v * size), size, size, "#1f8a4c", 3, [6, 4]);
    if (view.selected != null) outline(ctx, ...rc(view.selected).reverse().map((v) => v * size), size, size, "#1b2230", 3);
    if (view.hover != null) outline(ctx, ...rc(view.hover).reverse().map((v) => v * size), size, size, "rgba(255,255,255,.9)", 2);
    if (view.agent) drawAgent(ctx, img, view.agent);
  }

  function stateAt(px, py) {
    const c = Math.floor(px / size);
    const r = Math.floor(py / size);
    return r >= 0 && c >= 0 && r < rows && c < cols ? r * cols + c : null;
  }

  return {
    resize(maxWidth, maxHeight = 520) {
      size = Math.max(10, Math.min(96, Math.floor(maxWidth / cols), Math.floor(maxHeight / rows)));
    },
    draw,
    stateAt,
    cellSize: () => size,
  };
}
