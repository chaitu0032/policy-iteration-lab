// Renderer for grid worlds: FrozenLake (S/F/H/G) and CliffWalking (./S/C/G).

import { textColorFor, valueColor, formatValue } from "../colors.js";
import { drawArrow, drawText, drawTrail, roundRect, strokeCell, setupCanvas } from "./draw.js";
import { alphaTemplate, token } from "../theme.js";

// Tile colours echo gymnasium's sprites: pale ice, deep hole water, moss for the cliff meadow, gift gold.
const TILE = {
  S: "#dbe9f4",
  F: "#e4f0f8",
  ".": "#2b3a2c",
  H: "#10233a",
  G: "#efc35a",
  C: "#3d1a1f",
};

function drawBaseTile(ctx, type, x, y, size, envKey) {
  ctx.fillStyle = TILE[type] || TILE["."];
  ctx.fillRect(x, y, size, size);
  if (type === "H") {
    ctx.fillStyle = "#2a5a8c";
    ctx.beginPath();
    ctx.ellipse(x + size / 2, y + size / 2, size * 0.34, size * 0.24, 0, 0, Math.PI * 2);
    ctx.fill();
  } else if (type === "C") {
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, size, size);
    ctx.clip();
    ctx.strokeStyle = "rgba(240,120,110,.28)";
    ctx.lineWidth = 3;
    for (let i = -size; i < size; i += 9) {
      ctx.beginPath();
      ctx.moveTo(x + i, y + size);
      ctx.lineTo(x + i + size, y);
      ctx.stroke();
    }
    ctx.restore();
  } else if (type === "G") {
    drawText(ctx, envKey === "FrozenLake" ? "🎁" : "🏁", x + size / 2, y + size / 2, { size: size * 0.42 });
  }
}

export function createGridBoard(canvas, run) {
  const { layout, env } = run;
  const { rows, cols, cells } = layout;
  const terminal = new Set(layout.terminal_states);
  let size = 60;

  const cellOf = (s) => [Math.floor(s / cols), s % cols];
  const center = (s) => {
    const [r, c] = cellOf(s);
    return [c * size + size / 2, r * size + size / 2];
  };

  function resize(cellSize) {
    size = cellSize;
    return { width: cols * size, height: rows * size };
  }

  function draw(view) {
    const ctx = setupCanvas(canvas, cols * size, rows * size);
    ctx.fillStyle = token("bg");
    ctx.fillRect(0, 0, cols * size, rows * size);

    for (let s = 0; s < rows * cols; s += 1) {
      const [r, c] = cellOf(s);
      const x = c * size;
      const y = r * size;
      const type = cells[r][c];
      drawBaseTile(ctx, type, x, y, size, env.key);
      const v = view.values?.[s];
      const overlay = view.showValues && view.values && !terminal.has(s) && type !== "C";
      if (overlay) {
        ctx.fillStyle = valueColor(v, view.range, view.mode === "play" ? 0.55 : 0.92);
        ctx.fillRect(x, y, size, size);
        if (type === "S") drawText(ctx, "S", x + 8, y + 9, { size: 10, color: textColorFor(v, view.range), weight: 700 });
      } else if (type === "S") {
        drawText(ctx, "S", x + size / 2, y + size / 2, { size: size * 0.3, color: "#5b7fa3", weight: 700 });
      }
      ctx.strokeStyle = "rgba(10,14,20,.35)";
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);

      if (!terminal.has(s) && type !== "C") {
        const fg = overlay ? textColorFor(v, view.range) : "#14202c";
        const arrowColor = overlay ? fg : (type === "." ? "#dfe8d6" : "#2c3e52");
        if (view.showArrows && view.policy) {
          const vec = env.action_vectors[view.policy[s]];
          if (vec) drawArrow(ctx, x + size / 2, y + size / 2 - (view.showValues ? size * 0.08 : 0), vec[0], vec[1], size, arrowColor);
        }
        if (overlay && size >= 38) {
          drawText(ctx, formatValue(v), x + size / 2, y + size - Math.max(9, size * 0.16), { size: Math.max(9, Math.min(13, size * 0.2)), color: fg });
        }
      }
      if (view.showChanges && view.changed?.has(s)) strokeCell(ctx, x, y, size, token("warn"), 2.5);
    }

    if (view.startState != null) strokeCell(ctx, ...cellXY(view.startState), size, token("good"), 3, 3);
    if (view.selected != null) strokeCell(ctx, ...cellXY(view.selected), size, token("accent"), 3, 1);
    if (view.hover != null) strokeCell(ctx, ...cellXY(view.hover), size, "rgba(255,255,255,.7)", 1.5, 1);
    if (view.trail?.length) drawTrail(ctx, view.trail.map(center), size, alphaTemplate("accent-2"));
    if (view.agent != null) drawAgent(ctx, view.agent, view.agentStatus);
  }

  function cellXY(s) {
    const [r, c] = cellOf(s);
    return [c * size, r * size];
  }

  function drawAgent(ctx, s, status) {
    const [cx, cy] = center(s);
    const color = token(status === "success" ? "good" : status === "fail" ? "bad" : "accent-2");
    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.26, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#14141c";
    ctx.beginPath();
    ctx.arc(cx - size * 0.08, cy - size * 0.04, size * 0.04, 0, Math.PI * 2);
    ctx.arc(cx + size * 0.08, cy - size * 0.04, size * 0.04, 0, Math.PI * 2);
    ctx.fill();
    roundRect(ctx, cx - size * 0.07, cy + size * 0.07, size * 0.14, size * 0.03, 2);
    ctx.fill();
    ctx.restore();
  }

  function stateAt(px, py) {
    const c = Math.floor(px / size);
    const r = Math.floor(py / size);
    if (r < 0 || c < 0 || r >= rows || c >= cols) return null;
    return r * cols + c;
  }

  return { resize, draw, stateAt, dims: { rows, cols } };
}
