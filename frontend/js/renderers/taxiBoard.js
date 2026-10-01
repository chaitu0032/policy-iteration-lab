// Renderer for Taxi: 5x5 grid with walls, R/G/Y/B depots, taxi, passenger and destination.
// Values/policy are shown for one "slice" (passenger location, destination) of the 500 states.

import { formatValue, textColorFor, valueColor } from "../colors.js";
import { decodeTaxi, encodeTaxi, TAXI_IN_CAR } from "../taxi.js";
import { drawArrow, drawBadgeGlyph, drawPerson, drawText, drawTrail, roundRect, setupCanvas, strokeCell } from "./draw.js";
import { alphaTemplate, token } from "../theme.js";

// R, G, Y, B depots (gymnasium colours, softened for a dark board)
const LOC_COLORS = ["#ef6b6b", "#5cc98a", "#e9c74d", "#5e9cf0"];

export function createTaxiBoard(canvas, run) {
  const { layout, env } = run;
  const { rows, cols, walls, locs, loc_names: locNames } = layout;
  const pad = 6;
  let size = 80;

  const xy = (r, c) => [pad + c * size, pad + r * size];
  const centerOf = (r, c) => [pad + c * size + size / 2, pad + r * size + size / 2];

  function resize(cellSize) {
    size = cellSize;
    return { width: cols * size + 2 * pad, height: rows * size + 2 * pad };
  }

  function sliceOf(view) {
    if (view.agent != null) {
      const d = decodeTaxi(view.agent);
      return { passenger: d.passenger, destination: d.destination };
    }
    return view.slice || { passenger: 0, destination: 1 };
  }

  function draw(view) {
    const W = cols * size + 2 * pad;
    const H = rows * size + 2 * pad;
    const ctx = setupCanvas(canvas, W, H);
    ctx.fillStyle = token("bg");
    ctx.fillRect(0, 0, W, H);
    const slice = sliceOf(view);
    const isTerminalSlice = slice.passenger === slice.destination;

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const [x, y] = xy(r, c);
        const s = encodeTaxi(r, c, slice.passenger, slice.destination);
        ctx.fillStyle = "#2a2f37"; // asphalt
        ctx.fillRect(x, y, size, size);
        const locIdx = locs.findIndex(([lr, lc]) => lr === r && lc === c);
        if (locIdx >= 0) {
          ctx.fillStyle = `${LOC_COLORS[locIdx]}33`;
          ctx.fillRect(x, y, size, size);
        }
        const v = view.values?.[s];
        const overlay = view.showValues && view.values && !isTerminalSlice;
        if (overlay) {
          ctx.fillStyle = valueColor(v, view.range, view.mode === "play" ? 0.5 : 0.85);
          ctx.fillRect(x, y, size, size);
        }
        if (locIdx >= 0) {
          drawText(ctx, locNames[locIdx], x + 10, y + 11, { size: 12, color: LOC_COLORS[locIdx], weight: 700 });
        }
        const fg = overlay ? textColorFor(v, view.range) : "#e3e7ee";
        if (view.showArrows && view.policy && !isTerminalSlice) {
          const a = view.policy[s];
          const vec = env.action_vectors[a];
          const cy = y + size / 2 - (view.showValues ? size * 0.08 : 0);
          if (vec) drawArrow(ctx, x + size / 2, cy, vec[0], vec[1], size, fg);
          else drawBadgeGlyph(ctx, x + size / 2, cy, size, env.action_names[a][0], a === 4 ? "#7fd0ff" : "#f6c445");
        }
        if (overlay && size >= 40) {
          drawText(ctx, formatValue(v), x + size / 2, y + size - Math.max(9, size * 0.15), { size: Math.max(9, Math.min(13, size * 0.18)), color: fg });
        }
        ctx.strokeStyle = "rgba(255,255,255,.08)";
        ctx.strokeRect(x + 0.5, y + 0.5, size - 1, size - 1);
        if (view.showChanges && view.changed?.has(s)) strokeCell(ctx, x, y, size, token("warn"), 2.5);
        if (view.startState === s) strokeCell(ctx, x, y, size, token("good"), 3, 3);
        if (view.selected === s) strokeCell(ctx, x, y, size, token("accent"), 3, 1);
        if (view.hover === s) strokeCell(ctx, x, y, size, "rgba(255,255,255,.7)", 1.5, 1);
      }
    }
    drawWalls(ctx, W, H);
    drawDestination(ctx, slice.destination);
    if (slice.passenger !== TAXI_IN_CAR && !isTerminalSlice) {
      const [pr, pc] = locs[slice.passenger];
      const [cx, cy] = centerOf(pr, pc);
      drawPerson(ctx, cx + size * 0.28, cy - size * 0.18, size * 0.8);
    }
    if (view.trail?.length) {
      drawTrail(ctx, view.trail.map((s) => { const d = decodeTaxi(s); return centerOf(d.row, d.col); }), size, alphaTemplate("accent-2"));
    }
    if (view.agent != null) drawTaxi(ctx, decodeTaxi(view.agent), view.agentStatus);
  }

  function drawWalls(ctx, W, H) {
    ctx.save();
    ctx.strokeStyle = "#d9dee6";
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    roundRect(ctx, pad - 2, pad - 2, W - 2 * pad + 4, H - 2 * pad + 4, 6);
    ctx.stroke();
    for (const [r, c] of walls) {
      const x = pad + (c + 1) * size;
      ctx.beginPath();
      ctx.moveTo(x, pad + r * size + 3);
      ctx.lineTo(x, pad + (r + 1) * size - 3);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawDestination(ctx, dest) {
    const [r, c] = locs[dest];
    const [x, y] = xy(r, c);
    ctx.save();
    ctx.setLineDash([6, 4]);
    strokeCell(ctx, x, y, size, LOC_COLORS[dest], 3, 4);
    ctx.restore();
    drawText(ctx, "⚑", x + size - 13, y + 13, { size: 14, color: LOC_COLORS[dest] });
  }

  function drawTaxi(ctx, d, status) {
    const [cx, cy] = centerOf(d.row, d.col);
    const w = size * 0.56;
    const h = size * 0.36;
    const loaded = d.passenger === TAXI_IN_CAR;
    ctx.save();
    ctx.shadowColor = token(status === "success" ? "good" : "star");
    ctx.shadowBlur = 14;
    ctx.fillStyle = status === "success" ? token("good") : loaded ? "#ffd75e" : "#f6c445";
    roundRect(ctx, cx - w / 2, cy - h / 2, w, h, 8);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#14141c";
    roundRect(ctx, cx - w * 0.3, cy - h * 0.38, w * 0.6, h * 0.34, 4);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx - w * 0.3, cy + h / 2, size * 0.06, 0, Math.PI * 2);
    ctx.arc(cx + w * 0.3, cy + h / 2, size * 0.06, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    if (loaded) drawPerson(ctx, cx, cy - h * 0.1, size * 0.55, "#ff8fa3");
  }

  function stateAt(px, py, slice) {
    const c = Math.floor((px - pad) / size);
    const r = Math.floor((py - pad) / size);
    if (r < 0 || c < 0 || r >= rows || c >= cols) return null;
    const sl = slice || { passenger: 0, destination: 1 };
    return encodeTaxi(r, c, sl.passenger, sl.destination);
  }

  return { resize, draw, stateAt, dims: { rows, cols } };
}
