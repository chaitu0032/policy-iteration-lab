// Taxi drawn tile-by-tile from the 7x11 gymnasium map (road, medians, depots, hotel, passenger, cab),
// plus value / policy layers for one (passenger, destination) slice of the 500 states.

import { decodeTaxi, encodeTaxi, TAXI_IN_CAR } from "../taxi.js";
import { actionBadge, arrow, blit, footprints, lerp, outline, setupCanvas, valueLabel, valueTint } from "./overlay.js";
import { sprites } from "./sprites.js";

const DEPOT_COLORS = ["rgba(255,0,0,.5)", "rgba(0,255,0,.5)", "rgba(255,255,0,.5)", "rgba(0,0,255,.5)"];
const CAB = ["cab_front", "cab_rear", "cab_right", "cab_left"]; // South, North, East, West

function medianSprite(desc, y, x) {
  const ch = desc[y][x];
  if (ch === "|") {
    if (y === 0 || desc[y - 1][x] !== "|") return "gridworld_median_top";
    if (y === desc.length - 1 || desc[y + 1][x] !== "|") return "gridworld_median_bottom";
    return "gridworld_median_vert";
  }
  if (ch === "-") {
    if (x === 0 || desc[y][x - 1] !== "-") return "gridworld_median_left";
    if (x === desc[y].length - 1 || desc[y][x + 1] !== "-") return "gridworld_median_right";
    return "gridworld_median_horiz";
  }
  return null;
}

export function createTaxiWorld(canvas, run) {
  const { layout, env } = run;
  const desc = layout.desc;
  const tilesX = desc[0].length;
  const tilesY = desc.length;
  let tile = 50;

  // gymnasium get_surf_loc: map cell (r, c) -> pixel ((2c+1) * tile, (r+1) * tile)
  const cellXY = (r, c) => [(2 * c + 1) * tile, (r + 1) * tile];
  const centerOf = (s) => { const d = decodeTaxi(s); const [x, y] = cellXY(d.row, d.col); return [x + tile / 2, y + tile / 2]; };

  function sliceOf(view) {
    const s = view.agent ? view.agent.to : view.start;
    if (view.slice) return view.slice;
    const d = decodeTaxi(s ?? 0);
    return { passenger: d.passenger, destination: d.destination };
  }

  function drawMap(ctx, img) {
    for (let y = 0; y < tilesY; y += 1) {
      for (let x = 0; x < tilesX; x += 1) {
        blit(ctx, img.taxi_background, x * tile, y * tile, tile, tile);
        const m = medianSprite(desc, y, x);
        if (m) blit(ctx, img[m], x * tile, y * tile, tile, tile);
      }
    }
    layout.locs.forEach(([r, c], i) => {
      const [x, y] = cellXY(r, c);
      ctx.fillStyle = DEPOT_COLORS[i];
      ctx.fillRect(x, y + tile * 0.2, tile, tile); // gymnasium offsets depots by +10px at 50px tiles
    });
  }

  function drawLayers(ctx, view, slice) {
    const L = view.layers || {};
    if (slice.passenger === slice.destination) return;
    for (let r = 0; r < 5; r += 1) {
      for (let c = 0; c < 5; c += 1) {
        const s = encodeTaxi(r, c, slice.passenger, slice.destination);
        const [x, y] = cellXY(r, c);
        if (L.values && view.values) valueTint(ctx, x, y, tile, tile, view.values[s], view.range, view.tintAlpha ?? 0.6);
        if (L.arrows && view.policy) {
          const a = view.policy[s];
          const vec = env.action_vectors[a];
          const cy = y + tile / 2 - (L.values && tile >= 40 ? tile * 0.1 : 0);
          if (vec) arrow(ctx, x + tile / 2, cy, vec[0], vec[1], tile);
          else actionBadge(ctx, x + tile / 2, cy, tile, a === 4 ? "P" : "D", a === 4 ? "#7fd4ff" : "#ffd75e");
        }
        if (L.values && view.values && tile >= 40) valueLabel(ctx, view.values[s], x + tile / 2, y + tile * 0.8, tile);
        if (L.changed && view.changed?.has(s)) outline(ctx, x, y, tile, tile, "#f2a20c", 3);
        if (view.start === s) outline(ctx, x, y, tile, tile, "#1f8a4c", 3, [6, 4]);
        if (view.selected === s) outline(ctx, x, y, tile, tile, "#1b2230", 3);
        if (view.hover === s) outline(ctx, x, y, tile, tile, "rgba(255,255,255,.9)", 2);
      }
    }
  }

  function drawActors(ctx, img, view, slice) {
    const agent = view.agent;
    const state = agent ? agent.to : view.start;
    const d = state != null ? decodeTaxi(state) : null;
    const passenger = d ? d.passenger : slice.passenger;
    const dest = d ? d.destination : slice.destination;
    if (passenger < TAXI_IN_CAR) {
      const [r, c] = layout.locs[passenger];
      blit(ctx, img.passenger, ...cellXY(r, c), tile, tile);
    }
    const [dx, dy] = cellXY(...layout.locs[dest]);
    const hotel = () => blit(ctx, img.hotel, dx, dy - tile / 2, tile, tile, 170 / 255);
    if (!d) { hotel(); return; }
    let tx; let ty;
    if (agent) {
      const a = decodeTaxi(agent.from);
      const t = agent.t ?? 1;
      [tx, ty] = [lerp((2 * a.col + 1) * tile, (2 * d.col + 1) * tile, t), lerp((a.row + 1) * tile, (d.row + 1) * tile, t)];
    } else {
      [tx, ty] = cellXY(d.row, d.col);
    }
    const cab = () => blit(ctx, img[CAB[view.orientation ?? 0]], tx, ty, tile, tile);
    if (dy <= ty) { hotel(); cab(); } else { cab(); hotel(); } // same overlap order as gymnasium
  }

  function draw(view) {
    const img = sprites();
    const ctx = setupCanvas(canvas, tilesX * tile, tilesY * tile);
    const slice = sliceOf(view);
    drawMap(ctx, img);
    drawLayers(ctx, view, slice);
    if (view.layers?.path && view.path?.length > 1) footprints(ctx, view.path.map(centerOf), tile, "#7a3ff2");
    drawActors(ctx, img, view, slice);
  }

  function stateAt(px, py, slice) {
    const x = Math.floor(px / tile);
    const y = Math.floor(py / tile);
    if (y < 1 || y > 5 || x % 2 === 0 || x < 1 || x > 9) return null;
    const sl = slice || { passenger: 0, destination: 1 };
    return encodeTaxi(y - 1, (x - 1) / 2, sl.passenger, sl.destination);
  }

  return {
    resize(maxWidth, maxHeight = 520) {
      tile = Math.max(8, Math.min(80, Math.floor(maxWidth / tilesX), Math.floor(maxHeight / tilesY)));
    },
    draw,
    stateAt,
    cellSize: () => tile,
  };
}
