// Value colormaps. Sequential (viridis-like) for one-signed values, diverging when V spans 0.

// Sequential: deep indigo -> teal -> mint -> warm cream (perceptually ordered, readable on a dark UI).
const SEQUENTIAL = [
  [38, 36, 78], [44, 82, 130], [33, 128, 141], [64, 174, 132], [166, 214, 126], [246, 236, 170],
];
// Diverging: brick red (negative) -> slate (zero) -> sea teal (positive).
const DIVERGING = [
  [176, 58, 62], [224, 128, 102], [58, 66, 82], [84, 166, 176], [150, 222, 200],
];

function interpolate(stops, t) {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
}

export function valueRange(arrays) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const arr of arrays) {
    for (const v of arr) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  return [lo, hi];
}

export function valueColor(v, [lo, hi], alpha = 1) {
  let rgb;
  if (lo < 0 && hi > 0) {
    const m = Math.max(-lo, hi);
    rgb = interpolate(DIVERGING, (v + m) / (2 * m));
  } else {
    rgb = interpolate(SEQUENTIAL, (v - lo) / (hi - lo));
  }
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`;
}

export function textColorFor(v, range) {
  const [r, g, b] = valueColor(v, range).match(/\d+/g).map(Number);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#141821" : "#f3f5f8";
}

export function gradientCss(range) {
  const stops = Array.from({ length: 9 }, (_, i) => {
    const v = range[0] + ((range[1] - range[0]) * i) / 8;
    return `${valueColor(v, range)} ${(i / 8) * 100}%`;
  });
  return `linear-gradient(90deg, ${stops.join(",")})`;
}

export function formatValue(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
