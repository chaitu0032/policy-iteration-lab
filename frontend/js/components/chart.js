// Small dependency-free SVG chart: line and bar series over an integer x axis.

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
  parent?.appendChild(node);
  return node;
}

function niceTicks(lo, hi, count = 4) {
  if (lo === hi) return [lo];
  const step = 10 ** Math.floor(Math.log10((hi - lo) / count));
  const err = ((hi - lo) / count) / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = step * mult;
  const ticks = [];
  for (let t = Math.ceil(lo / s) * s; t <= hi + 1e-12; t += s) ticks.push(+t.toPrecision(12));
  return ticks;
}

function fmt(v, step = null) {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(0);
  if (step) return v.toFixed(Math.max(0, Math.min(6, -Math.floor(Math.log10(step)))));
  if (a >= 100) return v.toFixed(0);
  return +v.toFixed(3) + "";
}

/**
 * opts: { series: [{label, color, values, type: 'line'|'bar', dashed, axis: 'left'|'right'}],
 *         height, log, selected, onSelect(i), xLabel, xFormat(i), yMin, yMax }
 */
export function renderChart(container, opts) {
  const { series, height = 170, log = false, selected = null, onSelect, xFormat = (i) => i } = opts;
  container.innerHTML = "";
  container.classList.add("chart");
  const n = Math.max(0, ...series.map((s) => s.values.length));
  if (!n) {
    container.innerHTML = '<div class="note">No data yet.</div>';
    return;
  }
  const W = Math.max(280, container.clientWidth || 480);
  const H = height;
  const m = { l: 46, r: 12, t: 10, b: 24 };
  const tf = (v) => (log ? Math.log10(Math.max(v, 1e-16)) : v);
  const all = series.flatMap((s) => s.values.filter((v) => v != null && Number.isFinite(v)).map(tf));
  let lo = opts.yMin != null ? tf(opts.yMin) : Math.min(...all);
  let hi = opts.yMax != null ? tf(opts.yMax) : Math.max(...all);
  if (series.some((s) => s.type === "bar") && !log) lo = Math.min(lo, 0);
  if (lo === hi) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;

  const plotW = W - m.l - m.r;
  const plotH = H - m.t - m.b;
  const band = plotW / n;
  const x = (i) => m.l + band * (i + 0.5);
  const y = (v) => m.t + plotH * (1 - (tf(v) - lo) / (hi - lo));

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const ticks = log
    ? Array.from({ length: Math.floor(hi) - Math.ceil(lo) + 1 }, (_, k) => 10 ** (Math.ceil(lo) + k))
    : niceTicks(lo, hi);
  const thin = Math.max(1, Math.ceil(ticks.length / 6));
  const tickStep = !log && ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : null;
  ticks.filter((_, k) => k % thin === 0).forEach((t) => {
    const yy = y(t);
    el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, class: "grid" }, svg);
    el("text", { x: m.l - 6, y: yy + 3, "text-anchor": "end" }, svg).textContent = fmt(t, tickStep);
  });
  el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  const every = Math.max(1, Math.ceil(n / Math.floor(plotW / 34)));
  for (let i = 0; i < n; i += every) {
    el("text", { x: x(i), y: H - 8, "text-anchor": "middle" }, svg).textContent = xFormat(i);
  }

  if (selected != null && selected < n) {
    el("rect", { x: x(selected) - band / 2, y: m.t, width: band, height: plotH, class: "sel-band" }, svg);
  }

  const bars = series.filter((s) => s.type === "bar");
  bars.forEach((s, bi) => {
    const bw = (band * 0.7) / bars.length;
    s.values.forEach((v, i) => {
      if (v == null) return;
      const y0 = y(log ? 10 ** lo : 0);
      const y1 = y(v);
      el("rect", {
        x: x(i) - (band * 0.35) + bi * bw, y: Math.min(y0, y1), width: Math.max(1, bw - 1),
        height: Math.max(1, Math.abs(y0 - y1)), rx: 2, opacity: 0.9,
      }, svg).style.fill = s.color;
    });
  });

  series.filter((s) => s.type !== "bar").forEach((s) => {
    const pts = s.values.map((v, i) => (v == null || !Number.isFinite(v) ? null : [x(i), y(v)]));
    const d = pts.reduce((acc, p, i) => (p ? `${acc}${acc && pts[i - 1] ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}` : acc), "");
    el("path", { d, fill: "none", "stroke-width": 2, "stroke-dasharray": s.dashed ? "5 4" : "none" }, svg).style.stroke = s.color;
    if (n <= 80) {
      pts.forEach((p) => { if (p) el("circle", { cx: p[0], cy: p[1], r: 2.6 }, svg).style.fill = s.color; });
    }
  });

  if (onSelect) {
    for (let i = 0; i < n; i += 1) {
      const hit = el("rect", { x: x(i) - band / 2, y: m.t, width: band, height: plotH, class: "hit" }, svg);
      const title = el("title", {}, hit);
      title.textContent = `${xFormat(i)}: ` + series.map((s) => `${s.label} ${s.values[i] == null ? "–" : fmt(s.values[i])}`).join(" · ");
      hit.addEventListener("click", () => onSelect(i));
    }
  }

  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("");
  container.appendChild(legend);
}
