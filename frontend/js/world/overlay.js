// Analysis layer drawn on top of the pixel world: value tint + label, policy arrows, markers, path.

import { formatValue, valueColor } from "../colors.js";

export function setupCanvas(canvas, width, height) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false; // keep Gymnasium's pixel art crisp
  return ctx;
}

export function blit(ctx, img, x, y, w, h, alpha = 1) {
  if (alpha !== 1) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, x, y, w, h);
    ctx.restore();
  } else {
    ctx.drawImage(img, x, y, w, h);
  }
}

export function valueTint(ctx, x, y, w, h, v, range, alpha = 0.55) {
  ctx.fillStyle = valueColor(v, range, alpha);
  ctx.fillRect(x, y, w, h);
}

/** Value printed in the pixel HUD face with a dark outline so it reads on any sprite. */
export function valueLabel(ctx, v, cx, cy, size) {
  const px = Math.max(10, Math.min(16, Math.round(size * 0.2)));
  ctx.save();
  ctx.font = `600 ${px}px "Pixelify Sans", monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(20, 24, 36, 0.85)";
  const text = formatValue(v);
  ctx.strokeText(text, cx, cy);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, cx, cy);
  ctx.restore();
}

/** Chunky arrow: white fill with an ink outline (legible over ice, grass and asphalt). */
export function arrow(ctx, cx, cy, dr, dc, size) {
  const len = size * 0.3;
  const shaft = Math.max(2, size * 0.07);
  const head = Math.max(5, size * 0.16);
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.atan2(dr, dc));
  ctx.beginPath();
  ctx.moveTo(-len, -shaft);
  ctx.lineTo(len - head, -shaft);
  ctx.lineTo(len - head, -head);
  ctx.lineTo(len, 0);
  ctx.lineTo(len - head, head);
  ctx.lineTo(len - head, shaft);
  ctx.lineTo(-len, shaft);
  ctx.closePath();
  ctx.lineJoin = "miter";
  ctx.lineWidth = Math.max(2, size * 0.035);
  ctx.strokeStyle = "#1b2230";
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.restore();
}

/** Lettered disc for Taxi's Pickup / Dropoff actions. */
export function actionBadge(ctx, cx, cy, size, letter, fill) {
  const r = Math.max(7, size * 0.2);
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#1b2230";
  ctx.stroke();
  ctx.fillStyle = "#1b2230";
  ctx.font = `700 ${Math.round(r * 1.2)}px "Pixelify Sans", monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(letter, cx, cy + 1);
  ctx.restore();
}

export function outline(ctx, x, y, w, h, color, width = 3, dash = null) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.strokeRect(x + width / 2, y + width / 2, w - width, h - width);
  ctx.restore();
}

/** Footprint path: small squares (pixel style) fading in along the visited cells. */
export function footprints(ctx, points, size, color) {
  const n = points.length;
  points.forEach(([x, y], i) => {
    const s = Math.max(4, size * 0.12);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.25 + 0.6 * ((i + 1) / n);
    ctx.fillRect(Math.round(x - s / 2), Math.round(y - s / 2), s, s);
  });
  ctx.globalAlpha = 1;
}

export const lerp = (a, b, t) => a + (b - a) * t;
