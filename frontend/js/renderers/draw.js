// Shared canvas primitives for the board renderers.

export function setupCanvas(canvas, width, height) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

export function drawArrow(ctx, cx, cy, dr, dc, size, color) {
  const len = size * 0.32;
  const head = Math.max(4, size * 0.13);
  const x0 = cx - dc * len * 0.6;
  const y0 = cy - dr * len * 0.6;
  const x1 = cx + dc * len;
  const y1 = cy + dr * len;
  const angle = Math.atan2(dr, dc);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = Math.max(1.5, size * 0.05);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1 - Math.cos(angle) * head * 0.6, y1 - Math.sin(angle) * head * 0.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - head * Math.cos(angle - 0.5), y1 - head * Math.sin(angle - 0.5));
  ctx.lineTo(x1 - head * Math.cos(angle + 0.5), y1 - head * Math.sin(angle + 0.5));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export function drawBadgeGlyph(ctx, cx, cy, size, label, color) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, size * 0.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#14141c";
  ctx.font = `700 ${Math.round(size * 0.22)}px IBM Plex Mono, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, cx, cy + 1);
  ctx.restore();
}

export function drawText(ctx, text, x, y, { size = 11, color = "#fff", align = "center", weight = 500 } = {}) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px IBM Plex Mono, monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
  ctx.restore();
}

export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function strokeCell(ctx, x, y, size, color, width = 2, inset = 2) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  roundRect(ctx, x + inset, y + inset, size - 2 * inset, size - 2 * inset, 5);
  ctx.stroke();
  ctx.restore();
}

export function drawTrail(ctx, points, size, color) {
  if (points.length < 2) return;
  ctx.save();
  ctx.lineWidth = Math.max(2, size * 0.08);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  for (let i = 1; i < points.length; i += 1) {
    const alpha = 0.15 + 0.75 * (i / points.length);
    ctx.strokeStyle = color.replace("ALPHA", alpha.toFixed(2));
    ctx.beginPath();
    ctx.moveTo(points[i - 1][0], points[i - 1][1]);
    ctx.lineTo(points[i][0], points[i][1]);
    ctx.stroke();
  }
  ctx.restore();
}

export function drawPerson(ctx, cx, cy, size, color = "#ff8fa3") {
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy - size * 0.14, size * 0.09, 0, Math.PI * 2);
  ctx.fill();
  roundRect(ctx, cx - size * 0.1, cy - size * 0.03, size * 0.2, size * 0.24, size * 0.06);
  ctx.fill();
  ctx.restore();
}

export function cellSizeFor(cols, rows, maxWidth, maxHeight = 560) {
  const byW = Math.floor(maxWidth / cols);
  const byH = Math.floor(maxHeight / rows);
  return Math.max(22, Math.min(112, byW, byH));
}
