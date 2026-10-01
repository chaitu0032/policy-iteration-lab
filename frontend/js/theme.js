// Design tokens shared by CSS and JS. Charts use CSS var() strings; canvas code reads resolved values.

/** Resolved value of a CSS custom property on <body> (follows the active world's accent). */
export function token(name) {
  return getComputedStyle(document.body).getPropertyValue(`--${name}`).trim();
}

/** Chart / SVG colours (CSS variables, so they follow the world theme without re-rendering logic). */
export const SERIES = {
  primary: "var(--accent)",
  secondary: "var(--accent-2)",
  good: "var(--good)",
  warn: "var(--warn)",
  bad: "var(--bad)",
  star: "var(--star)",
  text: "var(--text)",
  q: ["var(--accent)", "var(--accent-2)", "var(--good)", "var(--warn)", "var(--bad)", "var(--star)"],
};

/** "#rrggbb" token -> "rgba(r,g,b,ALPHA)" template used by the trail drawer. */
export function alphaTemplate(name) {
  const hex = token(name).replace("#", "");
  const [r, g, b] = [0, 2, 4].map((i) => Number.parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r},${g},${b},ALPHA)`;
}
