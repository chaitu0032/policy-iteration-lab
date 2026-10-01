// Loads Gymnasium's own toy-text sprites (frontend/assets/gym, MIT licensed) once and caches them.

const NAMES = [
  "ice", "hole", "cracked_hole", "goal", "stool", "elf_left", "elf_down", "elf_right", "elf_up",
  "cookie", "mountain_bg1", "mountain_bg2", "mountain_cliff", "mountain_near-cliff1", "mountain_near-cliff2",
  "cab_front", "cab_rear", "cab_right", "cab_left", "passenger", "hotel", "taxi_background",
  "gridworld_median_left", "gridworld_median_horiz", "gridworld_median_right",
  "gridworld_median_top", "gridworld_median_vert", "gridworld_median_bottom",
];

let loading = null;

function load(name) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve([name, img]);
    img.onerror = () => reject(new Error(`Could not load sprite ${name}.png`));
    img.src = `assets/gym/${name}.png`;
  });
}

/** @returns {Promise<Record<string, HTMLImageElement>>} */
export function loadSprites() {
  if (!loading) loading = Promise.all(NAMES.map(load)).then(Object.fromEntries);
  return loading;
}

/** Synchronous access after loadSprites() resolved (renderers are only created afterwards). */
let cache = null;
export async function ensureSprites() {
  cache = await loadSprites();
  return cache;
}
export function sprites() {
  if (!cache) throw new Error("Sprites not loaded yet");
  return cache;
}
