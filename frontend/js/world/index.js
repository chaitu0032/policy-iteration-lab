// World factory: picks the gymnasium-style renderer for a run's environment.

import { createGridWorld } from "./gridWorlds.js";
import { createTaxiWorld } from "./taxiWorld.js";

export { ensureSprites } from "./sprites.js";

export function createWorld(canvas, run) {
  return run.layout.kind === "taxi" ? createTaxiWorld(canvas, run) : createGridWorld(canvas, run);
}
