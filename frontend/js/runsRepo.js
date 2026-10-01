// Run repository used by the app: per-user server storage when the backend offers it, otherwise this
// browser's IndexedDB. Runs travel gzip-compressed (Taxi runs are several MB of JSON).

import { api } from "./api.js";
import { runsDb } from "./runsDb.js";

let mode = "browser";

export function setStorageMode(next) {
  mode = next === "server" ? "server" : "browser";
}

export function storageMode() {
  return mode;
}

async function gzipJson(doc) {
  const stream = new Blob([JSON.stringify(doc)]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).blob();
}

async function gunzipJson(response) {
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return JSON.parse(await new Response(stream).text());
}

const serverRepo = {
  list: () => api.listRuns(),
  async get(id) {
    try {
      return await gunzipJson(await api.getRunGz(id));
    } catch (err) {
      if (/not found/i.test(err.message)) return null;
      throw err;
    }
  },
  async save(run) {
    await api.putRunGz(run.id, await gzipJson(run));
    return run;
  },
  remove: (id) => api.deleteRun(id),
};

/** Same interface as runsDb, routed to the active backend. */
export const runs = {
  list: () => (mode === "server" ? serverRepo : runsDb).list(),
  get: (id) => (mode === "server" ? serverRepo : runsDb).get(id),
  save: (run) => (mode === "server" ? serverRepo : runsDb).save(run),
  remove: (id) => (mode === "server" ? serverRepo : runsDb).remove(id),
};

/** Upload runs that were saved in this browser before accounts existed; returns how many moved. */
export async function migrateBrowserRuns(onProgress) {
  const local = await runsDb.list();
  let moved = 0;
  for (const summary of local) {
    const run = await runsDb.get(summary.id);
    if (run) {
      await serverRepo.save(run);
      await runsDb.remove(summary.id);
      moved += 1;
      onProgress?.(moved, local.length);
    }
  }
  return moved;
}

export async function browserRunCount() {
  try {
    return (await runsDb.list()).length;
  } catch {
    return 0;
  }
}
