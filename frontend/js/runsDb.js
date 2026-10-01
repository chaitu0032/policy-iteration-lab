// Run repository in the browser (IndexedDB) — the backend is stateless so it can run on Vercel.
// Falls back to an in-memory map when IndexedDB is unavailable (private mode, blocked storage).

const DB_NAME = "pi-lab";
const STORE = "runs";
const memory = new Map();
let dbPromise = null;

function openDb() {
  if (!("indexedDB" in window)) return Promise.resolve(null);
  if (!dbPromise) {
    dbPromise = new Promise((resolve) => {
      try {
        const req = indexedDB.open(DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: "id" });
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => {
          console.warn("[runsDb] IndexedDB unavailable, using memory store", req.error);
          resolve(null);
        };
      } catch (err) {
        console.warn("[runsDb] IndexedDB failed, using memory store", err);
        resolve(null);
      }
    });
  }
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await openDb();
  if (!db) return fn(null);
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const result = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
  });
}

export function summarize(run) {
  const best = run.iterations[run.optimal_index] || run.iterations[run.iterations.length - 1];
  return {
    id: run.id,
    created_at: run.created_at,
    env_key: run.env.key,
    title: run.env.title,
    options: run.env.options,
    gamma: run.config.gamma,
    eval_mode: run.config.eval_mode,
    n_iterations: run.iterations.length,
    status: run.status,
    v_start: best?.stats.v_start ?? null,
  };
}

export const runsDb = {
  async save(run) {
    const copy = structuredClone(run);
    await tx("readwrite", (store) => (store ? store.put(copy) : memory.set(copy.id, copy)));
    return run;
  },
  async get(id) {
    return tx("readonly", (store) => (store ? store.get(id) : structuredClone(memory.get(id))));
  },
  async list() {
    const all = await tx("readonly", (store) => (store ? store.getAll() : [...memory.values()]));
    return (all || []).map(summarize).sort((a, b) => b.created_at - a.created_at);
  },
  async remove(id) {
    await tx("readwrite", (store) => (store ? store.delete(id) : memory.delete(id)));
  },
};

export function newRunId() {
  const bytes = crypto.getRandomValues(new Uint8Array(6));
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function validateImportedRun(data) {
  const ok = data && typeof data === "object" && data.env?.key && data.layout && Array.isArray(data.iterations)
    && data.iterations.length > 0 && data.iterations.every((it) => Array.isArray(it.policy) && Array.isArray(it.V));
  if (!ok) throw new Error("Not a Policy Iteration Lab run file");
  return { ...data, id: newRunId(), created_at: Date.now() / 1000 };
}
