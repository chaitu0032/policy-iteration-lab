// Minimal immutable store: every update produces a new state object and notifies subscribers.

export function createStore(initial) {
  let state = Object.freeze({ ...initial });
  const listeners = new Set();

  return {
    get: () => state,
    set(patch) {
      const next = typeof patch === "function" ? patch(state) : patch;
      const changed = Object.keys(next).filter((k) => next[k] !== state[k]);
      if (!changed.length) return;
      state = Object.freeze({ ...state, ...next });
      listeners.forEach((fn) => fn(state, new Set(changed)));
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
  };
}
