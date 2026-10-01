// Thin fetch wrapper around the FastAPI backend. Every call throws an Error with a readable message.

// A request the server never answers (e.g. it was restarted mid-request) must not hang the UI.
const REQUEST_TIMEOUT_MS = 30_000;

// Sent on every call: the server rejects state-changing requests without it (CSRF protection).
const BASE_HEADERS = { "X-Requested-With": "pi-lab" };

async function send(method, url, { body, contentType, raw = false } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(url, {
      method,
      credentials: "same-origin",
      headers: { ...BASE_HEADERS, ...(contentType ? { "Content-Type": contentType } : {}) },
      body,
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`The server did not answer within ${REQUEST_TIMEOUT_MS / 1000}s. Restart it with ./run.sh and try again.`);
    }
    throw new Error("Cannot reach the server. Start it with ./run.sh, then try again.");
  } finally {
    clearTimeout(timer);
  }
  // An expired session elsewhere asks the user to sign in again; a failed sign-in attempt does not.
  if (response.status === 401 && !url.startsWith("/api/auth/")) {
    window.dispatchEvent(new CustomEvent("pi:auth-required"));
  }
  if (raw && response.ok) return response;
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    throw new Error(formatDetail(data?.detail) || `Request failed (${response.status})`);
  }
  return data;
}

function request(method, url, json) {
  return send(method, url, json === undefined ? {} : { body: JSON.stringify(json), contentType: "application/json" });
}

function formatDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => `${(d.loc || []).slice(1).join(".")}: ${d.msg}`).join("; ");
  }
  return JSON.stringify(detail);
}

export const api = {
  envs: () => request("GET", "/api/envs"),
  layout: (envKey, options) => request("POST", "/api/layout", { env_key: envKey, options }),
  step: (payload) => request("POST", "/api/pi/step", payload),
  infer: (payload) => request("POST", "/api/infer", payload),
  benchmark: (payload) => request("POST", "/api/benchmark", payload),
  render: (payload) => request("POST", "/api/render", payload),
  session: () => request("GET", "/api/session"),
  login: (username, password) => request("POST", "/api/auth/login", { username, password }),
  logout: () => request("POST", "/api/auth/logout"),
  listRuns: () => request("GET", "/api/runs"),
  getRunGz: (id) => send("GET", `/api/runs/${id}`, { raw: true }),
  putRunGz: (id, gzBlob) => send("PUT", `/api/runs/${id}`, { body: gzBlob, contentType: "application/gzip" }),
  deleteRun: (id) => request("DELETE", `/api/runs/${id}`),
};
