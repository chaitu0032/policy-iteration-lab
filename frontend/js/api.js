// Thin fetch wrapper around the FastAPI backend. Every call throws an Error with a readable message.

async function request(method, url, body) {
  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(`Cannot reach the server (${err.message}). Is uvicorn running?`);
  }
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
};
