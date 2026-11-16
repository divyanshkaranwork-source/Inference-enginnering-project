// API client: JSON calls + Server-Sent-Events streaming over fetch.
// The API key lives in sessionStorage (cleared when the tab closes), is sent
// only to this origin, and never appears in URLs.

const KEY = "inferlab_api_key";

export function getKey() {
  try { return sessionStorage.getItem(KEY) || ""; } catch { return ""; }
}

export function setKey(k) {
  try {
    if (k) sessionStorage.setItem(KEY, k);
    else sessionStorage.removeItem(KEY);
  } catch { /* storage disabled: key lives only for this page */ }
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function headers() {
  const h = { "Content-Type": "application/json" };
  const k = getKey();
  if (k) h.Authorization = `Bearer ${k}`;
  return h;
}

async function errorFrom(res) {
  let msg = `HTTP ${res.status}`;
  try {
    const j = await res.json();
    if (j && j.error) {
      msg = j.error.message || msg;
      if (Array.isArray(j.error.details) && j.error.details.length) {
        msg += ": " + j.error.details.map((d) => `${d.loc.slice(1).join(".")} ${d.msg}`).join("; ");
      }
    }
  } catch { /* not JSON */ }
  if (res.status === 401) msg = "Missing or invalid API key - paste it in the box at the top right.";
  if (res.status === 429) msg = "Rate limited - " + msg;
  return new ApiError(res.status, msg);
}

export async function api(path, body, method) {
  const res = await fetch(path, {
    method: method || (body === undefined ? "GET" : "POST"),
    headers: headers(),
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!res.ok) throw await errorFrom(res);
  return res.json();
}

// POST a streaming request and call onEvent(obj, tArrival) for each SSE data event.
export async function stream(path, body, onEvent, signal) {
  const res = await fetch(path, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
    credentials: "same-origin",
    cache: "no-store",
    signal,
  });
  if (!res.ok) throw await errorFrom(res);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") return;
        let obj;
        try { obj = JSON.parse(data); } catch { continue; }
        if (obj.error) throw new ApiError(obj.error.code || 500, obj.error.message);
        onEvent(obj, performance.now());
      }
    }
  }
}
