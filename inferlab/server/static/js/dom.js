// Tiny DOM helpers. Text is ALWAYS set via textContent / text nodes - never
// innerHTML - so model output and user input can never inject markup (XSS).

const SVG_NS = "http://www.w3.org/2000/svg";

function apply(el, props) {
  if (!props) return el;
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "text") el.textContent = String(v);
    else if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "style") for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
    else if (k === "dataset") for (const [p, val] of Object.entries(v)) el.dataset[p] = val;
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = Boolean(v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === undefined || c === null || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function h(tag, props, ...children) {
  return append(apply(document.createElement(tag), props), children);
}

export function s(tag, props, ...children) {
  return append(apply(document.createElementNS(SVG_NS, tag), props), children);
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function fmt(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  if (typeof x !== "number") return String(x);
  if (!Number.isFinite(x)) return x > 0 ? "∞" : "–∞";
  if (Number.isInteger(x) && Math.abs(x) < 1e7) return x.toLocaleString("en-US");
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 1e7)) return x.toExponential(digits);
  if (a >= 1000) return Math.round(x).toLocaleString("en-US");
  return x.toFixed(a >= 100 ? 0 : a >= 10 ? Math.min(digits, 1) : digits);
}

export function compact(x) {
  if (x === null || x === undefined) return "–";
  const a = Math.abs(x);
  if (a >= 1e12) return (x / 1e12).toFixed(1) + "T";
  if (a >= 1e9) return (x / 1e9).toFixed(1) + "B";
  if (a >= 1e6) return (x / 1e6).toFixed(1) + "M";
  if (a >= 1e4) return (x / 1e3).toFixed(1) + "K";
  return fmt(x, 1);
}

export function bytes(n) {
  if (n === null || n === undefined) return "–";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  let v = n;
  while (Math.abs(v) >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

export function ms(x) {
  if (x === null || x === undefined) return "–";
  if (x >= 1000) return (x / 1000).toFixed(2) + " s";
  return fmt(x, x < 10 ? 2 : 1) + " ms";
}

// One shared tooltip for every chart mark (data-tip attribute).
const tip = () => document.getElementById("tooltip");

export function attachTip(el, text) {
  el.dataset.tip = text;
  el.setAttribute("tabindex", "0");
  el.setAttribute("aria-label", text);
  const show = (x, y) => {
    const t = tip();
    t.textContent = el.dataset.tip;
    t.hidden = false;
    const r = t.getBoundingClientRect();
    t.style.left = Math.min(x + 12, window.innerWidth - r.width - 8) + "px";
    t.style.top = Math.max(8, y - r.height - 10) + "px";
  };
  el.addEventListener("mousemove", (e) => show(e.clientX, e.clientY));
  el.addEventListener("mouseleave", () => { tip().hidden = true; });
  el.addEventListener("focus", () => {
    const b = el.getBoundingClientRect();
    show(b.left + b.width / 2, b.top);
  });
  el.addEventListener("blur", () => { tip().hidden = true; });
  return el;
}
