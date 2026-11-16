// Reusable UI pieces (cards, fields, tables, stat tiles, async buttons).

import { h } from "./dom.js";

export function page(kicker, title, intro, ...body) {
  return h("div", null,
    h("div", { class: "page-head" },
      h("div", { class: "kicker", text: kicker }),
      h("h1", { text: title }),
      intro ? h("p", { text: intro }) : null),
    ...body);
}

export function card(title, subtitle, ...body) {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h2", { text: title }), subtitle ? h("p", { text: subtitle }) : null),
    ...body);
}

export function callout(strong, rest) {
  return h("div", { class: "callout" }, h("strong", { text: strong }), " ", rest);
}

export function stat(label, value, sub) {
  return h("div", { class: "stat" },
    h("div", { class: "label", text: label }),
    h("div", { class: "value", text: value }),
    sub ? h("div", { class: "sub", text: sub }) : null);
}

export function stats(items) {
  return h("div", { class: "stats" }, items.map((i) => stat(i[0], i[1], i[2])));
}

let uid = 0;

export function field(label, input, valueEl) {
  const id = input.id || `f${++uid}`;
  input.id = id;
  return h("div", { class: "field" }, h("label", { for: id }, label, valueEl || null), input);
}

export function select(options, value) {
  return h("select", null, options.map((o) => {
    const [v, t] = Array.isArray(o) ? o : [o, o];
    const opt = h("option", { value: v, text: t });
    if (String(v) === String(value)) opt.selected = true;
    return opt;
  }));
}

export function num(value, min, max, step = 1) {
  const input = h("input", { type: "number", min, max, step });
  input.value = String(value);
  return input;
}

export function text(value, placeholder = "") {
  return h("input", { type: "text", value, placeholder, spellcheck: "false" });
}

export function area(value, rows = 3) {
  return h("textarea", { rows, spellcheck: "false" }, value);
}

export function slider(label, value, min, max, step, fmtFn = (v) => String(v)) {
  // set min/max/step BEFORE value, otherwise the browser snaps the value to the default 0..100 range
  const input = h("input", { type: "range", min, max, step });
  input.value = String(value);
  const val = h("span", { class: "val", text: fmtFn(Number(value)) });
  input.addEventListener("input", () => { val.textContent = fmtFn(Number(input.value)); });
  return { input, el: field(label, input, val) };
}

export function check(label, checked) {
  const input = h("input", { type: "checkbox", checked });
  return { input, el: h("label", { class: "check" }, input, label) };
}

export function seg(options, value, onChange) {
  const el = h("div", { class: "seg", role: "tablist" });
  let cur = value;
  const buttons = options.map(([v, t]) => {
    const b = h("button", { type: "button", text: t, role: "tab" });
    b.addEventListener("click", () => {
      cur = v;
      buttons.forEach((x) => x.classList.toggle("on", x === b));
      onChange(v);
    });
    if (v === value) b.classList.add("on");
    return b;
  });
  buttons.forEach((b) => el.appendChild(b));
  return { el, get: () => cur };
}

export function table(columns, rows) {
  // columns: [{key, label, num?, fmt?}]
  return h("div", { class: "table-wrap" },
    h("table", null,
      h("thead", null, h("tr", null, columns.map((c) => h("th", { class: c.num ? "num" : "", text: c.label })))),
      h("tbody", null, rows.map((r) => h("tr", null, columns.map((c) => {
        const v = typeof c.key === "function" ? c.key(r) : r[c.key];
        const out = c.fmt ? c.fmt(v, r) : v;
        return h("td", { class: c.num ? "num" : "" }, out instanceof Node ? out : String(out ?? "–"));
      }))))));
}

export function message(kind, textContent) {
  const icon = { error: "✕", ok: "✓", info: "ℹ" }[kind] || "";
  return h("div", { class: `msg ${kind}`, role: kind === "error" ? "alert" : "status" },
    h("span", { class: "status-icon", "aria-hidden": "true", text: icon }), h("span", { text: textContent }));
}

// Button that runs an async action, shows a spinner and renders errors into `out`.
export function runButton(label, out, action, cls = "primary") {
  const b = h("button", { type: "button", class: cls, text: label });
  b.addEventListener("click", async () => {
    b.disabled = true;
    const old = b.textContent;
    b.textContent = "";
    b.append(h("span", { class: "spinner" }), " ", old);
    try {
      await action();
    } catch (err) {
      if (out) {
        out.replaceChildren(message("error", err.message || String(err)));
      }
    } finally {
      b.disabled = false;
      b.textContent = old;
    }
  });
  return b;
}

export function code(textContent) {
  return h("pre", { class: "code", text: textContent });
}
