// Small SVG chart kit. Conventions:
// thin marks (bars <= 20px, 2px lines, 4px rounded data ends), hairline grid,
// muted axis text, legend for >= 2 series, hover/focus tooltip on every mark,
// categorical colours in fixed slot order via classes c1..c8.

import { attachTip, clear, fmt, h, s } from "./dom.js";

const FONT = 11;

// Re-render on container resize so text stays 11px at every width.
export function responsive(draw, minWidth = 280) {
  const box = h("div", { class: "chart" });
  let last = 0;
  const render = () => {
    const w = Math.max(minWidth, Math.floor(box.clientWidth || 640));
    if (Math.abs(w - last) < 4) return;
    last = w;
    clear(box);
    box.appendChild(draw(w));
  };
  const ro = new ResizeObserver(render);
  ro.observe(box);
  requestAnimationFrame(render);
  return box;
}

export function legend(items) {
  return h("div", { class: "legend" },
    items.map((it) => h("span", { class: "key" }, h("span", { class: `sw ${it.cls}` }), it.name)));
}

export function niceTicks(min, max, count = 5) {
  if (!(max > min)) max = min + 1;
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => raw <= st) || 10 * mag;
  const start = Math.floor(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + step * 0.5; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

function hbarPath(x0, x1, y, t, r = 4) {
  const w = x1 - x0;
  if (w <= 0) return `M${x0},${y}h0`;
  const rr = Math.min(r, w, t / 2);
  return `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + t - rr}Q${x1},${y + t} ${x1 - rr},${y + t}H${x0}Z`;
}

function colPath(x, w, yb, yt, r = 4) {
  const hgt = yb - yt;
  if (hgt <= 0) return `M${x},${yb}h0`;
  const rr = Math.min(r, hgt, w / 2);
  return `M${x},${yb}V${yt + rr}Q${x},${yt} ${x + rr},${yt}H${x + w - rr}Q${x + w},${yt} ${x + w},${yt + rr}V${yb}Z`;
}

function textWidth(str) {
  return String(str).length * FONT * 0.58;
}

// ---------------------------------------------------------------------------
// Horizontal bars, one or more series grouped per category.
// categories: [label]; series: [{name, cls, values:[number]}]
export function hbars({ categories, series, valueFmt = (v) => fmt(v), maxValue, labelWidth }) {
  const groups = categories.length;
  const nS = series.length;
  return responsive((W) => {
    const lw = Math.min(labelWidth || Math.max(...categories.map(textWidth), 40) + 10, W * 0.42);
    const t = nS > 1 ? 9 : 14;
    const groupH = nS * t + (nS - 1) * 2 + 12;
    const H = groups * groupH + 22;
    const vmax = maxValue || Math.max(...series.flatMap((se) => se.values), 1e-12);
    const x0 = lw;
    const x1 = W - 64;
    const sx = (v) => x0 + (Math.max(v, 0) / vmax) * (x1 - x0);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    for (const tk of niceTicks(0, vmax, 4)) {
      if (tk > vmax * 1.001) continue;
      const x = sx(tk);
      svg.appendChild(s("line", { x1: x, x2: x, y1: 0, y2: H - 18, class: "gridline" }));
      svg.appendChild(s("text", { x, y: H - 4, "text-anchor": "middle" }, valueFmt(tk)));
    }
    categories.forEach((cat, gi) => {
      const gy = gi * groupH + 6;
      const label = String(cat);
      svg.appendChild(s("text", { x: lw - 8, y: gy + (nS * t + (nS - 1) * 2) / 2 + 4, "text-anchor": "end",
        class: "label-strong" }, label.length > 40 ? label.slice(0, 39) + "…" : label));
      series.forEach((se, si) => {
        const v = se.values[gi];
        const y = gy + si * (t + 2);
        const p = s("path", { d: hbarPath(x0, sx(v), y, t), class: `mark ${se.cls}` });
        attachTip(p, `${cat}\n${se.name}: ${valueFmt(v)}`);
        svg.appendChild(p);
        if (nS === 1 || si === nS - 1 || nS <= 2) {
          svg.appendChild(s("text", { x: sx(v) + 5, y: y + t / 2 + 4, class: "value-label" }, valueFmt(v)));
        }
      });
    });
    svg.appendChild(s("line", { x1: x0, x2: x0, y1: 0, y2: H - 18, class: "axis" }));
    return svg;
  });
}

// Vertical columns (single series, or stacked when several series).
export function columns({ labels, series, yLabel = "", valueFmt = (v) => fmt(v), height = 200, xEvery }) {
  return responsive((W) => {
    const H = height;
    const left = 44;
    const right = 10;
    const top = 10;
    const bottom = 24;
    const n = labels.length || 1;
    const totals = labels.map((_, i) => series.reduce((a, se) => a + (se.values[i] || 0), 0));
    const vmax = Math.max(...totals, 1e-12);
    const ticks = niceTicks(0, vmax, 4);
    const ymax = ticks[ticks.length - 1];
    const sy = (v) => top + (1 - v / ymax) * (H - top - bottom);
    const band = (W - left - right) / n;
    const bw = Math.max(1, Math.min(20, band * 0.7));
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    for (const tk of ticks) {
      svg.appendChild(s("line", { x1: left, x2: W - right, y1: sy(tk), y2: sy(tk), class: "gridline" }));
      svg.appendChild(s("text", { x: left - 6, y: sy(tk) + 4, "text-anchor": "end" }, valueFmt(tk)));
    }
    const every = xEvery || Math.ceil(n / Math.max(1, Math.floor((W - left) / 48)));
    labels.forEach((lab, i) => {
      const x = left + i * band + (band - bw) / 2;
      let base = 0;
      series.forEach((se, si) => {
        const v = se.values[i] || 0;
        if (v <= 0) return;
        const isTop = series.slice(si + 1).every((o) => !(o.values[i] > 0));
        const yb = sy(base) - (base > 0 ? 2 : 0); // 2px surface gap between stacked parts
        const yt = sy(base + v);
        const d = isTop ? colPath(x, bw, yb, yt) : `M${x},${yb}V${yt}H${x + bw}V${yb}Z`;
        const p = s("path", { d, class: `mark ${se.cls}` });
        const parts = series.map((o) => `${o.name}: ${valueFmt(o.values[i] || 0)}`).join("\n");
        attachTip(p, `${lab}\n${parts}`);
        svg.appendChild(p);
        base += v;
      });
      if (i % every === 0) {
        svg.appendChild(s("text", { x: x + bw / 2, y: H - 8, "text-anchor": "middle" }, String(lab)));
      }
    });
    svg.appendChild(s("line", { x1: left, x2: W - right, y1: sy(0), y2: sy(0), class: "axis" }));
    if (yLabel) {
      svg.appendChild(s("text", { x: 4, y: 10 }, yLabel));
    }
    return svg;
  });
}

// Line chart with crosshair tooltip. series: [{name, cls, points:[[x,y]]}]
export function lines({ series, xLabel = "", yLabel = "", xFmt = (v) => fmt(v), yFmt = (v) => fmt(v),
  height = 220, logY = false, yMin }) {
  return responsive((W) => {
    const H = height;
    const left = 48;
    const right = 14;
    const top = 14;
    const bottom = 30;
    const xs = series.flatMap((se) => se.points.map((p) => p[0]));
    const ys = series.flatMap((se) => se.points.map((p) => p[1])).filter((v) => Number.isFinite(v));
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs, xmin + 1);
    let ymin = yMin !== undefined ? yMin : Math.min(0, ...ys);
    let ymax = Math.max(...ys, ymin + 1e-9);
    const tf = logY ? (v) => Math.log10(Math.max(v, 1e-9)) : (v) => v;
    let ticks;
    if (logY) {
      ymin = 10 ** Math.floor(Math.log10(Math.max(Math.min(...ys), 1e-9)));
      ymax = 10 ** Math.ceil(Math.log10(Math.max(ymax, ymin * 10)));
      ticks = [];
      for (let v = ymin; v <= ymax * 1.0001; v *= 10) ticks.push(v);
    } else {
      ticks = niceTicks(ymin, ymax, 4);
      ymin = ticks[0];
      ymax = ticks[ticks.length - 1];
    }
    const sx = (v) => left + ((v - xmin) / (xmax - xmin)) * (W - left - right);
    const sy = (v) => top + (1 - (tf(v) - tf(ymin)) / (tf(ymax) - tf(ymin))) * (H - top - bottom);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    for (const tk of ticks) {
      svg.appendChild(s("line", { x1: left, x2: W - right, y1: sy(tk), y2: sy(tk), class: "gridline" }));
      svg.appendChild(s("text", { x: left - 6, y: sy(tk) + 4, "text-anchor": "end" }, yFmt(tk)));
    }
    for (const tk of niceTicks(xmin, xmax, Math.max(2, Math.floor(W / 110)))) {
      if (tk < xmin || tk > xmax) continue;
      svg.appendChild(s("text", { x: sx(tk), y: H - 14, "text-anchor": "middle" }, xFmt(tk)));
    }
    if (xLabel) svg.appendChild(s("text", { x: W - right, y: H - 2, "text-anchor": "end" }, xLabel));
    if (yLabel) svg.appendChild(s("text", { x: 4, y: 10 }, yLabel));
    svg.appendChild(s("line", { x1: left, x2: W - right, y1: H - bottom, y2: H - bottom, class: "axis" }));
    for (const se of series) {
      const pts = se.points.filter((p) => Number.isFinite(p[1]));
      if (!pts.length) continue;
      const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
      svg.appendChild(s("path", { d, class: `line ${se.cls}` }));
      const lp = pts[pts.length - 1];
      svg.appendChild(s("circle", { cx: sx(lp[0]), cy: sy(lp[1]), r: 4, class: `${se.cls} ring` }));
    }
    // crosshair
    const cross = s("line", { x1: 0, x2: 0, y1: top, y2: H - bottom, class: "axis", visibility: "hidden" });
    svg.appendChild(cross);
    const overlay = s("rect", { x: left, y: top, width: W - left - right, height: H - top - bottom,
      fill: "transparent", class: "mark" });
    const allX = [...new Set(xs)].sort((a, b) => a - b);
    const tipEl = () => document.getElementById("tooltip");
    overlay.addEventListener("mousemove", (e) => {
      const r = svg.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      const xv = xmin + ((px - left) / (W - left - right)) * (xmax - xmin);
      let best = allX[0];
      for (const x of allX) if (Math.abs(x - xv) < Math.abs(best - xv)) best = x;
      cross.setAttribute("x1", sx(best));
      cross.setAttribute("x2", sx(best));
      cross.setAttribute("visibility", "visible");
      const linesTxt = series.map((se) => {
        const p = se.points.find((q) => q[0] === best);
        return `${se.name}: ${p ? yFmt(p[1]) : "–"}`;
      });
      const t = tipEl();
      t.textContent = `${xLabel || "x"} ${xFmt(best)}\n${linesTxt.join("\n")}`;
      t.hidden = false;
      t.style.left = Math.min(e.clientX + 12, window.innerWidth - 220) + "px";
      t.style.top = e.clientY - 40 + "px";
    });
    overlay.addEventListener("mouseleave", () => {
      cross.setAttribute("visibility", "hidden");
      tipEl().hidden = true;
    });
    svg.appendChild(overlay);
    return svg;
  });
}

// Log-log roofline: attainable = min(peak, AI * bandwidth).
export function roofline({ peakTflops, bandwidthGBs, points }) {
  return responsive((W) => {
    const H = 310;
    const left = 56;
    const right = 16;
    const top = 26;
    const bottom = 36;
    const xmin = 0.1;
    const xmax = 10000;
    const ymin = 0.01;
    const ymax = 10 ** Math.ceil(Math.log10(peakTflops * 1.5));
    const lx = (v) => Math.log10(v);
    const sx = (v) => left + ((lx(v) - lx(xmin)) / (lx(xmax) - lx(xmin))) * (W - left - right);
    const sy = (v) => top + (1 - (lx(Math.max(v, ymin)) - lx(ymin)) / (lx(ymax) - lx(ymin))) * (H - top - bottom);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    for (let v = xmin; v <= xmax * 1.001; v *= 10) {
      svg.appendChild(s("line", { x1: sx(v), x2: sx(v), y1: top, y2: H - bottom, class: "gridline" }));
      svg.appendChild(s("text", { x: sx(v), y: H - bottom + 14, "text-anchor": "middle" }, fmt(v, 1)));
    }
    for (let v = ymin; v <= ymax * 1.001; v *= 10) {
      svg.appendChild(s("line", { x1: left, x2: W - right, y1: sy(v), y2: sy(v), class: "gridline" }));
      svg.appendChild(s("text", { x: left - 6, y: sy(v) + 4, "text-anchor": "end" }, fmt(v, 2)));
    }
    svg.appendChild(s("text", { x: W - right, y: H - 4, "text-anchor": "end" }, "arithmetic intensity (FLOP / byte)"));
    svg.appendChild(s("text", { x: 4, y: 11 }, "TFLOP/s"));
    const bw = bandwidthGBs / 1000; // TB/s -> TFLOP/s per FLOP/byte
    const ridge = peakTflops / bw;
    const pts = [];
    for (let v = xmin; v <= xmax; v *= 1.2) pts.push([v, Math.min(peakTflops, v * bw)]);
    pts.push([xmax, peakTflops]);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
    svg.appendChild(s("path", { d, class: "line c7" }));
    svg.appendChild(s("line", { x1: sx(ridge), x2: sx(ridge), y1: sy(peakTflops), y2: H - bottom, class: "axis" }));
    svg.appendChild(s("text", { x: sx(ridge) + 4, y: sy(peakTflops) + 14, class: "label-strong" },
      `ridge ${fmt(ridge, 0)} FLOP/B`));
    svg.appendChild(s("text", { x: sx(0.3), y: sy(0.3 * bw) - 8, class: "label-strong" }, "memory-bound"));
    svg.appendChild(s("text", { x: W - right - 4, y: sy(peakTflops) - 6, "text-anchor": "end", class: "label-strong" },
      "compute-bound"));
    for (const p of points) {
      const cls = p.phase === "prefill" ? "c2" : "c1";
      const c = s("circle", { cx: sx(Math.min(Math.max(p.ai, xmin), xmax)), cy: sy(p.tflops), r: 5,
        class: `mark ring ${cls}` });
      attachTip(c, `${p.label}\nAI ${fmt(p.ai, 1)} FLOP/B\n${fmt(p.tflops, 1)} TFLOP/s attainable\n${p.bound}-bound`);
      svg.appendChild(c);
    }
    return svg;
  });
}

// Gantt: rows (e.g. pipeline stages) x time blocks.
export function gantt({ rows, rowLabel, events, xFmt = (v) => fmt(v) + " ms" }) {
  return responsive((W) => {
    const left = 70;
    const right = 12;
    const rh = 22;
    const H = rows * (rh + 6) + 30;
    const tmax = Math.max(...events.map((e) => e.end), 1e-9);
    const sx = (v) => left + (v / tmax) * (W - left - right);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });
    for (const tk of niceTicks(0, tmax, Math.max(2, Math.floor(W / 120)))) {
      if (tk > tmax) continue;
      svg.appendChild(s("line", { x1: sx(tk), x2: sx(tk), y1: 0, y2: H - 22, class: "gridline" }));
      svg.appendChild(s("text", { x: sx(tk), y: H - 8, "text-anchor": "middle" }, xFmt(tk)));
    }
    for (let r = 0; r < rows; r++) {
      svg.appendChild(s("text", { x: left - 8, y: r * (rh + 6) + rh / 2 + 4, "text-anchor": "end",
        class: "label-strong" }, rowLabel(r)));
    }
    for (const e of events) {
      const x = sx(e.start);
      const w = Math.max(sx(e.end) - x - 2, 1); // 2px surface gap between touching blocks
      const y = e.row * (rh + 6);
      const rect = s("rect", { x, y, width: w, height: rh, rx: 4, class: `mark ${e.cls || "c1"}` });
      attachTip(rect, e.tip);
      svg.appendChild(rect);
      if (e.label && w > textWidth(e.label) + 8) {
        svg.appendChild(s("text", { x: x + 6, y: y + rh / 2 + 4, class: "inside-label" }, e.label));
      }
    }
    return svg;
  });
}

// Tall segmented bar (one whole split into parts) + optional budget marker.
export function segbar(parts, total, unitFmt) {
  const sum = parts.reduce((a, p) => a + p.value, 0);
  const whole = Math.max(total || sum, sum, 1e-12);
  const bar = h("div", { class: "segbar", role: "img",
    "aria-label": parts.map((p) => `${p.name} ${unitFmt(p.value)}`).join(", ") });
  for (const p of parts) {
    if (p.value <= 0) continue;
    const pct = (p.value / whole) * 100;
    const part = h("div", { class: `part ${p.cls}`, style: { width: pct + "%" } });
    if (pct > 12) part.textContent = unitFmt(p.value);
    attachTip(part, `${p.name}: ${unitFmt(p.value)} (${pct.toFixed(1)}%)`);
    bar.appendChild(part);
  }
  if (total && total > sum) {
    const free = h("div", { class: "part cn", style: { width: ((total - sum) / whole) * 100 + "%" } });
    if ((total - sum) / whole > 0.15) free.textContent = "free " + unitFmt(total - sum);
    attachTip(free, `free: ${unitFmt(total - sum)}`);
    bar.appendChild(free);
  }
  return bar;
}
