// App shell: hash router, navigation, API-key box, theme toggle.

import { api, getKey, setKey } from "./api.js";
import { clear, h } from "./dom.js";
import { message } from "./ui.js";
import overview from "./pages/overview.js";
import playground from "./pages/playground.js";
import { tokensPage, memoryPage, kvquantPage } from "./pages/foundations.js";
import { enginePage, batchingPage, specPage } from "./pages/runtime.js";
import kernels from "./pages/kernels.js";
import production from "./pages/production.js";
import parallel from "./pages/parallel.js";

const GROUPS = [
  { title: "Start", cls: "", pages: [overview, playground] },
  { title: "1 · Foundations", cls: "c4", pages: [tokensPage, memoryPage, kvquantPage] },
  { title: "2 · Runtime", cls: "c3", pages: [enginePage, batchingPage, specPage] },
  { title: "3 · Kernels", cls: "c2", pages: [kernels] },
  { title: "4 · Production", cls: "c1", pages: [production] },
  { title: "Scale-out", cls: "c7", pages: [parallel] },
];
const PAGES = Object.fromEntries(GROUPS.flatMap((g) => g.pages).map((p) => [p.id, p]));

const ctx = { overview: null, navigate };
let current = null;

function buildNav() {
  const nav = document.getElementById("nav");
  clear(nav);
  for (const g of GROUPS) {
    nav.appendChild(h("div", { class: "nav-group" },
      h("div", { class: "nav-group-title" }, g.cls ? h("span", { class: `dot ${g.cls}` }) : null, g.title),
      g.pages.map((p) => h("a", { class: "nav-link", href: `#/${p.id}`, dataset: { page: p.id }, text: p.title }))));
  }
}

function navigate(id) {
  location.hash = `#/${id}`;
}

async function loadOverview() {
  const badge = document.getElementById("model-badge");
  const backend = document.getElementById("backend-badge");
  try {
    ctx.overview = await api("/lab/overview");
    const m = ctx.overview.model;
    badge.replaceChildren(h("span", { class: "dot good" }),
      h("span", { text: `${m.name} · ${(m.n_params / 1e6).toFixed(1)}M params` }));
    backend.textContent = `backend: ${ctx.overview.backend.backend}`;
    return true;
  } catch (err) {
    ctx.overview = null;
    badge.replaceChildren(h("span", { class: "dot bad" }),
      h("span", { text: err.status === 401 ? "API key needed" : "offline" }));
    return false;
  }
}

async function route() {
  const id = (location.hash.replace(/^#\//, "") || "overview").split("?")[0];
  const pageDef = PAGES[id] || overview;
  if (current && current.teardown) current.teardown();
  current = pageDef;
  document.querySelectorAll(".nav-link").forEach((a) => a.classList.toggle("active", a.dataset.page === pageDef.id));
  const main = document.getElementById("main");
  clear(main);
  if (!ctx.overview) await loadOverview();
  if (!ctx.overview) {
    main.appendChild(h("div", { class: "page-head" }, h("h1", { text: "Connect to the server" })));
    main.appendChild(message("info",
      "This server requires an API key. It was printed when the server started (or saved to .inferlab_api_key). " +
      "Paste it into the box at the top right and press Save key."));
    return;
  }
  document.title = `${pageDef.title} · InferLab`;
  try {
    main.appendChild(pageDef.render(ctx));
  } catch (err) {
    main.appendChild(message("error", `Failed to render page: ${err.message}`));
  }
  main.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

function initTheme() {
  const root = document.documentElement;
  let saved = null;
  try { saved = localStorage.getItem("inferlab_theme"); } catch { /* ignore */ }
  if (saved === "dark" || saved === "light") root.dataset.theme = saved;
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("inferlab_theme", root.dataset.theme); } catch { /* ignore */ }
  });
}

function initKey() {
  const input = document.getElementById("api-key");
  input.value = getKey();
  document.getElementById("keyform").addEventListener("submit", async (e) => {
    e.preventDefault();
    setKey(input.value.trim());
    ctx.overview = null;
    await route();
  });
}

initTheme();
initKey();
buildNav();
window.addEventListener("hashchange", route);
route();
