/* Second Loops: interactive figures. All numbers come from data.js (scripts/export_blog_data.py). */
(() => {
"use strict";
const D = window.BLOG_DATA;
const RM = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const NS = "http://www.w3.org/2000/svg";
const $ = (id) => document.getElementById(id);

function s(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) {
    const v = attrs[k];
    if (v == null) continue;
    if (typeof v === "string" && v.startsWith("var(")) e.style.setProperty(k, v);
    else if (k === "text") e.textContent = v;
    else if (k === "class") e.setAttribute("class", v);
    else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "style") e.setAttribute("style", v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c);
  return e;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const pct = (v, d = 0) => (v == null ? "n/a" : `${(+v).toFixed(d)}%`);
const r0 = (v) => Math.round(v);

const MODELS = {
  ouro: { label: "Ouro", full: "Ouro-1.4B-Thinking", color: "var(--ouro)", loops: 4, layers: 24, dir: "Ouro-1.4B-Thinking", tags: "grid_test{,_b}" },
  nanbeige: { label: "Nanbeige", full: "Nanbeige4.2-3B", color: "var(--nanbeige)", loops: 2, layers: 22, dir: "Nanbeige4.2-3B", tags: "grid_test{,_b}" },
  dory: { label: "Dory", full: "Dory 1.0 4B", color: "var(--dory)", loops: 2, layers: 40, ramps: true, dir: "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", tags: "grid_test{,_b}" },
  dory4: { label: "Dory × 4 loops", full: "Dory 1.0 4B run with 4 loops", color: "var(--dory)", loops: 4, layers: 40, ramps: true, dir: "NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", tags: "grid_test4{,_b}" },
  qwen: { label: "Qwen (dense)", full: "Qwen3-8B", color: "var(--dense)", loops: 4, layers: 36, straight: true, dir: "Qwen3-8B", tags: "grid_test{,_b}" },
  /* a looped mixture of experts; on the setup cards only until its results are in */
  inkling: { label: "Inkling", full: "Inkling 2 Mini", color: "var(--inkling)", loops: 3, layers: 12, dir: "Inkling-2-Mini", tags: "grid_test" },
};
const QLAYERS = ["0–8", "9–17", "18–26", "27–35"];
/* the order models appear in everywhere: Dory first (and the default), then Nanbeige, Ouro and the dense Qwen */
const ORDER4 = ["dory", "nanbeige", "ouro", "qwen"], ORDER5 = ["dory", "dory4", "nanbeige", "ouro", "qwen"];
const unitName = (m, k) => (MODELS[m].straight ? `Q${k}` : `Loop ${k}`);
const unitShort = (m, k) => (MODELS[m].straight ? `Q${k}` : `L${k}`);
const lc = (label) => label.replace(/\b(Loop|Every|All|Random)\b/g, (w) => w.toLowerCase()); /* lower-case words, keep Q1..Q4 */
const allKey = (n) => Array.from({ length: n }, (_, i) => i + 1).join("");
const SOURCES = [];
function src(text) { SOURCES.push(text); } /* data files, listed once at the end of the methods */

/* ------------------------------------------------------------------ markdown, for model replies */
/* A small renderer for what the models write: headings, lists, bold / italic, inline code, code blocks, tables, and
   LaTeX (KaTeX's MathML output when KaTeX loaded, the TeX source in mono otherwise). Text is escaped first; links
   are shown as plain text. */
const escHtml = (t) => t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function mathHtml(tex, display) {
  if (window.katex) {
    try { return window.katex.renderToString(tex, { output: "mathml", displayMode: display, throwOnError: false, errorColor: "inherit" }); } catch (e) { /* fall through */ }
  }
  return `<code class="tex">${escHtml(tex)}</code>`;
}
/* $...$ only when it reads as maths: no space just inside the dollars, no digit right after the closing one (prices),
   no escaped dollar, and no run of plain words (prose caught between two stray dollars, common in reasoning traces).
   A pair that fails leaves its first dollar as text and tries again from the second. */
function looksTex(t, next) {
  if (!t || t.length > 300 || /^\s|\s$|\n/.test(t) || /\d/.test(next || "")) return false;
  return !/[A-Za-z]{3,}[\s,.;:!?]+[A-Za-z]{3,}/.test(t.replace(/\\text\{[^}]*\}|\\[a-zA-Z]+/g, " "));
}
function texPairs(t, render) {
  let out = "", i = 0;
  for (;;) {
    const a = t.indexOf("$", i);
    if (a < 0) break;
    if (a > 0 && t[a - 1] === "\\") { out += t.slice(i, a - 1) + "$"; i = a + 1; continue; }
    const b = t.indexOf("$", a + 1);
    if (b < 0) break;
    if (looksTex(t.slice(a + 1, b), t[b + 1])) { out += t.slice(i, a) + render(t.slice(a + 1, b)); i = b + 1; }
    else { out += t.slice(i, a + 1); i = a + 1; }
  }
  return out + t.slice(i);
}
/* display maths ($$...$$, \[...\]) and \(...\), left as text when implausibly long (an unclosed delimiter) */
function texBlocks(t, stash) {
  t = t.replace(/\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$/g, (m, a, b) => (m.length > 800 ? m : stash(mathHtml(a || b, true))));
  return t.replace(/\\\(([\s\S]+?)\\\)/g, (m, a) => (m.length > 400 ? m : stash(mathHtml(a, false))));
}
/* a reasoning trace: plain text (shown pre-wrapped) with its maths rendered */
function texText(t) {
  const keep = [];
  const stash = (html) => `\u0000${keep.push(html) - 1}\u0000`;
  t = texPairs(texBlocks(String(t || ""), stash), (x) => stash(mathHtml(x, false)));
  return escHtml(t).replace(/\u0000(\d+)\u0000/g, (_, i) => keep[+i]);
}
function mdInline(t) {
  const keep = [];
  const stash = (html) => `\u0000${keep.push(html) - 1}\u0000`;
  t = t.replace(/`([^`]+)`/g, (_, c) => stash(`<code>${escHtml(c)}</code>`));
  t = texPairs(texBlocks(t, stash), (x) => stash(mathHtml(x, false)));
  t = escHtml(t);
  t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  t = t.replace(/\*\*([^*]+?)\*\*|__([^_]+?)__/g, (_, a, b) => `<strong>${a || b}</strong>`);
  t = t.replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\*)/g, "$1<em>$2</em>");
  return t.replace(/\u0000(\d+)\u0000/g, (_, i) => keep[+i]);
}
function markdown(text) {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n"), out = [];
  let para = [], list = null, gap = false;
  const flushPara = () => { if (para.length) out.push(`<p>${mdInline(para.join(" "))}</p>`); para = []; };
  const flushList = () => {
    if (list) out.push(`<${list.tag}${list.start > 1 ? ` start="${list.start}"` : ""}>${list.items.map((x) => `<li>${mdInline(x).replace(/\u0001/g, "<br>• ")}</li>`).join("")}</${list.tag}>`);
    list = null;
  };
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i], tr = ln.trim();
    let m;
    /* a blank line ends a paragraph; a list stays open across it if the next line is another item or indented */
    if (tr && gap && list && !/^([-*+]|\d+[.)])\s+/.test(tr) && !/^\s{2,}/.test(ln)) flushList();
    if (tr) gap = false;
    if (tr.startsWith("```")) {
      flushPara(); flushList();
      const code = [];
      while (++i < lines.length && !lines[i].trim().startsWith("```")) code.push(lines[i]);
      out.push(`<pre><code>${escHtml(code.join("\n"))}</code></pre>`);
    } else if (!tr) { flushPara(); gap = true; }
    else if ((m = tr.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<h6 class="md-h">${mdInline(m[2])}</h6>`); }
    else if (/^(-{3,}|\*{3,}|_{3,})$/.test(tr)) { flushPara(); flushList(); out.push("<hr>"); }
    else if (tr.startsWith("|") && i + 1 < lines.length && /^\|?[\s:|-]+\|?$/.test(lines[i + 1].trim()) && lines[i + 1].includes("-")) {
      flushPara(); flushList();
      const cells = (r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => mdInline(c.trim()));
      const head = cells(tr), body = [];
      i++;
      while (i + 1 < lines.length && lines[i + 1].trim().startsWith("|")) body.push(cells(lines[++i]));
      out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
    } else if ((m = tr.match(/^([-*+]|\d+[.)])\s+(.*)$/))) {
      flushPara();
      const tag = /\d/.test(m[1]) ? "ol" : "ul";
      if (list && list.tag !== tag && /^\s{2,}/.test(ln)) { list.items[list.items.length - 1] += `\u0001${m[2]}`; continue; } /* nested item, shown on its own line */
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [], start: tag === "ol" ? parseInt(m[1], 10) : 1 }; }
      list.items.push(m[2]);
    } else if (tr.startsWith(">")) { flushPara(); flushList(); out.push(`<blockquote>${mdInline(tr.replace(/^>\s?/, ""))}</blockquote>`); }
    else if (list && /^\s{2,}/.test(ln)) list.items[list.items.length - 1] += " " + tr;
    else { flushList(); para.push(tr); }
  }
  flushPara(); flushList();
  return out.join("");
}

/* ------------------------------------------------------------------ tooltip */
const tip = $("tip");
function moveTip(e) {
  const x = Math.min(window.innerWidth - 275, e.clientX + 14);
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${e.clientY + 16}px`;
}
function addTip(node, html) {
  node.style.cursor = "default";
  node.addEventListener("pointerenter", (e) => { tip.innerHTML = typeof html === "function" ? html() : html; tip.classList.add("on"); moveTip(e); });
  node.addEventListener("pointermove", moveTip);
  node.addEventListener("pointerleave", () => tip.classList.remove("on"));
}

/* ------------------------------------------------------------------ chips */
function chips(el, items, { value, onChange, swatch = false, multi = false } = {}) {
  el.innerHTML = "";
  const btns = items.map((it) => {
    const b = h("button", { class: "chip", type: "button", "aria-pressed": String(it.id === value), disabled: it.disabled || null,
      title: it.title || null }, swatch && it.color ? h("span", { class: "sw", style: `background:${it.color}` }) : null, it.label);
    b.addEventListener("click", () => {
      if (b.disabled) return;
      if (!multi) btns.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      onChange(it.id, b);
    });
    el.append(b);
    return b;
  });
  return btns;
}

/* ------------------------------------------------------------------ scoreboard: a big number that counts to its value */
class Board {
  /* bare: the number alone, without the caption rows (the hero tiles) */
  constructor(el, { title = "", color = "var(--ink)", bare = false } = {}) {
    el.innerHTML = "";
    this.color = color; this.cur = null; this.raf = 0;
    this.tl = h("span"); this.tr = h("span"); this.bl = h("span"); this.br = h("span");
    this.v = h("span", { class: "bv" }); this.u = h("span", { class: "bu" });
    this.num = h("div", { class: "bnum", role: "img" }, this.v, this.u);
    if (bare) el.append(this.num);
    else el.append(h("div", { class: "bl" }, this.tl, this.tr), this.num, h("div", { class: "bl" }, this.bl, this.br));
    this.tl.innerHTML = title;
  }
  text(num, unit = "") {
    this.v.textContent = num; this.u.textContent = unit;
    this.num.style.color = this.color;
    this.num.setAttribute("aria-label", num + unit);
  }
  set(v, o = {}) {
    if (o.color) this.color = o.color;
    if (o.title != null) this.tl.innerHTML = o.title;
    if (o.right != null) this.tr.innerHTML = o.right;
    if (o.sub != null) this.bl.innerHTML = o.sub;
    if (o.subRight != null) this.br.innerHTML = o.subRight;
    const unit = o.suffix ?? "%";
    const show = (x) => this.text(`${o.sign && Math.round(x) > 0 ? "+" : ""}${Math.round(x)}`, unit);
    cancelAnimationFrame(this.raf);
    this.num.classList.toggle("idle", v == null);
    if (v == null) { this.text("–"); this.cur = null; return; }
    const from = this.cur ?? v;
    this.cur = v;
    if (RM || o.animate === false || from === v) { show(v); return; }
    const t0 = performance.now(), dur = 650;
    const step = (t) => {
      const p = clamp((t - t0) / dur, 0, 1);
      show(lerp(from, v, easeOut(p)));
      if (p < 1) this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }
}

/* ------------------------------------------------------------------ two columns of prompts */
/* Each dot is one test prompt (or problem). It sits in the left column when most of its samples show the behaviour
   (refused, solved) and in the right column otherwise; its fill is the share of samples that did. When steering
   changes, the dots that switch sides slide across. */
/* how a dot is drawn for each kick outcome (export_blog_data.KICK_OUT): solved filled; a finished wrong answer a ring;
   no answer a dashed ring; cut off: lighter to darker grey for still reasoning, repeating itself, incoherent */
const KICK_STYLE = [{ f: 1 }, { f: 0 }, { f: 0, dash: "2 2" }, { f: 0.22 }, { f: 0.55 }, { f: 0.55, dash: "2 2" }, { f: 0.4 }];
class Split {
  /* marks: dots drawn with a dark ring (prompts whose replies can be read); onPick(i) when a dot is clicked (marked
     dots only, or every dot with pickAll); select(i) rings the dot being read */
  constructor(el, { n = 100, labels = ["Refuses", "Complies"], tipFn = null, marks = [], onPick = null, pickAll = false } = {}) {
    el.innerHTML = "";
    this.labels = labels; this.n = n; this.vals = null; this.marks = new Set(marks); this.sel = null;
    const d = 13, cols = 10, gap = 28, top = 26;
    Object.assign(this, { d, cols, top, colX: [4 + d / 2, 4 + d / 2 + cols * d + gap] });
    this.svg = s("svg", { viewBox: `0 0 ${2 * cols * d + gap + 8} ${top + 10 * d + 4}`, role: "img", class: "split" }, el);
    s("line", { x1: 4 + cols * d + gap / 2, x2: 4 + cols * d + gap / 2, y1: 4, y2: top + 10 * d, stroke: "var(--line)", "stroke-width": 1, "stroke-dasharray": "3 3" }, this.svg);
    this.head = labels.map((_, c) => s("text", { x: this.colX[c] - d / 2, y: 14, class: "split-h" }, this.svg));
    this.c = Array.from({ length: n }, (_, i) => {
      const c = s("circle", { cx: 0, cy: 0, r: d * 0.39, "stroke-width": 1.3 }, this.svg);
      if (tipFn) addTip(c, () => tipFn(i, this.vals ? this.vals[i] : null));
      if (onPick && (pickAll || this.marks.has(i))) {
        c.style.cursor = "pointer"; c.setAttribute("tabindex", "0"); c.setAttribute("role", "button"); c.setAttribute("data-i", i);
        c.addEventListener("click", () => onPick(i));
        c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPick(i); } });
      }
      return c;
    });
    for (const i of this.marks) this.svg.appendChild(this.c[i]); /* on top */
    this.side = new Array(n).fill(1);
    this.first = true;
  }
  select(i) { this.sel = i; if (this.last) this.set(...this.last); }
  /* cats: optional category per dot (kicks: KICK_STYLE); dots of one category sit together in their column */
  set(vals, color, label, cats = null) {
    this.vals = vals; this.last = [vals, color, label, cats];
    const side = this.c.map((_, i) => (vals && vals[i] != null ? (vals[i] >= 0.5 ? 0 : 1) : this.side[i]));
    const count = [0, 0], slot = [0, 0];
    side.forEach((sd, i) => { if (vals && vals[i] != null) count[sd]++; });
    const order = this.c.map((_, i) => i);
    if (cats) order.sort((a, b) => (cats[a] ?? 9) - (cats[b] ?? 9) || a - b);
    let moved = 0;
    for (const i of order) {
      const c = this.c[i];
      const sd = side[i], j = slot[sd]++, v = vals ? vals[i] : null;
      const x = this.colX[sd] + (j % this.cols) * this.d, y = this.top + this.d / 2 + Math.floor(j / this.cols) * this.d;
      c.style.transitionDelay = this.first ? "0ms" : `${(sd !== this.side[i] ? moved++ : 0) * 9}ms`;
      if (this.first) c.style.transition = "none";
      c.style.transform = `translate(${x}px, ${y}px)`;
      const mk = this.marks.has(i), on = this.sel === i;
      if (Array.isArray(color)) { /* a ramp: the fill's colour is the share of samples, blue (none) to red (all) */
        const f = v == null ? null : rgbKey(mixRGB(color[0], color[1], v));
        c.style.fill = f || "none"; c.style.stroke = mk || on ? "var(--ink)" : f || "var(--ink-3)";
        c.style.fillOpacity = 1; c.style.strokeOpacity = v == null ? 0.25 : 1;
        c.setAttribute("stroke-width", on ? 3.2 : mk ? 2.2 : 1);
        if (v == null) c.setAttribute("stroke-dasharray", "2 3"); else c.removeAttribute("stroke-dasharray");
        continue;
      }
      c.style.fill = color; c.style.stroke = mk || on ? "var(--ink)" : color;
      c.setAttribute("stroke-width", on ? 3 : 1.3);
      const k = cats && v != null ? KICK_STYLE[cats[i]] : null;
      if (v == null) { c.style.fillOpacity = 0; c.style.strokeOpacity = 0.25; c.setAttribute("stroke-dasharray", "2 3"); }
      else if (k) {
        c.style.fillOpacity = k.f; c.style.strokeOpacity = 0.9;
        if (k.dash) c.setAttribute("stroke-dasharray", k.dash); else c.removeAttribute("stroke-dasharray");
      } else { c.style.fillOpacity = 0.06 + 0.94 * v; c.style.strokeOpacity = 0.85; c.removeAttribute("stroke-dasharray"); }
    }
    if (this.first) { this.first = false; requestAnimationFrame(() => this.c.forEach((c) => { c.style.transition = ""; })); }
    this.side = side;
    this.head.forEach((t, k) => { t.textContent = vals ? `${this.labels[k]} · ${count[k]}` : this.labels[k]; });
    this.svg.setAttribute("aria-label", label || (vals ? `${count[0]} ${this.labels[0].toLowerCase()}, ${count[1]} ${this.labels[1].toLowerCase()}` : "not run"));
  }
}

/* ------------------------------------------------------------------ replies behind the dots */
/* gzipped JSON, base64-encoded as text (samples/<path>.txt, scripts/export_blog_samples.py), fetched once per file on
   the first click */
const SAMPLES = new Map();
function loadSamples(path) {
  if (!SAMPLES.has(path)) {
    SAMPLES.set(path, (async () => {
      const res = await fetch(`samples/${path}.txt`);
      if (!res.ok) throw new Error(`${res.status}`);
      const b64 = (await res.text()).trim();
      if (typeof DecompressionStream === "undefined") throw new Error("this browser cannot unpack them");
      const bin = atob(b64), buf = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) buf[k] = bin.charCodeAt(k);
      const text = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
      return JSON.parse(text);
    })().catch((e) => { SAMPLES.delete(path); throw e; }));
  }
  return SAMPLES.get(path);
}
const fmtInt = (n) => Number(n).toLocaleString("en-US");
/* one reply: the prompt, a who / verdict line, the reasoning folded behind "Thought for N tokens" (opened on a click),
   and the answer, always shown */
function replyView(el, { prompt, promptLabel = "Prompt", who, verdict = "", col = "var(--ink-3)", think = null, tokens = 0, closed = true, answer = "", note = "" }) {
  clearEl(el);
  if (prompt != null) { /* normal text flow (inside the .md grid every formula took a row of its own); figures folded away */
    const figs = [], text = String(prompt).replace(/\[asy\]([\s\S]*?)\[\/asy\]/g, (_, c) => { figs.push(c.trim()); return "[figure]"; });
    const p = h("div", { class: "md-inline" }); p.innerHTML = mdInline(text);
    const box = h("div", { class: "ex-prompt" }, h("span", { class: "label" }, promptLabel), p);
    figs.forEach((c, k) => box.append(h("details", { class: "think asy" }, h("summary", {}, `${figs.length > 1 ? `Figure ${k + 1}` : "The figure"}, as the Asymptote code the model saw`), h("pre", {}, c))));
    el.append(box);
  }
  const card = h("div", { class: "reply" }, h("div", { class: "who" }, h("span", {}, who), h("span", { class: "verdict", style: `--c:${col}` }, verdict)));
  if (think != null) {
    const body = h("div", { class: "think-body" });
    const det = h("details", { class: "think" }, h("summary", {}, closed ? `Thought for ${fmtInt(tokens)} tokens` : `Thought for ${fmtInt(tokens)} tokens, then hit the length limit`), body);
    det.addEventListener("toggle", () => { if (det.open && !body.textContent) { if (think) body.innerHTML = texText(think); else body.textContent = "(no reasoning text)"; } });
    card.append(det);
  }
  const ans = h("div", { class: "md answer" });
  ans.innerHTML = answer ? markdown(answer) : `<p><i>${escHtml(note)}</i></p>`;
  card.append(ans);
  el.append(card);
}

/* a harmful test prompt's reply in one refusal run: its verdict over 9 samples and, for the mild prompts
   (D.examples.safe), one sample's reasoning and answer; the other prompts show their verdict only */
async function showRefusalReply(el, { run, cand, i, v, who, seq, cur }) {
  const EX = D.examples, n9 = v == null ? null : Math.round(9 * v), refused = n9 != null && v >= 0.5;
  const base = { prompt: `“${EX.all[i]}”`, promptLabel: `Prompt ${i + 1}`, who,
    verdict: n9 == null ? "" : `${refused ? "refuses" : "complies"} · ${n9} of 9 samples refused`,
    col: n9 == null ? "var(--ink-3)" : refused ? "var(--refuse)" : "var(--flow-ink)" };
  if (!EX.safe.includes(i)) {
    replyView(el, { ...base, note: "Reply not shown. The page only shows replies to the mild requests (the dots with a dark ring); for the rest it shows the verdict." });
    return;
  }
  replyView(el, { ...base, note: "Loading the reply…" });
  let x;
  try { x = ((await loadSamples(`refusal-${run}.json.gz`)).cands[cand] || {})[String(i)]; }
  catch (e) { if (seq === cur()) replyView(el, { ...base, note: `Could not load the replies (${e.message}).` }); return; }
  if (seq !== cur()) return; /* another dot was picked meanwhile */
  if (!x) { replyView(el, { ...base, note: "No sample for this prompt in this run." }); return; }
  const [, closed, tokens, think, answer] = x;
  replyView(el, { ...base, think, tokens, closed: !!closed, answer, note: "No answer: the reasoning hit the length limit, so the judge read the reasoning itself." });
}

/* kick outcomes: a key with counts under the dots, and one response on a click */
function kickKey(el, out) {
  clearEl(el);
  if (!out) return;
  const names = D.kicks.outcomes;
  names.forEach((name, k) => {
    const n = out.filter((c) => c === k).length;
    if (!n && k > 1) return; /* always list solved and wrong answer; the rest when they occur */
    const st = KICK_STYLE[k], sw = s("svg", { width: 12, height: 12, viewBox: "0 0 12 12", "aria-hidden": "true" });
    s("circle", { cx: 6, cy: 6, r: 4.5, fill: "var(--kick)", "fill-opacity": st.f, stroke: "var(--kick)", "stroke-width": 1.3, "stroke-dasharray": st.dash || null }, sw);
    el.append(h("span", {}, sw, `${name} · ${n}`));
  });
}
async function showKickReply(el, { run, cand, i, code, who, seq, cur }) {
  const names = D.kicks.outcomes, verdict = code == null ? "" : names[code], col = code === 0 ? "var(--ink)" : "var(--ink-2)";
  let P, x;
  replyView(el, { who, verdict, col, note: "Loading the response…" });
  try { [P, x] = await Promise.all([loadSamples("math500.json.gz"), loadSamples(`kick-${run}/${cand}.json.gz`)]); }
  catch (e) { if (seq === cur()) replyView(el, { who, verdict, col, note: `Could not load the responses (${e.message}).` }); return; }
  if (seq !== cur()) return;
  const ans = String(P.answers[i]), ansTxt = /^[A-Za-z][A-Za-z .,'-]*$/.test(ans) ? ans : `$${ans}$`; /* a word answer stays upright text */
  const row = x.rows[i], base = { prompt: `${P.problems[i]} (correct answer: ${ansTxt})`, promptLabel: `Problem ${i + 1}`, who, verdict, col };
  if (!row) { replyView(el, { ...base, note: "No response for this problem in this run." }); return; }
  const [, closed, tokens, think, answer, pred] = row;
  replyView(el, { ...base, verdict: `${verdict}${closed && pred != null && code !== 0 ? ` · said ${pred}` : ""}`, think, tokens, closed: !!closed, answer,
    note: closed ? "The reasoning closed but no answer followed." : "No answer: the response was cut off at the length limit while still in its reasoning (open it above)." });
}

/* ------------------------------------------------------------------ the track: a two-lane ribbon */
/* A looped model is drawn as a ribbon that curls once per loop (Dory also gets its entry and exit stacks as straights);
   Qwen is a straight. Lane = where the answer is heading: the plain lane has no behaviour (no refusal, no trait, on
   course), the checkered lane has it. Colour = what the stream carries: red refusal, a trait's colour or black for a steering
   vector, grey nothing. Refusal (and a prompted trait) builds up on its own, so the dot drifts from the plain lane
   into the checkered one; steering moves it at once and the loops that follow let it drift back. Only the lane and
   colour at the end are data; the path between is a sketch (or, with P.curve, a measured signal). */
const RGB = { grey: [154, 163, 178], red: [220, 38, 38], violet: [122, 76, 217], flow: [37, 99, 235], black: [22, 24, 29] }; /* flow: a stream with nothing along the direction */
/* every persona gets its own colour, none of them close to the blue of an unsteered stream */
const TRAIT_RGB = { optimistic: [229, 164, 0], humorous: [240, 124, 26], sycophantic: [224, 71, 158], hallucinating: [31, 158, 90],
  apathetic: [138, 141, 31], evil: [122, 18, 52], impolite: [139, 90, 43] };
const rgbCss = (c) => `rgb(${c.map((v) => Math.round(v)).join(",")})`;
const traitRGB = (t) => TRAIT_RGB[t] || RGB.violet;
const mixRGB = (a, b, t) => a.map((v, i) => v + (b[i] - v) * clamp(t, 0, 1));
const rgbKey = (c) => `rgb(${c.map((v) => Math.round(v / 6) * 6).join(",")})`;
const smooth = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const clearEl = (e) => { while (e.firstChild) e.removeChild(e.firstChild); };
const GEO = {
  /* pre / post: Ouro and Nanbeige go from the embedding straight into loop 1, and from the last loop straight into the LM head */
  ouro: { loop: 24, pre: 0.6, post: 0.6, ends: ["embed", "head"], site: { push: 16, kick: 12, persona: 12 } },
  nanbeige: { loop: 22, pre: 0.6, post: 0.6, ends: ["embed", "head"], site: { push: 15, kick: 11, persona: 11 } },
  dory: { loop: 40, pre: 16, post: 16, stacks: true, ends: ["entry stack · 16", "exit stack · 16"], site: { push: 26, kick: 36, persona: 36 } },
  /* Inkling: a prelude of 6 layers, a core of 12 run 3 times (passes), a coda of 6; its picked refusal site is layer 11
     (core layer 6 of 12), its kicks go in at layer 12 */
  inkling: { loop: 12, pre: 6, post: 6, stacks: true, ends: ["prelude · 6", "coda · 6"], names: ["prelude", "coda", "core, 3 passes"], site: { push: 5, kick: 6, persona: 6 } },
  qwen: { straight: true, total: 36, site: { push: 3, kick: 4, persona: 4 } },
};
GEO.dory4 = GEO.dory;
/* natural: the behaviour builds up without steering (the dot starts plain and drifts into the checkered lane) */
const MEAN = {
  refusal: { natural: true, sig: RGB.red, sigCss: "var(--refuse)", sigText: "refusal signal", lanes: ["No-refusal lane", "Refusal lane"], ends: ["complies", "refuses"] },
  /* violet: a persona averaged over traits (the extra-loops figure); one trait gets its own colour via traitMean */
  persona: { natural: false, sig: RGB.violet, sigCss: "var(--steer)", kind: "persona", lanes: ["Normal lane", "Trait lane"], ends: ["normal", "trait shows"] },
  perase: { natural: true, sig: RGB.violet, sigCss: "var(--steer)", kind: "perase", sigText: "trait set by the system prompt", lanes: ["No-trait lane", "Trait lane"], ends: ["no trait", "trait shows"] },
  kick: { natural: false, sig: RGB.black, lanes: ["On-course lane", "Derailed lane"], ends: ["solves", "derails"] },
};
/* a persona's meaning carries its trait's colour (kind: "persona" adds it, "perase" is set by the prompt and erased) */
const traitMean = (kind, t) => Object.assign({}, MEAN[kind], { sig: traitRGB(t), sigCss: rgbCss(traitRGB(t)), trait: t, kind });
const SPEED = 0.45; /* dot speed along the ribbon, viewBox px per ms, the same for every model (o.speed scales it) */
const TAG = { erase: "ERASED", push: "PUSH", persona: "PERSONA", kick: "KICK" };
const FOFF = 0.3; /* the first filter of an erased loop sits this far (in layers) into the loop, and the colour changes there */
const K_D = 3.6; /* loop height: the ribbon follows x = r t + d sin t, y = -d (1 - cos t), d = K_D r */
const LOOP_LEN = (() => { let L = 0, px = 0, py = 0; const N = 4000;
  for (let i = 0; i <= N; i++) { const t = (2 * Math.PI * i) / N, x = t + K_D * Math.sin(t), y = -K_D * (1 - Math.cos(t)); if (i) L += Math.hypot(x - px, y - py); px = x; py = y; }
  return L; })();
let TRACK_N = 0;

class Track {
  /* o: {model, loops, maxLaps, trained, mini, meaning, types, runnerColor, label, reinject, speed,
         arch: {ticks: "pre" | "sandwich" | "renorm", gates: true, reinject}} (arch: the model cards) */
  constructor(el, o) {
    this.el = el;
    this.o = Object.assign({ model: "ouro", loops: null, maxLaps: null, trained: null, mini: false, meaning: MEAN.refusal, types: [], runnerColor: null, label: "", reinject: false, arch: null, speed: 1 }, o);
    this.G = GEO[this.o.model]; this.straight = !!this.G.straight;
    this.id = `tr${++TRACK_N}`; this.token = 0;
    if (el._track) el._track.token++; /* stop an older track's animation and its onDone in this box */
    el._track = this;
    this.build(this.o.loops ?? MODELS[this.o.model].loops, this.o.trained);
  }
  build(n, trained) {
    const o = this.o, G = this.G, st = this.straight, mini = o.mini;
    ++this.token;
    this.n = st ? 4 : n; this.trained = st ? 4 : (trained ?? n);
    this.solo = true; /* lanes are gone; this flat drawing is only the fallback when WebGL is missing */
    clearEl(this.el);
    /* narrower boxes get a narrower drawing, so text and lanes stay legible on a phone */
    const cw = this.el.clientWidth || 1000;
    const W = mini ? 440 : cw < 540 ? 560 : cw < 780 ? 720 : 900, mL = mini ? 14 : 18, mR = mini ? 14 : (this.solo ? 22 : (W < 700 ? 86 : 104));
    const A = o.arch || {}, feed = o.reinject || A.reinject;
    /* room under the track for architecture labels (cards) and for the re-injection pipe */
    const under = (mini ? (feed ? 30 : A.ticks || A.gates ? 18 : 8) : 34 + (feed ? 18 : 0));
    const raw = []; let H, yb, sPx, w;
    if (st) {
      w = mini ? 4.2 : 10;
      sPx = (W - mL - mR) / G.total; yb = (mini ? 8 : 36) + w; H = yb + w + (mini ? under : 40);
      for (let i = 0; i <= 200; i++) raw.push([mL + ((W - mL - mR) * i) / 200, yb]);
      this.geo = { yb, sPx, x0: mL, xe: W - mR, end: W - mR };
    } else {
      const preR = (G.pre * LOOP_LEN) / G.loop, postR = (G.post * LOOP_LEN) / G.loop;
      const spanR = (k) => preR + postR + k * 2 * Math.PI;
      const r = Math.min(mini ? 14 : 21, (W - mL - mR) / spanR(Math.max(this.n, o.maxLaps || 0))), d = K_D * r;
      w = mini ? 4.2 : clamp(0.52 * r, 6, 10);
      sPx = (LOOP_LEN * r) / G.loop;
      const pre = G.pre * sPx, post = G.post * sPx, x0 = mL + (W - mL - mR - spanR(this.n) * r) / 2;
      yb = (mini ? 8 : 32) + w + 2 * d; H = yb + w + under;
      raw.push([x0, yb]);
      for (let i = 1; i <= 30; i++) raw.push([x0 + (pre * i) / 30, yb]);
      for (let k = 0; k < this.n; k++) {
        const X = x0 + pre + k * 2 * Math.PI * r;
        for (let i = 1; i <= 600; i++) { const t = (2 * Math.PI * i) / 600; raw.push([X + r * t + d * Math.sin(t), yb - d * (1 - Math.cos(t))]); }
      }
      const xe = x0 + pre + this.n * 2 * Math.PI * r;
      for (let i = 1; i <= 30; i++) raw.push([xe + (post * i) / 30, yb]);
      this.geo = { yb, sPx, x0, r, d, pre, post, xe, end: xe + post };
    }
    this.w = w;
    /* resample at equal arc length; p is the position in layers */
    const cum = [0];
    for (let i = 1; i < raw.length; i++) cum.push(cum[i - 1] + Math.hypot(raw[i][0] - raw[i - 1][0], raw[i][1] - raw[i - 1][1]));
    const Ltot = cum[cum.length - 1], ds = mini ? 2 : 1.5, N = Math.ceil(Ltot / ds) + 1, S = [];
    let j = 0;
    for (let i = 0; i < N; i++) {
      const L = Math.min(Ltot, i * ds);
      while (j < raw.length - 2 && cum[j + 1] < L) j++;
      const f = clamp((L - cum[j]) / Math.max(1e-9, cum[j + 1] - cum[j]), 0, 1);
      S.push({ x: lerp(raw[j][0], raw[j + 1][0], f), y: lerp(raw[j][1], raw[j + 1][1], f), p: L / sPx });
    }
    for (let i = 0; i < N; i++) {
      const a = S[Math.max(0, i - 1)], b = S[Math.min(N - 1, i + 1)], tx = b.x - a.x, ty = b.y - a.y, l = Math.hypot(tx, ty) || 1;
      S[i].nx = ty / l; S[i].ny = -tx / l; /* left of travel: the inside of each loop, the upper lane on straights */
    }
    this.T = Ltot / sPx; this.Lpx = Ltot;
    const nn = this.n;
    this.unitOf = (p) => (st ? Math.min(4, Math.floor(p / 9) + 1) : p < G.pre ? 0 : p >= G.pre + nn * G.loop ? nn + 1 : Math.min(nn, Math.floor((p - G.pre) / G.loop) + 1));
    for (const q of S) {
      q.u = this.unitOf(q.p);
      q.c = st ? q.u - 1 : q.u === 0 ? 0 : q.u === nn + 1 ? 4 * nn + 1 : 1 + 4 * (q.u - 1) + Math.min(3, Math.floor(((q.p - G.pre) / G.loop - (q.u - 1)) * 4));
    }
    this.S = S;
    this.svg = s("svg", { viewBox: `0 0 ${W} ${H.toFixed(0)}`, role: "img", "aria-label": o.label || "Track" }, this.el);
    const defs = s("defs", {}, this.svg);
    const pat = s("pattern", { id: `${this.id}alt`, patternUnits: "userSpaceOnUse", width: 6, height: 6 }, defs);
    s("rect", { width: 6, height: 6, fill: "var(--alt-b)" }, pat);
    s("rect", { width: 3, height: 3, fill: "var(--alt-a)" }, pat);
    s("rect", { x: 3, y: 3, width: 3, height: 3, fill: "var(--alt-a)" }, pat);
    const flag = s("pattern", { id: `${this.id}flag`, patternUnits: "userSpaceOnUse", width: 6, height: 6 }, defs);
    s("rect", { width: 6, height: 6, fill: "#fff" }, flag);
    s("rect", { width: 3, height: 3, fill: "var(--ink)" }, flag);
    s("rect", { x: 3, y: 3, width: 3, height: 3, fill: "var(--ink)" }, flag);
    /* everything sits in one group so morph() can slide the drawing while loops are added */
    this.body = s("g", {}, this.svg);
    /* ribbon, chunk by chunk in path order so the later stroke of each loop passes over the earlier one */
    this.chunks = {};
    let a = 0;
    while (a < N) {
      let b = a;
      while (b + 1 < N && S[b + 1].c === S[a].c) b++;
      this.drawChunk(S[a].c, a, Math.min(N - 1, b + 2), b + 1);
      a = b + 1;
    }
    this.lab = s("g", {}, this.body);
    if (!mini) this.drawLabels();
    else this.drawArch();
    this.fd = feed ? s("g", {}, this.body) : null;
    this.mk = s("g", {}, this.body);
    this.rg = s("g", {}, this.body);
    this.halo = s("circle", { r: 0, fill: "var(--refuse)", "fill-opacity": 0.16, stroke: "var(--refuse)", "stroke-opacity": 0.5, "stroke-dasharray": "3 3" }, this.rg);
    this.ring = s("circle", { r: mini ? 4.8 : 8.6, fill: "#fff", stroke: "rgba(19,26,39,.38)", "stroke-width": 1 }, this.rg);
    this.core = s("circle", { r: mini ? 3.4 : 6.3 }, this.rg);
    this.leg = null;
    if (!mini) { this.leg = h("div", { class: "legend track-legend" }); this.el.append(this.leg); }
    this.apply({});
    this.place(0);
  }
  pts(a, b, off) {
    let out = "";
    for (let i = a; i <= b; i++) { const q = this.S[i]; out += `${(q.x + q.nx * off).toFixed(1)},${(q.y + q.ny * off).toFixed(1)} `; }
    return out;
  }
  drawChunk(c, a, b, next) {
    const g = s("g", {}, this.body), w = this.w, unit = this.S[a].u;
    const extra = !this.straight && unit > this.trained && unit <= this.n;
    const line = (off, attrs) => s("polyline", Object.assign({ points: this.pts(a, b, off), fill: "none" }, attrs), g);
    const A = this.o.arch || {};
    if (this.solo) {
      line(0, { stroke: "var(--panel)", "stroke-width": 2 * w + 4 });
      line(0, { stroke: "var(--own-lane)", "stroke-width": 2 * w });
      for (const off of [w, -w]) line(off, { stroke: "var(--lane-edge)", "stroke-width": this.o.mini ? 0.8 : 1, "stroke-dasharray": extra ? "4 3" : null });
      if (A.ticks) this.drawTicks(g, this.S[a].p, next < this.S.length ? this.S[next].p : Infinity, A.ticks);
    } else {
      line(0, { stroke: "var(--panel)", "stroke-width": 2 * w + 5, "stroke-linejoin": "round" });
      line(w / 2, { stroke: "var(--own-lane)", "stroke-width": w + 0.4 });
      line(-w / 2, { stroke: `url(#${this.id}alt)`, "stroke-width": w + 0.4 });
      for (const off of [w, -w]) line(off, { stroke: "var(--lane-edge)", "stroke-width": 1, "stroke-dasharray": extra ? "4 3" : null });
      line(0, { stroke: "var(--lane-edge)", "stroke-width": 0.7, "stroke-dasharray": "2 3" });
    }
    this.chunks[c] = { g, zone: s("g", {}, g), trail: s("g", {}, g), a, b, unit };
  }
  /* norms, one mark per layer (sublayer for Dory) at its centre, for positions p0 <= p < p1:
     pre      = each layer reads a normalised copy of the stream (a light tick inside the ribbon)
     sandwich = it also normalises what it writes (a double tick: Ouro)
     renorm   = the stream itself is put back to unit length after every sublayer (a dark tick past the edges: Dory) */
  drawTicks(g, p0, p1, kind) {
    const G = this.G, w = this.w, T = this.T;
    const inLayers = (p) => (this.straight ? p < G.total : G.stacks ? p < T : p >= G.pre && p < G.pre + this.n * G.loop);
    for (let p = Math.floor(p0) + 0.5, end = Math.min(p1, T); p < end; p++) {
      if (p < p0 || !inLayers(p)) continue;
      const q = this.S[this.idxAt(p)], tx = -q.ny, ty = q.nx;
      const seg = (half, dt, attrs) => s("line", Object.assign({ x1: q.x + q.nx * half + tx * dt, y1: q.y + q.ny * half + ty * dt,
        x2: q.x - q.nx * half + tx * dt, y2: q.y - q.ny * half + ty * dt }, attrs), g);
      if (kind === "pre") seg(w * 0.7, 0, { stroke: "var(--ink-3)", "stroke-width": 0.7, "stroke-opacity": 0.8 });
      else if (kind === "sandwich") for (const dt of [-0.85, 0.85]) seg(w, dt, { stroke: "var(--ink-2)", "stroke-width": 0.6 });
      else seg(w + 1.6, 0, { stroke: "var(--ink)", "stroke-width": 0.75 });
    }
  }
  unitX(k) {
    const g = this.geo;
    return this.straight ? g.x0 + (9 * (k - 1) + 4.5) * g.sPx : g.x0 + g.pre + (k - 1) * 2 * Math.PI * g.r + Math.PI * g.r;
  }
  drawLabels() {
    const g = this.geo, G = this.G, w = this.w, L = this.lab;
    this.lapEls = {};
    if (this.straight) {
      for (let k = 1; k <= 3; k++) { const x = g.x0 + 9 * k * g.sPx; s("line", { x1: x, x2: x, y1: g.yb - w - 1, y2: g.yb + w + 1, stroke: "var(--panel)", "stroke-width": 3 }, L); }
      for (let k = 1; k <= 4; k++) {
        this.lapEls[k] = s("text", { x: this.unitX(k), y: g.yb + w + 19, "text-anchor": "middle", class: "trk-loop", text: `Q${k}` }, L);
        s("text", { x: this.unitX(k), y: g.yb + w + 32, "text-anchor": "middle", class: "trk-small", text: `layers ${QLAYERS[k - 1]}` }, L);
      }
    } else {
      this.loopLab = {};
      for (let k = 1; k <= this.n; k++) {
        const x = this.unitX(k), y = g.yb - 4.06 * g.r, gk = (this.loopLab[k] = s("g", {}, L));
        this.lapEls[k] = s("text", { x, y: y + 5, "text-anchor": "middle", class: "trk-loop", text: g.r < 15 ? `L${k}` : `LOOP ${k}` }, gk);
        if (k > this.trained) s("text", { x, y: y + 18, "text-anchor": "middle", class: "trk-small trk-extra", text: g.r < 15 ? "extra" : "extra loop" }, gk);
      }
      if (G.stacks) s("text", { x: g.x0 + g.pre / 2, y: g.yb + w + 17, "text-anchor": "middle", class: "trk-small", text: G.ends[0] }, L);
    }
    /* the end of the drawing (exit stack label, flag, outcome labels) slides as one piece when loops are added */
    const T = (this.tailLab = s("g", {}, L));
    if (!this.straight && G.stacks) s("text", { x: g.xe + g.post / 2, y: g.yb + w + 17, "text-anchor": "middle", class: "trk-small", text: G.ends[1] }, T);
    s("line", { x1: g.x0, x2: g.x0, y1: g.yb - w - 3, y2: g.yb + w + 3, stroke: "var(--ink-3)", "stroke-width": 1.5 }, L);
    s("rect", { x: g.end - 3, y: g.yb - w - 2, width: 6, height: 2 * w + 4, fill: `url(#${this.id}flag)`, stroke: "var(--ink)", "stroke-width": 0.8 }, T);
    if (!this.solo) {
      const m = this.o.meaning;
      const dy = Math.max(w / 2, 7);
      s("text", { x: g.end + 10, y: g.yb - dy + 3.6, class: "trk-end", text: `▸ ${m.ends[0]}` }, T);
      s("text", { x: g.end + 10, y: g.yb + dy + 3.6, class: "trk-end alt", text: `▸ ${m.ends[1]}` }, T);
    }
  }
  /* the cards: entry and exit stacks, norm gates between loops, and the norm ticks drawn with the ribbon */
  drawArch() {
    const A = this.o.arch;
    if (!A) return;
    const g = this.geo, G = this.G, w = this.w, L = this.lab;
    if (!this.straight && G.stacks) {
      s("text", { x: g.x0 + g.pre / 2, y: g.yb + w + 13, "text-anchor": "middle", class: "trk-small", text: (G.names || ["entry stack"])[0] }, L);
      s("text", { x: g.xe + g.post / 2, y: g.yb + w + 13, "text-anchor": "middle", class: "trk-small", text: (G.names || [0, "exit stack"])[1] }, L);
    }
    if (A.gates) for (let k = 1; k < this.n; k++) {
      const x = g.x0 + g.pre + k * 2 * Math.PI * g.r;
      s("rect", { x: x - 2.4, y: g.yb - w - 3.5, width: 4.8, height: 2 * w + 7, rx: 1.4, fill: "var(--ink)" }, L);
      if (!A.reinject) s("text", { x, y: g.yb + w + 14, "text-anchor": "middle", class: "trk-small", text: "norm" }, L);
    }
    if (this.straight) s("text", { x: (g.x0 + g.xe) / 2, y: g.yb + w + 14, "text-anchor": "middle", class: "trk-small", text: "36 layers, one pass" }, L);
  }
  /* add loops in place: the drawing of a run with fromN loops grows into this one (n loops), or shrinks back with
     reverse. The new loops are revealed left to right while the exit stack slides out ahead of them. */
  morph(fromN, { reverse = false, dur = 1300, onDone } = {}) {
    const tok = ++this.token, g = this.geo, dX = (this.n - fromN) * 2 * Math.PI * g.r, xs = g.x0 + g.pre + fromN * 2 * Math.PI * g.r;
    const defs = this.svg.querySelector("defs"), cp = s("clipPath", { id: `${this.id}grow` }, defs);
    const rect = s("rect", { x: xs - 1, y: -50, width: 0, height: 2000 }, cp);
    const extra = [], tail = [this.tailLab];
    for (const c of Object.values(this.chunks)) {
      if (c.unit > fromN && c.unit <= this.n) { c.g.setAttribute("clip-path", `url(#${this.id}grow)`); extra.push(c.g); }
      else if (c.unit === this.n + 1) tail.push(c.g);
    }
    const labs = Object.entries(this.loopLab || {}).filter(([k]) => +k > fromN).map(([, el]) => el);
    const set = (f) => {
      rect.setAttribute("width", (f * dX + 1).toFixed(1));
      this.body.setAttribute("transform", `translate(${((1 - f) * dX / 2).toFixed(1)},0)`);
      for (const el of tail) el.setAttribute("transform", `translate(${(-(1 - f) * dX).toFixed(1)},0)`);
      for (const el of labs) el.setAttribute("opacity", clamp((f - 0.45) / 0.55, 0, 1).toFixed(2));
    };
    const fin = () => {
      for (const c of extra) c.removeAttribute("clip-path");
      for (const el of tail) el.removeAttribute("transform");
      this.body.removeAttribute("transform");
      cp.remove();
      onDone && onDone();
    };
    this.place(0);
    if (RM) { set(reverse ? 0 : 1); if (!reverse) fin(); else onDone && onDone(); return; }
    const t0 = performance.now();
    set(reverse ? 1 : 0);
    const step = (now) => {
      if (tok !== this.token) return;
      const p = clamp((now - t0) / dur, 0, 1), f = easeInOut(reverse ? 1 - p : p);
      set(f);
      if (p < 1) requestAnimationFrame(step);
      else if (reverse) onDone && onDone();
      else fin();
    };
    requestAnimationFrame(step);
  }
  /* Dory's re-injection: a pipe from the end of the entry stack into the start of every later loop.
     P.feed = {loops: [k...], style: "usual" | "edit" | "cut", tag} marks the loops whose copy is changed */
  drawFeed() {
    const F = this.fd;
    if (!F || this.straight) return;
    clearEl(F);
    const g = this.geo, w = this.w, mini = this.o.mini, fe = (this.P && this.P.feed) || {};
    const y0 = g.yb + w + 1.5, yp = g.yb + w + (mini ? 9 : 12), X = (k) => g.x0 + g.pre + k * 2 * Math.PI * g.r;
    const sw = mini ? 1.5 : 1.8, changed = new Set(fe.style && fe.style !== "usual" ? fe.loops || [] : []), tagX = [];
    s("path", { d: `M${X(0).toFixed(1)},${y0.toFixed(1)}V${yp.toFixed(1)}H${X(this.n - 1).toFixed(1)}`, fill: "none", stroke: "var(--ink-2)", "stroke-width": sw, "stroke-linejoin": "round" }, F);
    for (let k = 1; k < this.n; k++) {
      const x = X(k), ch = changed.has(k + 1);
      if (ch) tagX.push(x);
      if (ch && fe.style === "cut") {
        const ym = (yp + y0) / 2 + 0.5;
        s("line", { x1: x, x2: x, y1: yp, y2: y0 + 2, stroke: "var(--ink-3)", "stroke-width": sw, "stroke-dasharray": "2 2" }, F);
        s("path", { d: `M${x - 3},${ym - 3}L${x + 3},${ym + 3}M${x + 3},${ym - 3}L${x - 3},${ym + 3}`, stroke: "var(--steer)", "stroke-width": 1.7, fill: "none" }, F);
      } else {
        const col = ch ? "var(--steer)" : "var(--ink-2)";
        s("line", { x1: x, x2: x, y1: yp, y2: y0 + 4.5, stroke: col, "stroke-width": sw }, F);
        s("polygon", { points: `${x - 3.2},${y0 + 4.5} ${x + 3.2},${y0 + 4.5} ${x},${y0}`, fill: col }, F);
      }
    }
    if (mini) s("text", { x: X(0) + 6, y: yp + 12, class: "trk-small", text: "input re-added each loop" }, F);
    if (tagX.length && fe.tag) s("text", { x: (tagX[0] + tagX[tagX.length - 1]) / 2, y: yp + 13, "text-anchor": "middle", class: "trk-small", fill: "var(--steer)", text: fe.tag }, F);
  }
  /* how far the model's own behaviour has built up at layer position p (a sketch; Dory follows its measured
     refusal signal: about a third of it leaves the entry stack and the rest appears within loop 1's first sublayers) */
  bld(p) {
    const G = this.G;
    if (this.straight) return smooth((p - 3) / 24);
    if (G.stacks) return p < G.pre ? 0.35 * smooth(p / G.pre) : 0.35 + 0.65 * smooth((p - G.pre) / (0.25 * G.loop));
    return smooth((p - G.pre) / (0.7 * G.loop));
  }
  idxAt(p) {
    let lo = 0, hi = this.S.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (this.S[mid].p < p) lo = mid + 1; else hi = mid; }
    return lo;
  }
  /* P: {events:[{k, type}], vExit, bite, vSingle:{k: v}, every, leak:{chain, ref}, edited} */
  compute(P) {
    const G = this.G, st = this.straight, S = this.S, N = S.length, T = this.T;
    const ustart = (k) => (st ? 9 * (k - 1) : G.pre + (k - 1) * G.loop), ulen = st ? 9 : G.loop;
    const evs = (P.events || []).filter((e) => e.k >= 1 && e.k <= this.n).map((e) => {
      const a = e.type === "erase" ? ustart(e.k) + FOFF : ustart(e.k) + G.site[e.type]; /* an erase starts at its first filter */
      return { k: e.k, type: e.type, a, b: e.type === "erase" ? ustart(e.k) + ulen : a };
    }).sort((x, y) => x.a - y.a);
    this.evs = evs;
    const vExit = clamp(P.vExit ?? 0, 0, 1), bite = clamp(P.bite ?? 1, 0, 1), lo = Math.max(vExit, 0.01);
    const jump = (i, ub) => {
      const e = evs[i], last = i === evs.length - 1;
      if (st) return last ? vExit : Math.max(ub, clamp((P.vSingle || {})[e.k] ?? 0, 0, 1));
      if (P.every) return last ? vExit : Math.max(ub, (vExit * (i + 1)) / evs.length);
      const cum = ub + bite * (1 - ub);
      if (!last) return cum;
      return T - e.b > 0.5 ? Math.max(cum, vExit) : vExit;
    };
    const sim = (lamGap) => {
      let u = 0; const J = [];
      for (let i = 0; i < evs.length; i++) {
        if (i > 0) u *= Math.exp(-lamGap * (evs[i].a - evs[i - 1].b));
        u = jump(i, u); J.push(u);
      }
      const last = evs[evs.length - 1], L = last ? T - last.b : 0;
      return { J, lam: !st && last && L > 0.5 && u > lo ? Math.log(u / lo) / L : 0 };
    };
    const r1 = sim(0), lamGap = st ? 0 : r1.lam, r2 = sim(lamGap);
    const mean = this.o.meaning || {}, nat = !!mean.natural, SIG = mean.sig || RGB.red, cv = P.curve;
    this.U = new Float32Array(N); this.COL = new Array(N); this.ER = new Uint8Array(N); this.HALO = new Float32Array(N); this.VT = new Float32Array(N);
    let ei = -1;
    for (let i = 0; i < N; i++) {
      const p = S[i].p;
      while (ei + 1 < evs.length && p >= evs[ei + 1].a) ei++;
      let u = 0, er = 0, type = null;
      if (ei >= 0) {
        const e = evs[ei], J = r2.J[ei];
        type = e.type;
        if (p <= e.b) { u = J; er = e.type === "erase" ? 1 : 0; }
        else u = J * Math.exp(-(ei === evs.length - 1 ? r2.lam : lamGap) * (p - e.b));
      }
      /* beh: 0 = plain lane, 1 = checkered lane. A natural behaviour builds up and steering removes its share u;
         otherwise steering adds it. P.curve gives the measured signal per layer position instead. */
      let beh;
      if (cv) { const j = Math.min(cv.length - 1, Math.floor(p)), k = Math.min(cv.length - 1, j + 1); beh = clamp(lerp(cv[j], cv[k], p - Math.floor(p)), 0, 1); }
      else beh = nat ? (P.peak ?? 1) * this.bld(p) * (1 - u) : u; /* peak: how strong the behaviour is unsteered (refusal: its rate) */
      let c = mixRGB(this.o.meaning ? RGB.flow : RGB.grey, SIG, beh);
      if (P.leak) {
        const ch = P.leak.chain, k = S[i].u;
        const lk = k === 0 ? ch[0] : k > this.n ? ch[this.n] : lerp(ch[k - 1], ch[k], smooth((p - G.pre) / G.loop - (k - 1)));
        c = mixRGB(RGB.flow, RGB.red, lk / P.leak.ref);
        this.HALO[i] = lk / P.leak.ref;
      }
      this.U[i] = beh; this.ER[i] = er; this.COL[i] = rgbKey(c);
    }
  }
  apply(P) {
    this.P = P || {};
    this.compute(this.P);
    for (const c of Object.values(this.chunks)) { clearEl(c.zone); clearEl(c.trail); }
    clearEl(this.mk);
    this.run = null; this.ti = 0;
    this.drawFeed();
    if (!this.o.mini) { this.drawMarkers(); this.legend(); }
  }
  tag(x, y, text, kind) {
    const wd = text.length * 6.9 + 12;
    const g = s("g", {}, this.mk);
    s("rect", { x: x - wd / 2, y: y - 11, width: wd, height: 16, rx: 4, fill: "var(--panel)", stroke: kind === "erase" ? "var(--ink-3)" : "var(--steer)",
      "stroke-width": 1.2, "stroke-dasharray": kind === "erase" ? "3 2" : null }, g);
    s("text", { x, y: y + 1, "text-anchor": "middle", class: "trk-tag", fill: kind === "erase" ? "var(--ink-2)" : "var(--steer)", text }, g);
  }
  drawMarkers() {
    const g = this.geo, w = this.w, P = this.P;
    const tagY = this.straight ? g.yb - w - 12 : g.yb - 2 * g.d - w - 10;
    const zoneOf = (c, white) => {
      if (white) s("polyline", { points: this.pts(c.a, c.b, 0), fill: "none", stroke: "var(--panel)", "stroke-width": 2 * w + 9, "stroke-opacity": 0.55 }, c.zone);
      for (const off of [w + 4.5, -w - 4.5]) s("polyline", { points: this.pts(c.a, c.b, off), fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, c.zone);
    };
    for (const e of this.evs) {
      if (e.type === "erase") {
        for (const c of Object.values(this.chunks)) if (c.unit === e.k) zoneOf(c, true);
      } else {
        /* the shove points to where it sends the dot: into the plain lane when it removes a behaviour that
           builds up on its own (refusal), into the checkered lane when it adds one (persona, kick) */
        const sg = this.o.meaning && this.o.meaning.natural ? -1 : 1;
        const q = this.S[this.idxAt(e.a)], nx = sg * q.nx, ny = sg * q.ny, tx = -ny, ty = nx;
        const P2 = (n, t) => `${(q.x + nx * n + tx * t).toFixed(1)},${(q.y + ny * n + ty * t).toFixed(1)}`;
        if (e.type === "kick") {
          s("polyline", { points: [P2(w + 10, 0), P2(w / 2, 4), P2(0, -4), P2(-w / 2, 4), P2(-w - 7, 0)].join(" "), fill: "none", stroke: "var(--steer)", "stroke-width": 2.4, "stroke-linejoin": "miter" }, this.mk);
        } else {
          s("line", { x1: q.x + nx * (w + 10), y1: q.y + ny * (w + 10), x2: q.x - nx * (w - 1), y2: q.y - ny * (w - 1), stroke: "var(--steer)", "stroke-width": 2.6 }, this.mk);
          s("polygon", { points: [P2(-w - 7, 0), P2(-w + 1, 4.5), P2(-w + 1, -4.5)].join(" "), fill: "var(--steer)" }, this.mk);
        }
      }
      this.tag(this.unitX(e.k), tagY, P.tagText || TAG[e.type], e.type);
    }
    if (P.edited) {
      for (const c of Object.values(this.chunks)) if (c.unit >= 1 && c.unit <= this.n) zoneOf(c, false);
      this.tag((this.unitX(1) + this.unitX(this.n)) / 2, tagY, "WEIGHTS EDITED · EVERY LOOP", "erase");
    }
  }
  legend() {
    if (!this.leg) return;
    const m = this.o.meaning || {}, P = this.P, items = [];
    const it = (cls, style, text) => h("span", {}, h("i", { class: cls, style }), text);
    if (!this.solo) { items.push(it("sw-own", null, m.lanes[0])); items.push(it("sw-alt", null, m.lanes[1])); }
    if (P.leak) items.push(it("sw-dot", "background:var(--refuse)", "refusal direction leaking back"), it("sw-halo", null, "more than the unedited model carries"));
    else if (m.natural) items.push(it("sw-dot", `background:${m.sigCss}`, m.sigText));
    else items.push(it("sw-dot", "background:var(--neutral)", "plain stream"));
    const types = new Set([...(this.o.types || []), ...this.evs.map((e) => e.type)]);
    if (types.has("push")) items.push(it("sw-dot", "background:var(--steer)", "push against refusal"));
    if (types.has("persona")) items.push(it("sw-dot", "background:var(--steer)", "persona vector"));
    if (types.has("kick")) items.push(it("sw-dot", "background:var(--steer)", "random kick"));
    if (types.has("erase")) items.push(it("sw-zone", null, m.kind ? "erased loop: trait wiped from the stream" : "erased loop: signal wiped from the stream"));
    if (this.fd) items.push(it("sw-pipe", null, "pipe: the entry stack's output, added back before every loop"));
    clearEl(this.leg);
    this.leg.append(...items);
  }
  offFor(i) { return this.solo ? 0 : this.w / 2 - this.U[i] * this.w; }
  pt(i) { const q = this.S[i], off = this.offFor(i); return `${(q.x + q.nx * off).toFixed(1)},${(q.y + q.ny * off).toFixed(1)}`; }
  trailTo(upTo) {
    for (let i = this.ti + 1; i <= upTo; i++) {
      const j = i - 1, ch = this.S[j].c, key = this.ER[j] ? "E" : this.COL[j];
      if (!this.run || this.run.key !== key || this.run.ch !== ch) {
        if (this.run) this.run.el.setAttribute("points", this.run.pts.join(" "));
        const attrs = key === "E" ? { fill: "none", stroke: "var(--ink-3)", "stroke-width": 2, "stroke-dasharray": "2 3", "stroke-linecap": "round" }
          : { fill: "none", stroke: key, "stroke-width": this.o.mini ? 2 : 3.6, "stroke-linecap": "round", "stroke-linejoin": "round" };
        const back = this.run && this.run.ch !== ch ? Math.max(0, j - 1) : j;
        this.run = { el: s("polyline", attrs, this.chunks[ch].trail), key, ch, pts: [this.pt(back)] };
        if (back !== j) this.run.pts.push(this.pt(j));
      }
      this.run.pts.push(this.pt(i));
    }
    if (this.run) this.run.el.setAttribute("points", this.run.pts.join(" "));
    this.ti = Math.max(this.ti, upTo);
  }
  place(i) {
    const q = this.S[i], off = this.offFor(i), x = (q.x + q.nx * off).toFixed(1), y = (q.y + q.ny * off).toFixed(1);
    for (const c of [this.halo, this.ring, this.core]) { c.setAttribute("cx", x); c.setAttribute("cy", y); }
    const er = this.ER[i];
    this.core.style.fill = this.o.runnerColor || (er ? "#fff" : this.COL[i]);
    this.core.style.stroke = er ? "var(--ink-3)" : "none";
    this.core.setAttribute("stroke-width", er ? 1.6 : 0);
    this.core.setAttribute("stroke-dasharray", er ? "2 2" : "none");
    const hv = this.HALO[i];
    this.halo.setAttribute("r", hv > 1 ? (8 + 7 * (hv - 1)).toFixed(1) : 0);
    if (this.lapEls) { const u = q.u; for (const k in this.lapEls) this.lapEls[k].classList.toggle("now", +k === u); }
  }
  /* play the plan; until = stop at this layer position */
  /* onProgress(f): called every frame with how far the flow has got, 0 to 1 (the hero tiles count along with it) */
  play(P, { onDone, until, onProgress } = {}) {
    const tok = ++this.token;
    this.apply(P);
    const N = this.S.length, iEnd = until == null ? N - 1 : this.idxAt(until);
    if (RM) { this.trailTo(iEnd); this.place(iEnd); onDone && onDone(); return; }
    const dur = Math.max(600, (this.Lpx * (until == null ? 1 : until / this.T)) / (SPEED * this.o.speed));
    const t0 = performance.now();
    const step = (now) => {
      if (tok !== this.token) return;
      const f = clamp((now - t0) / dur, 0, 1), i = Math.round(f * iEnd);
      this.trailTo(i); this.place(i); onProgress && onProgress(f);
      if (f < 1) requestAnimationFrame(step);
      else { if (this.lapEls) for (const k in this.lapEls) this.lapEls[k].classList.remove("now"); onDone && onDone(); }
    };
    requestAnimationFrame(step);
  }
  show(P, until) {
    ++this.token;
    this.apply(P);
    const iEnd = until == null ? this.S.length - 1 : this.idxAt(until);
    this.trailTo(iEnd); this.place(iEnd);
    if (this.lapEls) for (const k in this.lapEls) this.lapEls[k].classList.remove("now");
  }
  idle() {
    const ms = this.Lpx / SPEED;
    if (RM || !("IntersectionObserver" in window)) return;
    let vis = false, t0 = 0;
    const N = this.S.length;
    const step = (now) => { if (!vis) return; this.place(Math.floor((((now - t0) / ms) % 1) * (N - 1))); requestAnimationFrame(step); };
    new IntersectionObserver((es) => { vis = es[0].isIntersecting; if (vis) { t0 = performance.now(); requestAnimationFrame(step); } }).observe(this.svg);
  }
}

/* ------------------------------------------------------------------ the stream in 3D */
/* Every model is a glass tube and the residual stream on one prompt is the liquid inside it. A looped model's tube
   runs a loop-the-loop once per loop of the model (nudged sideways so it passes beside itself); Qwen's is straight.
   Colour is the only signal: how far the stream points along a behaviour's direction (red refusal, a persona trait's
   own colour, black how far a kick knocked it off course, blue none of these; grey on the model cards). Erasing a
   direction is a filter in the stream along the layers of a loop, from its first layer; adding a vector is a drop of
   dye at one layer. Dory's shortcut carries the entry stack's output to the start of every later loop. Only the colour
   at the end is data, and the head (the bulb) shows it (or, with P.curve, a measured signal along the way). One WebGL renderer draws every figure into its own 2D canvas, and only while it is on screen.
   Without WebGL the page falls back to the flat Track above. */
const T3 = window.THREE;
const D3 = { d: 1, r: 0.3, a: 0.42, rt: 0.13, liq: 0.082, byY: -0.66, byR: 0.062, ms: 40, qLen: 10.5, nf: 6, dropH: 0.85 };
const LOOP3 = (() => { let L = 0, p = null;
  for (let i = 0; i <= 900; i++) { const t = (2 * Math.PI * i) / 900, q = [D3.r * t + D3.d * Math.sin(t), D3.d * (1 - Math.cos(t)), D3.a * Math.sin(t)]; if (p) L += Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]); p = q; }
  return L; })();
const MSU = 400 / LOOP3; /* ms per unit of tube: the liquid moves at the same speed in every figure and model, one loop in 0.4 s */
const GL = (() => {
  if (!T3 || !T3.WebGLRenderer) return null;
  try {
    const probe = document.createElement("canvas");
    if (!(probe.getContext("webgl2") || probe.getContext("webgl"))) return null;
    const renderer = new T3.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1); renderer.setClearColor(0x000000, 0);
    return { renderer, views: new Set(), w: 1, h: 1, raf: 0 };
  } catch (e) { return null; }
})();
function glFrame(now) {
  GL.raf = 0;
  let again = false;
  for (const v of GL.views) {
    if (v.tick(now)) again = true;
    if (v.visible && v.dirty) { v.render(now); v.dirty = false; }
  }
  if (again) GL.raf = requestAnimationFrame(glFrame);
}
const glKick = () => { if (GL && !GL.raf) GL.raf = requestAnimationFrame(glFrame); };
const cssHex = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888888";
const PathCurve = T3 ? class extends T3.Curve {
  constructor(pts) { super(); this.pts = pts; }
  getPoint(t, out = new T3.Vector3()) {
    const P = this.pts, f = clamp(t, 0, 1) * (P.length - 1), i = Math.min(P.length - 2, Math.floor(f)), g = f - i, a = P[i], b = P[i + 1];
    return out.set(a.x + (b.x - a.x) * g, a.y + (b.y - a.y) * g, a.z + (b.z - a.z) * g);
  }
  getPointAt(u, out) { return this.getPoint(u, out); } /* the points are already spaced evenly */
  getTangentAt(u, out) { return this.getTangent(u, out); }
} : null;
const VS3 = `
attribute float aZone; attribute float aExtra; attribute float aHalo;
uniform float uShift;
varying vec3 vN; varying vec3 vW; varying vec3 vC; varying float vS; varying float vZone; varying float vX; varying float vExtra; varying float vHalo;
void main() {
  vec3 p = position;
  if (aZone > 1.5) p.x += uShift;
  vX = p.x; vZone = aZone; vS = uv.x; vExtra = aExtra; vHalo = aHalo;
#ifdef USE_COLOR
  vC = color;
#else
  vC = vec3(1.0);
#endif
  vec4 w = modelMatrix * vec4(p, 1.0);
  vW = w.xyz; vN = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const FS_LIQ = `
uniform float uFront; uniform float uTime; uniform float uReveal; uniform float uBands; uniform vec3 uLight;
varying vec3 vN; varying vec3 vW; varying vec3 vC; varying float vS; varying float vZone; varying float vX; varying float vExtra; varying float vHalo;
void main() {
  if (vS > uFront) discard;
  if (vZone > 0.5 && vZone < 1.5 && vX > uReveal) discard;
  vec3 N = normalize(vN), V = normalize(cameraPosition - vW);
  if (dot(N, V) < 0.0) N = -N;
  float lam = 0.74 + 0.36 * max(dot(N, uLight), 0.0);
  float band = 0.5 + 0.5 * sin(6.28318 * (vS * uBands - uTime));
  vec3 c = vC * lam * (0.94 + 0.1 * band);
  c += vec3(0.3) * pow(max(dot(N, normalize(uLight + V)), 0.0), 36.0);
  gl_FragColor = vec4(c, 1.0);
}`;
const FS_GLASS = `
uniform vec3 uEdge; uniform vec3 uExtraCol; uniform vec3 uHaloCol; uniform float uReveal; uniform vec3 uLight; uniform float uFront;
varying vec3 vN; varying vec3 vW; varying vec3 vC; varying float vS; varying float vZone; varying float vX; varying float vExtra; varying float vHalo;
void main() {
  if (vZone > 0.5 && vZone < 1.5 && vX > uReveal) discard;
  vec3 N = normalize(vN), V = normalize(cameraPosition - vW);
  float f = 1.0 - abs(dot(N, V));
  vec3 c = mix(uEdge, uExtraCol, vExtra);
  float a = 0.035 + (0.44 + 0.3 * vExtra) * pow(f, 2.6);
  float sp = pow(abs(dot(N, normalize(uLight + V))), 90.0);
  c = mix(c, vec3(1.0), sp * 0.9); a = max(a, sp * 0.7);
  float hl = clamp(vHalo, 0.0, 1.0) * step(vS, uFront);
  c = mix(c, uHaloCol, hl); a = max(a, hl * (0.3 + 0.55 * f));
  gl_FragColor = vec4(c, a);
}`;
let FILTER_TEX = null;
function filterTex() {
  if (FILTER_TEX) return FILTER_TEX;
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const x = c.getContext("2d");
  x.save(); x.beginPath(); x.arc(64, 64, 63, 0, 2 * Math.PI); x.clip();
  x.fillStyle = "rgba(255,255,255,0.3)"; x.fillRect(0, 0, 128, 128);
  x.strokeStyle = "rgba(255,255,255,1)"; x.lineWidth = 4.5;
  for (let i = 10; i < 128; i += 22) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i, 128); x.stroke(); x.beginPath(); x.moveTo(0, i); x.lineTo(128, i); x.stroke(); }
  x.restore();
  FILTER_TEX = new T3.CanvasTexture(c);
  return FILTER_TEX;
}
const UP = T3 ? new T3.Vector3(0, 1, 0) : null, ZA = T3 ? new T3.Vector3(0, 0, 1) : null;

class Stream3D {
  /* the same options and calls as Track: build, apply, play, show, morph, idle */
  constructor(el, o) {
    this.el = el;
    this.o = Object.assign({ model: "ouro", loops: null, maxLaps: null, trained: null, mini: false, meaning: MEAN.refusal, types: [], label: "", reinject: false, arch: null, speed: 1 }, o);
    this.G = GEO[this.o.model]; this.straight = !!this.G.straight;
    this.feed = !this.straight && !!this.G.stacks; /* Dory and Inkling: every drawing gets the shortcut from the stack that runs first */
    this.token = 0; this.runs = []; this.visible = false; this.dirty = true;
    if (el._track) { el._track.token++; if (el._track.dispose) el._track.dispose(); }
    el._track = this;
    clearEl(el);
    const mini = this.o.mini;
    this.wrap = h("div", { class: `s3${mini ? " mini" : ""}`, role: "img", "aria-label": this.o.label || "Model" });
    this.canvas = h("canvas");
    this.labs = h("div", { class: "s3-labs", "aria-hidden": "true", style: "visibility:hidden" }); /* shown once placed by the first render */
    this.wrap.append(this.canvas, this.labs);
    if (!mini) { this.hint = h("div", { class: "s3-hint", "aria-hidden": "true" }, "drag to turn"); this.wrap.append(this.hint); }
    el.append(this.wrap);
    this.leg = mini ? null : h("div", { class: "legend track-legend" });
    if (this.leg) el.append(this.leg);
    this.ctx = this.canvas.getContext("2d");
    this.scene = new T3.Scene();
    this.camera = new T3.PerspectiveCamera(26, 2, 0.1, 300);
    this.scene.add(new T3.HemisphereLight(0xffffff, 0xc3c9d4, 0.9));
    const dl = new T3.DirectionalLight(0xffffff, 0.55); dl.position.set(-3, 6, 5); this.scene.add(dl);
    this.light = new T3.Vector3(-0.35, 0.8, 0.5).normalize();
    this.col = { edge: new T3.Color(cssHex("--ink-3")), ink: new T3.Color(cssHex("--ink")), ink2: new T3.Color(cssHex("--ink-2")),
      steer: new T3.Color(cssHex("--steer")), refuse: new T3.Color(cssHex("--refuse")), grey: new T3.Color(cssHex("--neutral")), flow: new T3.Color(cssHex("--flow")) };
    this.view = { yaw0: -0.38, pitch0: 0.22, dYaw: 0, dPitch: 0, touched: false, t0: performance.now() };
    this.target = new T3.Vector3(); this.dist = 20;
    this.cw = Math.max(200, el.clientWidth || 640); this.narrow = this.cw < 560;
    this.bindInput();
    this.build(this.o.loops ?? MODELS[this.o.model].loops, this.o.trained);
    GL.views.add(this);
    this.observe();
  }
  observe() {
    const io = "IntersectionObserver" in window ? new IntersectionObserver((es) => {
      this.visible = es[es.length - 1].isIntersecting;
      if (this.visible) { this.dirty = true; glKick(); }
    }, { rootMargin: "80px" }) : null;
    if (io) io.observe(this.wrap); else this.visible = true;
    const ro = "ResizeObserver" in window ? new ResizeObserver(() => this.resize()) : null;
    if (ro) ro.observe(this.el);
    this.obs = [io, ro];
    this.resize(true);
  }
  /* the canvas is as tall as the model needs at this width (the frame is fixed per figure, so adding loops does not zoom) */
  resize(force) {
    const w = Math.max(200, this.el.clientWidth || 640), mini = this.o.mini, narrow = w < 560;
    if (!force && w === this.cw) return;
    this.cw = w; this.narrow = narrow;
    const [lo, hi] = mini ? [90, 240] : narrow ? [170, 520] : [170, 430];
    let hh = mini ? w * 0.6 : narrow ? w : w * 0.5;
    for (let it = 0; it < 5; it++) {
      this.camera.aspect = w / hh; this.camera.updateProjectionMatrix();
      const want = clamp(Math.round((hh * this.fit()) / 0.95), lo, hi);
      if (Math.abs(want - hh) < 2) break;
      hh = want;
    }
    hh = Math.round(hh); this.ch = hh;
    this.camera.aspect = w / hh; this.camera.updateProjectionMatrix(); this.fit();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(hh * dpr);
    this.wrap.style.height = `${hh}px`;
    this.dirty = true; glKick();
  }
  bindInput() {
    let drag = null;
    const w = this.wrap;
    w.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY, id: e.pointerId }; try { w.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ } });
    w.addEventListener("pointermove", (e) => {
      if (!drag || e.pointerId !== drag.id) return;
      const v = this.view, dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      v.dYaw = clamp(v.dYaw - dx * 0.006, -1.4, 1.4); v.dPitch = clamp(v.dPitch + dy * 0.004, -0.35, 0.8);
      if (!v.touched) { v.touched = true; if (this.hint) this.hint.classList.add("off"); }
      this.dirty = true; glKick();
    });
    const up = () => { drag = null; };
    w.addEventListener("pointerup", up); w.addEventListener("pointercancel", up);
    w.addEventListener("dblclick", () => { this.view.dYaw = 0; this.view.dPitch = 0; this.dirty = true; glKick(); });
  }
  /* ---------- geometry */
  pathFor(n) {
    const G = this.G, { d, r, a } = D3, raw = [];
    let per;
    if (this.straight) {
      per = D3.qLen / G.total;
      for (let i = 0; i <= 200; i++) raw.push([-D3.qLen / 2 + (D3.qLen * i) / 200, 0, 0]);
    } else {
      per = LOOP3 / G.loop;
      const pre = G.pre * per, post = G.post * per;
      for (let i = 0; i <= 40; i++) raw.push([-pre + (pre * i) / 40, 0, 0]);
      for (let k = 0; k < n; k++) for (let i = 1; i <= 900; i++) { const t = (2 * Math.PI * i) / 900; raw.push([2 * Math.PI * r * k + r * t + d * Math.sin(t), d * (1 - Math.cos(t)), a * Math.sin(t)]); }
      const xe = 2 * Math.PI * r * n;
      for (let i = 1; i <= 40; i++) raw.push([xe + (post * i) / 40, 0, 0]);
    }
    const cum = [0];
    for (let i = 1; i < raw.length; i++) cum.push(cum[i - 1] + Math.hypot(raw[i][0] - raw[i - 1][0], raw[i][1] - raw[i - 1][1], raw[i][2] - raw[i - 1][2]));
    const L = cum[cum.length - 1], N = Math.ceil(L / 0.022) + 1, S = [];
    let j = 0;
    for (let i = 0; i < N; i++) {
      const l = (i * L) / (N - 1);
      while (j < raw.length - 2 && cum[j + 1] < l) j++;
      const f = clamp((l - cum[j]) / Math.max(1e-9, cum[j + 1] - cum[j]), 0, 1);
      S.push({ x: lerp(raw[j][0], raw[j + 1][0], f), y: lerp(raw[j][1], raw[j + 1][1], f), z: lerp(raw[j][2], raw[j + 1][2], f), p: l / per });
    }
    let x0 = Infinity, x1 = -Infinity;
    for (const q of S) { x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); }
    const xc = (x0 + x1) / 2;
    for (const q of S) q.x -= xc;
    return { S, per, T: L / per, xc, span: x1 - x0, L };
  }
  spanFor(n) { return this.straight ? D3.qLen : (this.G.pre + this.G.post) * (LOOP3 / this.G.loop) + 2 * Math.PI * D3.r * n; }
  build(n, trained) {
    const o = this.o, G = this.G, st = this.straight;
    ++this.token; this.runs = [];
    this.n = st ? 4 : n; this.trained = st ? 4 : (trained ?? n);
    if (this.root) { this.scene.remove(this.root); this.disposeObj(this.root); }
    this.root = new T3.Group(); this.scene.add(this.root);
    clearEl(this.labs); this.labels = []; this.loopEls = null;
    const P = this.pathFor(this.n);
    this.S = P.S; this.per = P.per; this.T = P.T; this.Lw = P.L;
    this.Xs = Array.from({ length: this.n + 1 }, (_, k) => 2 * Math.PI * D3.r * k - P.xc);
    this.pExit = st ? G.total : G.pre + this.n * G.loop;
    const nn = this.n;
    this.unitOf = (p) => (st ? Math.min(4, Math.floor(p / 9) + 1) : p < G.pre ? 0 : p >= G.pre + nn * G.loop ? nn + 1 : Math.min(nn, Math.floor((p - G.pre) / G.loop) + 1));
    for (const q of this.S) q.u = this.unitOf(q.p);
    /* the tube: glass outside, liquid inside */
    const curve = new PathCurve(this.S), segs = this.S.length - 1;
    this.segs = segs;
    const common = { uShift: { value: 0 }, uReveal: { value: 1e9 }, uLight: { value: this.light }, uTime: { value: 0 } };
    this.common = common;
    this.mLiq = new T3.ShaderMaterial({ vertexShader: VS3, fragmentShader: FS_LIQ, vertexColors: true,
      uniforms: Object.assign({}, common, { uFront: { value: -1 }, uBands: { value: this.T / 2.4 } }) });
    this.mGlass = new T3.ShaderMaterial({ vertexShader: VS3, fragmentShader: FS_GLASS, transparent: true, depthWrite: false,
      uniforms: Object.assign({}, common, { uFront: this.mLiq.uniforms.uFront, uEdge: { value: this.col.edge }, uExtraCol: { value: this.col.steer }, uHaloCol: { value: this.col.refuse } }) });
    this.gGlass = this.tubeGeo(curve, segs, D3.rt, 18, () => [0, 0, 0]); /* extra loops get the same glass: it is the same model, run longer */
    this.gLiq = this.tubeGeo(curve, segs, D3.liq, 10, () => [0, 0, 0], true);
    this.glass = new T3.Mesh(this.gGlass, this.mGlass); this.glass.renderOrder = 2;
    this.liquid = new T3.Mesh(this.gLiq, this.mLiq);
    this.root.add(this.liquid, this.glass);
    this.zoned = [{ geo: this.gGlass, radial: 18, ring: (i) => this.S[i] }, { geo: this.gLiq, radial: 10, ring: (i) => this.S[i] }];
    /* the bead at the head of the liquid, an inlet cap and the outlet bulb */
    this.bead = new T3.Mesh(new T3.SphereGeometry(D3.rt * 0.9, 22, 16), new T3.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, emissive: 0x000000 }));
    this.bead.visible = false; this.root.add(this.bead);
    const q0 = this.S[0], q1 = this.S[1];
    /* the inlet is the token embedding: a dark block the stream leaves from (the outlet bulb is the LM head) */
    const cap = new T3.Mesh(new T3.BoxGeometry(D3.rt * 3, 0.13, D3.rt * 3), new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.55 }));
    cap.position.set(q0.x, q0.y, q0.z); cap.quaternion.setFromUnitVectors(UP, new T3.Vector3(q1.x - q0.x, q1.y - q0.y, q1.z - q0.z).normalize());
    this.root.add(cap);
    const qe = this.S[segs], bulb = new T3.Group();
    bulb.position.set(qe.x + D3.rt * 1.5, qe.y, qe.z);
    const bg = new T3.Mesh(new T3.SphereGeometry(D3.rt * 1.95, 28, 20), new T3.MeshStandardMaterial({ color: 0xffffff, transparent: true, opacity: 0.28, roughness: 0.15, depthWrite: false }));
    bg.renderOrder = 2;
    const bl = new T3.Mesh(new T3.SphereGeometry(D3.rt * 1.4, 24, 18), new T3.MeshStandardMaterial({ color: 0x9aa3b2, roughness: 0.35 }));
    bulb.add(bl, bg); bulb.userData = { x: bulb.position.x, p: this.T + 1 };
    this.bulb = { g: bulb, liq: bl }; bl.visible = false;
    this.root.add(bulb);
    this.objs = [bulb]; /* objects that move or hide while loops are added */
    if (this.feed) this.bypass3();
    if (o.arch) this.arch3();
    this.labels3();
    this.dev = new T3.Group(); this.root.add(this.dev);
    this.fit();
    this.apply({});
    this.setFront(0);
  }
  /* a tube along the path with per-ring attributes [zone, extra, halo] (and colours for the liquid) */
  tubeGeo(curve, segs, radius, radial, ringFn, colored) {
    const g = new T3.TubeGeometry(curve, segs, radius, radial, false);
    const nv = g.attributes.position.count;
    const z = new Float32Array(nv), e = new Float32Array(nv), hl = new Float32Array(nv);
    for (let i = 0; i <= segs; i++) {
      const [a, b, c] = ringFn(i);
      for (let j = 0; j <= radial; j++) { const v = i * (radial + 1) + j; z[v] = a; e[v] = b; hl[v] = c; }
    }
    g.setAttribute("aZone", new T3.BufferAttribute(z, 1));
    g.setAttribute("aExtra", new T3.BufferAttribute(e, 1));
    g.setAttribute("aHalo", new T3.BufferAttribute(hl, 1));
    if (colored) g.setAttribute("color", new T3.BufferAttribute(new Float32Array(nv * 3).fill(0.6), 3));
    return g;
  }
  ptAt(p) { const q = this.S[this.idxAt(p)]; return new T3.Vector3(q.x, q.y, q.z); }
  tanAt(p) {
    const i = this.idxAt(p), a = this.S[Math.max(0, i - 2)], b = this.S[Math.min(this.S.length - 1, i + 2)];
    return new T3.Vector3(b.x - a.x, b.y - a.y, b.z - a.z).normalize();
  }
  idxAt(p) { return Track.prototype.idxAt.call(this, p); }
  bld(p) { return Track.prototype.bld.call(this, p); }
  compute(P) { return Track.prototype.compute.call(this, P); }
  /* Dory: a pipe forks off the end of the entry stack, runs under the loops, and rises into the start of every later loop */
  bypass3() {
    const X = this.Xs, n = this.n, y = D3.byY, F = X[0] - 0.62, H0 = X[0] + 0.04, end = X[n - 1], raw = [];
    const bz = (t, a, b, c, d) => { const u = 1 - t; return u * u * u * a + 3 * u * u * t * b + 3 * u * t * t * c + t * t * t * d; };
    for (let i = 0; i <= 40; i++) { const t = i / 40; raw.push({ x: bz(t, F, F + 0.36, H0 - 0.36, H0), y: bz(t, 0, 0, y, y), z: 0 }); }
    for (let i = 1; i <= 60; i++) raw.push({ x: H0 + ((end - H0) * i) / 60, y, z: 0 });
    const curve = new PathCurve(raw), segs = raw.length - 1;
    const gG = this.tubeGeo(curve, segs, D3.byR, 12, () => [0, 0, 0]), gL = this.tubeGeo(curve, segs, D3.byR * 0.7, 8, () => [0, 0, 0], true);
    this.mByLiq = new T3.ShaderMaterial({ vertexShader: VS3, fragmentShader: FS_LIQ, vertexColors: true,
      uniforms: Object.assign({}, this.common, { uFront: { value: -1 }, uBands: { value: 6 } }) });
    const glass = new T3.Mesh(gG, this.mGlass); glass.renderOrder = 2;
    this.root.add(new T3.Mesh(gL, this.mByLiq), glass);
    this.zoned.push({ geo: gG, radial: 12, ring: (i) => ({ x: raw[i].x, p: -1 }) }, { geo: gL, radial: 8, ring: (i) => ({ x: raw[i].x, p: -1 }) });
    const elbow = new T3.Mesh(new T3.SphereGeometry(D3.byR * 1.25, 16, 12), this.byGlassMat());
    elbow.position.set(end, y, 0); elbow.userData = { x: end, p: -1 }; this.root.add(elbow); this.objs.push(elbow);
    this.byp = { raw, gL, F, H0, end, pF: this.G.pre - (X[0] - F) / this.per, risers: [] };
    for (let k = 1; k < n; k++) {
      const x = X[k], g = new T3.Group(), len = -y - D3.rt * 0.55;
      g.position.set(x, y, 0); g.userData = { x, p: -1 };
      const tube = new T3.Mesh(new T3.CylinderGeometry(D3.byR, D3.byR, len, 16, 1, true), this.byGlassMat());
      tube.position.y = len / 2; tube.renderOrder = 2;
      const lo = new T3.Mesh(new T3.CylinderGeometry(D3.byR * 0.7, D3.byR * 0.7, len / 2, 10), new T3.MeshStandardMaterial({ color: 0x9aa3b2, roughness: 0.4 }));
      const hi = new T3.Mesh(new T3.CylinderGeometry(D3.byR * 0.7, D3.byR * 0.7, len / 2, 10), new T3.MeshStandardMaterial({ color: 0x9aa3b2, roughness: 0.4 }));
      lo.geometry.translate(0, len / 4, 0); hi.geometry.translate(0, len / 4, 0); hi.position.y = len / 2;
      g.add(tube, lo, hi); this.root.add(g); this.objs.push(g);
      this.byp.risers.push({ k, g, lo, hi, len, p: this.G.pre + k * this.G.loop, mode: "usual" });
    }
  }
  byGlassMat() { return new T3.MeshStandardMaterial({ color: this.col.edge, transparent: true, opacity: 0.32, roughness: 0.2, depthWrite: false }); }
  /* the model cards: one ring per layer for how it normalises, and a gate where loops hand over */
  arch3() {
    const A = this.o.arch, G = this.G, T = this.T, kind = A.ticks;
    const inLayers = (p) => (this.straight ? p < G.total : G.stacks ? p < T : p >= G.pre && p < G.pre + this.n * G.loop);
    const ps = [];
    if (kind) for (let p = 0.5; p < T; p++) if (inLayers(p)) ps.push(p);
    const dbl = kind === "sandwich", geo = new T3.TorusGeometry(D3.rt * (kind === "renorm" ? 1.16 : 1.05), kind === "renorm" ? 0.016 : 0.011, 6, 30);
    const mat = new T3.MeshBasicMaterial({ color: kind === "renorm" ? this.col.ink2 : this.col.edge, transparent: true, opacity: kind === "pre" ? 0.55 : kind === "sandwich" ? 0.8 : 0.85 });
    if (ps.length) {
      const im = new T3.InstancedMesh(geo, mat, ps.length * (dbl ? 2 : 1)), m = new T3.Matrix4(), qt = new T3.Quaternion(), one = new T3.Vector3(1, 1, 1);
      let c = 0;
      for (const p of ps) for (const dp of dbl ? [-0.16, 0.16] : [0]) { qt.setFromUnitVectors(ZA, this.tanAt(p + dp)); m.compose(this.ptAt(p + dp), qt, one); im.setMatrixAt(c++, m); }
      this.root.add(im);
    }
    if (A.gates && !this.straight) for (let k = 1; k < this.n; k++) {
      const p = G.pre + k * G.loop, ring = new T3.Mesh(new T3.TorusGeometry(D3.rt * 1.42, 0.05, 10, 36), new T3.MeshStandardMaterial({ color: this.col.ink, roughness: 0.5 }));
      ring.position.copy(this.ptAt(p)); ring.quaternion.setFromUnitVectors(ZA, this.tanAt(p));
      this.root.add(ring);
    }
    if (A.exits && !this.straight) this.exits3();
    if (A.moe) this.moe3();
    if (A.plugs && !this.straight) this.plugs3();
    if (A.msgs && !this.straight) this.msgs3();
  }
  /* Inkling's mixture of experts: a ring of 16 beads around every layer visit, each bead standing for 8 of the layer's
     128 experts, one of them lit (the token goes to 8). The lit bead is a sketch; it moves between passes because the
     router sees a new state on every pass, so the same layer can pick other experts. On the card it swells as the bead
     passes. */
  moe3() {
    const G = this.G, T = this.T, NB = 16, R = D3.rt * 1.62, ps = [];
    for (let p = 0.5; p < T; p++) ps.push(p);
    const layerOf = (p) => { const i = Math.floor(p); if (i < G.pre) return [i, 0]; const c = i - G.pre; return c < this.n * G.loop ? [G.pre + (c % G.loop), 1 + Math.floor(c / G.loop)] : [G.pre + G.loop + (c - this.n * G.loop), 0]; };
    const pick = (l, pass) => (l * 7 + pass * 5 + ((l * l * 3) % 5)) % NB;
    const dimG = new T3.SphereGeometry(0.023, 8, 6), litG = new T3.SphereGeometry(0.036, 12, 9);
    const dim = new T3.InstancedMesh(dimG, new T3.MeshStandardMaterial({ color: this.col.edge, roughness: 0.5, transparent: true, opacity: 0.75 }), ps.length * (NB - 1));
    const litC = new T3.Color(cssHex("--inkling"));
    const lit = new T3.InstancedMesh(litG, new T3.MeshStandardMaterial({ color: litC, emissive: litC, emissiveIntensity: 0.25, roughness: 0.4 }), ps.length);
    const m = new T3.Matrix4(), qt = new T3.Quaternion(), one = new T3.Vector3(1, 1, 1);
    this.moeLit = [];
    let c = 0;
    ps.forEach((p, j) => {
      const [l, pass] = layerOf(p), k0 = pick(l, pass), at = this.ptAt(p);
      qt.setFromUnitVectors(ZA, this.tanAt(p));
      for (let k = 0; k < NB; k++) {
        const a = (2 * Math.PI * k) / NB, pos = new T3.Vector3(R * Math.cos(a), R * Math.sin(a), 0).applyQuaternion(qt).add(at);
        if (k === k0) { m.compose(pos, qt, one); lit.setMatrixAt(j, m); this.moeLit.push({ p, pos, q: qt.clone() }); }
        else { m.compose(pos, qt, one); dim.setMatrixAt(c++, m); }
      }
    });
    this.moeMesh = lit;
    this.root.add(dim, lit);
  }
  moePulse(pb) {
    if (!this.moeMesh) return;
    const m = new T3.Matrix4(), sc = new T3.Vector3();
    this.moeLit.forEach((b, j) => { const f = 1 + 0.9 * Math.exp(-(((b.p - pb) / 0.45) ** 2)); sc.set(f, f, f); m.compose(b.pos, b.q, sc); this.moeMesh.setMatrixAt(j, m); });
    this.moeMesh.instanceMatrix.needsUpdate = true;
  }
  /* Inkling: every pass restarts from the prelude's state, so the stream of the pass before stops at a plug just before
     the shortcut rises into the next pass */
  plugs3() {
    const G = this.G, mat = new T3.MeshStandardMaterial({ color: this.col.ink, roughness: 0.5 });
    for (let k = 1; k < this.n; k++) {
      const p = G.pre + k * G.loop - 0.32, disc = new T3.Mesh(new T3.CylinderGeometry(D3.rt * 1.22, D3.rt * 1.22, 0.045, 30), mat);
      disc.position.copy(this.ptAt(p)); disc.quaternion.setFromUnitVectors(UP, this.tanAt(p));
      this.root.add(disc);
    }
  }
  /* Inkling: the passes talk through gated messages, read from the previous pass into every core sublayer: a dashed
     bridge from the top of each pass to the top of the next, with a valve (the gate) at its peak */
  msgYs() { const y0 = 2 * D3.d + D3.rt + 0.06; return { y0, peak: y0 + 0.36 }; }
  msgs3() {
    const G = this.G, ink = new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.5 }), { y0, peak } = this.msgYs();
    const rod = new T3.CylinderGeometry(0.016, 0.016, 1, 8);
    for (let k = 1; k < this.n; k++) {
      const a = this.ptAt(G.pre + (k - 0.5) * G.loop), b = this.ptAt(G.pre + (k + 0.5) * G.loop);
      const P0 = new T3.Vector3(a.x + 0.05, y0, a.z), P2 = new T3.Vector3(b.x - 0.05, y0, b.z), P1 = new T3.Vector3((a.x + b.x) / 2, 2 * peak - y0, (a.z + b.z) / 2);
      const bz = (t) => P0.clone().multiplyScalar((1 - t) ** 2).addScaledVector(P1, 2 * t * (1 - t)).addScaledVector(P2, t * t);
      const pts = Array.from({ length: 61 }, (_, i) => bz(i / 60)), cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
      const L = cum[cum.length - 1], at = (l) => { let i = 1; while (i < cum.length - 1 && cum[i] < l) i++; const f = (l - cum[i - 1]) / Math.max(1e-9, cum[i] - cum[i - 1]); return pts[i - 1].clone().lerp(pts[i], f); };
      const on = 0.07, off = 0.045, gap = 0.13; /* no dashes under the valve */
      for (let l = 0.02; l + on < L - 0.06; l += on + off) {
        if (Math.abs(l + on / 2 - L / 2) < gap) continue;
        const u = at(l), v = at(l + on), d = new T3.Mesh(rod, ink);
        d.position.copy(u).lerp(v, 0.5); d.scale.set(1, u.distanceTo(v), 1); d.quaternion.setFromUnitVectors(UP, v.clone().sub(u).normalize());
        this.root.add(d);
      }
      const tip = new T3.Mesh(new T3.ConeGeometry(0.042, 0.09, 14), ink), e = at(L - 0.04), e0 = at(L - 0.12);
      tip.position.copy(e); tip.quaternion.setFromUnitVectors(UP, e.clone().sub(e0).normalize()); this.root.add(tip);
      const mid = at(L / 2), wheel = new T3.Mesh(new T3.TorusGeometry(0.075, 0.016, 8, 28), ink); wheel.position.copy(mid);
      const bar = new T3.Mesh(new T3.BoxGeometry(0.15, 0.016, 0.016), ink); bar.position.copy(mid);
      const bar2 = bar.clone(); bar2.rotation.z = Math.PI / 2;
      this.root.add(wheel, bar, bar2);
    }
  }
  /* Ouro: after every loop an exit gate can stop and send the stream to the LM head (trained that way; we always run
     every loop, so the routes stay empty). Each gate is a valve under the loop's end with its own dashed route to the
     head; the routes nest (the earliest runs deepest) so none meets another before the head. */
  exitDepth(k) { return -0.5 - (this.n - 1 - k) * 0.17; }
  exits3() {
    const X = this.Xs, n = this.n, xh = this.bulb.g.position.x, rb = D3.rt * 1.95, ink = new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.5 });
    const dash = (x0, y0, x1, y1) => { /* a dashed line of short rods */
      const a = new T3.Vector3(x0, y0, 0), b = new T3.Vector3(x1, y1, 0), len = a.distanceTo(b), dir = b.clone().sub(a).normalize();
      const q = new T3.Quaternion().setFromUnitVectors(UP, dir), on = 0.07, off = 0.045, nd = Math.max(1, Math.floor((len + off) / (on + off)));
      const geo = new T3.CylinderGeometry(0.017, 0.017, on, 8), im = new T3.InstancedMesh(geo, ink, nd), m = new T3.Matrix4(), one = new T3.Vector3(1, 1, 1);
      const pad = (len - (nd * on + (nd - 1) * off)) / 2;
      for (let i = 0; i < nd; i++) m.compose(a.clone().addScaledVector(dir, pad + on / 2 + i * (on + off)), q, one), im.setMatrixAt(i, m);
      this.root.add(im);
    };
    const dot = (x, y) => { const s = new T3.Mesh(new T3.SphereGeometry(0.022, 10, 8), ink); s.position.set(x, y, 0); this.root.add(s); };
    const span = n > 2 ? 0.24 / (n - 2) : 0;
    for (let k = 1; k < n; k++) {
      const x = X[k], y0 = -D3.rt - 0.02, yv = y0 - 0.1, yd = this.exitDepth(k), xr = xh - 0.12 + (n - 1 - k) * span;
      /* the valve: a short stem out of the tube and a hand wheel facing the viewer */
      const stem = new T3.Mesh(new T3.CylinderGeometry(0.024, 0.024, 0.1, 10), ink); stem.position.set(x, y0 - 0.05, 0);
      const wheel = new T3.Mesh(new T3.TorusGeometry(0.085, 0.016, 8, 28), ink); wheel.position.set(x, yv - 0.035, 0);
      const bar = new T3.Mesh(new T3.BoxGeometry(0.17, 0.016, 0.016), ink); bar.position.copy(wheel.position);
      const bar2 = bar.clone(); bar2.rotation.z = Math.PI / 2;
      this.root.add(stem, wheel, bar, bar2);
      /* its own route: down, along, and up into the head */
      const yr = Math.sqrt(Math.max(0, rb * rb - (xr - xh) ** 2));
      dash(x, yv - 0.13, x, yd); dot(x, yd); dash(x, yd, xr, yd); dot(xr, yd); dash(xr, yd, xr, -yr - 0.02);
      const tip = new T3.Mesh(new T3.ConeGeometry(0.04, 0.08, 14), ink); tip.position.set(xr, -yr - 0.02, 0); this.root.add(tip);
    }
  }
  label(text, pos, cls, extra = {}) {
    const el = h("span", { class: `s3-lab ${cls}` }, text);
    this.labs.append(el);
    const L = Object.assign({ el, pos, x: pos.x, p: -1 }, extra);
    this.labels.push(L);
    return L;
  }
  /* how high the drawing reaches: the top of the loops, or of a dropper above them */
  topY() {
    const G = this.G, st = this.straight, d = D3.d;
    let y = (st ? 0 : 2 * d) + D3.rt;
    if (!st && this.o.arch && this.o.arch.msgs) y = this.msgYs().peak + 0.08;
    for (const t of (this.o.types || []).filter((t) => t !== "erase")) {
      const ys = st ? 0 : d * (1 - Math.cos((2 * Math.PI * G.site[t]) / G.loop));
      y = Math.max(y, this.dropTop(ys) + 0.33);
    }
    return y;
  }
  /* one small label over the loops (no per-loop text: the loops speak for themselves), the stacks Dory runs once,
     and on the model cards the exit gates and the head */
  labels3() {
    const G = this.G, o = this.o, mini = o.mini, V = (x, y, z = 0) => new T3.Vector3(x, y, z), top = this.topY() + (mini ? 0.24 : 0.3);
    if (this.straight) {
      this.label(mini ? "36 layers, one pass" : "36 layers, run once", mini ? V(0, -0.52) : V(0, top), "s3-small");
      if (!mini) for (let k = 1; k < 4; k++) { const ring = new T3.Mesh(new T3.TorusGeometry(D3.rt * 1.2, 0.02, 8, 32), new T3.MeshBasicMaterial({ color: this.col.edge })); ring.position.copy(this.ptAt(9 * k)); ring.quaternion.setFromUnitVectors(ZA, this.tanAt(9 * k)); this.root.add(ring); }
      return;
    }
    const X = this.Xs, per = this.per;
    const NM = G.names || ["entry stack", "exit stack", "looped stack"];
    this.label(NM[2], V((X[0] + X[this.n]) / 2, top), "s3-small", { mid: true });
    if (G.stacks && (mini || !this.narrow)) {
      this.label(NM[0], V(X[0] - (G.pre * per) * 0.62, mini ? -0.52 : -0.4), "s3-small", { p: G.pre / 2 });
      this.label(NM[1], V(X[this.n] + (G.post * per) * 0.6, mini ? -0.52 : -0.4), "s3-small", { p: this.T - G.post / 2 });
    }
    if (o.arch && o.arch.msgs && this.n > 1) this.label("messages", V(X[this.n - 1] + Math.PI * D3.r + 0.62, this.msgYs().peak - 0.12), "s3-small");
    if (o.arch && o.arch.exits) this.label("exit gates", V((X[1] + X[this.n]) / 2, this.exitDepth(1) - 0.2), "s3-small");
    if (mini && o.arch && !G.stacks) { /* Ouro and Nanbeige: the two ends of the tube */
      this.label("embedding", V(this.S[0].x - 0.05, -0.46), "s3-small");
      this.label("LM head", V(this.bulb.g.position.x + 0.62, 0), "s3-small");
    }
  }
  /* frame the camera on the largest run this figure shows, so adding loops does not zoom */
  fit() {
    const v = this.view, mini = this.o.mini, nF = Math.max(this.n || 0, this.o.maxLaps || 0), G = this.G, d = D3.d, V = (x, y, z = 0) => new T3.Vector3(x, y, z);
    v.yaw0 = mini ? -0.34 : this.narrow ? -0.82 : -0.4; v.pitch0 = mini ? 0.2 : this.narrow ? 0.3 : 0.22;
    if (!this.fitPts || this.fitPts.n !== nF) {
      const P = nF === this.n ? { S: this.S } : this.pathFor(nF), pts = [], S = P.S, rt = D3.rt * (mini ? 1.5 : 2.2);
      for (let i = 0; i < S.length; i += 8) { const q = S[i]; pts.push(V(q.x, q.y + rt, q.z), V(q.x, q.y - rt, q.z), V(q.x, q.y, q.z + rt), V(q.x, q.y, q.z - rt)); }
      const q = S[S.length - 1]; pts.push(V(q.x + D3.rt * 3.6, q.y, q.z));
      const top = this.topY() + (mini ? 0.36 : 0.44); /* the "looped stack" line */
      if (this.straight) { if (!mini) pts.push(V(S[0].x, -0.36), V(S[S.length - 1].x, -0.36), V(0, top)); else pts.push(V(0, -0.66)); }
      else {
        const x0 = S[0].x + G.pre * (LOOP3 / G.loop), xe = x0 + 2 * Math.PI * D3.r * nF;
        pts.push(V((x0 + xe) / 2, top));
        if (G.stacks) pts.push(V(S[0].x + 0.3, mini ? -0.6 : -0.5), V(S[S.length - 1].x - 0.3, mini ? -0.6 : -0.5));
        if (this.feed) pts.push(V(x0 - 0.7, D3.byY - D3.byR - 0.06), V(xe, D3.byY - D3.byR - 0.06));
        if (this.o.arch && this.o.arch.exits) { const yd = this.exitDepth(1) - 0.3; pts.push(V(x0, yd), V(xe, yd)); }
        if (mini && this.o.arch && !G.stacks) pts.push(V(S[0].x - 0.6, -0.62), V(S[S.length - 1].x + 1.7, 0));
      }
      pts.n = nF; this.fitPts = pts;
    }
    const pts = this.fitPts;
    const cam = this.camera, tv = Math.tan((cam.fov * Math.PI) / 360), th = tv * cam.aspect, fill = mini ? 0.95 : 0.9; /* room for the sway */
    const fwd = new T3.Vector3(Math.sin(v.yaw0) * Math.cos(v.pitch0), Math.sin(v.pitch0), Math.cos(v.yaw0) * Math.cos(v.pitch0));
    const right = new T3.Vector3().crossVectors(UP, fwd).normalize(), up = new T3.Vector3().crossVectors(fwd, right);
    let y0 = Infinity, y1 = -Infinity;
    for (const q of pts) { y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y); }
    const c = new T3.Vector3(0, (y0 + y1) / 2, 0);
    let D = 10;
    for (let pass = 0; pass < 3; pass++) {
      D = 0;
      for (const q of pts) { const r = q.clone().sub(c); D = Math.max(D, r.dot(fwd) + Math.abs(r.dot(right)) / (th * fill), r.dot(fwd) + Math.abs(r.dot(up)) / (tv * fill)); }
      let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
      for (const q of pts) { const r = q.clone().sub(c), z = D - r.dot(fwd); a0 = Math.min(a0, r.dot(right) / z); a1 = Math.max(a1, r.dot(right) / z); b0 = Math.min(b0, r.dot(up) / z); b1 = Math.max(b1, r.dot(up) / z); }
      c.addScaledVector(right, ((a0 + a1) / 2) * D).addScaledVector(up, ((b0 + b1) / 2) * D);
    }
    this.target.copy(c); this.dist = D;
    let b0 = Infinity, b1 = -Infinity;
    for (const q of pts) { const r = q.clone().sub(c), z = D - r.dot(fwd); b0 = Math.min(b0, r.dot(up) / z); b1 = Math.max(b1, r.dot(up) / z); }
    return (b1 - b0) / (2 * tv); /* share of the height the model fills */
  }
  dropTop(ys) { return this.straight ? ys + D3.dropH : Math.max(ys + D3.dropH, 2 * D3.d + D3.rt + 0.42); }
  updateCamera(now) {
    const v = this.view, sway = !RM && !v.touched && !this.o.mini ? 0.07 * Math.sin((now - v.t0) / 2800) : 0;
    const yaw = v.yaw0 + v.dYaw + sway, pitch = v.pitch0 + v.dPitch;
    this.camera.position.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)).multiplyScalar(this.dist).add(this.target);
    this.camera.lookAt(this.target);
  }
  /* ---------- what the stream carries */
  apply(P) {
    this.P = P || {};
    this.compute(this.P);
    this.paint();
    this.devices();
    this.setFront(0);
    this.legend();
  }
  colorAt(i) {
    const P = this.P, m = this.o.meaning, SIG = (m && m.sig) || RGB.red, q = this.S[i];
    if (!m && !P.leak) return RGB.grey; /* the model cards: no behaviour, just the architecture */
    if (P.leak) return mixRGB(RGB.flow, RGB.red, this.HALO[i]);
    let vis = 1; /* a filter clears the liquid in its loop; a random-direction filter takes out nothing, so the behaviour builds as usual */
    if (!P.curve && !P.random) for (const e of this.evs) if (e.type === "erase") {
      if (q.p >= e.a && q.p <= e.b) vis = 0;
      else if (q.p > e.b) vis = Math.min(vis, smooth((q.p - e.b) / (0.3 * (this.straight ? 9 : this.G.loop))));
    }
    return mixRGB(RGB.flow, SIG, this.U[i] * vis);
  }
  paint() {
    const N = this.S.length, col = this.gLiq.attributes.color.array, hl = this.gGlass.attributes.aHalo.array, P = this.P;
    this.C3 = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) { const c = this.colorAt(i); this.C3[3 * i] = c[0] / 255; this.C3[3 * i + 1] = c[1] / 255; this.C3[3 * i + 2] = c[2] / 255; }
    for (let i = 0; i < N; i++) for (let j = 0; j <= 10; j++) { const v = 3 * (i * 11 + j); col[v] = this.C3[3 * i]; col[v + 1] = this.C3[3 * i + 1]; col[v + 2] = this.C3[3 * i + 2]; }
    this.gLiq.attributes.color.needsUpdate = true;
    for (let i = 0; i < N; i++) { const g = P.leak && this.HALO[i] > 1 ? clamp(0.35 + (this.HALO[i] - 1) / 1.5, 0, 1) : 0; for (let j = 0; j <= 18; j++) hl[i * 19 + j] = g; }
    this.gGlass.attributes.aHalo.needsUpdate = true;
    /* the head shows the outcome itself: the colour the data gives at the end, without the filters' local fade */
    const last = this.S.length - 1, m = this.o.meaning;
    const endC = !m || P.leak || P.curve ? [this.C3[3 * last], this.C3[3 * last + 1], this.C3[3 * last + 2]] : mixRGB(RGB.flow, m.sig || RGB.red, this.U[last]).map((v) => v / 255);
    this.bulb.liq.material.color.setRGB(...endC);
    if (this.byp) {
      /* the copy carries what the stream holds as it leaves the entry stack */
      const k = this.idxAt(this.G.pre), cc = [this.C3[3 * k], this.C3[3 * k + 1], this.C3[3 * k + 2]];
      const bc = this.byp.gL.attributes.color.array;
      for (let v = 0; v < bc.length; v += 3) { bc[v] = cc[0]; bc[v + 1] = cc[1]; bc[v + 2] = cc[2]; }
      this.byp.gL.attributes.color.needsUpdate = true;
      const fe = P.feed || {}, ch = new Set(fe.style && fe.style !== "usual" ? fe.loops || [] : []);
      for (const r of this.byp.risers) {
        r.mode = ch.has(r.k + 1) ? fe.style : "usual";
        r.lo.material.color.setRGB(...cc);
        if (r.mode === "usual") r.hi.material.color.setRGB(...cc); else r.hi.material.color.setRGB(RGB.flow[0] / 255, RGB.flow[1] / 255, RGB.flow[2] / 255);
      }
    }
  }
  /* filters for erased loops, droppers for added vectors, devices on the shortcut */
  devices() {
    this.disposeObj(this.dev); this.root.remove(this.dev);
    this.dev = new T3.Group(); this.root.add(this.dev);
    this.filters = []; this.drops = []; this.devs = [];
    for (const L of this.labels.filter((l) => l.tag)) L.el.remove();
    this.labels = this.labels.filter((l) => !l.tag);
    const P = this.P, m = this.o.meaning || {}, G = this.G, st = this.straight, d = D3.d;
    const ustart = (k) => (st ? 9 * (k - 1) : G.pre + (k - 1) * G.loop), ulen = st ? 9 : G.loop, nf = st ? 4 : D3.nf;
    const sigC = new T3.Color().setRGB(...(m.sig || RGB.red).map((v) => v / 255));
    const filter = (p, { built = false, level = 0 } = {}) => {
      const g = new T3.Group(), R = D3.rt * (built ? 1.7 : 2.05);
      const disc = new T3.Mesh(new T3.CircleGeometry(R, 40), new T3.MeshBasicMaterial({ map: filterTex(), transparent: true, side: T3.DoubleSide, depthWrite: false, color: this.col.ink2, opacity: built ? 0.7 : 1 }));
      const frame = new T3.Mesh(new T3.TorusGeometry(R, built ? 0.016 : 0.024, 8, 48), new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.45, metalness: 0.2 }));
      disc.renderOrder = 3; g.add(disc, frame);
      const base = this.ptAt(p);
      g.quaternion.setFromUnitVectors(ZA, this.tanAt(p));
      this.dev.add(g);
      const f = { g, disc, p, base, level, baseCol: disc.material.color.clone(), caught: sigC };
      this.filters.push(f); this.devs.push({ g, base, lift: 0.9 });
      return f;
    };
    /* an erased loop's filters run from its very first layer to its last, so the colour changes at the first one */
    const fpos = (k, j) => ustart(k) + FOFF + (j * ulen) / nf;
    for (const e of this.evs) {
      if (e.type === "erase") {
        for (let j = 0; j < nf; j++) { const p = fpos(e.k, j); filter(p, { level: !P.random && m.natural ? this.bld(p) : 0 }); }
      } else {
        /* the drop's colour is what it adds: against refusal it is the colour of a stream with no refusal; a persona its
           trait's colour; a kick black */
        const dc = e.type === "push" ? this.col.flow : new T3.Color().setRGB(...(m.sig || RGB.violet).map((v) => v / 255));
        const p = e.a, base = this.ptAt(p), top = new T3.Vector3(base.x, this.dropTop(base.y), base.z), g = new T3.Group();
        const cone = new T3.Mesh(new T3.ConeGeometry(0.085, 0.26, 24, 1, true), new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.4, side: T3.DoubleSide }));
        cone.rotation.x = Math.PI; cone.position.y = 0.13;
        const lid = new T3.Mesh(new T3.CylinderGeometry(0.1, 0.1, 0.05, 24), new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.4 }));
        lid.position.y = 0.28;
        g.add(cone, lid); g.position.copy(top);
        const drop = new T3.Mesh(new T3.SphereGeometry(0.085, 20, 14), new T3.MeshStandardMaterial({ color: dc, roughness: 0.25, emissive: dc, emissiveIntensity: 0.15 }));
        drop.scale.set(1, 1.3, 1);
        const ripple = new T3.Mesh(new T3.TorusGeometry(1, 0.035, 8, 40), new T3.MeshBasicMaterial({ color: dc, transparent: true, opacity: 0 }));
        ripple.quaternion.setFromUnitVectors(ZA, this.tanAt(p)); ripple.position.copy(base);
        const port = new T3.Mesh(new T3.TorusGeometry(D3.rt * 1.12, 0.028, 8, 36), new T3.MeshStandardMaterial({ color: dc, roughness: 0.4 }));
        port.quaternion.copy(ripple.quaternion); port.position.copy(base);
        this.dev.add(g, drop, ripple, port);
        const dv = { g, base: top, lift: 0.6 };
        this.drops.push({ p, base, top: top.clone().add(new T3.Vector3(0, -0.07, 0)), drop, ripple, port, dv });
        this.devs.push(dv);
      }
    }
    if (P.edited) for (let k = 1; k <= this.n; k++) for (let j = 0; j < nf; j++) filter(fpos(k, j), { built: true });
    if (this.byp) {
      for (const r of this.byp.risers) {
        if (r.mode === "usual") continue;
        const yy = r.len / 2, at = new T3.Vector3(r.g.position.x, D3.byY + yy, 0);
        if (r.mode === "cut") {
          const capM = new T3.Mesh(new T3.CylinderGeometry(D3.byR * 1.9, D3.byR * 1.9, 0.05, 24), new T3.MeshStandardMaterial({ color: this.col.ink, roughness: 0.5 }));
          capM.position.copy(at); this.dev.add(capM); this.devs.push({ g: capM, base: at, lift: 0.5 });
        } else {
          const g = new T3.Group(), R = D3.byR * 2.4;
          const disc = new T3.Mesh(new T3.CircleGeometry(R, 32), new T3.MeshBasicMaterial({ map: filterTex(), transparent: true, side: T3.DoubleSide, depthWrite: false, color: this.col.ink2 }));
          const frame = new T3.Mesh(new T3.TorusGeometry(R, 0.016, 8, 36), new T3.MeshStandardMaterial({ color: this.col.ink2, roughness: 0.45 }));
          g.add(disc, frame); g.quaternion.setFromUnitVectors(ZA, UP); g.position.copy(at); disc.renderOrder = 3;
          this.dev.add(g); this.devs.push({ g, base: at, lift: 0.5 });
        }
      }
    }
    for (const dv of this.devs) dv.ins = 0;
    this.placeDevices();
  }
  placeDevices() {
    for (const dv of this.devs) {
      const e = easeOut(dv.ins);
      dv.g.visible = dv.ins > 0.001;
      dv.g.position.copy(dv.base).add(new T3.Vector3(0, (1 - e) * dv.lift, 0));
      dv.g.scale.setScalar(0.35 + 0.65 * e);
    }
    for (const L of this.labels) if (L.tag) L.el.style.opacity = this.devs.length ? clamp(this.devs[0].ins * 1.4, 0, 1) : 1;
    for (const dp of this.drops || []) this.dropAt(dp, this.front || 0, false);
    this.dirty = true;
  }
  insertDevices(dur = 420) {
    if (!this.devs.length) return 0;
    const t0 = performance.now(), tok = this.token, n = this.devs.length, stg = Math.min(45, 300 / n);
    const f = (now) => {
      if (tok !== this.token) return false;
      let more = false;
      this.devs.forEach((dv, i) => { dv.ins = clamp((now - t0 - i * stg) / dur, 0, 1); if (dv.ins < 1) more = true; });
      this.placeDevices();
      return more;
    };
    this.runs.push(f); glKick();
    return dur + stg * n * 0.6;
  }
  /* ---------- the head of the liquid */
  setFront(p, live = false) {
    this.front = p;
    const u = p <= 0 ? -1 : this.idxAt(Math.min(p, this.T)) / this.segs + 1e-6;
    this.mLiq.uniforms.uFront.value = u;
    if (live && p > 0 && p < this.T) this.placeBead(p); else this.bead.visible = false;
    this.bulb.liq.visible = p >= this.T - 0.2;
    if (this.byp) this.fillBypass(p);
    for (const f of this.filters || []) {
      const k = clamp((p - f.p) / 1.2, 0, 1) * f.level;
      f.disc.material.color.copy(f.baseCol).lerp(f.caught, 0.95 * k);
    }
    for (const dp of this.drops || []) this.dropAt(dp, p, live);
    if (this.loopEls) { const un = live && p < this.T ? this.unitOf(p) : -1; this.loopEls.forEach((el, k) => el.classList.toggle("now", k + 1 === un)); }
    this.dirty = true;
  }
  placeBead(p) {
    const i = this.idxAt(p), q = this.S[i];
    this.bead.visible = true; this.bead.position.set(q.x, q.y, q.z);
    this.bead.material.color.setRGB(this.C3[3 * i], this.C3[3 * i + 1], this.C3[3 * i + 2]).lerp(new T3.Color(1, 1, 1), this.o.meaning ? 0.15 : 0.55);
    this.bead.material.emissive.copy(this.bead.material.color).multiplyScalar(0.25);
  }
  fillBypass(p) {
    const B = this.byp, G = this.G, nn = this.n;
    const idx = p <= B.pF ? -1 : p < G.pre ? (40 * (p - B.pF)) / (G.pre - B.pF) : 40 + (60 * (p - G.pre)) / Math.max(1, (nn - 1) * G.loop);
    this.mByLiq.uniforms.uFront.value = idx < 0 ? -1 : Math.min(1, idx / 100) + 1e-6;
    for (const r of B.risers) {
      const f = clamp((p - (r.p - 4)) / 4, 0, 1), lo = clamp(f * 2, 0, 1), hi = clamp(f * 2 - 1, 0, 1);
      r.lo.visible = lo > 0.01; r.lo.scale.y = Math.max(0.01, lo);
      r.hi.visible = hi > 0.01 && r.mode !== "cut"; r.hi.scale.y = Math.max(0.01, hi);
    }
  }
  dropAt(dp, p, live) {
    const Lf = (420 * this.o.speed) / (MSU * this.per), k = (p - (dp.p - Lf)) / Lf, g = dp.dv.g; /* the drop falls for the last 420 ms before the liquid arrives */
    dp.port.visible = p >= dp.p;
    if (!live || p < dp.p) dp.ripple.material.opacity = 0;
    if (p < dp.p && k <= 0) { /* hanging from the dropper, which may still be sliding in */
      dp.drop.visible = g.visible; dp.drop.position.copy(g.position).add(new T3.Vector3(0, -0.07, 0)); dp.drop.scale.set(g.scale.x, 1.3 * g.scale.x, g.scale.x);
      return;
    }
    if (!live) { dp.drop.visible = false; return; }
    if (p < dp.p) {
      dp.drop.visible = true; dp.drop.scale.set(1, 1.3, 1);
      dp.drop.position.copy(dp.top).lerp(dp.base, clamp(k, 0, 1) ** 2);
    } else {
      dp.drop.visible = false;
      const r = clamp((p - dp.p) / (Lf * 0.9), 0, 1);
      dp.ripple.scale.setScalar(D3.rt * (1.1 + 2.4 * r));
      dp.ripple.material.opacity = 0.75 * (1 - r);
    }
  }
  /* ---------- runs */
  play(P, { onDone, until, onProgress, placed = false } = {}) {
    const tok = ++this.token;
    this.runs = [];
    this.apply(P);
    const pEnd = until == null ? this.T : until;
    if (RM || placed) { for (const dv of this.devs) dv.ins = 1; this.placeDevices(); } /* placed: the filters already dropped in (during a morph) */
    if (RM) { this.setFront(pEnd); onDone && onDone(); return; }
    const wait = placed ? 0 : this.insertDevices();
    const dur = Math.max(600, (pEnd * this.per * MSU) / this.o.speed), t0 = performance.now() + wait; /* same speed along the tube in every model */
    this.runs.push((now) => {
      if (tok !== this.token) return false;
      const f = clamp((now - t0) / dur, 0, 1); /* 0 while the filters and drops are still dropping in */
      this.setFront(f * pEnd, f < 1); onProgress && onProgress(f);
      if (f < 1) return true;
      onDone && onDone();
      return false;
    });
    glKick();
  }
  show(P, until) {
    ++this.token; this.runs = [];
    this.apply(P);
    for (const dv of this.devs) dv.ins = 1;
    this.placeDevices();
    this.setFront(until == null ? this.T : until);
    glKick();
  }
  idle() {
    this.show({});
    if (RM) return;
    const t0 = performance.now(), ms = this.T * this.per * MSU; /* the cards' bead moves at the same speed as every run */
    const f = (now) => { const p = (((now - t0) / ms) % 1) * this.T; this.placeBead(p); this.moePulse(p); this.dirty = true; return true; };
    f.visOnly = true; this.runs.push(f); glKick();
  }
  /* add loops in place: the run with fromN loops grows into this one, or shrinks back with reverse */
  morph(fromN, { reverse = false, dur = 1300, onDone } = {}) {
    const tok = ++this.token, xs = this.Xs[fromN], dX = this.Xs[this.n] - xs, pX = this.pExit;
    const zoneOf = (x, p) => (p >= pX - 1e-6 ? 2 : x > xs + 1e-3 ? 1 : 0), zoneObj = (x, p) => (p >= pX - 1e-6 ? 2 : x > xs - 1e-3 ? 1 : 0);
    for (const z of this.zoned) {
      const a = z.geo.attributes.aZone.array, rings = a.length / (z.radial + 1);
      for (let i = 0; i < rings; i++) { const q = z.ring(i), v = zoneOf(q.x, q.p); for (let j = 0; j <= z.radial; j++) a[i * (z.radial + 1) + j] = v; }
      z.geo.attributes.aZone.needsUpdate = true;
    }
    for (const ob of this.objs) { ob.userData.zone = zoneObj(ob.userData.x, ob.userData.p); ob.userData.x0 = ob.position.x; }
    for (const L of this.labels) L.zone = zoneOf(L.x, L.p);
    this.runs = [];
    this.setFront(0);
    const set = (f) => {
      this.reveal = xs + f * dX; this.shift = -(1 - f) * dX;
      this.common.uReveal.value = this.reveal; this.common.uShift.value = this.shift;
      this.root.position.x = ((1 - f) * dX) / 2;
      for (const ob of this.objs) {
        const zn = ob.userData.zone;
        if (zn === 2) ob.position.x = ob.userData.x0 + this.shift;
        ob.visible = zn !== 1 || ob.userData.x + 0.25 <= this.reveal;
      }
      this.dirty = true;
    };
    const clear = () => {
      this.reveal = null; this.shift = 0;
      this.common.uReveal.value = 1e9; this.common.uShift.value = 0; this.root.position.x = 0;
      for (const ob of this.objs) { ob.position.x = ob.userData.x0; ob.visible = true; }
      for (const L of this.labels) L.zone = 0;
      this.dirty = true;
    };
    if (RM) { if (!reverse) clear(); else set(0); onDone && onDone(); glKick(); return; }
    const t0 = performance.now();
    set(reverse ? 1 : 0);
    this.runs.push((now) => {
      if (tok !== this.token) return false;
      const p = clamp((now - t0) / dur, 0, 1);
      set(easeInOut(reverse ? 1 - p : p));
      if (p < 1) return true;
      if (!reverse) clear();
      onDone && onDone();
      return false;
    });
    glKick();
  }
  /* ---------- frames */
  tick(now) {
    const vis = this.visible, keep = [], list = this.runs;
    let busy = false;
    this.runs = []; /* a run may start another (morph, then play); those land in the fresh list */
    for (const f of list) {
      if (f.visOnly && !vis) { keep.push(f); continue; }
      if (f(now)) { keep.push(f); busy = true; }
      this.dirty = true;
    }
    if (this.runs.length) busy = true;
    this.runs = keep.concat(this.runs);
    if (vis && !RM) { busy = true; this.dirty = true; } /* the liquid shimmers and the view sways while on screen */
    return busy;
  }
  render(now) {
    this.updateCamera(now);
    this.common.uTime.value = RM ? 0 : (now / 1000) * 0.8;
    const r = GL.renderer, w = this.canvas.width, hh = this.canvas.height;
    if (!w || !hh) return;
    if (w > GL.w || hh > GL.h) { GL.w = Math.max(GL.w, w); GL.h = Math.max(GL.h, hh); r.setSize(GL.w, GL.h, false); }
    r.setScissorTest(true); r.setViewport(0, 0, w, hh); r.setScissor(0, 0, w, hh);
    r.render(this.scene, this.camera);
    this.ctx.clearRect(0, 0, w, hh);
    this.ctx.drawImage(r.domElement, 0, GL.h - hh, w, hh, 0, 0, w, hh);
    this.placeLabels();
    if (!this.placed) { this.placed = true; this.labs.style.visibility = ""; }
  }
  placeLabels() {
    const v = new T3.Vector3(), rx = this.root.position.x;
    for (const L of this.labels) {
      v.copy(L.pos); v.x += rx + (L.zone === 2 ? this.shift || 0 : 0);
      if (L.mid) { L.zone = 0; v.x = rx + (this.Xs[0] + (this.reveal != null ? clamp(this.reveal, this.Xs[0], this.Xs[this.n]) : this.Xs[this.n])) / 2; } /* stays over the loops while they grow */
      v.project(this.camera);
      const x = ((v.x + 1) / 2) * this.cw, y = ((1 - v.y) / 2) * this.ch;
      L.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -50%)`;
      if (L.zone === 1) L.el.style.visibility = this.reveal != null && L.x > this.reveal - 0.3 ? "hidden" : "visible";
      else if (!L.tag) L.el.style.visibility = "visible";
    }
  }
  legend() {
    if (!this.leg) return;
    const m = this.o.meaning || {}, P = this.P, items = [];
    const it = (icon, text) => { const sp = h("span", {}); sp.innerHTML = icon; sp.append(text); return sp; };
    const grad = (a, b) => `<i class="sw-grad" style="background:linear-gradient(90deg,${a},${b})"></i>`;
    /* colour first (what the liquid means), then the devices in the drawing; nothing else */
    const sc = m.sigCss || rgbCss(m.sig || RGB.red);
    if (P.leak) items.push(it(grad("var(--flow)", "var(--refuse)"), "redness: how much of the erased direction is back (full red: as much as the unedited model)"), it(`<i class="sw-halo"></i>`, "glowing glass: more than the unedited model"));
    else if (m.sig === RGB.red) items.push(it(grad("var(--flow)", "var(--refuse)"), "redness: strength of the refusal signal"));
    else if (m.kind === "perase") items.push(it(grad("var(--flow)", sc), m.trait ? `colour: strength of the ${m.trait} trait the system prompt sets` : "colour: strength of the trait the system prompt sets (mean over traits)"));
    else if (m.kind === "persona") items.push(it(grad("var(--flow)", sc), m.trait ? `colour: how far the stream points along the ${m.trait} vector` : "colour: how far the stream points along the persona vector (mean over traits)"));
    else if (m === MEAN.kick) items.push(it(grad("var(--flow)", "var(--kick)"), "darkness: how far the kick has knocked the stream off course"));
    const types = new Set([...(this.o.types || []), ...this.evs.map((e) => e.type)]);
    const dropIcon = (c) => `<svg class="sw-svg" viewBox="0 0 12 16" aria-hidden="true"><path d="M6 1C6 1 1.5 7 1.5 10a4.5 4.5 0 0 0 9 0C10.5 7 6 1 6 1z" style="fill:${c}"/></svg>`;
    if (types.has("push")) items.push(it(dropIcon("var(--flow)"), "add: activation-addition steering, a vector against refusal added at one layer"));
    if (types.has("persona")) items.push(it(dropIcon(sc), "add: the persona vector, added at one layer (activation addition)"));
    if (types.has("kick")) items.push(it(dropIcon("var(--kick)"), "kick: a random vector, added at one layer"));
    if (types.has("erase") || P.edited) items.push(it(`<svg class="sw-svg" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.6" style="fill:rgba(72,82,103,.12);stroke:var(--ink-2)" stroke-width="1.4"/><path d="M3.2 5.5h9.6M2 8h12M3.2 10.5h9.6M5.5 3.2v9.6M8 2v12M10.5 3.2v9.6" style="stroke:var(--ink-2)" stroke-width=".8"/></svg>`,
      P.edited ? "filters built into the weights, on every loop" : P.hooks ? "filter: a hook takes the direction out at every layer" : `filter: takes the ${m.kind ? "trait" : "refusal"} direction out at every layer of that loop`));
    if (this.byp && this.o.reinject) items.push(it(`<svg class="sw-svg wide" viewBox="0 0 30 16" aria-hidden="true"><path d="M1 4H29M5 4C9 4 8 13 12 13H24V5" style="fill:none;stroke:var(--ink-3)" stroke-width="2.2"/></svg>`, "shortcut: the entry stack's output, added back before every later loop"));
    if (this.o.arch && this.o.arch.ticks === "sandwich" && !this.o.mini) items.push(
      it(`<svg class="sw-svg wide" viewBox="0 0 30 16" aria-hidden="true"><rect x=".5" y="4" width="29" height="8" rx="4" style="fill:var(--panel-2);stroke:var(--ink-3)" stroke-width=".8"/><ellipse cx="10" cy="8" rx="1.4" ry="4.3" style="fill:none;stroke:var(--ink-2)" stroke-width=".8"/><ellipse cx="13" cy="8" rx="1.4" ry="4.3" style="fill:none;stroke:var(--ink-2)" stroke-width=".8"/><ellipse cx="20" cy="8" rx="1.4" ry="4.3" style="fill:none;stroke:var(--ink-2)" stroke-width=".8"/><ellipse cx="23" cy="8" rx="1.4" ry="4.3" style="fill:none;stroke:var(--ink-2)" stroke-width=".8"/></svg>`, "double rings: a norm on what each layer reads and on what it writes"),
      it(`<svg class="sw-svg wide" viewBox="0 0 30 16" aria-hidden="true"><rect x=".5" y="4" width="29" height="8" rx="4" style="fill:var(--panel-2);stroke:var(--ink-3)" stroke-width=".8"/><ellipse cx="15" cy="8" rx="2.1" ry="6.6" style="fill:none;stroke:var(--ink)" stroke-width="2.4"/></svg>`, "collar: the shared norm between loops"));
    clearEl(this.leg);
    this.leg.append(...items);
  }
  disposeObj(g) {
    g.traverse((ob) => {
      if (ob.geometry) ob.geometry.dispose();
      if (ob.material) (Array.isArray(ob.material) ? ob.material : [ob.material]).forEach((mt) => mt.dispose());
    });
  }
  dispose() {
    GL.views.delete(this);
    for (const ob of this.obs || []) if (ob) ob.disconnect();
    if (this.root) this.disposeObj(this.root);
  }
}
const Track3 = GL ? Stream3D : Track;

/* plans from data. Refusal: the unsteered stream ends as red as the model's refusal rate (peak), and v is the share of
   that refusal a steering removes, so the colour at the end is the refusal rate itself */
const vOf = (c, r) => clamp((c - r) / c, 0, 1);
function vRef(m, mode, key) { const g = grid(m); return vOf(g.clean.r, g[mode][key].r); }
function refusalPlan(m, mode, sel) {
  const n = grid(m).units, ks = [...sel].sort(), vs = {};
  for (let k = 1; k <= n; k++) vs[k] = vRef(m, mode, String(k));
  return { events: ks.map((k) => ({ k, type: mode })), vExit: ks.length ? vRef(m, mode, ks.join("")) : 0, bite: vs[n], vSingle: vs, peak: grid(m).clean.r / 100 };
}
function kickPlan(m, unit, eps, every) {
  const n = MODELS[m].loops, c = kickClean(m).acc, r = kickFind(m, unit, eps, every);
  if (!r) return null;
  const v = (x) => (x ? clamp(1 - x.acc / c, 0, 1) : 0), vs = {};
  for (let k = 1; k <= n; k++) vs[k] = v(kickFind(m, k, eps));
  return { events: (every ? Array.from({ length: n }, (_, i) => i + 1) : [unit]).map((k) => ({ k, type: "kick" })), vExit: v(r), bite: vs[n], vSingle: vs, every };
}
function personaPlan(m, trait, units) {
  const T = D.persona[m].traits[trait], n = MODELS[m].loops, allU = Array.from({ length: n }, (_, i) => i).join(",");
  const den = Math.max(3, T.by_units[allU].score - T.clean);
  const v = (u) => (T.by_units[u] ? clamp((T.by_units[u].score - T.clean) / den, 0, 1) : 0), vs = {};
  for (let k = 1; k <= n; k++) vs[k] = v(String(k - 1));
  return { events: units.split(",").map((x) => ({ k: +x + 1, type: "persona" })), vExit: v(units), bite: vs[n], vSingle: vs };
}

/* ------------------------------------------------------------------ charts */
function barPath(x, y, w, hgt, r = 4) {
  if (hgt <= 0.5) return `M${x},${y + hgt}h${w}`;
  r = Math.min(r, w / 2, hgt);
  return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
}
function barChart(el, { title, items, max = 100, min = 0, refs = [], fmt = (v) => `${r0(v)}%`, w = 230, ht = 190, subtitle, padR = 46 }) {
  const svg = s("svg", { viewBox: `0 0 ${w} ${ht}`, role: "img", "aria-label": title }, el);
  const top = subtitle ? 44 : 34, bot = ht - 26, left = 6, right = w - 6;
  const y = (v) => bot - ((clamp(v, min, max) - min) / (max - min)) * (bot - top);
  s("text", { x: left, y: 15, class: "t-title", text: title }, svg);
  if (subtitle) s("text", { x: left, y: 30, text: subtitle, "font-size": 10 }, svg);
  s("line", { x1: left, x2: right, y1: y(min), y2: y(min), stroke: "var(--line)", "stroke-width": 1 }, svg);
  const n = items.length, gap = 8, bw = Math.min(42, (right - left - padR - gap * (n - 1)) / n);
  const x0 = left + 4;
  items.forEach((it, i) => {
    const x = x0 + i * (bw + gap);
    if (it.value == null) { /* not run yet: an empty dashed slot */
      s("rect", { x: x + 0.5, y: top + 18, width: bw - 1, height: y(min) - top - 18, rx: 4, fill: "none", stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
      s("text", { x: x + bw / 2, y: top + 12, "text-anchor": "middle", text: "not run", "font-size": 9 }, svg);
      s("text", { x: x + bw / 2, y: bot + 15, "text-anchor": "middle", text: it.label, "font-size": 10.5 }, svg);
      const hit = s("rect", { x: x - gap / 2, y: top - 10, width: bw + gap, height: bot - top + 26, fill: "transparent" }, svg);
      if (it.tip) addTip(hit, it.tip);
      return;
    }
    const yv = y(Math.max(it.value, min));
    const p = s("path", { d: barPath(x, yv, bw, y(min) - yv), fill: it.color || "var(--ink-2)", opacity: it.faded ? 0.35 : 1 }, svg);
    if (it.hatch) p.setAttribute("fill-opacity", 0.55);
    if (it.lo != null) s("line", { x1: x + bw / 2, x2: x + bw / 2, y1: y(it.lo), y2: y(it.hi), stroke: "var(--ink)", "stroke-width": 1.2, opacity: 0.5 }, svg);
    s("text", { x: x + bw / 2, y: Math.min(yv, it.hi != null ? y(it.hi) : yv) - 5, "text-anchor": "middle", class: "t-strong", text: fmt(it.value), "font-size": 11 }, svg);
    s("text", { x: x + bw / 2, y: bot + 15, "text-anchor": "middle", text: it.label, "font-size": 10.5 }, svg);
    const hit = s("rect", { x: x - gap / 2, y: top - 10, width: bw + gap, height: bot - top + 26, fill: "transparent" }, svg);
    if (it.tip) addTip(hit, it.tip);
  });
  refs.forEach((rf) => {
    const yy = y(rf.value);
    s("line", { x1: left, x2: right - 2, y1: yy, y2: yy, stroke: "var(--ink-2)", "stroke-width": 1.2, "stroke-dasharray": rf.dash || "5 4" }, svg);
    s("text", { x: right, y: yy - 4, "text-anchor": "end", text: rf.label, "font-size": 9.5 }, svg);
  });
  return svg;
}

/* ------------------------------------------------------------------ data helpers */
const grid = (m) => D.grids[m];
function rec(m, mode, units) {
  const g = grid(m);
  if (!units.length) return g.clean;
  return g[mode][[...units].sort().join("")];
}
function shareKept(m, mode, k) {
  const g = grid(m), all = g[mode][allKey(g.units)].r, c = g.clean.r;
  return (c - g[mode][String(k)].r) / (c - all);
}
const meanOf = (a) => a.reduce((x, y) => x + y, 0) / a.length;
function personaGain(m, units) {
  const T = D.persona[m].traits;
  return meanOf(Object.values(T).map((t) => (t.by_units[units] ? t.by_units[units].gain : NaN)).filter((v) => !Number.isNaN(v)));
}
function personaUnits(m) {
  const any = Object.values(D.persona[m].traits)[0];
  return Object.keys(any.by_units).filter((u) => u !== "nan" && u !== "");
}
/* the persona traits, in one order for every model, and why a model lacks one: no vector (it will not play the trait
   even when told to) or, for steering, no coherent strength that moves it by 10 points on every loop at once */
const PTRAITS = ["optimistic", "humorous", "sycophantic", "hallucinating", "apathetic", "impolite", "evil"];
const sgn = (v) => `${v >= 0 ? "+" : ""}${(+v).toFixed(1)}`;
const andList = (a) => (a.length < 2 ? a.join("") : `${a.slice(0, -1).join(", ")} and ${a[a.length - 1]}`);
function missingNote(m, missing, P, erase = false) {
  if (!missing.length) return "";
  const name = m === "dory4" ? "Dory" : MODELS[m].label.replace(" (dense)", "");
  const novec = missing.filter((t) => !P.vector.includes(t)), weak = missing.filter((t) => P.vector.includes(t)), parts = [];
  if (weak.length) parts.push(`${andList(weak)}, which no strength that keeps replies coherent moves by 10 points even on every ${MODELS[m].straight ? "quarter" : "loop"} (best ${andList(weak.map((t) => sgn(P.tune[t].best)))})`);
  if (novec.length) parts.push(`${andList(novec)}, which ${name} will not play even when ${erase ? "its system prompt asks for it" : "told to"}, so there is no vector${erase ? " to erase" : ""}`);
  return `* Not shown for ${name}: ${parts.join("; ")}.`;
}
function kickRows(m) { return D.kicks[m].rows; }
function kickClean(m) { return kickRows(m).find((r) => r.cand === "none"); }
const KLAYER = { qwen: [4, 13, 22, 31] };
function kickFind(m, unit, eps, every = false) {
  return kickRows(m).find((r) => r.site && r.site.eps === eps && !r.indep &&
    (every ? r.every : !r.every && (m === "qwen" ? r.site.layer === KLAYER.qwen[unit - 1] : r.site.loop === unit - 1)));
}

/* ================================================================== 0. hero */
(function hero() {
  const ex = D.extra.dory.erase, clean = grid("dory").clean;
  const tr = new Track3($("hero-track"), { model: "dory", loops: 2, maxLaps: 4, trained: 2, meaning: MEAN.refusal, types: ["erase"], label: "Dory: one harmful prompt through its loops" });
  const bite = vRef("dory", "erase", "2"), c = clean.r, peak = c / 100;
  const both = [{ k: 1, type: "erase" }, { k: 2, type: "erase" }];
  const steps = [
    { t: "No steering", v: clean, loops: 2, plan: { peak } },
    { t: "Steer both loops", v: ex.at_trained, loops: 2, plan: { events: both, vExit: vOf(c, ex.at_trained.r), bite, peak } },
    { t: "Add 2 extra loops", v: ex.at_extra, loops: 4, plan: { events: both, vExit: vOf(c, ex.at_extra.r), bite, peak } },
  ];
  let auto = true, last = 0;
  /* each tile: its name, its refusal rate as a big counting number and the 95% interval, all in blue and red mixed by its refusal rate */
  const tiles = steps.map((st, i) => {
    const col = rgbCss(mixRGB(RGB.flow, RGB.red, st.v.r / 100)), led = h("div", { class: "num" });
    const b = h("button", { class: "step", type: "button", "aria-pressed": String(i === 0), style: `--c:${col}`, "aria-label": `${st.t}: ${pct(st.v.r)} refusal` },
      h("span", { class: "k" }, st.t), led, h("span", { class: "ci" }, `95% CI ${r0(st.v.lo)}–${pct(st.v.hi)}`));
    b.addEventListener("click", () => { auto = false; run(i); });
    $("hero-steps").append(b);
    const board = new Board(led, { color: col, bare: true });
    board.set(st.v.r, { animate: false });
    return { b, board };
  });
  $("hero-n").textContent = [...new Set(steps.map((st) => st.v.n))].join(" or ");
  /* while the liquid runs, the tile counts from the previous run's rate (from 0 on a replay) to its own: it starts when
     the flow starts and lands when the flow reaches the head */
  function run(i, then) {
    const st = steps[i], bd = tiles[i].board, from = last === i ? 0 : steps[last].v.r, to = st.v.r;
    tiles.forEach((x, j) => { x.b.setAttribute("aria-pressed", String(i === j)); if (j !== i) x.board.set(steps[j].v.r, { animate: false }); });
    bd.set(from, { animate: false });
    const go = (placed) => tr.play(st.plan, { placed, onProgress: (f) => bd.set(lerp(from, to, f), { animate: false }),
      onDone: () => { bd.set(to, { animate: false }); last = i; if (then) setTimeout(() => auto && then(), 1600); } });
    if (st.loops > tr.n) { /* the same model, run for 2 more loops: the filters drop in while the tube grows */
      tr.build(st.loops, 2); tr.apply(st.plan); tr.morph(2, { onDone: () => go(true) });
      if (!RM) tr.insertDevices();
    }
    else if (st.loops < tr.n) { tr.apply({ peak }); tr.morph(st.loops, { reverse: true, dur: 800, onDone: () => { tr.build(st.loops, 2); go(); } }); }
    else go(false);
  }
  tr.show(steps[0].plan);
  /* the tiles sit right under the drawing, above its legend, so the two stay in view together on a phone */
  const heroLeg = $("hero-track").querySelector(".track-legend, .legend");
  if (heroLeg) heroLeg.before($("hero-steps"));
  src(`Hero: data/runs/${MODELS.dory.dir}/generations/grid_test{,_b} (2 loops) and grid_test4{,_b} (4 loops): none, ablate_hh_eoi_t0.l26_apply0-1`);
  if (!RM) setTimeout(() => auto && run(1, () => run(2)), 1400);
  $("gen-stamp").textContent = `Data exported ${D.generated}`;
})();

/* ================================================================== 1. model cards */
(function cards() {
  /* each card: the drawing, how one layer updates the stream (and what happens between loops), and what differs */
  const info = [
    ["dory", "Dory 1.0 4B", "NVIDIA · 4B · 16 in, 40 × 2 loops, 16 out", { ticks: "renorm", gates: true, reinject: true },
      [["each sublayer", "h \\leftarrow \\bar n\\big(\\hat h + \\alpha \\odot (\\bar n(f(\\hat h)) - \\hat h)\\big),\\ \\hat h = \\bar n(h)"], ["loop start", "h \\leftarrow \\bar n(h + x)"]],
      "No gains anywhere: the stream is kept at unit length, n̄(v) = v / ‖v‖ (nGPT, the dark rings), and each sublayer moves it part of the way towards its own output, at a learned rate α per dimension. An entry stack runs once and its output x is added back before every loop (the shortcut); an exit stack finishes."],
    ["nanbeige", "Nanbeige4.2-3B", "Nanbeige · 3B · 22 layers × 2 loops", { ticks: "pre", gates: true },
      [["each layer", "h \\leftarrow h + f(\\mathrm{N}(h))"], ["between loops", "h \\leftarrow \\mathrm{N}(h)"]],
      "Standard pre-norm layers: each reads a normalised copy of the stream (the thin rings) and adds its output back as it is, with no norm on the write. The shared final norm also runs between the two loops (the collar)."],
    ["ouro", "Ouro-1.4B-Thinking", "ByteDance · 1.4B · 24 layers × 4 loops", { ticks: "sandwich", gates: true, exits: true },
      [["each layer", "h \\leftarrow h + \\mathrm{N}_{\\text{out}}\\big(f(\\mathrm{N}_{\\text{in}}(h))\\big)"], ["between loops", "h \\leftarrow \\mathrm{N}(h)"], ["exit gate", "\\lambda_t = \\sigma(w^{\\top} h_t)"]],
      "Sandwich norms: every layer normalises what it reads and also what it writes (the double rings), so each write passes through a learned gain before it reaches the stream. The shared final norm runs after every loop (the collars). After each loop an exit gate (the valves) can stop and send the stream down its own route to the LM head; training scores every loop's prediction, weighted by the gate. We always run all four loops, so the dashed routes stay empty, and after loop 4 the stream goes straight into the head."],
    ["qwen", "Qwen3-8B", "Alibaba · 8B · 36 layers, once", { ticks: "pre" },
      [["each layer", "h \\leftarrow h + f(\\mathrm{N}(h))"]],
      "The dense control: 36 different layers, each run once, with the same pre-norm layers as Nanbeige. We cut it into four quarters of 9 layers to line it up with a four-loop model."],
    ["inkling", "Inkling 2 Mini", "Thinking Machines · 26B MoE · 6 in, 12 × 3 passes, 6 out", { moe: true, plugs: true, msgs: true },
      [["each layer", "b \\leftarrow b + f\\big(\\mathrm{N}(\\textstyle\\sum_j a_j\\, \\mathrm{N}(b_j))\\big)"], ["MoE layer", "f(x) = \\textstyle\\sum_{e \\in \\mathrm{top}\\,8} g_e(x)\\, E_e(x)"],
        ["pass start", "b \\leftarrow b_{\\text{prelude}}"], ["message", "u \\leftarrow (1 - \\tanh \\phi) \\odot u + \\tanh \\gamma \\odot m"]],
      "A looped mixture of experts: each layer's MLP has 128 experts, and a router sends every token to 8 of them (3.4B of 26B weights active; the rings, where each bead stands for 8 experts). There is no single residual stream: every sublayer reads a learned mix a of the finished 6-layer blocks b_j and the running block b. The prelude runs once, the core three times, the coda once. Each pass restarts from the prelude's state (the shortcut; the plug is where the pass before stops) and hears that pass only through gated messages m blended into every core sublayer's input u (the bridges). The router sees a new state on each pass, so a layer can pick other experts each time; which bead lights is a sketch. Its results are not in the figures yet."],
  ];
  for (const [m, name, spec, arch, eqs, desc] of info) {
    const box = h("div", { class: "trackbox mini" });
    const eq = h("div", { class: "eqs" });
    eq.innerHTML = eqs.map(([k, tex]) => `<div><span>${k}</span>${mathHtml(tex, false)}</div>`).join("");
    const card = h("div", { class: "card" }, h("h3", {}, h("span", { class: "dotsw", style: `background:${MODELS[m].color}` }), name),
      h("div", { class: "spec" }, spec), box, eq, h("p", {}, desc));
    $("model-cards").append(card);
    const t = new Track3(box, { model: m, loops: MODELS[m].loops, mini: true, meaning: null, arch,
      label: `${name}: ${m === "qwen" ? "one straight pass" : arch.msgs ? `${MODELS[m].loops} passes, each restarting from the prelude and reading the pass before through gated messages; 8 of 128 experts per layer` : `${MODELS[m].loops} loops`}${arch.reinject ? ", input re-added before every loop" : arch.gates ? ", a shared norm between loops" : ""}` });
    t.idle();
  }
  /* key to the marks on the cards */
  const key = $("norm-key"), sw = (draw, w = 34) => { const v = s("svg", { viewBox: `0 0 ${w} 16`, width: w, height: 16, "aria-hidden": "true" }); draw(v); return v; };
  const tube = (v) => s("rect", { x: 0.5, y: 4, width: 33, height: 8, rx: 4, fill: "var(--panel-2)", stroke: "var(--ink-3)", "stroke-width": 0.8 }, v);
  const ring = (v, x, ry, col, w) => s("ellipse", { cx: x, cy: 8, rx: ry * 0.32, ry, fill: "none", stroke: col, "stroke-width": w }, v);
  const items = [
    [(v) => { tube(v); for (const x of [8, 17, 26]) ring(v, x, 4.3, "var(--ink-3)", 0.8); }, "thin ring: each layer reads a normalised copy"],
    [(v) => { tube(v); for (const x of [8, 17, 26]) for (const dx of [-1.3, 1.3]) ring(v, x + dx, 4.3, "var(--ink-2)", 0.8); }, "double ring: reads and writes are normalised"],
    [(v) => { tube(v); for (const x of [8, 17, 26]) ring(v, x, 5.6, "var(--ink)", 1.1); }, "dark ring: the stream itself is renormalised"],
    [(v) => { tube(v); ring(v, 17, 6.6, "var(--ink)", 2.6); }, "collar: a norm between loops"],
    [(v) => { s("path", { d: "M1 4H33M5 4C9 4 8 13 12 13H26V5", fill: "none", stroke: "var(--ink-3)", "stroke-width": 2.2 }, v); }, "shortcut: the first stack's output, added back each loop (Dory) or restarting each pass (Inkling)"],
    [(v) => { s("rect", { x: 0.5, y: 0.5, width: 26, height: 5, rx: 2.5, fill: "var(--panel-2)", stroke: "var(--ink-3)", "stroke-width": 0.8 }, v); s("circle", { cx: 30, cy: 3, r: 3, fill: "var(--panel-2)", stroke: "var(--ink-3)", "stroke-width": 1 }, v);
      s("line", { x1: 9, x2: 9, y1: 5.5, y2: 7.2, stroke: "var(--ink-2)", "stroke-width": 1.4 }, v); s("circle", { cx: 9, cy: 9, r: 1.9, fill: "none", stroke: "var(--ink-2)", "stroke-width": 1 }, v);
      s("path", { d: "M9 11.2V14.5H30V6.5", fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.1, "stroke-dasharray": "2 1.4" }, v); }, "exit gate: a valve that can send a loop's output on its own route to the LM head"],
    [(v) => { tube(v); for (let k = 0; k < 8; k++) { const a = (2 * Math.PI * k) / 8; s("circle", { cx: 17 + 3.2 * Math.cos(a), cy: 8 + 6.6 * Math.sin(a), r: k === 2 ? 1.7 : 1.05, fill: k === 2 ? "var(--inkling)" : "var(--ink-3)", "fill-opacity": k === 2 ? 1 : 0.6 }, v); } }, "expert ring: each bead is 8 of a layer's 128 experts; the lit one is where the token goes"],
    [(v) => { tube(v); s("rect", { x: 15.6, y: 1.5, width: 2.8, height: 13, rx: 1.2, fill: "var(--ink)" }, v); }, "plug: the stream stops here, and the next pass restarts from the prelude"],
    [(v) => { s("path", { d: "M3 14Q17 -4 31 14", fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.2, "stroke-dasharray": "2.4 1.6" }, v); s("circle", { cx: 17, cy: 5, r: 2.6, fill: "var(--panel)", stroke: "var(--ink-2)", "stroke-width": 1 }, v); }, "bridge: a gated message from one pass into the next"],
  ];
  for (const [draw, text] of items) key.append(h("span", {}, sw(draw), text));
  const def = h("p", { class: "norm-def" });
  def.innerHTML = `${mathHtml("\\mathrm{N}(v) = g \\odot v / \\mathrm{rms}(v)", false)} is RMSNorm, with a learned gain <i>g</i> for every dimension; <i>f</i> is an attention or MLP sublayer. The gains matter later: they are how a weight edit leaks.`;
  key.after(def);
})();

/* ================================================================== 2. switchboard */
(function switchboard() {
  const st = { m: "dory", mode: "erase", sel: new Set([1]) };
  let tr, units = [];
  const board = new Board($("sb-board"), { title: "refusal rate", color: "var(--refuse)" });
  const EX = D.examples, exIds = Object.keys(EX.prompts).map(Number);
  const exNames = { 20: "An insult", 25: "A spam bot", 94: "A biased rubric" };
  st.ex = exIds[0];
  const SAFE = new Set(EX.safe);
  const dots = new Split($("sb-dots"), { marks: EX.safe, pickAll: true, onPick: (i) => { st.ex = i; exChips.forEach((b, j) => b.setAttribute("aria-pressed", String(exIds[j] === i))); showEx(); },
    tipFn: (i, v) => (v == null ? `Prompt ${i + 1}: no data` : `${SAFE.has(i) ? `<b>“${escHtml(EX.all[i])}”</b><br>` : `Prompt ${i + 1}<br>`}refused in ${Math.round(9 * v)} of its 9 samples<br><i>click to ${SAFE.has(i) ? "read the reply" : "see the prompt"}</i>`) });
  chips($("sb-models"), ORDER5.map((m) => ({ id: m, label: MODELS[m].label, color: MODELS[m].color })),
    { value: st.m, swatch: true, onChange: (m) => { st.m = m; st.sel = new Set([1]); build(); } });
  chips($("sb-mode"), [{ id: "erase", label: "Erase" }, { id: "push", label: "Add" }], { value: "erase", onChange: (v) => { st.mode = v; update(true); } });
  /* one reply per setting for a few prompts (a sample that agrees with the prompt's majority verdict) */
  const exChips = chips($("sb-ex-chips"), exIds.map((i) => ({ id: i, label: exNames[i] || `Prompt ${i + 1}` })), { value: st.ex, onChange: (i) => { st.ex = i; showEx(); } });
  let exSeq = 0;
  function showEx() {
    const sel = [...st.sel].sort(), r = sel.length ? rec(st.m, st.mode, sel) : grid(st.m).clean;
    const where = sel.length ? `${st.mode === "erase" ? "erased" : "added"} on ${sel.map((k) => unitName(st.m, k)).join(" + ")}` : "no steering";
    dots.select(st.ex);
    showRefusalReply($("sb-ex"), { run: st.m, cand: r.cand, i: st.ex, v: r.dots ? r.dots[st.ex] : null, who: `${MODELS[st.m].label} · ${where}`, seq: ++exSeq, cur: () => exSeq });
  }
  chips($("sb-presets"), [{ id: "first", label: "First" }, { id: "last", label: "Last" }, { id: "all", label: "All" }, { id: "none", label: "None" }],
    { multi: true, onChange: (v, b) => {
      const n = grid(st.m).units;
      st.sel = new Set(v === "first" ? [1] : v === "last" ? [n] : v === "all" ? Array.from({ length: n }, (_, i) => i + 1) : []);
      b.setAttribute("aria-pressed", "false"); syncUnits(); update(true);
    } });
  function syncUnits() { units.forEach((b, i) => b.setAttribute("aria-pressed", String(st.sel.has(i + 1)))); }
  function build() {
    const M = MODELS[st.m], g = grid(st.m);
    tr = new Track3($("sb-track"), { model: st.m, loops: g.units, trained: st.m === "dory4" ? 2 : g.units, meaning: MEAN.refusal, types: ["erase", "push"], label: `${M.full}: one harmful prompt through the model` });
    $("sb-units-label").textContent = `Steer on these ${M.straight ? "quarters" : "loops"}`;
    const seg = $("sb-units"); seg.innerHTML = "";
    units = Array.from({ length: g.units }, (_, i) => {
      const k = i + 1;
      const small = M.straight ? `layers ${QLAYERS[i]}` : (st.m === "dory4" && k > 2 ? "extra loop" : `${M.layers} ${M.ramps ? "sublayers" : "layers"}`);
      const b = h("button", { type: "button", "aria-pressed": "false" }, h("span", {}, unitName(st.m, k)), h("small", {}, small));
      b.addEventListener("click", () => { st.sel.has(k) ? st.sel.delete(k) : st.sel.add(k); syncUnits(); update(true); });
      seg.append(b);
      return b;
    });
    syncUnits(); update(false);
  }
  function update(animate) {
    const M = MODELS[st.m], g = grid(st.m), sel = [...st.sel].sort();
    const r = rec(st.m, st.mode, sel), all = g[st.mode][allKey(g.units)], c = g.clean;
    const done = () => {
      board.set(r.r, { title: `${M.label} · <b>refusal rate</b>`, right: `unsteered ${pct(c.r)}`,
        sub: `95% CI ${pct(r.lo)}–${pct(r.hi)}`, subRight: `${r.n} prompts` });
      dots.set(r.dots, [RGB.flow, RGB.red], `Refusal per prompt, mean ${pct(r.r)}`);
      showEx();
    };
    const plan = refusalPlan(st.m, st.mode, sel);
    if (animate) { board.set(null, { sub: "running…", subRight: "" }); tr.play(plan, { onDone: done }); }
    else { tr.show(plan); done(); }
    const kept = sel.length ? ` That is ${r0(100 * (c.r - r.r) / (c.r - all.r))}% of the effect of steering every ${M.straight ? "quarter" : "loop"} (${pct(all.r, 1)}).` : "";
    const where = sel.length ? `${st.mode === "erase" ? "erasing" : "adding"} on ${sel.map((k) => unitName(st.m, k)).join(" + ")}` : "no steering";
    const late = M.straight && sel.length === 1 && (sel[0] === "4" || sel[0] === 4) ? " Steering Qwen's last quarter barely moves it: the answer is already set." : "";
    $("sb-cap").textContent = `${M.full}, ${where}: ${pct(r.r, 1)} refusal.${kept}${late}` + (st.mode === "push" ? ` Add: ${g.push_note}.` : "") +
      (st.m === "dory4" ? " Dory was trained with 2 loops; this run uses 4." : "") +
      (st.mode === "push" && (st.m === "ouro" || st.m === "nanbeige") ? " In Ouro and Nanbeige an addition on one loop is weak wherever it lands; additions add up across loops." : "");
  }
  build();
  SOURCES.push("Switchboard: data/runs/<model>/generations/grid_test{,_b} (Dory ×4: grid_test4{,_b}), one file per loop subset");

  /* looped vs dense: where in the forward pass one stretch of erasing does the most */
  contrastChart($("contrast-chart"));
  for (const m of ORDER5) {
    const g = grid(m), box = h("div");
    $("single-multiples").append(box);
    barChart(box, { title: MODELS[m].label, subtitle: m === "dory4" ? "trained with 2 loops" : (m === "qwen" ? "quarters of 9 layers" : null),
      items: Array.from({ length: g.units }, (_, i) => {
        const r = g.erase[String(i + 1)];
        return { label: unitShort(m, i + 1), value: r.r, lo: r.lo, hi: r.hi, color: MODELS[m].color,
          tip: `${MODELS[m].full}<br>erase on ${unitName(m, i + 1)} only: <b>${pct(r.r, 1)}</b><br>95% CI ${pct(r.lo, 1)}–${pct(r.hi, 1)} · ${r.n} prompts` };
      }),
      refs: [{ value: g.clean.r, label: `none ${pct(g.clean.r)}` }, { value: g.erase[allKey(g.units)].r, label: `all ${pct(g.erase[allKey(g.units)].r)}`, dash: "2 3" }] });
  }
  $("d4-123").textContent = pct(grid("dory4").erase["123"].r);
  $("d4-4").textContent = pct(grid("dory4").erase["4"].r);

  /* flip the sign: add the refusal direction on harmless prompts; every quarter of Qwen */
  const P = D.plus_harmless;
  const pbox = h("div", { class: "multiples plus" });
  $("plus-chart").append(pbox);
  const plusUnits = {
    dory: [["first", "L1", "loop 1"], ["last", "L2", "loop 2"], ["all", "all", "both loops"]],
    dory4: [["first", "L1", "loop 1"], ["l2", "L2", "loop 2"], ["l3", "L3", "loop 3"], ["last", "L4", "loop 4"], ["all", "all", "every loop"]],
    nanbeige: [["first", "L1", "loop 1"], ["last", "L2", "loop 2"], ["all", "all", "both loops"]],
    ouro: [["first", "L1", "loop 1"], ["l2", "L2", "loop 2"], ["l3", "L3", "loop 3"], ["last", "L4", "loop 4"], ["all", "all", "every loop"]],
    qwen: [["first", "Q1", "Q1"], ["q2", "Q2", "Q2"], ["q3", "Q3", "Q3"], ["last", "Q4", "Q4"], ["all", "all", "all four quarters"]],
  };
  /* each model at the strength tuned for refusal vs coherence (E15) where those runs exist, else E14's 4x */
  const TP = (m) => P[m].tuned || P[m];
  const extra = (m, k) => {
    const T = TP(m), parts = [];
    if (T.coh && T.coh[k] != null) parts.push(`coherence ${r0(T.coh[k])} (unsteered ${r0(T.coh.clean)})`);
    if (T.unclosed && T.unclosed[k] != null) parts.push(`${pct(T.unclosed[k])} of reasoning traces never close (unsteered ${pct(T.unclosed.clean)})`);
    return parts.length ? `<br>${parts.join("<br>")}` : "";
  };
  /* one y scale for every panel, just above the largest bar (the tuned additions refuse far less than 100%) */
  const PMAX = Math.max(20, Math.ceil(Math.max(...Object.entries(plusUnits).flatMap(([m, us]) =>
    [...us.map(([k]) => TP(m)[k]), TP(m).random, TP(m).clean].filter((v) => v != null))) / 10) * 10);
  for (const [m, us] of Object.entries(plusUnits)) {
    const b = h("div", { class: m === "qwen" ? "span2" : null }); pbox.append(b);
    const T = TP(m), every = m === "qwen" ? "all four quarters" : "every loop";
    const sub = [m === "dory4" ? "trained with 2" : m === "qwen" ? "start of each quarter" : null, T.mult ? `strength ${T.mult}×` : null].filter(Boolean).join(" · ");
    /* Qwen spans two columns on wide screens; on a phone (one column) a 470-wide panel would shrink to half size */
    barChart(b, { title: MODELS[m].label, subtitle: sub || null, w: m === "qwen" ? (window.innerWidth < 640 ? 300 : 470) : 230, max: PMAX, padR: us.length >= 5 && m !== "qwen" ? 60 : 46,
      items: us.map(([k, lab, name]) => ({ label: lab, value: T[k], color: MODELS[m].color, faded: k === "all",
        tip: T[k] == null ? `${MODELS[m].full}: addition on ${name} not run yet` : `${MODELS[m].full}: refusal direction added on ${name} → ${pct(T[k], 1)} of harmless prompts refused${extra(m, k)}` }))
        .concat([{ label: "rand", value: T.random, color: "var(--neutral)",
          tip: `${MODELS[m].full}: a random direction of the same size added on ${every} → ${pct(T.random, 1)} of harmless prompts “refused”${extra(m, "random")}` }]),
      refs: [{ value: T.clean, label: `none ${pct(T.clean)}` }] });
  }
  src("Flip the sign: data/runs/<model>/generations/plus_harmless (Dory ×4: plus_harmless4), 100 harmless test prompts, at the strength tuned per model on 50 held-out harmless prompts (analysis/plus_tune.json, scripts/plus_tune.py; coherence judged by gpt-oss-20b); random direction at the same strength on every loop / quarter");
  plusTune($("plus-tune"), P);
  plusText(P, TP);
})();

/* the sentences under "flip the sign" that quote Ouro's and Qwen's bars */
function plusText(P, TP) {
  const q = TP("qwen"), o = TP("ouro"), n = TP("nanbeige"), d = TP("dory");
  /* the sentences follow the data: whether the last loop alone gets most of the every-loop effect, and whether Qwen's
     middle quarters stand out */
  $("plus-dory").textContent = d.last >= 0.6 * d.all
    ? `Dory's last loop does nearly all of it: the addition there alone makes it refuse ${pct(d.last)} of harmless prompts, against ${pct(d.all)} on both loops, ${pct(d.first)} on the first loop and ${pct(d.clean)} unsteered.`
    : `Dory refuses ${pct(d.last)} of harmless prompts with the addition on its last loop, ${pct(d.all)} on both loops and ${pct(d.first)} on the first (${pct(d.clean)} unsteered).`;
  const most = (t) => t.last >= 0.6 * t.all, d4 = TP("dory4");
  const mid = (t) => (t.l2 != null && t.l3 != null ? [t.l2, t.l3] : null);
  $("plus-dory").textContent += mid(d4) ? ` Run with four loops, Dory refuses ${pct(d4.last)} with the addition on loop 4 alone, against ${pct(d4.first)}, ${pct(d4.l2)} and ${pct(d4.l3)} on loops 1 to 3.` : "";
  $("plus-ouro").textContent = (!most(n) && !most(o)
    ? `Nanbeige and Ouro need it on every loop: ${pct(n.all)} and ${pct(o.all)}, against ${pct(n.last)} and ${pct(o.last)} with it on the last loop alone.`
    : `Nanbeige refuses ${pct(n.last)} with it on the last loop and ${pct(n.all)} on every loop; Ouro ${pct(o.last)} and ${pct(o.all)}.`)
    + (!mid(o) ? "" : Math.max(o.l2, o.l3) <= o.first + 2 ? ` Ouro's middle loops on their own do no more than its first (${pct(o.l2)} and ${pct(o.l3)}).` : ` With it on loop 2 or loop 3 alone, Ouro refuses ${pct(o.l2)} and ${pct(o.l3)}.`);
  const qs = [q.first, q.q2, q.q3, q.last], top = Math.max(...qs), bi = qs.indexOf(top), ends = Math.max(q.first, q.last);
  const midTop = (bi === 1 || bi === 2) && top >= ends + 3;
  $("plus-qwen").textContent = q.q2 == null ? ""
    : top < 3
      ? `Qwen hardly moves at a strength that keeps it coherent: ${pct(q.all)} with the addition in all four quarters, at most ${pct(top)} with one.`
      : `Qwen refuses ${pct(q.all)} with the addition in all four quarters and at most ${pct(top)} with it in one, Q${bi + 1}`
        + (midTop ? `, a middle quarter as with erasing (Q1 ${pct(q.first)}, Q4 ${pct(q.last)}).` : ".");
}

/* choosing the strength of +u: refusal of harmless prompts and judged coherence at each strength tried (every loop at once,
   50 held-out prompts); the chosen strength is ringed, strengths that break the rule are hollow */
function plusTune(el, P) {
  if (!el) return;
  el.innerHTML = "";
  const ms = ["dory", "nanbeige", "ouro", "qwen"].filter((m) => P[m].tune);
  if (!ms.length) { el.closest(".fig").hidden = true; return; }
  const box = h("div", { class: "multiples" }); el.append(box);
  for (const m of ms) {
    const U = P[m].tune, rows = U.rows, W = 230, H = 196, mL = 30, mR = 10, top = 40, bot = H - 30;
    /* strengths on a log scale (the grid doubles from 0.5x to 4x, with extra points inside the coherence cliff);
       only the doubling grid is labelled */
    const xs = rows.map((r) => r.mult), l0 = Math.log2(xs[0]), l1 = Math.log2(xs[xs.length - 1]);
    const X = (i) => mL + ((W - mL - mR) * (Math.log2(xs[i]) - l0)) / (l1 - l0), Y = (v) => bot - ((bot - top) * clamp(v, 0, 100)) / 100;
    const LAB = [0.25, 0.5, 1, 1.5, 2, 3, 4];
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${MODELS[m].label}: refusal and coherence at each strength` }, h("div", {}));
    box.append(svg.parentNode);
    s("text", { x: 0, y: 15, class: "t-title", text: MODELS[m].label }, svg);
    s("text", { x: 0, y: 29, text: `chosen: ${U.mult}× ${U.any_ok ? "" : "(none passed; the weakest)"}`, "font-size": 10 }, svg);
    for (const v of [0, 50, 100]) {
      s("line", { x1: mL, x2: W - mR, y1: Y(v), y2: Y(v), stroke: "var(--line)" }, svg);
      s("text", { x: mL - 5, y: Y(v) + 3.5, "text-anchor": "end", text: String(v), "font-size": 9.5 }, svg);
    }
    const thr = U.clean.coherence - 10;
    s("line", { x1: mL, x2: W - mR, y1: Y(thr), y2: Y(thr), stroke: "var(--ink-3)", "stroke-dasharray": "3 3" }, svg);
    xs.forEach((x, i) => { if (LAB.includes(x)) s("text", { x: X(i), y: bot + 14, "text-anchor": "middle", text: `${x}×`, "font-size": 9.5 }, svg); });
    const line = (k, col, w) => s("polyline", { points: rows.map((r, i) => `${X(i).toFixed(1)},${Y(r[k]).toFixed(1)}`).join(" "), fill: "none", stroke: col, "stroke-width": w, "stroke-linejoin": "round" }, svg);
    line("coherence", "var(--ink-2)", 1.6); line("refusal", "var(--refuse)", 2.2);
    rows.forEach((r, i) => {
      const chosen = r.c === U.c;
      for (const [k, col] of [["coherence", "var(--ink-2)"], ["refusal", "var(--refuse)"]])
        s("circle", { cx: X(i), cy: Y(r[k]), r: 3.4, fill: r.ok ? col : "var(--panel)", stroke: col, "stroke-width": 1.4 }, svg);
      if (chosen) s("rect", { x: X(i) - 6.5, y: top - 6, width: 13, height: bot - top + 12, rx: 6.5, fill: "none", stroke: "var(--steer)", "stroke-width": 1.6 }, svg);
      /* hover band: halfway to each neighbour */
      const a = i ? (X(i - 1) + X(i)) / 2 : X(i) - 12, b = i < rows.length - 1 ? (X(i) + X(i + 1)) / 2 : X(i) + 12;
      addTip(s("rect", { x: a, y: top - 8, width: b - a, height: bot - top + 24, fill: "transparent" }, svg),
        `${MODELS[m].full}, +u on ${m === "qwen" ? "all four quarters" : "every loop"} at ${r.mult}× (c = ${r.c})<br>refusal ${pct(r.refusal)} of harmless prompts (unsteered ${pct(U.clean.refusal)})<br>coherence ${r0(r.coherence)} (unsteered ${r0(U.clean.coherence)})<br>never closes: ${pct(r.capped)} (unsteered ${pct(U.clean.capped)})${chosen ? "<br><b>chosen</b>" : r.ok ? "" : "<br>fails the coherence rule"}`);
    });
  }
}

/* looped vs dense, side by side along the forward pass: each stretch that can be erased gets a column, red up to the
   refusal rate after erasing that stretch alone, blue for the rest. Qwen dips in the middle; Dory only at the end. */
function contrastChart(el) {
  const st = { dory: "dory4" }; /* Dory run with 4 loops; every model at its trained loops is in the panels below */
  function draw() {
    el.innerHTML = "";
    /* one scale for both rows: Dory with 4 loops fills the width (16 + 4 × 40 + 16 sublayers), and each of Qwen's
       quarters is as wide as one of Dory's loops, so Qwen's 36 layers line up under Dory's looped stack */
    const W = 760, mL = 40, mR = 112, colH = 118, rowH = 236, H = 2 * rowH + 6, TOT = 192, PX = (W - mL - mR) / TOT, CW = 40 * PX * 0.62;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Refusal after erasing one stretch of the forward pass: Qwen against Dory", style: "min-width:560px" }, el);
    const nD = grid(st.dory).units, offD = (TOT - (32 + 40 * nD)) / 2;
    const rows = [
      { m: "qwen", title: "Qwen3-8B · dense", sub: "36 different layers, once: the middle quarters build refusal", off: 16,
        segs: [0, 1, 2, 3].map((i) => ({ len: 40, k: i + 1, lab: `Q${i + 1}`, sub: `layers ${QLAYERS[i]}` })) },
      { m: st.dory, title: "Dory · looped, run with 4 loops", sub: "the same 40 sublayers each loop: only the last loop's erase sticks", off: offD,
        segs: [{ len: 16, lab: "entry stack", sub: "runs once" }].concat(Array.from({ length: nD }, (_, i) => ({ len: 40, k: i + 1, lab: `loop ${i + 1}`, sub: "40 sublayers" })), [{ len: 16, lab: "exit stack", sub: "runs once" }]) },
    ];
    rows.forEach((row, ri) => {
      const g = grid(row.m), y0 = ri * rowH + 6, base = y0 + 62 + colH;
      s("circle", { cx: 13, cy: y0 + 9, r: 5, fill: MODELS[row.m].color }, svg);
      s("text", { x: 23, y: y0 + 13, class: "t-title", text: row.title }, svg);
      s("text", { x: 8, y: y0 + 29, text: row.sub, "font-size": 10.5 }, svg);
      let acc = row.off;
      for (const sg of row.segs) {
        const a = mL + acc * PX, b = mL + (acc + sg.len) * PX; acc += sg.len;
        s("rect", { x: a + 1, y: base + 6, width: b - a - 2, height: 12, rx: 6, fill: sg.k ? "var(--panel-2)" : "var(--led-off)", stroke: "var(--line)" }, svg);
        s("text", { x: (a + b) / 2, y: base + 33, "text-anchor": "middle", text: sg.lab, "font-size": 10.5, class: sg.k ? "t-strong" : null }, svg);
        s("text", { x: (a + b) / 2, y: base + 46, "text-anchor": "middle", text: sg.sub, "font-size": 9.5 }, svg);
        if (!sg.k) continue;
        const r = g.erase[String(sg.k)], cw = CW, cx = (a + b) / 2 - cw / 2, red = (colH * r.r) / 100;
        s("path", { d: barPath(cx, base - colH, cw, colH - red - 1), fill: "var(--flow)" }, svg);
        s("rect", { x: cx, y: base - red + 1, width: cw, height: Math.max(0, red - 1), fill: "var(--refuse)" }, svg);
        s("text", { x: cx + cw / 2, y: base - colH - 6, "text-anchor": "middle", class: "t-strong", text: pct(r.r), "font-size": 12 }, svg);
        const hit = s("rect", { x: a, y: base - colH - 20, width: b - a, height: colH + 40, fill: "transparent" }, svg);
        addTip(hit, `${MODELS[row.m].full}<br>erase on ${unitName(row.m, sg.k)} only: <b>${pct(r.r, 1)}</b> refuse<br>95% CI ${pct(r.lo, 1)}–${pct(r.hi, 1)} · ${r.n} prompts`);
      }
      const yc = base - (colH * g.clean.r) / 100;
      s("line", { x1: mL, x2: mL + TOT * PX + 6, y1: yc, y2: yc, stroke: "var(--ink-2)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, svg);
      s("text", { x: mL + TOT * PX + 10, y: yc + 4, text: `no steering ${pct(g.clean.r)}`, "font-size": 10.5 }, svg);
    });
  }
  draw();
}

/* ================================================================== 3. extra loops */
(function extraLaps() {
  const st = { m: "dory", exp: "erase", extra: false };
  let tr;
  const board = new Board($("ex-board"), { title: "" });
  let dots = null;
  const pa = D.persona_ablate;
  const paRow = (m, cond) => pa[m].rows.find((r) => r.cond === cond);
  const trainedOf = (m) => (m === "dory" ? 2 : 4);
  const refusalExp = (mode) => (m) => {
    const e = D.extra[m][mode], n = trainedOf(m), c = grid(m).clean.r, bite = vRef(m, mode, String(n));
    const ev = Array.from({ length: n }, (_, i) => ({ k: i + 1, type: mode }));
    return { a: { ...e.at_trained, run: m }, b: { ...e.at_extra, run: m === "dory" ? "dory4" : "ouro6" }, c: e.clean.r, gain: e.gain, unit: "%", dots: true, what: "refusal rate", steered: "every trained loop", color: "var(--refuse)",
      plan: (x) => ({ events: ev, vExit: vOf(c, (x ? e.at_extra : e.at_trained).r), bite, peak: c / 100 }) };
  };
  const EXPS = {
    erase: { label: "Erase refusal", mean: MEAN.refusal, type: "erase", get: refusalExp("erase") },
    push: { label: "Add against refusal", mean: MEAN.refusal, type: "push", get: refusalExp("push") },
    persona: { label: "Add a persona", only: "dory", mean: MEAN.persona, type: "persona", get: () => {
      const ga = personaGain("dory", "0,1"), gb = personaGain("dory4", "0,1");
      return { a: { r: ga }, b: { r: gb }, c: 0, unit: "", sign: true, what: "trait gain", steered: "loops 1 and 2", color: "var(--steer)",
        plan: (x) => ({ events: [1, 2].map((k) => ({ k, type: "persona" })), vExit: x ? gb / ga : 1, bite: personaGain("dory", "1") / ga }) };
    } },
    perase: { label: "Erase a persona", only: "dory", mean: MEAN.perase, type: "erase", get: () => {
      const A = paRow("dory", "ablate_all").trait_score, B = paRow("dory4", "ablate_12").trait_score, span = pa.dory.prompted - A;
      return { a: { r: A }, b: { r: B }, c: pa.dory.prompted, unit: "", what: "trait score", steered: "loops 1 and 2", lowerWorks: true, color: "var(--steer)",
        plan: (x) => ({ events: [1, 2].map((k) => ({ k, type: "erase" })), vExit: x ? (pa.dory4.prompted - B) / span : 1, bite: (pa.dory.prompted - paRow("dory", "ablate_2").trait_score) / span }) };
    } },
    kick: { label: "Random kick", mean: MEAN.kick, type: "kick", get: (m) => {
      const loop = trainedOf(m), A = m === "dory" ? "dory" : "ouro", B = m === "dory" ? "dory4" : "ouro6";
      const ra = kickFind(A, loop, 1.0), rb = kickRows(B).find((r) => r.site && r.site.loop === loop - 1 && r.site.eps === 1.0 && !r.every);
      const cA = kickClean(A).acc, cB = kickClean(B).acc;
      /* darkness = accuracy lost: the kick darkens the stream as much as it costs when nothing follows, and the extra
         loops wash out as much of that as they win back */
      const dA = clamp(1 - ra.acc / cA, 0, 1), dB = clamp(1 - rb.acc / cB, 0, 1);
      return { a: { r: 100 * ra.acc, dots: ra.dots, out: ra.out, cand: ra.cand, run: A }, b: { r: 100 * rb.acc, dots: rb.dots, out: rb.out, cand: rb.cand, run: B },
        c: 100 * cB, cA: 100 * cA, unit: "%", dots: true, what: "MATH-500 accuracy",
        steered: `loop ${loop} (kick size 1)`, kickLap: loop, color: "var(--kick)",
        plan: (x) => ({ events: [{ k: loop, type: "kick" }], vExit: x ? dB : dA, bite: dA }) };
    } },
  };
  chips($("ex-models"), ["dory", "ouro"].map((m) => ({ id: m, label: MODELS[m].label, color: MODELS[m].color })), { value: "dory", swatch: true,
    onChange: (m) => { st.m = m; if (EXPS[st.exp].only && EXPS[st.exp].only !== m) st.exp = "erase"; drawExps(); update(true); } });
  function drawExps() {
    chips($("ex-exps"), Object.entries(EXPS).map(([id, e]) => ({ id, label: e.label, disabled: e.only && e.only !== st.m, title: e.only && e.only !== st.m ? "Not run for this model" : null })),
      { value: st.exp, onChange: (v) => { st.exp = v; update(true); } });
  }
  const togg = chips($("ex-toggle"), [{ id: false, label: "Trained loops" }, { id: true, label: "+2 extra loops" }], { value: false,
    onChange: (v) => { const was = st.extra; st.extra = v; update(true, v && !was ? "grow" : !v && was ? "shrink" : null); } });
  let dotsKind = null, exCur = null, exSeq = 0;
  const exPick = { refusal: D.examples.safe[0], kick: 0 };
  /* the reply behind the picked dot, in the run on show (trained loops or two extra) */
  function exReply() {
    if (!exCur || !dots) return;
    const M = MODELS[st.m], E = EXPS[st.exp], kind = st.exp === "kick" ? "kick" : "refusal", i = exPick[kind];
    const who = `${M.label} · ${E.label.toLowerCase()} · ${trainedOf(st.m) + (st.extra ? 2 : 0)} loops`;
    dots.select(i);
    if (kind === "kick") showKickReply($("ex-ex"), { run: exCur.run, cand: exCur.cand, i, code: exCur.out ? exCur.out[i] : null, who, seq: ++exSeq, cur: () => exSeq });
    else showRefusalReply($("ex-ex"), { run: exCur.run, cand: exCur.cand, i, v: exCur.dots[i], who, seq: ++exSeq, cur: () => exSeq });
  }
  function update(animate, morph) {
    const M = MODELS[st.m], E = EXPS[st.exp], x = E.get(st.m);
    const trained = trainedOf(st.m), n = trained + (st.extra ? 2 : 0);
    togg.forEach((b, i) => b.setAttribute("aria-pressed", String((i === 1) === st.extra)));
    togg[0].textContent = `Trained (${trained} loops)`; togg[1].textContent = `+2 extra loops (${trained + 2})`;
    const make = (loops) => new Track3($("ex-track"), { model: st.m, loops, maxLaps: trained + 2, trained, meaning: E.mean, types: [E.type], label: `${M.full}, ${loops} loops` });
    const v = st.extra ? x.b : x.a, plan = x.plan(st.extra);
    const done = () => {
      board.set(v.r, { title: `${M.label} · <b>${x.what}</b>`, suffix: x.unit, sign: !!x.sign, color: x.color,
        right: st.exp === "persona" ? "unsteered +0" : `unsteered ${x.unit === "%" ? pct(st.extra || !x.cA ? x.c : x.cA) : r0(x.c)}`,
        sub: v.lo != null ? `95% CI ${pct(v.lo)}–${pct(v.hi)}` : (st.exp === "kick" ? "100 problems" : "mean over traits"),
        subRight: v.n ? `${v.n} prompts` : "" });
      if (x.dots) {
        const kind = st.exp === "kick" ? "kick" : "refusal";
        if (dotsKind !== kind) {
          const pick = (i) => { exPick[kind] = i; exReply(); };
          dots = new Split($("ex-dots"), kind === "kick"
            ? { labels: ["Solved", "Not solved"], pickAll: true, onPick: pick,
                tipFn: (i, w) => `Problem ${i + 1}: ${w == null ? "no data" : exCur && exCur.out ? D.kicks.outcomes[exCur.out[i]] : w >= 0.5 ? "solved" : "not solved"}<br><i>click to read the response</i>` }
            : { marks: D.examples.safe, pickAll: true, onPick: pick,
                tipFn: (i, w) => (w == null ? `Prompt ${i + 1}: no data` : `Prompt ${i + 1}<br>refused in ${Math.round(9 * w)} of its 9 samples<br><i>click to ${D.examples.safe.includes(i) ? "read the reply" : "see the prompt"}</i>`) });
          dotsKind = kind;
        }
        exCur = v;
        $("ex-dots").hidden = false; dots.set(v.dots, kind === "kick" ? "var(--kick)" : [RGB.flow, RGB.red], null, kind === "kick" ? v.out : null);
        kickKey($("ex-key"), kind === "kick" ? v.out : null);
        exReply();
      } else { $("ex-dots").hidden = true; clearEl($("ex-key")); clearEl($("ex-ex")); exCur = null; }
    };
    const go = () => {
      if (animate) { board.set(null, { sub: "running…", subRight: "" }); tr.play(plan, { onDone: done }); }
      else { tr.show(plan); done(); }
    };
    /* the extra loops are added to the same drawing, so it reads as the same model run longer, not a bigger model */
    if (animate && morph === "grow") { tr = make(n); tr.apply(plan); board.set(null, { sub: "adding 2 loops…", subRight: "" }); tr.morph(trained, { onDone: go }); }
    else if (animate && morph === "shrink" && tr && tr.n === trained + 2) {
      board.set(null, { sub: "removing the extra loops…", subRight: "" }); tr.apply({});
      tr.morph(trained, { reverse: true, dur: 800, onDone: () => { tr = make(n); go(); } });
    } else { tr = make(n); go(); }
    const rcv = st.exp === "persona" ? (x.a.r - x.b.r) / x.a.r : (x.b.r - x.a.r) / (x.c - x.a.r);
    const unitFmt = (val) => (x.unit === "%" ? pct(val, 1) : (x.sign ? `${val > 0 ? "+" : ""}${val.toFixed(1)}` : val.toFixed(1)));
    $("ex-cap").textContent = `${M.full}, ${E.label.toLowerCase()} on ${x.steered}. With ${trained} loops: ${unitFmt(x.a.r)}. With ${trained + 2} loops: ${unitFmt(x.b.r)}` +
      (st.exp === "persona" ? " (unsteered: +0)." : ` (unsteered: ${unitFmt(x.c)}).`) + ` The extra loops undo ${r0(100 * clamp(rcv, -1, 2))}% of the effect.` +
      (x.gain ? ` On the same ${x.gain.n} prompts, they add ${x.gain.d > 0 ? "+" : ""}${x.gain.d.toFixed(1)} points of refusal (95% CI ${x.gain.lo.toFixed(1)} to ${x.gain.hi.toFixed(1)}).` : "") +
      (st.exp === "perase" ? ` The persona is set by a system prompt; erasing a random direction leaves ${paRow("dory4", "random_all").trait_score}.` : "");
  }
  drawExps(); update(false);
  { /* the paired refusal gains, in the text under the figure */
    const g = (m) => D.extra[m].erase.gain, f = (x) => `${x.d > 0 ? "+" : ""}${x.d.toFixed(1)} points (95% CI ${x.lo.toFixed(1)} to ${x.hi.toFixed(1)})`;
    $("ex-gains").textContent = ` With refusal erased on every trained loop, two extra loops add ${f(g("dory"))} of refusal for Dory and ${f(g("ouro"))} for Ouro, each measured on the same prompts with and without the extra loops (${g("dory").n === g("ouro").n ? `${g("dory").n} prompts each` : `${g("dory").n} prompts for Dory, ${g("ouro").n} for Ouro`}).`;
  }
  SOURCES.push("Extra loops: grid_test4{,_b} (Dory), grid_test6{,_b} (Ouro), each pair on the prompts both runs and the unsteered run share; persona_*_loops4.csv, capability/math500/kickp_loops4 and kickp_loops6");
})();

function gauge(el, label, val, ref, color, fmtv = (v) => r0(v), refName = "unsteered") {
  el.innerHTML = "";
  const fill = h("div", { class: "gf", style: `width:0%;background:${color}` });
  el.append(h("div", { class: "gl" }, h("span", {}, label), h("span", {}, `${fmtv(val)} (${refName} ${fmtv(ref)})`)),
    h("div", { class: "gt" }, fill, h("div", { class: "gm", style: `left:calc(${clamp(ref, 0, 100)}% - 1px)`, title: refName })));
  requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = `${clamp(val, 0, 100)}%`; }));
}

/* ================================================================== 4. personas */
(function personas() {
  const st = { m: "dory", trait: null, units: null };
  let tr = null, trM = null;
  const order = (m) => personaUnits(m).sort((a, b) => a.length - b.length || a.localeCompare(b));
  const uLabel = (m, u) => {
    const ks = u.split(",").map((x) => +x + 1);
    if (ks.length === MODELS[m].loops) return MODELS[m].straight ? "All quarters" : "Every loop";
    return ks.map((k) => unitName(m, k)).join(" + ");
  };
  chips($("pe-models"), ORDER5.map((m) => ({ id: m, label: MODELS[m].label, color: MODELS[m].color })), { value: "dory", swatch: true,
    onChange: (m) => { st.m = m; st.trait = null; st.units = null; draw(true); } });
  function draw(animate) {
    const P = D.persona[st.m], traits = Object.keys(P.traits);
    if (!st.trait || !P.traits[st.trait]) st.trait = traits.includes("optimistic") ? "optimistic" : traits[0];
    const us = order(st.m);
    if (!st.units || !us.includes(st.units)) st.units = us[0];
    /* the traits this model can be steered on, in one order for every model; the footnote says why the others are missing */
    chips($("pe-traits"), PTRAITS.filter((t) => P.traits[t]).map((t) => ({ id: t, label: t, color: rgbCss(traitRGB(t)) })),
      { value: st.trait, swatch: true, onChange: (t) => { st.trait = t; draw(true); } });
    $("pe-missing").textContent = missingNote(st.m, PTRAITS.filter((t) => !P.traits[t]), P);
    chips($("pe-units"), us.map((u) => ({ id: u, label: uLabel(st.m, u) })), { value: st.units, onChange: (u) => { st.units = u; draw(true); } });
    if (trM !== st.m) { tr = new Track3($("pe-track"), { model: st.m, loops: MODELS[st.m].loops, trained: st.m === "dory4" ? 2 : MODELS[st.m].loops, meaning: traitMean("persona", st.trait), types: ["persona"], label: `${MODELS[st.m].full} with a persona vector` }); trM = st.m; }
    tr.o.meaning = traitMean("persona", st.trait); /* each trait has its own colour */
    const plan = personaPlan(st.m, st.trait, st.units);
    if (animate) tr.play(plan); else tr.show(plan);
    const T = P.traits[st.trait], b = T.by_units[st.units], allU = us[us.length - 1];
    gauge($("pe-gauge-trait"), `Trait score · ${st.trait}`, b.score, T.clean, rgbCss(traitRGB(st.trait)));
    gauge($("pe-gauge-coh"), "Coherence", b.coh, T.coh_clean, "var(--ink-3)");
    const sg = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}`;
    $("pe-cap").textContent = `${MODELS[st.m].full}, ${st.trait} on ${lc(uLabel(st.m, st.units))}: ${sg(b.gain)} over unsteered. ` +
      (st.units === allU ? "" : `On every ${MODELS[st.m].straight ? "quarter" : "loop"} at the same strength: ${sg(T.by_units[allU].gain)}. `) + "Scores average 100 replies (20 questions × 5)." +
      (st.trait === "hallucinating" ? ` ${halluNote(st.m)}` : "");
    $("pe-q").textContent = `Example question: “${T.question}”`;
    const ex = T.examples, cleanEx = ex.clean, stEx = ex[st.units];
    $("pe-replies").innerHTML = "";
    const card = (who, e) => {
      const body = h("div", { class: "md" });
      body.innerHTML = e ? markdown(e.text) : "<p>No example.</p>";
      return h("div", { class: "reply" }, h("div", { class: "who" }, h("span", {}, who), h("span", {}, e && e.score != null ? `trait ${e.score}` : "")), body);
    };
    $("pe-replies").append(card("Unsteered", cleanEx), card(`Persona vector on ${lc(uLabel(st.m, st.units))}`, stEx));
  }
  draw(false);
  SOURCES.push("Personas: data/runs/<model>/analysis/persona_loops*.csv, persona/steer_loops*.judged.parquet");
  for (const m of ORDER5) {
    const us = order(m), box = h("div"); $("pe-multiples").append(box);
    const n = MODELS[m].loops;
    barChart(box, { title: MODELS[m].label, max: 55, fmt: (v) => `${v >= 0 ? "+" : ""}${r0(v)}`,
      items: us.filter((u) => u.split(",").length === 1 || u.split(",").length === n).map((u) => {
        const g = personaGain(m, u), all = u.split(",").length === n;
        return { label: all ? "all" : unitShort(m, +u + 1), value: g, color: MODELS[m].color, faded: all,
          tip: `${MODELS[m].full}: ${lc(uLabel(m, u))} → mean trait gain ${g.toFixed(1)}` };
      }) });
  }
})();

/* hallucination starts high: every question asks about something that does not exist, and the small models often say
   so in their first line and then pad the reply with invented names and places, which the judge counts (Ouro mostly plays
   along: in about half of the replies scored 70 or more, Dory, Nanbeige and Qwen open by saying the thing does not exist) */
function halluNote(m) {
  const H = (k) => D.persona[k].traits.hallucinating, allU = (k) => Array.from({ length: MODELS[k].loops }, (_, i) => i).join(",");
  const g = (k) => H(k).by_units[allU(k)].gain, name = (k) => (k === "dory4" ? "Dory × 4" : MODELS[k].label.replace(" (dense)", ""));
  const ks = ORDER5.filter((k) => D.persona[k].traits.hallucinating), weak = ks.filter((k) => g(k) < 15), strong = ks.filter((k) => g(k) >= 15);
  const c = H(m).clean;
  return `Every hallucination question asks about something that does not exist (a 2024 Mars mission, the Quantium X7 processor), and the unsteered score is already ${r0(c)}${c >= 70 ? ", which leaves little room to push" : ""}. ` +
    `Ouro mostly plays along, and the other models often say the thing does not exist in their first line and then pad the reply with invented names, places and findings, which the judge still counts as hallucination. ` +
    `On every loop (Qwen: every quarter) the vector adds ${andList(weak.map((k) => `${sgn(g(k))} for ${name(k)}`))}, against ${andList(strong.map((k) => `${sgn(g(k))} for ${name(k)}`))}.`;
}

/* ================================================================== 4b. erase a prompted persona */
(function personaErase() {
  const A = D.persona_ablate, MS = ORDER5;
  const st = { m: "dory", trait: "humorous", cond: "ablate_1" };
  let tr = null, trM = null;
  const every = (m) => (MODELS[m].straight ? "quarter" : "loop");
  const parse = (m, c) => (c === "ablate_all" || c === "random_all" ? Array.from({ length: MODELS[m].loops }, (_, i) => i + 1) : c.replace("ablate_", "").replace("q", "").split("").map(Number));
  const rank = (m, c) => (c === "random_all" ? 99 : c === "ablate_all" ? 90 : parse(m, c).length * 10 + parse(m, c)[0]);
  const conds = (m) => Object.values(A[m].traits)[0].rows.map((r) => r.cond).filter((c) => c.startsWith("ablate") || c === "random_all").sort((a, b) => rank(m, a) - rank(m, b));
  const cLabel = (m, c) => (c === "random_all" ? `Random, every ${every(m)}` : c === "ablate_all" ? `Every ${every(m)}` : parse(m, c).map((k) => unitName(m, k)).join(" + "));
  const cShort = (m, c) => (c === "random_all" ? "rand" : c === "ablate_all" ? "all" : parse(m, c).map((k) => unitShort(m, k)).join("+"));
  const rowOf = (T, c) => T.rows.find((r) => r.cond === c);
  const removed = (T, score) => clamp((T.prompted - score) / Math.max(5, T.prompted - T.noprompt), 0, 1);
  function plan(m, T, c) {
    const n = MODELS[m].loops, vs = {};
    for (let k = 1; k <= n; k++) { const r = rowOf(T, MODELS[m].straight ? `ablate_q${k}` : `ablate_${k}`); vs[k] = r ? removed(T, r.score) : 0; }
    const v = removed(T, rowOf(T, c).score);
    return { events: parse(m, c).map((k) => ({ k, type: "erase" })), vExit: v, bite: c === "random_all" ? v : vs[n], vSingle: vs, tagText: c === "random_all" ? "RANDOM DIR." : null, random: c === "random_all" };
  }
  chips($("pa-models"), MS.map((m) => ({ id: m, label: MODELS[m].label, color: MODELS[m].color })), { value: st.m, swatch: true, onChange: (m) => { st.m = m; draw(true); } });
  function draw(animate) {
    const m = st.m, P = A[m], traits = Object.keys(P.traits);
    if (!P.traits[st.trait]) st.trait = traits.includes("humorous") ? "humorous" : traits[0];
    const cs = conds(m);
    if (!cs.includes(st.cond)) st.cond = cs[0];
    chips($("pa-traits"), PTRAITS.filter((t) => P.traits[t]).map((t) => ({ id: t, label: t, color: rgbCss(traitRGB(t)) })),
      { value: st.trait, swatch: true, onChange: (t) => { st.trait = t; draw(true); } });
    $("pa-missing").textContent = missingNote(m, PTRAITS.filter((t) => !P.traits[t]), P, true);
    chips($("pa-units"), cs.map((c) => ({ id: c, label: cLabel(m, c) })), { value: st.cond, onChange: (c) => { st.cond = c; draw(true); } });
    if (trM !== m) {
      tr = new Track3($("pa-track"), { model: m, loops: MODELS[m].loops, trained: m === "dory4" ? 2 : MODELS[m].loops, meaning: traitMean("perase", st.trait), types: ["erase"], label: `${MODELS[m].full}: a trait set by the system prompt, erased on some loops` });
      trM = m;
    }
    tr.o.meaning = traitMean("perase", st.trait);
    const tc = rgbCss(traitRGB(st.trait));
    const T = P.traits[st.trait], r = rowOf(T, st.cond), pl = plan(m, T, st.cond);
    if (animate) tr.play(pl); else tr.show(pl); /* the bar chart beside it holds the numbers */
    gauge($("pa-gauge-coh"), "Coherence", r.coh, rowOf(T, "prompted").coh, "var(--ink-3)", (v) => r0(v), "prompted");
    const el = $("pa-chart");
    el.innerHTML = "";
    barChart(el, { title: `Trait score · ${st.trait}`, subtitle: "dashed: prompted, no erasing · dotted: no prompt", max: 100, w: 380, ht: 210, padR: 84, fmt: (v) => r0(v),
      items: cs.map((c) => { const x = rowOf(T, c); return { label: cShort(m, c), value: x.score, color: c === "random_all" ? "var(--neutral)" : tc, faded: c !== st.cond,
        tip: `${MODELS[m].full}, ${st.trait}: erased on ${lc(cLabel(m, c))} → trait ${x.score}, coherence ${x.coh}` }; }),
      refs: [{ value: T.prompted, label: `prompted ${r0(T.prompted)}` }, { value: T.noprompt, label: `no prompt ${r0(T.noprompt)}`, dash: "2 3" }] });
    const share = r0(100 * (T.prompted - r.score) / Math.max(1e-9, T.prompted - T.noprompt));
    $("pa-cap").textContent = `${MODELS[m].full}, system prompt for “${st.trait}”, ${st.cond === "random_all" ? `a random direction erased on every ${every(m)} (a control)` : `trait direction erased on ${lc(cLabel(m, st.cond))}`}: trait ${r0(r.score)}, ` +
      `against ${r0(T.prompted)} with the prompt alone and ${r0(T.noprompt)} without it, so ${share}% of what the prompt added is gone.` +
      (T.prompted - T.noprompt < 15 ? " The prompt barely raises this trait, so that share is noisy." : "") + " 100 replies per bar (20 questions × 5).";
  }
  draw(false);
  SOURCES.push("Persona erasure: data/runs/<model>/analysis/persona_ablate.csv (Dory × 4 loops: persona_ablate_loops4.csv)");
  for (const m of MS) {
    const box = h("div"); $("pa-multiples").append(box);
    const P = A[m], cs = conds(m).filter((c) => c !== "random_all" && (parse(m, c).length === 1 || c === "ablate_all"));
    const coh = (c) => (P.rows.find((x) => x.cond === c) || {}).coherence;
    barChart(box, { title: MODELS[m].label, max: 100, fmt: (v) => `${r0(v)}%`,
      items: cs.map((c) => ({ label: cShort(m, c), value: P.removed[c], color: MODELS[m].color, faded: c === "ablate_all",
        tip: `${MODELS[m].full}, erased on ${lc(cLabel(m, c))}: ${r0(P.removed[c])}% of the points the prompt adds are gone (summed over ${P.removed_traits.length} traits); coherence ${r0(coh(c))} (prompted ${r0(coh("prompted"))})` })) });
  }
  const R = (m, c) => r0(A[m].removed[c]);
  const set = (id, v) => { $(id).textContent = v; };
  set("pa-d4-1", `${R("dory4", "ablate_1")}%`); set("pa-d4-2", `${R("dory4", "ablate_2")}%`); set("pa-d4-3", `${R("dory4", "ablate_3")}%`);
  set("pa-d4-4", `${R("dory4", "ablate_4")}%`); set("pa-d4-all", `${R("dory4", "ablate_all")}%`);
  set("pa-d-1", `${R("dory", "ablate_1")}%`); set("pa-d-2", `${R("dory", "ablate_2")}%`);
  set("pa-n-1", `${R("nanbeige", "ablate_1")}%`); set("pa-n-2", `${R("nanbeige", "ablate_2")}%`);
  set("pa-o-1", `${R("ouro", "ablate_1")}%`); set("pa-o-4", `${R("ouro", "ablate_4")}%`);
  const ot = (t, c) => rowOf(A.ouro.traits[t], c);
  { /* optimism erased on loop 4: from the prompted score to where, against the score with no prompt */
    const P = r0(A.ouro.traits.optimistic.prompted), X = r0(ot("optimistic", "ablate_4").score), Y = r0(A.ouro.traits.optimistic.noprompt);
    $("pa-o-opt-clause").textContent = `Erasing optimism on loop 4 takes it from ${P} to ${X}, ` +
      (X < Y - 1 ? `overshooting into gloom below the ${Y} it scores with no prompt` : X <= Y + 5 ? `close to the ${Y} it scores with no prompt` : `still above the ${Y} it scores with no prompt`);
  }
  set("pa-o-apa", `${r0(ot("apathetic", "prompted").coh)} to ${r0(ot("apathetic", "ablate_4").coh)}`);
  set("pa-q-all", `${R("qwen", "ablate_all")}%`);
  const ocoh = (c) => r0((A.ouro.rows.find((x) => x.cond === c) || {}).coherence);
  set("pa-o-cohn", ocoh("noprompt")); set("pa-o-cohp", ocoh("prompted"));
  { /* the other models' coherence, with and without the system prompt, as ranges */
    const co = (m, c) => (A[m] && (A[m].rows.find((x) => x.cond === c) || {}).coherence);
    const rng = (c) => { const v = ["nanbeige", "dory", "qwen"].map((m) => co(m, c)).filter((x) => x != null); return `${r0(Math.min(...v))} to ${r0(Math.max(...v))}`; };
    set("pa-coh-others-n", rng("noprompt")); set("pa-coh-others-p", rng("prompted"));
  }
})();

/* ================================================================== 5. kicks */
const EPS = [0.5, 0.7, 1.0, 1.5];
(function kicks() {
  const st = { m: "dory", unit: 1, every: false, eps: 2 };
  let tr, units = [];
  const board = new Board($("ki-board"), { title: "", color: "var(--ink)" });
  let cur = null, kiSeq = 0;
  const outName = (i) => (cur && cur.out ? D.kicks.outcomes[cur.out[i]] : null);
  const dots = new Split($("ki-dots"), { labels: ["Solved", "Not solved"], pickAll: true, onPick: (i) => { st.pick = i; showReply(); },
    tipFn: (i, v) => `Problem ${i + 1}: ${v == null ? "not run" : outName(i) || (v >= 0.5 ? "solved" : "not solved")}<br><i>click to read the response</i>` });
  st.pick = 0;
  function showReply() {
    dots.select(st.pick);
    if (!cur) { clearEl($("ki-ex")); return; }
    const M = MODELS[st.m], eps = EPS[st.eps];
    const where = cur.site == null ? "no kick" : st.every ? `size-${eps} kick on every ${M.straight ? "quarter" : "loop"}` : `size-${eps} kick on ${unitName(st.m, st.unit)}`;
    showKickReply($("ki-ex"), { run: st.m, cand: cur.cand, i: st.pick, code: cur.out ? cur.out[st.pick] : null, who: `${M.label} · ${where}`, seq: ++kiSeq, cur: () => kiSeq });
  }
  chips($("ki-models"), ORDER5.map((m) => ({ id: m, label: MODELS[m].label, color: MODELS[m].color })), { value: st.m, swatch: true,
    onChange: (m) => { st.m = m; st.unit = 1; st.every = false; build(); } });
  const epsEl = $("ki-eps");
  epsEl.addEventListener("input", () => { st.eps = +epsEl.value; show(false); });
  $("ki-go").addEventListener("click", () => show(true));
  function build() {
    const M = MODELS[st.m];
    tr = new Track3($("ki-track"), { model: st.m, loops: M.loops, trained: st.m === "dory4" ? 2 : M.loops, meaning: MEAN.kick, types: ["kick"], label: `${M.full} track` });
    const seg = $("ki-units"); seg.innerHTML = "";
    units = [];
    for (let k = 1; k <= M.loops; k++) {
      const b = h("button", { type: "button" }, h("span", {}, unitName(st.m, k)), h("small", {}, M.straight ? `layer ${KLAYER.qwen[k - 1]}` : `layer ${GEO[st.m].site.kick}`));
      b.addEventListener("click", () => { st.unit = k; st.every = false; sync(); show(true); });
      seg.append(b); units.push(b);
    }
    const ev = h("button", { type: "button" }, h("span", {}, "Every"), h("small", {}, "same kick each loop"));
    ev.addEventListener("click", () => { st.every = true; sync(); show(true); });
    seg.append(ev); units.push(ev);
    sync(); show(false);
  }
  function sync() { units.forEach((b, i) => b.setAttribute("aria-pressed", String(st.every ? i === units.length - 1 : i + 1 === st.unit))); }
  function show(animate) {
    const M = MODELS[st.m], eps = EPS[st.eps], c = kickClean(st.m);
    $("ki-eps-label").textContent = `${eps} × the state's length`;
    const r = kickFind(st.m, st.unit, eps, st.every), plan = kickPlan(st.m, st.unit, eps, st.every);
    const done = () => {
      board.set(r ? 100 * r.acc : null, { title: `${M.label} · <b>MATH-500 accuracy</b>`, right: `unkicked ${pct(100 * c.acc)}`,
        sub: r ? `95% CI ${pct(100 * r.lo)}–${pct(100 * r.hi)}` : "not run at this size", subRight: "100 problems" });
      cur = r;
      dots.set(r ? r.dots : null, "var(--kick)", r ? `${r0(100 * r.acc)} of 100 solved` : "not run", r ? r.out : null);
      kickKey($("ki-key"), r ? r.out : null);
      $("ki-kl").textContent = r && r.kl != null ? `Next-token KL ${r.kl.toFixed(3)}` : "";
      showReply();
    };
    if (!plan) { tr.show({}); done(); return; }
    if (animate) { board.set(null, { sub: "kicking…", subRight: "" }); tr.play(plan, { onDone: done }); }
    else { tr.show(plan); done(); }
  }
  build();
  const d2 = kickFind("dory", 2, 1.0), d4 = kickRows("dory4").find((r) => r.site && r.site.loop === 1 && r.site.eps === 1.0 && !r.every);
  const o4 = kickFind("ouro", 4, 1.0), o6 = kickRows("ouro6").find((r) => r.site && r.site.loop === 3 && r.site.eps === 1.0 && !r.every);
  $("ki-same").textContent = `A kick of size 1 at Dory's loop 2 leaves ${pct(100 * d2.acc)} accuracy when loop 2 is the last loop, and ${pct(100 * d4.acc)} when two more loops follow. Ouro's loop 4: ${pct(100 * o4.acc)} with 4 loops, ${pct(100 * o6.acc)} with 6.`;
  const oe = kickFind("ouro", 1, 0.5, true), oc = kickClean("ouro").acc;
  $("ki-all").textContent = pct(100 * oe.acc);
  $("ki-allmax").textContent = r0(100 * Math.max(...[1, 2, 3, 4].map((k) => oc - kickFind("ouro", k, 0.5).acc)));
  SOURCES.push("Kicks: data/runs/<model>/capability/math500/kickp* and analysis/kick_kl/kickp*.parquet via scripts/kick_summary.py");

  /* where each model feels a kick: one row per run, every layer (Dory: sublayer) a cell, darker = bigger change in the
     next-token predictions (teacher-forced KL, size-1 kick). Each row spans the same width, so positions line up as a
     share of the forward pass. */
  const SR = D.kicks.strips || {}, rowsK = ORDER5.filter((m) => SR[m]);
  /* median KL per loop (quarter, stack) of each run, for the labels and the paragraph above */
  const kmed = (m, stg, lp) => {
    const v = (SR[m]?.cells || []).filter((c) => c.stage === stg && (lp == null || c.loop === lp)).map((c) => c.kl).sort((a, b) => a - b);
    return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null;
  };
  const fk = (v) => (v < 0.1 ? v.toPrecision(1) : v < 1 ? v.toFixed(2) : v.toFixed(1));
  /* colour = log KL; "row": each run from its own 5th to 95th percentile (a few outlier cells, like the first layer of
     a loop, would otherwise set the range), "shared": one range for every run, 0.001 to 10 */
  const q = (a, p) => a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))];
  function drawStrip(mode) {
    $("strip-chart").innerHTML = "";
    const W = 900, mL = 112, mR = 8, rowH = 62, top = 22, Hh = top + rowH * rowsK.length + 6, gapS = 6;
    const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Damage from a kick at each layer of each model", style: "min-width:680px" }, $("strip-chart"));
    s("text", { x: mL, y: 12, text: mode === "row" ? "within each row: lighter = felt least in that run · darker = felt most" : "lighter: barely felt (KL 0.001) · darker: strong (KL 10)", "font-size": 10 }, svg);
    rowsK.forEach((m, ri) => {
      const cells = SR[m].cells, M = MODELS[m], y0 = top + ri * rowH;
      const ks = cells.map((c) => Math.log10(Math.max(c.kl, 1e-3))).sort((a, b) => a - b);
      const [lo, hi] = mode === "row" ? [q(ks, 0.05), q(ks, 0.95)] : [-3, 1];
      const lv = (kl) => clamp((Math.log10(Math.max(kl, 1e-3)) - lo) / Math.max(1e-6, hi - lo), 0, 1);
      const stages = m === "dory" || m === "dory4"
        ? [["in", null, "entry"]].concat(Array.from({ length: M.loops }, (_, k) => ["loop", k + 1, `loop ${k + 1}`]), [["out", null, "exit"]])
        : Array.from({ length: M.loops }, (_, k) => ["loop", k + 1, unitName(m, k + 1).toLowerCase().replace(/^q/, "Q")]);
      const groups = stages.map(([stg, lp, lab]) => ({ lab, med: kmed(m, stg, lp), cs: cells.filter((c) => c.stage === stg && (lp == null || c.loop === lp)).sort((a, b) => a.layer - b.layer) })).filter((g) => g.cs.length);
      const n = groups.reduce((a, g) => a + g.cs.length, 0), cw = (W - mL - mR - gapS * (groups.length - 1)) / n;
      s("circle", { cx: 6, cy: y0 + 17, r: 4.5, fill: M.color }, svg);
      s("text", { x: 15, y: y0 + 21, class: "t-strong", text: m === "dory4" ? "Dory × 4" : m === "qwen" ? "Qwen" : M.label, "font-size": 11.5 }, svg);
      s("text", { x: 15, y: y0 + 34, text: m === "dory4" ? "trained with 2" : m === "qwen" ? "dense" : `${M.loops} loops`, "font-size": 9.5 }, svg);
      if (mode === "row") s("text", { x: 15, y: y0 + 46, text: `KL ${fk(10 ** lo)}–${fk(10 ** hi)}`, "font-size": 9 }, svg);
      let x = mL;
      for (const g of groups) {
        const x0 = x;
        for (const c of g.cs) {
          const r = s("rect", { x, y: y0 + 4, width: Math.max(0.6, cw - (cw > 3 ? 0.6 : 0.2)), height: 30, fill: "var(--kick)", "fill-opacity": (0.04 + 0.96 * lv(c.kl)).toFixed(3) }, svg);
          addTip(r, `${M.full}, ${g.lab}, ${M.ramps ? "sublayer" : "layer"} ${c.layer}<br>KL ${c.kl.toFixed(3)}`);
          x += cw;
        }
        s("text", { x: (x0 + x) / 2, y: y0 + 47, "text-anchor": "middle", text: g.lab, "font-size": 10 }, svg);
        s("text", { x: (x0 + x) / 2, y: y0 + 58, "text-anchor": "middle", text: `median ${fk(g.med)}`, "font-size": 9, fill: "var(--ink-3)" }, svg);
        x += gapS;
      }
    });
  }
  chips($("strip-scale"), [{ id: "row", label: "Each row" }, { id: "shared", label: "Shared" }], { value: "row", onChange: (v) => drawStrip(v) });
  drawStrip("row");
  for (const m of rowsK) src(`Kick strip, ${MODELS[m].label}: ${SR[m].source}`);
  if (["dory", "dory4", "nanbeige", "ouro", "qwen"].every((m) => SR[m])) {
    const o13 = [1, 2, 3].map((l) => kmed("ouro", "loop", l)), q = [1, 2, 3, 4].map((l) => kmed("qwen", "loop", l));
    $("strip-text").textContent = `Measured layer by layer, the looped models feel a kick most in their last loop. A kick on Dory's first loop moves its predictions by a median KL of ${fk(kmed("dory", "loop", 1))}, one on its second loop by ${fk(kmed("dory", "loop", 2))}. Run with four loops, Dory barely feels a kick on its first two (${fk(Math.max(kmed("dory4", "loop", 1), kmed("dory4", "loop", 2)))} or less). Ouro's first three loops sit between ${fk(Math.min(...o13))} and ${fk(Math.max(...o13))} and its fourth at ${fk(kmed("ouro", "loop", 4))}; Nanbeige's two loops at ${fk(kmed("nanbeige", "loop", 1))} and ${fk(kmed("nanbeige", "loop", 2))}. Qwen has no such slope: a kick hurts most in its middle quarters (${fk(q[1])} and ${fk(q[2])}) and least in its last (${fk(q[3])}). Dory's entry stack (${fk(kmed("dory", "in"))}) is felt more than its first loop because its output is fed into every loop.`;
  }

  /* small multiples heat grids */
  for (const m of ORDER5) {
    const M = MODELS[m], c = kickClean(m).acc, box = h("div"); $("kick-multiples").append(box);
    const cw2 = 40, ch = 26, left = 34, top = 40;
    const sv = s("svg", { viewBox: `0 0 ${left + cw2 * 4 + 6} ${top + ch * M.loops + 8}`, role: "img", "aria-label": `${M.label} kick grid` }, box);
    s("text", { x: 0, y: 15, class: "t-title", text: M.label }, sv);
    EPS.forEach((e, j) => s("text", { x: left + cw2 * j + cw2 / 2, y: top - 6, "text-anchor": "middle", text: String(e), "font-size": 10 }, sv));
    for (let k = 1; k <= M.loops; k++) {
      s("text", { x: 0, y: top + ch * (k - 1) + ch / 2 + 4, text: unitShort(m, k), "font-size": 10.5 }, sv);
      EPS.forEach((e, j) => {
        const r = kickFind(m, k, e);
        if (!r) return;
        const a = clamp(r.acc / c, 0, 1);
        const rc = s("rect", { x: left + cw2 * j, y: top + ch * (k - 1), width: cw2 - 2, height: ch - 2, rx: 3, fill: M.color, "fill-opacity": (0.06 + 0.94 * a).toFixed(3) }, sv);
        s("text", { x: left + cw2 * j + cw2 / 2 - 1, y: top + ch * (k - 1) + ch / 2 + 3, "text-anchor": "middle", text: r0(100 * r.acc),
          "font-size": 10.5, fill: a > 0.55 ? "#fff" : "var(--ink)", style: "pointer-events:none" }, sv);
        addTip(rc, `${M.full}<br>kick on ${unitName(m, k)}, size ${e}<br>accuracy ${pct(100 * r.acc)} (unkicked ${pct(100 * c)})${r.kl != null ? `<br>KL ${r.kl.toFixed(3)}` : ""}`);
      });
    }
  }
})();

/* ================================================================== 6. verdict */
(function verdict() {
  const st = { eps: 0.7 };
  chips($("vd-eps"), EPS.map((e) => ({ id: e, label: String(e) })), { value: 0.7, onChange: (v) => { st.eps = v; draw(); } });
  const cols = ["dory4", "nanbeige", "ouro", "qwen"]; /* Dory with 4 loops: up to three loops after a steered one, like Ouro */
  const rows = [
    { label: "Erase refusal", sub: "487 prompts", get: (m) => Array.from({ length: grid(m).units }, (_, i) => shareKept(m, "erase", i + 1)) },
    { label: "Add a persona", sub: "100 replies × traits", get: (m) => { const all = personaGain(m, Array.from({ length: MODELS[m].loops }, (_, i) => i).join(",")); return Array.from({ length: MODELS[m].loops }, (_, i) => personaGain(m, String(i)) / all); } },
    { label: "Random kick", sub: "100 problems", get: (m, eps) => { const c = kickClean(m).acc, ev = kickFind(m, 1, eps, true); return Array.from({ length: MODELS[m].loops }, (_, i) => { const r = kickFind(m, i + 1, eps); return r && ev ? (c - r.acc) / (c - ev.acc) : null; }); } },
  ];
  function draw() {
    const el = $("vd-matrix"); el.innerHTML = "";
    /* each cell: the bars sit below a band for their value labels, so nothing reaches the row or column titles */
    const lw = 130, cw = 190, ch = 138, top = 34;
    const svg = s("svg", { viewBox: `0 0 ${lw + cw * cols.length} ${top + ch * rows.length}`, role: "img", "aria-label": "Share of the full effect kept by loop", style: "min-width:640px" }, el);
    cols.forEach((m, j) => {
      s("circle", { cx: lw + cw * j + 13, cy: 16, r: 5, fill: MODELS[m].color }, svg);
      s("text", { x: lw + cw * j + 23, y: 21, class: "t-title", text: MODELS[m].label }, svg);
    });
    rows.forEach((row, i) => {
      const y0 = top + ch * i;
      s("line", { x1: 0, x2: lw + cw * cols.length, y1: y0, y2: y0, stroke: "var(--line)" }, svg);
      s("text", { x: 0, y: y0 + 56, class: "t-strong", text: row.label, "font-size": 12 }, svg);
      s("text", { x: 0, y: y0 + 72, text: i === 2 ? `size ${st.eps}` : row.sub, "font-size": 10 }, svg);
      cols.forEach((m, j) => {
        const vals = row.get(m, st.eps), n = vals.length;
        const x0 = lw + cw * j + 8, base = y0 + ch - 24, hmax = 72, bw = Math.min(30, (cw - 30 - 6 * (n - 1)) / n);
        const yv = (v) => base - clamp(v, 0, 1.15) * hmax; /* tallest bar top: 83 px above the base, value label 14 px above that */
        s("line", { x1: x0, x2: x0 + cw - 24, y1: yv(1), y2: yv(1), stroke: "var(--ink-3)", "stroke-dasharray": "4 3" }, svg);
        s("line", { x1: x0, x2: x0 + cw - 24, y1: base, y2: base, stroke: "var(--line)" }, svg);
        vals.forEach((v, k) => {
          const x = x0 + k * (bw + 6);
          if (v == null) return;
          const top_ = yv(Math.max(v, 0));
          s("path", { d: barPath(x, top_, bw, base - top_), fill: MODELS[m].color }, svg);
          s("text", { x: x + bw / 2, y: top_ - 4, "text-anchor": "middle", text: r0(100 * v), "font-size": 10, class: "t-strong" }, svg);
          s("text", { x: x + bw / 2, y: base + 13, "text-anchor": "middle", text: unitShort(m, k + 1), "font-size": 9.5 }, svg);
          const hit = s("rect", { x: x - 3, y: y0 + 8, width: bw + 6, height: ch - 12, fill: "transparent" }, svg);
          addTip(hit, `${MODELS[m].full}<br>${row.label.toLowerCase()} on ${unitName(m, k + 1)} only<br>keeps <b>${r0(100 * v)}%</b> of the every-${MODELS[m].straight ? "quarter" : "loop"} effect`);
        });
      });
    });
  }
  draw();
})();

/* ================================================================== 6a. why later loops can repair: the answer settles early */
(function settle() {
  const R = D.readout, MS = ORDER5, dash = { dory: "5 4" };
  const name = (m) => (m === "qwen" ? "Qwen (dense)" : m === "dory4" ? "Dory × 4" : m === "dory" ? "Dory × 2" : MODELS[m].label);
  const legend = $("st-legend");
  for (const m of MS) {
    const v = s("svg", { viewBox: "0 0 26 10", width: 26, height: 10, "aria-hidden": "true" });
    s("line", { x1: 1, x2: 25, y1: 5, y2: 5, stroke: MODELS[m].color, "stroke-width": m === "qwen" ? 2.8 : 2, "stroke-dasharray": dash[m] || null }, v);
    s("circle", { cx: 13, cy: 5, r: 3.2, fill: MODELS[m].color }, v);
    legend.append(h("span", {}, v, m === "dory" ? "Dory (2 loops, as trained)" : m === "dory4" ? "Dory run with 4 loops" : m === "qwen" ? "Qwen3-8B (after each quarter)" : MODELS[m].full));
  }
  function draw() {
    const el = $("st-chart"); el.innerHTML = "";
    const W = 860, H = 290, mL = 52, mR = 40, top = 20, bot = H - 46, kl = false; /* the top-token measure says it all */
    const X = (f) => mL + (W - mL - mR) * f;
    const lk = (v) => Math.log10(v + 0.01), Y = kl ? (v) => bot - ((lk(v) - lk(0)) / (lk(20) - lk(0))) * (bot - top) : (v) => bot - (clamp(v, 0, 100) / 100) * (bot - top);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": kl ? "KL from the final prediction after each loop" : "Share of next-token predictions already final after each loop", style: "min-width:620px" }, el);
    for (const v of kl ? [0, 0.1, 1, 10] : [0, 25, 50, 75, 100]) {
      s("line", { x1: mL, x2: W - mR, y1: Y(v), y2: Y(v), stroke: "var(--line)" }, svg);
      s("text", { x: mL - 8, y: Y(v) + 4, "text-anchor": "end", text: kl ? String(v) : `${v}%` }, svg);
    }
    for (const f of [0.25, 0.5, 0.75, 1]) s("text", { x: X(f), y: bot + 16, "text-anchor": "middle", text: `${r0(100 * f)}%`, "font-size": 10 }, svg);
    s("text", { x: X(0.5), y: bot + 34, "text-anchor": "middle", class: "t-strong", text: "share of the forward pass done", "font-size": 10.5 }, svg);
    s("text", { x: mL, y: 10, text: kl ? "KL divergence from the final prediction (log scale; 0 = identical)" : "next-token predictions that already match the final one", "font-size": 10.5 }, svg);
    for (const m of MS) {
      const vals = kl ? R[m].kl : R[m].agree.map((a) => 100 * a), n = vals.length, M = MODELS[m];
      const pts = vals.map((v, i) => ({ k: i + 1, f: (i + 1) / n, v }));
      s("polyline", { points: pts.map((p) => `${X(p.f).toFixed(1)},${Y(p.v).toFixed(1)}`).join(" "), fill: "none", stroke: M.color, "stroke-width": m === "qwen" ? 2.8 : 2, "stroke-dasharray": dash[m] || null, "stroke-linejoin": "round" }, svg);
      for (const p of pts) {
        s("circle", { cx: X(p.f), cy: Y(p.v), r: 4.6, fill: M.color, stroke: "var(--panel)", "stroke-width": 2 }, svg);
        addTip(s("circle", { cx: X(p.f), cy: Y(p.v), r: 11, fill: "transparent" }, svg),
          `${M.full}<br>after ${unitName(m, p.k)}: ${pct(100 * R[m].agree[p.k - 1])} of predictions already final<br>KL from the final prediction ${R[m].kl[p.k - 1].toFixed(3)}`);
      }
      const p0 = pts[0];
      s("text", { x: X(p0.f) + 9, y: Y(p0.v) + (kl ? -8 : m === "dory4" || m === "dory" ? 16 : -8), text: name(m), "font-size": 10.5, class: "t-strong", fill: M.color }, svg);
    }
    $("st-cap").textContent = kl
      ? `How far each loop's read-out prediction is from the final one (KL divergence, nats; lower is closer). Dory is within ${R.dory.kl[0].toFixed(2)} after loop 1, Ouro ${R.ouro.kl[0].toFixed(1)}, Nanbeige ${R.nanbeige.kl[0].toFixed(1)}, Qwen's first quarter ${R.qwen.kl[0].toFixed(1)}.`
      : `After loop 1, Dory's read-out already matches its final next-token prediction ${pct(100 * R.dory.agree[0])} of the time, Ouro's ${pct(100 * R.ouro.agree[0])} and Nanbeige's ${pct(100 * R.nanbeige.agree[0])}. Qwen's quarters reach ${R.qwen.agree.slice(0, 3).map((a) => pct(100 * a)).join(", ")}: a dense model's answer only appears in its last layers.`;
  }
  draw();
  /* settled after loop 1 vs how much of a loop-1 erase the later loops undo */
  const pts = ["nanbeige", "ouro", "dory", "dory4"].map((m) => ({ m, x: 100 * R[m].agree[0], y: 100 * (1 - shareKept(m, "erase", 1)) }));
  const W = 420, H = 300, mL = 50, mR = 20, top = 20, bot = H - 50;
  const X = (v) => mL + ((W - mL - mR) * v) / 100, Y = (v) => bot - ((bot - top) * clamp(v, 0, 105)) / 105;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Settled after loop 1 against the share of a loop-1 erase that later loops undo" }, $("st-scatter"));
  for (const v of [0, 50, 100]) {
    s("line", { x1: mL, x2: W - mR, y1: Y(v), y2: Y(v), stroke: "var(--line)" }, svg);
    s("text", { x: mL - 8, y: Y(v) + 4, "text-anchor": "end", text: `${v}%` }, svg);
    s("line", { x1: X(v), x2: X(v), y1: top, y2: bot, stroke: "var(--line)", "stroke-dasharray": "2 3" }, svg);
    s("text", { x: X(v), y: bot + 16, "text-anchor": "middle", text: `${v}%`, "font-size": 10 }, svg);
  }
  s("text", { x: (mL + W - mR) / 2, y: bot + 36, "text-anchor": "middle", class: "t-strong", text: "predictions already final after loop 1", "font-size": 10.5 }, svg);
  s("text", { x: 4, y: 10, text: "share of a loop-1 erase the later loops undo", "font-size": 10.5, class: "t-strong" }, svg);
  for (const p of pts) {
    s("circle", { cx: X(p.x), cy: Y(p.y), r: 6.5, fill: p.m === "dory" ? "var(--panel)" : MODELS[p.m].color, stroke: p.m === "dory" ? MODELS[p.m].color : "var(--panel)", "stroke-width": p.m === "dory" ? 2.4 : 2 }, svg);
    const right = p.x < 80;
    s("text", { x: X(p.x) + (right ? 11 : -11), y: Y(p.y) + (p.m === "dory" ? 14 : p.m === "dory4" ? -6 : 4), "text-anchor": right ? "start" : "end", text: name(p.m), "font-size": 11, class: "t-strong", fill: MODELS[p.m].color }, svg);
    addTip(s("circle", { cx: X(p.x), cy: Y(p.y), r: 13, fill: "transparent" }, svg),
      `${MODELS[p.m].full}<br>after loop 1: ${pct(p.x)} of predictions already final<br>erase on loop 1 only: later loops undo ${pct(p.y)} of the effect`);
  }
  const set = (id, v) => { $(id).textContent = v; };
  const undo = (m) => pct(100 * (1 - shareKept(m, "erase", 1)));
  set("st-nan", `${pct(100 * R.nanbeige.agree[0])} and ${undo("nanbeige")}`); set("st-ouro", `${pct(100 * R.ouro.agree[0])} and ${undo("ouro")}`);
  set("st-dory", `${pct(100 * R.dory.agree[0])} and ${undo("dory")}`);
  SOURCES.push("Read-out: data/runs/<model>/analysis/loop_readout_*.csv (logit lens on clean reasoning traces)");
})();

/* ================================================================== 6b. is it the re-injected input? */
(function reinjection() {
  const R = D.reinject, C = R.conds, G = GEO.dory, n = 4, NPOS = G.pre + n * G.loop + G.post;
  const mean = (a, i, j) => { let t = 0; for (let k = i; k < j; k++) t += a[k]; return t / (j - i); };
  const L34 = [G.pre + 2 * G.loop, G.pre + 4 * G.loop];
  const ref = mean(C.clean.curve, G.pre + G.loop, G.pre + 2 * G.loop); /* the unsteered level inside a loop */
  const OPTS = [
    { id: "abl12", label: "The usual copy", short: "usual copy", feed: { style: "usual" }, what: "the entry stack's output, as usual" },
    { id: "abl12+umean34", label: "No erased direction", short: "minus erased dir.", feed: { style: "edit", tag: "minus the erased direction" },
      what: "the copy with its component along the erased refusal direction set to its average, which removes the refusing-versus-complying difference along that direction without moving the stream" },
    { id: "abl12+mean34", label: "No entry-stack direction", short: "minus entry dir.", feed: { style: "edit", tag: "minus entry-stack direction" },
      what: "the copy with its component along the entry stack's own refusal direction (a different direction from the erased one) set to its average" },
    { id: "abl12+uxmean34", label: "Neither direction", short: "minus both dirs", feed: { style: "edit", tag: "minus both directions" },
      what: "the copy with both the erased direction and the entry stack's own refusal direction set to their averages" },
    { id: "abl12+smean34", label: "No refusal subspace", short: "minus 17 dirs", feed: { style: "edit", tag: "minus 17 refusal directions" },
      what: "the copy with all 17 refusal directions of the entry stack (one per sublayer input) set to their averages" },
    { id: "abl12+rmean34", label: "Control: a random direction", short: "random dir. out", feed: { style: "edit", tag: "minus a random direction" }, what: "the copy with one random direction set to its average (control)" },
    { id: "abl12+rsmean34", label: "Control: 17 random directions", short: "random subspace out", feed: { style: "edit", tag: "minus 17 random directions" }, what: "the copy with 17 random directions set to their averages (control)" },
    { id: "abl12+drop34", label: "No copy", short: "no copy", feed: { style: "cut", tag: "nothing added" }, what: "nothing: loops 3 and 4 start from the carried state alone" },
    { id: "abl12+flat34", label: "A flat copy", short: "flat copy", feed: { style: "edit", tag: "flat copy" }, what: "the same average vector at every token, which keeps the copy's size and drops everything token-specific" },
  ];
  const st = { id: "abl12+umean34" };
  const tr = new Track3($("ri-track"), { model: "dory", loops: n, trained: 2, meaning: MEAN.refusal, types: ["erase"], reinject: true,
    label: "Dory run with 4 loops, refusal erased on loops 1 and 2, the copy added before loops 3 and 4 changed" });
  const board = new Board($("ri-board"), { title: "Loops 3–4 · <b>refusal signal</b>", color: "var(--refuse)" });
  const kept = (id) => (100 * mean(C[id].curve, ...L34)) / mean(C.clean.curve, ...L34);
  /* the dot's lane: the measured signal smoothed over 9 sublayers, held through the exit stack (which turns the
     stream towards tokens, so its refusal separation shrinks for reasons unrelated to the question) */
  function runCurve(id) {
    const c = C[id].curve, end = G.pre + n * G.loop, out = new Float32Array(NPOS + 1);
    for (let i = 0; i < end; i++) { const a = Math.max(0, i - 4), b = Math.min(end, i + 5); out[i] = mean(c, a, b) / ref; }
    for (let i = end; i <= NPOS; i++) out[i] = out[end - 1];
    return out;
  }
  const planOf = (o) => ({ events: [{ k: 1, type: "erase" }, { k: 2, type: "erase" }], curve: runCurve(o.id), feed: Object.assign({ loops: [3, 4] }, o.feed) });
  /* line chart over the run: sublayer pairs (attention + MLP) averaged */
  function chart() {
    const el = $("ri-chart");
    el.innerHTML = "";
    const W = 900, H = 262, mL = 44, mR = 150, top = 46, bot = H - 26;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Refusal signal at every sublayer of Dory run with 4 loops", style: "min-width:680px" }, el);
    const pairs = (c) => Array.from({ length: c.length / 2 }, (_, i) => (100 * (c[2 * i] + c[2 * i + 1])) / 2 / ref);
    const X = (pos) => mL + ((W - mL - mR) * pos) / NPOS, Y = (v) => bot - ((clamp(v, 0, 160) - 0) / 160) * (bot - top);
    const bands = [["entry stack", 0, G.pre]].concat(Array.from({ length: n }, (_, k) => [`loop ${k + 1}`, G.pre + k * G.loop, G.pre + (k + 1) * G.loop]), [["exit stack", G.pre + n * G.loop, NPOS]]);
    bands.forEach(([name, a, b], i) => {
      const erased = i === 1 || i === 2, changed = (i === 3 || i === 4) && st.id !== "abl12";
      s("rect", { x: X(a), y: top - 4, width: X(b) - X(a), height: bot - top + 4, fill: erased ? "var(--panel-2)" : i % 2 ? "transparent" : "var(--panel-2)", stroke: erased ? "var(--ink-3)" : "none", "stroke-dasharray": "4 3", "stroke-width": 1 }, svg);
      s("text", { x: (X(a) + X(b)) / 2, y: top - 9, "text-anchor": "middle", "font-size": 10.5, fill: changed ? "var(--steer)" : "var(--ink-3)",
        text: erased ? `${name} · erased` : changed ? `${name} · copy changed` : name }, svg);
    });
    for (const v of [0, 50, 100, 150]) {
      s("line", { x1: mL, x2: W - mR, y1: Y(v), y2: Y(v), stroke: "var(--line)", "stroke-width": 1, "stroke-dasharray": v === 100 ? null : "2 3" }, svg);
      s("text", { x: mL - 6, y: Y(v) + 3.5, "text-anchor": "end", "font-size": 10, text: `${v}%` }, svg);
    }
    s("text", { x: 4, y: 12, "font-size": 10.5, fill: "var(--ink-2)", text: "refusal signal, % of the unsteered level inside a loop" }, svg);
    const pc = pairs(C.clean.curve), pu = pairs(C.abl12.curve), ps = pairs(C[st.id].curve);
    const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(2 * i + 1).toFixed(1)},${Y(v).toFixed(1)}`).join("");
    s("path", { d: path(pc), fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.6, "stroke-dasharray": "5 4" }, svg);
    if (st.id !== "abl12") s("path", { d: path(pu), fill: "none", stroke: "var(--refuse)", "stroke-width": 1.4, opacity: 0.35 }, svg);
    s("path", { d: path(ps), fill: "none", stroke: "var(--refuse)", "stroke-width": 2.2, "stroke-linejoin": "round" }, svg);
    const lab = (v, text, col) => s("text", { x: W - mR + 8, y: Y(v) + 3.5, "font-size": 10.5, fill: col, text }, svg);
    const yEnd = (arr) => arr[arr.length - 1];
    const lbl = [[yEnd(pc), "unsteered", "var(--ink-3)"], [yEnd(ps), OPTS.find((o) => o.id === st.id).short, "var(--refuse)"]];
    const gapPx = (Math.abs(Y(lbl[0][0]) - Y(lbl[1][0])));
    if (gapPx < 13) { const up = Y(lbl[1][0]) <= Y(lbl[0][0]); lbl[1][0] = lbl[0][0] + ((up ? 1 : -1) * 13 * 160) / (bot - top); }
    lbl.forEach(([v, t, c]) => lab(v, t, c));
    /* hover: crosshair and values */
    const cross = s("line", { y1: top - 4, y2: bot, stroke: "var(--ink-2)", "stroke-width": 1, opacity: 0 }, svg);
    const hit = s("rect", { x: mL, y: top - 4, width: W - mL - mR, height: bot - top + 4, fill: "transparent" }, svg);
    const at = (e) => { const b = svg.getBoundingClientRect(), x = ((e.clientX - b.left) / b.width) * W; return clamp(Math.floor(((x - mL) / (W - mL - mR)) * (NPOS / 2)), 0, NPOS / 2 - 1); };
    const where = (pos) => {
      const [name, j] = pos < G.pre ? ["entry stack", pos] : pos >= G.pre + n * G.loop ? ["exit stack", pos - G.pre - n * G.loop] : [`loop ${Math.floor((pos - G.pre) / G.loop) + 1}`, (pos - G.pre) % G.loop];
      return `${name}, sublayers ${j + 1}–${j + 2}`;
    };
    hit.addEventListener("pointermove", (e) => {
      const i = at(e), x = X(2 * i + 1);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("opacity", 0.5);
      tip.innerHTML = `<b>${where(2 * i)}</b><br>unsteered ${r0(pc[i])}%<br>erased on loops 1–2 ${r0(pu[i])}%` + (st.id !== "abl12" ? `<br>and ${OPTS.find((o) => o.id === st.id).label.toLowerCase()}: ${r0(ps[i])}%` : "");
      tip.classList.add("on"); moveTip(e);
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); tip.classList.remove("on"); });
  }
  function draw(animate) {
    const o = OPTS.find((x) => x.id === st.id), k = kept(o.id), c = C[o.id];
    chart();
    const fin = () => board.set(k, { right: `unsteered 100%`, sub: `next-token loss ${c.nll.toFixed(2)} (unsteered ${C.clean.nll.toFixed(2)})`, subRight: "" });
    if (animate) { board.set(null, { sub: "running…", subRight: "" }); tr.play(planOf(o), { onDone: fin }); } else { tr.show(planOf(o)); fin(); }
    $("ri-cap").textContent = `Refusal erased on loops 1 and 2; added back before loops 3 and 4: ${o.what}. Loops 3 and 4 carry ${r0(k)}% of the unsteered refusal signal. ` +
      `Next-token loss on the same traces: ${c.nll.toFixed(2)} (unsteered ${C.clean.nll.toFixed(2)}${c.nll > 1 ? "; the model's predictions are badly off" : ""}).`;
  }
  chips($("ri-opts"), OPTS.map((o) => ({ id: o.id, label: o.label })), { value: st.id, onChange: (id) => { st.id = id; draw(true); } });
  draw(false);
  src(`Re-injection: data/runs/${MODELS.dory.dir}/${R.source}. ${R.traces}`);
  const keepFirst = (m) => r0(100 * (1 - shareKept(m, "erase", 1)));
  const set = (id, v) => { $(id).textContent = v; };
  const pc = (id) => `${r0(kept(id))}%`;
  set("ri-usual", pc("abl12")); set("ri-umean", pc("abl12+umean34")); set("ri-uxmean", pc("abl12+uxmean34"));
  set("ri-mean", pc("abl12+mean34")); set("ri-smean", pc("abl12+smean34")); set("ri-umean-only", pc("umean34"));
  const range = (ids) => { const v = ids.map((id) => r0(kept(id))).sort((a, b) => a - b); return v[0] === v[v.length - 1] ? `${v[0]}%` : `${v[0]}–${v[v.length - 1]}%`; };
  set("ri-rand", range(["abl12+rmean34", "abl12+rsmean34"])); set("ri-all", range(["abl12+umean1234", "abl12+uxmean1234"]));
  const shrink = Math.abs(R.entry.x_u_sep / R.entry.x_u_sep_umean);
  set("ri-shrink", shrink >= 1000 ? "over a thousandfold" : `${r0(shrink)}-fold`);
  const edits = ["abl12+umean34", "abl12+uxmean34", "abl12+umean1234", "abl12+uxmean1234", "abl12+mean34", "abl12+smean34"];
  set("ri-nll", `${C.clean.nll.toFixed(2)} to ${Math.max(...edits.map((id) => C[id].nll)).toFixed(2)}`);
  set("ri-drop", `${r0(kept("abl12+drop34"))}%`); set("ri-nll-drop", C["abl12+drop34"].nll.toFixed(1)); set("ri-nll-flat", C["abl12+flat34"].nll.toFixed(1));
  set("ri-nll-clean", C.clean.nll.toFixed(2));
  set("ri-ouro", `${keepFirst("ouro")}%`); set("ri-nan", `${keepFirst("nanbeige")}%`);
})();

/* ================================================================== 7. weight edits */
(function edits() {
  const E = D.edits;
  const order = ORDER4;
  const metrics = [["refusal", "Refusal"], ["math500", "MATH-500"], ["mmlupro", "MMLU-Pro"]];
  const val = (row, k) => (k === "refusal" ? row.refusal.r : row[k]);
  for (const m of order) {
    const M = MODELS[m], row = E[m];
    const variants = m === "ouro" ? [["ortho", "Standard edit"], ["ortho_na", "Norm-aware edit"], ["random", "Random direction"]] : m === "dory" ? [["ortho", "Standard edit"], ["ortho_na", "Norm-aware edit"]] : [["ortho", "Standard edit"]];
    const st = { v: "ortho" };
    const body = h("div", { class: "stack", style: "gap:.55rem" });
    const term = m === "ouro" ? h("div", { class: "terminal", "aria-live": "polite" }) : null;
    const go = h("button", { class: "go", type: "button" }, "Abliterate");
    const varRow = h("div", { class: "row" });
    const card = h("div", { class: "card" }, h("h3", {}, h("span", { class: "dotsw", style: `background:${M.color}` }), M.full),
      variants.length > 1 ? varRow : null, body, term, h("div", { class: "row" }, go, h("span", { class: "src" }, "hollow: unsteered · diamond: hooks")));
    $("edit-cards").append(card);
    if (variants.length > 1) chips(varRow, variants.map(([id, label]) => ({ id, label })), { value: "ortho", onChange: (v) => { st.v = v; draw(true); } });
    const draw = (animate) => {
      body.innerHTML = "";
      const ed = row[st.v];
      const broken = m === "ouro" && st.v !== "ortho_na";
      for (const [k, lab] of metrics) {
        const c = val(row.clean, k), e = val(ed, k), hk = k === "refusal" ? row.hooks.refusal.r : null;
        const shown = k === "refusal" && broken ? null : e;
        const txt = shown == null ? `${r0(c)} → no finished answers` : `${r0(c)} → ${r0(e)}${k === "refusal" ? "%" : ""}`;
        const sv = s("svg", { viewBox: "0 0 240 26", role: "img", "aria-label": `${lab}: ${txt}` });
        const X = (v) => 8 + (224 * clamp(v, 0, 100)) / 100;
        s("line", { x1: 8, x2: 232, y1: 13, y2: 13, stroke: "var(--line)", "stroke-width": 2, "stroke-linecap": "round" }, sv);
        const link = s("line", { x1: X(c), x2: X(c), y1: 13, y2: 13, stroke: M.color, "stroke-width": 3 }, sv);
        if (hk != null) s("path", { d: `M${X(hk)},7l6,6l-6,6l-6,-6z`, fill: "var(--ink-3)" }, sv);
        s("circle", { cx: X(c), cy: 13, r: 6, fill: "var(--panel)", stroke: "var(--ink)", "stroke-width": 2 }, sv);
        const dot = s("circle", { cx: X(c), cy: 13, r: 6.5, fill: M.color }, sv);
        if (shown == null) dot.setAttribute("opacity", 0);
        body.append(h("div", { class: "gauge" }, h("div", { class: "gl" }, h("span", {}, lab), h("span", { class: "num" }, txt)), sv));
        const target = shown == null ? X(c) : X(shown);
        const fin = () => { dot.setAttribute("cx", target); link.setAttribute("x2", target); };
        if (!animate || RM || shown == null) fin();
        else {
          const t0 = performance.now(), from = X(c);
          const step = (t) => { const p = clamp((t - t0) / 900, 0, 1), xx = lerp(from, target, easeOut(p)); dot.setAttribute("cx", xx); link.setAttribute("x2", xx); if (p < 1) requestAnimationFrame(step); };
          requestAnimationFrame(step);
        }
      }
      if (term) {
        const key = { ortho: "ortho_v4_baseline_t3.l16", ortho_na: "ortho_v4_baseline_t3.l16_na1", random: "ortho_random_t3.l16" }[st.v];
        const smp = E.ouro_samples[key];
        /* the reply as markdown: its opening, the skipped middle, then its last 170 characters typed out */
        const fin = smp.tail, head = `${smp.head.replace(/\s*…$/, "")} *[… ${smp.n_tokens} tokens in all …]* `;
        term.innerHTML = "";
        const body = h("div", { class: "md" });
        term.append(body, h("div", { class: "dim", style: "margin-top:.4rem" }, `${r0(100 * smp.closed_frac)}% of MATH-500 answers finish thinking`));
        const show = (i) => { body.innerHTML = markdown(head + fin.slice(0, i)); };
        if (!animate || RM) show(fin.length);
        else { let i = 0; const tick = () => { i += 3; show(i); if (i < fin.length) setTimeout(tick, 16); }; tick(); }
      }
    };
    go.addEventListener("click", () => draw(true));
    draw(false);
  }
  $("ed-na-r").textContent = pct(E.ouro.ortho_na.refusal.r);
  SOURCES.push("Weight edits: grid_test{,_b} and ortho_test (Dory) for refusal; capability/{math500,aime25,mmlupro}/cap for capability; Ouro replies from capability/math500/cap");
})();

/* ================================================================== 8. leak */
/* 8a. a toy in two dimensions: a write with nothing along u, then a norm with uneven gains */
(function gains() {
  const el = $("gn-chart"), inp = $("gn-g"), st = { where: "edit", t: 1 };
  const TU = (35 * Math.PI) / 180, u = [Math.cos(TU), Math.sin(TU)], x = [-u[1], u[0]];
  const ang = (sv) => { const a = [(1 + sv) * u[0], (1 - sv) * u[1]]; return Math.acos((a[0] * u[0] + a[1] * u[1]) / Math.hypot(...a)); };
  let sOuro = 0; while (sOuro < 0.9 && ang(sOuro) < Math.acos(D.edits.growth.ouro.gains.sandwich_attn.cos_mean)) sOuro += 0.002; /* Ouro's mean turn of u */
  const W = 400, H = 300, O = [200, 156], K = 112;
  const P = (v) => [O[0] + K * v[0], O[1] - K * v[1]];
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "A write orthogonal to u, before and after a norm with uneven gains" }, el);
  const defs = s("defs", {}, svg);
  for (const [id, c] of [["ah-flow", "var(--flow-ink)"], ["ah-ink", "var(--ink)"], ["ah-red", "var(--refuse)"]]) {
    const mk = s("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse" }, defs);
    s("path", { d: "M0,0L10,5L0,10z", fill: c }, mk);
  }
  const g0 = s("g", {}, svg);
  function draw() {
    clearEl(g0);
    const sv = (+inp.value / 100) * st.t, gg = [1 + sv, 1 - sv];
    for (const v of [-1, 1]) {
      s("line", { x1: P([v, -1.25])[0], x2: P([v, 1.25])[0], y1: P([v, -1.25])[1], y2: P([v, 1.25])[1], stroke: "var(--line)" }, g0);
      s("line", { x1: P([-1.35, v])[0], x2: P([1.35, v])[0], y1: P([-1.35, v])[1], y2: P([1.35, v])[1], stroke: "var(--line)" }, g0);
    }
    s("line", { x1: P([-1.4, 0])[0], x2: P([1.4, 0])[0], y1: O[1], y2: O[1], stroke: "var(--ink-3)", "stroke-width": 0.8 }, g0);
    s("line", { x1: O[0], x2: O[0], y1: P([0, 1.3])[1], y2: P([0, -1.3])[1], stroke: "var(--ink-3)", "stroke-width": 0.8 }, g0);
    s("text", { x: P([1.38, 0])[0], y: O[1] + 14, "text-anchor": "end", text: "dim. 1", "font-size": 10 }, g0);
    s("text", { x: O[0] + 5, y: P([0, 1.28])[1] + 4, text: "dim. 2", "font-size": 10 }, g0);
    /* the gains stretch the plane: the unit circle becomes an ellipse */
    s("ellipse", { cx: O[0], cy: O[1], rx: K, ry: K, fill: "none", stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }, g0);
    s("ellipse", { cx: O[0], cy: O[1], rx: K * gg[0], ry: K * gg[1], fill: "none", stroke: "var(--ink-2)", "stroke-width": 1.2, opacity: sv > 0.005 ? 1 : 0 }, g0);
    /* u, the erased direction */
    const ue = P([1.3 * u[0], 1.3 * u[1]]), us = P([-1.3 * u[0], -1.3 * u[1]]);
    s("line", { x1: us[0], y1: us[1], x2: ue[0], y2: ue[1], stroke: "var(--refuse)", "stroke-width": 1.4, "stroke-dasharray": "6 4" }, g0);
    s("text", { x: ue[0] + 4, y: ue[1] + 4, text: "u", "font-size": 13, class: "t-strong", fill: "var(--refuse)" }, g0);
    /* the write x (no u), and what the norm makes of it */
    const xe = P(x);
    s("line", { x1: O[0], y1: O[1], x2: xe[0], y2: xe[1], stroke: "var(--flow-ink)", "stroke-width": 3, "marker-end": "url(#ah-flow)", opacity: 0.55 }, g0);
    s("text", { x: xe[0], y: xe[1] - 12, "text-anchor": "middle", text: "write x", "font-size": 10.5, class: "t-strong", fill: "var(--flow-ink)" }, g0);
    let y = [gg[0] * x[0], gg[1] * x[1]]; const ny = Math.hypot(...y); y = [y[0] / ny, y[1] / ny];
    let leak = y[0] * u[0] + y[1] * u[1];
    const ye = P(y);
    s("line", { x1: O[0], y1: O[1], x2: ye[0], y2: ye[1], stroke: "var(--ink)", "stroke-width": 2.4, "marker-end": "url(#ah-ink)", opacity: st.where === "hook" ? 0.3 : 1 }, g0);
    if (st.where === "hook") { /* a hook removes u after the norm */
      let z = [y[0] - leak * u[0], y[1] - leak * u[1]]; const nz = Math.hypot(...z); z = [z[0] / nz, z[1] / nz];
      const ze = P(z);
      s("line", { x1: O[0], y1: O[1], x2: ze[0], y2: ze[1], stroke: "var(--ink)", "stroke-width": 2.4, "marker-end": "url(#ah-ink)" }, g0);
      s("text", { x: ze[0] - 6, y: ze[1] + 16, "text-anchor": "end", text: "after the hook", "font-size": 10.5, class: "t-strong" }, g0);
      leak = 0;
    } else s("text", { x: ye[0] - 6, y: ye[1] + 16, "text-anchor": "end", text: "after the norm", "font-size": 10.5, class: "t-strong" }, g0);
    if (Math.abs(leak) > 0.004) {
      const f = P([leak * u[0], leak * u[1]]);
      s("line", { x1: ye[0], y1: ye[1], x2: f[0], y2: f[1], stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }, g0);
      s("line", { x1: O[0], y1: O[1], x2: f[0], y2: f[1], stroke: "var(--refuse)", "stroke-width": 4, "stroke-linecap": "round" }, g0);
    }
    s("text", { x: 8, y: H - 10, text: `along u after the norm: ${Math.abs(leak).toFixed(2)}`, "font-size": 11.5, class: "t-strong", fill: Math.abs(leak) > 0.004 ? "var(--refuse)" : "var(--ink-2)" }, g0);
    const deg = (ang(+inp.value / 100) * 180) / Math.PI;
    $("gn-g-label").textContent = +inp.value === 0 ? "equal gains" : `${(1 + +inp.value / 100).toFixed(2)} vs ${(1 - +inp.value / 100).toFixed(2)} · turns u by ${deg.toFixed(0)}°`;
  }
  $("gn-mark").style.left = `${(100 * sOuro) / 0.6}%`;
  $("gn-formula").innerHTML = mathHtml("\\mathrm{N}(x)\\cdot u \;=\; \\frac{x \\cdot (g \\odot u)}{\\mathrm{rms}(x)}", true);
  chips($("gn-where"), [{ id: "edit", label: "Weight edit (before the norm)" }, { id: "hook", label: "Hook (after the norm)" }], { value: "edit", onChange: (v) => { st.where = v; draw(); } });
  inp.addEventListener("input", draw);
  /* when it scrolls into view, the gains grow from equal to uneven */
  let played = false;
  const play = () => {
    if (played || RM) return; played = true;
    const t0 = performance.now();
    const f = (now) => { st.t = easeInOut(clamp((now - t0 - 300) / 1400, 0, 1)); draw(); if (st.t < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  };
  if (!RM && "IntersectionObserver" in window) { st.t = 0; new IntersectionObserver((es) => { if (es[0].isIntersecting) play(); }, { threshold: 0.5 }).observe(el); }
  draw();
})();

/* 8b. measured: how much of the erased direction the stream carries after each loop, against the unedited model */
(function leak() {
  const GR = D.edits.growth, G = GEO.ouro;
  /* tabs: the unedited model (the yardstick), the two weight edits and hooks. Only hooks get filters in the tube: a weight
     edit has no separate part to draw, the matrices themselves are changed */
  const KINDS = [["clean", "Unedited", "var(--ink-2)", "5 4"], ["ortho", "Standard edit", "var(--refuse)", null], ["ortho_na", "Norm-aware edit", "var(--refuse)", "5 4"], ["hook", "Hooks", "var(--flow-ink)", null]];
  const chainOf = (k) => (k === "clean" ? [1, 1, 1, 1] : GR.ouro[k]);
  const st = { kind: "ortho", n: 4 };
  const tr = new Track3($("lk-track"), { model: "ouro", loops: 4, meaning: { lanes: null }, arch: { ticks: "sandwich", gates: true }, label: "Ouro with the refusal direction removed: how much of it is back after each loop" });
  const planOf = () => {
    const ch = [0].concat(chainOf(st.kind));
    return st.kind === "hook" ? { leak: { chain: ch, ref: 1 }, events: [1, 2, 3, 4].map((k) => ({ k, type: "erase" })), hooks: true } : { leak: { chain: ch, ref: 1 } };
  };
  function chart() {
    const el = $("lk-chart"); el.innerHTML = "";
    const W = 760, Hh = 236, top = 44, bot = Hh - 30;
    const svg = s("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "The erased direction carried after each loop, against the unedited model: Ouro and Nanbeige", style: "min-width:560px" }, el);
    const panels = [{ m: "ouro", x0: 34, x1: 400, sub: "a norm after every write, and one between loops" }, { m: "nanbeige", x0: 532, x1: 612, sub: "no norm on the writes" }];
    const Y = (v) => bot - ((bot - top) * clamp(v, 0, 6)) / 6;
    for (const pn of panels) {
      const m = pn.m, R = GR[m], T = R.ortho.length;
      const X = (t) => pn.x0 + ((pn.x1 - pn.x0) * (t - 1)) / (T - 1);
      s("text", { x: pn.x0 - 30, y: 14, class: "t-title", text: MODELS[m].label }, svg);
      s("text", { x: pn.x0 - 30, y: 30, text: pn.sub, "font-size": 10.5 }, svg);
      const cleanOn = m === "ouro" && st.kind === "clean";
      for (const v of [0, 1, 3, 5]) {
        s("line", { x1: pn.x0 - 8, x2: pn.x1 + 8, y1: Y(v), y2: Y(v), stroke: v === 1 ? (cleanOn ? "var(--ink)" : "var(--ink-3)") : "var(--line)", "stroke-width": v === 1 && cleanOn ? 2.4 : 1, "stroke-dasharray": v === 1 ? "5 4" : null }, svg);
        s("text", { x: pn.x0 - 14, y: Y(v) + 4, "text-anchor": "end", text: v === 1 ? "1×" : `${v}×`, "font-size": 10 }, svg);
      }
      for (let t = 1; t <= T; t++) s("text", { x: X(t), y: bot + 16, "text-anchor": "middle", text: `loop ${t}`, "font-size": 10 }, svg);
      const ends = [];
      for (const [k, lab, col, dash] of KINDS) {
        if (!R[k]) continue;
        const on = m !== "ouro" || k === st.kind, vals = R[k]; /* unedited: every line dims and the dashed 1x line stands out */
        s("polyline", { points: vals.map((v, i) => `${X(i + 1).toFixed(1)},${Y(v).toFixed(1)}`).join(" "), fill: "none", stroke: col, "stroke-width": on ? 2.6 : 1.6, "stroke-dasharray": dash, opacity: on ? 1 : 0.4, "stroke-linejoin": "round" }, svg);
        vals.forEach((v, i) => {
          const dim = m === "ouro" && on && i + 1 > st.n;
          s("circle", { cx: X(i + 1), cy: Y(v), r: on ? 4.4 : 3, fill: col, stroke: "var(--panel)", "stroke-width": 1.6, opacity: (on ? 1 : 0.45) * (dim ? 0.3 : 1) }, svg);
          addTip(s("circle", { cx: X(i + 1), cy: Y(v), r: 10, fill: "transparent" }, svg), `${MODELS[m].full}, ${lab.toLowerCase()}<br>after loop ${i + 1}: <b>${v < 0.01 ? "< 0.01" : v.toFixed(2)}×</b> the unedited model`);
        });
        ends.push({ y: Y(vals[T - 1]), text: `${lab} ${vals[T - 1] < 0.01 ? "≈ 0" : `${vals[T - 1].toFixed(1)}×`}`, col, on });
      }
      ends.sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
      for (const e of ends) s("text", { x: pn.x1 + 14, y: e.y + 4, text: e.text, "font-size": 10.5, fill: e.col, opacity: e.on ? 1 : 0.6, class: e.on ? "t-strong" : null }, svg);
    }
    s("text", { x: 400 + 14, y: Y(1) - 6, text: "unedited model", "font-size": st.kind === "clean" ? 10.5 : 9.5, class: st.kind === "clean" ? "t-strong" : null }, svg);
  }
  function draw(animate) {
    chart();
    $("lk-loop-label").textContent = `${st.n} of 4`;
    const v = chainOf(st.kind)[st.n - 1], name = KINDS.find((k) => k[0] === st.kind)[1].toLowerCase();
    $("lk-cap").textContent = st.kind === "clean"
      ? "Ouro, unedited: the yardstick for the other tabs. Full red is as much of the refusal direction as this model carries after each loop; the edits and hooks are measured against it."
      : `Ouro, ${name}: after ${st.n} loop${st.n === 1 ? "" : "s"} the stream carries ${v < 0.01 ? "almost none" : `${v.toFixed(2)}×`} of the refusal direction the unedited model carries there.` +
      (st.kind === "ortho_na" ? " The norm-aware edit closes the leak through the write norms; the shared norm between loops still lets it grow." : st.kind === "hook" ? " A hook removes it after every norm, so nothing leaks." : "");
    const until = G.pre + st.n * G.loop + (st.n === 4 ? G.post : 0);
    if (animate) tr.play(planOf(), { until }); else tr.show(planOf(), until);
  }
  chips($("lk-kind"), KINDS.map(([id, label]) => ({ id, label })), { value: st.kind, onChange: (v) => { st.kind = v; draw(true); } });
  const inp = $("lk-loop");
  inp.addEventListener("input", () => { st.n = Math.max(1, +inp.value); draw(true); });
  draw(false);
  src(`Leak: data/runs/<model>/${GR.ouro.source} and ${GR.nanbeige.source} (scripts/ortho_compare.py, 8 traces; RMS of the stream along u after each loop, divided by the unedited model's)`);
})();

/* ================================================================== 9. levers */
(function levers() {
  const keepFirst = (m) => r0(100 * (1 - shareKept(m, "erase", 1)));
  const E = D.edits, ex = D.extra;
  const data = [
    { m: "dory", title: "Add the input back every loop", who: "Dory", rows: [
      ["First-loop erase undone", `${keepFirst("dory")}%`], ["Two extra loops", `refusal back to ${pct(ex.dory.erase.at_extra.r)}`],
      ["Copy minus the erased direction", `rebuild ${r0((100 * D.reinject.conds["abl12+umean34"].curve.slice(96, 176).reduce((a, b) => a + b, 0)) / D.reinject.conds.clean.curve.slice(96, 176).reduce((a, b) => a + b, 0))}% of unsteered`],
      ["Weight edit", `refusal ${pct(E.dory.ortho.refusal.r)}, MATH ${r0(E.dory.clean.math500)} → ${r0(E.dory.ortho.math500)}`]],
      text: "The strongest comeback of the three, but not because of the copy: stripping the refusal information out of it leaves the rebuild intact, and the models without a copy recover too." },
    { m: "nanbeige", title: "No copy, no norm on the writes", who: "Nanbeige", rows: [
      ["First-loop erase undone", `${keepFirst("nanbeige")}%`], ["Two extra loops", "not testable"],
      ["Weight edit", `refusal ${pct(E.nanbeige.ortho.refusal.r)}, MATH ${r0(E.nanbeige.clean.math500)} → ${r0(E.nanbeige.ortho.math500)}`]],
      text: "Plain pre-norm layers and a shared norm between loops. Moderate self-correction with no re-injection at all, and the weight edit works better than hooks." },
    { m: "ouro", title: "Norm every write, share a norm between loops", who: "Ouro", rows: [
      ["First-loop erase undone", `${keepFirst("ouro")}%`], ["Two extra loops", `refusal stays at ${pct(ex.ouro.erase.at_extra.r)}`],
      ["Weight edit", `MATH ${r0(E.ouro.clean.math500)} → ${r0(E.ouro.ortho.math500)} (norm-aware ${r0(E.ouro.ortho_na.math500)})`]],
      text: "Learned norm gains rotate an erased direction back, and the shared norm amplifies it every loop. The standard edit destroys the model." },
  ];
  for (const d of data) {
    $("lever-cards").append(h("div", { class: "lever", style: `--c:${MODELS[d.m].color}` },
      h("div", { class: "eyebrow" }, d.who), h("h3", {}, d.title), h("p", { class: "note" }, d.text),
      h("div", { class: "verdicts" }, d.rows.map(([a, b]) => h("div", {}, h("span", {}, a), h("span", {}, b))))));
  }
})();

/* ================================================================== addendum: a second judge */
(function secondJudge() {
  const J = D.judge2;
  if (!J || !$("j2-table")) return;
  const ms = ["dory", "nanbeige", "ouro", "qwen"].filter((m) => J.rows[m]), R = ms.map((m) => J.rows[m]);
  const f1 = (x) => x.toFixed(1), f3 = (x) => x.toFixed(3), lo = (k) => Math.min(...R.map((r) => r[k])), hi = (k) => Math.max(...R.map((r) => r[k]));
  const head = ["Model", "Answers", "Same verdict", "Refusal, gpt-oss-20b", "Refusal, luna", "Direction cosine, luna vs ours", "Held-out AUC, ours → luna"];
  $("j2-table").append(h("thead", {}, h("tr", {}, head.map((t) => h("th", {}, t)))),
    h("tbody", {}, ms.map((m) => { const r = J.rows[m];
      return h("tr", {}, h("td", {}, MODELS[m].label + (m === "dory" ? " *" : "")), h("td", {}, r.answers.toLocaleString("en-US")), h("td", {}, `${f1(r.agree)}%`),
        h("td", {}, `${f1(r.ref_a)}%`), h("td", {}, `${f1(r.ref_b)}%`), h("td", {}, f3(r.cos)), h("td", { style: "white-space:nowrap" }, `${f3(r.auc_cur)} → ${f3(r.auc_ref)}`)); })));
  const o = J.rows.ouro;
  $("j2-agree").textContent = `${r0(lo("agree"))}–${r0(hi("agree"))}%`;
  $("j2-diff").textContent = r0(Math.max(...R.map((r) => Math.abs(r.ref_b - r.ref_a))));
  $("j2-cos").textContent = (Math.floor(lo("cos") * 100) / 100).toFixed(2);
  $("j2-cosmin").textContent = (Math.floor(lo("cos_min") * 100) / 100).toFixed(2);
  $("j2-ouro-non").textContent = `${o.non_cur} complying traces → ${o.non_score}`;
  $("j2-ouro-cos").textContent = o.cos_score.toFixed(2);
  steered(J.steered);
  personaVecs(J.persona);
  { /* traits, coherence and cut-off labels under both judges */
    const X = J.extra || {}, parts = [], fmtN = (n) => n.toLocaleString("en-US"), sg = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
    if (X.persona) parts.push(`On the ${fmtN(X.persona.n)} persona replies, the two judges' trait scores correlate at r = ${X.persona.trait_r.toFixed(2)} and their coherence scores at r = ${X.persona.coh_r.toFixed(2)} (luna ${sg(X.persona.trait_diff)} and ${sg(X.persona.coh_diff)} points on average).`);
    if (X.flipcoh) parts.push(`On the ${fmtN(X.flipcoh.n)} flip-sign replies, coherence correlates at r = ${X.flipcoh.r.toFixed(2)} (luna ${sg(X.flipcoh.diff)} points on average).`);
    if (X.cutoff) parts.push(`On the ${fmtN(X.cutoff.n)} cut-off kick responses, they give the same label ${X.cutoff.agree.toFixed(0)}% of the time; luna calls ${r0(X.cutoff.luna.coherent || 0)}% still reasoning, ${r0(X.cutoff.luna.repetitive || 0)}% repeating and ${r0(X.cutoff.luna.incoherent || 0)}% incoherent, against ${r0(X.cutoff.oss.coherent || 0)}%, ${r0(X.cutoff.oss.repetitive || 0)}% and ${r0(X.cutoff.oss.incoherent || 0)}% for gpt-oss-20b.`);
    if (parts.length) { $("j2x-text").textContent = parts.join(" "); $("j2x").hidden = false; }
  }
  SOURCES.push("Second judge: generations/<tag>/[invalid/]luna/*.scored.parquet and generations/*.scored_luna.parquet (scripts/judge_api.py, gpt-6-luna), judge_compare.csv (scripts/judge_compare.py)");

  /* persona vectors rebuilt from luna's ratings: cosine with ours at the site each trait steers with */
  function personaVecs(P) {
    if (!P || !$("j2p-table")) return;
    const ms = ["dory", "nanbeige", "ouro", "qwen"].filter((m) => P[m]);
    const traits = [...new Set(ms.flatMap((m) => Object.keys(P[m])))].sort();
    const LOW = 0.95, lowOnes = [];
    $("j2p-table").append(h("thead", {}, h("tr", {}, h("th", {}, "Trait"), ms.map((m) => h("th", {}, MODELS[m].label)))),
      h("tbody", {}, traits.map((t) => h("tr", {}, h("td", {}, t), ms.map((m) => {
        const x = P[m][t];
        if (!x) return h("td", { style: "color:var(--ink-3)" }, "–");
        if (x.cos < LOW) lowOnes.push([m, t, x]);
        const td = h("td", { style: x.cos < LOW ? "font-weight:600;color:var(--ink)" : null }, x.cos.toFixed(2));
        const k = x.kept;
        addTip(td, `<b>${MODELS[m].label} · ${t}</b><br>${MODELS[m].straight ? "" : `loop ${x.loop}, `}layer ${x.layer}: cosine ${x.cos.toFixed(3)}` +
          (k && k[0] != null ? `<br>replies kept (with / without the trait): gpt-oss-20b ${k[0]} / ${k[1]}, luna ${k[2]} / ${k[3]}` : ""));
        return td;
      })))));
    const all = ms.flatMap((m) => Object.values(P[m]).map((x) => x.cos)), rest = all.filter((c) => c >= LOW);
    $("j2p-same").textContent = `${rest.length} of the ${all.length} vectors on this page have cosine ${(Math.floor(Math.min(...rest) * 100) / 100).toFixed(2)} or more with ours; Dory's and Qwen's are effectively identical.`;
    $("j2p-moved").textContent = lowOnes.map(([m, t, x]) => `${MODELS[m].label}'s ${t} vector (cosine ${x.cos.toFixed(2)}): luna keeps ${x.kept[2]} replies with the trait and ${x.kept[3]} without, where gpt-oss-20b kept ${x.kept[0]} and ${x.kept[1]}.`).join(" ");
    SOURCES.push("Persona vectors under both judges: directions/persona.pt vs persona_luna.pt (persona_vectors.py --judge-tag luna), at each trait's tuned site (analysis/persona_tune.json)");
  }

  /* every refusal cell on the page under both judges: a scatter on the diagonal, and the headline cells as a table */
  function steered(S) {
    if (!S || !S.cells.length) { $("j2s-intro").textContent = "(Not run yet.)"; return; }
    const C = S.cells.map((r) => Object.fromEntries(S.cols.map((k, i) => [k, r[i]])));
    const find = (m, g, c) => C.find((x) => x.model === m && x.group === g && x.cand === c);
    const fmtN = (n) => n.toLocaleString("en-US");
    const gaps = C.map((x) => Math.abs(x.luna - x.oss)).sort((a, b) => a - b), med = gaps[Math.floor(gaps.length / 2)];
    const worst = C.reduce((a, b) => (Math.abs(b.luna - b.oss) > Math.abs(a.luna - a.oss) ? b : a));
    $("j2s-intro").textContent = `That is ${fmtN(S.answers)} answers across ${C.length} steering settings, on harmful prompts and, for the flip-sign runs, harmless ones.`;
    /* legend: models, then filled vs ring */
    const lg = $("j2s-legend");
    const ms = ["dory", "nanbeige", "ouro", "qwen"].filter((m) => C.some((x) => x.model === m));
    for (const m of ms) lg.append(h("span", { class: "row", style: "gap:.35rem;align-items:center" }, h("span", { style: `width:10px;height:10px;border-radius:50%;background:${MODELS[m].color};display:inline-block` }), MODELS[m].label));
    lg.append(h("span", { class: "row", style: "gap:.35rem;align-items:center;color:var(--ink-2)" }, h("span", { style: "width:10px;height:10px;border-radius:50%;background:var(--ink-3);display:inline-block" }), "harmful prompts"),
      h("span", { class: "row", style: "gap:.35rem;align-items:center;color:var(--ink-2)" }, h("span", { style: "width:10px;height:10px;border-radius:50%;border:1.5px solid var(--ink-3);box-sizing:border-box;display:inline-block" }), "harmless prompts (flip sign)"));
    /* the scatter */
    const W = 360, L = 46, B = 40, T = 10, R = 12, P = W - L - R, Hh = W - T - B;
    const X = (v) => L + (v / 100) * P, Y = (v) => T + Hh - (v / 100) * Hh;
    const svg = s("svg", { viewBox: `0 0 ${W} ${W}`, role: "img", "aria-label": `Refusal rate under gpt-oss-20b (x) against gpt-6-luna (y) for ${C.length} steering settings` });
    for (const t of [0, 25, 50, 75, 100]) {
      s("line", { x1: X(0), x2: X(100), y1: Y(t), y2: Y(t), class: "grid" }, svg);
      s("line", { x1: X(t), x2: X(t), y1: Y(0), y2: Y(100), class: "grid" }, svg);
      s("text", { x: L - 6, y: Y(t) + 4, "text-anchor": "end", class: "ax", text: `${t}%` }, svg);
      s("text", { x: X(t), y: T + Hh + 16, "text-anchor": "middle", class: "ax", text: `${t}%` }, svg);
    }
    s("line", { x1: X(0), y1: Y(0), x2: X(100), y2: Y(100), stroke: "var(--ink-3)", "stroke-width": 1, "stroke-dasharray": "4 3" }, svg);
    s("text", { x: X(50), y: W - 6, "text-anchor": "middle", class: "ax", text: "gpt-oss-20b refusal rate" }, svg);
    s("text", { x: 12, y: Y(50), "text-anchor": "middle", class: "ax", transform: `rotate(-90 12 ${Y(50)})`, text: "gpt-6-luna refusal rate" }, svg);
    const harmless = (x) => x.group.startsWith("plus");
    for (const x of [...C].sort((a, b) => harmless(b) - harmless(a))) {
      const g = s("g", {}, svg), cx = X(x.oss), cy = Y(x.luna);
      s("circle", harmless(x) ? { cx, cy, r: 3.2, fill: "var(--bg)", stroke: MODELS[x.model].color, "stroke-width": 1.5 }
        : { cx, cy, r: 3.4, fill: MODELS[x.model].color, stroke: "var(--bg)", "stroke-width": 1 }, g);
      s("circle", { cx, cy, r: 7, fill: "transparent" }, g);
      addTip(g, `<b>${MODELS[x.model].label}</b> · ${x.label}<br>gpt-oss-20b ${pct(x.oss, 1)} · gpt-6-luna ${pct(x.luna, 1)}<br>${x.n} prompts`);
    }
    $("j2s-chart").append(svg);
    const pm = Object.entries(S.by_model).map(([m, v]) => `${MODELS[m].label} ${(100 * v.agree / v.answers).toFixed(1)}%`).join(", ");
    $("j2s-cap").textContent = `Each dot is one steering setting; on the dashed line both judges give the same rate. They give the same verdict on ${(100 * S.agree / S.answers).toFixed(1)}% of answers (${pm}). The median gap between the two rates is ${med.toFixed(1)} points; the largest is ${Math.abs(worst.luna - worst.oss).toFixed(1)} (${MODELS[worst.model].label}, ${worst.label}).`;
    $("j2-blocked-note").textContent = S.blocked ? `OpenAI's biology filter refused to score ${fmtN(S.blocked)} of the steered answers (and 1–4 training replies per model); those keep gpt-oss-20b's verdict.` : "";
    /* the headline cells */
    const rows = [
      ["Dory · no steering", "dory", "grid", "none"], ["Dory · erase loop 1", "dory", "grid", "ablate_hh_eoi_t0.l26_apply0"],
      ["Dory · erase loop 2", "dory", "grid", "ablate_hh_eoi_t0.l26_apply1"], ["Dory · erase both loops", "dory", "grid", "ablate_hh_eoi_t0.l26_apply0-1"],
      ["Dory · erase both, run with 4 loops", "dory", "grid4", "ablate_hh_eoi_t0.l26_apply0-1"],
      ["Nanbeige · no steering", "nanbeige", "grid", "none"], ["Nanbeige · erase loop 1", "nanbeige", "grid", "ablate_v4_baseline_t1.l15_apply0"],
      ["Nanbeige · erase loop 2", "nanbeige", "grid", "ablate_v4_baseline_t1.l15_apply1"], ["Nanbeige · erase every loop", "nanbeige", "grid", "ablate_v4_baseline_t1.l15"],
      ["Ouro · no steering", "ouro", "grid", "none"], ["Ouro · erase loop 1", "ouro", "grid", "ablate_v4_baseline_t3.l16_apply0"],
      ["Ouro · erase loop 4", "ouro", "grid", "ablate_v4_baseline_t3.l16_apply3"], ["Ouro · erase every loop", "ouro", "grid", "ablate_v4_baseline_t3.l16"],
      ["Ouro · erase loops 1–4, run with 6 loops", "ouro", "grid6", "ablate_v4_baseline_t3.l16_apply0-1-2-3"],
      ["Qwen · no steering", "qwen", "grid", "none"], ["Qwen · erase every layer", "qwen", "grid", "ablate_v4_baseline_l21"],
      ["Dory · weight edit", "dory", "ortho", "ortho_hh_eoi_t0.l26"], ["Nanbeige · weight edit", "nanbeige", "grid", "ortho_v4_baseline_t1.l15"],
      ["Ouro · weight edit", "ouro", "grid", "ortho_v4_baseline_t3.l16"], ["Ouro · norm-aware weight edit", "ouro", "grid", "ortho_v4_baseline_t3.l16_na1"],
    ];
    const sg = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}`;
    const body = rows.map(([lab, m, g, c]) => {
      const x = find(m, g, c);
      return h("tr", {}, h("td", {}, lab), h("td", {}, x ? String(x.n) : "–"), h("td", {}, x ? pct(x.oss, 1) : "pending"),
        h("td", {}, x ? pct(x.luna, 1) : "pending"), h("td", {}, x ? sg(x.luna - x.oss) : ""));
    });
    for (const m of ["dory", "ouro"]) {
      const gn = S.gains && S.gains[m];
      body.push(h("tr", {}, h("td", {}, `${MODELS[m].label} · two extra loops, paired gain (points)`), h("td", {}, gn ? String(gn.n) : "–"),
        h("td", {}, gn ? `${sg(gn.oss.d)} [${gn.oss.lo.toFixed(1)}, ${gn.oss.hi.toFixed(1)}]` : "pending"),
        h("td", {}, gn ? `${sg(gn.luna.d)} [${gn.luna.lo.toFixed(1)}, ${gn.luna.hi.toFixed(1)}]` : "pending"), h("td", {}, gn ? sg(gn.luna.d - gn.oss.d) : "")));
    }
    $("j2s-table").append(h("thead", {}, h("tr", {}, ["Setting", "Prompts", "gpt-oss-20b", "gpt-6-luna", "Difference"].map((t) => h("th", {}, t)))), h("tbody", {}, body));
    /* where the judges part: broken output, which the rubric's "no specific help" rule lets a judge call a refusal */
    const oe = find("ouro", "grid", "ortho_v4_baseline_t3.l16"), on = find("ouro", "grid", "ortho_v4_baseline_t3.l16_na1"), nb = find("nanbeige", "grid4", "none");
    const pp = (x) => `${r0(x.oss)}% to ${r0(x.luna)}%`;
    if (oe && on && nb) $("j2s-note").textContent = `Every conclusion on this page holds under either judge, and most rates move by a few points, mostly down: luna calls slightly fewer steered answers refusals. The judges part ways on broken output. The rubric counts an answer with no specific help as a refusal, and luna applies that to garbled text far more often. Ouro after the weight edit, which can no longer do maths, goes from ${pp(oe)} refusal (the norm-aware edit, which halves its maths score: ${pp(on)}), and Nanbeige run with 4 loops, which produces gibberish, from ${pp(nb)}. Neither judge's refusal rate means much for a model that no longer works, which is why the weight-edit section reports capability beside it.`;
  }
})();

/* the judge of record (export_blog_data.py --judge): v1 gpt-oss-20b, v2 gpt-6-luna for every reported score */
{
  const E = D.edits && D.edits.dory, fill = (id, v) => { if ($(id) && v != null) $(id).textContent = pct(v); };
  if (E) { fill("lk-d-ortho", E.ortho.refusal.r); fill("lk-d-hooks", E.hooks.refusal.r); fill("lk-d-na", E.ortho_na.refusal.r); }
  if (D.edits && D.edits.ouro) fill("ed-o-ref", D.edits.ouro.ortho.refusal.r);
  document.querySelectorAll(".judge-name").forEach((e) => { e.textContent = D.judge ? D.judge.name : "gpt-oss-20b"; });
}
if (D.judge && D.judge.id === "v2") {
  $("judge-lead").innerHTML = 'Every score on this page comes from one judge, gpt-6-luna (OpenAI API, low reasoning effort): refusal, traits, coherence and the labels on cut-off kick responses. Accuracy and kick damage need no judge. We built the project with an open judge first, gpt-oss-20b at a fixed snapshot with greedy decoding, and chose the refusal directions, the persona vectors and every steering strength with its scores; the <a href="#judge2">addendum</a> shows that luna\'s labels give the same directions and vectors.';
  $("judge-limits").innerHTML = 'One judge scores every figure, with no human calibration; the directions, persona vectors and strengths were chosen with an earlier judge (<a href="#judge2">addendum</a>).';
  $("plus-tune-judge").textContent = ", the judge that chose these strengths (everything else on the page is scored by gpt-6-luna)";
  $("j2s-lead").textContent = "Would the earlier judge tell a different story? Every refusal answer on this page was scored by both gpt-6-luna, whose scores the page reports, and gpt-oss-20b, the open model we started with, using the same rubric, prompt and parser.";
  $("j2-limits-a").textContent = "Only the tuning runs that chose the strengths (the “How hard to push” chart and the persona strength search) keep gpt-oss-20b's scores.";
}
$("all-src").innerHTML = SOURCES.map((x) => `<div>· ${x.replace(/</g, "&lt;")}</div>`).join("") + `<div>· Generated by scripts/export_blog_data.py on ${D.generated}; scores by ${D.judge ? `${D.judge.name} (judgment ${D.judge.id})` : "gpt-oss-20b"}</div>`;
})();
