"use strict";

const params = new URLSearchParams(location.search);
const TOKEN = params.get("t") || "";
const NOWAIT = params.get("nowait") === "1" || !!params.get("selftest");
const S = {
  lang: "en",
  strings: {},
  tab: "import",
  defaults: null,
  browse: null,
  files: [],
  opts: { languages: ["es", "en"], review: "balanced", correct: "off", second: "off", formats: ["pdf", "xlsx", "json", "txt"] },
  job: null,
  after: 0,
  docs: {},
  order: [],
  cur: { doc: null, page: 1, word: null },
  pages: {},
  only: false,
  fixedOnly: false,
  reviewTab: "page",
  involved: new Set(),
  ruleFocus: null,
  editing: false,
  error: null,
  pathText: null,
  addReport: null,
  exportState: { destination: "", pattern: "{name}", formats: ["pdf", "xlsx", "json", "txt"], picked: null, result: null },
  pending: Promise.resolve(),
};

function t(key, p) {
  let text = S.strings[key];
  if (text === undefined) return key;
  if (p) text = text.replace(/\{(\w+)\}/g, (_, name) => (p[name] === undefined ? `{${name}}` : p[name]));
  return text;
}

// A text with a plural: `key_one` is used when n is 1 and the language has it.
function tn(key, n, p) {
  const one = `${key}_one`;
  return t(n === 1 && S.strings[one] !== undefined ? one : key, { n, ...p });
}

// A page can be withheld from the fixes for two causes: too many of its words would change (a share), or its language is
// not one of the chosen ones (no share): each has its own text.
function withheldCode(code, params) {
  return code === "guard.warning.fixes_withheld" && params && params.cause === "language" ? "guard.warning.fixes_withheld_language" : code;
}

// A note about a page that holds for many pages (a whole document read with the wrong language ticked) is one note that
// lists its pages, not one box per page: 300 boxes pushed the progress of the read off the screen.
const WARNING_PAGES_SHOWN = 12;
function groupedWarnings(warnings) {
  const groups = [];
  const byKey = new Map();
  for (const w of warnings) {
    const page = w.params && typeof w.params.page === "number" ? w.params.page : null;
    if (page === null) {
      groups.push({ w, pages: [] });
      continue;
    }
    const rest = { ...w.params };
    delete rest.page;
    const key = JSON.stringify([w.doc, w.code, rest]);
    if (!byKey.has(key)) {
      const group = { w, pages: [] };
      byKey.set(key, group);
      groups.push(group);
    }
    const pages = byKey.get(key).pages;
    if (!pages.includes(page)) pages.push(page);
  }
  for (const g of groups) g.pages.sort((a, b) => a - b);
  return groups;
}

function warningBox(g) {
  const params = g.pages.length ? { ...g.w.params, page: g.pages[0] } : g.w.params;
  const more = g.pages.slice(1);
  const list = more.slice(0, WARNING_PAGES_SHOWN).join(", ") + (more.length > WARNING_PAGES_SHOWN ? ", ..." : "");
  return h("div", { class: "warn" }, g.w.doc ? `${g.w.doc}: ` : "", t(g.w.code, params), more.length ? h("div", { class: "small muted", "data-more": more.length }, tn("ui.read.warning_more", more.length, { pages: list })) : null);
}

function reasonText(r) {
  const p = r.params ? { ...r.params } : null;
  if (p && typeof p.conf === "number" && p.conf <= 1) p.conf = `${Math.round(p.conf * 100)} %`;
  return t(r.key, p);
}

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "value") el.value = v;
    else if (k === "checked" || k === "disabled" || k === "selected") el[k] = !!v;
    else el.setAttribute(k, v === true ? (k.startsWith("aria-") ? "true" : "") : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  if (tag === "button") el.querySelectorAll("kbd").forEach((k) => k.setAttribute("aria-hidden", "true"));
  return el;
}

// The attributes that make a clickable element of the page (a folder, a row of a table) work with the keyboard too: it
// takes the focus, is announced as a button and is pressed with Enter or Space, which do not reach the Review shortcuts.
function pressable(action) {
  return {
    tabindex: "0",
    role: "button",
    onclick: action,
    onkeydown: (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      e.preventDefault();
      e.stopPropagation();
      action();
    },
  };
}

function link(path, extra) {
  const q = new URLSearchParams({ t: TOKEN, ...(extra || {}) });
  return `${path}${path.includes("?") ? "&" : "?"}${q}`;
}

async function api(path, body) {
  const options = { headers: { "X-Token": TOKEN } };
  if (body !== undefined) {
    options.method = "POST";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  const json = await response.json();
  if (json.error) throw Object.assign(new Error(json.error.code), json.error);
  return json;
}

function errorText(error) {
  const code = error.code || "error.internal";
  return t(code, error.params || {}) + (error.params && error.params.detail ? ` (${error.params.detail})` : "");
}

function fail(error) {
  S.error = errorText(error);
  render();
}

function fmtTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "-";
  const s = Math.round(seconds);
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${s % 60} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

function pagesText(n) {
  return n === 1 ? t("ui.pages_one") : t("ui.pages", { n });
}

function fmtSize(bytes) {
  return bytes > 1e6 ? `${(bytes / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1e3))} KB`;
}

// the page view: image, boxes, zoom and pan on a canvas

const COLORS = {
  mismatch: ["#f26d6d", "rgba(242,109,109,0.18)", 2.5, []],
  doubtful: ["#f5c451", "rgba(245,196,81,0.18)", 2, []],
  unscored: ["#9ba3ad", "rgba(155,163,173,0.10)", 1.5, [5, 3]],
  ok: ["rgba(95,208,140,0.55)", null, 1, []],
  native: ["rgba(155,163,173,0.35)", null, 1, []],
  accepted: ["#5fd08c", "rgba(95,208,140,0.16)", 2, []],
  corrected: ["#b3a8ff", "rgba(179,168,255,0.16)", 2, []],
  kept: ["#62b8f5", "rgba(98,184,245,0.10)", 2, []],
  fixed: ["#62b8f5", "rgba(98,184,245,0.24)", 2.5, []],
  dropped: ["#62b8f5", null, 2, [4, 3]],
};

function boxClass(word) {
  if (word.flag) return word.flag;
  if (word.status === "absorbed") return word.absorbed_by ? "absorbed" : "dropped";
  if (word.status === "fixed") return word.review === "accepted" ? "kept" : "fixed";
  if (word.review) return word.review;
  return word.status;
}

const TOUCH_SLOP = 22; // a 44 px touch target

class PageView {
  constructor(canvas) {
    this.canvas = canvas;
    this.image = null;
    this.width = 1;
    this.height = 1;
    this.words = [];
    this.buckets = {};
    this.only = false;
    this.fixedOnly = false;
    this.selected = null;
    this.involved = new Set();
    this.view = { s: 1, x: 0, y: 0 };
    this.lastDraw = 0;
    this.moved = false;
    this.onPick = null;
    this.bind();
  }

  setPage(image, width, height, words) {
    const keepView = this.image && this.width === width && this.height === height;
    this.image = image;
    this.width = width;
    this.height = height;
    this.setWords(words);
    if (!keepView) this.fit();
    else this.draw();
  }

  setWords(words) {
    this.words = words;
    this.buckets = {};
    for (const w of words) (this.buckets[boxClass(w)] ||= []).push(w);
  }

  size() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr));
    const h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    return { w: r.width, h: r.height, dpr };
  }

  fit() {
    const { w, h } = this.size();
    const s = Math.min((w - 24) / this.width, (h - 24) / this.height);
    this.view = { s: s > 0 ? s : 1, x: (w - this.width * s) / 2, y: (h - this.height * s) / 2 };
    this.moved = false;
    this.draw();
  }

  zoomAt(factor, cx, cy) {
    const { w, h } = this.size();
    cx = cx ?? w / 2;
    cy = cy ?? h / 2;
    const v = this.view;
    const s = Math.min(8, Math.max(0.05, v.s * factor));
    this.view = { s, x: cx - ((cx - v.x) / v.s) * s, y: cy - ((cy - v.y) / v.s) * s };
    this.moved = true;
    this.draw();
  }

  reveal(word) {
    const { w, h } = this.size();
    const v = this.view;
    const [l, tp, r, b] = word.box;
    const sx = l * v.s + v.x, ex = r * v.s + v.x, sy = tp * v.s + v.y, ey = b * v.s + v.y;
    if (sx >= 20 && ex <= w - 20 && sy >= 20 && ey <= h - 20) return;
    this.view = { s: v.s, x: w / 2 - ((l + r) / 2) * v.s, y: h / 2 - ((tp + b) / 2) * v.s };
    this.moved = true;
    this.draw();
  }

  visibleCount() {
    let n = 0;
    for (const [cls, list] of Object.entries(this.buckets)) {
      if (this.shown(cls)) n += list.length;
    }
    return n;
  }

  isPending(cls) {
    return cls === "doubtful" || cls === "unscored" || cls === "mismatch";
  }

  isFixed(cls) {
    return cls === "fixed" || cls === "kept" || cls === "dropped";
  }

  shown(cls) {
    if (cls === "absorbed") return false;
    if (this.fixedOnly) return this.isFixed(cls);
    return !this.only || this.isPending(cls);
  }

  draw() {
    const started = performance.now();
    const { w, h, dpr } = this.size();
    const ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0f1113";
    ctx.fillRect(0, 0, w, h);
    const v = this.view;
    if (this.image) {
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(this.image, v.x, v.y, this.width * v.s, this.height * v.s);
    } else {
      ctx.fillStyle = "#e9e9e6";
      ctx.fillRect(v.x, v.y, this.width * v.s, this.height * v.s);
    }
    const order = ["native", "ok", "accepted", "corrected", "kept", "fixed", "dropped", "unscored", "doubtful", "mismatch"];
    for (const cls of order) {
      const list = this.buckets[cls];
      if (!list || !this.shown(cls)) continue;
      const [stroke, fill, width, dash] = COLORS[cls];
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.strokeStyle = stroke;
      ctx.beginPath();
      for (const word of list) {
        const [l, tp, r, b] = word.box;
        ctx.rect(l * v.s + v.x, tp * v.s + v.y, (r - l) * v.s, (b - tp) * v.s);
      }
      ctx.stroke();
      if (fill) {
        ctx.fillStyle = fill;
        ctx.beginPath();
        for (const word of list) {
          const [l, tp, r, b] = word.box;
          ctx.rect(l * v.s + v.x, tp * v.s + v.y, (r - l) * v.s, (b - tp) * v.s);
        }
        ctx.fill();
      }
    }
    ctx.setLineDash([]);
    this.drawCuts(ctx);
    const mark = (ids, color, width) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      for (const word of this.words) {
        if (!ids.has(word.id)) continue;
        const [l, tp, r, b] = word.box;
        ctx.strokeRect(l * v.s + v.x - 2, tp * v.s + v.y - 2, (r - l) * v.s + 4, (b - tp) * v.s + 4);
      }
    };
    mark(this.involved, "#62b8f5", 2.5);
    if (this.selected) mark(new Set([this.selected]), "#8b7cf6", 3);
    this.lastDraw = performance.now() - started;
  }

  drawCuts(ctx) {
    const v = this.view;
    ctx.strokeStyle = COLORS.fixed[0];
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const cls of ["fixed", "kept"]) {
      for (const word of this.buckets[cls] || []) {
        if (!this.shown(cls) || !word.fix || word.fix.op !== "segment" || !word.text) continue;
        const [l, tp, r, b] = word.box;
        for (let i = 1; i < word.text.length; i++) {
          if (word.text[i] !== " ") continue;
          const x = (l + ((r - l) * (i + 0.5)) / word.text.length) * v.s + v.x;
          ctx.moveTo(x, tp * v.s + v.y - 3);
          ctx.lineTo(x, b * v.s + v.y + 3);
        }
      }
    }
    ctx.stroke();
  }

  // The word under a point of the canvas. `slop` widens every box by that many screen pixels so a finger can hit a word
  // that is only a few pixels high: among the boxes that reach the point, the nearest one wins.
  hit(px, py, slop = 0) {
    const v = this.view;
    const x = (px - v.x) / v.s, y = (py - v.y) / v.s;
    for (let i = this.words.length - 1; i >= 0; i--) {
      const word = this.words[i];
      if (!this.shown(boxClass(word))) continue;
      const [l, tp, r, b] = word.box;
      if (x >= l && x <= r && y >= tp && y <= b) return word;
    }
    if (!slop) return null;
    let best = null;
    let bestDistance = slop;
    for (const word of this.words) {
      if (!this.shown(boxClass(word))) continue;
      const [l, tp, r, b] = word.box;
      const dx = Math.max(l * v.s + v.x - px, 0, px - (r * v.s + v.x));
      const dy = Math.max(tp * v.s + v.y - py, 0, py - (b * v.s + v.y));
      const distance = Math.hypot(dx, dy);
      if (distance <= bestDistance) {
        best = word;
        bestDistance = distance;
      }
    }
    return best;
  }

  // A view the user has not moved follows the size of its panel: the page stays fitted and centred.
  onResize() {
    if (!this.image) return;
    if (this.moved) this.draw();
    else this.fit();
  }

  bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, vx: this.view.x, vy: this.view.y, moved: false };
      c.setPointerCapture(e.pointerId);
      c.classList.add("dragging");
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      if (drag.moved) {
        this.view = { s: this.view.s, x: drag.vx + dx, y: drag.vy + dy };
        this.moved = true;
        this.draw();
      }
    });
    c.addEventListener("pointerup", (e) => {
      c.classList.remove("dragging");
      if (drag && !drag.moved && this.onPick) {
        const r = c.getBoundingClientRect();
        this.onPick(this.hit(e.clientX - r.left, e.clientY - r.top, e.pointerType === "mouse" ? 0 : TOUCH_SLOP));
      }
      drag = null;
    });
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const r = c.getBoundingClientRect();
        this.zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
      },
      { passive: false },
    );
    new ResizeObserver(() => this.onResize()).observe(c);
  }
}

let reviewView = null;
let readView = null;

// data

async function loadDocs() {
  const { docs } = await api("/api/docs");
  for (const d of docs) addDoc(d);
}

function addDoc(meta) {
  if (!S.docs[meta.id]) {
    S.order.push(meta.id);
    // a document read after the Export screen was first opened is ticked there like the others: it is not left out silently
    if (S.exportState.picked) S.exportState.picked.add(meta.id);
  }
  S.docs[meta.id] = meta;
  if (!S.cur.doc) S.cur.doc = meta.id;
}

// Another tab or window may have changed a document since this one last looked (a word accepted, the correction mode
// switched): when this tab gets the focus back it reads what the server holds, and redraws only if something differs.
async function refreshFromServer() {
  if (S.starting || (S.job && S.job.running)) return;
  const { docs } = await api("/api/docs");
  let changed = false;
  for (const d of docs) {
    if (S.docs[d.id] && JSON.stringify(S.docs[d.id]) === JSON.stringify(d)) continue;
    changed = true;
    dropPages(d.id);
    addDoc(d);
  }
  if (!changed) return;
  if (S.tab === "review") await showPage();
  render();
}

async function refreshDoc(id) {
  const { doc } = await api(`/api/doc/${id}`);
  addDoc(doc);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(Object.assign(new Error("image"), { code: "error.image_failed", params: {} }));
    img.src = url;
  });
}

async function pageData(doc, number) {
  const key = `${doc}:${number}`;
  if (!S.pages[key]) {
    const loading = (async () => {
      const [data, image] = await Promise.all([
        api(`/api/words/${doc}/${number}`),
        loadImage(link(`/api/page/${doc}/${number}.png`)),
      ]);
      applyFlags(data.words, new Set(data.flagged));
      // the data says whose page it is: a screen draws it only next to the document and page it belongs to
      return { doc, page: number, words: data.words, width: data.width, height: data.height, image };
    })();
    S.pages[key] = loading;
    // a page that could not be loaded is asked for again next time, not remembered as failed until the document is re-read
    loading.catch(() => {
      if (S.pages[key] === loading) delete S.pages[key];
    });
  }
  return S.pages[key];
}

function applyFlags(words, flagged) {
  for (const word of words) word.flag = flagged.has(word.id) ? "mismatch" : null;
}

function currentMeta() {
  return S.docs[S.cur.doc];
}

// Pages of a document that did not get their second reading (thorough mode ran out of RAM or time).
function noSecondReading(meta) {
  return (meta && meta.fixes && meta.fixes.second_reading_failed_pages) || [];
}

function wordOnPage(page, id) {
  return page.words.find((w) => w.id === id) || null;
}

function pageOf(id) {
  return Number(/^p(\d+)w/.exec(id)[1]);
}

// Point Review at a document: another document starts at its first page, with no word and no rule picked and the page
// view (not the table of fields, which only an invoice has).
function focusDoc(docId, wordId) {
  const changed = docId !== S.cur.doc;
  S.cur.doc = docId;
  S.editing = false;
  S.cur.word = wordId || null;
  if (wordId) S.cur.page = pageOf(wordId);
  else if (changed) S.cur.page = 1;
  if (changed) S.ruleFocus = null;
  S.reviewTab = "page";
}

async function select(docId, wordId) {
  focusDoc(docId, wordId);
  await showPage();
}

async function showPage() {
  const meta = currentMeta();
  if (!meta) return;
  if (!meta.pages[S.cur.page - 1]) S.cur.page = 1; // a page this document does not have
  // the filter of fixed words belongs to the document it was set on: one with nothing fixed would show an empty page
  if (S.fixedOnly && !meta.fixed.length) S.fixedOnly = false;
  const docId = S.cur.doc;
  const number = S.cur.page;
  const data = await pageData(docId, number);
  // the choice moved on while the page loaded: the showPage queued by that move draws the right one
  if (S.cur.doc !== docId || S.cur.page !== number) return;
  S.current = data;
  let word = S.cur.word ? wordOnPage(data, S.cur.word) : null;
  if (word && word.status === "absorbed" && word.absorbed_by) word = wordOnPage(data, word.absorbed_by);
  if (!word) S.cur.word = null;
  else S.cur.word = word.id;
  computeInvolved(word);
  if (reviewView && S.tab === "review" && S.reviewTab === "page") {
    reviewView.only = S.only;
    reviewView.fixedOnly = S.fixedOnly;
    reviewView.selected = word ? word.id : null;
    reviewView.involved = S.involved;
    reviewView.setPage(data.image, data.width, data.height, data.words);
    if (word) reviewView.reveal(word);
  }
  renderLeft();
  renderRight();
  renderFilter();
}

function renderFilter() {
  const button = document.getElementById("fixed-filter");
  const meta = currentMeta();
  if (!button || !meta) return;
  button.textContent = t("ui.review.fixed_filter", { n: meta.fixed.length });
  button.disabled = !meta.fixed.length;
  button.setAttribute("aria-pressed", S.fixedOnly ? "true" : "false");
  button.style.borderColor = S.fixedOnly ? "var(--blue)" : "";
}

// Whether the page data in hand is the page of the document and page that are selected (a load may be on its way).
function currentIsShown() {
  return !!S.current && S.current.doc === S.cur.doc && S.current.page === S.cur.page;
}

function wordOf(id) {
  if (!id || !currentIsShown()) return null;
  return wordOnPage(S.current, id);
}

function numbersOf(id) {
  const meta = currentMeta();
  return meta ? meta.numbers.filter((n) => n.word_ids.includes(id)) : [];
}

function suggestionOf(word) {
  if (!word) return null;
  if (word.suggestion) return word.suggestion;
  for (const n of numbersOf(word.id)) {
    const fix = n.suggestion;
    if (fix && n.word_ids.length === 1 && word.text.slice(fix.position, fix.position + 1) === fix.old) {
      return word.text.slice(0, fix.position) + fix.new + word.text.slice(fix.position + 1);
    }
  }
  return null;
}

/* the values a rule involves, located through the invoice cells */

function involvedBy(inv, wid, ruleKey) {
  if (!inv) return [];
  const ids = [];
  const take = (...cells) => cells.forEach((c) => c && ids.push(...c.word_ids));
  const has = (c) => c && c.word_ids.includes(wid);
  const line = inv.lines.find((l) => [l.description, l.quantity, l.unit_price, l.amount].some(has));
  const pair = inv.vat.find((v) => [v.rate, v.base, v.amount].some(has));
  switch (ruleKey) {
    case "checks.rule.line_amount":
      if (line) take(line.quantity, line.unit_price, line.amount);
      break;
    case "checks.rule.lines_sum":
      inv.lines.forEach((l) => take(l.amount));
      take(inv.subtotal);
      break;
    case "checks.rule.vat_amount":
      take(inv.subtotal);
      if (pair) take(pair.rate, pair.base, pair.amount);
      else inv.vat.forEach((v) => take(v.rate, v.base, v.amount));
      break;
    case "checks.rule.withholding_amount":
      take(inv.subtotal, inv.withholding_rate, inv.withholding);
      break;
    case "checks.rule.total":
      take(inv.subtotal, inv.withholding, inv.total);
      inv.vat.forEach((v) => take(v.amount));
      break;
    default:
  }
  return ids.filter((x) => x !== wid);
}

function computeInvolved(word) {
  S.involved = new Set();
  if (!word) return;
  if (word.fix && word.fix.applied && word.fix.op === "join") word.fix.ids.filter((id) => id !== word.id).forEach((id) => S.involved.add(id));
  const inv = currentMeta().invoice;
  for (const n of numbersOf(word.id)) {
    for (const rule of n.rules) {
      if (S.ruleFocus && S.ruleFocus !== rule.key) continue;
      involvedBy(inv, word.id, rule.key).forEach((x) => S.involved.add(x));
    }
  }
}

// actions

function queue(promise) {
  S.pending = S.pending.then(() => promise()).catch(fail);
  return S.pending;
}

function navList(meta) {
  return S.fixedOnly ? meta.fixed : meta.doubtful;
}

function dropPages(docId) {
  for (const key of Object.keys(S.pages)) if (key.startsWith(`${docId}:`)) delete S.pages[key];
}

function takeChanges(meta, result) {
  const keys = ["doubtful", "fixed", "fixes", "pages", "numbers", "numbers_checked", "invoice", "mode"];
  for (const key of keys) if (result[key] !== undefined) meta[key] = result[key];
}

async function act(kind, text) {
  const meta = currentMeta();
  const word = wordOf(S.cur.word);
  if (!meta || !word) return;
  const list = navList(meta);
  const index = list.findIndex((d) => d.id === word.id);
  const body = { doc: meta.id, word: word.id };
  if (kind === "correct") body.text = text;
  const result = await api(`/api/${kind}`, body);
  takeChanges(meta, result);
  if (kind === "undo") dropPages(meta.id);
  else {
    const page = S.current;
    page.words[page.words.findIndex((w) => w.id === word.id)] = result.word;
    applyFlags(page.words, new Set(result.flagged));
  }
  S.editing = false;
  if (kind === "reopen") return select(meta.id, word.id);
  const after = navList(meta);
  const still = after.findIndex((d) => d.id === word.id);
  let next = null;
  if (after.length) next = still >= 0 && S.fixedOnly ? after[(still + 1) % after.length] : index >= 0 ? after[Math.min(index, after.length - 1)] : null;
  await select(meta.id, next ? next.id : word.id);
}

async function undoPage() {
  const meta = currentMeta();
  if (!meta) return;
  const result = await api("/api/undo_page", { doc: meta.id, page: S.cur.page });
  takeChanges(meta, result);
  dropPages(meta.id);
  S.editing = false;
  await showPage();
}

async function setMode(mode) {
  const meta = currentMeta();
  if (!meta || meta.mode === mode) return;
  const result = await api("/api/mode", { doc: meta.id, mode });
  takeChanges(meta, result);
  dropPages(meta.id);
  S.editing = false;
  render();
  await showPage();
}

function isFixed(word) {
  return !!word && !!word.fix && word.fix.applied && (word.status === "fixed" || word.status === "absorbed");
}

async function step(direction) {
  const meta = currentMeta();
  if (!meta || !navList(meta).length) return;
  const list = navList(meta);
  const index = list.findIndex((d) => d.id === S.cur.word);
  let target;
  if (index < 0) target = direction > 0 ? list[0] : list[list.length - 1];
  else target = list[(index + direction + list.length) % list.length];
  await select(meta.id, target.id);
}

function startEdit() {
  if (!S.cur.word) return;
  S.editing = true;
  renderRight();
  const input = document.getElementById("edit-input");
  if (input) {
    input.focus();
    input.select();
  }
}

function onKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const target = e.target;
  const typing = target && (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA");
  if (typing) {
    if (target.id === "edit-input") {
      if (e.key === "Enter") {
        e.preventDefault();
        const text = target.value;
        queue(() => act("correct", text));
      } else if (e.key === "Escape") {
        e.preventDefault();
        S.editing = false;
        renderRight();
      }
    }
    return;
  }
  if (S.tab !== "review") return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (key === "n") queue(() => step(1));
  else if (key === "p") queue(() => step(-1));
  else if (key === "u" && isFixed(wordOf(S.cur.word))) queue(() => act("undo"));
  else if (key === "u" && canReopen(wordOf(S.cur.word))) queue(() => act("reopen"));
  else if (key === "Enter" && S.cur.word) {
    e.preventDefault();
    queue(() => act("accept"));
  } else if (key === "Tab") {
    // Tab takes the suggestion only when there is one to take, and never as Shift+Tab or from a control: everywhere
    // else it moves the focus, so a keyboard-only user can reach the tabs, the buttons and the zoom.
    const onControl = target && target !== document.body && target.closest && target.closest("button, a, summary, [tabindex]");
    if (!e.shiftKey && !onControl && S.cur.word && suggestionOf(wordOf(S.cur.word))) {
      e.preventDefault();
      queue(() => act("use_suggestion"));
    }
  } else if (key === "e" && S.cur.word) {
    e.preventDefault();
    startEdit();
  } else if (key === "Escape") {
    S.cur.word = null;
    S.ruleFocus = null;
    queue(showPage);
  } else if (key === "+" || key === "=") reviewView && reviewView.zoomAt(1.25);
  else if (key === "-") reviewView && reviewView.zoomAt(1 / 1.25);
  else if (key === "0") reviewView && reviewView.fit();
}

document.addEventListener("keydown", onKey);
window.addEventListener("focus", () => queue(refreshFromServer));

// screens

const TABS = ["import", "read", "review", "export"];

function projectName() {
  const first = S.files.find((f) => f.picked);
  const dir = S.browse ? pathParts(S.browse.path).parts.pop() : null;
  return first || dir ? dir : t("ui.project.none");
}

function topBar() {
  const enabled = { import: true, read: !!S.job, review: S.order.length > 0, export: S.order.length > 0 };
  return h(
    "header",
    { class: "top" },
    h("div", { class: "brand" }, h("div", { class: "logo" }, h("i")), h("b", null, "Extract"), h("span", { class: "sep" }, "/"), h("span", null, projectName())),
    h(
      "nav",
      { class: "tabs", "aria-label": t("ui.nav") },
      TABS.map((name) =>
        h("button", { class: "tab", "data-tab": name, disabled: !enabled[name], "aria-current": S.tab === name ? "page" : null, onclick: () => go(name) }, t(`ui.tab.${name}`)),
      ),
    ),
    h(
      "div",
      { class: "tools" },
      modeSwitch(),
      h("button", { class: "chip", id: "lang", onclick: () => setLang(S.lang === "en" ? "es" : "en") }, S.lang.toUpperCase()),
    ),
  );
}

function modeSwitch() {
  const meta = S.tab === "review" ? currentMeta() : null;
  if (!meta) return null;
  return h(
    "div",
    { class: "modeswitch", id: "modeswitch", title: t("ui.mode.title") },
    h("span", { class: "muted small" }, t("ui.mode.label")),
    h("div", { class: "seg" }, ["off", "safe", "more"].map((mode) => h("button", { "data-mode": mode, "aria-pressed": meta.mode === mode, onclick: () => queue(() => setMode(mode)) }, t(`ui.mode.${mode}`)))),
  );
}

// The top bar again, for a change that does not come from a click on this screen (a read that another tab or a reload
// started, a document read while this screen is Import): the Read, Review and Export tabs follow what exists.
function refreshTopBar() {
  const old = document.querySelector("header.top");
  if (old) old.replaceWith(topBar());
}

function go(tab) {
  S.tab = tab;
  S.error = null;
  render();
  if (tab === "review") queue(showPage);
}

function frame(side, main, aside) {
  return h("div", { class: "body" }, side, main, aside);
}

function clearError() {
  if (!S.error) return;
  S.error = null;
  document.querySelectorAll(".banner").forEach((el) => el.remove());
}

function errorBox() {
  return S.error ? h("div", { class: "error banner", role: "alert" }, h("span", { class: "grow" }, S.error), h("button", { class: "btn", "aria-label": t("ui.dismiss"), onclick: () => { S.error = null; render(); } }, "x")) : null;
}

const STORE = "extract.session";

function remember() {
  try {
    sessionStorage.setItem(STORE, JSON.stringify({ files: S.files, path: S.browse ? S.browse.path : null, opts: S.opts }));
  } catch (e) {}
}

function recall() {
  try {
    return JSON.parse(sessionStorage.getItem(STORE) || "null");
  } catch (e) {
    return null;
  }
}

function render() {
  remember();
  const root = document.getElementById("root");
  root.replaceChildren(h("div", { class: "app" }, topBar(), screen()));
  document.documentElement.lang = S.lang;
  document.title = "Extract";
  if (S.tab === "review") afterReviewMount();
  if (S.tab === "read") afterReadMount();
  if (window.__selftest) root.append(window.__selftest);
}

function screen() {
  return { import: importScreen, read: readScreen, review: reviewScreen, export: exportScreen }[S.tab]();
}

/* Import */

function kindLabel(kind) {
  return t(`ui.kind.${kind}`);
}

// What the read may take: the long figure is a page full of text, the short one a page with little on it.
function estimateRange(picked) {
  const scanned = picked.reduce((n, f) => n + (f.scanned_pages || 0), 0);
  const digital = picked.reduce((n, f) => n + (f.digital_pages || 0), 0);
  // the seconds were measured on one machine: this computer's own reads scale them (speed_factor)
  const speed = S.defaults.speed_factor || 1;
  const per = S.defaults.seconds_per_scanned_page * speed;
  const low = S.defaults.seconds_per_scanned_page_low * speed;
  let from = scanned * low + digital * S.defaults.seconds_per_digital_page;
  let to = scanned * per + digital * S.defaults.seconds_per_digital_page;
  if (S.opts.second === "thorough") {
    const extra = ((S.defaults.second_reading_thorough_cost || {}).seconds_per_page || 0) * speed;
    from += scanned * extra * (low / per);
    to += scanned * extra;
  }
  return [from, to];
}

function estimateLine(picked) {
  const pages = picked.reduce((n, f) => n + (f.pages || 0), 0);
  const [low, high] = estimateRange(picked);
  const a = fmtTime(low);
  const b = fmtTime(high);
  // until this computer has read something the line says it is for a typical laptop, not for "this PC"
  return tn(S.defaults.speed_measured ? "ui.import.estimate" : "ui.import.estimate_typical", pages, { pages, time: a === b ? a : t("ui.import.range", { from: a, to: b }) });
}

// The folders of a path and the path up to each one, for a computer that writes them with "/" (Linux, macOS) or with "\\"
// and a drive letter (Windows): a path of the second kind has one name only when split on "/".
function pathParts(path) {
  const parts = path.split(/[\\/]+/).filter(Boolean);
  const drive = /^[A-Za-z]:/.test(path);
  const unc = path.startsWith("\\\\");
  const upTo = (n) => (drive ? parts.slice(0, n).join("\\") + (n === 1 ? "\\" : "") : unc ? `\\\\${parts.slice(0, n).join("\\")}` : `/${parts.slice(0, n).join("/")}`);
  return { parts, upTo };
}

// The last folders of a path as buttons; the ones above the user's home folder are not offered.
function crumbs(b) {
  if (!b) return null;
  const { parts, upTo } = pathParts(b.path);
  const floor = pathParts(S.defaults.root).parts.length;
  const shown = parts.map((name, i) => ({ name, i, path: upTo(i + 1) })).slice(-3);
  return h(
    "div",
    { class: "crumbs small", id: "crumbs", "aria-label": t("ui.import.folder") },
    parts.length > shown.length ? h("span", { class: "muted" }, "...") : null,
    shown.map((c) => h("button", { class: "crumb", disabled: c.i + 1 < floor, onclick: () => browse(c.path) }, c.name)),
  );
}

function importScreen() {
  const picked = S.files.filter((f) => f.picked);
  const thoroughMissing = S.opts.second === "thorough" && !S.defaults.thorough_available;
  const side = h(
    "aside",
    { class: "side", "aria-label": t("ui.import.documents") },
    h("div", { class: "row between" }, h("span", { class: "label" }, t("ui.import.documents_count", { n: S.files.length }))),
    S.files.length === 0 ? h("div", { class: "note" }, t("ui.import.empty")) : null,
    S.files.map((f) =>
      h(
        "label",
        { class: `card ${f.picked ? "selected" : ""}` },
        h("div", { class: "row" }, h("input", { type: "checkbox", checked: f.picked, onchange: (e) => { f.picked = e.target.checked; render(); } }), h("span", { class: "name grow" }, f.name)),
        h("div", { class: "mono small muted" }, f.pages ? `${pagesText(f.pages)} · ${kindLabel(f.kind)} · ${fmtSize(f.size)}` : t("ui.import.inspecting")),
      ),
    ),
  );
  const b = S.browse;
  const shownPath = S.pathText !== null ? S.pathText : b ? b.path : "";
  const main = h(
    "section",
    { class: "main main-pad" },
    errorBox(),
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.import.folder")), h("div", { class: "row" }, h("input", { type: "text", id: "path", value: shownPath, title: shownPath, oninput: (e) => { S.pathText = e.target.value; clearError(); }, onkeydown: (e) => e.key === "Enter" && browse(e.target.value) }), h("button", { class: "btn", id: "browse", onclick: () => browse(document.getElementById("path").value) }, t("ui.import.browse")), h("button", { class: "btn", id: "add-all", onclick: addAll, disabled: !b || !b.pdfs.length }, tn("ui.import.add_all", b ? b.pdfs.length : 0))), crumbs(b), h("div", { class: "small muted", id: "where" }, t("ui.import.where"))),
    addReportBox(),
    b
      ? h(
          "div",
          { class: "list", id: "listing" },
          b.parent ? h("div", { class: "item folder", ...pressable(() => browse(b.parent)) }, "..") : null,
          b.dirs.map((d) => h("div", { class: "item folder", ...pressable(() => browse(`${b.path}/${d}`)) }, `${d}/`)),
          b.pdfs.map((f) => h("div", { class: "item" }, h("span", { class: "grow" }, f.name), h("span", { class: "mono small muted" }, fmtSize(f.size)), h("button", { class: "btn", onclick: () => addFile(`${b.path}/${f.name}`) }, t("ui.import.add")))),
          !b.dirs.length && !b.pdfs.length ? h("div", { class: "item muted" }, t("ui.import.nothing")) : null,
        )
      : null,
  );
  const aside = h(
    "aside",
    { class: "aside", "aria-label": t("ui.import.settings") },
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.import.languages")), ["es", "en"].map((lang) => h("label", { class: "check" }, h("input", { type: "checkbox", checked: S.opts.languages.includes(lang), onchange: (e) => toggle(S.opts.languages, lang, e.target.checked, 1) }), t(`ui.language.${lang}`)))),
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.import.outputs")), ["pdf", "xlsx", "json", "txt"].map((f) => h("label", { class: "check" }, h("input", { type: "checkbox", checked: S.opts.formats.includes(f), onchange: (e) => { toggle(S.opts.formats, f, e.target.checked, 1); S.exportState.formats = [...S.opts.formats]; } }), t(`ui.format.${f}`)))),
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.import.budget")), h("div", { class: "seg" }, ["fewer", "balanced", "more"].map((r) => h("button", { "aria-pressed": S.opts.review === r, "data-review": r, onclick: () => { S.opts.review = r; render(); } }, t(`ui.budget.${r}`)))), h("span", { class: "small muted" }, t(`ui.budget.${S.opts.review}.hint`))),
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.import.corrections")), h("div", { class: "seg" }, ["off", "safe", "more"].map((mode) => h("button", { "aria-pressed": S.opts.correct === mode, "data-correct": mode, onclick: () => { S.opts.correct = mode; render(); } }, t(`ui.mode.${mode}`)))), h("span", { class: "small muted" }, t(`ui.correct.${S.opts.correct}.hint`)), S.opts.correct === "safe" ? h("span", { class: "small muted", id: "safe-note" }, t("ui.correct.safe.note")) : null),
    secondReadingField(),
    h("div", { class: "grow" }),
    h("div", { class: "actionbar" }, h("button", { class: "btn primary wide", id: "read-button", disabled: picked.length === 0 || thoroughMissing || (S.job && S.job.running), onclick: startRead }, tn("ui.import.read", picked.length)), h("div", { class: "muted", style: "text-align:center", id: "estimate" }, picked.length ? estimateLine(picked) : t("ui.import.pick"))),
  );
  return frame(side, main, aside);
}

function oneDecimal(value) {
  return Number(value).toLocaleString(S.lang, { maximumFractionDigits: 1 });
}

function secondReadingHint() {
  if (S.opts.second !== "thorough") return [t("ui.second.off.hint")];
  const cost = S.defaults.second_reading_thorough_cost || {};
  const lines = [t("ui.second.thorough.hint", { seconds: oneDecimal(cost.seconds_per_page) })];
  if (cost.seconds_p95 && cost.reader_ram_mb) lines.push(t("ui.second.thorough.detail", { p95: oneDecimal(cost.seconds_p95), ram: oneDecimal(cost.reader_ram_mb / 1000) }));
  if (S.opts.correct !== "more") lines.push(t("ui.second.gain.other"));
  else {
    const lang = S.opts.languages.length === 1 ? S.opts.languages[0] : null;
    const byLanguage = cost.fixes_more_pct_by_language;
    const fixed = byLanguage ? byLanguage[lang] : lang === "es" ? cost.fixes_more_pct : undefined;
    lines.push(fixed === undefined || fixed === null ? t("ui.second.gain.unmeasured") : t("ui.second.gain.more", { fixed: oneDecimal(fixed), language: t(`ui.language.${lang}`) }));
  }
  if (!S.defaults.thorough_available) lines.push(t("ui.second.missing"));
  return lines;
}

function secondReadingField() {
  return h(
    "div",
    { class: "field" },
    h("span", { class: "label" }, t("ui.import.second")),
    h("div", { class: "seg" }, ["off", "thorough"].map((value) => h("button", { "aria-pressed": S.opts.second === value, "data-second": value, onclick: () => { S.opts.second = value; render(); } }, t(`ui.second.${value}`)))),
    secondReadingHint().map((line) => h("div", { class: "small muted second-hint" }, line)),
  );
}

function readOptions() {
  return { languages: S.opts.languages, review: S.opts.review, correct: S.opts.correct, second_reading: S.opts.second === "thorough" ? "thorough" : false };
}

function toggle(list, value, on, min) {
  const at = list.indexOf(value);
  if (on && at < 0) list.push(value);
  if (!on && at >= 0 && list.length > min) list.splice(at, 1);
  render();
}

async function browse(path) {
  S.pathText = path || "";
  S.addReport = null;
  try {
    S.error = null;
    S.browse = await api(`/api/browse?path=${encodeURIComponent(path || "")}`);
    S.browse.path = S.browse.path.replace(/\/$/, "");
    S.pathText = null;
  } catch (e) {
    S.error = errorText(e);
  }
  render();
}

// Add files: every one that cannot be read is listed with its own name and reason (never one generic message).
async function addPaths(paths) {
  const fresh = paths.filter((p) => !S.files.some((f) => f.path === p));
  S.addReport = null;
  if (fresh.length) {
    try {
      const result = await api("/api/inspect_many", { paths: fresh });
      for (const info of result.files) S.files.push({ ...info, picked: true });
      S.addReport = { already: paths.length - fresh.length, skipped: result.skipped.map((f) => ({ name: f.name, text: errorText(f) })) };
    } catch (e) {
      S.error = errorText(e);
    }
  } else if (paths.length) S.addReport = { already: paths.length, skipped: [] };
  render();
}

function addFile(path) {
  return addPaths([path]);
}

function addAll() {
  return addPaths(S.browse.pdfs.map((f) => `${S.browse.path}/${f.name}`));
}

function addReportBox() {
  const report = S.addReport;
  if (!report || (!report.skipped.length && !report.already)) return null;
  return h(
    "div",
    { class: "warn add-report", id: "add-report" },
    report.skipped.length ? h("div", null, h("b", null, tn("ui.import.skipped", report.skipped.length))) : null,
    report.skipped.length ? h("ul", { class: "skipped" }, report.skipped.map((f) => h("li", { "data-skipped": f.name }, h("b", null, f.name), ": ", f.text))) : null,
    report.already ? h("div", { class: "small muted" }, tn("ui.import.already", report.already)) : null,
  );
}

async function startRead(replace = false) {
  if (S.starting) return; // a double click is one read
  S.starting = true;
  const paths = S.files.filter((f) => f.picked).map((f) => f.path);
  S.error = null;
  const previous = S.job;
  S.job = { running: true, docs: paths.map((p) => ({ name: fileName(p), pages: 0, done: 0, status: "waiting", tiles: [] })), warnings: [], errors: [], startedAt: Date.now(), pagesDone: 0, pagesTotal: 0, doubtful: 0, checked: 0, current: -1 };
  try {
    await api("/api/read", { paths, options: readOptions(), replace });
  } catch (e) {
    S.job = previous || null;
    S.starting = false;
    // reading again a document the user has changed throws those changes away: ask first, never do it silently
    if (e.code === "error.edits_would_be_lost") return window.confirm(t("ui.read.confirm_replace", { names: (e.params.names || []).join(", ") })) ? startRead(true) : undefined;
    // a read already going on (started in another tab or before a reload) is shown, not reported as a failure
    if (e.code === "error.job_running") return go("read");
    return fail(e);
  }
  S.starting = false;
  go("read");
}

/* Read */

function handleEvent(ev) {
  if (ev.event === "job.started") {
    S.job = { running: true, docs: ev.names.map((name) => ({ name, pages: 0, done: 0, status: "waiting", tiles: [] })), warnings: [], errors: [], startedAt: ev.time * 1000, pagesDone: 0, pagesTotal: ev.pages, doubtful: 0, checked: 0, current: -1 };
    if (S.tab === "read") render();
    else refreshTopBar();
    return;
  }
  const job = S.job;
  if (!job) return;
  const doc = job.docs[ev.index];
  switch (ev.event) {
    case "doc.begin":
      job.current = ev.index;
      doc.pages = ev.pages;
      doc.status = "reading";
      doc.tiles = Array.from({ length: ev.pages }, () => ({ status: "wait", todo: 0 }));
      break;
    case "page.progress":
      doc.done = ev.done;
      break;
    case "page.read":
      if (doc.tiles[ev.page - 1]) doc.tiles[ev.page - 1] = { status: ev.status, todo: ev.doubtful };
      job.pagesDone += 1;
      (job.stamps ||= []).push(ev.time * 1000);
      job.doubtful += ev.doubtful;
      doc.fixedPages = (doc.fixedPages || 0) + (ev.fixed || 0);
      break;
    case "doc.read":
      doc.status = ev.status;
      doc.doc_id = ev.doc_id;
      doc.done = doc.pages;
      doc.todo = ev.doubtful;
      doc.fixed = ev.fixed;
      job.checked += ev.numbers_checked;
      doc.sameAs = ev.name !== doc.name ? ev.name : null; // a file with the content of one already open opens that one
      job.lastDoc = ev.doc_id;
      job.lastIndex = ev.index;
      dropPages(ev.doc_id); // the document may be a new reading of one that is open: the cached words are the old reading's
      refreshDoc(ev.doc_id).then(() => {
        if (S.tab === "read") render();
        else if (S.tab === "export") render();
        else {
          refreshTopBar();
          // Review shows what the server holds now: the list of documents, and the page of the one that was read again
          if (S.tab === "review") {
            renderLeft();
            if (S.cur.doc === ev.doc_id) queue(showPage);
          }
        }
      });
      break;
    case "job.warning":
      // with several documents a warning says which one it is about (the "same content" one names its own two files)
      job.warnings.push({ code: withheldCode(ev.code, ev.params), params: ev.params, doc: job.docs.length > 1 && doc && ev.code !== "job.warning.same_content" ? doc.name : null });
      break;
    case "job.error":
      job.errors.push({ code: ev.code, params: ev.params, path: ev.path });
      if (doc) doc.status = "failed"; // the document this error is about was not read: its card does not say Reading
      break;
    case "job.finished":
      job.running = false;
      job.stopping = false;
      job.endedAt = ev.time * 1000;
      job.status = ev.status;
      api("/api/defaults").then((d) => { S.defaults.speed_factor = d.speed_factor; S.defaults.speed_measured = d.speed_measured; }).catch(() => {});
      for (const d of job.docs) {
        // a document that was being read when the read stopped is not kept; the ones not yet started were never read
        if (ev.status !== "done" && d.status === "reading") d.status = ev.status === "stopped" ? "stopped" : "failed";
        else if (ev.status !== "done" && d.status === "waiting") d.status = "skipped";
        // a read that ended well has no document still waiting or being read: one with no result did not get read
        else if (ev.status === "done" && !d.doc_id && (d.status === "reading" || d.status === "waiting")) d.status = "failed";
      }
      break;
    default:
  }
  if (S.tab === "read") render();
}

async function pollLoop() {
  for (;;) {
    try {
      const { events } = await api(`/api/events?after=${S.after}${NOWAIT ? "&wait=0" : ""}`);
      const first = !S.live;
      for (const ev of events) {
        S.after = ev.n;
        handleEvent(ev);
      }
      S.live = true;
      // A page opened (or reloaded) while a read is going on shows that read, with its Stop button, not Import.
      if (first && S.job && S.job.running && S.tab === "import" && !params.get("tab")) go("read");
      if (NOWAIT) return;
    } catch (e) {
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

// Time left from the speed of the last pages, not from the whole read: pages differ a lot in how much text they hold.
function etaSeconds(stamps, remaining) {
  if (remaining <= 0) return 0;
  const recent = stamps.slice(-10);
  if (recent.length < 3) return NaN;
  const span = (recent[recent.length - 1] - recent[0]) / 1000;
  return span > 0 ? (span / (recent.length - 1)) * remaining : NaN;
}

// The document the middle of the Read screen is about: the one being read while the read goes on; once it is over, the
// last one that was read (the page drawn below is that document's, and its name and its page tiles go with it).
function shownCard(job) {
  if (!job.running && job.lastIndex !== undefined && job.docs[job.lastIndex]) return job.docs[job.lastIndex];
  return job.docs[Math.max(0, job.current)];
}

function readScreen() {
  const job = S.job;
  if (!job) return frame(h("aside", { class: "side" }), h("section", { class: "main main-pad" }, h("div", { class: "note" }, t("ui.read.none"))), h("aside", { class: "aside" }));
  const elapsed = ((job.endedAt || Date.now()) - job.startedAt) / 1000;
  const remaining = etaSeconds(job.stamps || [], job.pagesTotal - job.pagesDone);
  const side = h(
    "aside",
    { class: "side" },
    h("span", { class: "label" }, t("ui.read.documents")),
    job.docs.map((d, i) =>
      h(
        "div",
        { class: `card ${d === shownCard(job) ? "selected" : ""}`, "data-doc": d.status },
        h("div", { class: "name" }, d.name),
        h("div", { class: "bar" }, h("i", { style: `width:${d.pages ? (100 * d.done) / d.pages : 0}%` })),
        h("div", { class: "row between small" }, h("span", { class: "muted" }, t(`ui.read.status.${d.status}`)), h("span", { class: "mono muted" }, d.pages ? `${d.done}/${d.pages}` : "")),
        d.sameAs ? h("div", { class: "small muted", "data-same-as": d.sameAs }, t("ui.read.same_as", { name: d.sameAs })) : null,
        d.doc_id ? h("button", { class: "btn", onclick: () => { S.cur.page = 1; focusDoc(d.doc_id, null); go("review"); } }, t("ui.read.open")) : null,
      ),
    ),
  );
  const active = shownCard(job);
  const preview = job.lastDoc && S.docs[job.lastDoc];
  const main = h(
    "section",
    { class: "main main-pad" },
    errorBox(),
    job.errors.map((e) => h("div", { class: "error" }, `${errorText(e)}${e.path ? ` - ${fileName(e.path)}` : ""}`)),
    job.warnings.length ? h("div", { class: "warns", id: "warnings" }, groupedWarnings(job.warnings).map(warningBox)) : null,
    h("div", { class: "field" }, h("div", { class: "row between" }, h("span", { class: "label", id: "read-doc" }, !job.running && preview ? preview.name : active ? active.name : ""), h("span", { class: "mono" }, `${job.pagesDone}/${job.pagesTotal}`)), h("div", { class: "bar", style: "height:10px" }, h("i", { id: "total-bar", style: `width:${job.pagesTotal ? (100 * job.pagesDone) / job.pagesTotal : 0}%` }))),
    active && active.tiles.length ? h("div", { class: "tiles" }, active.tiles.map((tile, n) => h("div", { class: `tile ${tile.status} ${tile.todo ? "flag" : ""} ${n === active.done && job.running ? "now" : ""}`, title: tile.todo ? t("ui.read.tile_flagged", { n: tile.todo }) : "" }, n + 1))) : null,
    // the page below says whose it is: while the next document is being read it is the previous one's
    preview ? h("div", { class: "label", id: "read-preview" }, t("ui.read.preview", { name: preview.name, page: 1 })) : null,
    h("div", { class: "canvas-wrap readwrap", id: "read-canvas-wrap" }, h("canvas", { id: "read-canvas" })),
  );
  const aside = h(
    "aside",
    { class: "aside" },
    h("div", { class: "stats" }, h("div", { class: "stat" }, h("b", { id: "stat-doubtful" }, job.doubtful), h("span", { class: "muted small" }, t("ui.read.doubtful"))), h("div", { class: "stat" }, h("b", { id: "stat-checked" }, job.checked), h("span", { class: "muted small" }, t("ui.read.checked"))), h("div", { class: "stat" }, h("b", { id: "stat-fixed", class: "blue" }, job.docs.reduce((n, d) => n + (d.fixed ?? d.fixedPages ?? 0), 0)), h("span", { class: "muted small" }, t("ui.read.fixed")))),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.read.elapsed")), h("span", { class: "mono", id: "elapsed" }, fmtTime(elapsed))),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.read.eta")), h("span", { class: "mono", id: "eta" }, job.running ? (isNaN(remaining) ? t("ui.read.eta_wait") : fmtTime(remaining)) : "-")),
    job.status ? h("div", { class: "note", id: "job-status" }, t(`ui.read.finished.${job.status}`)) : null,
    h("div", { class: "grow" }),
    h(
      "div",
      { class: "actionbar" },
      h("button", { class: "btn danger wide", id: "stop", disabled: !job.running || job.stopping, onclick: () => { job.stopping = true; render(); api("/api/stop", {}).catch(fail); } }, job.stopping ? t("ui.read.stopping") : t("ui.read.stop")),
      S.order.length ? h("button", { class: `btn ${job.running ? "" : "primary"} wide`, id: "to-review", onclick: () => go("review") }, job.running ? t("ui.read.review_now") : t("ui.read.to_review")) : null,
    ),
  );
  return frame(side, main, aside);
}

// The screen is drawn again every second while a read goes on and on each event, and a page may take a while to load:
// the page that arrives is drawn on the view it was asked for, and only if no newer drawing of the screen has asked since.
let readMounts = 0;
async function afterReadMount() {
  const canvas = document.getElementById("read-canvas");
  const job = S.job;
  const mount = ++readMounts;
  if (!canvas || !job || !job.lastDoc) return;
  const meta = S.docs[job.lastDoc];
  if (!meta) return;
  const view = new PageView(canvas);
  readView = view;
  const data = await pageData(meta.id, 1);
  if (mount !== readMounts) return;
  view.setPage(data.image, data.width, data.height, data.words);
  canvas.dataset.doc = data.doc;
  canvas.dataset.page = String(data.page);
}

setInterval(() => {
  if (S.tab === "read" && S.job && S.job.running) render();
}, 1000);

/* Review */

function reviewScreen() {
  const meta = currentMeta();
  const side = h("aside", { class: "side", id: "left", "aria-label": t("ui.review.documents") });
  const main = h(
    "section",
    { class: "main" },
    h(
      "div",
      { class: "toolbar" },
      h("div", { class: "seg", style: "width:200px" }, ["page", "fields"].map((tab) => h("button", { "data-rtab": tab, "aria-pressed": S.reviewTab === tab, disabled: tab === "fields" && !(meta && meta.invoice), onclick: () => { S.reviewTab = tab; render(); queue(showPage); } }, t(`ui.review.tab.${tab}`)))),
      S.reviewTab === "fields" ? null : h("button", { class: "btn", id: "only", "aria-pressed": S.only, style: S.only ? "border-color:var(--accent)" : "", onclick: () => { S.only = !S.only; S.fixedOnly = false; if (reviewView) { reviewView.only = S.only; reviewView.fixedOnly = false; reviewView.draw(); } render(); } }, t("ui.review.only_doubtful")),
      S.reviewTab === "fields" ? null : h("button", { class: "btn", id: "fixed-filter", "aria-pressed": S.fixedOnly, style: S.fixedOnly ? "border-color:var(--blue)" : "", disabled: !meta || !meta.fixed.length, onclick: () => { S.fixedOnly = !S.fixedOnly; S.only = false; if (reviewView) { reviewView.fixedOnly = S.fixedOnly; reviewView.only = false; reviewView.draw(); } render(); } }, t("ui.review.fixed_filter", { n: meta ? meta.fixed.length : 0 })),
      h("div", { class: "grow" }),
      S.reviewTab === "fields" ? null : h("div", { class: "legend" }, h("span", null, h("i", { style: "background:var(--amber)" }), t("ui.legend.doubtful")), h("span", null, h("i", { style: "background:var(--muted)" }), t("ui.legend.unscored")), h("span", null, h("i", { style: "background:var(--green)" }), t("ui.legend.accepted")), h("span", null, h("i", { style: "background:var(--accent-soft)" }), t("ui.legend.corrected")), h("span", null, h("i", { style: "background:var(--blue)" }), t("ui.legend.fixed"))),
      S.reviewTab === "page" ? h("div", { class: "row" }, h("button", { class: "btn", onclick: () => reviewView && reviewView.zoomAt(1 / 1.25), "aria-label": t("ui.review.zoom_out") }, "-"), h("button", { class: "btn", onclick: () => reviewView && reviewView.fit() }, t("ui.review.fit")), h("button", { class: "btn", onclick: () => reviewView && reviewView.zoomAt(1.25), "aria-label": t("ui.review.zoom_in") }, "+")) : null,
    ),
    S.reviewTab === "page" ? h("div", { class: "canvas-wrap" }, h("canvas", { id: "page-canvas" })) : h("div", { class: "main-pad", id: "fields" }, fieldsTable(meta)),
  );
  const aside = h("aside", { class: "aside", id: "right" });
  return frame(side, main, aside);
}

function afterReviewMount() {
  const canvas = document.getElementById("page-canvas");
  if (canvas) {
    reviewView = new PageView(canvas);
    reviewView.onPick = (word) => {
      if (word) queue(() => select(S.cur.doc, word.id));
    };
    if (currentIsShown()) {
      reviewView.only = S.only;
      reviewView.fixedOnly = S.fixedOnly;
      reviewView.selected = S.cur.word;
      reviewView.involved = S.involved;
      reviewView.setPage(S.current.image, S.current.width, S.current.height, S.current.words);
    }
  } else reviewView = null;
  renderLeft();
  renderRight();
}

// The badges of a document: Ready only when nothing is left to check and every page was read.
function docBadges(d) {
  return [
    d.doubtful.length ? h("span", { class: "badge amber" }, t("ui.review.to_check", { k: d.doubtful.length })) : d.status === "partial" ? null : h("span", { class: "badge green" }, t("ui.review.ready")),
    d.fixed.length ? h("span", { class: "badge blue" }, tn("ui.review.fixed_count", d.fixed.length)) : null,
    d.status === "partial" ? h("span", { class: "badge red" }, t("ui.review.partial")) : null,
  ];
}

function docStateText(d) {
  return d.doubtful.length ? t("ui.review.to_check", { k: d.doubtful.length }) : d.status === "partial" ? t("ui.review.partial") : t("ui.review.ready");
}

function canReopen(word) {
  return !!word && !isFixed(word) && (word.review === "accepted" || (word.review === "corrected" && word.status !== "fixed" && word.status !== "absorbed"));
}

function renderLeft() {
  const left = document.getElementById("left");
  if (!left) return;
  left.replaceChildren(
    ...[h("span", { class: "label" }, t("ui.review.documents_count", { n: S.order.length })),
    S.order.map((id) => {
      const d = S.docs[id];
      const selected = id === S.cur.doc;
      return h(
        "div",
        null,
        h(
          "button",
          { class: `card ${selected ? "selected" : ""}`, "data-doc": id, onclick: () => queue(() => { S.cur.page = 1; return select(id, null); }) },
          h("div", { class: "name" }, d.name),
          h("div", { class: "row wrap" }, docBadges(d), noSecondReading(d).length ? h("span", { class: "badge amber", "data-second-failed": noSecondReading(d).length }, t("ui.second.failed_badge", { n: noSecondReading(d).length })) : null, h("span", { class: "muted small" }, pagesText(d.pages.length))),
        ),
        selected
          ? h("div", { class: "row", style: "flex-wrap:wrap;padding:6px 4px;gap:6px" }, d.pages.map((p) => h("button", { class: "btn", "data-page": p.number, style: p.number === S.cur.page ? "border-color:var(--accent)" : "", onclick: () => { S.cur.page = p.number; S.cur.word = null; queue(showPage); } }, p.number, p.status === "unreadable" ? h("span", { class: "badge red", style: "margin-left:6px" }, "!") : null, p.todo ? h("span", { class: "badge amber", style: "margin-left:6px" }, p.todo) : null, p.fixed ? h("span", { class: "badge blue", style: "margin-left:6px" }, p.fixed) : null)))
          : null,
      );
    })].flat()
  );
}

function renderRight() {
  const right = document.getElementById("right");
  if (!right) return;
  const meta = currentMeta();
  const word = wordOf(S.cur.word);
  const page = meta && meta.pages[S.cur.page - 1];
  const parts = [h("span", { class: "label" }, t("ui.review.selected"))];
  if (page && page.status === "unreadable") parts.push(h("div", { class: "warn", id: "unreadable" }, t("ui.review.unreadable", { page: page.number }), " ", t(page.reason || "page.reason.low_confidence")));
  ((meta && meta.notes) || []).filter((n) => n.page === S.cur.page).forEach((n) => parts.push(h("div", { class: "warn", id: `note-${n.key.split(".").pop()}` }, t(n.key, n))));
  const brake = meta && meta.fixes.withheld.find((w) => w.page === S.cur.page);
  if (brake) parts.push(h("div", { class: "warn", id: "withheld" }, t(withheldCode("guard.warning.fixes_withheld", brake), { page: brake.page, share: Math.round(brake.share * 100) })));
  const noSecond = noSecondReading(meta);
  if (noSecond.length) parts.push(h("div", { class: "warn", id: "second-failed" }, t("ui.second.failed_doc", { pages: noSecond.join(", ") }), noSecond.includes(S.cur.page) ? h("span", { id: "second-failed-page" }, " ", t("ui.second.failed_page", { page: S.cur.page })) : null));
  if (page && page.fixed) parts.push(h("button", { class: "btn", id: "undo-page", onclick: () => queue(undoPage) }, t("ui.fix.undo_page", { n: page.fixed })));
  if (!word) {
    parts.push(h("div", { class: "note" }, t(S.reviewTab === "fields" ? "ui.fields.pick" : "ui.review.pick")));
  } else if (isFixed(word)) {
    parts.push(fixPanel(meta, word));
  } else {
    const suggestion = suggestionOf(word);
    const unscored = word.status === "unscored" && !word.review;
    parts.push(
      cropView(meta.id, word),
      h("div", { class: "reading", id: "reading" }, word.text),
      word.original ? h("div", { class: "small muted" }, t("ui.review.original", { text: word.original })) : null,
      h("div", { class: "row" }, stateBadges(word)),
      unscored ? h("div", { class: "note" }, t("ui.review.not_checked")) : null,
      suggestion ? h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.review.suggestion")), h("span", { class: "mono", id: "suggestion" }, suggestion)) : null,
      word.fix && !word.fix.applied && word.review === "accepted" ? h("div", { class: "note", id: "fix-undone" }, t("ui.fix.undone", { text: word.fix.op === "drop" ? t("ui.fix.removed") : word.fix.text })) : null,
      word.reasons.length ? h("ul", { class: "reasons" }, word.reasons.map((r) => h("li", null, reasonText(r)))) : null,
      numbersOf(word.id).map((n) => numberBlock(n)),
      S.editing ? h("input", { type: "text", id: "edit-input", value: word.text, "aria-label": t("ui.review.edit") }) : null,
      h("div", { class: "row" }, h("button", { class: "btn grow", id: "accept", disabled: word.review === "accepted", onclick: () => queue(() => act("accept")) }, t("ui.review.accept"), h("kbd", null, "Enter")), h("button", { class: "btn grow", id: "use", disabled: !suggestion, onclick: () => queue(() => act("use_suggestion")) }, t("ui.review.use"), h("kbd", null, "Tab"))),
      h("button", { class: "btn", id: "edit", onclick: startEdit }, t("ui.review.edit"), h("kbd", null, "E")),
      canReopen(word) ? h("button", { class: "btn", id: "reopen", onclick: () => queue(() => act("reopen")) }, t("ui.review.reopen"), h("kbd", null, "U")) : null,
    );
  }
  const todo = meta ? navList(meta).length : 0;
  parts.push(h("div", { class: "row" }, h("button", { class: "btn grow", id: "prev", disabled: !todo, onclick: () => queue(() => step(-1)) }, t("ui.review.prev_button"), h("kbd", null, "P")), h("button", { class: "btn grow", id: "next", disabled: !todo, onclick: () => queue(() => step(1)) }, t("ui.review.next_button"), h("kbd", null, "N"))));
  parts.push(h("div", { class: "grow" }), h("div", { class: "hint" }, h("kbd", null, "N"), " ", t("ui.keys.next"), " · ", h("kbd", null, "P"), " ", t("ui.keys.previous"), h("br"), h("kbd", null, "Enter"), " ", t("ui.keys.accept"), " · ", h("kbd", null, "Tab"), " ", t("ui.keys.use"), h("br"), h("kbd", null, "U"), " ", t("ui.keys.undo"), " · ", h("kbd", null, "E"), " ", t("ui.keys.edit"), " · ", h("kbd", null, "Esc"), " ", t("ui.keys.deselect")));
  right.replaceChildren(...parts.flat().filter(Boolean));
  if (S.editing) {
    const input = document.getElementById("edit-input");
    if (input) { input.focus(); input.select(); }
  }
}

function sourceLabel(fix) {
  const s = fix.source;
  if (s.startsWith("reading:")) return t("ui.fix.source.reading", { reader: s.slice(8) });
  if (s.startsWith("rule:")) return t("ui.fix.source.rule");
  const known = { dictionary: 1, fold: 1, punct: 1, page: 1, ctc_alt: 1, masked: 1 };
  if (known[s]) return t(`ui.fix.source.${s}`);
  return t(`ui.fix.source.${["join", "segment", "drop"].includes(s) ? "structure" : "other"}`);
}

function fixPanel(meta, word) {
  const fix = word.fix;
  const kept = word.review === "accepted";
  const final = fix.op === "drop" ? t("ui.fix.removed") : word.text;
  return [
    cropView(meta.id, word),
    h("div", { class: "fixrow", id: "fix-before" }, h("span", { class: "label" }, t("ui.fix.before")), h("span", { class: "mono" }, word.original || "")),
    h("div", { class: "fixrow", id: "fix-after" }, h("span", { class: "label" }, t("ui.fix.after")), h("span", { class: "mono blue", id: "reading" }, final)),
    h("div", { class: "row" }, h("span", { class: "badge blue", id: "word-state" }, t("ui.state.fixed")), kept ? h("span", { class: "badge green", id: "fix-kept" }, t("ui.fix.kept")) : null, fix.op === "join" ? h("span", { class: "small muted" }, t("ui.fix.joined", { n: fix.ids.length })) : null, fix.op === "segment" ? h("span", { class: "small muted" }, t("ui.fix.split")) : null),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.fix.from")), h("span", { id: "fix-source" }, sourceLabel(fix))),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.fix.confidence")), h("span", { class: "mono", id: "fix-confidence" }, `${Math.round(fix.p_fix * 100)} %`)),
    fix.reasons.length ? h("ul", { class: "reasons", id: "fix-reasons" }, fix.reasons.map((r) => h("li", null, reasonText(r)))) : null,
    h("div", { class: "row" }, h("button", { class: "btn grow", id: "undo", onclick: () => queue(() => act("undo")) }, t("ui.fix.undo"), h("kbd", null, "U")), h("button", { class: "btn grow", id: "keep", disabled: kept, onclick: () => queue(() => act("accept")) }, t("ui.fix.keep"), h("kbd", null, "Enter"))),
  ];
}

const CROP_PAD = 8;

function cropView(docId, word) {
  const mark = h("div", { class: "crop-mark" });
  const img = h("img", { id: "crop", src: link(`/api/crop/${docId}/${word.id}.png`, { v: word.text }), alt: t("ui.review.crop") });
  img.addEventListener("load", () => {
    const [l, tp, r, b] = word.box;
    const x0 = Math.max(0, Math.floor(l) - CROP_PAD), y0 = Math.max(0, Math.floor(tp) - CROP_PAD);
    const w = img.naturalWidth, hgt = img.naturalHeight;
    Object.assign(mark.style, { left: `${((l - x0) / w) * 100}%`, top: `${((tp - y0) / hgt) * 100}%`, width: `${((r - l) / w) * 100}%`, height: `${((b - tp) / hgt) * 100}%` });
  });
  return h("div", { class: "crop" }, h("div", { class: "crop-frame" }, img, mark));
}

function stateBadges(word) {
  const numbers = numbersOf(word.id);
  const worst = ["mismatch", "checked", "unverified"].map((st) => numbers.find((n) => n.status === st)).find(Boolean);
  const unreviewed = !word.review;
  const badges = [];
  if (!(worst && word.status === "ok" && unreviewed)) {
    const tone = word.review === "accepted" ? "green" : word.review === "corrected" ? "accent" : word.status === "doubtful" ? "amber" : "";
    badges.push(h("span", { class: `badge ${tone}`, id: "word-state" }, t(`ui.state.${word.review || word.status}`)));
  }
  if (worst) {
    const tone = worst.status === "checked" ? "green" : worst.status === "mismatch" ? "red" : "";
    badges.push(h("span", { class: `badge ${tone}`, id: "number-state" }, t(`ui.status.${worst.status}`)));
  }
  if (word.p_wrong !== null && word.status === "doubtful") badges.push(h("span", { class: "muted small mono" }, `${Math.round(word.p_wrong * 100)} %`));
  return badges;
}

function numberBlock(n) {
  if (!n.rules.length && !n.suggestion) return null;
  return h(
    "div",
    { class: "field", "data-number": n.text },
    h("div", { class: "row between" }, h("span", { class: "label" }, t("ui.review.rules")), h("span", { class: `badge ${n.status === "checked" ? "green" : n.status === "mismatch" ? "red" : ""}` }, t(`ui.status.${n.status}`))),
    n.rules.map((r) => h("div", { class: `rule ${r.result}`, "data-rule": r.key, style: "cursor:pointer", onclick: () => { S.ruleFocus = S.ruleFocus === r.key ? null : r.key; queue(showPage); } }, h("span", { class: "grow" }, t(r.key, r.params)), h("span", { class: `badge ${r.result === "pass" ? "green" : r.result === "fail" ? "red" : ""}` }, t(`checks.result.${r.result}`)))),
    S.involved.size ? h("div", { class: "small muted", id: "involved" }, tn("ui.review.involved", S.involved.size)) : null,
  );
}

function fieldsTable(meta) {
  const inv = meta && meta.invoice;
  if (!inv) return h("div", { class: "note" }, t("ui.fields.none"));
  const rows = [];
  const add = (label, cell) => cell && rows.push(h("tr", { class: "pick", ...pressable(() => cell.word_ids[0] && (S.reviewTab = "page", queue(() => select(meta.id, cell.word_ids[0])).then(render))) }, h("td", null, label), h("td", { class: "mono" }, cell.text), h("td", null, h("span", { class: `badge ${cell.status === "checked" ? "green" : cell.status === "mismatch" ? "red" : ""}` }, t(`ui.status.${cell.status}`)))));
  for (const f of ["number", "date", "subtotal", "withholding_rate", "withholding", "total"]) add(t(`ui.field.${f}`), inv[f]);
  inv.vat.forEach((v, i) => { add(`${t("ui.field.vat_rate")} ${i + 1}`, v.rate); add(`${t("ui.field.vat_base")} ${i + 1}`, v.base); add(`${t("ui.field.vat_amount")} ${i + 1}`, v.amount); });
  const lines = inv.lines.map((l, i) => h("tr", { class: "pick" }, h("td", null, i + 1), ["description", "quantity", "unit_price", "amount"].map((c) => h("td", { class: "mono", ...(l[c] && l[c].word_ids[0] ? pressable(() => (S.reviewTab = "page", queue(() => select(meta.id, l[c].word_ids[0])).then(render))) : {}) }, l[c] ? h("span", null, l[c].text, " ", h("span", { class: `badge ${l[c].status === "checked" ? "green" : l[c].status === "mismatch" ? "red" : ""}` }, t(`ui.status.${l[c].status}`))) : "-"))));
  return h("div", { class: "field", style: "gap:18px" }, h("table", { class: "fields" }, h("thead", null, h("tr", null, h("th", null, t("ui.fields.field")), h("th", null, t("ui.fields.value")), h("th", null, t("ui.fields.status")))), h("tbody", null, rows)), inv.lines.length ? h("table", { class: "fields" }, h("thead", null, h("tr", null, h("th", null, "#"), ["description", "quantity", "unit_price", "amount"].map((c) => h("th", null, t(`ui.field.${c}`))))), h("tbody", null, lines)) : null);
}

/* Export */

function exportPreview(name) {
  const stem = name.replace(/\.[^.]*$/, "");
  const today = new Date().toISOString().slice(0, 10);
  return S.exportState.pattern.replace("{name}", stem).replace("{date}", today);
}

function exportKey(ids, ex) {
  return JSON.stringify([ids, ex.destination, ex.pattern, ex.formats]);
}

function fileName(path) {
  return path.split(/[\\/]/).pop();
}

// The names that were taken: a written file whose name does not start with the name the pattern gives.
function renamedFiles(result, pattern) {
  const wanted = Object.fromEntries(Object.keys(result.written).map((id) => [id, S.docs[id] ? expandName(pattern, S.docs[id].name) : null]));
  return Object.entries(result.written).some(([id, paths]) => wanted[id] && paths.some((p) => !fileName(p).startsWith(`${wanted[id]}.`)));
}

function expandName(pattern, name) {
  return pattern.replace("{name}", name.replace(/\.[^.]*$/, "")).replace("{date}", new Date().toISOString().slice(0, 10)).trim();
}

function exportSummary(result) {
  const files = Object.values(result.written).flat();
  const docs = Object.keys(result.written).length;
  return h(
    "div",
    { class: "success", id: "export-summary", role: "status" },
    h("div", { class: "label" }, t("ui.export.done")),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.export.summary_documents")), h("b", { class: "mono", "data-docs": docs }, docs)),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.export.summary_files")), h("b", { class: "mono", "data-files": files.length }, files.length)),
    h("div", { class: "kv" }, h("span", { class: "muted" }, t("ui.export.summary_folder")), h("span", { class: "mono path", id: "export-folder", title: result.destination }, result.destination)),
    h("div", { class: "row" }, h("button", { class: "btn", id: "open-folder", onclick: () => api("/api/open_folder", { path: result.destination }).catch(fail) }, t("ui.export.open_folder"))),
    renamedFiles(result, result.pattern) ? h("div", { class: "small muted", id: "export-renamed" }, t("ui.export.renamed")) : null,
    h("details", null, h("summary", null, t("ui.export.written")), h("ul", { class: "written", id: "export-written" }, Object.entries(result.written).map(([id, paths]) => h("li", null, h("b", null, S.docs[id] ? S.docs[id].name : id), h("div", { class: "mono small muted" }, paths.map(fileName).join("  ")))))),
    Object.entries(result.failed).map(([id, e]) => h("div", { class: "error" }, t("ui.export.failed", { name: S.docs[id] ? S.docs[id].name : id }), ": ", errorText(e))),
  );
}

function exportScreen() {
  const ex = S.exportState;
  if (!ex.picked) ex.picked = new Set(S.order);
  if (!ex.destination) ex.destination = S.defaults.export_dir;
  const picked = S.order.filter((id) => ex.picked.has(id));
  const first = S.docs[picked[0] || S.order[0]];
  const done = !!ex.result && ex.result.key === exportKey(picked, ex);
  const side = h("aside", { class: "side" }, h("span", { class: "label" }, t("ui.export.documents")), S.order.map((id) => h("label", { class: `card ${ex.picked.has(id) ? "selected" : ""}` }, h("div", { class: "row" }, h("input", { type: "checkbox", checked: ex.picked.has(id), onchange: (e) => { e.target.checked ? ex.picked.add(id) : ex.picked.delete(id); render(); } }), h("span", { class: "name grow" }, S.docs[id].name)), h("div", { class: "small muted" }, docStateText(S.docs[id])), noSecondReading(S.docs[id]).length ? h("div", { class: "warn small", "data-export-second-failed": id }, t("ui.second.failed_doc", { pages: noSecondReading(S.docs[id]).join(", ") })) : null)));
  const submit = (e) => {
    if (e.key === "Enter" && picked.length) {
      e.preventDefault();
      doExport(picked);
    }
  };
  const main = h(
    "section",
    { class: "main main-pad" },
    errorBox(),
    ex.result ? exportSummary(ex.result) : null,
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.export.destination")), h("input", { type: "text", id: "destination", value: ex.destination, title: ex.destination, oninput: (e) => { ex.destination = e.target.value; clearError(); }, onkeydown: submit })),
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.export.pattern")), h("input", { type: "text", id: "pattern", value: ex.pattern, oninput: (e) => { ex.pattern = e.target.value; clearError(); const p = document.getElementById("preview"); if (p && first) p.textContent = exportPreview(first.name); }, onkeydown: submit }), h("div", { class: "small muted" }, t("ui.export.example"), " ", h("span", { class: "mono", id: "preview" }, first ? exportPreview(first.name) : ""), h("span", { class: "mono muted" }, ex.formats.length ? `.${ex.formats[0] === "pdf" ? "searchable.pdf" : ex.formats[0]}` : ""))),
    h("div", { class: "field" }, h("span", { class: "label" }, t("ui.export.formats")), ["pdf", "xlsx", "json", "txt"].map((f) => h("label", { class: "check" }, h("input", { type: "checkbox", checked: ex.formats.includes(f), onchange: (e) => toggle(ex.formats, f, e.target.checked, 1) }), t(`ui.format.${f}`)))),
    h("div", { class: "note" }, t("ui.export.note")),
  );
  const aside = h("aside", { class: "aside" }, h("div", { class: "stat" }, h("b", null, picked.length), h("span", { class: "muted small" }, tn("ui.export.selected", picked.length))), h("div", { class: "grow" }), h("div", { class: "actionbar" }, h("button", { class: "btn primary wide", id: "export", disabled: !picked.length, onclick: () => doExport(picked) }, done ? t("ui.export.again") : tn("ui.export.go", picked.length)), h("button", { class: "btn wide", onclick: () => go("review") }, t("ui.export.cancel"))));
  return frame(side, main, aside);
}

async function doExport(ids) {
  const ex = S.exportState;
  const key = exportKey(ids, ex);
  try {
    S.error = null;
    ex.result = { ...(await api("/api/export", { docs: ids, destination: ex.destination, pattern: ex.pattern, formats: ex.formats, language: S.lang })), key, pattern: ex.pattern };
  } catch (e) {
    S.error = errorText(e);
  }
  render();
  const summary = document.getElementById("export-summary");
  if (summary && summary.scrollIntoView) summary.scrollIntoView({ block: "nearest" });
}

// language and start

async function setLang(lang) {
  const { strings } = await api(`/api/i18n/${lang}.json`);
  S.strings = strings;
  S.lang = lang;
  try {
    localStorage.setItem("extract.lang", lang);
  } catch (e) {}
  render();
  if (S.tab === "review") queue(showPage);
}

async function start() {
  S.defaults = await api("/api/defaults");
  const saved = recall();
  if (saved) {
    S.files = Array.isArray(saved.files) ? saved.files : [];
    Object.assign(S.opts, saved.opts || {});
  }
  S.opts.correct = params.get("correct") || (saved && saved.opts && saved.opts.correct) || S.defaults.correct || "off";
  S.fixedOnly = params.get("filter") === "fixed";
  S.opts.second = (params.get("second") || (saved && saved.opts && saved.opts.second) || S.defaults.second_reading) === "thorough" ? "thorough" : "off";
  const wanted = (params.get("languages") || "").split(",").filter((l) => l === "es" || l === "en");
  if (wanted.length) S.opts.languages = [...new Set(wanted)];
  let lang = params.get("lang") || S.defaults.lang;
  try {
    lang = lang || localStorage.getItem("extract.lang");
  } catch (e) {}
  lang = lang || (navigator.language || "en").slice(0, 2);
  const { strings } = await api(`/api/i18n/${lang === "es" ? "es" : "en"}.json`);
  S.strings = strings;
  S.lang = lang === "es" ? "es" : "en";
  await loadDocs();
  const tab = params.get("tab");
  if (tab && TABS.includes(tab)) S.tab = tab;
  else if (S.order.length) S.tab = "review";
  if (!S.browse) {
    try {
      S.browse = await api(`/api/browse?path=${encodeURIComponent(params.get("path") || (saved && saved.path) || S.defaults.root)}`);
      S.browse.path = S.browse.path.replace(/\/$/, "");
    } catch (e) {}
  }
  render();
  for (const path of (params.get("add") || "").split(";").filter(Boolean)) await addFile(path);
  const pick = params.get("pick");
  if (S.order.length && params.get("doc")) S.cur.doc = S.order[Number(params.get("doc"))] || S.cur.doc;
  if (pick) {
    S.cur.page = pageOf(pick);
    S.cur.word = pick;
  }
  if (params.get("rtab") === "fields") S.reviewTab = "fields";
  if (S.tab === "review") {
    render();
    await queue(showPage);
  }
  if (S.tab === "export" && params.get("export") === "1" && S.order.length) {
    // a link that exports everything on arrival: it lets a screenshot show the screen after an export
    S.exportState.picked = new Set(S.order);
    await doExport(S.order);
  }
  pollLoop();
  if (params.get("selftest") === "fixes") await runFixesSelftest();
  else if (params.get("selftest") === "second") await runSecondSelftest();
  else if (params.get("selftest") === "ui") document.body.dataset.ready = "1";
  else if (params.get("selftest")) await runSelftest();
  document.body.dataset.ready = "1";
}

// self-test: drives the real key handlers and writes the result into the page

function press(key) {
  document.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
  return S.pending;
}

async function runSelftest() {
  const checks = [];
  const check = (name, ok, detail) => checks.push({ name, ok: !!ok, detail: detail === undefined ? "" : String(detail) });
  try {
    const withSuggestion = S.order.find((id) => S.docs[id].doubtful.length >= 4);
    check("a document with doubtful words is open", withSuggestion);
    S.tab = "review";
    S.cur = { doc: withSuggestion, page: 1, word: null };
    render();
    await queue(showPage);
    const meta = currentMeta();
    const list0 = meta.doubtful.map((d) => d.id);
    await press("n");
    check("N selects the first doubtful word", S.cur.word === list0[0], S.cur.word);
    await press("n");
    check("N again selects the second", S.cur.word === list0[1], S.cur.word);
    await press("p");
    check("P goes back to the first", S.cur.word === list0[0], S.cur.word);
    check("the right panel shows the reading", document.getElementById("reading").textContent === wordOf(S.cur.word).text);

    let guard = 0;
    while (!suggestionOf(wordOf(S.cur.word)) && guard++ < list0.length) await press("n");
    const target = S.cur.word;
    const suggestion = suggestionOf(wordOf(target));
    check("a doubtful word with a suggestion is selected", suggestion, target);
    await press("Tab");
    const used = wordOf(target);
    check("Tab uses the suggestion", used && used.text === suggestion && used.review === "corrected", used && `${used.text} ${used.review}`);
    check("a used suggestion leaves the list", !currentMeta().doubtful.some((d) => d.id === target));

    const second = S.cur.word;
    const before = currentMeta().doubtful.length;
    await press("Enter");
    const accepted = wordOf(second);
    check("Enter accepts the selected word", accepted && accepted.review === "accepted", accepted && accepted.review);
    check("an accepted word leaves the list", currentMeta().doubtful.length === before - 1);

    const third = S.cur.word;
    await press("e");
    const input = document.getElementById("edit-input");
    check("E opens the edit box", input && document.activeElement === input);
    input.value = "EDITED";
    input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    await S.pending;
    const edited = wordOf(third);
    check("Enter in the box stores the correction", edited && edited.text === "EDITED" && edited.review === "corrected", edited && edited.text);
    const log = await api(`/api/doc/${currentMeta().id}`);
    check("the corrections changed the pending count", log.doc.doubtful.length === before - 2, log.doc.doubtful.length);

    const total = reviewView.visibleCount();
    document.getElementById("only").click();
    const only = reviewView.visibleCount();
    check("Only doubtful hides the other boxes", only < total && reviewView.only, `${only} of ${total}`);
    document.getElementById("only").click();
    check("Only doubtful toggles back", !reviewView.only && reviewView.visibleCount() === total);

    const invoiceDoc = S.order.find((id) => S.docs[id].invoice && S.docs[id].invoice.lines.length);
    check("an invoice document with lines is open", invoiceDoc);
    const failing = S.docs[invoiceDoc];
    check("a number that does not add up counts as to check", failing.doubtful.length > 0 && failing.doubtful[0].status === "mismatch", failing.doubtful[0] && failing.doubtful[0].status);
    S.cur = { doc: invoiceDoc, page: 1, word: null };
    S.ruleFocus = null;
    await queue(showPage);
    const inv = currentMeta().invoice;
    const candidates = [inv.total, inv.subtotal, ...(inv.lines[0] ? [inv.lines[0].amount] : [])].filter(Boolean);
    let found = null;
    for (const cell of candidates) {
      const id = cell.word_ids[0];
      await select(invoiceDoc, id);
      if (S.involved.size) { found = id; break; }
    }
    check("a number with rules highlights the values involved", found && S.involved.size > 0, found && [...S.involved].join(","));
    check("the rules are listed in the panel", document.querySelectorAll("[data-rule]").length > 0, document.getElementById("right").innerText.slice(0, 300));
    const badge = document.getElementById("number-state");
    check("a word of a failing number shows the number status, not Fine", badge && badge.textContent === t("ui.status.mismatch") && !document.getElementById("word-state"), badge && badge.textContent);
    check("the highlighted words are drawn", reviewView.involved === S.involved && reviewView.involved.size > 0);
    await press("Escape");
    check("Esc deselects", S.cur.word === null && S.involved.size === 0);

  } catch (e) {
    check("the self-test ran to the end", false, `${e && e.message} ${e && e.stack}`);
  }
  const pass = checks.every((c) => c.ok);
  const out = h("pre", { id: "selftest-result", "data-pass": pass ? "1" : "0" }, JSON.stringify({ pass, checks }, null, 1));
  window.__selftest = out;
  document.getElementById("root").append(out);
}

async function runSecondSelftest() {
  const checks = [];
  const check = (name, ok, detail) => checks.push({ name, ok: !!ok, detail: detail === undefined ? "" : String(detail) });
  const hints = () => [...document.querySelectorAll(".second-hint")].map((e) => e.textContent).join(" | ");
  try {
    S.tab = "import";
    render();
    check("off is the default and the order says no second reading", S.opts.second === "off" && readOptions().second_reading === false, JSON.stringify(readOptions()));
    document.querySelector('[data-second="thorough"]').click();
    check("Thorough is sent as the word thorough", readOptions().second_reading === "thorough", JSON.stringify(readOptions()));
    document.querySelector('[data-correct="off"]').click();
    check("with corrections Off the gain is said to need More", hints().includes(t("ui.second.gain.other")), hints());
    document.querySelector('[data-correct="more"]').click();
    check("with both languages no figure is mixed in", hints().includes(t("ui.second.gain.unmeasured")), hints());
    S.opts.languages = ["es"];
    render();
    check("without p95 and RAM in the defaults the detail line is absent", !hints().includes(t("ui.second.thorough.detail", { p95: "6", ram: "1.6" })));
    const keep = S.defaults.second_reading_thorough_cost;
    S.defaults.second_reading_thorough_cost = { ...keep, seconds_p95: 6, reader_ram_mb: 1600, fixes_more_pct_by_language: { es: 2.3, en: 3 } };
    render();
    check("p95 and RAM show when the defaults carry them", hints().includes(t("ui.second.thorough.detail", { p95: oneDecimal(6), ram: oneDecimal(1.6) })), hints());
    S.opts.languages = ["en"];
    render();
    check("the gain is read per language", hints().includes(oneDecimal(3)) && !hints().includes(oneDecimal(2.3)), hints());
    S.defaults.second_reading_thorough_cost = keep;
    S.opts.languages = ["es"];
    render();
    const cost = S.defaults.second_reading_thorough_cost;
    check("with Spanish and More the figure of the defaults shows, in the language's format", hints().includes(oneDecimal(cost.fixes_more_pct)) && hints().includes(oneDecimal(cost.seconds_per_page)), hints());
    S.files = [{ path: "/x.pdf", name: "x.pdf", picked: true, pages: 4, scanned_pages: 4, digital_pages: 0, kind: "scanned", size: 1 }];
    const picked = S.files;
    S.opts.second = "off";
    render();
    const [off, offHigh] = estimateRange(picked);
    check("the page shows the estimate without the second reading", document.body.textContent.includes(estimateLine(picked)), estimateLine(picked));
    S.opts.second = "thorough";
    render();
    const [on, onHigh] = estimateRange(picked);
    check("the estimate adds the scanned pages times the measured seconds", Math.abs(onHigh - offHigh - 4 * S.defaults.second_reading_thorough_cost.seconds_per_page) < 1e-9 && on > off, `${offHigh} -> ${onHigh}`);
    check("the page shows the longer estimate", document.body.textContent.includes(estimateLine(picked)), estimateLine(picked));
    const missing = t("ui.second.missing");
    S.defaults.thorough_available = false;
    render();
    check("a missing model file is said before reading", hints().includes(missing), hints());
    S.defaults.thorough_available = true;
    render();
    check("an installed model file adds no warning", !hints().includes(missing), hints());
    document.querySelector('[data-second="off"]').click();
    check("Off goes back to false", readOptions().second_reading === false);
  } catch (e) {
    check("the self-test ran to the end", false, e && e.stack ? e.stack : e);
  }
  const out = document.createElement("pre");
  out.id = "selftest-result";
  out.textContent = JSON.stringify({ checks });
  document.body.append(out);
}

async function runFixesSelftest() {
  const checks = [];
  const check = (name, ok, detail) => checks.push({ name, ok: !!ok, detail: detail === undefined ? "" : String(detail) });
  const click = async (selector) => {
    document.querySelector(selector).click();
    await S.pending;
  };
  try {
    const fixedDoc = S.order.find((id) => S.docs[id].fixed.length === 5);
    const brakeDoc = S.order.find((id) => S.docs[id].fixes.withheld.length > 0);
    check("a document with 5 applied fixes is open", fixedDoc);
    check("a document with a withheld page is open", brakeDoc);
    S.tab = "review";
    S.cur = { doc: fixedDoc, page: 1, word: null };
    render();
    await queue(showPage);
    const meta = currentMeta();
    check("the filter counts the fixed words", document.getElementById("fixed-filter").textContent === t("ui.review.fixed_filter", { n: 5 }), document.getElementById("fixed-filter").textContent);

    await click('[data-mode="safe"]');
    check("Safe re-applies only the safe fixes, without reading again", meta.mode === "safe" && meta.fixed.length === 2, meta.fixed.length);
    await click('[data-mode="more"]');
    check("More applies the rest", meta.mode === "more" && meta.fixed.length === 5, meta.fixed.length);

    await click("#fixed-filter");
    check("Fixed (n) shows only the fixed boxes", reviewView.fixedOnly && reviewView.visibleCount() === 5, reviewView.visibleCount());
    await press("n");
    const first = wordOf(S.cur.word);
    check("N walks the fixed words", first && first.status === "fixed" && S.cur.word === meta.fixed[0].id, `${S.cur.word} ${first && first.status} ${meta.fixed[0].id}`);
    check("the panel shows the original and the final text", document.getElementById("fix-before").textContent.includes(first.original) && document.getElementById("reading").textContent === first.text);
    check("the join draws its members as one group", first.fix.op === "join" && S.involved.size === first.fix.ids.length - 1 && reviewView.involved === S.involved, S.involved.size);
    check("the source and the reasons are shown", document.getElementById("fix-source").textContent.length > 0 && document.querySelectorAll("#fix-reasons li").length > 0);
    check("the confidence is shown", /\d+ %/.test(document.getElementById("fix-confidence").textContent));

    await press("Enter");
    check("Enter keeps the fix and moves on", wordOf(first.id).review === "accepted" && wordOf(first.id).status === "fixed" && meta.fixed.length === 5 && S.cur.word !== first.id, S.cur.word);
    const second = wordOf(S.cur.word);
    await press("u");
    check("U undoes the fix: the text comes back and the count drops", wordOf(second.id).status !== "fixed" && wordOf(second.id).text === second.original && meta.fixed.length === 4, meta.fixed.length);
    check("the filter count follows", document.getElementById("fixed-filter").textContent === t("ui.review.fixed_filter", { n: 4 }));

    await click("#undo-page");
    check("Undo all on this page empties the fixed list", meta.fixed.length === 0 && meta.pages[0].fixed === 0, meta.fixed.length);
    check("the filter is disabled with nothing fixed", document.getElementById("fixed-filter").disabled);

    await click('[data-mode="off"]');
    check("Off keeps the text as read", meta.mode === "off" && meta.fixed.length === 0);
    await click('[data-mode="safe"]');
    check("an undone fix is not applied again by a mode switch", meta.mode === "safe" && meta.fixed.length === 0, meta.fixed.length);

    S.cur = { doc: brakeDoc, page: 1, word: null };
    S.fixedOnly = false;
    render();
    await queue(showPage);
    const notice = document.getElementById("withheld");
    check("the page brake shows its notice", notice && notice.textContent.includes("1"), notice && notice.textContent);
    check("nothing was applied on the braked page", currentMeta().fixed.length === 0);
    check("no notice when every page had its second reading", !document.getElementById("second-failed"));
    currentMeta().fixes.second_reading_failed_pages = [1, 3];
    render();
    check("a document with pages without second reading shows the list and the page sentence", document.getElementById("second-failed") && document.getElementById("second-failed").textContent.includes("1, 3") && document.getElementById("second-failed-page"), document.getElementById("second-failed") && document.getElementById("second-failed").textContent);
    check("its card carries the count badge", document.querySelector("[data-second-failed]") && document.querySelector("[data-second-failed]").getAttribute("data-second-failed") === "2");
    currentMeta().fixes.second_reading_failed_pages = [3];
    render();
    check("the page sentence is absent on a page that had its second reading", document.getElementById("second-failed") && !document.getElementById("second-failed-page"));
    S.tab = "export";
    render();
    check("Export warns on the document too", document.querySelector("[data-export-second-failed]") && document.querySelector("[data-export-second-failed]").textContent.includes("3"));
    currentMeta().fixes.second_reading_failed_pages = [];
    render();
    check("an empty list removes the Export notice", !document.querySelector("[data-export-second-failed]"));
    S.tab = "review";
    render();
    await click('[data-mode="off"]');
    await click('[data-mode="more"]');
    check("the mode switch re-applies without reading again", currentMeta().mode === "more" && document.querySelector('[data-mode="more"]').getAttribute("aria-pressed") === "true");
  } catch (e) {
    check("the self-test ran to the end", false, `${e && e.message} ${e && e.stack}`);
  }
  const pass = checks.every((c) => c.ok);
  const out = h("pre", { id: "selftest-result", "data-pass": pass ? "1" : "0" }, JSON.stringify({ pass, checks }, null, 1));
  window.__selftest = out;
  document.getElementById("root").append(out);
}

async function runPerf() {
  const canvas = h("canvas", { style: "width:1200px;height:700px" });
  document.getElementById("root").append(canvas);
  const words = Array.from({ length: 1500 }, (_, i) => ({ id: `p1w${i}`, text: "x", box: [(i % 50) * 30, Math.floor(i / 50) * 20, (i % 50) * 30 + 26, Math.floor(i / 50) * 20 + 16], status: ["ok", "doubtful", "unscored", "native"][i % 4], review: null }));
  const view = new PageView(canvas);
  view.setPage(null, 1500, 600, words);
  view.selected = "p1w7";
  view.involved = new Set(["p1w8", "p1w9"]);
  const times = [];
  for (let i = 0; i < 15; i++) {
    view.draw();
    times.push(view.lastDraw);
  }
  times.sort((a, b) => a - b);
  const median = times[7];
  const pass = median < 50 && view.visibleCount() === 1500;
  document.getElementById("root").append(h("pre", { id: "perf-result", "data-pass": pass ? "1" : "0" }, JSON.stringify({ pass, boxes: view.visibleCount(), median_ms: median, worst_ms: times[14] })));
}

if (params.get("perf")) runPerf();
else start().catch((e) => {
  document.getElementById("root").textContent = `Extract: ${e.message}`;
});
