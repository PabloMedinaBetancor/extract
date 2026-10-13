"use strict";
// Self-test of the screen fixes of 5.0.1: it drives the real page and writes the result into the DOM, as the other
// self-tests do (tests/test_ui_v501.py reads it). `?selftest=ui&section=NAME` runs one section.

(function () {
  const params = new URLSearchParams(location.search);
  const section = params.get("section") || "";
  const checks = [];
  const check = (name, ok, detail) => checks.push({ name, ok: !!ok, detail: detail === undefined ? "" : String(detail) });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function until(cond, tries = 200) {
    for (let i = 0; i < tries; i++) {
      if (await cond()) return true;
      await sleep(50);
    }
    return false;
  }
  const $ = (selector) => document.querySelector(selector);
  const typeInto = (el, value) => {
    el.value = value;
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };
  const key = (el, name) => el.dispatchEvent(new KeyboardEvent("keydown", { key: name, bubbles: true }));

  const sections = {};

  sections.export = async () => {
    S.tab = "export";
    S.exportState.destination = params.get("dest");
    S.exportState.formats = ["txt", "json"];
    S.exportState.picked = null;
    render();
    const n = S.order.length;
    check("before exporting the button names the count", $("#export").textContent === tn("ui.export.go", n), $("#export").textContent);
    check("no summary before exporting", !$("#export-summary"));
    $("#export").click();
    check("a summary appears at the top of the page", await until(() => $("#export-summary")));
    const summary = $("#export-summary");
    check("it counts the documents", summary.querySelector("[data-docs]").dataset.docs === String(n), summary.textContent);
    check("it counts the files", summary.querySelector("[data-files]").dataset.files === String(2 * n), summary.textContent);
    check("it names the folder once", $("#export-folder").textContent === params.get("dest"));
    const written = $("#export-written").textContent;
    check("the file list gives names, not long paths", written.includes("doubtful.txt") && !written.includes(params.get("dest")), written);
    check("the button now says Export again", $("#export").textContent === t("ui.export.again"), $("#export").textContent);
    check("an Open folder action is offered", $("#open-folder") && $("#open-folder").textContent === t("ui.export.open_folder"));
    $("#open-folder").click();
    await sleep(300);
    check("nothing was renamed the first time", !$("#export-renamed"));
    typeInto($("#pattern"), "{name}-b");
    check("changing the pattern puts the count back on the button", $("#export").textContent === tn("ui.export.go", n) || $("#export").textContent === t("ui.export.again"), $("#export").textContent);
    S.exportState.pattern = "{name}";
    render();
    check("the first choice is the one that was exported", $("#export").textContent === t("ui.export.again"));
    key($("#destination"), "Enter");
    check("Enter in the folder field exports again", await until(() => $("#export-renamed")), document.body.innerText.slice(0, 300));
    check("the second export explains the -2 names", $("#export-renamed").textContent === t("ui.export.renamed"));
  };

  sections.exportlink = async () => {
    check("a link with export=1 shows the screen after an export", await until(() => $("#export-summary")));
    check("it exported every document", $("#export-summary [data-docs]").dataset.docs === String(S.order.length));
    check("and the button says Export again", $("#export").textContent === t("ui.export.again"));
  };

  sections.addall = async () => {
    check("the folder is open", await until(() => $("#add-all") && !$("#add-all").disabled));
    $("#add-all").click();
    check("a report names the files that were skipped", await until(() => $("#add-report")));
    const bad = ["corrupt.pdf", "empty.pdf", "locked.pdf", "text.pdf"];
    for (const name of bad) {
      const item = document.querySelector(`[data-skipped="${name}"]`);
      check(`${name} is named with its own reason`, item && item.textContent.length > name.length + 4, item && item.textContent);
    }
    const locked = document.querySelector('[data-skipped="locked.pdf"]');
    check("the locked PDF says it is protected by a password", locked && locked.textContent.includes(t("error.password_protected")), locked && locked.textContent);
    check("the report counts the skipped files", $("#add-report").textContent.includes(tn("ui.import.skipped", bad.length)), $("#add-report").textContent);
    check("the readable files were added", S.files.length === 2 && S.files.every((f) => f.pages > 0), S.files.map((f) => f.name).join(","));
    check("no generic banner replaces the report", !$(".banner"));
    await addFile(`${params.get("path")}/empty.pdf`);
    const one = document.querySelectorAll("[data-skipped]");
    check("a single Add of a bad file also names it", one.length === 1 && one[0].dataset.skipped === "empty.pdf", one.length);
    await addAll();
    check("adding the same files again says they were already there", $("#add-report").textContent.includes(tn("ui.import.already", 2)) || $("#add-report").textContent.includes(tn("ui.import.skipped", 4)), $("#add-report").textContent);
  };

  // The app in an iframe of a given size: layout rules are answered by the iframe's own viewport.
  async function frameOf(width, height, query) {
    const frame = document.createElement("iframe");
    Object.assign(frame.style, { position: "absolute", top: "0", left: "0", width: `${width}px`, height: `${height}px`, border: "0", opacity: "0", pointerEvents: "none" });
    frame.src = `${location.origin}/?t=${TOKEN}&nowait=1&lang=${params.get("lang") || "en"}&${query}`;
    document.body.append(frame);
    await until(() => frame.contentDocument && frame.contentDocument.body && frame.contentDocument.body.dataset.ready === "1", 400);
    return frame;
  }
  const inside = (rect, w, h) => rect.left >= -1 && rect.right <= w + 1 && rect.top >= -1 && rect.bottom <= h + 1;

  sections.layout = async () => {
    const pdf = params.get("pdf");
    const sizes = [[480, 800], [800, 1000], [1100, 800], [1200, 1920], [1920, 1200], [2400, 900]];
    for (const [w, h] of sizes) {
      const label = `${w}x${h}`;
      for (const tab of ["import", "review", "export"]) {
        const frame = await frameOf(w, h, `tab=${tab}&doc=0&pick=p1w1&add=${encodeURIComponent(pdf)}`);
        const doc = frame.contentDocument;
        check(`${label} ${tab}: the page does not scroll sideways`, doc.documentElement.scrollWidth <= w, doc.documentElement.scrollWidth);
        const tabs = [...doc.querySelectorAll(".tab")].map((el) => el.getBoundingClientRect());
        check(`${label} ${tab}: the four tabs are on screen`, tabs.length === 4 && tabs.every((r) => r.left >= -1 && r.right <= w + 1), JSON.stringify(tabs.map((r) => [r.left, r.right])));
        if (tab === "import") {
          const read = () => doc.getElementById("read-button").getBoundingClientRect();
          check(`${label} import: the Read button is on screen`, inside(read(), w, h), JSON.stringify(read()));
          doc.querySelector('[data-second="thorough"]').click();
          await sleep(100);
          check(`${label} import: and stays on screen with Thorough on`, inside(read(), w, h), JSON.stringify(read()));
        }
        if (tab === "review") {
          const canvas = doc.getElementById("page-canvas").getBoundingClientRect();
          check(`${label} review: the page view is big enough to use`, canvas.width >= Math.min(300, w - 24) && canvas.height >= 300, `${canvas.width}x${canvas.height}`);
          const right = doc.getElementById("right").getBoundingClientRect();
          check(`${label} review: the word panel is reachable`, right.left >= -1 && right.right <= w + 1 && right.width >= Math.min(280, w - 40), JSON.stringify([right.left, right.right]));
          const view = frame.contentWindow.eval("JSON.stringify([reviewView.view, reviewView.width, reviewView.height, reviewView.canvas.getBoundingClientRect().width, reviewView.canvas.getBoundingClientRect().height])");
          const [v, pw, ph, cw, ch] = JSON.parse(view);
          check(`${label} review: the page is centred in its view`, Math.abs(v.x - (cw - pw * v.s) / 2) < 2 && Math.abs(v.y - (ch - ph * v.s) / 2) < 2, view);
        }
        frame.remove();
      }
    }
    // a window that grows after the page is open: the page is fitted again, not left in the corner
    const frame = await frameOf(1200, 900, "tab=review&doc=0&pick=p1w1");
    const state = () => JSON.parse(frame.contentWindow.eval("JSON.stringify([reviewView.view, reviewView.width, reviewView.height, reviewView.canvas.getBoundingClientRect().width, reviewView.canvas.getBoundingClientRect().height])"));
    frame.style.width = "2400px";
    await sleep(200);
    frame.contentWindow.eval("reviewView.onResize()"); // what the ResizeObserver of the canvas calls
    const [v, pw, ph, cw, ch] = state();
    check("after the window grows the page is centred again", Math.abs(v.x - (cw - pw * v.s) / 2) < 2 && Math.abs(v.y - (ch - ph * v.s) / 2) < 2, JSON.stringify(state()));
    frame.contentWindow.eval("reviewView.zoomAt(1.5)");
    const before = state()[0];
    frame.style.width = "1500px";
    await sleep(200);
    frame.contentWindow.eval("reviewView.onResize()");
    check("a view the user zoomed is left as it is", Math.abs(state()[0].s - before.s) < 1e-9, JSON.stringify([before, state()[0]]));
    frame.remove();
  };

  // The Read screen is fed the events the server would send.
  sections.read = async () => {
    const id = S.order[0];
    S.tab = "read";
    handleEvent({ event: "job.started", names: ["a.pdf", "b.pdf", "c.pdf"], pages: 30, time: 1000 });
    handleEvent({ event: "doc.begin", index: 0, pages: 10, name: "a.pdf", time: 1000 });
    for (let i = 1; i <= 10; i++) handleEvent({ event: "page.read", index: 0, page: i, status: "ocr", doubtful: 0, fixed: 0, time: 1000 + 2 * i });
    handleEvent({ event: "doc.read", index: 0, doc_id: id, name: "a.pdf", status: "done", doubtful: 1, fixed: 0, numbers_checked: 0, unreadable: [], time: 1021 });
    await sleep(400);
    check("the Read screen stays when a document is read: no jump to Review", S.tab === "read", S.tab);
    check("a Review button is offered while the read goes on", $("#to-review") && $("#to-review").textContent === t("ui.read.review_now"), $("#to-review") && $("#to-review").textContent);
    $("#to-review").click();
    check("it opens Review when pressed", S.tab === "review", S.tab);
    S.tab = "read";
    render();
    check("the time left comes from the last pages: 2 s a page, 20 left, 40 s", Math.abs(etaSeconds(S.job.stamps, 20) - 40) < 1e-9, etaSeconds(S.job.stamps, 20));
    check("with fewer than 3 pages it says it is calculating", isNaN(etaSeconds([0, 1000], 5)));
    const slow = [0, 90000, ...Array.from({ length: 10 }, (_, i) => 91000 + i * 1000)];
    check("a slow start does not weigh on the time left", Math.abs(etaSeconds(slow, 10) - 10) < 1e-9, etaSeconds(slow, 10));
    check("nothing left means no time left", etaSeconds([0, 1000, 2000], 0) === 0);
    handleEvent({ event: "doc.begin", index: 1, pages: 100, name: "b.pdf", time: 1022 });
    for (let i = 1; i <= 18; i++) handleEvent({ event: "page.read", index: 1, page: i, status: "ocr", doubtful: 0, fixed: 0, time: 1022 + i });
    check("while reading the card says Reading", document.body.textContent.includes(t("ui.read.status.reading")));
    $("#stop").click();
    check("Stop says it is stopping and cannot be pressed twice", $("#stop").textContent === t("ui.read.stopping") && $("#stop").disabled, $("#stop").textContent);
    handleEvent({ event: "job.finished", status: "stopped", time: 1050 });
    const cards = [...document.querySelectorAll("[data-doc]")].map((c) => c.textContent);
    check("the document that was being read says it was not kept", cards[1].includes(t("ui.read.status.stopped")), cards[1]);
    check("the document never started says it was not read", cards[2].includes(t("ui.read.status.skipped")), cards[2]);
    check("the document already read is still Read", cards[0].includes(t("ui.read.status.done")), cards[0]);
    check("the note says what is kept and what is not", $("#job-status").textContent === t("ui.read.finished.stopped"));
    check("the time elapsed stops when the read ends", $("#elapsed").textContent === fmtTime(50), $("#elapsed").textContent);
  };

  async function setupReview() {
    const doubtfulId = S.order.find((id) => S.docs[id].doubtful.length >= 4);
    S.tab = "review";
    S.cur = { doc: doubtfulId, page: 1, word: null };
    render();
    await queue(showPage);
    return doubtfulId;
  }

  sections.badges = async () => {
    const doubtfulId = await setupReview();
    // Ready and Partial never together
    const base = S.docs[doubtfulId];
    const text = (d) => docBadges(d).filter(Boolean).map((b) => b.textContent).join("|");
    check("a partial document with nothing to check is not also Ready", text({ ...base, doubtful: [], fixed: [], status: "partial" }) === t("ui.review.partial"), text({ ...base, doubtful: [], fixed: [], status: "partial" }));
    check("a complete document with nothing to check is Ready", text({ ...base, doubtful: [], fixed: [], status: "done" }) === t("ui.review.ready"));
    check("a partial document with words to check says both", text({ ...base, fixed: [], status: "partial" }) === `${t("ui.review.to_check", { k: base.doubtful.length })}|${t("ui.review.partial")}`, text({ ...base, fixed: [], status: "partial" }));
    check("the Export list says Partial too", docStateText({ ...base, doubtful: [], status: "partial" }) === t("ui.review.partial"));
    check("the badges of a card wrap instead of overflowing", document.querySelector("#left .card .row.wrap"));
  };

  sections.fields = async () => {
    await setupReview();
    const invoiceId = S.order.find((id) => S.docs[id].invoice);
    S.cur = { doc: invoiceId, page: 1, word: null };
    S.reviewTab = "fields";
    render();
    await queue(showPage);
    check("the Fields view has no Only doubtful toggle", !$("#only"));
    check("it has no Fixed filter", !$("#fixed-filter"));
    check("it has no colour legend", !$(".legend"));
    check("its hint is about the rows, not about clicking a word", $("#right").textContent.includes(t("ui.fields.pick")) && !$("#right").textContent.includes(t("ui.review.pick")), $("#right").textContent.slice(0, 120));
    S.reviewTab = "page";
    render();
    await queue(showPage);
    check("the page view has them back", $("#only") && $("#fixed-filter") && $(".legend"));
  };

  sections.reviewtouch = async () => {
    const doubtfulId = await setupReview();
    check("the keyboard hints of a button are hidden from screen readers", document.querySelectorAll("button kbd").length > 0 && [...document.querySelectorAll("button kbd")].every((k) => k.getAttribute("aria-hidden") === "true"));
    check("a Previous and a Next button exist for touch", $("#prev") && $("#next") && !$("#next").disabled);
    $("#next").click();
    await S.pending;
    const first = S.cur.word;
    check("Next selects the first doubtful word", first === S.docs[doubtfulId].doubtful[0].id, first);
    $("#next").click();
    await S.pending;
    check("Next goes on to the second", S.cur.word === S.docs[doubtfulId].doubtful[1].id, S.cur.word);
    $("#prev").click();
    await S.pending;
    check("Previous goes back", S.cur.word === first, S.cur.word);
  };

  sections.reopen = async () => {
    const doubtfulId = await setupReview();
    await press("n");
    const first = S.cur.word;
    const before = S.docs[doubtfulId].doubtful.length;
    $("#accept").click();
    await S.pending;
    check("Accept takes the word off the list", S.docs[doubtfulId].doubtful.length === before - 1);
    await select(doubtfulId, first);
    check("an accepted word has its Accept button off", $("#accept").disabled);
    check("and a Reopen button", $("#reopen") && $("#reopen").textContent.startsWith(t("ui.review.reopen")));
    $("#reopen").click();
    await S.pending;
    check("Reopen puts it back on the list, undecided", S.docs[doubtfulId].doubtful.length === before && wordOf(first).review === null, wordOf(first).review);
    check("and stays on the word", S.cur.word === first && !$("#reopen") && !$("#accept").disabled);
    await press("Enter");
    await select(doubtfulId, first);
    await press("u");
    check("U reopens an accepted word", wordOf(first).review === null && S.docs[doubtfulId].doubtful.length === before, wordOf(first).review);
    // a retyped word
    await select(doubtfulId, first);
    const original = wordOf(first).text;
    await press("e");
    typeInto($("#edit-input"), "RETYPED");
    key($("#edit-input"), "Enter");
    await S.pending;
    await select(doubtfulId, first);
    check("a retyped word offers Reopen", wordOf(first).text === "RETYPED" && $("#reopen"));
    $("#reopen").click();
    await S.pending;
    check("Reopen gives back the text it was read with", wordOf(first).text === original && wordOf(first).review === null, wordOf(first).text);
  };

  sections.touch = async () => {
    check("a finger gets a 44 px target", TOUCH_SLOP === 22);
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;left:0;top:0;width:600px;height:600px";
    document.body.append(canvas);
    const view = new PageView(canvas);
    view.setWords([
      { id: "p1w1", text: "a", box: [100, 100, 140, 108], status: "doubtful", review: null },
      { id: "p1w2", text: "b", box: [100, 300, 140, 308], status: "ok", review: null },
      { id: "p1w3", text: "c", box: [100, 330, 140, 338], status: "absorbed", absorbed_by: "p1w2", review: null },
    ]);
    view.view = { s: 1, x: 0, y: 0 };
    check("a point just below a word hits nothing with a mouse", view.hit(120, 122, 0) === null);
    check("and the word with a finger", view.hit(120, 122, TOUCH_SLOP) && view.hit(120, 122, TOUCH_SLOP).id === "p1w1");
    check("a point far from every word hits nothing even with a finger", view.hit(120, 200, TOUCH_SLOP) === null);
    check("the nearest word wins", view.hit(120, 296, TOUCH_SLOP).id === "p1w2" && view.hit(120, 322, TOUCH_SLOP).id === "p1w2");
    check("an absorbed word is never hit", !view.hit(120, 336, TOUCH_SLOP) || view.hit(120, 336, TOUCH_SLOP).id !== "p1w3");
    view.view = { s: 0.5, x: 0, y: 0 };
    check("the slop is in screen pixels, whatever the zoom", view.hit(60, 58, TOUCH_SLOP).id === "p1w1");
    canvas.remove();
    const meta = document.querySelector('meta[name="viewport"]');
    check("the keyboard resizes the page instead of covering it", meta.content.includes("interactive-widget=resizes-content"), meta.content);
    check("the page says it is not to be translated by the browser", document.documentElement.getAttribute("translate") === "no" && document.querySelector('meta[name="google"]'));
    check("the page language follows the interface", document.documentElement.lang === S.lang);
    S.lang = "es";
    render();
    check("and changes with it", document.documentElement.lang === "es");
  };

  sections.plural = async () => {
    check("1 fixed word uses the singular text", tn("ui.review.fixed_count", 1) === t("ui.review.fixed_count_one"), tn("ui.review.fixed_count", 1));
    check("2 fixed words use the plural text with the count", tn("ui.review.fixed_count", 2) === t("ui.review.fixed_count", { n: 2 }) && tn("ui.review.fixed_count", 2).includes("2"), tn("ui.review.fixed_count", 2));
    check("the singular text names the number one", tn("ui.review.fixed_count", 1).includes("1"), tn("ui.review.fixed_count", 1));
    check("Read 1 document", tn("ui.import.read", 1) === t("ui.import.read_one"));
    check("Read 3 documents", tn("ui.import.read", 3) === t("ui.import.read", { n: 3 }));
    check("Export 1 document", tn("ui.export.go", 1) === t("ui.export.go_one"));
    check("a count of zero uses the plural", tn("ui.review.fixed_count", 0) === t("ui.review.fixed_count", { n: 0 }));
  };

  sections.spanish = async () => {
    for (const [tab, doc] of [["review", 2], ["review", 0], ["import", 0]]) {
      const frame = await frameOf(1280, 800, `tab=${tab}&doc=${doc}&add=${encodeURIComponent(params.get("pdf"))}`);
      const d = frame.contentDocument;
      const side = d.querySelector(tab === "review" ? "#left" : ".side");
      check(`${tab} doc ${doc}: the side panel has no horizontal scroll`, side.scrollWidth <= side.clientWidth, `${side.scrollWidth} > ${side.clientWidth}`);
      const edge = side.getBoundingClientRect().right;
      const outside = [...side.querySelectorAll(".badge")].filter((b) => b.getBoundingClientRect().right > edge + 1);
      check(`${tab} doc ${doc}: every badge is inside the panel`, outside.length === 0, outside.map((b) => b.textContent).join(","));
      frame.remove();
    }
  };

  sections.estimate = async () => {
    S.tab = "import";
    render();
    S.files = [{ path: "/x.pdf", name: "x.pdf", picked: true, pages: 100, scanned_pages: 100, digital_pages: 0, kind: "scanned", size: 1 }];
    S.opts.second = "off";
    render();
    const [low, high] = estimateRange(S.files);
    check("the estimate has a short and a long figure", low > 0 && low < high, `${low} ${high}`);
    check("and the page shows both", $("#estimate").textContent === estimateLine(S.files) && $("#estimate").textContent.includes(t("ui.import.range", { from: fmtTime(low), to: fmtTime(high) })), $("#estimate").textContent);
    S.files[0].pages = 1;
    S.files[0].scanned_pages = 1;
    render();
    check("one page is said in the singular", $("#estimate").textContent === estimateLine(S.files) && !$("#estimate").textContent.includes("1 pages"), $("#estimate").textContent);
  };

  // The estimate follows how fast this computer has read: the figures scale, and the line says "this PC" only then.
  sections.speedestimate = async () => {
    S.tab = "import";
    S.files = [{ path: "/x.pdf", name: "x.pdf", picked: true, pages: 100, scanned_pages: 100, digital_pages: 0, kind: "scanned", size: 1 }];
    S.opts.second = "off";
    S.defaults.speed_factor = 1;
    S.defaults.speed_measured = false;
    render();
    const [low1, high1] = estimateRange(S.files);
    check("before any read the line is for a typical laptop", $("#estimate").textContent === tn("ui.import.estimate_typical", 100, { pages: 100, time: t("ui.import.range", { from: fmtTime(low1), to: fmtTime(high1) }) }) && !$("#estimate").textContent.includes("on this PC"), $("#estimate").textContent);
    S.defaults.speed_factor = 2;
    S.defaults.speed_measured = true;
    render();
    const [low2, high2] = estimateRange(S.files);
    check("a computer twice as slow doubles both figures", Math.abs(low2 - 2 * low1) < 1e-9 && Math.abs(high2 - 2 * high1) < 1e-9, `${low1}-${high1} ${low2}-${high2}`);
    check("and the line says it is for this PC", $("#estimate").textContent.includes("on this PC"), $("#estimate").textContent);
    S.opts.second = "thorough";
    render();
    const [, high3] = estimateRange(S.files);
    check("the second reading is scaled too", high3 - high2 > 0 && Math.abs((high3 - high2) - 2 * 100 * ((S.defaults.second_reading_thorough_cost || {}).seconds_per_page || 0)) < 1e-6, `${high2} ${high3}`);
  };

  // Reading again a document the user has changed asks first.
  sections.replaceedits = async () => {
    const realApi = api;
    const realConfirm = window.confirm;
    const posted = [];
    const asked = [];
    api = async (path, body) => {
      posted.push([path, body]);
      if (path === "/api/read" && !body.replace) {
        const error = new Error("edits");
        error.code = "error.edits_would_be_lost";
        error.params = { names: ["a.pdf", "b.pdf"] };
        throw error;
      }
      return {};
    };
    try {
      S.files = [{ path: "/x/a.pdf", name: "a.pdf", picked: true, pages: 1, scanned_pages: 1, digital_pages: 0, kind: "scanned", size: 1 }];
      window.confirm = (text) => { asked.push(text); return true; };
      await startRead();
      check("it asks, naming the files", asked.length === 1 && asked[0] === t("ui.read.confirm_replace", { names: "a.pdf, b.pdf" }), JSON.stringify(asked));
      check("and reads again with replace only after the yes", posted.length === 2 && !posted[0][1].replace && posted[1][1].replace === true, JSON.stringify(posted.map((p) => p[1].replace)));
      posted.length = 0;
      S.starting = false;
      window.confirm = () => false;
      await startRead();
      check("after a no nothing is read", posted.length === 1 && !posted[0][1].replace && S.starting === false);
    } finally {
      api = realApi;
      window.confirm = realConfirm;
    }
  };

  // A document changed in another tab shows here as soon as this tab gets the focus back.
  sections.refocus = async () => {
    const id = await setupReview();
    const other = S.docs[id].mode === "more" ? "safe" : "more";
    const before = S.docs[id].mode;
    await api("/api/mode", { doc: id, mode: other }); // what a second tab does
    check("this tab still shows the old mode: it was not told", S.docs[id].mode === before, S.docs[id].mode);
    window.dispatchEvent(new Event("focus"));
    await S.pending;
    check("with the focus back it shows the mode the server holds", S.docs[id].mode === other, S.docs[id].mode);
    check("and the buttons say so", document.querySelector(`[data-mode="${other}"]`).getAttribute("aria-pressed") === "true");
    const view = document.getElementById("root");
    window.dispatchEvent(new Event("focus"));
    await S.pending;
    check("nothing changed: nothing is redrawn", document.getElementById("root") === view && S.docs[id].mode === other);
  };

  // A computer that writes its paths with backslashes and a drive letter gets the same folder buttons and names.
  sections.winpaths = async () => {
    const real = S.defaults.root;
    try {
      S.defaults.root = "C:\\Users\\me";
      check("a Windows path is split in its folders", JSON.stringify(pathParts("C:\\Users\\me\\Documents\\Invoices 2026").parts) === JSON.stringify(["C:", "Users", "me", "Documents", "Invoices 2026"]), JSON.stringify(pathParts("C:\\Users\\me\\Documents").parts));
      const { upTo } = pathParts("C:\\Users\\me\\Documents");
      check("and the path up to each folder is a Windows path", upTo(1) === "C:\\" && upTo(2) === "C:\\Users" && upTo(4) === "C:\\Users\\me\\Documents", [upTo(1), upTo(2), upTo(4)].join(" | "));
      check("a path with slashes is as before", JSON.stringify(pathParts("/srv/me/docs").parts) === JSON.stringify(["srv", "me", "docs"]) && pathParts("/srv/me/docs").upTo(2) === "/srv/me");
      check("a network path keeps its two backslashes", pathParts("\\\\srv\\share\\x").upTo(2) === "\\\\srv\\share", pathParts("\\\\srv\\share\\x").upTo(2));
      S.tab = "import";
      S.browse = { path: "C:\\Users\\me\\Documents\\Invoices", parent: "C:\\Users\\me\\Documents", dirs: [], pdfs: [] };
      render();
      const buttons = [...document.querySelectorAll("#crumbs .crumb")];
      check("the folder buttons name the last three folders", buttons.map((b) => b.textContent).join("|") === "me|Documents|Invoices", buttons.map((b) => b.textContent).join("|"));
      check("the ones above the home folder are not offered", buttons[0].disabled === false && !buttons[1].disabled);
      check("the name of the folder in the title is the last one", projectName() === "Invoices" || projectName() === t("ui.project.none"), projectName());
      check("a file name is the last part of its path", fileName("C:\\Users\\me\\a b.pdf") === "a b.pdf" && fileName("/srv/me/a b.pdf") === "a b.pdf");
    } finally {
      S.defaults.root = real;
    }
  };

  sections.import = async () => {
    S.tab = "import";
    render();
    check("the screen says where the folders are", $("#where") && $("#where").textContent === t("ui.import.where"));
    const list = () => ($("#listing") ? $("#listing").getBoundingClientRect().top : null);
    const top = list();
    typeInto($("#path"), "/nonexistent-folder");
    key($("#path"), "Enter");
    check("a wrong path gives an error", await until(() => $(".banner")));
    check("the typed path stays in the box", $("#path").value === "/nonexistent-folder", $("#path").value);
    check("the error does not push the page down", list() === top, `${top} -> ${list()}`);
    check("the error floats over the page", getComputedStyle($(".banner")).position === "fixed");
    typeInto($("#path"), "/nonexistent-folder-2");
    check("typing again clears the error", !$(".banner"));
    key($("#path"), "Enter");
    await until(() => $(".banner"));
    $(".banner button").click();
    check("the error can be dismissed", !$(".banner"));
    await browse(params.get("deep"));
    const names = params.get("deep").split("/").filter(Boolean);
    check("a deep folder shows its last folders as buttons", $("#crumbs") && [...$("#crumbs").querySelectorAll(".crumb")].map((c) => c.textContent).join("/") === names.slice(-3).join("/"), $("#crumbs") && $("#crumbs").textContent);
    check("with the full path in the box and its tooltip", $("#path").value === params.get("deep") && $("#path").title === params.get("deep"));
    $("#crumbs").querySelector(".crumb").click();
    await until(() => S.browse.path.endsWith(`/${names[names.length - 3]}`));
    check("a button goes to that folder", S.browse.path.endsWith(`/${names[names.length - 3]}`), S.browse.path);
    // thorough mode without its models
    S.files = [{ path: "/x.pdf", name: "x.pdf", picked: true, pages: 2, scanned_pages: 2, digital_pages: 0, kind: "scanned", size: 1 }];
    S.opts.second = "thorough";
    S.defaults.thorough_available = false;
    render();
    check("Read cannot be pressed in Thorough mode when the models are missing", $("#read-button").disabled);
    S.defaults.thorough_available = true;
    render();
    check("it can when they are installed", !$("#read-button").disabled);
  };

  sections.reload = async () => {
    const pdf = params.get("pdf");
    const folder = params.get("deep");
    const frame = await frameOf(1280, 800, `tab=import&add=${encodeURIComponent(pdf)}`);
    await until(() => frame.contentWindow.eval("S.files.length") === 1);
    await frame.contentWindow.eval(`browse(${JSON.stringify(folder)})`);
    check("one file is in the list before the reload", frame.contentWindow.eval("S.files.length") === 1);
    frame.contentWindow.location.reload();
    await sleep(300);
    await until(() => frame.contentDocument.readyState === "complete" && frame.contentDocument.body && frame.contentDocument.body.dataset.ready === "1", 400);
    const w = frame.contentWindow;
    check("after a reload the list of documents is still there", w.eval("S.files.length") === 1, w.eval("S.files.length"));
    check("and says so", w.document.querySelector(".side .label").textContent === w.eval('t("ui.import.documents_count", { n: 1 })'), w.document.querySelector(".side .label").textContent);
    check("the folder is the one that was open", w.document.getElementById("path").value === folder, w.document.getElementById("path").value);
    check("the choices made are kept", w.eval("S.opts.languages.length") >= 1);
    frame.remove();
  };

  // The server is in the middle of a read (the test holds it) when this page opens.
  sections.reloadread = async () => {
    check("a page opened during a read shows the Read screen, not Import", await until(() => S.tab === "read"), S.tab);
    const tab = document.querySelector('[data-tab="read"]');
    check("the Read tab can be pressed", tab && !tab.disabled);
    check("the Stop button is there and can be pressed", $("#stop") && !$("#stop").disabled);
    $("#stop").click();
    check("Stop is accepted", $("#stop").disabled && $("#stop").textContent === t("ui.read.stopping"), $("#stop").textContent);
  };

  // The same read, seen by a page that stays on Import (a second tab that was open before the read started).
  sections.readelsewhere = async () => {
    check("the page knows a read is going on", await until(() => S.job && S.job.running));
    check("and its Read tab is enabled although it is on Import", S.tab === "import" && !document.querySelector('[data-tab="read"]').disabled);
    await addFile(params.get("pdf"));
    S.job.running = false; // the page had not seen the read when the button was pressed
    await startRead();
    check("pressing Read while another read runs opens the Read screen", S.tab === "read", S.tab);
    check("and says nothing is wrong", !S.error && !$(".banner"), S.error);
  };

  sections.doubleclick = async () => {
    await addFile(params.get("pdf"));
    const button = $("#read-button");
    check("the Read button can be pressed", button && !button.disabled);
    button.click();
    button.click();
    check("the Read screen opens", await until(() => S.tab === "read"), S.tab);
    await sleep(300);
    check("a double click is one read, with no error", !S.error && !$(".banner") && S.job.errors.length === 0, S.error);
  };

  // Tab is the key of "use the suggestion" in Review, and it must not trap the keyboard.
  sections.tabkey = async () => {
    const tab = (shift = false) => {
      const ev = new KeyboardEvent("keydown", { key: "Tab", shiftKey: shift, bubbles: true, cancelable: true });
      document.dispatchEvent(ev);
      return ev.defaultPrevented;
    };
    await setupReview();
    check("with no word selected Tab moves the focus", !tab());
    check("and so does Shift+Tab", !tab(true));
    let withSuggestion = null;
    let without = null;
    for (const id of S.order) {
      for (const item of S.docs[id].doubtful) {
        await select(id, item.id);
        if (suggestionOf(wordOf(item.id))) withSuggestion ||= [id, item.id];
        else without ||= [id, item.id];
      }
    }
    check("the fixtures have words with and without a suggestion", withSuggestion && without);
    await select(...without);
    check("a selected word with no suggestion leaves Tab to the browser", !tab());
    await select(...withSuggestion);
    check("Shift+Tab is always left to the browser", !tab(true));
    const button = $("#accept");
    button.focus();
    const ev = new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true });
    button.dispatchEvent(ev);
    check("Tab pressed on a button moves the focus on", !ev.defaultPrevented);
    document.body.focus();
    const word = S.cur.word;
    check("Tab on a word with a suggestion still takes it", tab());
    await S.pending;
    check("and the word is corrected", wordOf(word) && wordOf(word).review === "corrected", wordOf(word) && wordOf(word).review);
    await press("Escape");
    check("Esc leaves nothing selected and Tab free again", !tab());
  };

  // A document read again (other options) is a new reading: the pages cached from the old one must go.
  sections.reread = async () => {
    const id = await setupReview();
    await pageData(id, 1);
    const old = S.pages[`${id}:1`];
    check("the page is cached after it was shown", !!old);
    S.tab = "read";
    handleEvent({ event: "job.started", names: ["a.pdf"], pages: 1, time: 1 });
    handleEvent({ event: "doc.begin", index: 0, pages: 1, name: "a.pdf", time: 1 });
    handleEvent({ event: "doc.read", index: 0, doc_id: id, name: "a.pdf", status: "done", doubtful: 0, fixed: 0, numbers_checked: 0, unreadable: [], time: 2 });
    check("reading the document again drops its cached pages (the Read screen may load the page afresh)", S.pages[`${id}:1`] !== old);
  };

  // A page whose fixes were withheld because of its language has no share of changed words: its text must not show one.
  sections.withheld = async () => {
    const id = await setupReview();
    const meta = S.docs[id];
    meta.fixes.withheld = [{ page: 1, cause: "language" }];
    S.cur = { doc: id, page: 1, word: null };
    render();
    await queue(showPage);
    const language = $("#withheld");
    check("the Review panel says the language is the reason", language && language.textContent === t("guard.warning.fixes_withheld_language", { page: 1 }), language && language.textContent);
    check("with no unresolved placeholder and no NaN", language && !/[{}]|NaN/.test(language.textContent), language && language.textContent);
    meta.fixes.withheld = [{ page: 1, share: 0.25 }];
    render();
    await queue(showPage);
    check("a page with too many changes still says the share", $("#withheld") && $("#withheld").textContent.includes("25 %"), $("#withheld") && $("#withheld").textContent);
    S.tab = "read";
    handleEvent({ event: "job.started", names: ["a.pdf"], pages: 1, time: 1 });
    handleEvent({ event: "job.warning", code: "guard.warning.fixes_withheld", params: { page: 1, cause: "language" }, index: 0, time: 2 });
    handleEvent({ event: "job.warning", code: "guard.warning.fixes_withheld", params: { page: 2, share: 0.5 }, index: 0, time: 2 });
    const notes = [...document.querySelectorAll(".warn")].map((el) => el.textContent);
    check("the Read screen shows both causes in words", notes.length === 2 && !notes.some((n) => /[{}]|NaN/.test(n)), JSON.stringify(notes));
    check("the language one is the language text", notes[0] === t("guard.warning.fixes_withheld_language", { page: 1 }), notes[0]);
  };

  // A warning of the Read screen names its document when there are several; the notes of a page show in the Review panel.
  sections.pagenotes = async () => {
    S.tab = "read";
    handleEvent({ event: "job.started", names: ["a.pdf", "b.pdf"], pages: 2, time: 1 });
    handleEvent({ event: "job.warning", code: "page.warning.low_resolution", params: { page: 1, dpi: 70 }, index: 1, time: 2 });
    handleEvent({ event: "job.warning", code: "job.warning.same_content", params: { name: "c.pdf", same_as: "a.pdf" }, index: 0, time: 2 });
    const notes = [...document.querySelectorAll(".warn")].map((el) => el.textContent);
    check("a warning names the document it is about", notes[0] === "b.pdf: " + t("page.warning.low_resolution", { page: 1, dpi: 70 }), JSON.stringify(notes));
    check("the resolution is in the words", notes[0].includes("70 dpi"), notes[0]);
    check("the same-content warning keeps its own two names", notes[1] === t("job.warning.same_content", { name: "c.pdf", same_as: "a.pdf" }), JSON.stringify(notes));
    const id = S.order.find((i) => S.docs[i].pages.length >= 2); // notes of page 2 need a document that has one
    S.tab = "review";
    S.cur = { doc: id, page: 1, word: null };
    render();
    await queue(showPage);
    S.docs[id].notes = [{ key: "page.warning.sideways", page: 1 }, { key: "page.warning.low_resolution", page: 1, dpi: 70 }, { key: "page.warning.sideways", page: 2 }];
    S.tab = "review";
    S.cur = { doc: id, page: 1, word: null };
    render();
    await queue(showPage);
    const sideways = $("#note-sideways");
    const low = $("#note-low_resolution");
    check("the page says its lines run up or down", sideways && sideways.textContent === t("page.warning.sideways", { page: 1 }), sideways && sideways.textContent);
    check("the page says its real resolution", low && low.textContent === t("page.warning.low_resolution", { page: 1, dpi: 70 }), low && low.textContent);
    check("with no unresolved placeholder", !/[{}]|NaN/.test(sideways.textContent + low.textContent));
    S.cur = { doc: id, page: 2, word: null };
    render();
    await queue(showPage);
    check("another page has only its own note", $("#note-sideways") && !$("#note-low_resolution") && $("#note-sideways").textContent === t("page.warning.sideways", { page: 2 }));
  };

  // A note that holds for every page of a document is one box, and it does not push the progress off the screen.
  sections.warninggroup = async () => {
    S.tab = "read";
    handleEvent({ event: "job.started", names: ["a.pdf"], pages: 40, time: 1 });
    for (let page = 40; page >= 1; page--) {
      handleEvent({ event: "job.warning", code: "guard.warning.fixes_withheld", params: { page, cause: "language" }, index: 0, time: 2 });
    }
    handleEvent({ event: "job.warning", code: "page.warning.low_resolution", params: { page: 3, dpi: 70 }, index: 0, time: 2 });
    handleEvent({ event: "job.warning", code: "page.warning.low_resolution", params: { page: 3, dpi: 70 }, index: 0, time: 2 });
    const boxes = [...document.querySelectorAll("#warnings .warn")];
    check("40 pages with the same note and one more are two boxes", boxes.length === 2, boxes.length);
    check("the first names the lowest page", boxes[0].textContent.startsWith(t("guard.warning.fixes_withheld_language", { page: 1 })), boxes[0].textContent);
    const more = boxes[0].querySelector("[data-more]");
    check("it counts the other pages", more && more.dataset.more === "39" && more.textContent.includes("39"), more && more.textContent);
    check("it lists a few and not all of them", more && more.textContent.includes("2, 3") && more.textContent.includes("...") && !more.textContent.includes("40"), more && more.textContent);
    check("a note of one page has no second line", !boxes[1].querySelector("[data-more]"), boxes[1].textContent);
    check("the progress stays in view", $("#total-bar").getBoundingClientRect().top < 450, $("#total-bar").getBoundingClientRect().top);
  };

  // The folders of the Import list and the rows of the Fields table are for the keyboard too.
  sections.keyboardlist = async () => {
    S.tab = "import";
    await browse(params.get("path"));
    const folders = [...document.querySelectorAll("#listing .item.folder")];
    check("the list has folders", folders.length >= 2, folders.length);
    check("every folder can take the focus and says it is a button", folders.every((f) => f.tabIndex === 0 && f.getAttribute("role") === "button"));
    const sub = folders.find((f) => f.textContent.trim() !== "..");
    sub.focus();
    check("a folder really has the focus", document.activeElement === sub);
    const wanted = `${S.browse.path}/${sub.textContent.trim().replace(/\/$/, "")}`;
    sub.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
    check("Enter opens the folder", await until(() => S.browse.path === wanted), S.browse.path);
    const up = document.querySelector("#listing .item.folder");
    up.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true, cancelable: true }));
    check("Space on .. goes back up", await until(() => S.browse.path === params.get("path")), S.browse.path);

    const invoiceId = S.order.find((id) => S.docs[id].invoice);
    S.tab = "review";
    S.cur = { doc: invoiceId, page: 1, word: null };
    render();
    await queue(showPage);
    const first = S.docs[invoiceId].doubtful[0].id;
    await select(invoiceId, first);
    const before = wordOf(first).review;
    S.reviewTab = "fields";
    render();
    const rows = [...document.querySelector("table.fields").querySelectorAll("tr.pick")];
    check("the rows of the Fields table take the focus", rows.length >= 1 && rows.every((r) => r.tabIndex === 0), `${rows.length} rows, ${rows.filter((r) => r.tabIndex === 0).length} focusable`);
    const cells = [...document.querySelectorAll("table.fields td[tabindex]")];
    check("so do the cells of the invoice lines", cells.length === 0 || cells.every((c) => c.getAttribute("role") === "button"));
    const target = rows.find((r) => r.querySelector(".badge"));
    target.focus();
    target.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true }));
    await until(() => S.reviewTab === "page");
    await S.pending;
    check("Enter on a row shows its word on the page", S.reviewTab === "page" && S.cur.word, S.reviewTab);
    check("and does not accept the word that was selected before", wordOf(first).review === before, wordOf(first).review);
  };

  // ---- 5.0.6: a screen never shows the name, page, fields or notes of one document next to the data of another ----

  // The events of a read that is over, as a page that opens late (a reload, a second window) receives them all at once.
  // `fail` is the index of a document whose read failed; `noBegin` says it failed before its reading began.
  function feedRead(ids, { fail = -1, noBegin = false, names } = {}) {
    S.tab = "read";
    names = names || ids.map((id) => S.docs[id].name);
    handleEvent({ event: "job.started", names, pages: ids.length, time: 1 });
    ids.forEach((id, i) => {
      if (i === fail) {
        if (!noBegin) handleEvent({ event: "doc.begin", index: i, pages: 1, name: names[i], time: 2 + i });
        handleEvent({ event: "job.error", code: "error.file_not_readable", params: { file: names[i] }, path: `/x/${names[i]}`, index: i, time: 2 + i });
        return;
      }
      handleEvent({ event: "doc.begin", index: i, pages: 1, name: names[i], time: 2 + i });
      handleEvent({ event: "page.read", index: i, page: 1, status: "ocr", doubtful: 0, fixed: 0, time: 2 + i });
      handleEvent({ event: "doc.read", index: i, doc_id: id, name: S.docs[id].name, status: "done", doubtful: 0, fixed: 0, numbers_checked: 0, unreadable: [], time: 2 + i });
    });
    handleEvent({ event: "job.finished", status: "done", time: 10 });
  }

  async function openReview(doc, page = 1) {
    S.tab = "review";
    S.cur = { doc, page, word: null };
    render();
    await queue(showPage);
  }

  // The pages of the first documents arrive in the reverse order of the events, as they do when a page opened late gets
  // three "document read" events at once: the page drawn is the last document's, and the screen says whose it is.
  sections.readpreview = async () => {
    const ids = S.order.slice(0, 3);
    const real = pageData;
    const delay = { [ids[0]]: 700, [ids[1]]: 350 };
    pageData = async (doc, number) => {
      await sleep(delay[doc] || 0);
      return real(doc, number);
    };
    feedRead(ids);
    await sleep(1500);
    pageData = real;
    const last = ids[2];
    const mine = await real(last, 1);
    const canvas = $("#read-canvas");
    check("the page drawn is the last document's, not the one whose page arrived last", !!readView && readView.words === mine.words, readView && readView.words.length);
    check("the canvas says whose page it is", canvas && canvas.dataset.doc === last && canvas.dataset.page === "1", canvas && canvas.dataset.doc);
    check("a caption names that document and the page", $("#read-preview") && $("#read-preview").textContent === t("ui.read.preview", { name: S.docs[last].name, page: 1 }), $("#read-preview") && $("#read-preview").textContent);
    check("the name above the progress is that document's", $("#read-doc") && $("#read-doc").textContent === S.docs[last].name, $("#read-doc") && $("#read-doc").textContent);
  };

  // The last document of the read could not be read: the name, the tiles and the page are those of the one before it.
  sections.readfailed = async () => {
    const ids = S.order.slice(0, 3);
    feedRead(ids, { fail: 2 });
    await sleep(800);
    const cards = [...document.querySelectorAll(".side .card")];
    check("the document that failed does not say Reading", cards[2] && cards[2].dataset.doc === "failed", cards[2] && cards[2].dataset.doc);
    check("the page drawn is the last document that was read", $("#read-canvas") && $("#read-canvas").dataset.doc === ids[1], $("#read-canvas") && $("#read-canvas").dataset.doc);
    check("the name above the progress is the name of that document", $("#read-doc") && $("#read-doc").textContent === S.docs[ids[1]].name, $("#read-doc") && $("#read-doc").textContent);
    check("and so is the card that is marked", document.querySelector(".side .card.selected .name").textContent === S.docs[ids[1]].name, document.querySelector(".side .card.selected .name").textContent);
    check("the tiles are those of that document", document.querySelectorAll(".tiles .tile").length === S.docs[ids[1]].pages.length, document.querySelectorAll(".tiles .tile").length);
    S.job = null;
    feedRead(ids, { fail: 2, noBegin: true });
    await sleep(300);
    const again = [...document.querySelectorAll(".side .card")];
    check("a file that could not even be opened is not left Waiting", again[2] && again[2].dataset.doc === "failed", again[2] && again[2].dataset.doc);
  };

  // A file with the content of one that is open opens that one: its card says so.
  sections.readsame = async () => {
    const ids = S.order.slice(0, 2);
    S.tab = "read";
    handleEvent({ event: "job.started", names: ["copy-of-first.pdf"], pages: 1, time: 1 });
    handleEvent({ event: "doc.begin", index: 0, pages: 1, name: "copy-of-first.pdf", time: 2 });
    handleEvent({ event: "doc.read", index: 0, doc_id: ids[0], name: S.docs[ids[0]].name, status: "done", doubtful: 0, fixed: 0, numbers_checked: 0, unreadable: [], time: 3 });
    handleEvent({ event: "job.finished", status: "done", time: 4 });
    await sleep(500);
    const note = document.querySelector("[data-same-as]");
    check("the card of the copy says whose content it has", note && note.dataset.sameAs === S.docs[ids[0]].name && note.textContent === t("ui.read.same_as", { name: S.docs[ids[0]].name }), note && note.textContent);
    check("the name above the page is the document that opens", $("#read-doc") && $("#read-doc").textContent === S.docs[ids[0]].name, $("#read-doc") && $("#read-doc").textContent);
  };

  // From a document on its page 3 to one that has no page 3.
  sections.reviewswitch = async () => {
    const [a, , c] = S.order;
    await openReview(c);
    document.querySelector('[data-page="3"]').click();
    await S.pending;
    check("page 3 of the three-page document is shown", S.cur.page === 3 && S.current.page === 3 && S.current.doc === c, `${S.cur.page} ${S.current.page}`);
    document.querySelector(`[data-doc="${a}"]`).click();
    await S.pending;
    check("choosing a one-page document does not ask for its page 3", !S.error && !$(".banner"), S.error);
    check("it starts at its first page", S.cur.doc === a && S.cur.page === 1, `${S.cur.doc} ${S.cur.page}`);
    check("the page on the canvas is that document's", S.current.doc === a && S.current.page === 1 && reviewView.words === S.current.words, S.current.doc);
    check("the page buttons are that document's", document.querySelectorAll("[data-page]").length === S.docs[a].pages.length, document.querySelectorAll("[data-page]").length);
  };

  // Open in Review from the Read screen while the page of the other document is not there yet (or never comes).
  sections.openfromread = async () => {
    const [a, b, c] = S.order;
    await openReview(c);
    document.querySelector('[data-page="3"]').click();
    await S.pending;
    feedRead([a, b]);
    await sleep(400);
    dropPages(a);
    dropPages(b);
    const real = pageData;
    pageData = async () => {
      throw Object.assign(new Error("image"), { code: "error.image_failed", params: {} });
    };
    const open = [...document.querySelectorAll(".side .card button")][1];
    open.click();
    check("Review opens on the document that was chosen", S.tab === "review" && S.cur.doc === b && S.cur.page === 1, `${S.tab} ${S.cur.doc} ${S.cur.page}`);
    check("the canvas does not draw the page of the document that was open before", reviewView.words.length === 0, reviewView.words.length);
    await S.pending;
    await sleep(100);
    check("the page that could not be loaded is reported", !!S.error, S.error);
    check("and the right panel does not show a word of the other document", !document.getElementById("reading"), document.getElementById("right").innerText.slice(0, 80));
    pageData = real;
    S.error = null;
    await queue(showPage);
    check("asked again, the page loads: a failure is not remembered", !S.error && S.current && S.current.doc === b && reviewView.words === S.current.words, S.error);
  };

  // The document on screen is read again (here, or by a second window): Review shows the new reading.
  sections.rereadreview = async () => {
    const a = S.order[0];
    await openReview(a);
    const w = S.docs[a].doubtful[0].id;
    await queue(() => select(a, w));
    check("the word is open before the other window accepts it", wordOf(w) && !wordOf(w).review, wordOf(w) && wordOf(w).review);
    await api("/api/accept", { doc: a, word: w }); // what the other window did
    handleEvent({ event: "job.started", names: ["x.pdf"], pages: 1, time: 1 });
    handleEvent({ event: "doc.begin", index: 0, pages: 1, name: "x.pdf", time: 2 });
    handleEvent({ event: "doc.read", index: 0, doc_id: a, name: S.docs[a].name, status: "done", doubtful: 0, fixed: 0, numbers_checked: 0, unreadable: [], time: 3 });
    await sleep(700);
    await S.pending;
    check("the page in Review is the one the server holds now", wordOf(w) && wordOf(w).review === "accepted", wordOf(w) && wordOf(w).review);
    check("the list of words to check does not hold the accepted one", !S.docs[a].doubtful.some((d) => d.id === w));
    // a document read while Review is open appears in its list
    const extra = S.order[2];
    const kept = S.docs[extra];
    S.order = S.order.slice(0, 2);
    delete S.docs[extra];
    renderLeft();
    check("one document less in the list", document.querySelectorAll("#left [data-doc]").length === 2);
    handleEvent({ event: "doc.read", index: 0, doc_id: extra, name: kept.name, status: "done", doubtful: 0, fixed: 0, numbers_checked: 0, unreadable: [], time: 4 });
    await sleep(500);
    check("a document read while Review is open is listed there", document.querySelectorAll("#left [data-doc]").length === 3 && !!document.querySelector(`#left [data-doc="${extra}"]`), document.querySelectorAll("#left [data-doc]").length);
  };

  // The filter of fixed words belongs to the document it was set on.
  sections.fixedfilter = async () => {
    const withFixes = S.order.find((id) => S.docs[id].fixed.length > 0);
    const without = S.order.find((id) => !S.docs[id].fixed.length);
    check("a document with fixed words and one without are open", withFixes && without);
    await openReview(withFixes);
    $("#fixed-filter").click();
    check("the filter is on", S.fixedOnly && reviewView.fixedOnly);
    document.querySelector(`[data-doc="${without}"]`).click();
    await S.pending;
    check("another document with nothing fixed is not drawn through the filter", !S.fixedOnly && !reviewView.fixedOnly && reviewView.visibleCount() > 0, `${S.fixedOnly} ${reviewView.visibleCount()}`);
    check("the button says the filter is off and cannot be pressed", $("#fixed-filter").getAttribute("aria-pressed") !== "true" && $("#fixed-filter").disabled);
    // a rule picked in one document is not carried to another
    S.ruleFocus = "checks.rule.total";
    document.querySelector(`[data-doc="${withFixes}"]`).click();
    await S.pending;
    check("a rule picked in a document is dropped when another is chosen", S.ruleFocus === null, S.ruleFocus);
  };

  // A document read after the Export screen was first opened is ticked there like the others.
  sections.exportnew = async () => {
    const c = S.order[2];
    const meta = S.docs[c];
    S.order = S.order.slice(0, 2);
    delete S.docs[c];
    S.exportState.picked = null;
    S.tab = "export";
    render();
    check("the two documents are ticked", S.exportState.picked.size === 2 && $("#export").textContent === tn("ui.export.go", 2), $("#export").textContent);
    addDoc(meta);
    render();
    check("a document read later is ticked too", S.exportState.picked.has(c) && $("#export").textContent === tn("ui.export.go", 3), $("#export").textContent);
  };

  // A document read while the Export screen is open is listed there.
  sections.exportevent = async () => {
    const c = S.order[2];
    const meta = S.docs[c];
    S.order = S.order.slice(0, 2);
    delete S.docs[c];
    S.exportState.picked = null;
    S.tab = "export";
    render();
    check("the Export screen lists two documents", document.querySelectorAll(".side .card").length === 2, document.querySelectorAll(".side .card").length);
    handleEvent({ event: "job.started", names: [meta.name], pages: 1, time: 1 });
    handleEvent({ event: "doc.begin", index: 0, pages: 1, name: meta.name, time: 2 });
    handleEvent({ event: "doc.read", index: 0, doc_id: c, name: meta.name, status: "done", doubtful: 0, fixed: 0, numbers_checked: 0, unreadable: [], time: 3 });
    await sleep(500);
    check("after a document is read the screen lists three", document.querySelectorAll(".side .card").length === 3, document.querySelectorAll(".side .card").length);
    check("and the new one is ticked", $("#export").textContent === tn("ui.export.go", 3), $("#export").textContent);
  };

  async function run() {
    await until(() => document.body.dataset.ready === "1", 400);
    try {
      if (!sections[section]) check("a known section was asked for", false, section);
      else await sections[section]();
    } catch (e) {
      check("the self-test ran to the end", false, `${e && e.message} ${e && e.stack}`);
    }
    const pass = checks.every((c) => c.ok);
    const out = document.createElement("pre");
    out.id = "selftest-result";
    out.dataset.pass = pass ? "1" : "0";
    out.textContent = JSON.stringify({ pass, checks }, null, 1);
    document.body.append(out);
  }

  window.UI_SECTIONS = sections;
  run();
})();
