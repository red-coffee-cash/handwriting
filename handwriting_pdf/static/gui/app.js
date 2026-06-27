// Worksheet layout editor frontend. No build step, no frameworks.
// All coordinates are exchanged with the backend in PDF points
// (PyMuPDF convention: origin top-left, y down). The canvas is drawn
// at `scale` px per PDF point, so canvas<->pdf conversion is a single
// multiply/divide -- no axis flip needed.

const state = {
  session: null,
  pageNum: 0,
  activeQid: null,
  tool: "select",
  scale: 1,
  pageSize: { width: 1, height: 1 },
  pageImage: null,
  history: {}, // qid -> { stack: [strokesArray, ...], pointer: int }
  drag: null, // in-progress pointer interaction
};

const canvas = document.getElementById("stage");
const ctx = canvas.getContext("2d");

function $(id) { return document.getElementById(id); }

async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(url, opts);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `${method} ${url} failed (${resp.status})`);
  return data;
}

function setStatus(msg) { $("status-msg").textContent = msg || ""; }

// ---------- session / page loading ----------

function showUpload() {
  document.getElementById("upload-screen").style.display = "flex";
  document.getElementById("app").style.display = "none";
}

function showApp() {
  document.getElementById("upload-screen").style.display = "none";
  document.getElementById("app").style.display = "flex";
}

async function loadSession() {
  const resp = await fetch("/api/session");
  if (resp.status === 404) {
    showUpload();
    return;
  }
  state.session = await resp.json();
  showApp();
  if (!state.activeQid) {
    const first = state.session.questions.find(q => !q.deleted);
    if (first) { state.activeQid = first.id; state.pageNum = first.box.page; }
  }
  renderQuestionList();
  await loadPage(state.pageNum);
}

async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("pdf", file);
  const uploadStatus = $("upload-status");
  uploadStatus.textContent = "Extracting questions...";
  try {
    const resp = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error(data.error || "Upload failed.");
    state.activeQid = null;
    state.history = {};
    await loadSession();
  } catch (e) {
    uploadStatus.textContent = e.message;
  }
}

function pageCount() { return state.session.pages.length; }

async function loadPage(n) {
  state.pageNum = Math.max(0, Math.min(n, pageCount() - 1));
  const size = await api("GET", `/api/page/${state.pageNum}/size`);
  state.pageSize = size;

  const img = new Image();
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
    img.src = `/api/page/${state.pageNum}.png?ts=${Date.now()}`;
  });
  state.pageImage = img;

  const displayWidth = Math.min(1000, Math.max(600, size.width));
  state.scale = displayWidth / size.width;
  canvas.width = Math.round(size.width * state.scale);
  canvas.height = Math.round(size.height * state.scale);

  $("page-label").textContent = `Page ${state.pageNum + 1} / ${pageCount()}`;
  renderQuestionList();
  draw();
}

// ---------- coordinate conversion ----------

function canvasToPdf(x, y) { return [x / state.scale, y / state.scale]; }
function pdfToCanvas(x, y) { return [x * state.scale, y * state.scale]; }

function eventToPdf(evt) {
  const rect = canvas.getBoundingClientRect();
  const cx = evt.clientX - rect.left;
  const cy = evt.clientY - rect.top;
  return canvasToPdf(cx, cy);
}

// ---------- question helpers ----------

function getActiveQuestion() {
  if (!state.activeQid) return null;
  return state.session.questions.find(q => q.id === state.activeQid) || null;
}

function questionsOnPage(n) {
  return state.session.questions.filter(q => q.box.page === n);
}

function ensureHistory(qid) {
  if (!state.history[qid]) {
    const q = state.session.questions.find(qq => qq.id === qid);
    state.history[qid] = { stack: [cloneStrokes(q.strokes || [])], pointer: 0 };
  }
  return state.history[qid];
}

function cloneStrokes(strokes) { return JSON.parse(JSON.stringify(strokes || [])); }

function pushHistory(qid, strokes) {
  const h = ensureHistory(qid);
  h.stack = h.stack.slice(0, h.pointer + 1);
  h.stack.push(cloneStrokes(strokes));
  h.pointer = h.stack.length - 1;
}

async function commitStrokes(qid, strokes, record) {
  const q = state.session.questions.find(qq => qq.id === qid);
  q.strokes = strokes;
  if (record) pushHistory(qid, strokes);
  await api("PUT", `/api/session/questions/${qid}/strokes`, { strokes });
  draw();
}

async function undo() {
  const qid = state.activeQid;
  if (!qid) return;
  const h = ensureHistory(qid);
  if (h.pointer <= 0) { setStatus("Nothing to undo."); return; }
  h.pointer -= 1;
  const strokes = cloneStrokes(h.stack[h.pointer]);
  await commitStrokes(qid, strokes, false);
  setStatus("Undid last edit.");
}

async function redo() {
  const qid = state.activeQid;
  if (!qid) return;
  const h = ensureHistory(qid);
  if (h.pointer >= h.stack.length - 1) { setStatus("Nothing to redo."); return; }
  h.pointer += 1;
  const strokes = cloneStrokes(h.stack[h.pointer]);
  await commitStrokes(qid, strokes, false);
  setStatus("Redid edit.");
}

// ---------- drawing ----------

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.pageImage) ctx.drawImage(state.pageImage, 0, 0, canvas.width, canvas.height);

  for (const q of questionsOnPage(state.pageNum)) {
    if (q.deleted) continue;
    drawBox(q);
    drawStrokes(q.strokes || [], q.id === state.activeQid ? "#111" : "#444");
  }

  if (state.drag && (state.drag.type === "box-draw" || state.drag.type === "freeform-draw")) drawDragBox(state.drag);
  if (state.drag && state.drag.type === "pen") drawLivePoints(state.drag.canvasPts, "#111");
  if (state.drag && state.drag.type === "eraser") drawEraserCursor(state.drag);

  if (state.tool === "transform") {
    const aq = getActiveQuestion();
    if (aq && !aq.deleted && aq.strokes && aq.strokes.length) drawTransformBox(aq);
  }
}

function drawBox(q) {
  const [x0, y0] = pdfToCanvas(q.box.x0, q.box.y0);
  const [x1, y1] = pdfToCanvas(q.box.x1, q.box.y1);
  ctx.save();
  ctx.strokeStyle = q.id === state.activeQid ? "#3a6df0" : "#aaa";
  ctx.lineWidth = q.id === state.activeQid ? 2 : 1;
  ctx.setLineDash(q.id === state.activeQid ? [] : [4, 3]);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.restore();

  // Suppress the box's own resize handles while the transform tool is active,
  // so they don't overlap the green stroke-transform handles.
  if (q.id === state.activeQid && state.tool !== "transform") {
    for (const [hx, hy] of boxHandles(q.box)) {
      const [cx, cy] = pdfToCanvas(hx, hy);
      ctx.fillStyle = "#3a6df0";
      ctx.fillRect(cx - 4, cy - 4, 8, 8);
    }
  }
}

function boxHandles(box) {
  const mx = (box.x0 + box.x1) / 2, my = (box.y0 + box.y1) / 2;
  return [
    [box.x0, box.y0], [mx, box.y0], [box.x1, box.y0],
    [box.x0, my], [box.x1, my],
    [box.x0, box.y1], [mx, box.y1], [box.x1, box.y1],
  ];
}

const HANDLE_NAMES = ["nw", "n", "ne", "w", "e", "sw", "s", "se"];

function hitHandle(box, pdfX, pdfY) {
  const tolPdf = 6 / state.scale;
  const handles = boxHandles(box);
  for (let i = 0; i < handles.length; i++) {
    const [hx, hy] = handles[i];
    if (Math.abs(hx - pdfX) <= tolPdf && Math.abs(hy - pdfY) <= tolPdf) return HANDLE_NAMES[i];
  }
  return null;
}

function pointInBox(box, x, y) {
  return x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1;
}

// ---------- resize (transform) tool helpers ----------
// The transform tool moves/scales a question's already-rendered strokes
// directly, with no regenerate. It works on the bounding box of the strokes
// (not the answer box, which is left untouched).

function strokesBBox(strokes) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of strokes || []) {
    for (const [x, y] of s.points || []) {
      if (x < x0) x0 = x;
      if (y < y0) y0 = y;
      if (x > x1) x1 = x;
      if (y > y1) y1 = y;
    }
  }
  if (!isFinite(x0)) return null;
  return { x0, y0, x1, y1 };
}

// For a grabbed handle, return the moving handle point, the fixed anchor
// (opposite corner/edge), and which axis an edge-handle drag follows.
function handleAndAnchor(handle, b) {
  const mx = (b.x0 + b.x1) / 2, my = (b.y0 + b.y1) / 2;
  switch (handle) {
    case "nw": return { h: [b.x0, b.y0], a: [b.x1, b.y1], axis: "both" };
    case "ne": return { h: [b.x1, b.y0], a: [b.x0, b.y1], axis: "both" };
    case "sw": return { h: [b.x0, b.y1], a: [b.x1, b.y0], axis: "both" };
    case "se": return { h: [b.x1, b.y1], a: [b.x0, b.y0], axis: "both" };
    case "n": return { h: [mx, b.y0], a: [mx, b.y1], axis: "y" };
    case "s": return { h: [mx, b.y1], a: [mx, b.y0], axis: "y" };
    case "w": return { h: [b.x0, my], a: [b.x1, my], axis: "x" };
    case "e": return { h: [b.x1, my], a: [b.x0, my], axis: "x" };
  }
}

// Single uniform scale factor (aspect-preserving). Edge handles scale along
// their axis; corner handles use the anchor->pointer distance ratio.
function scaleFactor(info, px, py) {
  const [hx, hy] = info.h, [ax, ay] = info.a;
  let s;
  if (info.axis === "x") {
    s = (hx - ax) === 0 ? 1 : (px - ax) / (hx - ax);
  } else if (info.axis === "y") {
    s = (hy - ay) === 0 ? 1 : (py - ay) / (hy - ay);
  } else {
    const dh = Math.hypot(hx - ax, hy - ay);
    s = dh === 0 ? 1 : Math.hypot(px - ax, py - ay) / dh;
  }
  return Math.max(0.1, s);  // clamp to avoid collapsing or flipping
}

function translateStrokes(strokes, dx, dy) {
  return strokes.map(s => ({ ...s, points: s.points.map(([x, y]) => [x + dx, y + dy]) }));
}

function scaleStrokesAbout(strokes, anchor, s) {
  const [ax, ay] = anchor;
  return strokes.map(st => ({
    ...st,
    points: st.points.map(([x, y]) => [ax + (x - ax) * s, ay + (y - ay) * s]),
  }));
}

function drawTransformBox(q) {
  const b = strokesBBox(q.strokes);
  if (!b) return;
  const [x0, y0] = pdfToCanvas(b.x0, b.y0);
  const [x1, y1] = pdfToCanvas(b.x1, b.y1);
  ctx.save();
  ctx.strokeStyle = "#1f8a3d";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 3]);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.setLineDash([]);
  for (const [hx, hy] of boxHandles(b)) {
    const [cx, cy] = pdfToCanvas(hx, hy);
    ctx.fillStyle = "#1f8a3d";
    ctx.fillRect(cx - 4, cy - 4, 8, 8);
  }
  ctx.restore();
}

function drawStrokes(strokes, color) {
  for (const s of strokes) {
    if (!s.points || s.points.length < 2) continue;
    ctx.beginPath();
    ctx.strokeStyle = s.source === "user" ? "#111" : color;
    ctx.lineWidth = Math.max(1, (s.width_pt || 1.4) * state.scale);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    const [x0, y0] = pdfToCanvas(s.points[0][0], s.points[0][1]);
    ctx.moveTo(x0, y0);
    for (let i = 1; i < s.points.length; i++) {
      const [x, y] = pdfToCanvas(s.points[i][0], s.points[i][1]);
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function drawLivePoints(canvasPts, color) {
  if (canvasPts.length < 2) return;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = parseFloat($("pen-width").value) * state.scale;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.moveTo(canvasPts[0][0], canvasPts[0][1]);
  for (let i = 1; i < canvasPts.length; i++) ctx.lineTo(canvasPts[i][0], canvasPts[i][1]);
  ctx.stroke();
}

function drawDragBox(drag) {
  const [x0, y0] = pdfToCanvas(drag.x0, drag.y0);
  const [x1, y1] = pdfToCanvas(drag.x1, drag.y1);
  ctx.save();
  ctx.strokeStyle = "#3a6df0";
  ctx.lineWidth = 2;
  ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
  ctx.restore();
}

function drawEraserCursor(drag) {
  if (!drag.lastCanvas) return;
  const radiusPx = parseFloat($("eraser-size").value) * state.scale / 2;
  ctx.save();
  ctx.strokeStyle = "#d62828";
  ctx.setLineDash([3, 2]);
  ctx.beginPath();
  ctx.arc(drag.lastCanvas[0], drag.lastCanvas[1], radiusPx, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// ---------- sidebar ----------

function renderQuestionList() {
  const ul = $("question-list");
  ul.innerHTML = "";
  for (const q of state.session.questions) {
    const li = document.createElement("li");
    li.className = (q.id === state.activeQid ? "active " : "") + (q.deleted ? "deleted" : "");
    const textDiv = document.createElement("div");
    textDiv.className = "q-text";
    if (q.source === "manual") {
      const badge = document.createElement("span");
      badge.className = "badge-manual";
      badge.textContent = "Manual";
      textDiv.appendChild(badge);
      textDiv.appendChild(document.createTextNode(q.text || "(empty)"));
    } else {
      textDiv.textContent = q.text;
    }
    const metaDiv = document.createElement("div");
    metaDiv.className = "q-meta";
    metaDiv.textContent = `p.${q.box.page + 1} | ${q.answer ? "answered" : "no answer"}`;
    li.appendChild(textDiv);
    li.appendChild(metaDiv);
    li.addEventListener("click", () => selectQuestion(q.id));
    ul.appendChild(li);
  }
  $("question-list-empty").style.display = state.session.questions.length ? "none" : "flex";
  updateActionButtons();
}

function updateActionButtons() {
  const q = getActiveQuestion();
  $("edit-text-btn").style.display = q && !q.deleted ? "" : "none";
}

async function selectQuestion(qid) {
  state.activeQid = qid;
  const q = state.session.questions.find(qq => qq.id === qid);
  ensureHistory(qid);
  if (q.box.page !== state.pageNum) {
    await loadPage(q.box.page);
  } else {
    renderQuestionList();
    draw();
  }
}

// ---------- tool handling ----------

const TOOL_INFO = {
  select: { name: "Select", hint: "move or resize a question's answer box", cursor: "default" },
  box: { name: "Box", hint: "drag to draw an answer box", cursor: "crosshair" },
  transform: { name: "Resize", hint: "drag inside to move, or a handle to scale the handwriting", cursor: "move" },
  freeform: { name: "Freeform", hint: "drag a box, then type your own text", cursor: "crosshair" },
  pen: { name: "Pen", hint: "draw freehand strokes", cursor: "crosshair" },
  eraser: { name: "Eraser", hint: "drag over strokes to erase parts of them", cursor: "crosshair" },
};

function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll(".tool-btn").forEach(b => b.classList.toggle("active", b.dataset.tool === tool));
  const info = TOOL_INFO[tool] || { name: tool, hint: "", cursor: "default" };
  const ind = $("tool-indicator");
  ind.textContent = info.name;
  ind.title = info.hint;
  $("status-msg").textContent = info.hint ? `— ${info.hint}` : "";
  canvas.style.cursor = info.cursor;
  // Redraw so tool-specific overlays update immediately (e.g. the transform
  // tool's stroke handles appear and the box handles hide) without waiting
  // for the next pointer event.
  if (state.session) draw();
}

canvas.addEventListener("pointerdown", onPointerDown);
canvas.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);

function onPointerDown(evt) {
  const [px, py] = eventToPdf(evt);
  const q = getActiveQuestion();

  if (state.tool === "select" && q) {
    const handle = hitHandle(q.box, px, py);
    if (handle) {
      state.drag = { type: "resize", handle, qid: q.id, orig: { ...q.box } };
      return;
    }
    if (pointInBox(q.box, px, py)) {
      state.drag = { type: "move", qid: q.id, orig: { ...q.box }, start: [px, py] };
      return;
    }
  }

  if (state.tool === "transform" && q && q.strokes && q.strokes.length) {
    const b = strokesBBox(q.strokes);
    if (b) {
      const handle = hitHandle(b, px, py);
      if (handle) {
        state.drag = { type: "strokes-scale", qid: q.id, info: handleAndAnchor(handle, b), orig: cloneStrokes(q.strokes) };
        return;
      }
      if (pointInBox(b, px, py)) {
        state.drag = { type: "strokes-move", qid: q.id, start: [px, py], orig: cloneStrokes(q.strokes) };
        return;
      }
    }
  }

  if (state.tool === "box" && q) {
    state.drag = { type: "box-draw", qid: q.id, x0: px, y0: py, x1: px, y1: py };
    return;
  }

  if (state.tool === "freeform") {
    state.drag = { type: "freeform-draw", x0: px, y0: py, x1: px, y1: py };
    return;
  }

  if (state.tool === "pen" && q) {
    state.drag = { type: "pen", qid: q.id, pdfPts: [[px, py]], canvasPts: [[evt.offsetX, evt.offsetY]] };
    return;
  }

  if (state.tool === "eraser" && q) {
    state.drag = { type: "eraser", qid: q.id, pdfPts: [[px, py]], lastCanvas: [evt.offsetX, evt.offsetY] };
    return;
  }
}

function onPointerMove(evt) {
  if (!state.drag) return;
  const [px, py] = eventToPdf(evt);
  const d = state.drag;

  if (d.type === "box-draw" || d.type === "freeform-draw") {
    d.x1 = px; d.y1 = py;
    draw();
  } else if (d.type === "move") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    const dx = px - d.start[0], dy = py - d.start[1];
    q.box.x0 = d.orig.x0 + dx; q.box.x1 = d.orig.x1 + dx;
    q.box.y0 = d.orig.y0 + dy; q.box.y1 = d.orig.y1 + dy;
    draw();
  } else if (d.type === "resize") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    applyResize(q.box, d.handle, px, py);
    draw();
  } else if (d.type === "strokes-move") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    q.strokes = translateStrokes(d.orig, px - d.start[0], py - d.start[1]);
    draw();
  } else if (d.type === "strokes-scale") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    q.strokes = scaleStrokesAbout(d.orig, d.info.a, scaleFactor(d.info, px, py));
    draw();
  } else if (d.type === "pen") {
    d.pdfPts.push([px, py]);
    d.canvasPts.push([evt.offsetX, evt.offsetY]);
    draw();
  } else if (d.type === "eraser") {
    d.pdfPts.push([px, py]);
    d.lastCanvas = [evt.offsetX, evt.offsetY];
    draw();
  }
}

function applyResize(box, handle, px, py) {
  if (handle.includes("n")) box.y0 = py;
  if (handle.includes("s")) box.y1 = py;
  if (handle.includes("w")) box.x0 = px;
  if (handle.includes("e")) box.x1 = px;
}

async function onPointerUp(evt) {
  const d = state.drag;
  if (!d) return;
  state.drag = null;

  if (d.type === "box-draw") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    q.box.x0 = Math.min(d.x0, d.x1); q.box.x1 = Math.max(d.x0, d.x1);
    q.box.y0 = Math.min(d.y0, d.y1); q.box.y1 = Math.max(d.y0, d.y1);
    await saveBox(q);
  } else if (d.type === "freeform-draw") {
    const x0 = Math.min(d.x0, d.x1), x1 = Math.max(d.x0, d.x1);
    const y0 = Math.min(d.y0, d.y1), y1 = Math.max(d.y0, d.y1);
    if (x1 - x0 < 4 || y1 - y0 < 4) {
      setStatus("Box too small -- drag out a larger area for the freeform box.");
    } else {
      try {
        const data = await api("POST", "/api/session/questions/freeform", {
          page: state.pageNum, x0, y0, x1, y1, text: "",
        });
        state.session.questions.push(data.question);
        state.activeQid = data.question.id;
        ensureHistory(data.question.id);
        renderQuestionList();
        openFreeformModal(data.question.id);
      } catch (e) {
        setStatus(e.message);
      }
    }
  } else if (d.type === "move" || d.type === "resize") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    normalizeBox(q.box);
    await saveBox(q);
  } else if (d.type === "strokes-move" || d.type === "strokes-scale") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    await commitStrokes(d.qid, q.strokes, true);
    setStatus(d.type === "strokes-scale" ? "Resized text." : "Moved text.");
  } else if (d.type === "pen") {
    if (d.pdfPts.length >= 2) {
      const q = state.session.questions.find(qq => qq.id === d.qid);
      const widthPt = parseFloat($("pen-width").value);
      const strokes = (q.strokes || []).concat([{ points: d.pdfPts, source: "user", width_pt: widthPt }]);
      await commitStrokes(q.id, strokes, true);
      setStatus("Added pen stroke.");
    }
  } else if (d.type === "eraser") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    const radius = parseFloat($("eraser-size").value) / 2;
    const newStrokes = eraseStrokes(q.strokes || [], d.pdfPts, radius);
    await commitStrokes(q.id, newStrokes, true);
    setStatus("Erased.");
  }
  draw();
}

function normalizeBox(box) {
  if (box.x1 < box.x0) [box.x0, box.x1] = [box.x1, box.x0];
  if (box.y1 < box.y0) [box.y0, box.y1] = [box.y1, box.y0];
}

async function saveBox(q) {
  q.box.user_edited = true;
  await api("POST", `/api/session/questions/${q.id}/box`, {
    x0: q.box.x0, y0: q.box.y0, x1: q.box.x1, y1: q.box.y1,
  });
  draw();
}

// Eraser: drop any stroke point within `radius` (pdf points) of any point
// on the eraser drag path, splitting each stroke into separate sub-strokes
// at the gaps this creates (rather than just deleting whole strokes).
function eraseStrokes(strokes, eraserPath, radius) {
  const result = [];
  for (const stroke of strokes) {
    let current = [];
    for (const pt of stroke.points) {
      const erased = eraserPath.some(ep => {
        const dx = ep[0] - pt[0], dy = ep[1] - pt[1];
        return Math.sqrt(dx * dx + dy * dy) <= radius;
      });
      if (erased) {
        if (current.length >= 2) result.push({ ...stroke, points: current });
        current = [];
      } else {
        current.push(pt);
      }
    }
    if (current.length >= 2) result.push({ ...stroke, points: current });
  }
  return result;
}

// ---------- freeform text modal ----------
// Shared by both manual (freeform) entries and edits to an extracted
// question's prompt text -- the two cases differ only in what Render does:
// manual entries render immediately (no Gemma); extracted questions just
// save the edited text and leave generation to the existing Generate button.

function openFreeformModal(qid) {
  const q = state.session.questions.find(qq => qq.id === qid);
  $("freeform-text").value = q.text || "";
  $("freeform-render-btn").querySelector(".btn-label").textContent = q.source === "manual" ? "Render" : "Save";
  $("freeform-modal").style.display = "flex";
  $("freeform-modal").dataset.qid = qid;
  $("freeform-text").focus();
}

function closeFreeformModal() {
  $("freeform-modal").style.display = "none";
  delete $("freeform-modal").dataset.qid;
}

$("freeform-cancel-btn").addEventListener("click", closeFreeformModal);

$("freeform-render-btn").addEventListener("click", async () => {
  const qid = $("freeform-modal").dataset.qid;
  const q = state.session.questions.find(qq => qq.id === qid);
  if (!q) { closeFreeformModal(); return; }
  const text = $("freeform-text").value;
  if (q.source === "manual" && !text.trim()) {
    setStatus("Enter some text first.");
    return;
  }
  q.text = text;
  try {
    await api("POST", `/api/session/questions/${qid}/text`, { text });
    if (q.source === "manual") {
      setStatus("Rendering...");
      const data = await api("POST", `/api/session/questions/${qid}/generate`);
      q.answer = data.answer;
      q.strokes = data.strokes;
      pushHistory(qid, q.strokes);
      setStatus(data.warning || "Rendered.");
    } else {
      setStatus("Text updated. Click Generate to get a new answer.");
    }
  } catch (e) {
    setStatus(e.message);
  }
  closeFreeformModal();
  renderQuestionList();
  draw();
});

$("edit-text-btn").addEventListener("click", () => {
  const q = getActiveQuestion();
  if (q) openFreeformModal(q.id);
});

// ---------- action buttons ----------

$("generate-btn").addEventListener("click", async () => {
  const q = getActiveQuestion();
  if (!q) return;
  setStatus("Generating answer...");
  try {
    const data = await api("POST", `/api/session/questions/${q.id}/generate`);
    q.answer = data.answer;
    q.strokes = data.strokes;
    pushHistory(q.id, q.strokes);
    setStatus(data.warning || "Generated.");
  } catch (e) {
    setStatus(e.message);
  }
  renderQuestionList();
  draw();
});

$("regenerate-btn").addEventListener("click", async () => {
  const q = getActiveQuestion();
  if (!q) return;
  setStatus("Regenerating...");
  try {
    const data = await api("POST", `/api/session/questions/${q.id}/regenerate`);
    q.strokes = data.strokes;
    pushHistory(q.id, q.strokes);
    setStatus(data.warning || "Regenerated.");
  } catch (e) {
    setStatus(e.message);
  }
  draw();
});

$("delete-q-btn").addEventListener("click", async () => {
  const q = getActiveQuestion();
  if (!q) return;
  await api("DELETE", `/api/session/questions/${q.id}`);
  q.deleted = true;
  renderQuestionList();
  draw();
});

$("restore-q-btn").addEventListener("click", async () => {
  const q = getActiveQuestion();
  if (!q) return;
  await api("POST", `/api/session/questions/${q.id}/restore`);
  q.deleted = false;
  renderQuestionList();
  draw();
});

$("confirm-btn").addEventListener("click", async () => {
  $("confirm-status").textContent = "Rendering...";
  try {
    await api("POST", "/api/session/confirm");
    $("confirm-status").textContent = "Done. Downloading finished PDF...";
    window.location.href = "/api/render";
  } catch (e) {
    $("confirm-status").textContent = e.message;
  }
});

$("new-worksheet-btn").addEventListener("click", async () => {
  if (!window.confirm("Start a new worksheet? This discards the current session.")) return;
  await api("POST", "/api/session/reset");
  state.session = null;
  state.activeQid = null;
  state.history = {};
  $("upload-status").textContent = "";
  $("pdf-input").value = "";
  showUpload();
});

$("upload-btn").addEventListener("click", () => {
  const file = $("pdf-input").files[0];
  if (!file) { $("upload-status").textContent = "Choose a PDF first."; return; }
  uploadPdf(file);
});

const dropZone = $("upload-box");
["dragenter", "dragover"].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
  })
);
dropZone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (file) uploadPdf(file);
});

$("undo-btn").addEventListener("click", undo);
$("redo-btn").addEventListener("click", redo);

$("prev-page").addEventListener("click", () => loadPage(state.pageNum - 1));
$("next-page").addEventListener("click", () => loadPage(state.pageNum + 1));

document.querySelectorAll(".tool-btn").forEach(btn => {
  btn.addEventListener("click", () => setTool(btn.dataset.tool));
});

setTool("select");
loadSession().catch(e => setStatus(e.message));
