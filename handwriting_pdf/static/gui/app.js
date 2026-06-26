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

function setStatus(msg) { $("status-bar").textContent = msg || ""; }

// ---------- session / page loading ----------

async function loadSession() {
  state.session = await api("GET", "/api/session");
  if (!state.activeQid) {
    const first = state.session.questions.find(q => !q.deleted);
    if (first) { state.activeQid = first.id; state.pageNum = first.box.page; }
  }
  renderQuestionList();
  await loadPage(state.pageNum);
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

  if (state.drag && state.drag.type === "box-draw") drawDragBox(state.drag);
  if (state.drag && state.drag.type === "pen") drawLivePoints(state.drag.canvasPts, "#1f4fd6");
  if (state.drag && state.drag.type === "eraser") drawEraserCursor(state.drag);
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

  if (q.id === state.activeQid) {
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

function drawStrokes(strokes, color) {
  for (const s of strokes) {
    if (!s.points || s.points.length < 2) continue;
    ctx.beginPath();
    ctx.strokeStyle = s.source === "user" ? "#1f4fd6" : color;
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
    textDiv.textContent = q.text;
    const metaDiv = document.createElement("div");
    metaDiv.className = "q-meta";
    metaDiv.textContent = `p.${q.box.page + 1} | ${q.answer ? "answered" : "no answer"}`;
    li.appendChild(textDiv);
    li.appendChild(metaDiv);
    li.addEventListener("click", () => selectQuestion(q.id));
    ul.appendChild(li);
  }
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

function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll(".tool-btn").forEach(b => b.classList.toggle("active", b.dataset.tool === tool));
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

  if (state.tool === "box" && q) {
    state.drag = { type: "box-draw", qid: q.id, x0: px, y0: py, x1: px, y1: py };
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

  if (d.type === "box-draw") {
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
  } else if (d.type === "move" || d.type === "resize") {
    const q = state.session.questions.find(qq => qq.id === d.qid);
    normalizeBox(q.box);
    await saveBox(q);
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
  await api("POST", "/api/session/confirm");
  $("confirm-status").textContent = "Layout confirmed.";
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
