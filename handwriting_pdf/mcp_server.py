"""MCP connector for the worksheet handwriting pipeline.

Exposes the local Flask app (gui_app.py) as MCP tools over stdio, so a
Claude client (Claude Code, Claude Desktop) can drive the whole workflow:
upload a worksheet PDF, read the detected questions and their proposed
answer boxes, write Claude-solved answers as handwriting strokes, look at
a composited preview to verify placement, and save the finished PDF.

The Flask app stays the single owner of session state -- this server is a
thin HTTP proxy, so the browser GUI and Claude can even be used side by
side on the same session.

Run the Flask app first:
    python worksheet_cli.py serve --session my_worksheet.json --port 5000

Register with Claude Code (from the repo root):
    claude mcp add handwriting \
        -e HANDWRITING_URL=http://127.0.0.1:5000 \
        -- python handwriting_pdf/mcp_server.py

Coordinates everywhere are PDF points, origin at the page's top-left,
y increasing downward. Page PNGs are rendered at 2x, so pixel/2 = point.
"""
import os

import requests

try:
    # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer, Image
except ImportError:  # mcp 1.x named the same class FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer, Image

BASE_URL = os.environ.get("HANDWRITING_URL", "http://127.0.0.1:5000").rstrip("/")
PAGE_RENDER_ZOOM = 2.0  # matches gui_app.PAGE_RENDER_ZOOM

mcp = MCPServer(
    "handwriting",
    instructions=(
        "Tools for filling a worksheet PDF with generated handwriting. "
        "Typical flow: upload_worksheet -> solve each question yourself -> "
        "answer_question (or place_answer for boxes detection missed) -> "
        "preview_page to visually verify -> finalize. Coordinates are PDF "
        "points, origin top-left, y down."
    ),
)


def _request(method, path, **kwargs):
    kwargs.setdefault("timeout", 300)
    try:
        resp = requests.request(method, BASE_URL + path, **kwargs)
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot reach the handwriting server at {BASE_URL}. Start it "
            f"first: python worksheet_cli.py serve --session "
            f"my_worksheet.json --port {BASE_URL.rsplit(':', 1)[-1]}"
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error") or resp.text[:300]
        except ValueError:
            detail = resp.text[:300]
        raise RuntimeError(f"{method} {path} failed ({resp.status_code}): {detail}")
    return resp


def _json(method, path, **kwargs):
    return _request(method, path, **kwargs).json()


def _question_summary(q):
    box = q["box"]
    return {
        "id": q["id"],
        "source": q.get("source", "detected"),
        "page": box["page"],
        "box": {k: round(box[k], 1) for k in ("x0", "y0", "x1", "y1")},
        "question_text": q.get("text", ""),
        "answered": bool(q.get("strokes")),
        "answer_text": (q.get("answer") or {}).get("raw", ""),
    }


def _session_questions():
    session = _json("GET", "/api/session")
    qs = [q for q in session.get("questions", []) if not q.get("deleted")]
    return [_question_summary(q) for q in qs]


@mcp.tool()
def upload_worksheet(pdf_path: str) -> dict:
    """Start a fresh session from a worksheet PDF. Detects the questions on
    each page and proposes an answer box for each one. Returns the detected
    questions with their ids, pages, box coordinates (PDF points, origin
    top-left), and text. Review the proposed boxes before answering; use
    place_answer for anything detection missed."""
    path = os.path.abspath(os.path.expanduser(pdf_path))
    if not os.path.isfile(path):
        raise RuntimeError(f"No such file: {path}")
    _json("POST", "/api/session/reset")
    with open(path, "rb") as f:
        _request("POST", "/api/upload",
                 files={"pdf": (os.path.basename(path), f, "application/pdf")})
    questions = _session_questions()
    return {"ok": True, "num_questions": len(questions), "questions": questions}


@mcp.tool()
def list_questions() -> dict:
    """List the current session's questions: id, page, answer box (PDF
    points), question text, and whether each is answered yet."""
    questions = _session_questions()
    return {"ok": True, "num_questions": len(questions), "questions": questions}


@mcp.tool()
def get_page(page: int) -> list:
    """Get a page of the ORIGINAL worksheet as an image, plus its size in
    PDF points. Use it to read the problems and to work out coordinates
    for place_answer: the PNG is rendered at 2x, so divide pixel
    coordinates by 2 to get PDF points (origin top-left, y down)."""
    size = _json("GET", f"/api/page/{page}/size")
    png = _request("GET", f"/api/page/{page}.png").content
    note = (
        f"Page {page}: {size['width']:.0f} x {size['height']:.0f} PDF points. "
        f"Image rendered at {PAGE_RENDER_ZOOM}x -- pixel / {PAGE_RENDER_ZOOM} = point."
    )
    return [Image(data=png, format="png"), note]


@mcp.tool()
def answer_question(question_id: str, answer_text: str) -> dict:
    """Write an answer into a question's box as generated handwriting.
    Formatting rules for answer_text: wrap math in $...$; put each step of
    multi-step work on its own line (\\n); matrices via
    \\begin{pmatrix}...\\end{pmatrix}, piecewise via \\begin{cases}; use
    simple LaTeX (\\frac, \\sqrt, powers, \\sum, \\int, \\lim, Greek);
    plain prose stays outside the $. A returned warning means the answer
    had to be shrunk to fit -- consider a bigger box (update_box)."""
    result = _json("POST", f"/api/session/questions/{question_id}/generate",
                   json={"text": answer_text})
    return {"ok": True, "warning": result.get("warning")}


@mcp.tool()
def place_answer(page: int, x0: float, y0: float, x1: float, y1: float,
                 answer_text: str) -> dict:
    """Create a new answer box at the given PDF-point coordinates (origin
    top-left, y down; x0<x1, y0<y1) and write answer_text into it as
    handwriting (same formatting rules as answer_question). For a question
    the detector missed: place the box in the blank space below or beside
    the printed question, roughly 25-30 points tall per expected line."""
    created = _json("POST", "/api/session/questions/freeform",
                    json={"page": page, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "text": answer_text})
    qid = created["question"]["id"]
    result = _json("POST", f"/api/session/questions/{qid}/generate", json={})
    return {"ok": True, "question_id": qid, "warning": result.get("warning")}


@mcp.tool()
def update_box(question_id: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    """Move/resize a question's answer box (PDF points) and re-render its
    existing answer to fit the new box. Use after preview_page shows an
    answer overlapping the printed question or running out of room."""
    _json("POST", f"/api/session/questions/{question_id}/box",
          json={"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    session = _json("GET", "/api/session")
    q = next((q for q in session.get("questions", []) if q["id"] == question_id), None)
    warning = None
    if q and q.get("answer"):
        result = _json("POST", f"/api/session/questions/{question_id}/regenerate")
        warning = result.get("warning")
    return {"ok": True, "warning": warning}


@mcp.tool()
def preview_page(page: int) -> list:
    """Get a page image WITH the handwritten answers composited on, exactly
    as the final PDF will look. Call this after answering a page's
    questions and visually check: answers inside their boxes, not covering
    printed text, legible size. Fix problems with update_box, then
    re-preview."""
    png = _request("GET", f"/api/page/{page}/preview.png").content
    return [Image(data=png, format="png"),
            f"Preview of page {page} with current handwriting composited."]


@mcp.tool()
def finalize(output_path: str) -> dict:
    """Confirm the session, composite all handwriting onto the source PDF,
    and save the finished worksheet to output_path. Do this only after
    preview_page checks pass."""
    out = os.path.abspath(os.path.expanduser(output_path))
    _json("POST", "/api/session/confirm")
    pdf_bytes = _request("GET", "/api/render").content
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "wb") as f:
        f.write(pdf_bytes)
    return {"ok": True, "output_path": out, "bytes": len(pdf_bytes)}


if __name__ == "__main__":
    mcp.run()
