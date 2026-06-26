"""Local browser-based GUI for the worksheet pipeline's placement/editing step.

Runs entirely on localhost (Flask). The user: places/resizes an answer box
per detected question, triggers generation (Gemma answer -> handwriting
strokes) for each, and can hand-edit the resulting strokes with a pen tool
(draw additional strokes) and an eraser tool (cut existing strokes), plus
regenerate/undo/redo, before confirming the layout. All coordinates cross
the wire in PDF points; the frontend converts to/from canvas pixels using
the page size returned by /api/page/<n>/size.
"""
import io
import os

from flask import Flask, jsonify, request, send_file, send_from_directory

import fitz

import gemma_client
import layout_session as ls
import pdf_compose
import pdf_extract
import render_box

STATIC_DIR = "static/gui"
PAGE_RENDER_ZOOM = 2.0


def create_app(session_path, ollama_url=gemma_client.DEFAULT_OLLAMA_URL, model=gemma_client.DEFAULT_MODEL):
    app = Flask(__name__, static_folder=None)
    state = {"doc": None}

    def _doc():
        if state["doc"] is None:
            session = ls.load(session_path)
            state["doc"] = fitz.open(session["source_pdf"])
        return state["doc"]

    def _workspace_dir():
        return os.path.dirname(os.path.abspath(session_path)) or "."

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/gui/<path:filename>")
    def static_files(filename):
        return send_from_directory(STATIC_DIR, filename)

    @app.get("/api/session")
    def get_session():
        if not os.path.exists(session_path):
            return jsonify({"exists": False}), 404
        return jsonify(ls.load(session_path))

    @app.post("/api/upload")
    def upload_pdf():
        """Accept a worksheet PDF uploaded from the browser, extract its
        questions, and create a fresh session at session_path -- this is
        what lets the whole pipeline run from the GUI with no CLI step."""
        file = request.files.get("pdf")
        if file is None or not file.filename:
            return jsonify({"ok": False, "error": "No PDF file provided."}), 400

        workspace_dir = _workspace_dir()
        os.makedirs(workspace_dir, exist_ok=True)
        source_pdf_path = os.path.join(workspace_dir, "source.pdf")
        file.save(source_pdf_path)

        doc = pdf_extract.load_pdf(source_pdf_path)
        questions = pdf_extract.build_question_records(doc)
        pages = [{"width": p.rect.width, "height": p.rect.height} for p in doc]
        session = ls.new_session(source_pdf_path, pages, questions)
        ls.save(session, session_path)
        state["doc"] = None
        return jsonify({"ok": True, "session": session})

    @app.post("/api/session/reset")
    def reset_session():
        """Discard the current session so the GUI shows the upload screen
        again, for starting a new worksheet without restarting the server."""
        if os.path.exists(session_path):
            os.remove(session_path)
        state["doc"] = None
        return jsonify({"ok": True})

    @app.get("/api/page/<int:page_num>.png")
    def page_png(page_num):
        doc = _doc()
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(PAGE_RENDER_ZOOM, PAGE_RENDER_ZOOM))
        return send_file(io.BytesIO(pix.tobytes("png")), mimetype="image/png")

    @app.get("/api/page/<int:page_num>/size")
    def page_size(page_num):
        rect = _doc()[page_num].rect
        return jsonify({"width": rect.width, "height": rect.height})

    @app.post("/api/session/questions/<qid>/box")
    def update_box(qid):
        session = ls.load(session_path)
        q = ls.get_question(session, qid)
        body = request.get_json()
        q["box"] = {
            "page": q["box"]["page"],
            "x0": body["x0"], "y0": body["y0"], "x1": body["x1"], "y1": body["y1"],
            "user_edited": True,
        }
        ls.save(session, session_path)
        return jsonify({"ok": True})

    @app.post("/api/session/questions/<qid>/generate")
    def generate_answer(qid):
        session = ls.load(session_path)
        q = ls.get_question(session, qid)
        try:
            raw = gemma_client.generate_answer(q["text"], model=model, ollama_url=ollama_url)
        except gemma_client.GemmaClientError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 502
        runs = gemma_client.split_runs(raw)
        q["answer"] = {"raw": raw, "runs": runs}
        q["seed"] = q.get("seed", 0)
        strokes, warning = render_box.render_answer_in_box(runs, q["box"], seed=q["seed"])
        q["strokes"] = strokes
        ls.save(session, session_path)
        return jsonify({"ok": True, "answer": q["answer"], "strokes": strokes, "warning": warning})

    @app.post("/api/session/questions/<qid>/regenerate")
    def regenerate_answer(qid):
        session = ls.load(session_path)
        q = ls.get_question(session, qid)
        if not q.get("answer"):
            return jsonify({"ok": False, "error": "No answer generated yet."}), 400
        q["seed"] = q.get("seed", 0) + 1000
        strokes, warning = render_box.render_answer_in_box(q["answer"]["runs"], q["box"], seed=q["seed"])
        q["strokes"] = strokes
        ls.save(session, session_path)
        return jsonify({"ok": True, "strokes": strokes, "warning": warning})

    @app.put("/api/session/questions/<qid>/strokes")
    def put_strokes(qid):
        session = ls.load(session_path)
        q = ls.get_question(session, qid)
        body = request.get_json()
        q["strokes"] = body["strokes"]
        ls.save(session, session_path)
        return jsonify({"ok": True})

    @app.delete("/api/session/questions/<qid>")
    def delete_question(qid):
        session = ls.load(session_path)
        q = ls.get_question(session, qid)
        q["deleted"] = True
        ls.save(session, session_path)
        return jsonify({"ok": True})

    @app.post("/api/session/questions/<qid>/restore")
    def restore_question(qid):
        session = ls.load(session_path)
        q = ls.get_question(session, qid)
        q["deleted"] = False
        ls.save(session, session_path)
        return jsonify({"ok": True})

    @app.post("/api/session/confirm")
    def confirm_session():
        session = ls.load(session_path)
        session["confirmed"] = True
        ls.save(session, session_path)
        return jsonify({"ok": True})

    @app.get("/api/render")
    def render_and_download():
        """Composite the session's strokes onto the source PDF and return
        the result as a downloadable file, so finishing a worksheet never
        requires leaving the browser."""
        out_path = os.path.join(_workspace_dir(), "filled.pdf")
        pdf_compose.compose(session_path, out_path)
        return send_file(out_path, as_attachment=True, download_name="filled_worksheet.pdf")

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--ollama-url", default=gemma_client.DEFAULT_OLLAMA_URL)
    parser.add_argument("--model", default=gemma_client.DEFAULT_MODEL)
    args = parser.parse_args()

    app = create_app(args.session, ollama_url=args.ollama_url, model=args.model)
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
