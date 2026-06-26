"""Composite a finalized layout session's strokes onto the source PDF.

Reads the same session JSON the GUI (gui_app.py) edits: each active,
non-deleted question's `strokes` list (absolute PDF-point coordinates,
PyMuPDF convention) is drawn directly onto its page with
Page.new_shape()/draw_polyline()/finish()/commit(). Questions with no
strokes yet (never generated, or deleted) are skipped.
"""
import fitz

import layout_session as ls

STROKE_COLOR = (0, 0, 0)
DEFAULT_STROKE_WIDTH = 1.4


def compose(session_path, out_path):
    """Render every active question's strokes onto the session's source
    PDF and save the result to `out_path`. Returns the number of strokes
    drawn."""
    session = ls.load(session_path)
    doc = fitz.open(session["source_pdf"])

    drawn = 0
    for q in ls.active_questions(session):
        for stroke in q.get("strokes") or []:
            points = stroke["points"]
            if len(points) < 2:
                continue
            page = doc[q["box"]["page"]]
            shape = page.new_shape()
            shape.draw_polyline(points)
            shape.finish(
                width=stroke.get("width_pt", DEFAULT_STROKE_WIDTH),
                color=STROKE_COLOR,
                fill=None,
            )
            shape.commit()
            drawn += 1

    doc.save(out_path)
    doc.close()
    return drawn


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    drawn = compose(args.session, args.out)
    print(f"Drew {drawn} strokes -> {args.out}")


if __name__ == "__main__":
    main()
