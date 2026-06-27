"""Session state shared across the extract / serve / render CLI stages.

A session is a single JSON file that accumulates state as it moves through
the pipeline: `extract` populates questions (text + suggested boxes),
`serve` (the GUI) lets the user confirm/edit boxes, generates an answer +
strokes per question, and lets the user hand-edit those strokes with the
pen/eraser tools; `render` reads the finalized strokes and composites them
onto the source PDF. Keeping it as one file (rather than passing objects
between separate process invocations) is what makes the GUI step resumable
and inspectable.

All box/stroke coordinates are in PDF points, in PyMuPDF's page coordinate
space (origin top-left, y increasing downward) -- the space pdf_compose.py
draws directly into.

Schema (top-level dict):
  version: int
  source_pdf: str
  pages: [{"width": float, "height": float}, ...]
  questions: [{
      "id": str,
      "text": str,
      "page": int,
      "match_bbox": [x0, y0, x1, y1] | null,
      "box": {"page": int, "x0", "y0", "x1", "y1", "user_edited": bool},
      "answer": {"raw": str, "runs": [{"kind": "text"|"math", "value": str}]} | null,
      "strokes": [{"points": [[x, y], ...], "source": "generated"|"user"}],
      "deleted": bool,
      "source": "manual" | absent,  # present only for user-created freeform
                                     # boxes; their "text" is rendered directly
                                     # (no Gemma call) instead of being an
                                     # extracted question prompt
  }, ...]
  confirmed: bool
"""
import json
import os
import uuid

SCHEMA_VERSION = 1


def new_session(source_pdf, pages, questions):
    return {
        "version": SCHEMA_VERSION,
        "source_pdf": source_pdf,
        "pages": pages,
        "questions": questions,
        "confirmed": False,
    }


def new_freeform_question(session, page, box, text=""):
    """A user-drawn box with user-typed text, rendered without Gemma --
    otherwise identical to an extracted question record so it flows through
    the same generate/regenerate/strokes/compose pipeline unmodified."""
    q = {
        "id": f"manual-{uuid.uuid4().hex[:8]}",
        "text": text,
        "page": page,
        "match_bbox": None,
        "box": {
            "page": page,
            "x0": box["x0"], "y0": box["y0"], "x1": box["x1"], "y1": box["y1"],
            "user_edited": True,
        },
        "answer": None,
        "strokes": [],
        "deleted": False,
        "seed": 0,
        "source": "manual",
    }
    session["questions"].append(q)
    return q


def load(path):
    with open(path) as f:
        session = json.load(f)
    if session.get("version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported session schema version {session.get('version')!r} in {path}"
        )
    return session


def save(session, path):
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(session, f, indent=2)
    os.replace(tmp_path, path)


def get_question(session, qid):
    for q in session["questions"]:
        if q["id"] == qid:
            return q
    raise KeyError(f"No question with id {qid!r}")


def active_questions(session):
    """Questions not marked deleted, in stored order."""
    return [q for q in session["questions"] if not q.get("deleted")]
