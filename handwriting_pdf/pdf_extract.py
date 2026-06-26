"""Extract question text and suggested answer-box placements from a PDF.

This is intentionally a heuristic, best-effort pass: it finds lines that
look like the start of a new question (e.g. "1.", "2)", "Q3:"), locates
their on-page bounding box, and proposes an empty-space box below/after
each question as a starting point for the answer. None of this needs to
be exact -- the GUI step lets the user drag/resize every box before
anything is rendered, per the "semi-automatic with confirmation" design.
"""
import re

import fitz  # PyMuPDF

QUESTION_RE = re.compile(r"^\s*(?:\(?\d+[.)]|\(?[a-zA-Z][.)]|Q\d+[:.]?)\s+\S")


def load_pdf(path):
    return fitz.open(path)


def extract_page_texts(doc):
    """Return a list of raw text strings, one per page."""
    return [page.get_text() for page in doc]


def split_into_questions(page_text):
    """Split a page's text into question chunks at lines that look like a
    new question/item start. Returns a list of (question_text,) strings.
    Text before the first match (if any) is dropped -- typically headers."""
    lines = page_text.splitlines()
    starts = [i for i, line in enumerate(lines) if QUESTION_RE.match(line)]
    if not starts:
        return []
    starts.append(len(lines))
    chunks = []
    for start, end in zip(starts[:-1], starts[1:]):
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def locate_question_bbox(page, question_text):
    """Find the on-page bounding box for a question's text via verbatim
    search. Falls back to searching just the first line if the full
    (possibly multi-line) text isn't found as a single search hit."""
    first_line = question_text.splitlines()[0].strip()
    hits = page.search_for(first_line)
    if not hits:
        # Try a shorter prefix in case of mid-word wraps or odd spacing.
        hits = page.search_for(first_line[:40])
    if not hits:
        return None
    rect = hits[0]
    for h in hits[1:]:
        rect |= h
    return [rect.x0, rect.y0, rect.x1, rect.y1]


def suggest_answer_box(page, question_bbox, next_question_bbox, page_rect):
    """Propose an answer box: starts just below the question's bbox, runs
    to just above the next question's bbox (or the bottom margin if this
    is the last question on the page), spans a comfortable answer height."""
    margin = 36
    x0 = question_bbox[0]
    y0 = question_bbox[3] + 6
    x1 = page_rect.width - margin
    if next_question_bbox is not None:
        y1_limit = next_question_bbox[1] - 6
    else:
        y1_limit = page_rect.height - margin
    default_height = 60
    y1 = min(y0 + default_height, y1_limit) if y1_limit > y0 else y0 + default_height
    y1 = max(y1, y0 + 20)
    return [x0, y0, x1, y1]


def build_question_records(doc):
    """Top-level entry point: returns a list of question dicts ready to be
    embedded in a layout_session, one per detected question across all
    pages: {id, text, page, match_bbox, box}."""
    records = []
    qid = 0
    for page_index, page in enumerate(doc):
        page_text = page.get_text()
        chunks = split_into_questions(page_text)
        bboxes = []
        for chunk in chunks:
            bboxes.append(locate_question_bbox(page, chunk))

        for i, (chunk, bbox) in enumerate(zip(chunks, bboxes)):
            if bbox is None:
                continue
            next_bbox = None
            for nb in bboxes[i + 1:]:
                if nb is not None:
                    next_bbox = nb
                    break
            box = suggest_answer_box(page, bbox, next_bbox, page.rect)
            records.append({
                "id": f"q{qid}",
                "text": chunk,
                "page": page_index,
                "match_bbox": bbox,
                "box": {
                    "page": page_index,
                    "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                    "user_edited": False,
                },
            })
            qid += 1
    return records
