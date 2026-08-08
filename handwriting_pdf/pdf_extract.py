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


MIN_BOX_HEIGHT = 36
DEFAULT_BOX_HEIGHT = 60
PAGE_MARGIN = 36
BOX_GAP = 6


def text_line_bboxes(page):
    """Every text line's bbox on the page, in reading order. Used both to
    measure a question's true extent and to find the blank areas between
    questions where an answer can actually go."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if "".join(s["text"] for s in line["spans"]).strip():
                lines.append(list(line["bbox"]))
    return lines


BLOCK_LINE_GAP = 10  # a question's own lines sit closer together than this


def question_block_bbox(line_bboxes, first_line_bbox, next_first_line_bbox):
    """The question's full on-page extent, not just its opening line.

    locate_question_bbox only matches the first line, so a question that
    carries a piecewise definition or a displayed formula under it reports
    a bbox ending partway through itself -- and an answer box anchored to
    that lands on top of the question.

    A question is a run of closely-spaced lines: absorb lines downward
    while each is within BLOCK_LINE_GAP of the run so far, and stop at the
    first real gap. That gap is the blank answer area (or, on the last
    question of a page, the space above the footer), so neither gets
    counted as part of the question.
    """
    top = first_line_bbox[1]
    limit = next_first_line_bbox[1] if next_first_line_bbox else float("inf")
    # 1pt of slack: sub/superscript runs can sit a hair above their line.
    candidates = sorted((lb for lb in line_bboxes
                         if lb[1] >= top - 1 and lb[1] < limit - 1),
                        key=lambda lb: lb[1])
    bbox = list(first_line_bbox)
    for lb in candidates:
        if lb[1] - bbox[3] > BLOCK_LINE_GAP:
            break
        bbox[0] = min(bbox[0], lb[0])
        bbox[1] = min(bbox[1], lb[1])
        bbox[2] = max(bbox[2], lb[2])
        bbox[3] = max(bbox[3], lb[3])
    return bbox


def _free_gaps_below(line_bboxes, y_from, page_rect):
    """Vertical gaps below `y_from` that no text occupies, as (top, bottom)."""
    spans = sorted((lb[1], lb[3]) for lb in line_bboxes)
    gaps = []
    cursor = y_from
    for top, bottom in spans:
        if bottom <= cursor:
            continue
        if top > cursor:
            gaps.append((cursor, top))
        cursor = max(cursor, bottom)
    bottom_limit = page_rect.height - PAGE_MARGIN
    if cursor < bottom_limit:
        gaps.append((cursor, bottom_limit))
    return gaps


def suggest_answer_box(page, question_bbox, next_question_bbox, page_rect):
    """Propose an answer box in the gap between this question and the next.

    Returns None when that gap is too thin to write in -- the caller then
    places the box in free space elsewhere (see fallback_answer_box) rather
    than drawing it over the following question's text.
    """
    y0 = question_bbox[3] + BOX_GAP
    if next_question_bbox is not None:
        y1_limit = next_question_bbox[1] - BOX_GAP
    else:
        y1_limit = page_rect.height - PAGE_MARGIN
    if y1_limit - y0 < MIN_BOX_HEIGHT:
        return None
    return [question_bbox[0], y0, page_rect.width - PAGE_MARGIN,
            min(y0 + DEFAULT_BOX_HEIGHT, y1_limit)]


def fallback_answer_box(question_bbox, page_rect, occupied):
    """Place a box for a question with no room before the next one -- a
    stem like "3. Let f(t) = {...}" whose parts follow immediately.

    Picks the largest empty gap below the question, treating both printed
    text and already-placed boxes as occupied, so these never land on the
    page's content or on another question's answer space.
    """
    y0 = question_bbox[3] + BOX_GAP
    gaps = [g for g in _free_gaps_below(occupied, y0, page_rect)
            if g[1] - g[0] >= MIN_BOX_HEIGHT + BOX_GAP]
    if gaps:
        top, bottom = max(gaps, key=lambda g: g[1] - g[0])
        return [question_bbox[0], top + BOX_GAP,
                page_rect.width - PAGE_MARGIN,
                min(top + BOX_GAP + DEFAULT_BOX_HEIGHT, bottom)]
    # Page is full; keep the box on the page and let the user move it.
    y1 = min(y0 + MIN_BOX_HEIGHT, page_rect.height - PAGE_MARGIN)
    return [question_bbox[0], y1 - MIN_BOX_HEIGHT,
            page_rect.width - PAGE_MARGIN, y1]


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

        line_bboxes = text_line_bboxes(page)
        # Two passes: first give every question the gap that follows it,
        # then fit the leftovers (questions with no room before the next
        # one) into whatever space those passes left free. Doing it in this
        # order stops a stem from claiming the work area that belongs to
        # the sub-questions underneath it.
        blocks, boxes, pending = {}, {}, []
        for i, (chunk, bbox) in enumerate(zip(chunks, bboxes)):
            if bbox is None:
                continue
            next_bbox = None
            for nb in bboxes[i + 1:]:
                if nb is not None:
                    next_bbox = nb
                    break
            blocks[i] = question_block_bbox(line_bboxes, bbox, next_bbox)
            box = suggest_answer_box(page, blocks[i], next_bbox, page.rect)
            if box is None:
                pending.append(i)
            else:
                boxes[i] = box
        occupied = line_bboxes + list(boxes.values())
        for i in pending:
            boxes[i] = fallback_answer_box(blocks[i], page.rect, occupied)
            occupied.append(boxes[i])

        for i, (chunk, bbox) in enumerate(zip(chunks, bboxes)):
            if bbox is None:
                continue
            box = boxes[i]
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
