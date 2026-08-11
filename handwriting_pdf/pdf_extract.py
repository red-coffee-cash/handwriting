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

_MARKER = r"\(?\d+[.)]|\(?[a-zA-Z][.)]|Q\d+[:.]?"
# A question marker at the start of a line, with its text following.
QUESTION_RE = re.compile(r"^\s*(?:%s)\s+\S" % _MARKER)
# A marker sitting alone on its line. Exporters routinely break
# "f.  y'' + 6y' + 9y = ..." so the marker lands on its own line with the
# formula on the next -- those questions were being dropped entirely.
# Accepted only when a content line follows (see _question_start_lines),
# since a bare "x." is far more often the tail of an expression like 1/x.
MARKER_ONLY_RE = re.compile(r"^\s*(?:%s)\s*$" % _MARKER)


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


def page_lines(page):
    """The page's text lines as (text, bbox), in reading order.

    Questions are chunked from this same list rather than from a separate
    page.get_text() pass, so each question's bbox is the bbox of the line
    that starts it. The old approach searched the page for a question's
    first line, which on documents with short repeated openings ("a. y",
    "b. y", ...) matched the wrong occurrence and scattered answer boxes
    to unrelated parts of the page.
    """
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"])
            if text.strip():
                out.append((text, list(line["bbox"])))
    return out


def _question_start_lines(lines):
    """Indices of lines that begin a question, given (text, bbox) lines."""
    starts = []
    for i, (text, _) in enumerate(lines):
        if QUESTION_RE.match(text):
            starts.append(i)
        elif MARKER_ONLY_RE.match(text):
            # Only a real marker if actual content follows it.
            nxt = lines[i + 1][0] if i + 1 < len(lines) else ""
            if nxt.strip() and not MARKER_ONLY_RE.match(nxt) and not QUESTION_RE.match(nxt):
                starts.append(i)
    return starts


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
# Floor for boxes packed into shared leftover space: enough for one line at
# the smallest render tier (render_box.LINE_HEIGHT_TIERS[-1]).
MIN_FALLBACK_HEIGHT = 12
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


def suggest_answer_box(page, question_bbox, next_question_bbox, page_rect,
                       line_bboxes=()):
    """Propose an answer box in the gap between this question and the next.

    The gap is bounded by the next question *and* by the next line of
    printed text below -- a page footer sits below the last question but
    is not a question, and a box run down to the bottom margin would be
    drawn straight through it.

    Returns None when the gap is too thin to write in; the caller then
    packs the box into free space elsewhere (allocate_fallback_boxes).
    """
    y0 = question_bbox[3] + BOX_GAP
    if next_question_bbox is not None:
        y1_limit = next_question_bbox[1] - BOX_GAP
    else:
        y1_limit = page_rect.height - PAGE_MARGIN
    for lb in line_bboxes:
        if lb[1] >= y0:
            y1_limit = min(y1_limit, lb[1] - BOX_GAP)
    if y1_limit - y0 < MIN_BOX_HEIGHT:
        return None
    return [question_bbox[0], y0, page_rect.width - PAGE_MARGIN,
            min(y0 + DEFAULT_BOX_HEIGHT, y1_limit)]


def allocate_fallback_boxes(order, blocks, page_rect, occupied):
    """Place boxes for the questions that have no usable gap after them.

    A tightly-set list ("a." through "j.", a couple of points apart) puts
    every item in this bucket at once. They are packed into the page's
    remaining free space **in document order**, top to bottom, sharing it
    out evenly -- so the answers read in the same order as the questions
    instead of scattering to whichever gap happened to be biggest, and
    none of them overlap each other or the printed text.
    """
    if not order:
        return {}
    gaps = [g for g in _free_gaps_below(occupied, 0, page_rect)
            if g[1] - g[0] >= MIN_FALLBACK_HEIGHT + BOX_GAP]

    def room_from(gap_index, cursor):
        """Free space still available at or after (gap_index, cursor)."""
        total = 0.0
        for gj in range(gap_index, len(gaps)):
            top, bottom = gaps[gj]
            total += max(0.0, bottom - max(top, cursor if gj == gap_index else top))
        return total

    boxes = {}
    gi, cursor = 0, (gaps[0][0] if gaps else page_rect.height)
    for idx, qi in enumerate(order):
        block = blocks[qi]
        # Re-derive the height each time from what is actually left and how
        # many questions still need a home, so a long list degrades into
        # uniformly short boxes instead of running out and overlapping.
        remaining = len(order) - idx
        share = room_from(gi, cursor) / remaining - BOX_GAP
        height = max(MIN_FALLBACK_HEIGHT, min(DEFAULT_BOX_HEIGHT, share))
        y0 = None
        while gi < len(gaps):
            top, bottom = gaps[gi]
            start = max(cursor, top, block[3] + BOX_GAP)
            if bottom - start >= height:
                y0 = start
                break
            gi += 1
            if gi < len(gaps):
                cursor = gaps[gi][0]
        if y0 is None:
            # Genuinely out of room (more items than the page has blank
            # space). Stack tight below the last box and squeeze against
            # whatever text comes next, so the proposal stays inside the
            # free space even when it ends up too small to write in -- the
            # user moves these. Overlapping the page's own text would be
            # worse than a box that is obviously too short.
            y0 = min(cursor, page_rect.height - PAGE_MARGIN - MIN_FALLBACK_HEIGHT)
            limit = page_rect.height - PAGE_MARGIN
            for lb in occupied:
                if lb[1] >= y0:
                    limit = min(limit, lb[1] - BOX_GAP)
            height = max(1.0, min(MIN_FALLBACK_HEIGHT, limit - y0))
        cursor = y0 + height + BOX_GAP
        boxes[qi] = [block[0], y0, page_rect.width - PAGE_MARGIN, y0 + height]
    return boxes


def build_question_records(doc):
    """Top-level entry point: returns a list of question dicts ready to be
    embedded in a layout_session, one per detected question across all
    pages: {id, text, page, match_bbox, box}."""
    records = []
    qid = 0
    for page_index, page in enumerate(doc):
        lines = page_lines(page)
        starts = _question_start_lines(lines)
        chunks, bboxes = [], []
        for si, start in enumerate(starts):
            end = starts[si + 1] if si + 1 < len(starts) else len(lines)
            chunk = "\n".join(t for t, _ in lines[start:end]).strip()
            if not chunk:
                continue
            chunks.append(chunk)
            bboxes.append(list(lines[start][1]))

        line_bboxes = [b for _, b in lines]
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
            box = suggest_answer_box(page, blocks[i], next_bbox, page.rect,
                                     line_bboxes=line_bboxes)
            if box is None:
                pending.append(i)
            else:
                boxes[i] = box
        occupied = line_bboxes + list(boxes.values())
        boxes.update(allocate_fallback_boxes(pending, blocks, page.rect, occupied))

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
