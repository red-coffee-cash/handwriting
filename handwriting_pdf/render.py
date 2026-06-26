"""Render sampled pen strokes onto a PDF page.

Path construction follows the original reference implementation
(sjvasquez/handwriting-synthesis demo.py `_draw`): straight-line segments
only, broken into a new sub-path every time a point's end-of-stroke flag is
set, plus a 1.5x coordinate scale applied before the denoise/align cleanup.
Legibility comes from the density of points the RNN emits, not from curve
fitting, so no bezier smoothing is applied here.
"""
import textwrap

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

import drawing
from sample import sample_strokes

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN = 72  # 1 inch
LINE_HEIGHT = 48
MAX_LINE_CHARS = 70


def wrap_text(text, max_chars=MAX_LINE_CHARS):
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=max_chars) or [""]
        lines.extend(wrapped)
    return lines


def _strokes_to_path_segments(offsets):
    """offsets: (N, 3) array of (dx, dy, eos). Returns list of point-lists,
    each representing one continuous pen-down stroke to draw as straight lines."""
    offsets = offsets.copy()
    offsets[:, :2] *= 1.5
    coords = drawing.offsets_to_coords(offsets)
    coords = drawing.denoise(coords)
    coords[:, :2] = drawing.align(coords[:, :2])

    segments = []
    current = []
    for x, y, eos in coords:
        current.append((x, y))
        if eos == 1:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def render_pdf(text, out_path, bias=0.75, style_prime=True, seed=None):
    lines = wrap_text(text)
    usable_width = PAGE_WIDTH - 2 * MARGIN
    usable_height = PAGE_HEIGHT - 2 * MARGIN
    lines_per_page = max(1, int(usable_height // LINE_HEIGHT))

    c = canvas.Canvas(out_path, pagesize=LETTER)
    c.setLineWidth(1)
    c.setLineJoin(1)  # round
    c.setLineCap(1)  # round

    for page_start in range(0, len(lines), lines_per_page):
        page_lines = lines[page_start:page_start + lines_per_page]
        y = PAGE_HEIGHT - MARGIN - LINE_HEIGHT * 0.75

        for line in page_lines:
            if line:
                offsets = sample_strokes(
                    line, bias=bias, style_prime=style_prime, seed=seed,
                )
                segments = _strokes_to_path_segments(offsets)

                all_x = [p[0] for seg in segments for p in seg]
                x_min, x_max = (min(all_x), max(all_x)) if all_x else (0.0, 0.0)
                line_width = x_max - x_min
                scale = min(1.0, usable_width / line_width) if line_width > 0 else 1.0
                x_offset = MARGIN
                baseline_y = y

                for seg in segments:
                    path = c.beginPath()
                    sx, sy = seg[0]
                    path.moveTo(x_offset + (sx - x_min) * scale, baseline_y + sy * scale)
                    for px, py in seg[1:]:
                        path.lineTo(x_offset + (px - x_min) * scale, baseline_y + py * scale)
                    c.drawPath(path, stroke=1, fill=0)

            y -= LINE_HEIGHT

        c.showPage()

    c.save()
