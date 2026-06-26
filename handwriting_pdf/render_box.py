"""Lay out an answer's text+math runs into a confirmed on-page box.

Takes the {"runs": [...]} produced by gemma_client.split_runs and turns it
into a list of stroke point-lists positioned in absolute PDF-page
coordinates (PyMuPDF convention: origin top-left, y increasing downward),
ready for pdf_compose.py to draw directly onto the page.

Text runs are sampled from the handwriting RNN (sample.sample_strokes);
math runs go through math_render.render_math_strokes. Both return strokes
in the same "list of point-lists" shape with a shared convention -- origin
at the run's own baseline-left, y increasing *upward* -- so they can be
concatenated along a line and then flipped into page coordinates together.

Fitting strategy: try progressively smaller line heights (a "shrink-tier"
search) until the wrapped content fits the box both horizontally and
vertically; if even the smallest tier doesn't fit vertically, apply one
uniform extra scale-down as a last resort and report a warning rather than
silently clipping content outside the box.
"""
import numpy as np

import drawing
import math_render
from sample import sample_strokes

LINE_HEIGHT_TIERS = [28, 22, 18, 14]
BOX_PADDING = 4
WORD_GAP_FACTOR = 0.35  # gap between tokens on a line, as a fraction of line height
CHAR_WIDTH_FACTOR = 0.55  # rough avg char width as a fraction of line height; wrap estimate only

_math_width_cache = {}


def _tokenize_runs(runs):
    """Flatten text/math runs into a sequence of ("text", word) / ("math",
    value) tokens, splitting text runs on whitespace so wrapping can break
    between words while math runs stay atomic."""
    tokens = []
    for run in runs:
        if run["kind"] == "text":
            for word in run["value"].split():
                tokens.append(("text", word))
        else:
            tokens.append(("math", run["value"]))
    return tokens


def _estimate_math_width(value, line_height):
    cached = _math_width_cache.get(value)
    if cached is None:
        _, w, _ = math_render.render_math_strokes(value, font_size_pt=24, jitter=False)
        cached = w
        _math_width_cache[value] = cached
    return cached * (line_height / 24.0)


def _estimate_token_width(token, line_height):
    kind, value = token
    if kind == "text":
        return len(value) * line_height * CHAR_WIDTH_FACTOR
    return _estimate_math_width(value, line_height)


def _wrap_tokens(tokens, line_height, usable_width):
    lines, current, current_width = [], [], 0.0
    gap = line_height * WORD_GAP_FACTOR
    for token in tokens:
        w = _estimate_token_width(token, line_height)
        added = w + (gap if current else 0.0)
        if current and current_width + added > usable_width:
            lines.append(current)
            current, current_width = [], 0.0
            added = w
        current.append(token)
        current_width += added
    if current:
        lines.append(current)
    return lines


def _group_line_tokens(line_tokens):
    """Merge consecutive text tokens into single groups (one sample_strokes
    call each, for natural cursive joins), keeping math tokens separate."""
    groups = []
    buf = []
    for kind, value in line_tokens:
        if kind == "text":
            buf.append(value)
        else:
            if buf:
                groups.append(("text", " ".join(buf)))
                buf = []
            groups.append(("math", value))
    if buf:
        groups.append(("text", " ".join(buf)))
    return groups


def _render_line(line_tokens, line_height, bias, style_prime, seed):
    """Render one wrapped line. Returns (strokes, width, height) with
    strokes positioned along a shared baseline at y=0, y-up, x starting
    at 0."""
    groups = _group_line_tokens(line_tokens)
    gap = line_height * WORD_GAP_FACTOR
    strokes = []
    x_cursor = 0.0
    max_y = 0.0
    min_y = 0.0
    seed_counter = 0
    for kind, value in groups:
        if kind == "text":
            offsets = sample_strokes(
                value, bias=bias, style_prime=style_prime,
                seed=None if seed is None else seed + seed_counter,
            )
            seed_counter += 1
            segments = drawing.strokes_to_path_segments(offsets)
            # Original RNN output is tuned for ~LINE_HEIGHT=48pt rendering
            # (see render.py); rescale to this tier's line height.
            rnn_scale = line_height / 48.0
            group_pts = [np.asarray(seg, dtype=float) * rnn_scale for seg in segments]
        else:
            math_seed = 0 if seed is None else seed + seed_counter
            seed_counter += 1
            group_strokes, _, _ = math_render.render_math_strokes(
                value, font_size_pt=line_height * 0.85, jitter=True, seed=math_seed,
            )
            group_pts = [np.asarray(s, dtype=float) for s in group_strokes]

        if not group_pts:
            continue
        all_pts = np.concatenate(group_pts, axis=0)
        group_min_x = float(all_pts[:, 0].min())
        group_min_y = float(all_pts[:, 1].min())
        group_max_y = float(all_pts[:, 1].max())
        offset_x = x_cursor - group_min_x
        for pts in group_pts:
            shifted = pts.copy()
            shifted[:, 0] += offset_x
            strokes.append(shifted)
        group_width = float(all_pts[:, 0].max()) - group_min_x
        x_cursor += group_width + gap
        min_y = min(min_y, group_min_y)
        max_y = max(max_y, group_max_y)

    width = max(0.0, x_cursor - gap) if groups else 0.0
    height = max_y - min_y
    return strokes, width, height


def render_answer_in_box(answer_runs, box, bias=0.75, style_prime=True, seed=None):
    """Render an answer's runs to fit inside `box`
    ({"x0","y0","x1","y1"} in absolute PDF-page points, PyMuPDF convention).

    Returns (strokes, warning) where strokes is a list of {"points": [[x,
    y], ...], "source": "generated"} dicts in absolute page coordinates,
    and warning is None or a string describing a last-resort fallback that
    was applied (so the caller can surface it instead of silently clipping).
    """
    tokens = _tokenize_runs(answer_runs)
    if not tokens:
        return [], None

    usable_width = max(1.0, (box["x1"] - box["x0"]) - 2 * BOX_PADDING)
    usable_height = max(1.0, (box["y1"] - box["y0"]) - 2 * BOX_PADDING)

    chosen = None
    for tier_index, line_height in enumerate(LINE_HEIGHT_TIERS):
        lines = _wrap_tokens(tokens, line_height, usable_width)
        rendered_lines = [
            _render_line(line, line_height, bias, style_prime,
                         seed=None if seed is None else seed + 1000 * i)
            for i, line in enumerate(lines)
        ]
        total_height = line_height * len(rendered_lines)
        if total_height <= usable_height or tier_index == len(LINE_HEIGHT_TIERS) - 1:
            chosen = (line_height, rendered_lines, total_height)
            break

    line_height, rendered_lines, total_height = chosen
    warning = None
    extra_scale = 1.0
    if total_height > usable_height:
        extra_scale = usable_height / total_height
        warning = (
            f"Answer didn't fit box even at the smallest line height "
            f"({LINE_HEIGHT_TIERS[-1]}pt); scaled down by {extra_scale:.2f}x as a last resort."
        )

    strokes = []
    for i, (line_strokes, line_width, _line_h) in enumerate(rendered_lines):
        line_scale = min(1.0, usable_width / line_width) if line_width > 0 else 1.0
        combined_scale = line_scale * extra_scale
        baseline_y = box["y0"] + BOX_PADDING + (i + 1) * line_height * extra_scale - line_height * 0.25 * extra_scale
        for pts in line_strokes:
            abs_x = box["x0"] + BOX_PADDING + pts[:, 0] * combined_scale
            abs_y = baseline_y - pts[:, 1] * combined_scale
            strokes.append({
                "points": np.stack([abs_x, abs_y], axis=1).tolist(),
                "source": "generated",
            })

    return strokes, warning
