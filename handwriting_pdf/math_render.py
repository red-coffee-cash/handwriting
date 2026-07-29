"""Render LaTeX/mathtext snippets as hand-sketched vector strokes.

The handwriting RNN's alphabet (drawing.alphabet) has no math operators or
Greek letters, so math is rendered through a completely separate path and
composited alongside RNN-generated text strokes by render_box.py.

Approach (selected after a prototype comparison -- see
prototypes/math_skeleton_compare.py -- with the user):
  1. Render the mathtext expression to a high-resolution raster, using a
     handwriting font (Caveat) as the custom mathtext fontset. Most glyphs
     (letters, digits, basic operators) come out in that handwriting font;
     mathtext gracefully falls back to its default font only for glyphs
     Caveat lacks (Greek letters, big-operator glyphs like sigma/integral).
  2. Skeletonize the raster to a 1px-wide centerline. This is the key fix
     over an earlier outline-stroking attempt: stroking a filled glyph's
     *outline* leaves bold/fallback-font glyphs looking like hollow
     cartoon outlines, whereas the skeleton centerline is a genuine
     single-pen-stroke path regardless of which font supplied the glyph.
  3. Trace the skeleton's pixel-adjacency graph into open polylines (one
     per edge between branch/endpoint nodes, plus one per stroke loop with
     no branch points, e.g. the bowl of an 'o').
  4. Apply a smooth, low-amplitude, long-period sine wobble along each
     stroke's local normal direction, parametrized by cumulative arc
     length rather than point index -- a slow natural hand-drift rather
     than jittery per-point noise, so it stays legible at any glyph size.

Output is a list of point-lists (one per stroke) in PDF points, same data
shape as drawing.strokes_to_path_segments, so callers can draw both with
the same code path.
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
from skimage.morphology import skeletonize

FONT_NAME = "Caveat"
FONT_PATH = os.path.join(os.path.dirname(__file__), "weights", "Caveat.ttf")
RASTER_DPI = 400

# Locked-in tremor parameters (prototype variant "D3: medium smooth wave").
TREMOR_AMP_PT = 0.6
TREMOR_WAVELENGTH_PT = 28.0
STROKE_WIDTH_PT = 1.4

_font_registered = False


def _ensure_font_registered():
    global _font_registered
    if _font_registered:
        return
    fm.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = FONT_NAME
    matplotlib.rcParams["mathtext.it"] = f"{FONT_NAME}:italic"
    matplotlib.rcParams["mathtext.bf"] = f"{FONT_NAME}:bold"
    matplotlib.rcParams["mathtext.cal"] = FONT_NAME
    _font_registered = True


def _rasterize_to_mask(fig, t, dpi, ismath):
    """Tighten `fig` around text artist `t`, render, and return (mask,
    px_per_pt, baseline_from_bottom_px). Shared tail of _rasterize and
    _rasterize_plain.

    The text is re-anchored with va="baseline" at a known figure position,
    so the baseline row in the rendered image is exact by construction --
    baseline_from_bottom_px is its distance above the image bottom. (Font
    metrics alone can't be trusted for this: get_window_extent for mathtext
    returns a full em box, not the ascent+descent layout box, so "bbox
    bottom = baseline - descent" does not hold.) The metric descent is only
    used to reserve enough room below the baseline for descenders.
    """
    renderer = fig.canvas.get_renderer()
    bbox = t.get_window_extent(renderer)
    try:
        _, _, descent_px = renderer.get_text_width_height_descent(
            t.get_text(), t.get_fontproperties(), ismath=ismath)
    except ValueError:
        # The artist itself decides math-ness per string (an odd number of
        # unescaped $ renders literally), so a forced ismath=True parse can
        # fail where draw() succeeded. Fall back to plain-text metrics
        # rather than letting model garbage crash the whole render.
        _, _, descent_px = renderer.get_text_width_height_descent(
            t.get_text(), t.get_fontproperties(), ismath=False)
    pad = 8
    fig_w_px = bbox.width + 2 * pad
    fig_h_px = bbox.height + 2 * pad
    fig.set_size_inches(fig_w_px / fig.dpi, fig_h_px / fig.dpi)
    x_frac = pad / fig_w_px
    y_frac = (descent_px + pad) / fig_h_px
    t.set_position((x_frac, y_frac))
    t.set_va("baseline")
    t.set_ha("left")
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    alpha = img[:, :, 3].astype(float)
    mask = alpha > 64
    px_per_pt = dpi / 72.0
    # The canvas may round the requested figure size to whole pixels; the
    # baseline lands at y_frac of the *actual* height.
    baseline_from_bottom_px = y_frac * h
    return mask, px_per_pt, baseline_from_bottom_px


def _rasterize_plain(snippet, size_pt, dpi=RASTER_DPI):
    """Fallback for malformed mathtext: render the literal characters in the
    Caveat handwriting font with math parsing disabled, so unparseable LaTeX
    still produces readable strokes instead of an exception."""
    literal = snippet.strip()
    if len(literal) >= 2 and literal.startswith("$") and literal.endswith("$"):
        literal = literal[1:-1]
    fig = plt.figure(figsize=(8, 2), dpi=dpi)
    fig.patch.set_alpha(0)
    fp = fm.FontProperties(fname=FONT_PATH)
    t = fig.text(0.02, 0.5, literal, fontsize=size_pt, va="center", ha="left",
                 fontproperties=fp, parse_math=False)
    fig.canvas.draw()
    return _rasterize_to_mask(fig, t, dpi, ismath=False)


def _rasterize(snippet, size_pt, dpi=RASTER_DPI):
    """Render a mathtext snippet to a binary numpy mask (True = ink) and
    return (mask, px_per_pt, baseline_from_bottom_px) where px_per_pt
    converts mask pixel distances to PDF points at the given font size and
    baseline_from_bottom_px locates the text baseline above the mask
    bottom."""
    fig = plt.figure(figsize=(8, 2), dpi=dpi)
    fig.patch.set_alpha(0)
    t = fig.text(0.02, 0.5, snippet, fontsize=size_pt, va="center", ha="left")
    try:
        fig.canvas.draw()
    except Exception:
        # Invalid mathtext/LaTeX from the model (unbalanced braces,
        # unsupported macros, ...) raises during layout. Re-render the raw
        # content as plain handwriting-font text so it stays legible instead
        # of crashing the whole generate request.
        plt.close(fig)
        return _rasterize_plain(snippet, size_pt, dpi=dpi)
    return _rasterize_to_mask(fig, t, dpi, ismath=True)


def _skeleton_to_polylines(mask):
    """Trace a skeletonized boolean mask into a list of (N,2) point arrays
    in (x, y) pixel coordinates, y measured downward (image convention)."""
    skel = skeletonize(mask)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:
        return []
    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    g = nx.Graph()
    g.add_nodes_from(pixel_set)
    neighbors8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for (x, y) in pixel_set:
        for dx, dy in neighbors8:
            nb = (x + dx, y + dy)
            if nb in pixel_set:
                g.add_edge((x, y), nb)

    polylines = []
    visited_edges = set()

    def edge_key(a, b):
        return (a, b) if a <= b else (b, a)

    special = [n for n in g.nodes if g.degree(n) != 2]

    def walk(start, nxt):
        path = [start, nxt]
        prev, cur = start, nxt
        while g.degree(cur) == 2 and cur != start:
            nbrs = [n for n in g.neighbors(cur) if n != prev]
            if not nbrs:
                break
            prev, cur = cur, nbrs[0]
            path.append(cur)
        return path

    for n in special:
        for nb in g.neighbors(n):
            ek = edge_key(n, nb)
            if ek in visited_edges:
                continue
            path = walk(n, nb)
            for a, b in zip(path[:-1], path[1:]):
                visited_edges.add(edge_key(a, b))
            polylines.append(np.array(path, dtype=float))

    # Remaining edges belong to pure loops (no special nodes, e.g. an 'o').
    seen_in_loop = set()
    for comp in nx.connected_components(g):
        comp_edges = [
            edge_key(a, b) for a, b in g.subgraph(comp).edges if edge_key(a, b) not in visited_edges
        ]
        if not comp_edges:
            continue
        start = comp_edges[0][0]
        if start in seen_in_loop:
            continue
        cur = start
        prev = None
        loop = [cur]
        while True:
            nbrs = [n for n in g.neighbors(cur) if n != prev and edge_key(cur, n) not in visited_edges]
            if not nbrs:
                break
            nxt = nbrs[0]
            visited_edges.add(edge_key(cur, nxt))
            loop.append(nxt)
            seen_in_loop.add(nxt)
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(loop) > 2:
            polylines.append(np.array(loop, dtype=float))

    return polylines


def _jitter_polylines(polylines, tremor_amp, wavelength_pt, seed):
    """Smooth hand-tremor wobble along each stroke's local normal,
    parametrized by cumulative arc length (points already in pt space)."""
    rng = np.random.default_rng(seed)
    out = []
    for poly in polylines:
        n = len(poly)
        if n < 2:
            out.append(poly)
            continue
        seg_lengths = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        phase = rng.uniform(0, 2 * np.pi)
        tangent = np.gradient(poly, axis=0)
        tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tnorm[tnorm == 0] = 1.0
        tangent /= tnorm
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        wobble = tremor_amp * np.sin(2 * np.pi * arc / wavelength_pt + phase)
        out.append(poly + normal * wobble[:, None])
    return out


def _polylines_to_points(polylines, mask_height_px, px_per_pt, baseline_from_bottom_px):
    """Flip y (image space is y-down), shift so y=0 is the text *baseline*
    (which sits baseline_from_bottom_px above the mask bottom), and rescale
    pixels -> PDF points. Descenders come out with negative y."""
    out = []
    for poly in polylines:
        pts = poly.copy()
        pts[:, 1] = (mask_height_px - pts[:, 1]) - baseline_from_bottom_px
        pts /= px_per_pt
        out.append(pts)
    return out


# --- LaTeX -> mathtext normalization -------------------------------------
# matplotlib's mathtext supports a subset of LaTeX; these are the common
# macros models emit that it rejects, mapped to supported spellings.
# Patterns use \b so e.g. \le never matches inside \left or \leq.
_MACRO_FIXUPS = [
    (re.compile(r"\\le\b"), r"\\leq"),
    (re.compile(r"\\ge\b"), r"\\geq"),
    (re.compile(r"\\ne\b"), r"\\neq"),
    (re.compile(r"\\iff\b"), r"\\Leftrightarrow"),
    (re.compile(r"\\implies\b"), r"\\Rightarrow"),
    (re.compile(r"\\impliedby\b"), r"\\Leftarrow"),
    (re.compile(r"\\tfrac\b"), r"\\frac"),
    (re.compile(r"\\displaystyle\b"), ""),
    (re.compile(r"\\pmod\s*\{([^{}]*)\}"), r"\\ (\\mathrm{mod}\\ \1)"),
    (re.compile(r"\\bmod\b"), r"\\ \\mathrm{mod}\\ "),
]


def _normalize_mathtext(s):
    for pat, rep in _MACRO_FIXUPS:
        s = pat.sub(rep, s)
    return s


# --- matrix / cases environments ------------------------------------------
# mathtext has no \begin{...} support at all, so matrix-family and cases
# environments are composited here: each cell rendered through the normal
# pipeline, laid out on a grid, delimiters drawn scaled to the stack.
_MATRIX_ENVS = {
    "pmatrix": ("(", ")"),
    "bmatrix": ("[", "]"),
    "Bmatrix": ("{", "}"),
    "vmatrix": ("|", "|"),
    "Vmatrix": ("|", "|"),
    "matrix": (None, None),
    "smallmatrix": (None, None),
    "cases": ("{", None),
}
_MATRIX_ENV_RE = re.compile(
    r"\\begin\{(pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|smallmatrix"
    r"|cases)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

# Fraction of the font size the math axis sits above the baseline; grids
# (like mathtext's own fractions) are centered vertically on this axis.
_AXIS_FRACTION = 0.26


def _split_nested(s, sep):
    """Split `s` on `sep` occurrences that are not inside a nested
    \\begin{...}\\end{...} block."""
    parts, buf, depth, i = [], [], 0, 0
    while i < len(s):
        if s.startswith("\\begin", i):
            depth += 1
            buf.append("\\begin")
            i += 6
            continue
        if s.startswith("\\end", i):
            depth = max(0, depth - 1)
            buf.append("\\end")
            i += 4
            continue
        if depth == 0 and s.startswith(sep, i):
            parts.append("".join(buf))
            buf = []
            i += len(sep)
            continue
        buf.append(s[i])
        i += 1
    parts.append("".join(buf))
    return parts


def _snippet_strokes(inner, size_pt):
    """Render bare mathtext content (no $ delimiters) to baseline-anchored
    stroke arrays via the raster/skeleton pipeline. No jitter."""
    mask, px_per_pt, baseline_px = _rasterize(f"${inner}$", size_pt=size_pt)
    polylines = _skeleton_to_polylines(mask)
    return _polylines_to_points(polylines, mask.shape[0], px_per_pt, baseline_px)


def _strokes_extent(strokes):
    """(min_x, min_y, max_x, max_y) over stroke arrays, or None if empty."""
    if not strokes:
        return None
    all_pts = np.concatenate(strokes, axis=0)
    return (float(all_pts[:, 0].min()), float(all_pts[:, 1].min()),
            float(all_pts[:, 0].max()), float(all_pts[:, 1].max()))


def _shift(strokes, dx, dy):
    return [s + np.array([dx, dy]) for s in strokes]


def _delimiter_strokes(ch, font_size_pt, target_height, center_y):
    """Render a delimiter glyph in the plain handwriting font and scale it
    (uniformly in y, proportionally in x) to span `target_height`, centered
    vertically on `center_y`. Returns (strokes, width)."""
    mask, px_per_pt, baseline_px = _rasterize_plain(ch, font_size_pt)
    polylines = _skeleton_to_polylines(mask)
    strokes = _polylines_to_points(polylines, mask.shape[0], px_per_pt, baseline_px)
    ext = _strokes_extent(strokes)
    if ext is None or ext[3] - ext[1] <= 0:
        return [], 0.0
    x0, y0, x1, y1 = ext
    scale = target_height / (y1 - y0)
    # Stretch mostly vertically; let width grow only mildly so a tall "("
    # stays slender like a hand-drawn big paren.
    x_scale = min(scale, 1.0 + 0.25 * (scale - 1.0)) if scale > 1.0 else scale
    mid_y = (y0 + y1) / 2.0
    out = []
    for s in strokes:
        pts = s.copy()
        pts[:, 0] = (pts[:, 0] - x0) * x_scale
        pts[:, 1] = (pts[:, 1] - mid_y) * scale + center_y
        out.append(pts)
    return out, (x1 - x0) * x_scale


def _render_env_grid(env, body, font_size_pt):
    """Composite a matrix-family or cases environment: render each cell,
    lay the cells out on a grid centered on the math axis, and add scaled
    delimiters. Returns baseline-anchored strokes (possibly empty)."""
    cell_size = font_size_pt * 0.9
    rows = [r for r in (_split_nested(body, "\\\\")) if r.strip()]
    grid = []
    for row in rows:
        cells = [c.strip() for c in _split_nested(row, "&")]
        rendered = []
        for cell in cells:
            strokes = _snippet_strokes(cell, cell_size) if cell else []
            rendered.append((strokes, _strokes_extent(strokes)))
        grid.append(rendered)
    if not grid:
        return []

    n_cols = max(len(r) for r in grid)
    col_w = [0.0] * n_cols
    row_asc, row_desc = [], []
    min_asc = 0.45 * cell_size  # empty/short rows still take vertical room
    for cells in grid:
        asc, desc = min_asc, 0.0
        for j, (_, ext) in enumerate(cells):
            if ext is None:
                continue
            col_w[j] = max(col_w[j], ext[2] - ext[0])
            asc = max(asc, ext[3])
            desc = max(desc, -ext[1])
        row_asc.append(asc)
        row_desc.append(desc)

    col_gap = 0.55 * font_size_pt
    row_gap = 0.4 * font_size_pt
    grid_w = sum(col_w) + col_gap * max(0, n_cols - 1)
    grid_h = sum(a + d for a, d in zip(row_asc, row_desc)) \
        + row_gap * max(0, len(grid) - 1)
    axis_y = _AXIS_FRACTION * font_size_pt
    top_y = axis_y + grid_h / 2.0

    strokes = []
    y_cursor = top_y
    for i, cells in enumerate(grid):
        row_baseline = y_cursor - row_asc[i]
        for j, (cell_strokes, ext) in enumerate(cells):
            if ext is None:
                continue
            x_off = sum(col_w[:j]) + col_gap * j
            # center the cell inside its column
            x_off += (col_w[j] - (ext[2] - ext[0])) / 2.0 - ext[0]
            strokes.extend(_shift(cell_strokes, x_off, row_baseline))
        y_cursor = row_baseline - row_desc[i] - row_gap

    left_ch, right_ch = _MATRIX_ENVS[env]
    delim_h = grid_h * 1.1
    delim_gap = 0.18 * font_size_pt
    out = []
    x_cursor = 0.0
    if left_ch:
        d_strokes, d_w = _delimiter_strokes(left_ch, font_size_pt, delim_h, axis_y)
        out.extend(_shift(d_strokes, x_cursor, 0.0))
        x_cursor += d_w + delim_gap
    out.extend(_shift(strokes, x_cursor, 0.0))
    x_cursor += grid_w
    if right_ch:
        x_cursor += delim_gap
        d_strokes, d_w = _delimiter_strokes(right_ch, font_size_pt, delim_h, axis_y)
        out.extend(_shift(d_strokes, x_cursor, 0.0))
    return out


def _compose_snippet(inner, font_size_pt):
    """Render mathtext content, compositing any matrix/cases environments
    it contains alongside the ordinary mathtext segments on one baseline."""
    segments = []
    pos = 0
    for m in _MATRIX_ENV_RE.finditer(inner):
        if m.start() > pos:
            segments.append(("math", inner[pos:m.start()]))
        segments.append(("env", m.group(1), m.group(2)))
        pos = m.end()
    if pos < len(inner):
        segments.append(("math", inner[pos:]))

    if len(segments) == 1 and segments[0][0] == "math":
        return _snippet_strokes(inner, font_size_pt)

    strokes = []
    x_cursor = 0.0
    seg_gap = 0.35 * font_size_pt
    for seg in segments:
        if seg[0] == "math":
            if not seg[1].strip():
                continue
            seg_strokes = _snippet_strokes(seg[1].strip(), font_size_pt)
        else:
            seg_strokes = _render_env_grid(seg[1], seg[2], font_size_pt)
        ext = _strokes_extent(seg_strokes)
        if ext is None:
            continue
        strokes.extend(_shift(seg_strokes, x_cursor - ext[0], 0.0))
        x_cursor += (ext[2] - ext[0]) + seg_gap
    return strokes


def render_math_strokes(snippet, font_size_pt=24, jitter=True, seed=0):
    """Render a mathtext snippet to hand-sketched strokes.

    `snippet` is the bare LaTeX/mathtext content (e.g. r"\frac{a}{b}"),
    matching what gemma_client.split_runs hands back for math runs --
    surrounding $ delimiters are added automatically if not already
    present, since matplotlib only parses text between them as mathtext.

    Returns (strokes, width_pt, height_pt):
      strokes    -- list of (N, 2) point arrays in PDF points, origin at
                    the snippet's baseline-left, y-up, descenders negative
                    (matches the baseline convention render_box.py uses to
                    mix these with RNN handwriting strokes on one line).
      width_pt, height_pt -- bounding size, for layout/scaling.
    """
    _ensure_font_registered()
    inner = snippet.strip()
    if len(inner) >= 2 and inner.startswith("$") and inner.endswith("$"):
        inner = inner[1:-1]
    # Escape any remaining $ (e.g. currency "$5" routed here because the
    # RNN alphabet has no $ glyph) so the wrapped string always has exactly
    # one balanced $...$ pair -- an odd $ count makes matplotlib render the
    # string literally, delimiters and all.
    inner = inner.replace("\\$", "$").replace("$", "\\$")
    inner = _normalize_mathtext(inner).strip()
    if not inner:
        return [], 0.0, 0.0
    strokes = _compose_snippet(inner, font_size_pt)
    if jitter:
        strokes = _jitter_polylines(strokes, TREMOR_AMP_PT, TREMOR_WAVELENGTH_PT, seed=seed)

    if strokes:
        all_pts = np.concatenate(strokes, axis=0)
        width_pt = float(all_pts[:, 0].max() - all_pts[:, 0].min())
        height_pt = float(all_pts[:, 1].max() - all_pts[:, 1].min())
    else:
        width_pt = height_pt = 0.0
    return strokes, width_pt, height_pt
