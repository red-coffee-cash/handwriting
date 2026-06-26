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


def _rasterize(snippet, size_pt, dpi=RASTER_DPI):
    """Render a mathtext snippet to a binary numpy mask (True = ink) and
    return (mask, px_per_pt) where px_per_pt converts mask pixel distances
    to PDF points at the given font size."""
    fig = plt.figure(figsize=(8, 2), dpi=dpi)
    fig.patch.set_alpha(0)
    t = fig.text(0.02, 0.5, snippet, fontsize=size_pt, va="center", ha="left")
    fig.canvas.draw()
    bbox = t.get_window_extent(fig.canvas.get_renderer())
    pad = 6
    fig.set_size_inches((bbox.width + 2 * pad) / fig.dpi, (bbox.height + 2 * pad) / fig.dpi)
    t.set_position((0, 0))
    t.set_va("bottom")
    t.set_ha("left")
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    alpha = img[:, :, 3].astype(float)
    mask = alpha > 64
    # px_per_pt: pixels per PDF point at this rasterization's font size.
    px_per_pt = dpi / 72.0
    return mask, px_per_pt


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


def _polylines_to_points(polylines, mask_height_px, px_per_pt):
    """Flip y (image space is y-down) and rescale pixels -> PDF points."""
    out = []
    for poly in polylines:
        pts = poly.copy()
        pts[:, 1] = mask_height_px - pts[:, 1]
        pts /= px_per_pt
        out.append(pts)
    return out


def render_math_strokes(snippet, font_size_pt=24, jitter=True, seed=0):
    """Render a $...$-delimited mathtext snippet to hand-sketched strokes.

    Returns (strokes, width_pt, height_pt):
      strokes    -- list of (N, 2) point arrays in PDF points, origin at
                    the snippet's bottom-left, y-up (matches the RNN's own
                    stroke convention so render_box.py can mix the two).
      width_pt, height_pt -- bounding size, for layout/scaling.
    """
    _ensure_font_registered()
    mask, px_per_pt = _rasterize(snippet, size_pt=font_size_pt)
    polylines = _skeleton_to_polylines(mask)
    h_px = mask.shape[0]
    strokes = _polylines_to_points(polylines, h_px, px_per_pt)
    if jitter:
        strokes = _jitter_polylines(strokes, TREMOR_AMP_PT, TREMOR_WAVELENGTH_PT, seed=seed)

    if strokes:
        all_pts = np.concatenate(strokes, axis=0)
        width_pt = float(all_pts[:, 0].max())
        height_pt = float(all_pts[:, 1].max())
    else:
        width_pt = height_pt = 0.0
    return strokes, width_pt, height_pt
