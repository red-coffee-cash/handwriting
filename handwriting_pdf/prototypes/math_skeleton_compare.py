"""Prototype iteration 2: skeleton-traced math rendering.

Iteration 1 (math_render_compare.py) drew each glyph's *outline contour*
stroked-only. That looks fine for simple Caveat letters (the font's own
strokes are already thin) but produces a hollow cartoon-outline for bold
filled glyphs -- which is exactly what happens for every symbol Caveat
lacks (sigma, integral signs, partial-derivative dels), since those fall
back to a bold default math font.

This iteration fixes that by never tracing outlines at all. Instead it:
  1. Rasterizes the whole math expression at high DPI via matplotlib
     (mathtext, with Caveat as the custom fontset, same as before).
  2. Binarizes and skeletonizes the raster (skimage) to a 1px-wide
     centerline -- the actual path a pen would have drawn.
  3. Walks the skeleton's pixel-adjacency graph (networkx) and emits
     one open polyline per edge between branch/endpoint nodes, plus
     one closed polyline per skeleton loop that has no branch points
     (e.g. the bowl of an 'o' or 'e').
  4. Converts pixel coords to point space and (optionally) applies a
     light jitter -- now safe, since we're wobbling a thin centerline
     rather than the outline of a filled shape.

This is font-agnostic: it treats Caveat-covered and fallback-font
glyphs identically, so sigma/integral/partial symbols come out as the
same kind of single-stroke line as the handwritten letters around them.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
from skimage.morphology import skeletonize

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sample import sample_strokes
from render import _strokes_to_path_segments

FONT_PATH = "/tmp/Caveat.ttf"
DPI = 400  # raster resolution for skeletonization; higher = smoother trace

SNIPPETS = [
    r"$x^2+y^2=r^2$",
    r"$\alpha+\beta=\gamma$",
    r"$\frac{a+b}{c}$",
    r"$\sum_{i=1}^n i = \frac{n(n+1)}{2}$",
    r"$\sqrt{2}\approx 1.414$",
    r"$\iint_D f(x,y)\,dx\,dy$",
    r"$\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}$",
    r"$\frac{\partial^2 f}{\partial x \partial y}$",
]


def rasterize(snippet, size_pt, dpi=DPI):
    """Render a mathtext snippet to a binary numpy mask (True = ink)."""
    fig = plt.figure(figsize=(8, 2))
    fig.patch.set_alpha(0)
    t = fig.text(0.02, 0.5, snippet, fontsize=size_pt, va="center", ha="left")
    fig.canvas.draw()
    bbox = t.get_window_extent(fig.canvas.get_renderer())
    pad = 6
    fig.set_size_inches(
        (bbox.width + 2 * pad) / fig.dpi, (bbox.height + 2 * pad) / fig.dpi
    )
    t.set_position((0, 0))
    t.set_va("bottom")
    t.set_ha("left")
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img = np.asarray(buf, dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    alpha = img[:, :, 3].astype(float)
    ink = alpha > 64  # text drawn at full alpha; background fully transparent
    return ink


def skeleton_to_polylines(mask):
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

    # Walk a simple chain starting at a special node along degree-2 pixels
    # until hitting another special node (or closing a loop).
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


def jitter_polylines(polylines, tremor_amp, wavelength_pt, seed, px_per_pt=1.0):
    """Smooth hand-tremor wobble, not random per-point noise.

    Real handwriting wobble is a slow, smooth wave along the pen's travel,
    not high-frequency jaggedness -- so phase is driven by cumulative arc
    length (in points) rather than by point index. That also keeps the
    wavelength visually consistent regardless of how many skeleton pixels
    a given stroke happens to contain, so tiny glyphs (exponents, indices)
    don't get more wave cycles crammed into them than large ones.
    """
    rng = np.random.default_rng(seed)
    out = []
    for poly in polylines:
        n = len(poly)
        if n < 2:
            out.append(poly)
            continue
        seg_lengths = np.linalg.norm(np.diff(poly, axis=0), axis=1) / px_per_pt
        arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        phase = rng.uniform(0, 2 * np.pi)
        # perpendicular (normal) direction at each point, so the wobble
        # pushes sideways off the stroke's own heading rather than just
        # vertically -- matches how a wavering pen actually moves.
        tangent = np.gradient(poly, axis=0)
        tnorm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tnorm[tnorm == 0] = 1.0
        tangent /= tnorm
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        wobble = tremor_amp * np.sin(2 * np.pi * arc / wavelength_pt + phase)
        jittered = poly + normal * wobble[:, None] * px_per_pt
        out.append(jittered)
    return out


def polylines_to_points(polylines, mask_height, px_per_pt):
    """Flip y (image space is y-down) and rescale pixels -> points."""
    out = []
    for poly in polylines:
        pts = poly.copy()
        pts[:, 1] = mask_height - pts[:, 1]
        pts /= px_per_pt
        out.append(pts)
    return out


def draw_polylines(c, polylines, x0, y0, stroke_width=1.4):
    c.setLineWidth(stroke_width)
    c.setLineJoin(1)
    c.setLineCap(1)
    for poly in polylines:
        if len(poly) < 2:
            continue
        path = c.beginPath()
        path.moveTo(x0 + poly[0, 0], y0 + poly[0, 1])
        for px, py in poly[1:]:
            path.lineTo(x0 + px, y0 + py)
        c.drawPath(path, stroke=1, fill=0)


def draw_rnn_sample(c, text, x0, y0, scale=1.0, seed=0):
    offsets = sample_strokes(text, bias=0.75, style_prime=True, seed=seed)
    segments = _strokes_to_path_segments(offsets)
    c.setLineWidth(1)
    for seg in segments:
        path = c.beginPath()
        sx, sy = seg[0]
        path.moveTo(x0 + sx * scale, y0 + sy * scale)
        for px, py in seg[1:]:
            path.lineTo(x0 + px * scale, y0 + py * scale)
        c.drawPath(path, stroke=1, fill=0)


def render_snippet(c, snippet, x0, y0, target_pt_height=24, jitter=None):
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.rm"] = "Caveat"
    matplotlib.rcParams["mathtext.it"] = "Caveat:italic"
    matplotlib.rcParams["mathtext.bf"] = "Caveat:bold"
    matplotlib.rcParams["mathtext.cal"] = "Caveat"

    mask = rasterize(snippet, size_pt=40)
    polylines = skeleton_to_polylines(mask)

    px_per_pt = mask.shape[0] / (target_pt_height * 2.2)  # rough; rescale below
    # Rescale so the mask's pixel height maps to a sane on-page size directly
    # by fixing px-per-point from the desired cap height instead.
    h_px = mask.shape[0]
    px_per_pt = h_px / (target_pt_height * 2.6)

    pts_polylines = polylines_to_points(polylines, h_px, px_per_pt)
    if jitter:
        tremor_amp, wavelength_pt = jitter
        pts_polylines = jitter_polylines(
            pts_polylines, tremor_amp, wavelength_pt, seed=hash(snippet) % 1000
        )
    draw_polylines(c, pts_polylines, x0, y0)


def main():
    fm.fontManager.addfont(FONT_PATH)
    out_path = os.path.join(os.path.dirname(__file__), "prototype_out", "math_skeleton.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER
    margin = 50
    row_h = 75

    variants = [
        ("D1: no jitter, thicker stroke", None),
        ("D2: subtle smooth wave (amp=0.35pt, wavelength=36pt)", (0.35, 36)),
        ("D3: medium smooth wave (amp=0.6pt, wavelength=28pt)", (0.6, 28)),
    ]

    for label, jitter in variants:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, height - 35, "Math rendering -- skeleton-traced (iteration 2)")
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(margin, height - 60, label)
        for si, snippet in enumerate(SNIPPETS):
            y0 = height - 95 - si * row_h
            render_snippet(c, snippet, margin + 10, y0, jitter=jitter)
        c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - 35, "Reference: real RNN handwriting (for scale/style comparison)")
    draw_rnn_sample(c, "x squared plus y squared equals r squared", margin, height - 100, scale=1.0, seed=42)
    draw_rnn_sample(c, "partial f over partial x plus partial f over partial y", margin, height - 160, scale=1.0, seed=7)
    c.showPage()

    c.save()
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
