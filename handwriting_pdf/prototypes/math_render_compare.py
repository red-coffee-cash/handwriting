"""Prototype: compare math-rendering strategies for the worksheet pipeline.

The handwriting RNN's alphabet (drawing.alphabet) has no math operators or
Greek letters, so math must be rendered through a different path. This
script generates a side-by-side comparison PDF of candidate strategies, to
be reviewed by hand before any of them are wired into the real pipeline.

Candidates:
  A: matplotlib mathtext's default font (DejaVu Sans), vector outline
     extracted via TextPath, vertices jittered to look hand-sketched.
  B: same, but using a registered handwriting font (Caveat) as the custom
     mathtext fontset, so plain letters/digits/most operators render in a
     genuine handwriting font; mathtext falls back to its default font only
     for glyphs Caveat lacks (Greek letters, big-operator glyphs).

Both candidates produce the same data shape as the RNN's own output --
a list of point-lists, one per pen-stroke -- so whichever is chosen slots
into the existing straight-line-segment rendering pipeline unchanged.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sample import sample_strokes
from render import _strokes_to_path_segments

FONT_PATH = "/tmp/Caveat.ttf"
SNIPPETS = [
    r"$x^2+y^2=r^2$",
    r"$\alpha+\beta=\gamma$",
    r"$\frac{a+b}{c}$",
    r"$\sum_{i=1}^n i = \frac{n(n+1)}{2}$",
    r"$\sqrt{2}\approx 1.414$",
]


def polygons_for(snippet, fontset, size=24):
    matplotlib.rcParams["mathtext.fontset"] = fontset
    if fontset == "custom":
        matplotlib.rcParams["mathtext.rm"] = "Caveat"
        matplotlib.rcParams["mathtext.it"] = "Caveat:italic"
        matplotlib.rcParams["mathtext.bf"] = "Caveat:bold"
        matplotlib.rcParams["mathtext.cal"] = "Caveat"
    tp = TextPath((0, 0), snippet, size=size, prop=FontProperties())
    return tp.to_polygons()


def jitter_polygons(polygons, point_sigma, tremor_amp, tremor_freq, seed):
    rng = np.random.default_rng(seed)
    out = []
    for poly in polygons:
        poly = np.asarray(poly, dtype=float)
        n = len(poly)
        t = np.linspace(0, 1, n)
        tremor = tremor_amp * np.sin(2 * np.pi * tremor_freq * t + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(scale=point_sigma, size=(n, 2))
        jittered = poly.copy()
        jittered[:, 1] += tremor
        jittered += noise
        out.append(jittered)
    return out


def draw_polygons(c, polygons, x0, y0, scale, stroke_width=0.6):
    c.setLineWidth(stroke_width)
    for poly in polygons:
        if len(poly) < 2:
            continue
        path = c.beginPath()
        path.moveTo(x0 + poly[0, 0] * scale, y0 + poly[0, 1] * scale)
        for px, py in poly[1:]:
            path.lineTo(x0 + px * scale, y0 + py * scale)
        path.close()
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


def main():
    fm.fontManager.addfont(FONT_PATH)
    out_path = os.path.join(os.path.dirname(__file__), "prototype_out", "math_compare.pdf")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    variants = [
        ("A1: DejaVu, jitter=0 (baseline, no sketch)", "dejavusans", 0.0, 0.0),
        ("A2: DejaVu, light jitter", "dejavusans", 0.4, 3.0),
        ("A3: DejaVu, heavy jitter", "dejavusans", 1.1, 6.0),
        ("B1: Caveat font, jitter=0", "custom", 0.0, 0.0),
        ("B2: Caveat font, light jitter", "custom", 0.4, 3.0),
        ("B3: Caveat font, heavy jitter", "custom", 1.1, 6.0),
    ]

    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER
    margin = 50
    row_h = 70

    # One variant per page so nothing runs off the bottom.
    for label, fontset, sigma, tremor in variants:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, height - 35, "Math rendering candidates -- for sign-off")
        c.setFont("Helvetica-Oblique", 11)
        c.drawString(margin, height - 60, label)
        for si, snippet in enumerate(SNIPPETS):
            y0 = height - 90 - si * row_h
            polys = polygons_for(snippet, fontset)
            if sigma or tremor:
                polys = jitter_polygons(polys, sigma, tremor, tremor_freq=2.5, seed=si)
            draw_polygons(c, polys, margin + 10, y0, scale=1.0)
        c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - 35, "Reference: real RNN handwriting (for scale/style comparison)")
    draw_rnn_sample(c, "x squared plus y squared equals r squared", margin, height - 100, scale=1.0, seed=42)
    draw_rnn_sample(c, "the sum of i equals n times n plus one over two", margin, height - 160, scale=1.0, seed=7)
    c.showPage()
    c.save()
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
