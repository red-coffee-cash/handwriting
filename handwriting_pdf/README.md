# handwriting_pdf

Turns plain text into a PDF of generated, "authentic looking" handwriting.

## How it works

This is a PyTorch port of the handwriting synthesis network from Alex
Graves, ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/abs/1308.0850)
(section 5): a 3-layer LSTM stack with a Gaussian-window soft-attention
mechanism over the input character sequence, feeding a mixture-density-network
(MDN) output head that predicts a distribution over the next pen offset
`(dx, dy, end-of-stroke)`. Generation is autoregressive: each sampled offset is
fed back in as the next input, one timestep at a time.

The model weights are ported from the pretrained TensorFlow checkpoint
published in [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)
(trained on IAM-OnDB), so no training was needed — `model.py` is a clean
reimplementation of the same architecture with the exact same gate layout as
TensorFlow's `LSTMCell`, so the pretrained kernel/bias tensors load directly.

Generated strokes are "primed" on a bundled real handwriting sample
(`weights/style-9-*.npy`) before free-running generation, biasing the output
toward that sample's consistent cursive style (Graves 2013, §5.3).

Output is rendered as vector paths (straight line segments between sampled
points, broken at each pen-lift) directly onto a PDF page via ReportLab — no
raster images involved.

## Installation

One-line installer (clones the repo, creates a venv, installs Python deps,
and checks for an Ollama install/model pull):

```
curl -fsSL https://raw.githubusercontent.com/red-coffee-cash/handwriting/claude/text-handwriting-ml-pdf-rj2g4w/install.sh | bash
```

Or manually:

```
git clone https://github.com/red-coffee-cash/handwriting.git
cd handwriting/handwriting_pdf
pip install -r requirements.txt
```

## Usage

```
python text_to_handwriting.py --text "Hello, world!" --out hello.pdf
python text_to_handwriting.py --file letter.txt --out letter.pdf --bias 1.0 --seed 42
```

Options:
- `--bias`: neatness, 0 = unbiased/messiest, ~0.75-1.0 = neat. Default 0.75.
- `--no-style-prime`: skip priming, use the model's unconditional default style.
- `--seed`: fix the random seed for reproducible output.

## Files

- `drawing.py` — stroke preprocessing/postprocessing (alphabet encoding, denoise, align).
- `model.py` — the PyTorch `HandwritingRNN` and weight loading.
- `sample.py` — autoregressive sampling loop (`sample_strokes`).
- `render.py` — stroke-to-PDF rendering and page/line layout.
- `text_to_handwriting.py` — CLI entry point.
- `weights/` — pretrained weights (`model_weights.npz`) and the priming sample.

## Worksheet pipeline (fill in a PDF worksheet with generated handwriting)

A second, separate pipeline takes a worksheet PDF, generates an answer for
each detected question with a local Gemma model (via [Ollama](https://ollama.com)),
renders those answers as handwriting (math expressions included, via a
handwriting-font skeleton-traced renderer), and lets you place/edit
everything in a browser GUI before compositing the result back onto the PDF.

No API keys are involved — Ollama serves Gemma fully locally. Install it and
pull a model first:

```
ollama pull gemma4:12b
```

### Quick start (everything in the browser)

The GUI is self-contained: start it once, then upload, edit, and download
without touching the CLI again.

```
pip install -r requirements.txt
python worksheet_cli.py serve --session session.json
# -> open http://127.0.0.1:5000, upload a worksheet PDF, and go
```

In the browser:
1. Upload a worksheet PDF — questions and suggested answer boxes are
   detected automatically.
2. Pick a question, adjust its answer box (the **Select**/**Box** tools),
   click **Generate** to get an answer from Gemma, and hand-edit the
   resulting strokes with the **Pen**/**Eraser** tools if needed.
3. Click **"Finish: Render & Download PDF"** to composite everything onto
   the original PDF and download it — no separate render step required.
4. Click **"Start New Worksheet"** to discard the session and upload another.

The toolbar has four tools: **Select** (move/resize an answer box via its
handles), **Box** (draw a new answer box from scratch), **Pen** (freehand
strokes, for fixing/adding handwriting by hand), and **Eraser** (click-drag
to remove parts of strokes near the cursor). Every edit auto-saves to the
session file, and Undo/Redo walk a per-question history of stroke edits.

### Scripted / CLI usage

The same steps are also available as separate CLI subcommands, useful for
automation or batch processing:

```
python worksheet_cli.py extract --pdf worksheet.pdf --session session.json
python worksheet_cli.py serve --session session.json   # edit in the GUI, then confirm
python worksheet_cli.py render --session session.json --out filled.pdf
```

Or do all three in one go with `worksheet_cli.py run --pdf worksheet.pdf
--session session.json --out filled.pdf` (serves the GUI, then renders
automatically once you confirm and stop the server with Ctrl-C).

### Files

- `pdf_extract.py` — heuristic question detection + answer-box suggestion from a worksheet PDF.
- `gemma_client.py` — queries a local Ollama-served Gemma model for an answer, split into text/math runs.
- `math_render.py` — renders a LaTeX math snippet as handwriting-style strokes (rasterize a handwriting font, skeletonize to centerlines, add a smooth arc-length-parametrized "hand tremor").
- `render_box.py` — lays out an answer's text+math runs to fit inside a confirmed on-page box, shrinking line height in tiers before falling back to a uniform scale-down.
- `layout_session.py` — the session JSON schema shared across extract/serve/render (questions, boxes, answers, strokes, confirmation state).
- `gui_app.py` + `static/gui/` — the local Flask + vanilla-JS placement/editing GUI (box placement, generate/regenerate, pen/eraser hand-editing, undo/redo).
- `pdf_compose.py` — composites a confirmed session's strokes onto the source PDF.
- `worksheet_cli.py` — ties the above into `extract` / `serve` / `render` / `run` subcommands.
