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

## Usage

```
pip install -r requirements.txt
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
