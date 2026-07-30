---
name: solve-worksheet
description: >
  Solve a worksheet PDF and fill it with generated handwriting via the
  handwriting MCP connector. Use when the user asks to solve, fill in,
  answer, or complete a worksheet/homework PDF with handwritten answers,
  or mentions the handwriting server. Requires the local handwriting
  server to be running and the "handwriting" MCP server to be registered.
---

# Solve a worksheet with handwritten answers

You drive the local handwriting pipeline through the `handwriting` MCP
tools. You do the solving; the tools do the rendering.

## Prerequisites

The Flask server must be running. If tool calls fail with "Cannot reach
the handwriting server", tell the user to start it:

```
cd handwriting_pdf
source .venv/bin/activate    # .venv\Scripts\activate on Windows
python worksheet_cli.py serve --session my_worksheet.json --port 5000
```

No Ollama/Gemma model is needed for this workflow — you provide the
answers yourself.

## Workflow

1. **Upload**: `upload_worksheet(pdf_path)` with the absolute path to the
   user's PDF. It returns detected questions, each with a proposed answer
   box (`page`, `x0,y0,x1,y1` in PDF points, origin top-left, y down).
2. **Read the problems**: use the extracted `question_text` per question;
   if it looks truncated or garbled, also read the PDF directly (Read
   tool) or call `get_page(page)` and read the image.
3. **Solve each question yourself.** Be careful and show work when the
   problem asks for it.
4. **Write answers**: `answer_question(question_id, answer_text)` for
   detected questions. If detection missed a question or a proposed box is
   clearly wrong, use `place_answer(page, x0, y0, x1, y1, answer_text)` /
   `update_box(...)` — pick blank space below or beside the printed
   question.
5. **Verify visually**: after answering each page's questions, call
   `preview_page(page)` and LOOK at the image: answers inside their
   boxes, not overlapping printed text, big enough to read. Fix issues
   with `update_box` (it re-renders the existing answer) and re-preview.
   Heed `warning` values — they mean the answer was shrunk to fit, so
   enlarge the box if there's room.
6. **Finalize**: `finalize(output_path)` writes the finished PDF. Tell the
   user where it landed.

## Answer formatting rules (the renderer's contract)

- Wrap ALL math in `$...$`. Plain words stay outside the dollars.
- One derivation step per line: separate steps with `\n` in answer_text.
  Do NOT use `\begin{aligned}` — write separate lines instead.
- Matrices: `$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$` (also
  bmatrix/vmatrix). Piecewise: `$\begin{cases} 1 & x > 0 \\ 0 & x \le 0
  \end{cases}$`.
- Simple LaTeX only: `\frac`, `\sqrt`, powers/subscripts, `\sum`, `\int`,
  `\lim`, Greek letters, `\leq/\geq/\neq` (aliases like `\le` are
  auto-fixed but prefer the full forms).
- Currency like `$5` in prose is fine — it is detected as money, not math.
- Keep answers concise; a worksheet box fits ~25–30 pt of height per line
  of handwriting. Multi-line answers need proportionally taller boxes.

## Box placement tips

- Coordinates are PDF points (1/72 inch), origin at the page's TOP-LEFT,
  y increasing DOWNWARD. `get_page` images are 2x: pixel / 2 = point.
- A letter page is 612 x 792 pt. Leave a few points of margin inside
  ruled answer areas.
- Prefer the detector's proposed boxes; only re-place when the preview
  shows a real problem.
