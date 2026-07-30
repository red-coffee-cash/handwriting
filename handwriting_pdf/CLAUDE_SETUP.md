# Solve worksheets with Claude Code — setup guide

This guide takes you from nothing to a first filled-in worksheet, using
Claude Code as the "brain": Claude reads the worksheet PDF, solves the
problems itself (no Ollama/Gemma needed), places the answers, checks its
own work against a rendered preview, and saves the finished PDF — all by
driving this project's local server through an MCP connector.

```
you                    Claude Code                 this project
───────────────────────────────────────────────────────────────────
"solve hw3.pdf"  ──▶   reads the PDF,       ──▶   handwriting server
                       solves the problems        renders strokes,
                       calls MCP tools     ◀──    returns previews
                       verifies previews   ──▶    writes filled.pdf
```

## 1. Install Claude Code

You need a Claude account (Pro/Max subscription or an Anthropic Console
account with API billing).

**macOS**

```bash
# Native installer:
curl -fsSL https://claude.ai/install.sh | bash
# or, with Homebrew:
brew install --cask claude-code
```

**Linux**

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

**Windows**

```powershell
# PowerShell (native installer):
irm https://claude.ai/install.ps1 | iex
```

Or use WSL and follow the Linux steps inside it (recommended if you
already work in WSL — keep everything, this project included, on the same
side).

**Verify and sign in**

```bash
claude --version     # prints a version -> installed
claude               # first launch opens a browser to sign in; pick your
                     # Claude account and authorize
claude doctor        # optional: checks the install for common problems
```

## 2. Install this project

```bash
git clone https://github.com/red-coffee-cash/handwriting.git
cd handwriting/handwriting_pdf
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

(Or use the one-line `install.sh` from the repo README — when it asks
about Ollama you can skip it entirely: `SKIP_MODEL=1`. Claude does the
solving in this workflow, so the local Gemma model is optional.)

## 3. Start the handwriting server

From `handwriting_pdf/`, with the venv active:

```bash
python worksheet_cli.py serve --session my_worksheet.json --port 5000
```

Leave this running in its own terminal. The browser GUI at
http://127.0.0.1:5000 keeps working alongside Claude — they share the
same session, so you can watch Claude's edits appear live.

## 4. Register the MCP connector with Claude Code

From the repo root (`handwriting/`), one time:

```bash
claude mcp add handwriting \
  -e HANDWRITING_URL=http://127.0.0.1:5000 \
  -- ./handwriting_pdf/.venv/bin/python handwriting_pdf/mcp_server.py
```

On Windows (PowerShell, no line continuations):

```powershell
claude mcp add handwriting -e HANDWRITING_URL=http://127.0.0.1:5000 -- handwriting_pdf\.venv\Scripts\python.exe handwriting_pdf\mcp_server.py
```

Notes:
- The `python` used must be the project venv's python (it needs the
  `mcp` and `requests` packages installed in step 2).
- If you chose a different `--port` in step 3, match it in
  `HANDWRITING_URL`.
- Check it worked: run `claude` in the repo and type `/mcp` — you should
  see `handwriting` listed with its 8 tools.

## 5. First run

1. Put a worksheet PDF somewhere handy, e.g. `~/Downloads/hw3.pdf`.
2. Start `claude` **from the repo root** (so the `solve-worksheet` skill
   in `.claude/skills/` is picked up).
3. Say:

   > Use the solve-worksheet skill to solve ~/Downloads/hw3.pdf and save
   > the result next to it.

4. What you'll see, in order:
   - a permission prompt the first time each MCP tool is used — approve
     them (or approve "always" for this project);
   - `upload_worksheet(...)` — the server detects the questions and
     proposes answer boxes;
   - Claude reading the PDF and solving the problems;
   - `answer_question(...)` / `place_answer(...)` calls as it writes each
     answer in generated handwriting;
   - `preview_page(...)` calls where Claude looks at a rendered image of
     the filled page and fixes any box that overlaps text or ran out of
     room;
   - `finalize(...)` — the finished PDF path is reported at the end.
5. Open the output PDF and enjoy the handwriting.

## Troubleshooting

- **"Cannot reach the handwriting server"** — the Flask server from step
  3 isn't running, or the port in `HANDWRITING_URL` doesn't match. It
  takes ~20–60 s to start (PyTorch import); wait for "Running on
  http://127.0.0.1:5000" before asking Claude to work.
- **`handwriting` missing from `/mcp`** — re-run the `claude mcp add`
  command from the repo root, and confirm the python path in it exists.
  `claude mcp list` shows what's registered; `claude mcp remove
  handwriting` lets you redo it.
- **Tool calls fail with import errors** — the registered python isn't
  the venv one, or `pip install -r requirements.txt` didn't run there.
- **Skill doesn't trigger** — make sure you launched `claude` from the
  repo root; skills load from `.claude/skills/` in the project you start
  in. You can always invoke it explicitly: `/solve-worksheet`.
- **You updated the code** — restart the Flask server; it loads Python
  modules once at startup.
