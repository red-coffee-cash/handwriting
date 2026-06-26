#!/usr/bin/env python3
"""CLI for the worksheet pipeline: extract questions from a PDF, serve the
placement/editing GUI, and render the finalized handwriting onto the PDF.

Subcommands:
  extract  PDF -> session JSON (detected questions + suggested answer boxes)
  serve    session JSON -> local GUI (place boxes, generate answers, hand-edit
           strokes with pen/eraser) until you confirm the layout in-browser
  render   confirmed session JSON -> finished PDF with handwriting composited in
  run      convenience: extract (if needed) + serve + render in one go
"""
import argparse
import sys

import gemma_client
import layout_session as ls
import pdf_compose
import pdf_extract


def cmd_extract(args):
    doc = pdf_extract.load_pdf(args.pdf)
    questions = pdf_extract.build_question_records(doc)
    pages = [{"width": page.rect.width, "height": page.rect.height} for page in doc]
    session = ls.new_session(args.pdf, pages, questions)
    ls.save(session, args.session)
    print(f"Extracted {len(questions)} question(s) -> {args.session}", file=sys.stderr)


def cmd_serve(args):
    import gui_app
    app = gui_app.create_app(args.session, ollama_url=args.ollama_url, model=args.model)
    print(f"Serving GUI at http://127.0.0.1:{args.port}  (Ctrl-C to stop)", file=sys.stderr)
    app.run(port=args.port, debug=False)


def cmd_render(args):
    session = ls.load(args.session)
    if not session.get("confirmed") and not args.force:
        print(
            "Session is not confirmed yet. Run `serve` and click "
            "'Confirm Layout & Finish', or pass --force to render anyway.",
            file=sys.stderr,
        )
        sys.exit(1)
    drawn = pdf_compose.compose(args.session, args.out)
    print(f"Drew {drawn} strokes -> {args.out}", file=sys.stderr)


def cmd_run(args):
    import os
    if not os.path.exists(args.session):
        cmd_extract(args)
    else:
        print(f"Using existing session {args.session}", file=sys.stderr)

    import gui_app
    app = gui_app.create_app(args.session, ollama_url=args.ollama_url, model=args.model)
    print(f"Serving GUI at http://127.0.0.1:{args.port}  (Ctrl-C when done)", file=sys.stderr)
    try:
        app.run(port=args.port, debug=False)
    except KeyboardInterrupt:
        pass

    session = ls.load(args.session)
    if session.get("confirmed"):
        drawn = pdf_compose.compose(args.session, args.out)
        print(f"Drew {drawn} strokes -> {args.out}", file=sys.stderr)
    else:
        print(
            "Session was not confirmed; skipping render. "
            "Run `worksheet_cli.py render` once it is.",
            file=sys.stderr,
        )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract questions from a PDF into a session file.")
    p_extract.add_argument("--pdf", required=True)
    p_extract.add_argument("--session", required=True)
    p_extract.set_defaults(func=cmd_extract)

    p_serve = sub.add_parser("serve", help="Serve the placement/editing GUI for a session.")
    p_serve.add_argument("--session", required=True)
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--ollama-url", default=gemma_client.DEFAULT_OLLAMA_URL)
    p_serve.add_argument("--model", default=gemma_client.DEFAULT_MODEL)
    p_serve.set_defaults(func=cmd_serve)

    p_render = sub.add_parser("render", help="Composite a confirmed session's strokes onto its source PDF.")
    p_render.add_argument("--session", required=True)
    p_render.add_argument("--out", required=True)
    p_render.add_argument("--force", action="store_true", help="Render even if not confirmed yet.")
    p_render.set_defaults(func=cmd_render)

    p_run = sub.add_parser("run", help="extract (if needed) + serve + render in one go.")
    p_run.add_argument("--pdf", required=True)
    p_run.add_argument("--session", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--port", type=int, default=5000)
    p_run.add_argument("--ollama-url", default=gemma_client.DEFAULT_OLLAMA_URL)
    p_run.add_argument("--model", default=gemma_client.DEFAULT_MODEL)
    p_run.set_defaults(func=cmd_run)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
