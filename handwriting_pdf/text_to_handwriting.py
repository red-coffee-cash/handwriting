#!/usr/bin/env python3
"""CLI: turn plain text into a PDF of generated handwriting."""
import argparse
import sys

from render import render_pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to render.")
    src.add_argument("--file", help="Path to a text file to render.")

    parser.add_argument("--out", required=True, help="Output PDF path.")
    parser.add_argument(
        "--bias", type=float, default=0.75,
        help="Neatness bias (higher = neater, less varied strokes). Default 0.75.",
    )
    parser.add_argument(
        "--no-style-prime", action="store_true",
        help="Skip style priming; use the model's unconditional default style.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible output.")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        text = args.text

    render_pdf(
        text,
        args.out,
        bias=args.bias,
        style_prime=not args.no_style_prime,
        seed=args.seed,
    )
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
