#!/usr/bin/env bash
# One-line installer:
#   curl -fsSL https://raw.githubusercontent.com/red-coffee-cash/handwriting/claude/text-handwriting-ml-pdf-rj2g4w/install.sh | bash
#
# Clones (or updates) the repo, creates a virtualenv, installs Python
# dependencies, and checks for a local Ollama install + model pull.
set -euo pipefail

REPO_URL="https://github.com/red-coffee-cash/handwriting.git"
INSTALL_DIR="${HANDWRITING_INSTALL_DIR:-$PWD/handwriting}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:4b}"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
info()  { printf '\033[34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[32m✔\033[0m %s\n' "$*"; }
warn()  { printf '\033[33m!\033[0m %s\n' "$*"; }

bold "Handwriting PDF pipeline installer"

if [ -f "./handwriting_pdf/worksheet_cli.py" ]; then
  INSTALL_DIR="$PWD"
  info "Already inside a checkout ($INSTALL_DIR), skipping clone."
elif [ -d "$INSTALL_DIR/.git" ]; then
  info "Found existing checkout at $INSTALL_DIR, pulling latest..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  info "Cloning $REPO_URL into $INSTALL_DIR..."
  git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR/handwriting_pdf"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  warn "$PYTHON_BIN not found. Install Python 3.9+ and re-run this script."
  exit 1
fi
ok "Found $($PYTHON_BIN --version)"

if [ ! -d ".venv" ]; then
  info "Creating virtualenv (.venv)..."
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
ok "Virtualenv ready."

info "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
ok "Dependencies installed."

if command -v ollama >/dev/null 2>&1; then
  ok "Found ollama."
  if [ "${SKIP_MODEL:-0}" != "1" ]; then
    info "Pulling model $OLLAMA_MODEL (set SKIP_MODEL=1 to skip)..."
    ollama pull "$OLLAMA_MODEL" || warn "Model pull failed -- pull it manually later with: ollama pull $OLLAMA_MODEL"
  fi
else
  warn "ollama not found. Install it from https://ollama.com/download, then run:"
  warn "  ollama pull $OLLAMA_MODEL"
fi

echo
bold "Setup complete!"
cat <<EOF

  cd $INSTALL_DIR/handwriting_pdf
  source .venv/bin/activate
  python worksheet_cli.py serve --session my_worksheet.json

Then open http://127.0.0.1:5000 in your browser and upload a worksheet PDF.
EOF
