"""Generate worksheet answers via a local Gemma model served by Ollama.

No API keys involved -- Ollama serves the model on localhost. We ask for
JSON-constrained output (Ollama's `"format": "json"` request field) so the
response is a structured {"answer": "..."} object rather than free text we
have to scrape, and ask the model to wrap any mathematical notation in
$...$ LaTeX delimiters so split_runs() can hand math substrings off to
math_render.py separately from the plain-text runs that go through the
handwriting RNN.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time

import requests

# Explicit 127.0.0.1, not "localhost" -- on dual-stack systems "localhost"
# can resolve to ::1 first, and since Ollama only binds IPv4 by default that
# produces a connection-refused error even while IPv4 access works fine.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:12b"
REQUEST_TIMEOUT = 300
STARTUP_TIMEOUT = 30

# Handle to the `ollama serve` process we launched (if any), plus the temp
# file capturing its stderr. Kept so we can detect an immediate crash and
# surface the real reason instead of a generic timeout -- and so a failed
# start can be retried on the next request rather than latching forever.
_ollama_proc = None
_ollama_stderr_path = None

SYSTEM_PROMPT = (
    "You are completing a worksheet. Given a question, respond with a JSON "
    'object of the form {"answer": "..."}. The answer should be concise '
    "and directly answer the question. If the answer involves mathematical "
    "notation, wrap each math expression in single dollar signs using LaTeX "
    "syntax, e.g. \"The result is $x^2 + 1$.\" Do not wrap plain words or "
    "numbers in dollar signs -- only use them for actual math notation.\n"
    "For a math problem, show only the work and the final solution -- the "
    "calculation steps and the answer, all as math. Do NOT write any "
    "explanation of why, any reasoning in words, or any English sentences "
    "at all. The ONLY exception is if the problem itself explicitly asks "
    "for an explanation, justification, or proof; in that case, write the "
    "explanation or proof as required.\n"
    "Use simple LaTeX only: \\frac, \\sqrt, powers and subscripts, \\sum, "
    "\\int, \\lim, Greek letters, \\leq, \\geq, \\neq. Put each step of "
    "your work on its own line using \\n newlines in the JSON string; do "
    "NOT use \\begin{aligned} or other alignment environments. Matrices "
    "may be written with \\begin{pmatrix} ... \\end{pmatrix} and piecewise "
    "definitions with \\begin{cases} ... \\end{cases}.\n"
    "Respond with only the JSON object, no other text."
)


class GemmaClientError(RuntimeError):
    pass


def generate_answer(question_text, model=DEFAULT_MODEL, ollama_url=DEFAULT_OLLAMA_URL):
    """Query Ollama for an answer to a single question. Returns the raw
    answer string (which may contain $...$ math runs). Retries once on a
    malformed JSON response before falling back to treating the raw model
    output as a plain-text answer."""
    _ensure_ollama_running(ollama_url)
    raw = _call_ollama(question_text, model, ollama_url)
    answer = _parse_answer_json(raw)
    if answer is None:
        raw_retry = _call_ollama(question_text, model, ollama_url, retry_hint=True)
        answer = _parse_answer_json(raw_retry)
        if answer is None:
            answer = raw_retry.strip() or raw.strip()

    if not answer.strip():
        raise GemmaClientError(
            "The model returned an empty answer. It may have produced no "
            "output or only an empty JSON field; try regenerating."
        )
    return answer


def _ollama_reachable(ollama_url):
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=2)
        return True
    except requests.RequestException:
        return False


def _read_ollama_stderr():
    """Return whatever `ollama serve` wrote to stderr, trimmed, or ''."""
    if not _ollama_stderr_path or not os.path.exists(_ollama_stderr_path):
        return ""
    try:
        with open(_ollama_stderr_path, "r", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _ensure_ollama_running(ollama_url):
    """Start `ollama serve` if it isn't already up. No-op once it's
    reachable. If a process we launched dies, surface its stderr and allow a
    fresh launch on the next call rather than latching into a permanent
    failure state."""
    global _ollama_proc, _ollama_stderr_path
    if _ollama_reachable(ollama_url):
        return

    # If we previously launched one and it has since exited, fold its stderr
    # into the error below and clear the handle so we can relaunch.
    crashed_stderr = ""
    if _ollama_proc is not None and _ollama_proc.poll() is not None:
        crashed_stderr = _read_ollama_stderr()
        _ollama_proc = None

    if shutil.which("ollama") is None:
        raise GemmaClientError(
            "Ollama is not running and the `ollama` command was not found "
            "on PATH. Install it from https://ollama.com/download, then "
            "run `ollama serve`."
        )

    if _ollama_proc is None:
        stderr_file = tempfile.NamedTemporaryFile(
            prefix="ollama-serve-", suffix=".log", delete=False
        )
        _ollama_stderr_path = stderr_file.name
        _ollama_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            start_new_session=True,
        )
        stderr_file.close()

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _ollama_reachable(ollama_url):
            return
        if _ollama_proc.poll() is not None:
            # The server we just launched exited before becoming reachable.
            stderr = _read_ollama_stderr()
            _ollama_proc = None
            raise GemmaClientError(_startup_error(ollama_url, stderr))
        time.sleep(0.5)
    raise GemmaClientError(_startup_error(ollama_url, crashed_stderr or _read_ollama_stderr()))


def _startup_error(ollama_url, stderr):
    msg = f"Could not reach Ollama at {ollama_url} after {STARTUP_TIMEOUT}s."
    if stderr:
        msg += f" `ollama serve` reported:\n{stderr}"
    else:
        msg += (
            " The server may still be loading the model, or another process "
            "may be bound to that port. Try again, or start `ollama serve` "
            "manually to see the error."
        )
    return msg


def _call_ollama(question_text, model, ollama_url, retry_hint=False):
    prompt = f"Question: {question_text}\n\nRespond with the JSON object now."
    if retry_hint:
        prompt = (
            "Your previous response was not valid JSON. "
            'Respond with ONLY a JSON object like {"answer": "..."}.\n\n'
            f"Question: {question_text}"
        )
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }
    try:
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.Timeout as exc:
        raise GemmaClientError(
            f"Ollama timed out after {REQUEST_TIMEOUT}s for model '{model}'. "
            "The model may still be loading into memory (the first request "
            "after startup is slowest) -- try again."
        ) from exc
    except requests.HTTPError as exc:
        # Surface Ollama's own error text (e.g. 'model not found'), which is
        # in the JSON body and otherwise lost behind a bare status code.
        detail = _http_error_detail(exc)
        raise GemmaClientError(f"Ollama returned an error: {detail}") from exc
    except requests.RequestException as exc:
        raise GemmaClientError(f"Ollama request failed: {exc}") from exc
    return resp.json().get("response", "")


def _http_error_detail(exc):
    resp = exc.response
    if resp is None:
        return str(exc)
    try:
        body = resp.json()
        if isinstance(body, dict) and body.get("error"):
            return f"{resp.status_code} {body['error']}"
    except ValueError:
        pass
    return f"{resp.status_code} {resp.text[:300]}".strip()


def _parse_answer_json(raw):
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
        return obj["answer"]
    return None


# Matches an explicitly delimited math span, most specific delimiter first:
# $$...$$ (display), \[...\], \(...\), then plain $...$. A $...$ span whose
# content starts with a digit or whitespace might be currency instead of
# math ("costs $5 and $3 more" pairs into "$5 and $"); split_runs resolves
# that ambiguity with _inline_span_is_currency rather than a regex guard,
# so legitimate digit-start math like "$2x + 1$" or "$2\pi r$" still counts.
_DELIM_MATH_RE = re.compile(
    r"\$\$(?P<display>.+?)\$\$"
    r"|\\\[(?P<bracket>.+?)\\\]"
    r"|\\\((?P<paren>.+?)\\\)"
    r"|\$(?P<inline>[^$]+)\$",
    re.DOTALL,
)

_BARE_NUMBER_RE = re.compile(r"\d[\d,]*(\.\d+)?")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _inline_span_is_currency(content):
    """Heuristic for a $...$ span whose content starts with a digit or
    whitespace: it's money-flavored prose (not math) if it is a bare number
    ("$5$") or contains a multi-letter English word ("$5 and $" from
    "costs $5 and $3"). Digit-start algebra like "2x + 1" or "2\\pi r" has
    neither, so it stays math."""
    stripped = content.strip()
    if _BARE_NUMBER_RE.fullmatch(stripped):
        return True
    return any(_ALPHA_WORD_RE.fullmatch(tok) for tok in stripped.split())


# Alignment-only environments: mathtext can't parse them, but their content
# is just rows of ordinary math, so split_runs unwraps them and emits one
# math run per row. (Matrix/cases environments are NOT in this list -- they
# carry 2D structure and are composited specially by math_render.)
_ALIGN_ENV_RE = re.compile(
    r"\\begin\{(aligned|align\*?|gathered|gather\*?|eqnarray\*?|split)\}"
    r"(.*?)\\end\{\1\}",
    re.DOTALL,
)


def _split_top_level_rows(value):
    """Split a math snippet on \\\\ row separators that are not nested
    inside a \\begin{...}\\end{...} block, dropping top-level alignment &s
    (matrix/cases cell separators are below top level and untouched)."""
    rows, buf, depth, i = [], [], 0, 0
    n = len(value)
    while i < n:
        if value.startswith("\\begin", i):
            depth += 1
            buf.append("\\begin")
            i += 6
            continue
        if value.startswith("\\end", i):
            depth = max(0, depth - 1)
            buf.append("\\end")
            i += 4
            continue
        if depth == 0 and value.startswith("\\\\", i):
            rows.append("".join(buf))
            buf = []
            i += 2
            continue
        if depth == 0 and value[i] == "&":
            i += 1
            continue
        buf.append(value[i])
        i += 1
    rows.append("".join(buf))
    return [r.strip() for r in rows if r.strip()]


def _explode_math_value(value):
    """Turn one delimited math span into a list of row snippets: unwrap
    alignment-only environments, then split multi-line derivations on
    top-level \\\\ so each step can go on its own rendered line."""
    value = _ALIGN_ENV_RE.sub(lambda m: m.group(2), value)
    return _split_top_level_rows(value)

# Bare (undelimited) LaTeX detection over whitespace tokens of a text chunk.
# A "strong" token unambiguously signals LaTeX: a \command macro or a
# superscript/subscript. A "weak" token is operator/number material (digits,
# + - * / = < > braces parens punctuation, no letters) that may belong to a
# surrounding expression but never triggers math on its own -- so plain
# prose and arithmetic like "12 + 7 = 19" stay text (render_box already
# routes undrawable characters through the math renderer).
_STRONG_LATEX_RE = re.compile(r"\\[a-zA-Z]+|[\^_]")
_WEAK_MATH_RE = re.compile(r"^[0-9+\-*/=<>(){}\[\].,:%|&\\]+$")


def _classify_token(tok):
    if _STRONG_LATEX_RE.search(tok):
        return "strong"
    if _WEAK_MATH_RE.match(tok):
        return "weak"
    return "text"


def _split_bare_latex(chunk):
    """Split a $-free text chunk into ("text"|"math", value) pieces,
    pulling out maximal token spans of strong/weak math tokens that contain
    at least one strong token -- so "\\frac{1}{2} + \\sqrt{2}" stays one
    atomic math run instead of being word-wrapped into five fragments."""
    tokens = chunk.split()
    pieces = []
    text_buf = []
    i = 0
    while i < len(tokens):
        if _classify_token(tokens[i]) == "text":
            text_buf.append(tokens[i])
            i += 1
            continue
        j = i
        has_strong = False
        while j < len(tokens) and _classify_token(tokens[j]) != "text":
            has_strong = has_strong or _classify_token(tokens[j]) == "strong"
            j += 1
        if has_strong:
            if text_buf:
                pieces.append(("text", " ".join(text_buf)))
                text_buf = []
            pieces.append(("math", " ".join(tokens[i:j])))
        else:
            text_buf.extend(tokens[i:j])
        i = j
    if text_buf:
        pieces.append(("text", " ".join(text_buf)))
    return pieces


def split_runs(answer_text):
    """Split an answer string into text/math/break runs.

    Recognizes $...$, $$...$$, \\(...\\), and \\[...\\] delimited math, plus
    undelimited LaTeX the model forgot to wrap (see _split_bare_latex).
    Multi-line answers and multi-row math (aligned blocks, top-level \\\\)
    produce {"kind": "break"} runs so the renderer can keep each derivation
    step on its own line. Returns a list of {"kind": "text" | "math",
    "value": str} / {"kind": "break"} dicts, in order. Empty text runs are
    omitted; unpaired delimiters are left as text.
    """
    runs = []

    def add_break():
        if runs and runs[-1]["kind"] != "break":
            runs.append({"kind": "break"})

    def add_text(chunk):
        for li, line in enumerate(chunk.split("\n")):
            if li:
                add_break()
            if line.strip():
                for kind, value in _split_bare_latex(line):
                    runs.append({"kind": kind, "value": value})

    def add_math(value):
        for ri, row in enumerate(_explode_math_value(value)):
            if ri:
                add_break()
            runs.append({"kind": "math", "value": row})

    pos = 0
    scan = 0
    while True:
        m = _DELIM_MATH_RE.search(answer_text, scan)
        if m is None:
            break
        if m.lastgroup == "inline":
            content = m.group("inline")
            if (content[:1].isdigit() or content[:1].isspace()) \
                    and _inline_span_is_currency(content):
                # Money, not math ("$5 and $" from "costs $5 and $3").
                # Leave it in the text stream and retry just past the
                # opening $ so a real math span later on isn't missed.
                scan = m.start() + 1
                continue
        if m.start() > pos:
            add_text(answer_text[pos:m.start()])
        math_chunk = m.group(m.lastgroup).strip()
        if math_chunk:
            add_math(math_chunk)
        pos = scan = m.end()
    if pos < len(answer_text):
        add_text(answer_text[pos:])

    while runs and runs[-1]["kind"] == "break":
        runs.pop()
    return runs
