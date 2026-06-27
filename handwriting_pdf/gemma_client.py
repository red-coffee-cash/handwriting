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
REQUEST_TIMEOUT = 120
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


# Matches a $...$ math span. The opening `$` must be followed by a character
# that isn't a digit or whitespace, so currency like "$5", "$ 50", or
# "costs $5 and $3 more" is NOT misread as a math run -- a real math span
# starts with a letter, backslash macro, brace, sign, etc. (A bare "$5$"
# meaning the literal number 5 falls through to the text path, where the RNN
# draws "5" just fine, so nothing is lost.) Content still may not contain $.
_MATH_RUN_RE = re.compile(r"\$([^\d\s$][^$]*)\$")


def split_runs(answer_text):
    """Split an answer string into alternating text/math runs.

    Returns a list of {"kind": "text" | "math", "value": str} dicts, in
    order, covering the whole input. Empty text runs (e.g. answer starts
    or ends with math) are omitted.
    """
    runs = []
    pos = 0
    for m in _MATH_RUN_RE.finditer(answer_text):
        if m.start() > pos:
            text_chunk = answer_text[pos:m.start()]
            if text_chunk.strip():
                runs.append({"kind": "text", "value": text_chunk})
        math_chunk = m.group(1).strip()
        if math_chunk:
            runs.append({"kind": "math", "value": math_chunk})
        pos = m.end()
    if pos < len(answer_text):
        tail = answer_text[pos:]
        if tail.strip():
            runs.append({"kind": "text", "value": tail})
    return runs
