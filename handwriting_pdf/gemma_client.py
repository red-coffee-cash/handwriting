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
import re
import shutil
import subprocess
import time

import requests

# Explicit 127.0.0.1, not "localhost" -- on dual-stack systems "localhost"
# can resolve to ::1 first, and since Ollama only binds IPv4 by default that
# produces a connection-refused error even while IPv4 access works fine.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:12b"
REQUEST_TIMEOUT = 120
STARTUP_TIMEOUT = 30

_ollama_launched = False

SYSTEM_PROMPT = (
    "You are completing a worksheet. Given a question, respond with a JSON "
    'object of the form {"answer": "..."}. The answer should be concise '
    "and directly answer the question. If the answer involves mathematical "
    "notation, wrap each math expression in single dollar signs using LaTeX "
    "syntax, e.g. \"The result is $x^2 + 1$.\" Do not wrap plain words or "
    "numbers in dollar signs -- only use them for actual math notation. "
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
    if answer is not None:
        return answer

    raw_retry = _call_ollama(question_text, model, ollama_url, retry_hint=True)
    answer = _parse_answer_json(raw_retry)
    if answer is not None:
        return answer

    return raw_retry.strip() or raw.strip()


def _ollama_reachable(ollama_url):
    try:
        requests.get(f"{ollama_url}/api/tags", timeout=2)
        return True
    except requests.RequestException:
        return False


def _ensure_ollama_running(ollama_url):
    """Start `ollama serve` if it isn't already up. No-op once it's
    reachable; only attempted once per process even if startup fails."""
    global _ollama_launched
    if _ollama_reachable(ollama_url):
        return

    if shutil.which("ollama") is None:
        raise GemmaClientError(
            "Ollama is not running and the `ollama` command was not found "
            "on PATH. Install it from https://ollama.com/download, then "
            "run `ollama serve`."
        )

    if not _ollama_launched:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _ollama_launched = True

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _ollama_reachable(ollama_url):
            return
        time.sleep(0.5)
    raise GemmaClientError(f"Timed out waiting for Ollama to start at {ollama_url}.")


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
    except requests.RequestException as exc:
        raise GemmaClientError(f"Ollama request failed: {exc}") from exc
    return resp.json().get("response", "")


def _parse_answer_json(raw):
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
        return obj["answer"]
    return None


# Matches a $...$ span; math content must not itself contain a literal $.
_MATH_RUN_RE = re.compile(r"\$([^$]+)\$")


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
