"""Ollama HTTP client: chat, streaming, embeddings, health."""
import json

import httpx

from . import config

_client = httpx.Client(base_url=config.OLLAMA_URL, timeout=httpx.Timeout(300, connect=5))


def _options():
    return {"num_ctx": config.NUM_CTX}


def chat(messages, model=None, tools=None, temperature=0.2) -> dict:
    """Non-streaming chat. Returns the response `message` dict (may contain tool_calls)."""
    body = {
        "model": model or config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": config.KEEP_ALIVE,
        "options": {**_options(), "temperature": temperature},
    }
    if tools:
        body["tools"] = tools
    r = _client.post("/api/chat", json=body)
    r.raise_for_status()
    return r.json()["message"]


def chat_stream(messages, model=None, temperature=0.2):
    """Yields content tokens."""
    body = {
        "model": model or config.CHAT_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": config.KEEP_ALIVE,
        "options": {**_options(), "temperature": temperature},
    }
    with _client.stream("POST", "/api/chat", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            part = json.loads(line)
            token = part.get("message", {}).get("content", "")
            if token:
                yield token


def embed(texts: list[str], prefix: str) -> list[list[float]]:
    """nomic-embed-text needs task prefixes: 'search_document: ' / 'search_query: '."""
    r = _client.post(
        "/api/embed",
        json={
            "model": config.EMBED_MODEL,
            "input": [prefix + t for t in texts],
            "keep_alive": config.KEEP_ALIVE,
        },
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def health() -> dict:
    try:
        r = _client.get("/api/tags", timeout=5)
        r.raise_for_status()
        installed = [m["name"] for m in r.json()["models"]]

        def has(name):
            return any(i == name or i.split(":")[0] == name for i in installed)

        missing = [m for m in (config.CHAT_MODEL, config.FAST_MODEL, config.EMBED_MODEL) if not has(m)]
        return {"ollama": "up", "missing_models": missing, "ok": not missing}
    except httpx.HTTPError as e:
        return {"ollama": "down", "error": str(e), "ok": False}
