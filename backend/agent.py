"""Intent router + tool loop + response orchestration.

respond() is a generator of SSE-ready event dicts:
  {"type":"token","content":str} ... {"type":"citations","citations":[...]}
  {"type":"done","message_id":int}
"""
import json
import re

from . import config, llm, prompts, retrieval
from .tracing import Trace

TOOLS = [
    {"type": "function", "function": {
        "name": "search_corpus",
        "description": "Search the user's paper library for passages relevant to a query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query"},
            "paper_id": {"type": "integer", "description": "optional: restrict to one paper"},
        }, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_papers",
        "description": "List all papers in the user's library with ids, titles, authors, years.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "summarize_paper",
        "description": "Get a summary of one paper (problem, method, results, limitations).",
        "parameters": {"type": "object", "properties": {
            "paper_id": {"type": "integer"}}, "required": ["paper_id"]}}},
    {"type": "function", "function": {
        "name": "compare_papers",
        "description": "Retrieve evidence from two or more papers about one aspect, for comparison.",
        "parameters": {"type": "object", "properties": {
            "paper_ids": {"type": "array", "items": {"type": "integer"}},
            "aspect": {"type": "string", "description": "what to compare, e.g. 'training method'"},
        }, "required": ["paper_ids", "aspect"]}}},
]


def route_intent(question: str) -> str:
    try:
        msg = llm.chat([{"role": "user", "content": prompts.ROUTER.format(question=question)}],
                       model=config.FAST_MODEL, temperature=0)
        word = re.sub(r"\W", "", msg["content"].strip().split()[0]).lower()
        if word in ("simple_qa", "needs_tools", "chitchat", "simpleqa", "needstools"):
            return {"simpleqa": "simple_qa", "needstools": "needs_tools"}.get(word, word)
    except Exception:
        pass
    return "simple_qa"


def rewrite_query(question: str, history: list[dict]) -> str:
    if not history:
        return question
    hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:])
    try:
        msg = llm.chat([{"role": "user",
                         "content": prompts.REWRITE.format(history=hist, question=question)}],
                       model=config.FAST_MODEL, temperature=0)
        rewritten = msg["content"].strip().strip('"')
        return rewritten if 0 < len(rewritten) < 400 else question
    except Exception:
        return question


def validate_citations(text: str, n_sources: int) -> list[int]:
    """Markers appearing in text that map to real sources (1-indexed)."""
    return sorted({int(m) for m in re.findall(r"\[(\d{1,2})\]", text) if 1 <= int(m) <= n_sources})


def format_sources(sources: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {s['title']} (p.{s['page_start']}"
        + (f"-{s['page_end']}" if s['page_end'] != s['page_start'] else "")
        + f"): {s['text'][:1500]}"
        for i, s in enumerate(sources))


def summarize(conn, paper_id: int) -> str:
    row = conn.execute("SELECT title, summary FROM papers WHERE id=?", (paper_id,)).fetchone()
    if not row:
        return f"No paper with id {paper_id}."
    if row["summary"]:
        return row["summary"]
    texts = [r["text"] for r in conn.execute(
        "SELECT text FROM chunks WHERE paper_id=? ORDER BY id", (paper_id,))]
    if not texts:
        return "Paper has no ingested text."
    body = "\n".join(texts)
    excerpt = body[:12000] + ("\n[...]\n" + body[-3000:] if len(body) > 15000 else "")
    msg = llm.chat([{"role": "user",
                     "content": prompts.SUMMARIZE.format(title=row["title"], text=excerpt)}])
    summary = msg["content"].strip()
    conn.execute("UPDATE papers SET summary=? WHERE id=?", (summary, paper_id))
    conn.commit()
    return summary


def _add_sources(sources: list[dict], chunks: list[dict]) -> str:
    """Register chunks in the global source list; return them formatted with their [n]."""
    out = []
    for c in chunks:
        idx = next((i for i, s in enumerate(sources) if s["chunk_id"] == c["chunk_id"]), None)
        if idx is None:
            sources.append(c)
            idx = len(sources) - 1
        out.append(f"[{idx + 1}] {c['title']} (p.{c['page_start']}): {c['text'][:1200]}")
    return "\n\n".join(out) if out else "No relevant passages found."


def run_tool(conn, name: str, args: dict, sources: list[dict]) -> str:
    if name == "search_corpus":
        res = retrieval.hybrid_search(conn, str(args.get("query", "")), k=6,
                                      paper_id=args.get("paper_id"))
        return _add_sources(sources, res["chunks"])
    if name == "list_papers":
        rows = conn.execute(
            "SELECT id, title, authors, year, status FROM papers ORDER BY id").fetchall()
        return "\n".join(f"id={r['id']}: {r['title']} — {r['authors'] or '?'} "
                         f"({r['year'] or '?'}) [{r['status']}]" for r in rows) or "Library is empty."
    if name == "summarize_paper":
        return summarize(conn, int(args["paper_id"]))
    if name == "compare_papers":
        parts = []
        for pid in list(args.get("paper_ids", []))[:4]:
            res = retrieval.hybrid_search(conn, str(args.get("aspect", "")), k=3, paper_id=int(pid))
            title = (res["chunks"][0]["title"] if res["chunks"] else f"paper {pid}")
            parts.append(f"=== {title} ===\n" + _add_sources(sources, res["chunks"]))
        return "\n\n".join(parts)
    return f"Unknown tool {name}."


def _agent_loop(conn, question: str, history: list[dict], sources: list[dict], trace: Trace):
    """Yields tokens for the final answer; gathers sources via tools along the way."""
    messages = ([{"role": "system", "content": prompts.AGENT_SYSTEM}]
                + history + [{"role": "user", "content": question}])
    for step in range(config.MAX_AGENT_STEPS):
        with trace.stage(f"agent_step_{step + 1}") as p:
            msg = llm.chat(messages, tools=TOOLS)
            calls = msg.get("tool_calls") or []
            p["tool_calls"] = [c["function"]["name"] for c in calls]
            if not calls:
                p["final"] = True
                yield msg.get("content", "")
                return
            messages.append(msg)
            for call in calls:
                fn = call["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                result = run_tool(conn, fn["name"], args, sources)
                p.setdefault("results", []).append({"tool": fn["name"], "args": args,
                                                    "chars": len(result)})
                messages.append({"role": "tool", "tool_name": fn["name"], "content": result})
    # step budget exhausted: force a final synthesis over what was gathered
    messages.append({"role": "user",
                     "content": "Answer now with what you have, citing sources like [1]."})
    with trace.stage("agent_final"):
        yield from llm.chat_stream(messages)


def respond(conn, chat_id: int, question: str):
    trace = Trace()
    history = [dict(m) for m in conn.execute(
        "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT 8",
        (chat_id,))][::-1]
    conn.execute("INSERT INTO messages(chat_id, role, content) VALUES (?,?,?)",
                 (chat_id, "user", question))
    conn.execute("UPDATE chats SET title=? WHERE id=? AND title='New chat'",
                 (question[:60], chat_id))
    conn.commit()

    with trace.stage("route") as p:
        intent = route_intent(question)
        p["intent"] = intent

    sources: list[dict] = []
    parts: list[str] = []

    def stream(tokens):
        for t in tokens:
            parts.append(t)
            yield {"type": "token", "content": t}

    if intent == "chitchat":
        msgs = ([{"role": "system", "content": prompts.CHITCHAT_SYSTEM}]
                + history + [{"role": "user", "content": question}])
        with trace.stage("generate"):
            yield from stream(llm.chat_stream(msgs, model=config.FAST_MODEL))
    elif intent == "needs_tools":
        yield from stream(_agent_loop(conn, question, history, sources, trace))
    else:  # simple_qa fast path
        with trace.stage("rewrite") as p:
            query = rewrite_query(question, history)
            p["rewritten_query"] = query
        with trace.stage("retrieve") as p:
            res = retrieval.hybrid_search(conn, query)
            sources = res["chunks"]
            p.update(max_cosine=res["max_cosine"], confident=res["confident"],
                     chunks=[{k: c[k] for k in ("chunk_id", "title", "page_start", "score", "cosine")}
                             for c in sources])
        if not res["confident"]:
            sources = []
            yield from stream([prompts.NOT_FOUND])
        else:
            msgs = ([{"role": "system",
                      "content": prompts.ANSWER_SYSTEM.format(sources=format_sources(sources))}]
                    + history + [{"role": "user", "content": question}])
            with trace.stage("generate"):
                yield from stream(llm.chat_stream(msgs))

    answer = "".join(parts)
    valid = validate_citations(answer, len(sources))
    citations = [{"marker": n, "chunk_id": sources[n - 1]["chunk_id"],
                  "paper_id": sources[n - 1]["paper_id"], "title": sources[n - 1]["title"],
                  "page_start": sources[n - 1]["page_start"], "page_end": sources[n - 1]["page_end"],
                  "snippet": sources[n - 1]["text"][:300]} for n in valid]

    cur = conn.execute("INSERT INTO messages(chat_id, role, content) VALUES (?,?,?)",
                       (chat_id, "assistant", answer))
    message_id = cur.lastrowid
    conn.executemany("INSERT INTO citations(message_id, chunk_id, marker) VALUES (?,?,?)",
                     [(message_id, c["chunk_id"], c["marker"]) for c in citations])
    trace.persist(conn, message_id)
    conn.commit()

    yield {"type": "citations", "citations": citations}
    yield {"type": "done", "message_id": message_id}
