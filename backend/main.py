"""ResearchMind API — single FastAPI process. Run: uvicorn backend.main:app"""
import hashlib
import json

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import agent, config, db, ingest, llm, retrieval

app = FastAPI(title="ResearchMind")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                   allow_methods=["*"], allow_headers=["*"])


def get_conn():
    return db.connect()


@app.get("/health")
def health():
    return llm.health()


# ---------- papers ----------

@app.post("/papers", status_code=202)
def upload_paper(file: UploadFile, background: BackgroundTasks):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are supported")
    data = file.file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(400, "file too large (100MB max)")
    sha = hashlib.sha256(data).hexdigest()
    conn = get_conn()
    try:
        dup = conn.execute("SELECT id FROM papers WHERE sha256=?", (sha,)).fetchone()
        if dup:
            raise HTTPException(409, f"already uploaded (paper id {dup['id']})")
        config.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{sha[:12]}.pdf"
        (config.PAPERS_DIR / filename).write_bytes(data)
        cur = conn.execute(
            "INSERT INTO papers(title, filename, sha256) VALUES (?,?,?)",
            (file.filename, filename, sha))
        conn.commit()
        paper_id = cur.lastrowid
    finally:
        conn.close()
    background.add_task(ingest.ingest_paper, paper_id)
    return {"id": paper_id, "status": "pending"}


@app.get("/papers")
def list_papers():
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, title, authors, year, num_pages, status, error, added_at "
            "FROM papers ORDER BY id DESC")]
    finally:
        conn.close()


@app.get("/papers/{paper_id}")
def get_paper(paper_id: int):
    conn = get_conn()
    try:
        r = conn.execute("SELECT id, title, authors, year, num_pages, status, summary, error, "
                         "added_at FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not r:
            raise HTTPException(404, "paper not found")
        return dict(r)
    finally:
        conn.close()


@app.delete("/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: int):
    conn = get_conn()
    try:
        r = conn.execute("SELECT filename FROM papers WHERE id=?", (paper_id,)).fetchone()
        if not r:
            raise HTTPException(404, "paper not found")
        conn.execute("DELETE FROM papers WHERE id=?", (paper_id,))
        conn.commit()
        (config.PAPERS_DIR / r["filename"]).unlink(missing_ok=True)
    finally:
        conn.close()
    retrieval.invalidate_cache()


@app.get("/papers/{paper_id}/pdf")
def get_pdf(paper_id: int):
    conn = get_conn()
    try:
        r = conn.execute("SELECT filename, title FROM papers WHERE id=?", (paper_id,)).fetchone()
    finally:
        conn.close()
    if not r:
        raise HTTPException(404, "paper not found")
    return FileResponse(config.PAPERS_DIR / r["filename"], media_type="application/pdf")


@app.post("/papers/{paper_id}/summary")
def summarize_paper(paper_id: int):
    conn = get_conn()
    try:
        if not conn.execute("SELECT 1 FROM papers WHERE id=?", (paper_id,)).fetchone():
            raise HTTPException(404, "paper not found")
        return {"summary": agent.summarize(conn, paper_id)}
    finally:
        conn.close()


# ---------- search (debug/explore + eval) ----------

class SearchRequest(BaseModel):
    query: str
    k: int = config.TOP_K_FUSED
    paper_id: int | None = None


@app.post("/search")
def search(req: SearchRequest):
    conn = get_conn()
    try:
        return retrieval.hybrid_search(conn, req.query, k=req.k, paper_id=req.paper_id)
    finally:
        conn.close()


# ---------- chats ----------

@app.post("/chats")
def create_chat():
    conn = get_conn()
    try:
        cur = conn.execute("INSERT INTO chats DEFAULT VALUES")
        conn.commit()
        return {"id": cur.lastrowid, "title": "New chat"}
    finally:
        conn.close()


@app.get("/chats")
def list_chats():
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT id, title, created_at FROM chats ORDER BY id DESC")]
    finally:
        conn.close()


@app.get("/chats/{chat_id}")
def get_chat(chat_id: int):
    conn = get_conn()
    try:
        chat = conn.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not chat:
            raise HTTPException(404, "chat not found")
        messages = []
        for m in conn.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY id", (chat_id,)):
            cits = [dict(c) for c in conn.execute(
                "SELECT ci.marker, ci.chunk_id, ch.paper_id, ch.page_start, ch.page_end, p.title, "
                "substr(ch.text, 1, 300) AS snippet "
                "FROM citations ci JOIN chunks ch ON ch.id = ci.chunk_id "
                "JOIN papers p ON p.id = ch.paper_id "
                "WHERE ci.message_id=? ORDER BY ci.marker", (m["id"],))]
            messages.append({**dict(m), "citations": cits})
        return {**dict(chat), "messages": messages}
    finally:
        conn.close()


class MessageRequest(BaseModel):
    content: str


@app.post("/chats/{chat_id}/messages")
def send_message(chat_id: int, req: MessageRequest):
    conn = get_conn()
    if not conn.execute("SELECT 1 FROM chats WHERE id=?", (chat_id,)).fetchone():
        conn.close()
        raise HTTPException(404, "chat not found")
    busy = conn.execute("SELECT count(*) FROM papers WHERE status IN ('pending','processing')"
                        ).fetchone()[0]
    if busy:
        conn.close()
        raise HTTPException(423, f"{busy} paper(s) still being analyzed — chat unlocks when "
                                 "ingestion finishes")

    def events():
        try:
            for ev in agent.respond(conn, chat_id, req.content):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            conn.close()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


@app.get("/messages/{message_id}/trace")
def get_trace(message_id: int):
    conn = get_conn()
    try:
        rows = conn.execute("SELECT stage, payload, duration_ms FROM traces "
                            "WHERE message_id=? ORDER BY id", (message_id,)).fetchall()
        if not rows:
            raise HTTPException(404, "no trace for this message")
        return [{"stage": r["stage"], "duration_ms": r["duration_ms"],
                 "payload": json.loads(r["payload"])} for r in rows]
    finally:
        conn.close()
