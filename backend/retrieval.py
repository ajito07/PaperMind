"""Hybrid retrieval: FTS5 BM25 + brute-force cosine over embeddings, fused with RRF."""
import re
import threading

import numpy as np

from . import config, llm

# ponytail: whole-corpus embedding matrix cached in RAM (~30MB / 10k chunks);
# swap for an ANN index if the library ever outgrows brute force.
_cache_lock = threading.Lock()
_cache = None  # (chunk_ids: np.ndarray, matrix: np.ndarray unit-normalized)


def invalidate_cache():
    global _cache
    with _cache_lock:
        _cache = None


def _matrix(conn):
    global _cache
    with _cache_lock:
        if _cache is None:
            rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            if len(rows):
                m = np.frombuffer(b"".join(r["embedding"] for r in rows), dtype=np.float32)
                m = m.reshape(len(rows), config.EMBED_DIM).copy()
                m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
            else:
                m = np.zeros((0, config.EMBED_DIM), dtype=np.float32)
            _cache = (ids, m)
        return _cache


def vector_search(conn, query: str, k: int, paper_id: int | None = None) -> list[tuple[int, float]]:
    """Returns [(chunk_id, cosine)] best-first."""
    ids, m = _matrix(conn)
    if not len(ids):
        return []
    q = np.asarray(llm.embed([query], "search_query: ")[0], dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-9
    sims = m @ q
    if paper_id is not None:
        keep = {r["id"] for r in conn.execute("SELECT id FROM chunks WHERE paper_id=?", (paper_id,))}
        mask = np.array([i in keep for i in ids])
        sims = np.where(mask, sims, -1.0)
    top = np.argsort(sims)[::-1][:k]
    return [(int(ids[i]), float(sims[i])) for i in top if sims[i] > -1.0]


def bm25_search(conn, query: str, k: int, paper_id: int | None = None) -> list[int]:
    """Returns chunk_ids best-first. Query is sanitized into quoted OR terms."""
    terms = re.findall(r"\w+", query)
    if not terms:
        return []
    match = " OR ".join(f'"{t}"' for t in terms)
    sql = ("SELECT chunks_fts.rowid AS cid FROM chunks_fts "
           "JOIN chunks ON chunks.id = chunks_fts.rowid WHERE chunks_fts MATCH ?")
    args = [match]
    if paper_id is not None:
        sql += " AND chunks.paper_id = ?"
        args.append(paper_id)
    sql += " ORDER BY chunks_fts.rank LIMIT ?"
    args.append(k)
    return [r["cid"] for r in conn.execute(sql, args)]


def rrf(ranked_lists: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def hybrid_search(conn, query: str, k: int = config.TOP_K_FUSED,
                  paper_id: int | None = None) -> dict:
    """Returns {"chunks": [...], "max_cosine": float, "confident": bool}."""
    vec = vector_search(conn, query, config.TOP_K_EACH, paper_id)
    bm = bm25_search(conn, query, config.TOP_K_EACH, paper_id)
    cosine_by_id = dict(vec)
    fused = rrf([[cid for cid, _ in vec], bm])
    top_ids = sorted(fused, key=fused.get, reverse=True)[:k]

    chunks = []
    for cid in top_ids:
        r = conn.execute(
            "SELECT c.id, c.paper_id, c.page_start, c.page_end, c.text, p.title "
            "FROM chunks c JOIN papers p ON p.id = c.paper_id WHERE c.id=?", (cid,)).fetchone()
        if r:
            chunks.append({
                "chunk_id": r["id"], "paper_id": r["paper_id"], "title": r["title"],
                "page_start": r["page_start"], "page_end": r["page_end"], "text": r["text"],
                "score": round(fused[cid], 5), "cosine": round(cosine_by_id.get(cid, 0.0), 4),
            })
    max_cosine = max((c["cosine"] for c in chunks), default=0.0)
    return {"chunks": chunks, "max_cosine": max_cosine,
            "confident": max_cosine >= config.MIN_COSINE}
