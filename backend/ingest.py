"""PDF → page-tracked chunks → embeddings → SQLite."""
import bisect
import json
import re
from dataclasses import dataclass

import fitz  # PyMuPDF
import numpy as np

from . import config, db, llm, prompts


@dataclass
class Chunk:
    text: str
    page_start: int
    page_end: int


def chunk_pages(pages: list[tuple[int, str]],
                size: int = config.CHUNK_CHARS,
                overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Sliding window over concatenated page text; every chunk keeps its page range.

    ponytail: no heading-aware splitting — paragraph/sentence boundary snapping
    is enough until eval says otherwise.
    """
    starts, page_nos, parts = [], [], []
    pos = 0
    for no, text in pages:
        text = text.strip()
        if not text:
            continue
        text += "\n\n"
        starts.append(pos)
        page_nos.append(no)
        parts.append(text)
        pos += len(text)
    full = "".join(parts)

    def page_of(offset: int) -> int:
        return page_nos[bisect.bisect_right(starts, offset) - 1]

    chunks = []
    start = 0
    while start < len(full):
        end = min(start + size, len(full))
        if end < len(full):
            cut = full.rfind("\n\n", start + size // 2, end)
            if cut == -1:
                cut = full.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        text = full[start:end].strip()
        if text:
            chunks.append(Chunk(text, page_of(start), page_of(end - 1)))
        if end >= len(full):
            break
        start = end - overlap
    return chunks


def _extract_metadata(doc, first_page_text: str) -> dict:
    title = (doc.metadata.get("title") or "").strip()
    if len(title) < 5:
        # 8B fallback: pull title/authors/year off page 1
        try:
            msg = llm.chat(
                [{"role": "user", "content": prompts.METADATA.format(text=first_page_text[:2500])}],
                model=config.FAST_MODEL,
            )
            meta = json.loads(re.search(r"\{.*\}", msg["content"], re.S).group())
            return {
                "title": str(meta.get("title") or "")[:300] or "Untitled",
                "authors": str(meta.get("authors") or "")[:300] or None,
                "year": int(meta["year"]) if str(meta.get("year") or "").isdigit() else None,
            }
        except Exception:
            first_line = next((l.strip() for l in first_page_text.splitlines() if len(l.strip()) > 5),
                              "Untitled")
            return {"title": first_line[:300], "authors": None, "year": None}
    return {"title": title[:300], "authors": (doc.metadata.get("author") or None), "year": None}


def ingest_paper(paper_id: int) -> None:
    """Background task: parse, chunk, embed, store. Sets papers.status."""
    conn = db.connect()
    try:
        row = conn.execute("SELECT * FROM papers WHERE id=?", (paper_id,)).fetchone()
        conn.execute("UPDATE papers SET status='processing' WHERE id=?", (paper_id,))
        conn.commit()

        doc = fitz.open(config.PAPERS_DIR / row["filename"])
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
        # first page is often an image-only cover; use the first page with text
        first_text = next((t for _, t in pages if t.strip()), "")
        meta = _extract_metadata(doc, first_text)
        chunks = chunk_pages(pages)
        if not chunks:
            raise ValueError("no extractable text (scanned PDF? OCR not supported)")

        embeddings = []
        for i in range(0, len(chunks), 32):
            embeddings += llm.embed([c.text for c in chunks[i:i + 32]], "search_document: ")

        conn.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))  # idempotent re-ingest
        for c, e in zip(chunks, embeddings):
            conn.execute(
                "INSERT INTO chunks(paper_id, page_start, page_end, text, embedding) VALUES (?,?,?,?,?)",
                (paper_id, c.page_start, c.page_end, c.text,
                 np.asarray(e, dtype=np.float32).tobytes()),
            )
        conn.execute(
            "UPDATE papers SET title=?, authors=?, year=?, num_pages=?, status='ready', error=NULL WHERE id=?",
            (meta["title"], meta["authors"], meta["year"], len(doc), paper_id),
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        conn.execute("UPDATE papers SET status='failed', error=? WHERE id=?", (str(e)[:500], paper_id))
        conn.commit()
    finally:
        conn.close()
        from . import retrieval
        retrieval.invalidate_cache()
