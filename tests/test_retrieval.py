import numpy as np
import pytest

from backend import config, db, retrieval


@pytest.fixture
def conn(tmp_path, monkeypatch):
    conn = db.connect(tmp_path / "test.db")
    retrieval.invalidate_cache()
    yield conn
    conn.close()
    retrieval.invalidate_cache()


def seed(conn, texts_and_vecs):
    conn.execute("INSERT INTO papers(id, title, filename, sha256) VALUES (1,'Paper A','a.pdf','x')")
    for i, (text, vec) in enumerate(texts_and_vecs):
        v = np.zeros(config.EMBED_DIM, dtype=np.float32)
        v[: len(vec)] = vec
        conn.execute(
            "INSERT INTO chunks(id, paper_id, page_start, page_end, text, embedding) "
            "VALUES (?,?,?,?,?,?)", (i + 1, 1, i + 1, i + 1, text, v.tobytes()))
    conn.commit()


def test_rrf_prefers_items_ranked_by_both():
    scores = retrieval.rrf([[1, 2, 3], [2, 1, 4]])
    assert scores[1] == scores[2] > scores[3]
    assert scores[3] > 0 and scores[4] > 0


def test_bm25_finds_keyword(conn):
    seed(conn, [("attention is all you need", [1, 0]),
                ("convolutional networks for images", [0, 1])])
    ids = retrieval.bm25_search(conn, "attention mechanism", k=5)
    assert ids[0] == 1


def test_bm25_survives_special_chars(conn):
    seed(conn, [("attention is all you need", [1, 0])])
    assert retrieval.bm25_search(conn, 'what is "attention" (really)? OR NOT', k=5) == [1]
    assert retrieval.bm25_search(conn, "???", k=5) == []


def test_vector_search_and_confidence_gate(conn, monkeypatch):
    seed(conn, [("about transformers", [1.0, 0.0]),
                ("about biology", [0.0, 1.0])])
    q = np.zeros(config.EMBED_DIM)
    q[0] = 1.0
    monkeypatch.setattr(retrieval.llm, "embed", lambda texts, prefix: [q.tolist()])

    res = retrieval.hybrid_search(conn, "transformers", k=2)
    assert res["chunks"][0]["chunk_id"] == 1
    assert res["max_cosine"] == pytest.approx(1.0, abs=1e-3)
    assert res["confident"]

    off = np.zeros(config.EMBED_DIM)
    off[10] = 1.0  # orthogonal to everything ingested
    monkeypatch.setattr(retrieval.llm, "embed", lambda texts, prefix: [off.tolist()])
    retrieval.invalidate_cache()
    res = retrieval.hybrid_search(conn, "zzzz qqqq", k=2)
    assert not res["confident"]


def test_paper_filter(conn, monkeypatch):
    seed(conn, [("attention paper text", [1.0, 0.0])])
    conn.execute("INSERT INTO papers(id, title, filename, sha256) VALUES (2,'Paper B','b.pdf','y')")
    v = np.zeros(config.EMBED_DIM, dtype=np.float32)
    v[0] = 1.0
    conn.execute("INSERT INTO chunks(id, paper_id, page_start, page_end, text, embedding) "
                 "VALUES (99, 2, 1, 1, 'attention in paper B', ?)", (v.tobytes(),))
    conn.commit()
    q = np.zeros(config.EMBED_DIM)
    q[0] = 1.0
    monkeypatch.setattr(retrieval.llm, "embed", lambda texts, prefix: [q.tolist()])
    res = retrieval.hybrid_search(conn, "attention", k=5, paper_id=2)
    assert {c["chunk_id"] for c in res["chunks"]} == {99}
