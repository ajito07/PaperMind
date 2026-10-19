import pytest
from fastapi.testclient import TestClient

from backend import config, main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    return TestClient(main.app)


def test_chat_locked_while_ingesting(client, monkeypatch):
    monkeypatch.setattr(main.agent, "respond",
                        lambda conn, cid, q: iter([{"type": "done", "message_id": 1}]))
    chat_id = client.post("/chats").json()["id"]
    from backend import db
    conn = db.connect()
    conn.execute("INSERT INTO papers(title, filename, sha256, status) "
                 "VALUES ('t','f.pdf','abc','processing')")
    conn.commit()

    r = client.post(f"/chats/{chat_id}/messages", json={"content": "hi"})
    assert r.status_code == 423
    assert "analyzed" in r.json()["detail"]

    conn.execute("UPDATE papers SET status='failed'")  # failed papers must NOT lock chat
    conn.commit()
    conn.close()
    # respond() would call Ollama; reaching it means the lock is open
    r = client.post(f"/chats/{chat_id}/messages", json={"content": "hi"})
    assert r.status_code != 423


def test_message_to_missing_chat_404s(client):
    assert client.post("/chats/999/messages", json={"content": "hi"}).status_code == 404
