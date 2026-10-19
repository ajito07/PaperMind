from backend import agent


def test_validate_citations_keeps_real_markers():
    assert agent.validate_citations("Claim [1]. Another [2][3].", 3) == [1, 2, 3]


def test_validate_citations_drops_invented_markers():
    assert agent.validate_citations("Fact [1], fake [7], huge [99].", 2) == [1]
    assert agent.validate_citations("No citations at all.", 5) == []
    assert agent.validate_citations("[3]", 0) == []


def test_router_falls_back_to_simple_qa_on_llm_failure(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("ollama down")
    monkeypatch.setattr(agent.llm, "chat", boom)
    assert agent.route_intent("what is attention?") == "simple_qa"


def test_router_parses_model_output(monkeypatch):
    monkeypatch.setattr(agent.llm, "chat", lambda *a, **k: {"content": " Chitchat.\n"})
    assert agent.route_intent("hello!") == "chitchat"
    monkeypatch.setattr(agent.llm, "chat", lambda *a, **k: {"content": "needs_tools"})
    assert agent.route_intent("compare paper 1 and 2") == "needs_tools"
    monkeypatch.setattr(agent.llm, "chat", lambda *a, **k: {"content": "banana"})
    assert agent.route_intent("?") == "simple_qa"


def test_rewrite_falls_back_to_question(monkeypatch):
    monkeypatch.setattr(agent.llm, "chat", lambda *a, **k: {"content": "x" * 900})
    q = "what dataset did they use?"
    assert agent.rewrite_query(q, [{"role": "user", "content": "hi"}]) == q
    assert agent.rewrite_query(q, []) == q  # no history → no LLM call needed


def test_format_sources_numbering_and_pages():
    src = [{"title": "T", "page_start": 3, "page_end": 3, "text": "abc"},
           {"title": "U", "page_start": 4, "page_end": 6, "text": "def"}]
    out = agent.format_sources(src)
    assert "[1] T (p.3): abc" in out
    assert "[2] U (p.4-6): def" in out
