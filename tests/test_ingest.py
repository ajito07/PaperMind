from backend.ingest import chunk_pages


def pages_of(n_pages, chars_per_page=3000):
    para = "The quick brown fox studies transformer attention. " * 12 + "\n\n"
    page = (para * (chars_per_page // len(para) + 1))[:chars_per_page]
    return [(i + 1, page) for i in range(n_pages)]


def test_chunks_cover_all_pages_and_are_ordered():
    chunks = chunk_pages(pages_of(10))
    assert chunks, "should produce chunks"
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 10
    for c in chunks:
        assert 1 <= c.page_start <= c.page_end <= 10
    starts = [c.page_start for c in chunks]
    assert starts == sorted(starts)


def test_chunk_size_bounds():
    chunks = chunk_pages(pages_of(5), size=1000, overlap=200)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_overlap_repeats_content():
    chunks = chunk_pages(pages_of(3), size=1000, overlap=300)
    assert len(chunks) >= 2
    # tail of chunk i overlaps head of chunk i+1
    assert chunks[0].text[-100:].strip()[:40] in chunks[1].text[:500]


def test_empty_and_blank_pages():
    assert chunk_pages([]) == []
    assert chunk_pages([(1, "   "), (2, "")]) == []
    chunks = chunk_pages([(1, ""), (2, "real content here on page two.")])
    assert chunks[0].page_start == 2
