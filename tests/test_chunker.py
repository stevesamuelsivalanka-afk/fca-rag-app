from backend.app.rag.chunker import chunk_text

def test_chunking():
    chunks=chunk_text('Paragraph one.\n\nParagraph two.', {'year':2025}, max_chars=30, overlap=5)
    assert chunks
    assert all(c.metadata['year']==2025 for c in chunks)
