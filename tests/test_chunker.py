"""Tests für den Chunker."""
from rag.chunker import chunk_markdown, estimate_tokens
from rag.config import ChunkingCfg

CFG = ChunkingCfg(target_tokens=100, max_tokens=160, overlap_tokens=20)


def test_heading_paths():
    """Überschriften bilden den Pfad, Titel dient als Fallback."""
    body = "i " * 100 + "\n\n# A\n\n" + "x " * 100 + "\n\n## B\n\n" + "y " * 100
    paths = [c.heading_path for c in chunk_markdown(body, CFG, "Titel")]
    assert paths[0] == "Titel"
    assert "A" in paths and "A > B" in paths


def test_empty_document():
    """Leerer Text ergibt keine Chunks."""
    assert chunk_markdown("", CFG) == []


def test_small_sections_are_merged():
    """Kleine Abschnitte werden verschmolzen."""
    chunks = chunk_markdown("# A\n\nkurz\n\n# B\n\nauch kurz\n", CFG)
    assert len(chunks) == 1 and "kurz" in chunks[0].text and "auch kurz" in chunks[0].text


def test_large_section_split_by_paragraphs():
    """Große Abschnitte werden an Absatzgrenzen geteilt und bleiben unter max_tokens."""
    body = "# A\n\n" + "\n\n".join(f"Absatz {i}. " + "wort " * 60 for i in range(8))
    chunks = chunk_markdown(body, CFG)
    assert len(chunks) > 2
    assert all(c.token_count <= CFG.max_tokens for c in chunks)


def test_code_block_not_split():
    """Ein Code-Block mit Leerzeilen bleibt in einem Chunk; # im Code ist keine Überschrift."""
    code = "```python\n# kein Titel\n\nprint(1)\n\nprint(2)\n```"
    chunks = chunk_markdown(f"# A\n\n{code}\n", CFG)
    assert any(code in c.text for c in chunks)
    assert all(c.heading_path == "A" for c in chunks)


def test_table_not_split():
    """Eine Tabelle über max_tokens wird nicht zerschnitten."""
    table = "\n".join(f"| zeile {i} | {'w' * 30} |" for i in range(30))
    chunks = chunk_markdown(f"# T\n\n{table}\n", CFG)
    assert any(table in c.text for c in chunks)


def test_giant_paragraph_hard_split():
    """Ein einzelner Riesen-Absatz ohne Satzzeichen wird hart geteilt."""
    chunks = chunk_markdown("# A\n\n" + "a" * 5000, CFG)
    assert len(chunks) > 1 and all(c.token_count <= CFG.max_tokens for c in chunks)


def test_estimate_tokens_minimum():
    """Mindestens ein Token."""
    assert estimate_tokens("") == 1
