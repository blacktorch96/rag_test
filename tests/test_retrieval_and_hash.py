"""Tests für RRF, Hash-Logik, Frontmatter und Evaluationskennzahlen."""
from rag.evaluate import first_rank, summarize
from rag.indexer import plan_changes
from rag.loader import sha256_bytes, split_frontmatter
from rag.retriever import rrf


def test_rrf_prefers_items_in_both_lists():
    """Treffer in beiden Listen schlagen Treffer in nur einer."""
    fused = rrf([[1, 2, 3], [3, 4, 1]], k=60)
    assert [i for i, _ in fused][:2] == [1, 3]
    assert fused[0][1] > fused[-1][1]


def test_rrf_empty_and_single():
    """Leere Eingaben und eine einzelne Liste funktionieren."""
    assert rrf([], 60) == [] and rrf([[]], 60) == []
    assert [i for i, _ in rrf([[7, 5]], 60)] == [7, 5]


def test_plan_changes():
    """Unveränderte werden übersprungen, geänderte/neue indexiert, fehlende entfernt."""
    stored = {"a.md": "1", "b.md": "2", "c.md": "3"}
    current = {"a.md": "1", "b.md": "X", "d.md": "4"}
    assert plan_changes(stored, current) == (["b.md", "d.md"], ["c.md"])


def test_sha256_stable():
    """Gleicher Inhalt, gleicher Hash; anderer Inhalt, anderer Hash."""
    assert sha256_bytes(b"abc") == sha256_bytes(b"abc") != sha256_bytes(b"abd")


def test_frontmatter():
    """Frontmatter wird abgetrennt, ohne Frontmatter bleibt der Text unverändert."""
    meta, body = split_frontmatter('---\ntitle: "Hallo"\ndate: 2024-01-01\n---\n# Text\n')
    assert meta == {"title": "Hallo", "date": "2024-01-01"} and body == "# Text\n"
    assert split_frontmatter("# Text") == ({}, "# Text")
    assert split_frontmatter("---\nkein ende") == ({}, "---\nkein ende")


def test_metrics():
    """Recall@k und MRR aus Rängen; Dateien zählen einmal."""
    assert first_rank(["a", "a", "b"], "b") == 2 and first_rank(["a"], "z") is None
    rep = summarize("hybrid", [1, 3, None, 6])
    assert rep.recall == {1: 0.25, 3: 0.5, 6: 0.75}
    assert abs(rep.mrr - (1 + 1 / 3 + 1 / 6) / 4) < 1e-9
