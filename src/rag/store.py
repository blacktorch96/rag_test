"""SQLite-Schema, Schreiboperationen und Suche (sqlite-vec plus FTS5)."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from .chunker import Chunk
from .errors import IndexNotReady, ModelMismatch

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL, indexed_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL, heading_path TEXT NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    heading_path, text, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@dataclass(frozen=True)
class ChunkRow:
    """Ein Chunk samt Dokumentpfad aus der Datenbank."""
    chunk_id: int
    path: str
    heading_path: str
    text: str


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    """Öffnet die Datenbank mit geladenem sqlite-vec; schreibgeschützt oder mit Schema."""
    if readonly:
        if not path.is_file():
            raise IndexNotReady(f"Kein Index gefunden ({path}). Bitte zuerst `rag index` ausführen.")
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    if not readonly:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Liest einen Wert aus der Meta-Tabelle."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Schreibt einen Wert in die Meta-Tabelle."""
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def check_model(conn: sqlite3.Connection, model: str) -> None:
    """Bricht ab, wenn der Index mit einem anderen Embedding-Modell erzeugt wurde."""
    stored = get_meta(conn, "embedding_model")
    if stored is not None and stored != model:
        raise ModelMismatch(
            f"Der Index wurde mit '{stored}' erzeugt, konfiguriert ist '{model}'. "
            "Bitte `rag index --rebuild` ausführen.")


def ensure_vec(conn: sqlite3.Connection, model: str, dim: int) -> None:
    """Legt die Vektortabelle an und merkt sich Modell und Dimension."""
    stored = get_meta(conn, "embedding_dim")
    if stored is not None and int(stored) != dim:
        raise ModelMismatch(f"Embedding-Dimension {dim} passt nicht zum Index ({stored}).")
    conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
                 f"embedding float[{dim}] distance_metric=cosine)")
    set_meta(conn, "embedding_model", model)
    set_meta(conn, "embedding_dim", str(dim))


def document_hashes(conn: sqlite3.Connection) -> dict[str, str]:
    """Liefert Pfad -> SHA-256 aller indexierten Dokumente."""
    return dict(conn.execute("SELECT path, sha256 FROM documents"))


def delete_document(conn: sqlite3.Connection, path: str) -> None:
    """Entfernt ein Dokument mit allen Chunks, FTS- und Vektoreinträgen."""
    row = conn.execute("SELECT id FROM documents WHERE path = ?", (path,)).fetchone()
    if row is None:
        return
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (row[0],))]
    for table in ("chunks_fts", "chunks_vec"):
        if table == "chunks_vec" and get_meta(conn, "embedding_dim") is None:
            continue
        conn.executemany(f"DELETE FROM {table} WHERE rowid = ?", [(i,) for i in ids])
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (row[0],))
    conn.execute("DELETE FROM documents WHERE id = ?", (row[0],))


def add_document(conn: sqlite3.Connection, path: str, sha: str,
                 chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Fügt ein Dokument mit Chunks und Embeddings ein (Commit durch den Aufrufer)."""
    doc_id = conn.execute("INSERT INTO documents (path, sha256, indexed_at) "
                          "VALUES (?, ?, datetime('now', 'localtime'))", (path, sha)).lastrowid
    for ord_, (chunk, vec) in enumerate(zip(chunks, vectors)):
        cid = conn.execute(
            "INSERT INTO chunks (document_id, ord, heading_path, text, token_count) VALUES (?,?,?,?,?)",
            (doc_id, ord_, chunk.heading_path, chunk.text, chunk.token_count)).lastrowid
        conn.execute("INSERT INTO chunks_fts (rowid, heading_path, text) VALUES (?,?,?)",
                     (cid, chunk.heading_path, chunk.text))
        conn.execute("INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                     (cid, sqlite_vec.serialize_float32(vec)))


def stats(conn: sqlite3.Connection) -> dict[str, object]:
    """Liefert Anzahl Dokumente/Chunks, letzte Indexierung und das Embedding-Modell."""
    docs, last = conn.execute("SELECT COUNT(*), MAX(indexed_at) FROM documents").fetchone()
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return {"documents": docs, "chunks": chunks, "last_indexed": last,
            "embedding_model": get_meta(conn, "embedding_model")}


def search_vec(conn: sqlite3.Connection, vector: list[float], k: int) -> list[int]:
    """Gibt die Chunk-IDs der k ähnlichsten Vektoren zurück (beste zuerst)."""
    rows = conn.execute("SELECT rowid FROM chunks_vec WHERE embedding MATCH ? AND k = ? "
                        "ORDER BY distance", (sqlite_vec.serialize_float32(vector), k))
    return [r[0] for r in rows]


def search_bm25(conn: sqlite3.Connection, question: str, k: int) -> list[int]:
    """Volltextsuche (BM25); die Frage wird in ODER-verknüpfte Wörter zerlegt."""
    words = re.findall(r"\w+", question)
    if not words:
        return []
    query = " OR ".join('"' + w + '"' for w in words)
    rows = conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                        "ORDER BY bm25(chunks_fts) LIMIT ?", (query, k))
    return [r[0] for r in rows]


def load_chunks(conn: sqlite3.Connection, ids: list[int]) -> dict[int, ChunkRow]:
    """Lädt Chunks samt Dokumentpfad zu den gegebenen IDs."""
    marks = ",".join("?" * len(ids))
    rows = conn.execute("SELECT c.id, d.path, c.heading_path, c.text FROM chunks c "
                        f"JOIN documents d ON d.id = c.document_id WHERE c.id IN ({marks})", ids)
    return {r[0]: ChunkRow(*r) for r in rows}
