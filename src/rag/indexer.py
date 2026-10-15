"""Orchestriert Loader, Chunker, Embedder und Store (inkrementell)."""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import loader, store
from .chunker import Chunk, chunk_markdown
from .config import Config
from .ollama_client import OllamaClient

log = logging.getLogger(__name__)
BATCH_SIZE = 32


@dataclass
class IndexResult:
    """Zusammenfassung eines Indexierungslaufs."""
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0


def plan_changes(stored: dict[str, str], current: dict[str, str]) -> tuple[list[str], list[str]]:
    """Bestimmt neue/geänderte und entfernte Pfade anhand der Hashes."""
    changed = sorted(p for p, h in current.items() if stored.get(p) != h)
    removed = sorted(set(stored) - set(current))
    return changed, removed


def embed_chunks(client: OllamaClient, chunks: list[Chunk]) -> list[list[float]]:
    """Embeddet Chunks in Batches; der Überschriftenpfad wird dem Text vorangestellt."""
    texts = [f"{c.heading_path}\n{c.text}" for c in chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(client.embed(texts[start:start + BATCH_SIZE]))
    return vectors


def run_index(cfg: Config, client: OllamaClient, limit: int | None = None, rebuild: bool = False,
              progress: Callable[[int, int, str], None] | None = None) -> IndexResult:
    """Indexiert neue und geänderte Dateien und entfernt gelöschte aus dem Index."""
    db_path: Path = cfg.store.path
    if rebuild:
        for suffix in ("", "-wal", "-shm", "-journal"):
            Path(str(db_path) + suffix).unlink(missing_ok=True)
    files = loader.scan(cfg.corpus.root)
    current = {rel: loader.sha256_bytes(p.read_bytes()) for rel, p in files.items()}
    conn = store.connect(db_path)
    try:
        store.check_model(conn, cfg.ollama.embed_model)
        changed, removed = plan_changes(store.document_hashes(conn), current)
        result = IndexResult(skipped=len(current) - len(changed), removed=len(removed))
        for rel in removed:
            store.delete_document(conn, rel)
        conn.commit()
        todo = changed[:limit] if limit else changed
        for i, rel in enumerate(todo, 1):
            if progress:
                progress(i, len(todo), rel)
            result.chunks += _index_file(conn, cfg, client, rel, files[rel])
            result.indexed += 1
        return result
    finally:
        conn.close()


def _index_file(conn: sqlite3.Connection, cfg: Config, client: OllamaClient, rel: str, path: Path) -> int:
    """Indexiert eine Datei; Embedding zuerst, damit Fehler den alten Stand nicht zerstören."""
    doc = loader.read_document(rel, path)
    title = doc.meta.get("title") or Path(rel).stem
    chunks = chunk_markdown(doc.body, cfg.chunking, title)
    vectors = embed_chunks(client, chunks) if chunks else []
    if vectors:
        store.ensure_vec(conn, cfg.ollama.embed_model, len(vectors[0]))
    store.delete_document(conn, rel)
    store.add_document(conn, rel, doc.sha256, chunks, vectors)
    conn.commit()
    log.info("%s: %d Chunks", rel, len(chunks))
    return len(chunks)
