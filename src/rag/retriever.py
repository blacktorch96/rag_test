"""Hybride Suche: Vektor und BM25, zusammengeführt per Reciprocal Rank Fusion."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import store
from .config import Config
from .ollama_client import OllamaClient

MODES = ("hybrid", "vector", "bm25")


@dataclass(frozen=True)
class Hit:
    """Ein Suchtreffer mit Quelle und Score."""
    chunk_id: int
    path: str
    heading_path: str
    text: str
    score: float


def rrf(rankings: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: Summe von 1/(k+rang) je Liste, beste zuerst."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def search(conn: sqlite3.Connection, cfg: Config, client: OllamaClient, question: str,
           mode: str = "hybrid", k: int | None = None,
           query_vec: list[float] | None = None) -> list[Hit]:
    """Sucht die besten Chunks im gewählten Modus (`hybrid`, `vector`, `bm25`)."""
    if mode not in MODES:
        raise ValueError(f"Unbekannter Suchmodus: {mode}")
    r = cfg.retrieval
    rankings: list[list[int]] = []
    if mode in ("hybrid", "vector"):
        store.check_model(conn, cfg.ollama.embed_model)
        if store.get_meta(conn, "embedding_dim") is not None:
            vec = query_vec or client.embed([question])[0]
            rankings.append(store.search_vec(conn, vec, r.top_k_vector))
    if mode in ("hybrid", "bm25"):
        rankings.append(store.search_bm25(conn, question, r.top_k_bm25))
    fused = rrf(rankings, r.rrf_k)[: k or r.final_k]
    rows = store.load_chunks(conn, [cid for cid, _ in fused])
    return [Hit(cid, rows[cid].path, rows[cid].heading_path, rows[cid].text, score)
            for cid, score in fused if cid in rows]
