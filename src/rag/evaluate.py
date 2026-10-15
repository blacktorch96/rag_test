"""Auswertung der Suche anhand von Testfragen (Recall@k, MRR)."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from . import retriever
from .config import Config
from .errors import ConfigError
from .ollama_client import OllamaClient

KS = (1, 3, 6)
POOL = 20  # so viele Chunks werden je Frage geholt, bevor auf Dateien reduziert wird


@dataclass
class ModeReport:
    """Kennzahlen eines Suchmodus; `ranks` hat je Frage den Rang oder None."""
    mode: str
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ranks: list[int | None] = field(default_factory=list)


def load_questions(path: Path) -> list[dict[str, str]]:
    """Liest die JSONL-Datei mit `question` und `expected_path`."""
    if not path.is_file():
        raise ConfigError(f"Fragendatei nicht gefunden: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ConfigError(f"Fragendatei ist leer: {path}")
    return rows


def first_rank(paths: list[str], expected: str) -> int | None:
    """Rang (ab 1) der erwarteten Datei in der Trefferliste, doppelte Dateien zählen einmal."""
    unique = list(dict.fromkeys(paths))
    return unique.index(expected) + 1 if expected in unique else None


def summarize(mode: str, ranks: list[int | None]) -> ModeReport:
    """Berechnet Recall@k und MRR aus den Rängen."""
    n = len(ranks)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    mrr = sum(1 / r for r in ranks if r) / n
    return ModeReport(mode, recall, mrr, ranks)


def run_eval(conn: sqlite3.Connection, cfg: Config, client: OllamaClient,
             questions: list[dict[str, str]]) -> list[ModeReport]:
    """Wertet alle drei Modi aus; das Query-Embedding wird je Frage nur einmal berechnet."""
    ranks: dict[str, list[int | None]] = {m: [] for m in retriever.MODES}
    for q in questions:
        vec = client.embed([q["question"]])[0]
        for mode in retriever.MODES:
            hits = retriever.search(conn, cfg, client, q["question"], mode, POOL, vec)
            ranks[mode].append(first_rank([h.path for h in hits], q["expected_path"]))
    return [summarize(m, ranks[m]) for m in retriever.MODES]
