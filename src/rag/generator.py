"""Prompt-Bau und Antwortgenerierung."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from . import retriever, store
from .config import Config
from .errors import IndexNotReady
from .ollama_client import OllamaClient
from .retriever import Hit

NO_ANSWER = "Dazu finde ich nichts in den Dokumenten"
SYSTEM_PROMPT = (
    "Du beantwortest Fragen ausschließlich auf Basis des gelieferten Kontexts. "
    "Antworte auf Deutsch, knapp und sachlich. Nenne die verwendeten Quellen im Text als [1], [2] usw. "
    f"Wenn der Kontext die Antwort nicht enthält, antworte ausdrücklich: \"{NO_ANSWER}.\" "
    "Erfinde nichts und nutze kein Wissen außerhalb des Kontexts."
)


@dataclass(frozen=True)
class Answer:
    """Antworttext und die nummerierten Quellen (Reihenfolge = Nummer 1..n)."""
    text: str
    hits: list[Hit]


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    """Baut die Chat-Nachrichten mit nummeriertem Kontext."""
    context = "\n\n".join(
        f"[{i}] Datei: {h.path} | Abschnitt: {h.heading_path}\n{h.text}" for i, h in enumerate(hits, 1))
    user = f"Kontext:\n{context}\n\nFrage: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def clean_answer(text: str) -> str:
    """Entfernt eventuelle <think>-Blöcke und umgebende Leerzeichen."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def format_sources(hits: list[Hit]) -> str:
    """Listet die Quellen mit Dateipfad und Überschriftenpfad als Text auf."""
    return "\n".join(f"[{i}] {h.path} ({h.heading_path})" for i, h in enumerate(hits, 1))


def answer_question(conn: sqlite3.Connection, cfg: Config, client: OllamaClient,
                    question: str, mode: str = "hybrid") -> Answer:
    """Führt Retrieval und Generierung aus und liefert Antwort samt Quellen."""
    if store.stats(conn)["chunks"] == 0:
        raise IndexNotReady("Der Index ist leer. Bitte zuerst `rag index` ausführen.")
    hits = retriever.search(conn, cfg, client, question, mode)
    if not hits:
        return Answer(NO_ANSWER + ".", [])
    return Answer(clean_answer(client.chat(build_messages(question, hits))), hits)
