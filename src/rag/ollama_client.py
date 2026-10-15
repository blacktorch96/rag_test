"""HTTP-Wrapper für Ollama (Embeddings und Chat)."""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from .config import OllamaCfg
from .errors import OllamaError

log = logging.getLogger(__name__)


class OllamaClient:
    """Dünner Client mit Wiederholungsversuchen und lesbaren Fehlern."""

    def __init__(self, cfg: OllamaCfg, timeout: float = 600.0, retries: int = 3) -> None:
        """Erzeugt den Client; `retries` ist die Zahl der Versuche je Anfrage."""
        self.cfg = cfg
        self.retries = retries
        self._http = httpx.Client(base_url=cfg.base_url, timeout=timeout)

    def close(self) -> None:
        """Schließt die HTTP-Verbindung."""
        self._http.close()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Sendet eine POST-Anfrage; Verbindungsfehler und 5xx werden mit Backoff wiederholt."""
        last = ""
        for attempt in range(self.retries):
            try:
                resp = self._http.post(path, json=payload)
            except httpx.TransportError as exc:
                last = f"Ollama nicht erreichbar unter {self.cfg.base_url} ({exc.__class__.__name__})"
            else:
                if resp.status_code < 400:
                    return resp.json()
                last = _error_text(resp, payload.get("model", "?"))
                if resp.status_code < 500:
                    raise OllamaError(last)
            log.warning("Ollama-Versuch %d/%d fehlgeschlagen: %s", attempt + 1, self.retries, last)
            if attempt + 1 < self.retries:
                time.sleep(2 ** attempt)
        raise OllamaError(last)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Berechnet Embeddings für mehrere Texte in einer Anfrage."""
        data = self._post("/api/embed", {"model": self.cfg.embed_model, "input": texts})
        vectors = data.get("embeddings", [])
        if len(vectors) != len(texts):
            raise OllamaError("Ollama lieferte eine unerwartete Zahl an Embeddings")
        return vectors

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Erzeugt eine Antwort ohne Streaming; `num_ctx` wird explizit gesetzt."""
        data = self._post("/api/chat", {
            "model": self.cfg.chat_model, "messages": messages, "stream": False,
            "think": False, "options": {"num_ctx": self.cfg.num_ctx, "temperature": 0.2},
        })
        return data.get("message", {}).get("content", "")

    def ping(self) -> tuple[bool, list[str]]:
        """Prüft die Erreichbarkeit (ohne Wiederholung) und liefert die Modellnamen."""
        try:
            resp = self._http.get("/api/tags", timeout=3.0)
            resp.raise_for_status()
            return True, [m["name"] for m in resp.json().get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError):
            return False, []


def _error_text(resp: httpx.Response, model: str) -> str:
    """Formt eine Fehlerantwort von Ollama in eine verständliche Meldung um."""
    try:
        detail = resp.json().get("error", "")
    except ValueError:
        detail = resp.text[:200]
    if resp.status_code == 404:
        return f"Modell '{model}' nicht gefunden (ollama pull {model}). {detail}".strip()
    return f"Ollama-Fehler {resp.status_code}: {detail}"
