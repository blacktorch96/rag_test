"""Routen: /, /ask, /status, /health (keine Geschäftslogik)."""
from __future__ import annotations

import sqlite3

from flask import Blueprint, abort, current_app, g, jsonify, render_template, request

from .. import generator, retriever, store
from ..config import Config
from ..errors import ConfigError, RagError
from ..ollama_client import OllamaClient

bp = Blueprint("rag", __name__)


def _cfg() -> Config:
    """Konfiguration der laufenden App, bezogen auf das per `profile` gewählte Profil."""
    if "cfg" not in g:
        try:
            g.cfg = current_app.extensions["rag_cfg"].with_profile(request.values.get("profile"))
        except ConfigError:
            abort(400, "Unbekanntes Profil")
    return g.cfg


@bp.app_context_processor
def _profiles() -> dict:
    """Stellt Profilliste und aktives Profil allen Templates bereit."""
    cfg = _cfg()
    return {"profiles": list(cfg.profiles), "profile": cfg.profile}


def _client() -> OllamaClient:
    """Ollama-Client der laufenden App."""
    return current_app.extensions["rag_client"]


def _db() -> sqlite3.Connection:
    """Eigene, schreibgeschützte Verbindung pro Request."""
    if "db" not in g:
        g.db = store.connect(_cfg().store.path, readonly=True)
    return g.db


@bp.teardown_app_request
def _close_db(_exc: BaseException | None) -> None:
    """Schließt die Request-Verbindung."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


@bp.get("/")
def index():
    """Startseite mit Fragefeld."""
    return render_template("index.html", modes=retriever.MODES, mode="hybrid", question="")


@bp.post("/ask")
def ask():
    """Führt Retrieval und Generierung aus und zeigt Antwort mit Quellen."""
    question = request.form.get("question", "").strip()
    mode = request.form.get("mode", "hybrid")
    show = request.form.get("show_chunks") == "on"
    ctx = {"modes": retriever.MODES, "mode": mode, "question": question, "show_chunks": show}
    if not question or mode not in retriever.MODES:
        return render_template("index.html", error="Bitte eine Frage und einen gültigen Modus angeben.",
                               **ctx), 400
    try:
        answer = generator.answer_question(_db(), _cfg(), _client(), question, mode)
    except RagError as exc:
        return render_template("index.html", error=str(exc), **ctx), 503
    return render_template("result.html", answer=answer, **ctx)


def _state() -> dict:
    """Sammelt Index- und Ollama-Zustand, ohne bei Fehlern abzubrechen."""
    cfg = _cfg()
    reachable, models = _client().ping()
    state = {"ollama": reachable, "index": False, "documents": 0, "chunks": 0,
             "last_indexed": None, "embed_model": cfg.ollama.embed_model,
             "chat_model": cfg.ollama.chat_model, "error": None,
             "embed_available": cfg.ollama.embed_model in models or f"{cfg.ollama.embed_model}:latest" in models,
             "chat_available": cfg.ollama.chat_model in models or f"{cfg.ollama.chat_model}:latest" in models}
    try:
        info = store.stats(_db())
        state.update(index=info["chunks"] > 0, documents=info["documents"],
                     chunks=info["chunks"], last_indexed=info["last_indexed"])
    except (RagError, sqlite3.Error) as exc:
        state["error"] = str(exc)
    return state


@bp.get("/status")
def status():
    """Statusseite mit Indexgröße, Modellen und Ollama-Erreichbarkeit."""
    return render_template("status.html", s=_state())


@bp.get("/health")
def health():
    """JSON-Gesundheitsstatus für Monitoring (503 bei Problemen)."""
    s = _state()
    ok = s["ollama"] and s["index"]
    body = {"status": "ok" if ok else "error", "ollama": s["ollama"], "index": s["index"]}
    return jsonify(body), 200 if ok else 503
