"""Kommandozeile: index, ask, eval, serve."""
from __future__ import annotations

import functools
import logging
import sys
from pathlib import Path
from typing import Callable

import typer

from . import evaluate, generator, indexer, retriever, store
from .config import Config, load_config
from .errors import RagError
from .ollama_client import OllamaClient

app = typer.Typer(add_completion=False, help="Lokale RAG-Testpipeline.")
_state: dict = {"config": None, "profile": None}


@app.callback()
def main(config: Path = typer.Option(None, "--config", help="Pfad zur config.toml"),
         profile: str = typer.Option(None, "--profile", "-p", help="Profil aus config.toml"),
         verbose: bool = typer.Option(False, "--verbose", "-v", help="Ausführliches Logging")) -> None:
    """Setzt Konfigurationspfad, Logging und UTF-8-Ausgabe."""
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    _state["config"] = config
    _state["profile"] = profile


def guarded(func: Callable) -> Callable:
    """Wandelt erwartete Fehler in eine kurze Meldung mit Exit-Code 1 um."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Führt den Befehl aus und fängt RagError ab."""
        try:
            return func(*args, **kwargs)
        except RagError as exc:
            typer.echo(f"Fehler: {exc}", err=True)
            raise typer.Exit(1) from exc
    return wrapper


def _cfg() -> Config:
    """Lädt die Konfiguration gemäß --config."""
    return load_config(_state["config"], _state["profile"])


@app.command()
@guarded
def index(limit: int = typer.Option(None, help="Höchstens N Dateien indexieren"),
          rebuild: bool = typer.Option(False, help="Index löschen und neu aufbauen")) -> None:
    """Indexiert neue und geänderte Markdown-Dateien."""
    cfg = _cfg()
    client = OllamaClient(cfg.ollama)

    def progress(i: int, total: int, rel: str) -> None:
        """Zeigt den Fortschritt je Datei."""
        typer.echo(f"[{i}/{total}] {rel}")

    res = indexer.run_index(cfg, client, limit, rebuild, progress)
    typer.echo(f"Fertig: {res.indexed} indexiert ({res.chunks} Chunks), "
               f"{res.skipped} unverändert übersprungen, {res.removed} entfernt.")


@app.command()
@guarded
def ask(question: str, show_chunks: bool = typer.Option(False, help="Chunk-Texte anzeigen"),
        mode: str = typer.Option("hybrid", help="hybrid, vector oder bm25")) -> None:
    """Beantwortet eine Frage anhand des Index."""
    if mode not in retriever.MODES:
        raise typer.BadParameter(f"mode muss einer von {', '.join(retriever.MODES)} sein")
    cfg = _cfg()
    client = OllamaClient(cfg.ollama)
    conn = store.connect(cfg.store.path, readonly=True)
    answer = generator.answer_question(conn, cfg, client, question, mode)
    typer.echo(answer.text)
    if answer.hits:
        typer.echo("\nQuellen:\n" + generator.format_sources(answer.hits))
    if show_chunks:
        for i, hit in enumerate(answer.hits, 1):
            typer.echo(f"\n--- [{i}] {hit.path} | {hit.heading_path} | Score {hit.score:.4f}\n{hit.text}")


@app.command(name="eval")
@guarded
def eval_command(questions: Path = typer.Option(Path("eval/questions.jsonl"), help="JSONL-Datei")) -> None:
    """Wertet Recall@k und MRR für alle Suchmodi aus."""
    cfg = _cfg()
    client = OllamaClient(cfg.ollama)
    conn = store.connect(cfg.store.path, readonly=True)
    rows = evaluate.load_questions(questions)
    reports = evaluate.run_eval(conn, cfg, client, rows)
    typer.echo(f"{'Modus':<8}" + "".join(f"Recall@{k:<3}" for k in evaluate.KS) + "MRR")
    for rep in reports:
        typer.echo(f"{rep.mode:<8}" + "".join(f"{rep.recall[k]:<9.2f}" for k in evaluate.KS) + f"{rep.mrr:.3f}")
    typer.echo("\nRang des ersten Treffers (-: nicht gefunden):")
    for i, row in enumerate(rows):
        ranks = "  ".join(f"{r.mode}={r.ranks[i] or '-'}" for r in reports)
        typer.echo(f"{i + 1:>3}. {ranks}  {row['question'][:60]}")


@app.command()
@guarded
def serve(host: str = typer.Option(None, help="Host (Standard aus config.toml)"),
          port: int = typer.Option(None, help="Port (Standard aus config.toml)"),
          dev: bool = typer.Option(False, "--dev", help="Flask-Dev-Server statt waitress")) -> None:
    """Startet die Weboberfläche (Standard Port 5080)."""
    from .web.app import create_app
    cfg = _cfg()
    host, port = host or cfg.web.host, port or cfg.web.port
    flask_app = create_app(cfg)
    typer.echo(f"Weboberfläche auf http://{host}:{port}")
    if dev:
        flask_app.run(host=host, port=port, debug=True)
    else:
        from waitress import serve as waitress_serve
        waitress_serve(flask_app, host=host, port=port, threads=cfg.web.threads)
