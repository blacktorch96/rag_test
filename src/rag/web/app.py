"""Application Factory der Weboberfläche."""
from __future__ import annotations

from flask import Flask

from ..config import Config
from ..ollama_client import OllamaClient
from .routes import bp


def create_app(cfg: Config) -> Flask:
    """Erzeugt die Flask-App; Konfiguration und Ollama-Client hängen an `app.extensions`."""
    app = Flask(__name__)
    app.extensions["rag_cfg"] = cfg
    app.extensions["rag_client"] = OllamaClient(cfg.ollama)
    app.register_blueprint(bp)
    return app
