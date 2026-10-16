"""Konfiguration laden und validieren."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from .errors import ConfigError


@dataclass(frozen=True)
class CorpusCfg:
    """Pfad zum Markdown-Korpus."""
    root: Path = Path(".")


@dataclass(frozen=True)
class StoreCfg:
    """Ort der SQLite-Datei."""
    path: Path = Path("rag.db")


@dataclass(frozen=True)
class OllamaCfg:
    """Ollama-Verbindung und Modelle."""
    base_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    chat_model: str = "qwen3.5:9b"
    num_ctx: int = 8192


@dataclass(frozen=True)
class ChunkingCfg:
    """Chunk-Größen in (geschätzten) Tokens."""
    target_tokens: int = 500
    max_tokens: int = 800
    overlap_tokens: int = 60


@dataclass(frozen=True)
class RetrievalCfg:
    """Parameter der hybriden Suche."""
    top_k_vector: int = 20
    top_k_bm25: int = 20
    final_k: int = 6
    rrf_k: int = 60


@dataclass(frozen=True)
class WebCfg:
    """Einstellungen der Weboberfläche."""
    host: str = "127.0.0.1"
    port: int = 5080
    threads: int = 4


@dataclass(frozen=True)
class ProfileCfg:
    """Ein Profil: eigener Dokumentenordner und eigene SQLite-Datei."""
    root: Path = Path(".")
    db: Path = Path("rag.db")


@dataclass(frozen=True)
class Config:
    """Gesamtkonfiguration; `profile` nennt das aktive Profil ("" = ohne Profile)."""
    corpus: CorpusCfg = field(default_factory=CorpusCfg)
    store: StoreCfg = field(default_factory=StoreCfg)
    ollama: OllamaCfg = field(default_factory=OllamaCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    web: WebCfg = field(default_factory=WebCfg)
    profiles: dict[str, ProfileCfg] = field(default_factory=dict)
    profile: str = ""

    def with_profile(self, name: str | None) -> "Config":
        """Liefert die Konfiguration mit Korpus und Datenbank des Profils `name`."""
        if not self.profiles:
            if name:
                raise ConfigError(f"Unbekanntes Profil: {name}")
            return self
        name = name or self.profile or next(iter(self.profiles))
        if name not in self.profiles:
            raise ConfigError(f"Unbekanntes Profil: {name} (verfügbar: {', '.join(self.profiles)})")
        prof = self.profiles[name]
        return replace(self, corpus=CorpusCfg(root=prof.root), store=StoreCfg(path=prof.db),
                       profile=name)


def _section(cls: type, data: dict[str, Any], name: str, base: Path) -> Any:
    """Baut eine Teil-Konfiguration und prüft Schlüssel und Typen."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unbekannte Schlüssel in [{name}]: {', '.join(sorted(unknown))}")
    values: dict[str, Any] = {}
    for key, value in data.items():
        default = getattr(cls(), key)
        if isinstance(default, Path):
            path = Path(str(value))
            value = path if path.is_absolute() else base / path
        elif not isinstance(value, type(default)) or isinstance(value, bool):
            raise ConfigError(f"[{name}] {key} hat den falschen Typ")
        values[key] = value
    return cls(**values)


def load_config(path: Path | None = None, profile: str | None = None) -> Config:
    """Lädt die TOML-Datei (Standard: $RAG_CONFIG oder ./config.toml).

    Mit `[profiles.<name>]`-Abschnitten (Schlüssel `root`, `db`) wird `profile` aktiviert;
    ohne Angabe gilt `default_profile` bzw. das erste Profil.
    """
    path = Path(path or os.environ.get("RAG_CONFIG", "config.toml")).resolve()
    if not path.is_file():
        raise ConfigError(f"Konfigurationsdatei nicht gefunden: {path}")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    default_profile = raw.pop("default_profile", "")
    raw_profiles = raw.pop("profiles", {})
    if not isinstance(default_profile, str) or not isinstance(raw_profiles, dict):
        raise ConfigError("default_profile muss Text und [profiles] eine Tabelle sein")
    parts = {"corpus": CorpusCfg, "store": StoreCfg, "ollama": OllamaCfg,
             "chunking": ChunkingCfg, "retrieval": RetrievalCfg, "web": WebCfg}
    unknown = set(raw) - set(parts)
    if unknown:
        raise ConfigError(f"Unbekannte Abschnitte: {', '.join(sorted(unknown))}")
    profiles = {}
    for name, data in raw_profiles.items():
        if not isinstance(data, dict) or set(data) != {"root", "db"}:
            raise ConfigError(f"[profiles.{name}] benötigt genau die Schlüssel root und db")
        profiles[name] = _section(ProfileCfg, data, f"profiles.{name}", path.parent)
    cfg = Config(**{n: _section(c, raw.get(n, {}), n, path.parent) for n, c in parts.items()},
                 profiles=profiles, profile=default_profile)
    cfg = cfg.with_profile(profile)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Prüft Wertebereiche und Abhängigkeiten zwischen Einstellungen."""
    ch = cfg.chunking
    if not 0 < ch.target_tokens <= ch.max_tokens:
        raise ConfigError("chunking: 0 < target_tokens <= max_tokens erforderlich")
    if not 0 <= ch.overlap_tokens < ch.target_tokens:
        raise ConfigError("chunking: overlap_tokens muss kleiner als target_tokens sein")
    if not 0 < cfg.web.port < 65536:
        raise ConfigError("web: port muss zwischen 1 und 65535 liegen")
    r = cfg.retrieval
    if min(r.top_k_vector, r.top_k_bm25, r.final_k, r.rrf_k) < 1:
        raise ConfigError("retrieval: alle Werte müssen >= 1 sein")
