"""Markdown-Dateien finden, lesen und hashen."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError


@dataclass(frozen=True)
class Document:
    """Ein gelesenes Dokument ohne Frontmatter."""
    path: str
    sha256: str
    meta: dict[str, str]
    body: str


def scan(root: Path) -> dict[str, Path]:
    """Findet rekursiv alle *.md; Schlüssel ist der relative Pfad mit '/'."""
    if not root.is_dir():
        raise ConfigError(f"Korpus-Verzeichnis nicht gefunden: {root}")
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*.md")) if p.is_file()}


def sha256_bytes(data: bytes) -> str:
    """Berechnet den SHA-256-Hash als Hex-String."""
    return hashlib.sha256(data).hexdigest()


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Trennt optionales YAML-Frontmatter (einfache `key: value`-Zeilen) ab."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1 or text[end + 4:end + 5] not in ("", "\n"):
        return {}, text
    meta: dict[str, str] = {}
    for line in text[4:end].split("\n"):
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "#", "-")):
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[end + 5:].lstrip("\n")


def read_document(rel: str, path: Path) -> Document:
    """Liest eine Datei (UTF-8, auch mit BOM) und liefert Hash, Metadaten und Text."""
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    meta, body = split_frontmatter(text)
    return Document(rel, sha256_bytes(raw), meta, body)
