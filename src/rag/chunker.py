"""Überschriftenbasiertes Chunking mit Rücksicht auf Code-Blöcke und Tabellen."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .config import ChunkingCfg

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """Ein Textabschnitt mit Überschriftenpfad."""
    heading_path: str
    text: str
    token_count: int


def estimate_tokens(text: str) -> int:
    """Grobe Token-Schätzung für deutschen Text (ca. 3,5 Zeichen je Token)."""
    return max(1, math.ceil(len(text) / 3.5))


def parse_sections(body: str, title: str) -> list[tuple[str, list[str]]]:
    """Teilt den Text an Überschriften (außerhalb von Code-Blöcken) in Abschnitte."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    path, lines, in_fence = title, [], False
    for line in body.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        head = None if in_fence else HEADING.match(line)
        if head is None:
            lines.append(line)
            continue
        sections.append((path, lines))
        level = len(head.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, head.group(2)))
        path, lines = " > ".join(t for _, t in stack), []
    sections.append((path, lines))
    return sections


def to_blocks(lines: list[str]) -> list[str]:
    """Gruppiert Zeilen zu Absatz-Blöcken; Code-Blöcke bleiben zusammen."""
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _pack(parts: list[str], sep: str, limit: int) -> list[str]:
    """Fasst Teile bis zur Token-Grenze zusammen; zu lange Teile werden hart geteilt."""
    out: list[str] = []
    cur = ""
    for part in parts:
        while estimate_tokens(part) > limit:
            if cur:
                out.append(cur)
                cur = ""
            cut = int(limit * 3.5)
            out.append(part[:cut])
            part = part[cut:]
        if cur and estimate_tokens(cur + sep + part) > limit:
            out.append(cur)
            cur = part
        else:
            cur = cur + sep + part if cur else part
    if cur:
        out.append(cur)
    return out


def split_block(block: str, limit: int) -> list[str]:
    """Teilt zu große Absätze; Code-Blöcke und Tabellen bleiben ungeteilt."""
    atomic = FENCE.match(block) or block.lstrip().startswith("|")
    if estimate_tokens(block) <= limit or atomic:
        return [block]
    lines = _pack(block.split("\n"), "\n", limit)
    return [piece for line in lines for piece in _pack(SENTENCE.split(line), " ", limit)]


def pack_section(blocks: list[str], cfg: ChunkingCfg) -> list[str]:
    """Packt Blöcke zu Chunks um `target_tokens` (höchstens `max_tokens`) mit Überlappung."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for block in (p for b in blocks for p in split_block(b, cfg.max_tokens)):
        tokens = estimate_tokens(block)
        if cur and (size + tokens > cfg.max_tokens or size >= cfg.target_tokens):
            chunks.append("\n\n".join(cur))
            last = cur[-1]
            keep = estimate_tokens(last) <= cfg.overlap_tokens
            cur, size = ([last], estimate_tokens(last)) if keep else ([], 0)
        cur.append(block)
        size += tokens
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_markdown(body: str, cfg: ChunkingCfg, title: str = "Dokument") -> list[Chunk]:
    """Zerlegt Markdown in Chunks, kleine Abschnitte werden mit Nachbarn verschmolzen."""
    raw: list[tuple[str, str]] = []
    for path, lines in parse_sections(body, title):
        raw.extend((path, text) for text in pack_section(to_blocks(lines), cfg))
    small = cfg.target_tokens // 4
    merged: list[tuple[str, str]] = []
    for path, text in raw:
        if merged:
            prev_path, prev = merged[-1]
            fits = estimate_tokens(prev) + estimate_tokens(text) <= cfg.max_tokens
            if fits and (estimate_tokens(prev) < small or estimate_tokens(text) < small):
                label = path.split(" > ")[-1] if path != prev_path else ""
                joined = f"{prev}\n\n{label}\n{text}" if label else f"{prev}\n\n{text}"
                merged[-1] = (prev_path, joined)
                continue
        merged.append((path, text))
    return [Chunk(p, t, estimate_tokens(t)) for p, t in merged]
