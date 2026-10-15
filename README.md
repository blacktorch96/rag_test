# RAG-Testpipeline (lokal, Ollama)

Lokale RAG-Pipeline für Markdown-Dokumente: indexiert einen Korpus in eine einzelne SQLite-Datei (`sqlite-vec` + FTS5), sucht hybrid (Vektor + BM25, Reciprocal Rank Fusion) und beantwortet Fragen auf Deutsch mit Quellenangabe. Bedienung per CLI oder Flask-Weboberfläche. Alle Modelle laufen lokal über Ollama.

Die vollständige Spezifikation steht in [rag-test-sdd.md](rag-test-sdd.md).

## Voraussetzungen

- Python 3.12 oder neuer und [`uv`](https://docs.astral.sh/uv/)
- Laufendes [Ollama](https://ollama.com) unter `http://localhost:11434`
- Ein Embedding- und ein Chat-Modell:

```bash
ollama pull bge-m3
ollama pull qwen3.5:9b      # oder ein anderes Chat-Modell
```

## Einrichtung

```bash
uv sync
```

Dann `config.toml` anpassen, mindestens:

| Schlüssel | Bedeutung |
|---|---|
| `corpus.root` | Verzeichnis mit den `*.md`-Dateien |
| `ollama.embed_model` | Embedding-Modell (Standard `bge-m3`) |
| `ollama.chat_model` | Chat-Modell (Standard `qwen3.5:9b`) |
| `store.path` | SQLite-Datei, relativ zur `config.toml` (Standard `rag.db`) |
| `web.host` / `web.port` | Standard `127.0.0.1:5080` |

Eine andere Konfigurationsdatei geben Sie mit `--config pfad/zur/config.toml` oder der Umgebungsvariable `RAG_CONFIG` an.

## Benutzung

```bash
uv run rag index                    # neue/geänderte Dateien indexieren
uv run rag index --limit 100        # nur die ersten 100 Dateien (zum Testen)
uv run rag index --rebuild          # Index löschen und komplett neu aufbauen

uv run rag ask "Wie viele Urlaubstage gibt es?"
uv run rag ask "..." --show-chunks  # zusätzlich die Chunk-Texte ausgeben
uv run rag ask "..." --mode bm25    # hybrid (Standard), vector oder bm25

uv run rag eval                     # Recall@1/3/6 und MRR je Suchmodus
uv run rag serve                    # Weboberfläche auf Port 5080 (waitress)
uv run rag serve --dev              # Flask-Dev-Server mit Debug
```

Weitere Optionen: `rag -v ...` für ausführliches Logging, `rag serve --host H --port P`.

Der Index ist inkrementell: Pro Datei wird ein SHA-256-Hash gespeichert, unveränderte Dateien werden übersprungen, geänderte neu indexiert und gelöschte entfernt. Wechselt man das Embedding-Modell, bricht das Programm mit einer Meldung ab. Dann hilft `rag index --rebuild`.

## Weboberfläche

| Route | Inhalt |
|---|---|
| `GET /` | Fragefeld, Suchmodus, Option „Chunks anzeigen" |
| `POST /ask` | Antwort mit nummerierten Quellen und aufklappbarem Chunk-Text |
| `GET /status` | Dokumente, Chunks, letzte Indexierung, Modelle, Ollama-Erreichbarkeit |
| `GET /health` | JSON `{"status": "ok"}` (HTTP 200) bzw. Fehlerstatus (HTTP 503) |

Es gibt **keine Authentifizierung**. Den Host nur bewusst auf `0.0.0.0` setzen und dann per Reverse Proxy oder Firewall absichern. Die Oberfläche ändert den Index nie, Indexieren bleibt Aufgabe der CLI.

## Evaluation

`eval/questions.jsonl` enthält pro Zeile eine Testfrage:

```json
{"question": "Wie viele Urlaubstage gibt es?", "expected_path": "personal/urlaub.md"}
```

`expected_path` ist der Pfad relativ zu `corpus.root` mit `/`. Die vorhandene Zeile ist nur ein Platzhalter. Für eine aussagekräftige Auswertung 15 bis 30 echte Fragen eintragen. `rag eval` vergleicht die Modi hybrid, nur Vektor und nur BM25 und zeigt je Frage den Rang des ersten Treffers (mehrere Chunks derselben Datei zählen einmal).

## Projektstruktur

```
config.toml           Konfiguration
src/rag/
  config.py           Konfiguration laden und validieren
  loader.py           Dateien finden, lesen, Frontmatter, Hash
  chunker.py          Überschriftenbasiertes Chunking
  ollama_client.py    Embeddings und Chat über HTTP (mit Retry)
  store.py            SQLite-Schema, Schreiben, Suche
  indexer.py          Inkrementelle Indexierung
  retriever.py        Hybride Suche mit RRF
  generator.py        Prompt und Antwortgenerierung
  evaluate.py         Recall@k und MRR
  cli.py              Befehle index, ask, eval, serve
  web/                Flask-App (Factory, Routen, Templates, CSS)
eval/questions.jsonl  Testfragen
tests/                pytest
```

## Entwicklung

```bash
uv run pytest
```

Konventionen: Python 3.12+, Typannotationen, Docstrings für alle Funktionen/Klassen, höchstens 150 Zeilen je `.py`-Datei.

## Betrieb unter Linux

`corpus.root` in der `config.toml` anpassen, dann `uv sync` und `uv run rag serve` (nutzt `waitress`, Threads über `web.threads`). Pfade werden intern mit `/` gespeichert, alle Dateien sind UTF-8.
