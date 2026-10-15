# SDD: RAG-Testpipeline (lokal, Ollama)

## 1. Ziel

Eine lokale RAG-Pipeline, die Markdown-Dokumente indexiert und Fragen mit Quellenangabe beantwortet. Sie dient als Test, um Trefferqualität und Machbarkeit zu bewerten.

**Nicht-Ziele:** Multi-User, Authentifizierung, Reranking (zunächst), Agenten-Logik, Cloud-Dienste, Indexierung über die Weboberfläche (bleibt Aufgabe der CLI).

## 2. Rahmenbedingungen

| Punkt | Vorgabe |
|---|---|
| Korpus | unter 1000 Markdown-Dateien, ca. 15 MB, überwiegend Deutsch |
| LLM und Embeddings | ausschließlich lokales Ollama (`http://localhost:11434`) |
| Sprache | Python, Paketverwaltung mit `uv` |
| Entwicklung | Windows, Betrieb später Linux (pfadneutral mit `pathlib`, UTF-8 überall) |
| Persistenz | eine einzelne SQLite-Datei (`sqlite-vec` plus FTS5) |
| Oberfläche | Flask-Webseite auf Port **5080** (Standard-Host `127.0.0.1`, konfigurierbar) |
| Serverbetrieb | Entwicklung mit Flask-Dev-Server, Betrieb unter Linux mit `waitress` |

## 3. Architektur

```
Indexierung:  Markdown -> Chunker -> Embedder (Ollama) -> Store (SQLite: Vektor + FTS5)
Abfrage:      Frage -> Embedder -> Retriever (Vektor + BM25, RRF) -> Generator (Ollama) -> Antwort + Quellen
Oberfläche:   Browser -> Flask (Port 5080) -> Retriever/Generator -> HTML-Antwortseite
```

Die Flask-Schicht enthält keine Geschäftslogik, sie ruft nur `retriever` und `generator` auf. Dadurch bleiben CLI und Weboberfläche identisch im Verhalten.

## 4. Projektstruktur

Jede Python-Datei hat maximal 150 Zeilen, jede Funktion und Methode einen Docstring.

```
rag-test/
  pyproject.toml
  config.toml
  src/rag/
    config.py       # Konfiguration laden und validieren
    loader.py       # Markdown-Dateien finden, lesen, Hash berechnen
    chunker.py      # Überschriftenbasiertes Chunking
    ollama_client.py# HTTP-Wrapper für Embeddings und Chat
    store.py        # SQLite-Schema, Insert, Delete, Suche
    indexer.py      # Orchestriert loader, chunker, embedder, store
    retriever.py    # Hybride Suche mit Reciprocal Rank Fusion
    generator.py    # Prompt-Bau und Antwortgenerierung
    evaluate.py     # Auswertung anhand Testfragen
    cli.py          # Befehle: index, ask, eval, serve
    web/
      app.py        # create_app() als Application Factory
      routes.py     # Routen: /, /ask, /status, /health
      templates/    # base.html, index.html, result.html, status.html
      static/       # style.css (kein CDN, alles lokal)
  eval/questions.jsonl
  tests/
```

## 5. Datenmodell

**documents:** `id`, `path` (relativ zum Korpus-Root), `sha256`, `indexed_at`
**chunks:** `id`, `document_id`, `ord`, `heading_path`, `text`, `token_count`
**chunks_fts:** FTS5-Tabelle über `heading_path` und `text`
**chunks_vec:** `sqlite-vec`-Tabelle mit Embedding je Chunk
**meta:** Schlüssel-Wert-Tabelle, enthält mindestens `embedding_model` und `embedding_dim`

Beim Start wird das konfigurierte Embedding-Modell gegen `meta` geprüft. Bei Abweichung bricht das Programm mit klarer Meldung ab und verlangt eine Neuindexierung.

## 6. Konfiguration (`config.toml`)

```toml
[corpus]
root = "C:/pfad/zum/korpus"       # unter Linux anpassbar

[ollama]
base_url = "http://localhost:11434"
embed_model = "bge-m3"
chat_model = "qwen3:8b"           # austauschbar
num_ctx = 8192

[chunking]
target_tokens = 500
max_tokens = 800
overlap_tokens = 60

[retrieval]
top_k_vector = 20
top_k_bm25 = 20
final_k = 6
rrf_k = 60

[web]
host = "127.0.0.1"                # "0.0.0.0" nur bewusst setzen (keine Authentifizierung)
port = 5080
threads = 4                       # waitress im Betrieb
```

## 7. Funktionale Anforderungen

**F1 Laden:** Rekursiv alle `*.md` unter `corpus.root` lesen. Optionales YAML-Frontmatter wird abgetrennt und als Metadaten behandelt, nicht gechunkt.

**F2 Chunking:** Teilen an Markdown-Überschriften. Zu große Abschnitte werden an Absatzgrenzen weiter geteilt, zu kleine mit dem Nachbarn verschmolzen. Jeder Chunk trägt seinen Überschriftenpfad (z. B. `Titel > Abschnitt > Unterabschnitt`), der bei Embedding und Anzeige dem Text vorangestellt wird. Code-Blöcke und Tabellen werden nicht mitten im Block zerschnitten.

**F3 Inkrementelles Indexieren:** Pro Datei wird ein SHA-256-Hash gespeichert. Unveränderte Dateien werden übersprungen, geänderte neu indexiert (alte Chunks vorher löschen), entfernte Dateien aus dem Index genommen.

**F4 Embedding:** Chunks werden in Batches an Ollama geschickt. Fehler führen zu begrenzten Wiederholungsversuchen mit Backoff, danach zu einer klaren Fehlermeldung. Fortschrittsanzeige in der Konsole.

**F5 Hybride Suche:** Vektorsuche und FTS5-BM25 laufen getrennt, die Ergebnisse werden per Reciprocal Rank Fusion zusammengeführt. Rückgabe sind die besten `final_k` Chunks mit Quelle, Überschriftenpfad und Score.

**F6 Generierung:** Der Prompt weist das Modell an, nur auf Basis des gelieferten Kontexts zu antworten, Quellen als `[1]`, `[2]` usw. zu nennen und bei fehlender Information ausdrücklich „Dazu finde ich nichts in den Dokumenten" zu antworten. Antwort auf Deutsch. `num_ctx` wird bei jedem Aufruf explizit gesetzt. Am Ende werden die verwendeten Quellen mit Dateipfad aufgelistet.

**F7 CLI:**
- `rag index [--limit N] [--rebuild]`
- `rag ask "Frage" [--show-chunks]`
- `rag eval`
- `rag serve [--host H] [--port P] [--dev]` startet die Weboberfläche (Standard Port 5080; ohne `--dev` über `waitress`)

**F8 Weboberfläche (Flask, Port 5080):**
- `GET /`: Startseite mit Fragefeld, Auswahl des Suchmodus (hybrid, nur Vektor, nur BM25) und Checkbox „Chunks anzeigen".
- `POST /ask`: führt Retrieval und Generierung aus und rendert die Antwortseite mit Antworttext, nummerierten Quellen (Dateipfad, Überschriftenpfad) und je Quelle einem aufklappbaren Chunk-Text. Während der Generierung zeigt die Seite einen Ladehinweis. Streaming ist in Version 1 nicht erforderlich.
- `GET /status`: Anzahl Dokumente und Chunks, Zeitpunkt der letzten Indexierung, konfigurierte Modelle, Erreichbarkeit von Ollama.
- `GET /health`: JSON `{"status": "ok"}` bzw. Fehlerstatus (für Monitoring unter Linux).
- Fehlerfälle (Ollama nicht erreichbar, leerer Index, Modell fehlt) erscheinen als verständliche Fehlermeldung auf der Seite, nicht als Stacktrace.
- Dokumentinhalte und Modellantworten werden ausschließlich escaped ausgegeben (Jinja-Autoescaping), niemals als rohes HTML.
- Pro Request eigene, schreibgeschützte SQLite-Verbindung. Der Index wird nicht über die Weboberfläche verändert.
- Die Oberfläche ist schlicht und responsiv, ohne externe Skripte, Schriften oder CDNs.

## 8. Evaluation

`eval/questions.jsonl` enthält 15 bis 30 Zeilen der Form `{"question": "...", "expected_path": "relativ/pfad.md"}`.

`rag eval` gibt aus:
- **Recall@k** für k = 1, 3, 6 (steht die erwartete Datei unter den Treffern?)
- **MRR**
- pro Frage: Rang des ersten korrekten Treffers
- Vergleich der Modi „nur Vektor", „nur BM25" und „hybrid"

Die Antwortqualität wird beim Test manuell stichprobenartig bewertet.

## 9. Nicht-funktionale Anforderungen

- Vollständiger Index für den Testkorpus in vertretbarer Zeit, Abfrage ohne Generierung unter 1 Sekunde
- Alle Dateien UTF-8, Pfade intern relativ und mit `/` gespeichert
- Ausgehend keine Netzwerkzugriffe außer auf den konfigurierten Ollama-Host, eingehend nur der Webport (Standard 5080, lokal gebunden)
- Logging über `logging`, keine `print`-Ausgaben außerhalb der CLI-Ausgabe
- Reproduzierbar: `uv sync` und `uv run rag ...` genügen

## 10. Code-Konventionen

- Python 3.12 oder neuer, Typannotationen überall
- Docstrings für alle Funktionen, Methoden und Klassen
- Maximal 150 Zeilen je `.py`-Datei, bei Überschreitung aufteilen
- Abhängigkeiten minimal halten: `httpx`, `sqlite-vec`, `typer`, `flask`, `waitress`, `pytest`
- Tests für Chunker (Struktur- und Randfälle), RRF und Hash-Logik

## 11. Abnahmekriterien

1. `uv run rag index` indexiert den Testkorpus vollständig, ein zweiter Lauf überspringt alle unveränderten Dateien.
2. `uv run rag ask` liefert eine deutsche Antwort mit nachvollziehbaren Quellen.
3. `uv run rag eval` läuft durch und zeigt Recall@k und MRR je Modus.
4. Modellwechsel beim Embedding wird erkannt und verhindert gemischte Vektoren.
5. `uv run rag serve` startet die Seite auf Port 5080, eine Frage im Browser liefert Antwort und Quellen.
6. `/status` und `/health` funktionieren auch bei nicht erreichbarem Ollama und melden den Zustand korrekt.
7. Alle Dateien erfüllen die Zeilen- und Docstring-Vorgabe.

## 12. Umsetzungsreihenfolge

1. Projektgerüst, Konfiguration, Ollama-Client (Verbindungstest)
2. Loader und Chunker mit Tests
3. Store und Indexer, Test mit `--limit 100`
4. Retriever (erst Vektor, dann BM25, dann RRF)
5. Generator und `ask`
6. Flask-Oberfläche (App Factory, `/`, `/ask`, `/status`, `/health`, `serve`)
7. Evaluation und Auswertung der drei Modi

## 13. Offene Entscheidungen (vor Start klären)

- Welches Chat-Modell läuft auf der vorhandenen Hardware stabil?
- Haben die Markdown-Dateien Frontmatter, das als Metadatum (z. B. Titel, Datum) nutzbar ist?
- Soll Reranking nach dem ersten Test ergänzt werden?
- Soll die Seite nur lokal erreichbar sein oder im Netzwerk (dann bewusst `0.0.0.0` setzen und über Reverse Proxy bzw. Firewall absichern, da es keine Anmeldung gibt)?
