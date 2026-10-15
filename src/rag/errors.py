"""Fehlerklassen mit verständlichen Meldungen für CLI und Weboberfläche."""


class RagError(Exception):
    """Basisklasse aller erwarteten Fehler."""


class ConfigError(RagError):
    """Konfiguration fehlt oder ist ungültig."""


class OllamaError(RagError):
    """Ollama nicht erreichbar oder Modell/Anfrage fehlerhaft."""


class IndexNotReady(RagError):
    """Der Index existiert nicht oder ist leer."""


class ModelMismatch(RagError):
    """Das Embedding-Modell passt nicht zum vorhandenen Index."""
