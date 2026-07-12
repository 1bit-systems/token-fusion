"""Content-type detection — the Cortex stage logic extracted for reuse."""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional

CONTENT_PATTERNS: list[tuple[str, str, str]] = [
    # (content_type, language, regex_pattern)
    ("json", "json", r'^\s*[\{\[].*[\}\]]\s*$'),
    ("json", "json", r'^\s*\{.*"\s*:'),
    ("log", "log", r'(?m)^\d{4}[-\d]+\s+\d{2}:\d{2}:\d{2}'),
    ("log", "log", r'(?m)^(INFO|DEBUG|WARN|ERROR|TRACE|FATAL)\s+'),
    ("diff", "diff", r'(?m)^(diff --git|\+\+\+|---|@@ -\d+,\d+ \+\d+,\d+)'),
    ("search", "search", r'(?m)^\S+:\d+:\s'),
    ("code", "python", r'(?m)^(import |from |def |class |@)\s'),
    ("code", "go", r'(?m)^(package |import |func |type |struct |interface )'),
    ("code", "rust", r'(?m)^(use |fn |pub |struct |enum |impl |trait )'),
    ("code", "typescript", r'(?m)^(import |export |interface |type |function |class |const )'),
    ("code", "javascript", r'(?m)^(import |export |function |class |const |let |var )'),
    ("code", "java", r'(?m)^(import |package |public |private |class |interface )'),
    ("code", "cpp", r"(?m)^(?:#include|using |namespace |template |class |struct |void |int |auto )"),
    ("code", "ruby", r'(?m)^(require |def |class |module |attr_)'),
    ("code", "php", r'(?m)^(<\?php|namespace |use |function |class )'),
    ("code", "shell", r'(?m)^(#!/|export |alias |function |if |for |while )'),
    ("code", "sql", r"(?m)^\s*(SELECT|FROM|WHERE|INSERT|CREATE|ALTER|DROP|UPDATE|DELETE)\b"),
    ("code", "yaml", r"(?m)^\s*\w[\w-]*:"),
]

LANGUAGE_EXTENSIONS: dict[str, str] = {
    ".py": "python", ".go": "go", ".rs": "rust", ".ts": "typescript",
    ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c",
    ".hpp": "cpp", ".rb": "ruby", ".php": "php", ".sh": "shell",
    ".bash": "shell", ".zsh": "shell", ".sql": "sql", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".toml": "toml", ".md": "markdown",
    ".rst": "markdown", ".html": "html", ".css": "css",
}

class ContentType(str, Enum):
    CODE = "code"
    JSON = "json"
    LOG = "log"
    DIFF = "diff"
    SEARCH = "search"
    TEXT = "text"
    UNKNOWN = "unknown"

@dataclass
class ContentProfile:
    content_type: ContentType
    language: Optional[str]
    confidence: float

def detect(text: str, filename_hint: Optional[str] = None) -> ContentProfile:
    """Detect content type and language from text content + optional filename."""
    text_stripped = text.strip()
    if not text_stripped:
        return ContentProfile(ContentType.TEXT, None, 1.0)

    # Filename-based detection
    if filename_hint and "." in filename_hint:
        ext = f".{filename_hint.rsplit('.', 1)[-1]}"
        if ext in LANGUAGE_EXTENSIONS:
            return ContentProfile(ContentType.CODE, LANGUAGE_EXTENSIONS[ext], 0.95)

    # Pattern-based detection
    best: Optional[tuple[str, str, float]] = None
    lines = text_stripped.split("\n")
    n_lines = len(lines)

    for ctype, lang, pattern in CONTENT_PATTERNS:
        try:
            matches = len(re.findall(pattern, text_stripped))
            if matches > 0:
                confidence = min(1.0, matches / max(3, n_lines * 0.1))
                if best is None or confidence > best[2]:
                    best = (ctype, lang, confidence)
        except re.error:
            continue

    if best:
        return ContentProfile(
            content_type=ContentType(best[0]),
            language=best[1],
            confidence=best[2],
        )

    return ContentProfile(ContentType.TEXT, None, 0.5)
