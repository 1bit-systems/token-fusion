"""RewindStore — hash-addressed reversible retrieval for compressed content."""

import hashlib
import json
from typing import Optional

REWIND_MARKER_PREFIX = "[rewind:"
REWIND_MARKER_SUFFIX = "]"


class RewindStore:
    """Hash-addressed LRU store for reversible compression.
    
    Stores original content by its SHA-256 hash so compressed output
    can contain markers like [rewind:deadbeef...] that an LLM can
    use to retrieve the original via a tool call.
    """

    def __init__(self, max_entries: int = 10_000):
        self._store: dict[str, str] = {}
        self._max_entries = max_entries
        self._access_order: list[str] = []

    def store(self, original: str, marker_id: Optional[str] = None) -> str:
        """Store original content and return its marker ID (first 16 hex chars of SHA-256)."""
        if marker_id is None:
            marker_id = hashlib.sha256(original.encode()).hexdigest()[:16]

        if marker_id not in self._store:
            # Evict LRU if at capacity
            if len(self._store) >= self._max_entries:
                oldest = self._access_order.pop(0)
                self._store.pop(oldest, None)

            self._store[marker_id] = original
            self._access_order.append(marker_id)
        else:
            # Bump in access order
            if marker_id in self._access_order:
                self._access_order.remove(marker_id)
            self._access_order.append(marker_id)

        return marker_id

    def retrieve(self, marker_id: str) -> Optional[str]:
        """Retrieve original content by marker ID."""
        if marker_id in self._store:
            # Bump LRU
            self._access_order.remove(marker_id)
            self._access_order.append(marker_id)
            return self._store[marker_id]
        return None

    def make_marker(self, marker_id: str) -> str:
        """Create a rewind marker string for embedding in compressed output."""
        return f"{REWIND_MARKER_PREFIX}{marker_id}{REWIND_MARKER_SUFFIX}"

    def parse_marker(self, text: str) -> Optional[str]:
        """Extract marker ID from a rewind marker string."""
        if text.startswith(REWIND_MARKER_PREFIX) and text.endswith(REWIND_MARKER_SUFFIX):
            return text[len(REWIND_MARKER_PREFIX):-len(REWIND_MARKER_SUFFIX)]
        return None

    def contains_marker(self, text: str) -> bool:
        """Check if text contains any rewind marker."""
        return REWIND_MARKER_PREFIX in text

    def clear(self) -> None:
        self._store.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {
            "entries": len(self._store),
            "max_entries": self._max_entries,
        }
