"""SimHash fingerprinting for semantic deduplication."""

from typing import Sequence
import re

_FINGERPRINT_BITS = 64

def _tokenize(text: str) -> list[str]:
    """Simple whitespace-aware tokenization with punctuation splitting."""
    return [t for t in re.split(r"(\s+)", text) if t.strip()]

def fingerprint(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for a string."""
    tokens = _tokenize(text)
    v = [0] * _FINGERPRINT_BITS

    for token in tokens:
        # Use Python's built-in hash (seeded, consistent within process)
        h = hash(token)
        for i in range(_FINGERPRINT_BITS):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fp = 0
    for i in range(_FINGERPRINT_BITS):
        if v[i] > 0:
            fp |= (1 << i)

    return fp

def hamming_distance(fp1: int, fp2: int) -> int:
    """Compute Hamming distance between two SimHash fingerprints."""
    return (fp1 ^ fp2).bit_count()

def similarity(fp1: int, fp2: int) -> float:
    """Similarity score [0,1] from SimHash fingerprints."""
    bits = _FINGERPRINT_BITS
    dist = hamming_distance(fp1, fp2)
    return 1.0 - (dist / bits)

def deduplicate(texts: list[str], threshold: float = 0.85) -> list[tuple[str, int, list[int]]]:
    """Deduplicate a list of text blocks.
    
    Returns list of (text, fingerprint, duplicate_indices) for unique texts.
    """
    fps = [fingerprint(t) for t in texts]
    result = []
    seen = set()

    for i, (text, fp) in enumerate(zip(texts, fps)):
        if i in seen:
            continue
        dupes = []
        for j in range(i + 1, len(texts)):
            if j not in seen and similarity(fp, fps[j]) >= threshold:
                dupes.append(j)
                seen.add(j)
        result.append((text, fp, dupes))
        seen.add(i)

    return result
