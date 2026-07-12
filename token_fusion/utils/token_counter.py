"""Token counting utilities — heuristic and accurate."""

import re
from typing import Optional

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Rough token estimator: ~4 chars per token for English text
CHARS_PER_TOKEN = 4.0
# Code is denser: ~3 chars per token
CODE_CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """Heuristic token count when tiktoken isn't available."""
    ratio = CODE_CHARS_PER_TOKEN if is_code else CHARS_PER_TOKEN
    return max(1, int(len(text) / ratio))


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Accurate token count using tiktoken (falls back to heuristic)."""
    if not HAS_TIKTOKEN:
        return estimate_tokens(text)

    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return estimate_tokens(text)


def format_report(stats: dict) -> str:
    """Format compression stats as a human-readable table."""
    before = stats.get("original_tokens", 0)
    after = stats.get("compressed_tokens", 0)
    reduction = stats.get("reduction_pct", 0)
    time_ms = stats.get("time_ms", 0)
    stages = stats.get("stages", {})

    lines = [
        "Token Compression Report",
        "─" * 50,
        f"  Before:  {before:>10,} tokens",
        f"  After:   {after:>10,} tokens",
        f"  Saved:   {before - after:>10,} tokens ({reduction:.1f}%)",
        f"  Time:    {time_ms:.1f} ms",
        "",
        "  Stage Results:",
    ]

    for stage_name, s in stages.items():
        if s.get("applied"):
            lines.append(
                f"    {stage_name:<20s}  "
                f"reduced {s.get('reduction_pct', 0):>5.1f}%  "
                f"in {s.get('time_ms', 0):>5.1f}ms"
            )

    return "\n".join(lines)
