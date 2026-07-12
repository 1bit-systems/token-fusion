"""Stage 7 — LogCrunch: folds repeated log lines with occurrence counts."""

import re
from collections import Counter
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_LOG_LINE_RE = re.compile(r'(?m)^.*$')


class LogCrunchStage(FusionStage):
    """Folds repeated log lines with occurrence counts.
    
    When the same log line appears multiple times consecutively,
    replaces repeats with a single occurrence + count.
    Only applies to log content.
    """
    name = "LogCrunch"
    order = 16

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "log" or "log" in ctx.text.lower()[:500]

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        lines = original.split("\n")

        # Consecutive dedup with counts
        compressed_lines = []
        i = 0
        n_folded = 0

        while i < len(lines):
            line = lines[i]
            count = 1
            while i + count < len(lines) and lines[i + count] == line:
                count += 1

            if count > 2:
                compressed_lines.append(f"[x{count}] {line}")
                n_folded += count - 1
            else:
                for _ in range(count):
                    compressed_lines.append(line)
            i += count

        compressed = "\n".join(compressed_lines)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"lines_folded": n_folded, "lines_original": len(lines)},
        )
