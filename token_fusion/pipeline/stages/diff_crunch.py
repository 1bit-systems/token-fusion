"""Stage 9 — DiffCrunch: folds unchanged context lines in git diffs."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_DIFF_HUNK_RE = re.compile(r'(@@ -\d+,\d+ \+\d+,\d+ @@.*?)(?=\n@@|$)', re.DOTALL)
_UNCHANGED_LINE_RE = re.compile(r'(?m)^ .*$')


class DiffCrunchStage(FusionStage):
    """Folds unchanged surrounding context lines in git diffs.
    
    In diff hunks, consecutive context lines are replaced with a
    count summary. Only changed lines (+/-) are preserved in full.
    """
    name = "DiffCrunch"
    order = 18

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "diff" or "diff --git" in ctx.text[:200]

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text

        def _compress_hunk(m: re.Match) -> str:
            header = m.group(1).split("\n")[0]
            body_lines = m.group(1).split("\n")[1:]
            compressed = [header]

            i = 0
            while i < len(body_lines):
                line = body_lines[i]
                if line.startswith(" "):
                    # Count consecutive context lines
                    count = 1
                    while i + count < len(body_lines) and body_lines[i + count].startswith(" "):
                        count += 1
                    if count > 3:
                        compressed.append(f"  ... {count} unchanged lines ...")
                    else:
                        compressed.extend(body_lines[i:i + count])
                    i += count
                else:
                    compressed.append(line)
                    i += 1

            return "\n".join(compressed)

        compressed = _DIFF_HUNK_RE.sub(_compress_hunk, original)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
        )
