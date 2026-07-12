"""Stage 8 — SearchCrunch: deduplicates search/grep results."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_SEARCH_LINE_RE = re.compile(r'(?m)^(\S+?):(\d+):(.+)$')
_SEARCH_RESULT_HEADER_RE = re.compile(r'(?m)^{-50,}')


class SearchCrunchStage(FusionStage):
    """Deduplicates and compresses search/grep results.
    
    Groups results by file, removes duplicate file references,
    and truncates long result listings.
    Only applies to search/grep content.
    """
    name = "SearchCrunch"
    order = 17

    def should_apply(self, ctx: FusionContext) -> bool:
        content = ctx.text[:1000]
        return (
            ctx.content_profile.content_type.value == "search"
            or bool(_SEARCH_LINE_RE.search(content))
        )

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text

        # Parse search results: group by filename
        file_groups: dict[str, list[tuple[int, str]]] = {}
        other_lines: list[str] = []

        for line in original.split("\n"):
            m = _SEARCH_LINE_RE.match(line)
            if m:
                fname, lineno, content = m.group(1), int(m.group(2)), m.group(3)
                if fname not in file_groups:
                    file_groups[fname] = []
                file_groups[fname].append((lineno, content))
            else:
                other_lines.append(line)

        if not file_groups:
            return FusionResult(
                content=original,
                original_tokens=_estimate_tokens(original),
                compressed_tokens=_estimate_tokens(original),
                metadata={"search_results_found": 0},
            )

        # Compress: keep max 10 lines per file with range notation
        compressed = list(other_lines)
        n_deduped = 0

        for fname, entries in sorted(file_groups.items()):
            entries.sort()
            compressed.append(f"{fname}: ({len(entries)} matches)")
            if len(entries) > 10:
                # Show first 3 and last 2
                for lineno, content in entries[:3]:
                    compressed.append(f"  {lineno}:{content}")
                compressed.append(f"  ... {len(entries) - 5} more matches ...")
                for lineno, content in entries[-2:]:
                    compressed.append(f"  {lineno}:{content}")
                n_deduped += len(entries) - 10
            else:
                for lineno, content in entries:
                    compressed.append(f"  {lineno}:{content}")

        result_text = "\n".join(compressed)

        return FusionResult(
            content=result_text,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(result_text),
            metadata={
                "files_found": len(file_groups),
                "lines_deduped": n_deduped,
                "total_matches": sum(len(v) for v in file_groups.values()),
            },
        )
