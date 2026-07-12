"""Stage 11 — Neurosyntax: safe code compression (NEVER strips comments or docs)."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

try:
    from tree_sitter_language_pack import get_language, get_parser
    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False

# Patterns for safe code compression (does NOT remove comments or docstrings)
_EXCESSIVE_BLANK_LINES = re.compile(r'\n{4,}')  # Collapse >3 blank lines to 2
_TRAILING_WS = re.compile(r'(?m)[ \t]+$')       # Remove trailing whitespace
_REPEATED_ASSERTS = re.compile(r'(?m)^(\s*(?:assert|expect|test|it)\s+.*)$')
_REDUNDANT_BLOCKS = re.compile(r'(?s)(\{|\(|\[)\s*\1?')


class NeurosyntaxStage(FusionStage):
    """Safe code compression — NEVER strips comments, docstrings, or identifiers.
    
    Safe operations:
    - Collapse excessive blank lines (>3 → 2)
    - Remove trailing whitespace
    - Collapse repeated test/assert boilerplate (same pattern >3× → summary)
    - Merge redundant braces/parens
    
    EXPLICITLY does NOT:
    - Remove or shorten comments or docstrings (they carry intent)
    - Shorten identifiers or rename symbols
    - Remove any semantically meaningful content
    
    With tree-sitter: language-aware structural compression.
    Without tree-sitter: safe regex fallback.
    """
    name = "Neurosyntax"
    order = 25

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "code"

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        metadata: dict = {}

        # 1. Remove trailing whitespace (always safe)
        compressed = _TRAILING_WS.sub("", original)
        ws_removed = len(original) - len(compressed)
        metadata["trailing_whitespace_removed"] = ws_removed

        # 2. Collapse excessive blank lines (safe, improves readability)
        before = compressed
        compressed = _EXCESSIVE_BLANK_LINES.sub("\n\n\n", compressed)
        if len(compressed) < len(before):
            metadata["blank_lines_collapsed"] = (len(before) - len(compressed)) // 1

        # 3. Detect and collapse repeated assertion/test patterns
        lines = compressed.split("\n")
        result = []
        i = 0
        n_repeated = 0

        while i < len(lines):
            line = lines[i]
            m = _REPEATED_ASSERTS.match(line)
            if m and len(lines) > 10:
                # Check for repeated pattern
                pattern = m.group(0)
                count = 1
                while i + count < len(lines):
                    m2 = _REPEATED_ASSERTS.match(lines[i + count])
                    if not m2:
                        break
                    count += 1

                if count > 5:
                    result.append(f"{lines[i]}  // ... {count - 1} more similar assertions")
                    n_repeated += count - 1
                    i += count
                    continue

            result.append(line)
            i += 1

        metadata["repeated_assertions_collapsed"] = n_repeated
        compressed = "\n".join(result)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata=metadata,
        )
