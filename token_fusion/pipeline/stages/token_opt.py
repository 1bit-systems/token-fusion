"""Stage 13 — TokenOpt: tokenizer-aware format optimization."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

# Patterns that waste tokens in most tokenizers
_BOLD_ITALIC_RE = re.compile(r'\*{1,3}([^*]+)\*{1,3}')
_EXTRA_WS_RE = re.compile(r' {2,}')
_TRAILING_WS_RE = re.compile(r'(?m)[ \t]+$')
_EMPTY_LINES_RE = re.compile(r'\n{3,}')


class TokenOptStage(FusionStage):
    """Token-format optimization.
    
    Strips bold/italic markers from markdown (saves ~2 tokens each).
    Normalizes whitespace: multiple spaces -> single, trailing ws removed.
    Collapses excessive blank lines.
    """
    name = "TokenOpt"
    order = 40

    def should_apply(self, ctx: FusionContext) -> bool:
        # Applies to all content types
        return bool(_BOLD_ITALIC_RE.search(ctx.text)) or bool(_EXTRA_WS_RE.search(ctx.text))

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        n_formatting_stripped = 0

        # Strip bold/italic markers (keep the text inside)
        def _strip_fmt(m: re.Match) -> str:
            nonlocal n_formatting_stripped
            n_formatting_stripped += 1
            return m.group(1)

        compressed = _BOLD_ITALIC_RE.sub(_strip_fmt, original)

        # Normalize whitespace
        compressed = _EXTRA_WS_RE.sub(" ", compressed)
        compressed = _TRAILING_WS_RE.sub("", compressed)
        compressed = _EMPTY_LINES_RE.sub("\n\n", compressed)
        compressed = compressed.strip()

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"formatting_stripped": n_formatting_stripped},
        )
