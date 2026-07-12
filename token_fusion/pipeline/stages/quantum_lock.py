"""Stage 1 — QuantumLock: isolates dynamic content in system prompts."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_DYNAMIC_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|<[^>]+>\{[^}]*\}|__[A-Z_]+__")

class QuantumLockStage(FusionStage):
    """Isolates dynamic content references in system prompts for KV-cache efficiency.
    
    Replaces dynamic/variable portions with stable placeholders so the bulk of
    the system prompt hits the KV cache. Only applies to system-role messages.
    """
    name = "QuantumLock"
    order = 3

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.role == "system" and bool(_DYNAMIC_PLACEHOLDER_RE.search(ctx.text))

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        replacements = []
        
        def _replace(m: re.Match) -> str:
            token = m.group(0)
            placeholder = f"[QL:{len(replacements)}]"
            replacements.append((placeholder, token))
            return placeholder

        compressed = _DYNAMIC_PLACEHOLDER_RE.sub(_replace, original)
        
        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"replacements": len(replacements)},
        )
