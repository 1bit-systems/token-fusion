"""Stage 2 — Cortex: auto-detects content type and programming language."""

from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens
from token_fusion.utils.content_type import detect as detect_content

class CortexStage(FusionStage):
    """Auto-detects content type and programming language for downstream stages.
    
    Passes detection results via metadata. This stage is a no-op for compression
    but essential for gating in downstream stages.
    """
    name = "Cortex"
    order = 5

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "unknown" or ctx.metadata.get("force_detect")

    def apply(self, ctx: FusionContext) -> FusionResult:
        profile = detect_content(ctx.text)
        return FusionResult(
            content=ctx.text,
            original_tokens=_estimate_tokens(ctx.text),
            compressed_tokens=_estimate_tokens(ctx.text),
            metadata={
                "content_type": profile.content_type.value,
                "language": profile.language,
                "confidence": profile.confidence,
            },
        )
