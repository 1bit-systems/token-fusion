"""Stage 12 — Nexus: ML-based token-level compression (with stopword fallback)."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

try:
    from transformers import pipeline as hf_pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Stopwords for fallback (text-only)
_STOPWORDS = set(
    "a an the is it in on at of to for and or but not with by from as be this that"
    " these those was were are been being have has had do does did will would shall"
    " should may might must can could about into through during before after above"
    " below between out off over under again further then once here there when where"
    " why how all each every both few more most other some such no nor only own same"
    " so than too very just also because if then else".split()
)


class NexusStage(FusionStage):
    """ML token-level compression with stopword removal fallback.
    
    With transformers: uses a small model to predict token importance and
    drops unimportant tokens (similar to LLMLingua).
    Without transformers: stopword removal for text content.
    
    Does NOT touch code content — code has no stopwords.
    Falls back gracefully when no ML is available.
    """
    name = "Nexus"
    order = 35

    def should_apply(self, ctx: FusionContext) -> bool:
        # Only apply to text content, not code
        return ctx.content_profile.content_type.value == "text" and len(ctx.text) > 200

    def _apply_stopword_removal(self, text: str) -> tuple[str, dict]:
        """Remove common stopwords from text content."""
        words = re.findall(r'\b\w+\b', text)
        total = len(words)
        kept = [w for w in words if w.lower() not in _STOPWORDS]
        removed = total - len(kept)

        # Rebuild text — preserve approximate structure
        # Simple approach: reconstruct with kept words
        compressed = " ".join(kept)
        return compressed, {"stopwords_removed": removed, "mode": "stopword_fallback"}

    def _apply_ml_compression(self, text: str) -> tuple[str, dict]:
        """Use transformer model for token importance scoring."""
        try:
            # Use a small fill-mask model to estimate token importance
            unmasker = hf_pipeline("fill-mask", model="bert-base-uncased", top_k=1)

            words = text.split()
            if len(words) < 10:
                return text, {"mode": "ml", "tokens_removed": 0}

            # Sample a few words for importance scoring
            import random
            sample_size = min(20, len(words))
            sample_indices = sorted(random.sample(range(len(words)), sample_size))

            important_indices = set(range(len(words)))
            for idx in sample_indices:
                try:
                    # Try to predict this word from context
                    masked = words.copy()
                    masked[idx] = "[MASK]"
                    snippet = " ".join(masked[max(0, idx - 10):idx + 10])
                    result = unmasker(snippet)
                    # If model easily predicts it, it might be less important
                    if result and result[0]["score"] > 0.5:
                        important_indices.discard(idx)
                except Exception:
                    continue

            # Keep only important words
            kept = [words[i] for i in sorted(important_indices)]
            removed = len(words) - len(kept)
            compressed = " ".join(kept)

            return compressed, {"mode": "ml", "tokens_removed": removed, "tokens_original": len(words)}
        except Exception:
            return self._apply_stopword_removal(text)

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text

        if HAS_TRANSFORMERS:
            compressed, stats = self._apply_ml_compression(original)
        else:
            compressed, stats = self._apply_stopword_removal(original)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata=stats,
        )
