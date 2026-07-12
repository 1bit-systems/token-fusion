"""Stage 5 — SemanticDedup: SimHash fingerprint deduplication."""

from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens
from token_fusion.utils.simhash import fingerprint, similarity

SIMILARITY_THRESHOLD = 0.85

class SemanticDedupStage(FusionStage):
    """Deduplicates semantically similar content blocks using SimHash fingerprints.
    
    Splits text into line-group blocks, fingerprints each, and keeps only the first
    occurrence of each near-duplicate block.
    """
    name = "SemanticDedup"
    order = 12

    def should_apply(self, ctx: FusionContext) -> bool:
        # Only useful on larger texts
        return len(ctx.text) > 500

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        lines = original.split("\n")

        # Group lines into blocks of ~5 lines
        block_size = max(1, len(lines) // 20)
        blocks = ["\n".join(lines[i:i + block_size]) for i in range(0, len(lines), block_size)]

        # Deduplicate blocks
        kept_blocks = []
        kept_fps = []
        n_dupes = 0

        for block in blocks:
            if not block.strip():
                kept_blocks.append(block)
                continue

            fp = fingerprint(block)
            is_dup = False
            for existing_fp in kept_fps:
                if similarity(fp, existing_fp) >= SIMILARITY_THRESHOLD:
                    is_dup = True
                    n_dupes += 1
                    break

            if not is_dup:
                kept_blocks.append(block)
                kept_fps.append(fp)

        compressed = "\n".join(kept_blocks)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"blocks_deduplicated": n_dupes, "blocks_original": len(blocks), "blocks_kept": len(kept_blocks)},
        )
