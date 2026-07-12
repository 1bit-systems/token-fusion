"""Stage 5 — SemanticDedup: confidence-guided content deduplication.

Uses CoMe-style confidence scoring (information density, structural salience)
instead of raw SimHash fingerprints. Low-confidence blocks are merged into
their nearest high-confidence neighbor rather than being dropped entirely.
"""

from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens
from token_fusion.utils.simhash import fingerprint, similarity
import statistics
import math

MIN_CONFIDENCE_THRESHOLD = 0.3  # Blocks below this get merged
MERGE_SIMILARITY_THRESHOLD = 0.7  # Blocks above this similarity get merged regardless of confidence


class SemanticDedupStage(FusionStage):
    """Confidence-guided content deduplication.
    
    Instead of dropping duplicate blocks (SimHash approach), computes a confidence
    score for each block based on:
    - Information density: unique vocabulary / total words
    - Structural salience: presence of headers, code, patterns
    - Entropy: Shannon entropy of character distribution
    
    Low-confidence blocks (boilerplate, repeated code, log noise) are merged into
    the nearest high-confidence block, preserving the information from both.
    """
    name = "SemanticDedup"
    order = 12

    def should_apply(self, ctx: FusionContext) -> bool:
        return len(ctx.text) > 500

    def _confidence_score(self, text: str) -> float:
        """Compute confidence score for a text block.
        
        Combines:
        1. Information density: unique words / total words (higher = more informative)
        2. Structural signals: headers, code patterns, numbers
        3. Shannon entropy: higher entropy = more information
        4. Length penalty: very short blocks are likely noise
        """
        if not text.strip():
            return 0.0
            
        words = text.split()
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        # Information density
        unique_words = len(set(w.lower() for w in words))
        density = unique_words / max(1, total_words)
        
        # Structural signals
        has_header = int(text.strip().startswith(("# ", "## ", "### ", "diff --git", "---", "+++", "@@")))
        has_code = int(any(kw in text for kw in ("def ", "class ", "import ", "fn ", "func ")))
        has_numbers = int(any(c.isdigit() for c in text[:100]))
        structural = (has_header + has_code + has_numbers) / 3.0
        
        # Shannon entropy (character-level)
        if len(text) > 0:
            char_counts = {}
            for c in text.lower():
                char_counts[c] = char_counts.get(c, 0) + 1
            entropy = 0
            for count in char_counts.values():
                p = count / len(text)
                if p > 0:
                    entropy -= p * math.log2(p)
            # Normalize: max entropy for ASCII is ~log2(95) ≈ 6.6
            normalized_entropy = min(1.0, entropy / 6.6)
        else:
            normalized_entropy = 0
        
        # Length penalty: blocks < 20 chars are likely noise
        length_factor = min(1.0, len(text) / 100)
        
        # Weighted combination
        score = (
            0.40 * density +
            0.25 * structural +
            0.25 * normalized_entropy +
            0.10 * length_factor
        )
        
        return min(1.0, max(0.0, score))

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        content_type = ctx.content_profile.content_type.value
        lines = original.split("\n")
        
        # Group into blocks (intelligent sizing based on content length)
        target_blocks = max(5, min(30, len(lines) // 5))
        block_size = max(1, len(lines) // target_blocks)
        blocks = ["\n".join(lines[i:i + block_size]) for i in range(0, len(lines), block_size)]
        
        # For structured content (JSON, code, diff), preserve block order!
        # Only use confidence-guided merging for free-text content.
        is_structured = content_type in ("json", "code", "diff", "log", "search")
        
        # Score each block, keep position
        scored_blocks: list[tuple[int, str, float, int]] = []  # (position, text, confidence, fingerprint)
        for pos, block in enumerate(blocks):
            if not block.strip():
                continue
            conf = self._confidence_score(block)
            fp = fingerprint(block)
            scored_blocks.append((pos, block, conf, fp))
        
        if not scored_blocks:
            return FusionResult(content=original, original_tokens=_estimate_tokens(original),
                                compressed_tokens=_estimate_tokens(original), metadata={})
        
        if is_structured:
            # Order-preserving dedup: keep blocks in original order, merge only duplicates
            keepers: list[tuple[str, float, int]] = []
            merged_count = 0
            
            for pos, text, conf, fp in sorted(scored_blocks, key=lambda x: x[0]):
                # Check if similar to any existing keeper
                is_dup = False
                for k_text, k_conf, k_fp in keepers:
                    if similarity(fp, k_fp) >= MERGE_SIMILARITY_THRESHOLD:
                        is_dup = True
                        merged_count += 1
                        break
                
                if not is_dup:
                    keepers.append((text, conf, fp))
            
            compressed = "\n".join(t for t, _, _ in keepers)
        else:
            # For free text: use confidence-guided merging
            keepers = []
            merged_count = 0
            
            for pos, text, conf, fp in scored_blocks:
                if conf >= MIN_CONFIDENCE_THRESHOLD:
                    # Check similarity against existing keepers
                    found = False
                    for ki, (k_text, k_conf, k_fp) in enumerate(keepers):
                        if similarity(fp, k_fp) >= MERGE_SIMILARITY_THRESHOLD:
                            keepers[ki] = (k_text + "\n" + text, max(k_conf, conf), k_fp)
                            merged_count += 1
                            found = True
                            break
                    if not found:
                        keepers.append((text, conf, fp))
                else:
                    # Low confidence: merge into most similar keeper
                    if keepers:
                        best_ki, best_sim = 0, 0
                        for ki, (k_text, k_conf, k_fp) in enumerate(keepers):
                            sim = similarity(fp, k_fp)
                            if sim > best_sim:
                                best_sim, best_ki = sim, ki
                        if best_sim > 0.3:
                            keepers[best_ki] = (keepers[best_ki][0] + "\n" + text,
                                                max(keepers[best_ki][1], conf),
                                                keepers[best_ki][2])
                            merged_count += 1
                        else:
                            keepers.append((text, conf, fp))
                    else:
                        keepers.append((text, conf, fp))
            
            compressed = "\n".join(t for t, _, _ in keepers)
        
        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={
                "blocks_merged": merged_count,
                "blocks_original": len(blocks),
                "blocks_kept": len(keepers),
            },
        )
