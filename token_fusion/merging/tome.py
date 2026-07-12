"""ToMe-style token merging — merge most similar token pairs."""

from __future__ import annotations
from typing import Optional, Any
import math

import numpy as np
from token_fusion.merging.base import TokenMerger, MergingConfig, MergingResult


class ToMeMerger(TokenMerger):
    """ToMe (Token Merging) — similarity-based bipartite soft merging.
    
    Splits tokens into two sets and merges each token with its most similar
    partner in the other set. Based on "Token Merging: Your ViT But Faster"
    (Bolya et al., ICML 2023).
    """

    name = "ToMe"

    def __init__(
        self,
        config: Optional[MergingConfig] = None,
        merge_mode: str = "average",
    ):
        super().__init__(config)
        self.merge_mode = merge_mode  # "average", "sum", "weighted"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between two sets of vectors."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
        return a_norm @ b_norm.T

    def merge(self, tokens: np.ndarray, **kwargs) -> MergingResult:
        """Merge tokens using bipartite soft matching.
        
        Args:
            tokens: (N, D) array of token embeddings
            **kwargs: 
                importance: Optional (N,) importance scores per token
                
        Returns:
            MergingResult with reduced token set
        """
        n_tokens = tokens.shape[0]
        if n_tokens <= 2:
            return MergingResult(
                merged_tokens=tokens,
                merge_indices=[],
                original_count=n_tokens,
                merged_count=n_tokens,
            )

        target = max(2, int(n_tokens * (1 - self.config.reduction_ratio)))
        n_merge = n_tokens - target
        importance = kwargs.get("importance")

        # Split tokens into two sets: first half, second half
        a_idx = n_tokens // 2
        a = tokens[:a_idx]
        b = tokens[a_idx:]

        if len(a) == 0 or len(b) == 0:
            return MergingResult(
                merged_tokens=tokens,
                merge_indices=[],
                original_count=n_tokens,
                merged_count=n_tokens,
            )

        # Compute similarity matrix
        sim = self._cosine_similarity(a, b)

        # Bipartite matching: greedily merge most similar pairs
        a_idx_arr = np.arange(len(a))
        b_idx_arr = np.arange(len(b))
        merge_pairs: list[tuple[int, int]] = []
        a_used = set()
        b_used = set()

        while len(merge_pairs) < n_merge and len(a_used) < len(a) and len(b_used) < len(b):
            # Find the most similar unused pair
            mask_a = np.array([i not in a_used for i in a_idx_arr])
            mask_b = np.array([j not in b_used for j in b_idx_arr])
            masked_sim = sim.copy()
            masked_sim[~mask_a] = -1
            masked_sim[:, ~mask_b] = -1

            max_val = np.max(masked_sim)
            if max_val < self.config.similarity_threshold:
                break

            i, j = np.unravel_index(np.argmax(masked_sim), masked_sim.shape)
            a_used.add(i)
            b_used.add(j)

            # Map back to original token indices
            orig_i = i  # already in range [0, a_idx)
            orig_j = a_idx + j  # offset by a_idx
            merge_pairs.append((orig_i, orig_j))

        # Build merged token set
        kept_indices = list(set(range(n_tokens)) - {j for _, j in merge_pairs})
        mergee_indices = {j for _, j in merge_pairs}  # indices being merged INTO keepers

        # Create merged tokens
        new_tokens = []
        used_mergees = set()

        for i, (keep_idx, merge_idx) in enumerate(merge_pairs):
            if self.merge_mode == "average":
                merged = (tokens[keep_idx] + tokens[merge_idx]) / 2
            elif self.merge_mode == "sum":
                merged = tokens[keep_idx] + tokens[merge_idx]
            elif self.merge_mode == "weighted" and importance is not None:
                w1, w2 = importance[keep_idx], importance[merge_idx]
                total = w1 + w2 + 1e-10
                merged = (tokens[keep_idx] * w1 + tokens[merge_idx] * w2) / total
            else:
                merged = (tokens[keep_idx] + tokens[merge_idx]) / 2

            new_tokens.append(merged)
            used_mergees.add(merge_idx)

        # Add remaining unmerged tokens
        for i in range(n_tokens):
            if i not in {x for pair in merge_pairs for x in pair}:
                new_tokens.append(tokens[i])

        result = np.stack(new_tokens[:target])

        return MergingResult(
            merged_tokens=result,
            merge_indices=merge_pairs,
            original_count=n_tokens,
            merged_count=result.shape[0],
            metadata={"merge_mode": self.merge_mode, "target": target},
        )
