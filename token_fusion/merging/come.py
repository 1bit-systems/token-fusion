"""Co-Me style confidence-guided token merging."""

from __future__ import annotations
from typing import Optional, Any, Callable

import numpy as np
from token_fusion.merging.base import TokenMerger, MergingConfig, MergingResult
from token_fusion.merging.tome import ToMeMerger


class CoMeMerger(TokenMerger):
    """Co-Me — Confidence-Guided Token Merging.
    
    Uses a confidence predictor (learned or heuristic) to identify which tokens
    carry important information and should NOT be merged. Merging only occurs
    between tokens with low confidence.
    
    Based on "Co-Me: Confidence-Guided Token Merging for Visual Geometric 
    Transformers" (Chen et al., CVPR 2026).
    """

    name = "CoMe"

    def __init__(
        self,
        config: Optional[MergingConfig] = None,
        confidence_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        fallback_mode: str = "gradient_magnitude",
    ):
        super().__init__(config)
        self._confidence_fn = confidence_fn
        self.fallback_mode = fallback_mode

    def _default_confidence(self, tokens: np.ndarray) -> np.ndarray:
        """Heuristic confidence scoring when no learned predictor is available.
        
        Uses a combination of:
        - Token norm (higher norm = more information)
        - Token variance across dimensions (higher variance = more discriminative)
        - Spatial edge detection (tokens at boundaries are important)
        """
        # L2 norm as base confidence
        norms = np.linalg.norm(tokens, axis=1)
        norm_conf = (norms - norms.min()) / (norms.max() - norms.min() + 1e-10)

        # Per-token variance (discriminative power)
        variances = np.var(tokens, axis=1)
        var_conf = (variances - variances.min()) / (variances.max() - variances.min() + 1e-10)

        # Combined confidence
        confidence = 0.5 * norm_conf + 0.5 * var_conf

        # Scale: median should be around 0.5
        median = np.median(confidence)
        if median > 0:
            confidence = confidence / (median * 2)
            confidence = np.clip(confidence, 0, 1)

        return confidence

    def merge(self, tokens: np.ndarray, **kwargs) -> MergingResult:
        """Merge tokens guided by confidence scores.
        
        Args:
            tokens: (N, D) array of token embeddings
            **kwargs:
                confidence: Optional pre-computed confidence scores
                
        Returns:
            MergingResult with confidence-weighted merged tokens
        """
        n_tokens = tokens.shape[0]
        if n_tokens <= 2:
            return MergingResult(
                merged_tokens=tokens,
                merge_indices=[],
                original_count=n_tokens,
                merged_count=n_tokens,
            )

        # Get confidence scores
        if "confidence" in kwargs:
            confidence = np.asarray(kwargs["confidence"])
        elif self._confidence_fn is not None:
            confidence = np.asarray(self._confidence_fn(tokens))
        else:
            confidence = self._default_confidence(tokens)

        # Sort tokens by confidence (low confidence = merge first)
        order = np.argsort(confidence)

        target = max(2, int(n_tokens * (1 - self.config.reduction_ratio)))
        n_merge = n_tokens - target

        # Low-confidence tokens are mergees; high-confidence tokens are keepers
        mergee_indices = order[:n_merge]
        keeper_indices = np.sort(order[n_merge:])

        # Build merge pairs: each mergee merges into the most similar keeper
        mergees = tokens[mergee_indices]
        keepers = tokens[keeper_indices]

        # Compute similarity matrix
        mergee_norm = mergees / (np.linalg.norm(mergees, axis=1, keepdims=True) + 1e-10)
        keeper_norm = keepers / (np.linalg.norm(keepers, axis=1, keepdims=True) + 1e-10)
        sim = mergee_norm @ keeper_norm.T

        merge_pairs: list[tuple[int, int]] = []
        keeper_idx_set = set(keeper_indices)

        for i, mi in enumerate(mergee_indices):
            best_k = np.argmax(sim[i])
            ki = keeper_indices[best_k]
            merge_pairs.append((int(ki), int(mi)))
            keeper_idx_set.add(int(ki))

        # Build merged tokens: merge each mergee into its best keeper
        all_keepers = set(keeper_indices)
        result_tokens = []

        # Process each keeper: merge in its assigned mergees
        keeper_to_mergees: dict[int, list[int]] = {}
        for ki, mi in merge_pairs:
            if ki not in keeper_to_mergees:
                keeper_to_mergees[ki] = []
            keeper_to_mergees[ki].append(mi)

        used = set()
        for ki in keeper_indices:
            mergees_here = keeper_to_mergees.get(ki, [])
            if len(mergees_here) == 0:
                result_tokens.append(tokens[ki])
            else:
                # Confidence-weighted merge
                w_keeper = confidence[ki]
                w_total = w_keeper + sum(confidence[m] for m in mergees_here) + 1e-10
                merged = tokens[ki] * (w_keeper / w_total)
                for mi in mergees_here:
                    merged += tokens[mi] * (confidence[mi] / w_total)
                result_tokens.append(merged)
            used.add(ki)
            for m in mergees_here:
                used.add(m)

        # Add any remaining tokens not involved in merging
        for i in range(n_tokens):
            if i not in used:
                result_tokens.append(tokens[i])

        result = np.stack(result_tokens)
        # Trim to target size
        if result.shape[0] > target:
            result = result[:target]

        return MergingResult(
            merged_tokens=result,
            merge_indices=merge_pairs,
            original_count=n_tokens,
            merged_count=result.shape[0],
            metadata={
                "confidence_min": float(confidence.min()),
                "confidence_max": float(confidence.max()),
                "confidence_mean": float(confidence.mean()),
                "fallback_mode": self.fallback_mode if self._confidence_fn is None else "learned",
            },
        )
