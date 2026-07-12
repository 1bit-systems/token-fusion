"""Adaptive token-level routing — FreeFuse-style token routing."""

from __future__ import annotations
from typing import Optional, Any, Callable

import numpy as np
from token_fusion.merging.base import TokenMerger, MergingConfig, MergingResult


class AdaptiveRouter(TokenMerger):
    """Adaptive Token-Level Routing.
    
    Routes tokens to specialized processing paths based on their content/semantics.
    Inspired by FreeFuse's adaptive routing for multi-subject LoRA fusion.
    
    Unlike ToMe/CoMe which merge similar tokens, this router assigns each
    token to a processing "expert" or "path" based on learned routing decisions.
    """
    
    name = "AdaptiveRouter"

    def __init__(
        self,
        config: Optional[MergingConfig] = None,
        num_routes: int = 2,
        router_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(config)
        self.num_routes = num_routes
        self._router_fn = router_fn

    def _default_router(self, tokens: np.ndarray) -> np.ndarray:
        """Default routing: cluster tokens by their dominant PCA component.
        
        Falls back to random assignment if tokens are too uniform.
        """
        n = tokens.shape[0]
        if n <= self.num_routes:
            return np.arange(n) % self.num_routes

        # Simple routing: mean activation across dimensions as routing signal
        means = tokens.mean(axis=1)
        
        # Use percentiles to create routing boundaries
        boundaries = np.percentile(means, np.linspace(0, 100, self.num_routes + 1)[1:-1])
        routes = np.zeros(n, dtype=int)
        for i, bound in enumerate(boundaries):
            routes[means > bound] = i + 1
            
        return routes

    def merge(self, tokens: np.ndarray, **kwargs) -> MergingResult:
        """Route tokens to processing paths.
        
        Instead of merging, this assigns each token to a route/expert.
        Tokens on the same route can then be processed uniformly.
        
        Args:
            tokens: (N, D) token embeddings
            **kwargs:
                route_fn: Optional custom routing function
                
        Returns:
            MergingResult with assignment metadata. The merged_tokens
            are the same tokens but with routing annotations.
        """
        n_tokens = tokens.shape[0]

        # Get routing assignments
        if "route_fn" in kwargs:
            routes = np.asarray(kwargs["route_fn"](tokens))
        elif self._router_fn is not None:
            routes = np.asarray(self._router_fn(tokens))
        else:
            routes = self._default_router(tokens)

        routes = np.clip(routes, 0, self.num_routes - 1).astype(int)

        # Group tokens by route
        route_groups: dict[int, list[int]] = {}
        for i, r in enumerate(routes):
            if r not in route_groups:
                route_groups[r] = []
            route_groups[r].append(i)

        # Build merge pairs: for tokens in the same route group,
        # merge similar ones (to reduce token count per route)
        merge_pairs = []
        kept_indices = set()
        mergee_indices = set()

        for route_id, indices in route_groups.items():
            if len(indices) <= 1:
                kept_indices.add(indices[0])
                continue

            # Within each route group, use ToMe-style bipartite merging
            group_tokens = tokens[indices]
            n_group = len(group_tokens)
            a_idx = n_group // 2

            if a_idx == 0 or a_idx >= n_group:
                kept_indices.update(indices)
                continue

            a = group_tokens[:a_idx]
            b = group_tokens[a_idx:]

            a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
            b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
            sim = a_norm @ b_norm.T

            used_a = set()
            used_b = set()
            target_merge = max(0, len(indices) - max(1, int(len(indices) * 0.7)))

            for _ in range(target_merge):
                mask_a = np.array([i not in used_a for i in range(len(a))])
                mask_b = np.array([j not in used_b for j in range(len(b))])
                masked_sim = sim.copy()
                masked_sim[~mask_a] = -1
                masked_sim[:, ~mask_b] = -1

                if masked_sim.max() < self.config.similarity_threshold:
                    break

                i, j = np.unravel_index(masked_sim.argmax(), masked_sim.shape)
                used_a.add(i)
                used_b.add(j)

                orig_i = indices[i]
                orig_j = indices[a_idx + j]
                merge_pairs.append((orig_i, orig_j))
                kept_indices.add(orig_i)
                mergee_indices.add(orig_j)

            # Add unmerged tokens
            for i, idx in enumerate(indices):
                if i in used_a or (a_idx + i - len(a)) in used_b if i >= a_idx else False:
                    continue
                if idx not in kept_indices and idx not in mergee_indices:
                    if i < a_idx and i not in used_a:
                        kept_indices.add(idx)
                    elif i >= a_idx and (i - a_idx) not in used_b:
                        kept_indices.add(idx)

        # Build merged tokens (same as input, no actual merging, just routing)
        result_tokens = tokens.copy()

        return MergingResult(
            merged_tokens=result_tokens,
            merge_indices=merge_pairs,
            original_count=n_tokens,
            merged_count=len(result_tokens),
            metadata={
                "num_routes": self.num_routes,
                "route_distribution": {str(k): len(v) for k, v in route_groups.items()},
                "merge_pairs_in_route": len(merge_pairs),
            },
        )
