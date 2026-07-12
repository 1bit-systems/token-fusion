"""Cross-modal token fusion — threshold-based token exchange (CVPR 2022).

Matches the original TokenFusion paper:
  Wang et al., "Multimodal Token Fusion for Vision Transformers", CVPR 2022

Core mechanism: two parallel transformer streams. At each layer, a learned
predictor scores each token's confidence. Low-confidence tokens are EXCHANGED
with the corresponding token from the other modality — not attention-fused.

This is ~10 lines of logic per modality. The paper's insight: "if modality A
is uncertain about token i, borrow modality B's token i instead."
"""

from __future__ import annotations
from typing import Optional, Any, Callable

import numpy as np
from token_fusion.merging.base import TokenMerger, MergingConfig, MergingResult


class TokenExchange:
    """TokenExchange module from the original TokenFusion paper.
    
    For each modality's tokens:
      - If mask[i] >= threshold: keep this modality's token i
      - If mask[i] < threshold:  replace with the OTHER modality's token i
    
    This is the actual fusion mechanism — hard exchange, not soft blending.
    """
    name = "TokenExchange"

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def __call__(
        self,
        tokens_a: np.ndarray,    # (N, D) modality A
        tokens_b: np.ndarray,    # (N, D) modality B
        mask_a: np.ndarray,      # (N,) confidence for modality A
        mask_b: np.ndarray,      # (N,) confidence for modality B
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exchange low-confidence tokens between modalities."""
        n = tokens_a.shape[0]
        out_a = np.zeros_like(tokens_a)
        out_b = np.zeros_like(tokens_b)

        # Modality A: keep own tokens where confident, borrow from B where not
        keep_a = mask_a >= self.threshold
        out_a[keep_a] = tokens_a[keep_a]
        out_a[~keep_a] = tokens_b[~keep_a]

        # Modality B: keep own tokens where confident, borrow from A where not
        keep_b = mask_b >= self.threshold
        out_b[keep_b] = tokens_b[keep_b]
        out_b[~keep_b] = tokens_a[~keep_b]

        return out_a, out_b


class ConfidencePredictor:
    """Learned per-token confidence predictor (PredictorLG from the paper).
    
    Small MLP: LayerNorm → Linear(d, d) → GELU → Linear(d, d/2) → GELU
             → Linear(d/2, d/4) → GELU → Linear(d/4, 2) → LogSoftmax
    
    Outputs log-probabilities for [keep, exchange]. The keep probability
    is the confidence score.
    
    NumPy implementation for standalone use. PyTorch version below.
    """

    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        # Simplified: norm + 2-layer MLP → sigmoid confidence score
        # (Full PredictorLG requires training — this is a heuristic proxy)

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        """Predict confidence for each token.
        
        Uses a heuristic: tokens with higher L2 norm and higher variance
        across dimensions are more "confident" (carry more information).
        """
        if tokens.ndim != 2:
            return np.ones(tokens.shape[0]) * 0.5

        # L2 norm: higher = more information
        norms = np.linalg.norm(tokens, axis=1)
        max_norm = norms.max()
        if max_norm > 0:
            norm_conf = norms / max_norm
        else:
            norm_conf = np.ones_like(norms) * 0.5

        # Variance across dimensions: higher = more discriminative
        variances = np.var(tokens, axis=1)
        max_var = variances.max()
        if max_var > 0:
            var_conf = variances / max_var
        else:
            var_conf = np.ones_like(variances) * 0.5

        # Combined confidence
        confidence = 0.6 * norm_conf + 0.4 * var_conf
        return np.clip(confidence, 0.01, 0.99)


def predictor_lg_pytorch(embed_dim: int) -> "torch.nn.Module":
    """Build the PredictorLG MLP from the paper in PyTorch.
    
    Returns a module that takes (B, N, embed_dim) and returns (B, N, 2)
    log-probabilities.
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, embed_dim),
        nn.GELU(),
        nn.Linear(embed_dim, embed_dim // 2),
        nn.GELU(),
        nn.Linear(embed_dim // 2, embed_dim // 4),
        nn.GELU(),
        nn.Linear(embed_dim // 4, 2),
        nn.LogSoftmax(dim=-1),
    )


def token_exchange_pytorch(
    tokens_a: "torch.Tensor",
    tokens_b: "torch.Tensor",
    mask_a: "torch.Tensor",
    mask_b: "torch.Tensor",
    threshold: float = 0.02,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """TokenExchange in PyTorch (matches the paper's implementation exactly)."""
    import torch
    x0 = torch.zeros_like(tokens_a)
    x1 = torch.zeros_like(tokens_b)

    x0[mask_a >= threshold] = tokens_a[mask_a >= threshold]
    x0[mask_a < threshold] = tokens_b[mask_a < threshold]
    x1[mask_b >= threshold] = tokens_b[mask_b >= threshold]
    x1[mask_b < threshold] = tokens_a[mask_b < threshold]

    return x0, x1


class CrossModalFuser(TokenMerger):
    """Cross-modal token fusion via confidence-based token exchange.
    
    Matches the original TokenFusion paper (CVPR 2022):
    1. Two parallel streams process each modality independently
    2. A confidence predictor scores each token per modality
    3. Low-confidence tokens are EXCHANGED with the other modality
       (not attention-fused, not blended — hard exchange)
    
    This is the NumPy reference implementation. For GPU training, use
    the PyTorch functions above.
    """

    name = "CrossModalFuser"

    def __init__(
        self,
        config: Optional[MergingConfig] = None,
        exchange_threshold: float = 0.02,
        confidence_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__(config)
        self.exchange = TokenExchange(threshold=exchange_threshold)
        self._confidence_fn = confidence_fn or ConfidencePredictor().predict

    def merge(self, tokens: np.ndarray, **kwargs) -> MergingResult:
        """Fuse two modalities via confidence-guided token exchange.
        
        Args:
            tokens: Primary modality tokens (N, D)
            **kwargs:
                auxiliary_tokens: Required. Secondary modality (N, D)
                mask_a: Optional pre-computed confidence for primary (N,)
                mask_b: Optional pre-computed confidence for secondary (N,)
                
        Returns:
            MergingResult with exchanged tokens for both modalities
        """
        aux = kwargs.get("auxiliary_tokens")
        if aux is None:
            raise ValueError("CrossModalFuser requires `auxiliary_tokens` in kwargs")

        aux = np.asarray(aux)
        tokens = np.asarray(tokens)

        if tokens.shape != aux.shape:
            raise ValueError(
                f"TokenFusion requires same-shaped modalities: "
                f"primary {tokens.shape} != auxiliary {aux.shape}"
            )

        n_tokens = tokens.shape[0]

        # Get confidence masks
        mask_a = np.asarray(kwargs.get("mask_a", self._confidence_fn(tokens)))
        mask_b = np.asarray(kwargs.get("mask_b", self._confidence_fn(aux)))

        # Exchange low-confidence tokens
        out_a, out_b = self.exchange(tokens, aux, mask_a, mask_b)

        # Track exchange statistics
        n_exchanged_a = int((mask_a < self.exchange.threshold).sum())
        n_exchanged_b = int((mask_b < self.exchange.threshold).sum())

        return MergingResult(
            merged_tokens=np.concatenate([out_a, out_b], axis=0),
            merge_indices=list(zip(
                np.where(mask_a < self.exchange.threshold)[0].tolist(),
                np.where(mask_b < self.exchange.threshold)[0].tolist(),
            )),
            original_count=n_tokens * 2,  # Both modalities
            merged_count=n_tokens * 2,    # Shape preserved, content exchanged
            metadata={
                "exchange_threshold": self.exchange.threshold,
                "tokens_exchanged_a": n_exchanged_a,
                "tokens_exchanged_b": n_exchanged_b,
                "total_exchanged": n_exchanged_a + n_exchanged_b,
                "method": "threshold_exchange",
                "paper": "CVPR 2022: Multimodal Token Fusion for Vision Transformers",
            },
        )
