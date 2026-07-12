"""Cross-modal token fusion via attention — fusing tokens across modalities."""

from __future__ import annotations
from typing import Optional, Any

import numpy as np
from token_fusion.merging.base import TokenMerger, MergingConfig, MergingResult


class CrossModalFuser(TokenMerger):
    """Cross-modal token fusion via learned attention.
    
    Fuses tokens from one modality into another's token sequence using
    cross-modal attention and a learned gating mechanism.
    
    Based on "Multimodal Token Fusion for Vision Transformers" 
    (Wang et al., CVPR 2022) and "Tokenization, Fusion, and Augmentation"
    (Zhang et al., AAAI 2025).
    
    This is a NumPy implementation for standalone use. For GPU training,
    use the PyTorch equivalents.
    """

    name = "CrossModalFuser"

    def __init__(
        self,
        config: Optional[MergingConfig] = None,
        num_heads: int = 4,
    ):
        super().__init__(config)
        self.num_heads = num_heads

    def merge(self, tokens: np.ndarray, **kwargs) -> MergingResult:
        """Cross-modal fusion: fuse auxiliary_tokens into primary tokens.
        
        Args:
            tokens: Primary modality tokens (N, D)
            **kwargs:
                auxiliary_tokens: Required. Tokens from secondary modality (M, D)
                attention_mask: Optional (N,) mask for primary tokens
                
        Returns:
            MergingResult with fused tokens (N, D) where auxiliary info
            has been fused into primary tokens
        """
        aux = kwargs.get("auxiliary_tokens")
        if aux is None:
            raise ValueError("CrossModalFuser requires `auxiliary_tokens` in kwargs")

        aux = np.asarray(aux)
        n_primary = tokens.shape[0]
        n_aux = aux.shape[0]
        d_model = tokens.shape[1]

        # Simple cross-modal attention (NumPy version)
        # Q = primary, K/V = auxiliary
        Q = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-10)
        K = aux / (np.linalg.norm(aux, axis=1, keepdims=True) + 1e-10)

        # Scaled dot-product attention: primary attends to auxiliary
        attn_scores = Q @ K.T / np.sqrt(d_model)
        
        # Softmax over auxiliary tokens
        attn_weights = np.exp(attn_scores - attn_scores.max(axis=1, keepdims=True))
        attn_weights = attn_weights / attn_weights.sum(axis=1, keepdims=True)

        # Weighted sum of auxiliary values
        context = attn_weights @ aux

        # Gated fusion: learnable gate (alpha) determines how much auxiliary info to blend
        # For simplicity, use a sigmoid-gated average
        gate = 0.5  # fixed gate (would be learned in real implementation)
        
        # Compute a per-token gate from attention confidence
        max_attn = attn_weights.max(axis=1)  # (N,) — how focused each primary token is
        per_token_gate = 0.3 + 0.4 * max_attn  # range [0.3, 0.7]

        # Fuse: alpha * primary + (1-alpha) * context from aux
        fused = np.zeros_like(tokens)
        for i in range(n_primary):
            g = per_token_gate[i]
            fused[i] = g * tokens[i] + (1 - g) * context[i]

        # Track merge pairs (primary token i attended to aux token j primarily)
        primary_attends_to = attn_weights.argmax(axis=1)
        merge_pairs = list(zip(range(n_primary), (int(x) for x in primary_attends_to)))

        return MergingResult(
            merged_tokens=fused,
            merge_indices=merge_pairs,
            original_count=n_primary * n_aux,  # total tokens across both modalities
            merged_count=n_primary,  # output is just primary
            metadata={
                "num_aux_tokens": n_aux,
                "num_heads": self.num_heads,
                "gate_type": "attention_confidence",
                "mean_gate": float(per_token_gate.mean()),
            },
        )


def cross_modal_fusion_pytorch(
    primary_tokens: "torch.Tensor",
    auxiliary_tokens: "torch.Tensor",
    num_heads: int = 4,
    dropout: float = 0.1,
) -> "torch.Tensor":
    """PyTorch cross-modal fusion with multi-head attention.
    
    Uses nn.MultiheadAttention for proper learned fusion.
    
    Args:
        primary_tokens: (N, D) primary modality
        auxiliary_tokens: (M, D) auxiliary modality
        num_heads: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        (N, D) fused tokens
    """
    import torch.nn as nn
    
    d_model = primary_tokens.shape[-1]
    mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
    
    # Primary attends to auxiliary: Q=primary, K/V=auxiliary
    fused, attn_weights = mha(
        query=primary_tokens.unsqueeze(0),  # (1, N, D)
        key=auxiliary_tokens.unsqueeze(0),   # (1, M, D)
        value=auxiliary_tokens.unsqueeze(0), # (1, M, D)
    )
    
    return fused.squeeze(0)
