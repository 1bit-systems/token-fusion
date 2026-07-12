"""Tests for the Token Merging module (Option B)."""

import sys
sys.path.insert(0, "/home/bcloud/token-fusion")

import numpy as np
from token_fusion.merging import ToMeMerger, CoMeMerger, CrossModalFuser, AdaptiveRouter
from token_fusion.merging.base import MergingConfig


def test_tome_merging():
    """ToMe merger should reduce token count."""
    np.random.seed(42)
    tokens = np.random.randn(20, 64).astype(np.float32)
    
    config = MergingConfig(reduction_ratio=0.5, similarity_threshold=0.1)
    merger = ToMeMerger(config)
    result = merger(tokens)
    
    print(f"  ToMe: {result.original_count} → {result.merged_count} tokens ({result.reduction_pct:.1f}%)")
    assert result.merged_count < result.original_count, "Should reduce token count"
    assert result.merged_count >= 2, "Should keep at least 2 tokens"
    assert result.merged_count == 10, f"Should reduce to 10 tokens (50%), got {result.merged_count}"


def test_tome_weighted():
    """ToMe with weighted merging should work."""
    np.random.seed(42)
    tokens = np.random.randn(20, 64).astype(np.float32)
    importance = np.random.rand(20).astype(np.float32)
    
    config = MergingConfig(reduction_ratio=0.5, similarity_threshold=0.3)
    merger = ToMeMerger(config, merge_mode="weighted")
    result = merger(tokens, importance=importance)
    
    print(f"  ToMe weighted: {result.original_count} → {result.merged_count} tokens")
    assert result.merged_count < result.original_count


def test_come_merging():
    """CoMe merger should use confidence-guided merging."""
    np.random.seed(42)
    tokens = np.random.randn(20, 64).astype(np.float32)
    # Create some tokens with higher "importance"
    tokens[:5] = tokens[:5] * 3  # Higher norm = higher confidence
    
    config = MergingConfig(reduction_ratio=0.5)
    merger = CoMeMerger(config)
    result = merger(tokens)
    
    print(f"  CoMe: {result.original_count} → {result.merged_count} tokens ({result.reduction_pct:.1f}%)")
    assert result.merged_count < result.original_count
    # High-confidence tokens should be keepers, not mergees
    high_conf_tokens = set(range(5))
    mergee_set = {m for _, m in result.merge_indices}
    # At least some high-confidence tokens should NOT be mergees
    assert len(high_conf_tokens - mergee_set) > 0 or len(result.merge_indices) == 0, \
        "High-confidence tokens should prefer to be keepers"


def test_come_custom_confidence():
    """CoMe with custom confidence function."""
    np.random.seed(42)
    tokens = np.random.randn(20, 64).astype(np.float32)
    
    def my_confidence(t):
        return np.linalg.norm(t, axis=1)
    
    config = MergingConfig(reduction_ratio=0.4)
    merger = CoMeMerger(config, confidence_fn=my_confidence)
    result = merger(tokens)
    
    print(f"  CoMe custom confidence: {result.original_count} → {result.merged_count} tokens")
    assert "fallback_mode" in result.metadata
    assert result.metadata["fallback_mode"] == "learned"


def test_cross_modal_fusion():
    """Cross-modal fusion should blend tokens from two modalities."""
    np.random.seed(42)
    primary = np.random.randn(10, 32).astype(np.float32)
    auxiliary = np.random.randn(5, 32).astype(np.float32)
    
    config = MergingConfig()
    fuser = CrossModalFuser(config, num_heads=2)
    result = fuser(primary, auxiliary_tokens=auxiliary)
    
    print(f"  CrossModal: {result.original_count} total tokens → {result.merged_count} fused tokens")
    assert result.merged_tokens.shape == (10, 32), "Should keep primary token shape"


def test_adaptive_routing():
    """Adaptive router should assign tokens to routes."""
    np.random.seed(42)
    tokens = np.random.randn(30, 64).astype(np.float32)
    
    config = MergingConfig(similarity_threshold=0.3)
    router = AdaptiveRouter(config, num_routes=3)
    result = router(tokens)
    
    print(f"  AdaptiveRouter: {result.original_count} tokens → {result.merged_count} routed")
    assert result.merged_tokens.shape == tokens.shape, "Should not change token count"
    assert "route_distribution" in result.metadata
    routes = result.metadata["route_distribution"]
    print(f"    Route distribution: {routes}")
    assert len(routes) <= 3, f"Expected ≤3 routes, got {len(routes)}"


def test_no_merge_when_few_tokens():
    """Should not merge when tokens are ≤ 2."""
    tokens = np.random.randn(2, 64).astype(np.float32)
    config = MergingConfig(reduction_ratio=0.9)
    
    for Merger in [ToMeMerger, CoMeMerger]:
        merger = Merger(config)
        result = merger(tokens)
        assert result.merged_count == 2, f"{Merger.__name__} should not merge 2 tokens"
    print(f"  No-merge guard: works for ToMe and CoMe")


if __name__ == "__main__":
    print("\n=== Merging Tests (Option B) ===\n")
    test_tome_merging()
    test_tome_weighted()
    test_come_merging()
    test_come_custom_confidence()
    test_cross_modal_fusion()
    test_adaptive_routing()
    test_no_merge_when_few_tokens()
    print("\n✓ All merging tests passed!\n")
