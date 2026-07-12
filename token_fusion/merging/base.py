"""Base classes for token merging algorithms."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod
import time


@dataclass
class MergingConfig:
    """Configuration for token merging."""
    reduction_ratio: float = 0.5          # Target ratio of tokens to keep
    similarity_threshold: float = 0.7     # Minimum similarity to merge
    confidence_threshold: float = 0.3     # Minimum confidence to keep a token
    max_iterations: int = 1               # Merge iterations per layer
    use_cuda: bool = False                # Use CUDA if available
    device: str = "cpu"


@dataclass
class MergingResult:
    """Result from a token merging operation."""
    merged_tokens: Any                       # numpy or torch array
    merge_indices: list[tuple[int, int]]     # (kept, merged) pairs
    original_count: int
    merged_count: int
    time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduction_pct(self) -> float:
        if self.original_count == 0:
            return 0.0
        return (1 - self.merged_count / self.original_count) * 100


class TokenMerger(ABC):
    """Base class for all token merging algorithms."""

    name: str = "base_merger"

    def __init__(self, config: Optional[MergingConfig] = None):
        self.config = config or MergingConfig()

    @abstractmethod
    def merge(self, tokens: Any, **kwargs) -> MergingResult:
        """Merge a set of tokens and return the reduced set."""
        ...

    def __call__(self, tokens: Any, **kwargs) -> MergingResult:
        start = time.perf_counter()
        result = self.merge(tokens, **kwargs)
        result.time_ms = (time.perf_counter() - start) * 1000
        return result
