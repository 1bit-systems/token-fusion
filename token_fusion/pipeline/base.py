"""Base classes for the Fusion Pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
from abc import ABC, abstractmethod
import time

from token_fusion.utils.content_type import ContentProfile, ContentType


@dataclass(frozen=True)
class FusionContext:
    """Immutable context flowing through the pipeline."""
    text: str
    content_profile: ContentProfile
    original_tokens: int = 0
    role: Optional[str] = None  # system, user, assistant, tool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result from a single pipeline stage."""
    content: str
    original_tokens: int
    compressed_tokens: int
    stage_name: str = ""
    time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compressed_tokens / self.original_tokens) * 100


class FusionStage(ABC):
    """Base class for all Fusion Pipeline stages."""
    name: str = "unnamed_stage"
    order: int = 100

    @abstractmethod
    def should_apply(self, ctx: FusionContext) -> bool:
        """Gate: should this stage run on this context?"""
        ...

    @abstractmethod
    def apply(self, ctx: FusionContext) -> FusionResult:
        """Execute compression on the context."""
        ...

    def __call__(self, ctx: FusionContext) -> Optional[FusionResult]:
        """Run stage with gating + timing."""
        if not self.should_apply(ctx):
            return None
        start = time.perf_counter()
        result = self.apply(ctx)
        result.time_ms = (time.perf_counter() - start) * 1000
        result.stage_name = self.name
        return result


_TOKEN_ESTIMATE = 4.0  # chars per token

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // int(_TOKEN_ESTIMATE))
