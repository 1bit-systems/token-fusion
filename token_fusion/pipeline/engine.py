"""FusionEngine — orchestrates the 14-stage Fusion Pipeline."""

from __future__ import annotations
from typing import Optional, Callable
import time

from token_fusion.pipeline.base import (
    FusionStage, FusionContext, FusionResult, _estimate_tokens,
)
from token_fusion.pipeline.rewind import RewindStore
from token_fusion.pipeline.stages import ALL_STAGES, DEFAULT_STAGES
from token_fusion.utils.content_type import detect as detect_content, ContentProfile, ContentType
from token_fusion.utils.token_counter import estimate_tokens


class FusionEngine:
    """The Fusion Pipeline orchestrator.
    
    Uses 12 default stages that provide meaningful token reduction.
    Abbrev and TokenOpt stages exist but save <5 tokens on typical content
    and are excluded by default. Enable with include_micro_opt=True.
    
    Usage:
        engine = FusionEngine()
        result = engine.compress("some text", content_type="code")
        
        # With reversible compression:
        engine = FusionEngine(enable_rewind=True)
        result = engine.compress(large_json, content_type="json")
        original = engine.rewind_store.retrieve(marker_id)
    """

    def __init__(
        self,
        stages: Optional[list[FusionStage]] = None,
        enable_rewind: bool = False,
        rewind_max_entries: int = 10_000,
        include_micro_opt: bool = False,
    ):
        if stages is not None:
            self._stages = stages
        elif include_micro_opt:
            self._stages = [cls() for cls in ALL_STAGES]
        else:
            self._stages = [cls() for cls in DEFAULT_STAGES]
        self._stages.sort(key=lambda s: s.order)
        self.rewind_store = RewindStore(max_entries=rewind_max_entries) if enable_rewind else None
        self._enable_rewind = enable_rewind

    @property
    def pipeline(self) -> list[FusionStage]:
        """Access the stage list (for in-place modifications like adding custom stages)."""
        return self._stages

    def add_stage(self, stage: FusionStage) -> FusionEngine:
        """Add a custom stage to the pipeline."""
        self._stages.append(stage)
        self._stages.sort(key=lambda s: s.order)
        return self

    def compress(
        self,
        text: str,
        content_type: Optional[str] = None,
        language: Optional[str] = None,
        role: Optional[str] = None,
        return_intermediates: bool = False,
    ) -> dict:
        """Run the full compression pipeline on a text input.
        
        Args:
            text: Input text to compress
            content_type: Optional hint ("code", "json", "log", "diff", "search", "text")
            language: Optional language hint ("python", "go", etc.)
            role: Message role ("system", "user", "assistant", "tool")
            return_intermediates: If True, include output from every stage
            
        Returns:
            dict with keys: compressed, stats, markers (if rewind enabled),
                            intermediates (if requested)
        """
        start_time = time.perf_counter()

        # Detect content profile
        if content_type:
            profile = ContentProfile(ContentType(content_type), language, 1.0)
        else:
            profile = detect_content(text)

        # Build initial context
        ctx = FusionContext(
            text=text,
            content_profile=profile,
            original_tokens=estimate_tokens(text, is_code=profile.content_type == ContentType.CODE),
            role=role,
        )

        # Run stages
        original_tokens = ctx.original_tokens
        stage_results: list[dict] = []
        current_text = text

        for stage in self._stages:
            # Update context with current text
            ctx = FusionContext(
                text=current_text,
                content_profile=profile,
                original_tokens=estimate_tokens(text, is_code=profile.content_type == ContentType.CODE),
                role=role,
            )

            result = stage(ctx)
            if result is not None:
                current_text = result.content

                entry = {
                    "name": result.stage_name,
                    "applied": True,
                    "reduction_pct": result.reduction_pct,
                    "time_ms": result.time_ms,
                    "metadata": result.metadata,
                }
                stage_results.append(entry)
            else:
                entry = {"name": stage.name, "applied": False}
                stage_results.append(entry)

        # Calculate final stats
        compressed_tokens = estimate_tokens(current_text, is_code=profile.content_type == ContentType.CODE)
        total_time_ms = (time.perf_counter() - start_time) * 1000
        reduction_pct = (
            (1 - compressed_tokens / original_tokens) * 100
            if original_tokens > 0 else 0
        )

        # Rewind: store original if enabled
        markers = []
        if self._enable_rewind and self.rewind_store:
            marker_id = self.rewind_store.store(text)
            markers.append({"id": marker_id, "marker": self.rewind_store.make_marker(marker_id)})

        result_dict: dict = {
            "compressed": current_text,
            "stats": {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "reduction_pct": reduction_pct,
                "saved_tokens": original_tokens - compressed_tokens,
                "time_ms": total_time_ms,
                "stages": {s["name"]: s for s in stage_results if s["applied"]},
                "stage_count": sum(1 for s in stage_results if s.get("applied")),
            },
        }

        if markers:
            result_dict["markers"] = markers

        if return_intermediates:
            result_dict["intermediates"] = stage_results

        return result_dict

    def compress_messages(
        self,
        messages: list[dict],
        return_intermediates: bool = False,
    ) -> dict:
        """Compress a list of chat messages.
        
        First runs cross-message dedup, then per-message pipeline.
        
        Args:
            messages: List of {"role": ..., "content": ...} dicts
            
        Returns:
            dict with keys: per_message (list), cross_message_dedup (dict), stats
        """
        start_time = time.perf_counter()

        # Per-message compression
        per_message_results = []
        total_before = 0
        total_after = 0

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            result = self.compress(
                content,
                role=role,
                return_intermediates=return_intermediates,
            )
            per_message_results.append({
                "role": role,
                "original": content,
                "compressed": result["compressed"],
                "stats": result["stats"],
            })
            total_before += result["stats"]["original_tokens"]
            total_after += result["stats"]["compressed_tokens"]

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return {
            "per_message": per_message_results,
            "stats": {
                "total_original_tokens": total_before,
                "total_compressed_tokens": total_after,
                "total_reduction_pct": (
                    (1 - total_after / total_before) * 100 if total_before > 0 else 0
                ),
                "total_time_ms": total_time_ms,
            },
        }
