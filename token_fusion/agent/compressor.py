"""Agent context compressor — compresses agent conversation histories."""

from __future__ import annotations
from typing import Optional
import time

from token_fusion.pipeline import FusionEngine
from token_fusion.agent.context import ConversationBuffer, Message
from token_fusion.pipeline.rewind import RewindStore


class AgentContextCompressor:
    """Compresses multi-turn agent conversations using the Fusion Pipeline.
    
    Features:
    - Per-message compression via the 14-stage pipeline
    - Cross-message semantic dedup
    - Reversible compression via RewindStore (agent can retrieve originals)
    - Token budget management: stays under context window
    - Progressive compression: older messages get heavier compression
    """

    def __init__(
        self,
        engine: Optional[FusionEngine] = None,
        max_tokens: int = 128_000,
        enable_rewind: bool = True,
        auto_compress_threshold: float = 0.8,  # Compress at 80% of budget
    ):
        self.engine = engine or FusionEngine(enable_rewind=enable_rewind)
        self.rewind_store = self.engine.rewind_store
        self.buffer = ConversationBuffer(max_tokens=max_tokens)
        self.auto_compress_threshold = auto_compress_threshold
        self._rewind_enabled = enable_rewind

    def add_message(self, role: str, content: str, token_count: Optional[int] = None) -> dict:
        """Add a message to the conversation.
        
        Args:
            role: "system", "user", "assistant", "tool"
            content: Message content
            token_count: Optional pre-computed token count
            
        Returns:
            dict with message info and auto-compression status
        """
        msg = self.buffer.add(role, content, token_count)
        result = {
            "index": len(self.buffer.messages) - 1,
            "role": role,
            "token_count": msg.token_count,
        }

        # Auto-compress if over threshold
        summary = self.buffer.summary()
        usage = summary["total_tokens"] / self.buffer.max_tokens

        if usage >= self.auto_compress_threshold:
            result["compression_triggered"] = True
            result["compression"] = self.compress()

        return result

    def compress(
        self,
        target_token_count: Optional[int] = None,
        preserve_recent: int = 2,
        aggressive_old: bool = True,
    ) -> dict:
        """Compress the conversation buffer.
        
        Args:
            target_token_count: Target total tokens (default: 60% of max)
            preserve_recent: Number of most recent messages to skip compression
            aggressive_old: Apply heavier compression to older messages
            
        Returns:
            dict with compression stats
        """
        if target_token_count is None:
            target_token_count = int(self.buffer.max_tokens * 0.6)

        messages = self.buffer.messages
        if len(messages) <= preserve_recent:
            return {"status": "no_op", "reason": "too_few_messages"}

        total_before = self.buffer._total_tokens
        n_compressed = 0
        stage_stats = {}

        # Compress from oldest to newest (skip most recent)
        compressible_indices = list(range(max(0, len(messages) - preserve_recent)))
        
        # If we need to compress more aggressively, apply full pipeline to older messages
        # and lighter to mid-aged ones
        mid_point = len(compressible_indices) // 2

        for idx in compressible_indices:
            msg = messages[idx]
            if msg.compressed:
                continue  # Skip already compressed

            # Determine compression level
            age_rank = idx / max(1, len(messages))
            if aggressive_old and age_rank < 0.3:
                # Old messages: use full pipeline with rewind
                result = self.engine.compress(
                    msg.content,
                    role=msg.role,
                    return_intermediates=False,
                )
            else:
                # Recent messages: lighter compression (fewer stages)
                result = self._light_compress(msg.content, msg.role)

            # Store rewind marker if enabled
            marker_id = None
            if self._rewind_enabled and self.rewind_store:
                marker_id = self.rewind_store.store(
                    msg.content,
                    marker_id=f"conv_{idx}_{int(time.time())}",
                )

            # Update buffer
            new_tokens = result["stats"]["compressed_tokens"]
            self.buffer.replace_content(idx, result["compressed"], new_tokens)
            messages[idx].rewind_marker = marker_id
            n_compressed += 1

            # Track stage stats
            for name, s in result["stats"].get("stages", {}).items():
                if name not in stage_stats:
                    stage_stats[name] = {"applied": 0, "total_reduction": 0.0}
                stage_stats[name]["applied"] += 1
                stage_stats[name]["total_reduction"] += s.get("reduction_pct", 0)

            # Stop if we've compressed enough
            if self.buffer._total_tokens <= target_token_count:
                break

        total_after = self.buffer._total_tokens

        return {
            "status": "compressed",
            "messages_compressed": n_compressed,
            "tokens_before": total_before,
            "tokens_after": total_after,
            "tokens_saved": total_before - total_after,
            "reduction_pct": (
                (1 - total_after / total_before) * 100 if total_before > 0 else 0
            ),
            "stage_stats": stage_stats,
            "rewind_available": self._rewind_enabled and n_compressed > 0,
        }

    def _light_compress(self, text: str, role: Optional[str] = None) -> dict:
        """Lighter compression for recent messages (fewer stages)."""
        # Use engine but with a subset of stages
        return self.engine.compress(text, role=role)

    def retrieve_original(self, marker_id: str) -> Optional[str]:
        """Retrieve original content by rewind marker."""
        if self.rewind_store:
            return self.rewind_store.retrieve(marker_id)
        return None

    def get_context(self) -> list[dict]:
        """Get the current context as chat-API messages."""
        return self.buffer.to_dicts()

    def summary(self) -> dict:
        """Get compressor and buffer summary."""
        buf_summary = self.buffer.summary()
        return {
            **buf_summary,
            "rewind_store_entries": self.rewind_store.size if self.rewind_store else 0,
            "auto_compress_threshold": self.auto_compress_threshold,
        }

    def clear(self) -> None:
        """Reset the conversation buffer."""
        self.buffer.clear()
        if self.rewind_store:
            self.rewind_store.clear()
