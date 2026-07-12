"""Conversation buffer — tracks multi-turn agent conversations."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = 0.0
    token_count: int = 0
    compressed: bool = False
    rewind_marker: Optional[str] = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ConversationBuffer:
    """Circular buffer for multi-turn agent conversations with compression awareness."""
    max_tokens: int = 128_000      # Context window limit
    max_messages: int = 500        # Absolute message limit
    
    messages: list[Message] = field(default_factory=list)
    _total_tokens: int = 0

    def add(self, role: str, content: str, token_count: Optional[int] = None) -> Message:
        """Add a message to the buffer."""
        msg = Message(
            role=role,
            content=content,
            token_count=token_count or self._estimate(content),
        )
        self.messages.append(msg)
        self._total_tokens += msg.token_count
        self._maybe_prune()
        return msg

    def _estimate(self, text: str) -> int:
        """Rough token estimate."""
        return max(1, len(text) // 4)

    def _maybe_prune(self) -> None:
        """Prune oldest messages if over limits."""
        while len(self.messages) > self.max_messages:
            removed = self.messages.pop(0)
            self._total_tokens -= removed.token_count

        # Prune by token count: keep recent, drop oldest
        while self._total_tokens > self.max_tokens and len(self.messages) > 2:
            # Never drop system prompt (first message) or the last 2 messages
            removed = self.messages.pop(1) if len(self.messages) > 3 else self.messages.pop(1)
            self._total_tokens -= removed.token_count

    def replace_content(self, index: int, new_content: str, 
                        new_token_count: Optional[int] = None) -> None:
        """Replace content of a message (used after compression)."""
        if 0 <= index < len(self.messages):
            old = self.messages[index]
            diff = (new_token_count or self._estimate(new_content)) - old.token_count
            self.messages[index].content = new_content
            if new_token_count:
                self.messages[index].token_count = new_token_count
            self.messages[index].compressed = True
            self._total_tokens += diff

    def to_dicts(self) -> list[dict]:
        """Export messages as chat-API format dicts."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def summary(self) -> dict:
        """Get buffer summary stats."""
        return {
            "total_messages": len(self.messages),
            "total_tokens": self._total_tokens,
            "max_tokens": self.max_tokens,
            "compressed_count": sum(1 for m in self.messages if m.compressed),
        }

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()
        self._total_tokens = 0
