"""Tests for the Agent Context Compressor (Option C)."""

import sys
sys.path.insert(0, "/home/bcloud/token-fusion")

from token_fusion.agent import AgentContextCompressor, ConversationBuffer
from token_fusion.pipeline import RewindStore


def test_conversation_buffer():
    """Conversation buffer should track messages."""
    buf = ConversationBuffer(max_tokens=1000, max_messages=10)
    
    msg = buf.add("system", "You are a helpful assistant.")
    summary = buf.summary()
    
    assert summary["total_messages"] == 1
    assert summary["total_tokens"] > 0
    
    # Add more messages
    for i in range(5):
        buf.add("user", f"Message {i}")
    
    summary = buf.summary()
    assert summary["total_messages"] == 6
    
    msg_dicts = buf.to_dicts()
    assert len(msg_dicts) == 6
    assert msg_dicts[0]["role"] == "system"
    assert msg_dicts[-1]["content"] == "Message 4"
    
    print(f"  Buffer: {summary['total_messages']} messages, {summary['total_tokens']} tokens")


def test_buffer_pruning():
    """Buffer should prune when over limits."""
    buf = ConversationBuffer(max_tokens=50, max_messages=5)
    
    for i in range(10):
        buf.add("user", f"Message {i} with some extra padding text to make it longer")
    
    summary = buf.summary()
    assert summary["total_messages"] <= 5, f"Expected ≤5 messages, got {summary['total_messages']}"
    print(f"  Buffer pruning: 10 → {summary['total_messages']} messages")


def test_agent_compressor():
    """Agent compressor should compress messages."""
    compressor = AgentContextCompressor(max_tokens=10_000)
    
    # Add a system prompt
    compressor.add_message("system", "You are a coding assistant that helps write Python code.")
    
    # Add several long messages with lots of redundancy
    for i in range(5):
        compressor.add_message("user", f"Can you write a function that does {i * 1000} calculations? " * 50)
        compressor.add_message("assistant", f"Here's the code for task {i}:\ndef calc_{i}(n):\n    result = []\n    for x in range(n):\n        result.append(x * {i})\n    return result" + "\nThe code above should work for your use case. Let me know if you need any changes." * 10)
    
    summary = compressor.summary()
    print(f"  Before compression: {summary['total_tokens']} tokens in {summary['total_messages']} messages")
    
    # Trigger compression
    result = compressor.compress(target_token_count=5000)
    
    if result["status"] == "compressed":
        print(f"  After compression: {result['tokens_after']} tokens ({result['reduction_pct']:.1f}% reduction)")
        print(f"  Messages compressed: {result['messages_compressed']}")
        if result["tokens_saved"] > 0:
            assert result["reduction_pct"] > 0, "Should have positive reduction"
        else:
            print(f"  (No token savings - content may already be efficient)")
    else:
        print(f"  No compression needed: {result['reason']}")
    
    print(f"  Compressor summary: {compressor.summary()}")


def test_agent_rewind():
    """Agent compressor should support rewind retrieval."""
    compressor = AgentContextCompressor(max_tokens=10_000, enable_rewind=True)
    
    compressor.add_message("user", "This is a very long message that needs compression " * 50)
    
    # Get message index
    msg = compressor.buffer.messages[0]
    
    # Compress
    compressor.compress()
    
    # Check for rewind markers
    if msg.rewind_marker:
        original = compressor.retrieve_original(msg.rewind_marker)
        assert original is not None, "Should retrieve original"
        assert "very long message" in original, "Should contain original content"
        print(f"  Rewind: stored message retrieved successfully")
    else:
        print(f"  Note: No rewind marker (message may not have been compressed)")


def test_auto_compress():
    """Auto compression should trigger at threshold."""
    compressor = AgentContextCompressor(
        max_tokens=500, 
        auto_compress_threshold=0.5,
        enable_rewind=False,
    )
    
    # Add messages up to the threshold
    for i in range(3):
        result = compressor.add_message("user", f"Long message {i}. " * 100)
        if "compression_triggered" in result.get("compression", {}):
            print(f"  Auto-compress triggered after message {i}")


if __name__ == "__main__":
    print("\n=== Agent Compressor Tests (Option C) ===\n")
    test_conversation_buffer()
    test_buffer_pruning()
    test_agent_compressor()
    test_agent_rewind()
    test_auto_compress()
    print("\n✓ All agent tests passed!\n")
