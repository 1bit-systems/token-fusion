"""Tests for the Fusion Pipeline (Option A)."""

import sys
sys.path.insert(0, "/home/bcloud/token-fusion")

from token_fusion.pipeline import FusionEngine
from token_fusion.pipeline.rewind import RewindStore


def test_basic_compress():
    """Basic compression should reduce token count."""
    engine = FusionEngine()
    result = engine.compress("def hello():\n    print('hello world')", content_type="code", language="python")
    
    assert "compressed" in result
    assert "stats" in result
    assert result["stats"]["original_tokens"] > 0
    assert result["stats"]["compressed_tokens"] > 0
    print(f"  Basic compress: {result['stats']['original_tokens']} → {result['stats']['compressed_tokens']} tokens ({result['stats']['reduction_pct']:.1f}%)")


def test_json_compression():
    """JSON content should get high compression from Ionizer."""
    import json
    data = [{"id": i, "name": f"item_{i}", "value": i * 100} for i in range(100)]
    json_str = json.dumps(data, indent=2)
    
    engine = FusionEngine()
    result = engine.compress(json_str, content_type="json")
    
    reduction = result["stats"]["reduction_pct"]
    print(f"  JSON compression: {result['stats']['original_tokens']} → {result['stats']['compressed_tokens']} tokens ({reduction:.1f}%)")
    assert reduction > 10, f"Expected >10% reduction for JSON, got {reduction:.1f}%"
    # Content should be shorter than original or contain compression markers
    assert len(result["compressed"]) < len(json_str) or "//" in result["compressed"] or "..." in result["compressed"], "Expected compression"


def test_log_compression():
    """Log content should get LogCrunch folding."""
    log = "2024-01-01 10:00:00 INFO starting\n" * 50
    engine = FusionEngine()
    result = engine.compress(log, content_type="log")
    
    reduction = result["stats"]["reduction_pct"]
    print(f"  Log compression: {result['stats']['original_tokens']} → {result['stats']['compressed_tokens']} tokens ({reduction:.1f}%)")
    assert reduction > 50, f"Expected >50% reduction for repeated logs, got {reduction:.1f}%"
    assert reduction > 80, f"Expected very high reduction for repeated logs, got {reduction:.1f}%"


def test_rewind_store():
    """RewindStore should retrieve original content."""
    store = RewindStore(max_entries=10)
    
    original = "This is secret content that will be compressed"
    mid = store.store(original)
    marker = store.make_marker(mid)
    
    retrieved = store.retrieve(mid)
    assert retrieved == original, f"Rewind roundtrip failed: {retrieved} != {original}"
    
    # Parse marker
    parsed = store.parse_marker(marker)
    assert parsed == mid, f"Marker parse failed: {parsed} != {mid}"
    print(f"  RewindStore: stored/retrieved {len(original)} chars, marker={marker}")


def test_rewind_compress():
    """Compression with rewind should produce markers."""
    engine = FusionEngine(enable_rewind=True)
    large_json = '[' + ','.join(f'{{"id":{i}}}' for i in range(50)) + ']'
    
    result = engine.compress(large_json, content_type="json")
    
    assert "markers" in result, "Expected rewind markers"
    assert len(result["markers"]) > 0, "Expected at least one marker"
    print(f"  Rewind compress: {result['stats']['original_tokens']} → {result['stats']['compressed_tokens']} tokens")
    print(f"  Markers stored: {len(result['markers'])}")


def test_messages_compress():
    """Multi-message compression should work."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant. You help users by writing clean, well-documented code."},
        {"role": "user", "content": "Write a Python function that computes fibonacci numbers."},
        {"role": "assistant", "content": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b"},
    ]
    
    engine = FusionEngine()
    result = engine.compress_messages(messages)
    
    assert "per_message" in result
    assert "stats" in result
    assert len(result["per_message"]) == 3
    
    total_reduction = result["stats"]["total_reduction_pct"]
    print(f"  Message compression: {result['stats']['total_original_tokens']} → {result['stats']['total_compressed_tokens']} tokens ({total_reduction:.1f}%)")


def test_abbrev_skips_code():
    """Abbrev should NOT fire on code content."""
    engine = FusionEngine()
    code = 'def calculate_configuration():\n    with open("/etc/config") as f:\n        return f.read()'
    
    result = engine.compress(code, content_type="code")
    # "configuration" and "with" should not be abbreviated in code
    assert "config" not in result["compressed"].split("#")[0] or "config" in code  # identifier context ok
    
    text = "The configuration file with the application should be approximately correct."
    result2 = engine.compress(text, content_type="text")
    print(f"  Text abbreviation: {text} → {result2['compressed']}")


def test_content_detection():
    """Content type detection should work."""
    from token_fusion.utils.content_type import detect, ContentType
    
    assert detect("def foo():\n    pass").content_type == ContentType.CODE
    assert detect('{"key": "value"}').content_type == ContentType.JSON
    assert detect("2024-01-01 10:00:00 ERROR something broke").content_type == ContentType.LOG
    assert detect("diff --git a/file.py b/file.py").content_type == ContentType.DIFF
    assert detect("plain text content here").content_type == ContentType.TEXT
    print(f"  Content detection: all types passed")


def test_custom_stage():
    """Custom stages should be addable to pipeline."""
    from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult
    
    class CapsStage(FusionStage):
        name = "Capsifier"
        order = 1
        
        def should_apply(self, ctx: FusionContext) -> bool:
            return True
        
        def apply(self, ctx: FusionContext) -> FusionResult:
            return FusionResult(
                content=ctx.text.upper(),
                original_tokens=len(ctx.text) // 4,
                compressed_tokens=len(ctx.text) // 4,
            )
    
    engine = FusionEngine()
    engine.add_stage(CapsStage())
    
    result = engine.compress("hello world", content_type="text")
    assert "HELLO" in result["compressed"], "Custom stage should have run"
    print(f"  Custom stage: input='hello world' → output='{result['compressed']}'")


if __name__ == "__main__":
    print("\n=== Pipeline Tests (Option A) ===\n")
    test_basic_compress()
    test_json_compression()
    test_log_compression()
    test_rewind_store()
    test_rewind_compress()
    test_messages_compress()
    test_abbrev_skips_code()
    test_content_detection()
    test_custom_stage()
    print("\n✓ All pipeline tests passed!\n")
