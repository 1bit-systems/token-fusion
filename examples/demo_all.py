#!/usr/bin/env python3
"""Full-stack demo: Pipeline (A) + Merging (B) + Agent (C)."""

import sys
sys.path.insert(0, "/home/bcloud/token-fusion")

print("=" * 65)
print("  TOKEN FUSION — Full-Stack Demo")
print("  Pipeline · Token Merging · Agent Context Compression")
print("=" * 65)

# ─── Option A: Pipeline ─────────────────────────────────────
print("\n" + "─" * 65)
print("  A) FUSION PIPELINE — 14-Stage Token Compression")
print("─" * 65)

from token_fusion.pipeline import FusionEngine

engine = FusionEngine(enable_rewind=True)

# A1: Python code
code = """def fibonacci(n):
    # Return n-th fibonacci number
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
"""

result = engine.compress(code, content_type="code", language="python")
print(f"\n  Code compression:     {result['stats']['original_tokens']:>5} → {result['stats']['compressed_tokens']:>5} tokens"
      f"  ({result['stats']['reduction_pct']:.1f}%)")
print(f"  Stages applied:       {result['stats']['stage_count']}/14")
print(f"  Compressed ({len(result['compressed'])} chars):")
print(f"    {result['compressed'][:120]}...")

# A2: JSON tool results
import json
tool_results = json.dumps([
    {"file": f"src/module_{i}.py", "error": None, "lint_issues": 0, "coverage": 0.85 + i * 0.01}
    for i in range(200)
], indent=2)

result = engine.compress(tool_results, content_type="json")
print(f"\n  JSON compression:     {result['stats']['original_tokens']:>5} → {result['stats']['compressed_tokens']:>5} tokens"
      f"  ({result['stats']['reduction_pct']:.1f}%)")
print(f"  Rewind markers:       {len(result.get('markers', []))}")
print(f"  Compressed ({len(result['compressed'])} chars):")
print(f"    {result['compressed'][:150]}...")

# A3: Agent logs
log = "\n".join([f"2024-01-01 10:00:{s:02d} INFO agent dispatched task_{s}" for s in range(100)])
result = engine.compress(log, content_type="log")
print(f"\n  Log compression:      {result['stats']['original_tokens']:>5} → {result['stats']['compressed_tokens']:>5} tokens"
      f"  ({result['stats']['reduction_pct']:.1f}%)")

# A4: Chat messages
messages = [
    {"role": "system", "content": "You are a coding assistant specialized in Python and data science."},
    {"role": "user", "content": "Write a pandas pipeline to clean and normalize a dataset with missing values."},
    {"role": "assistant", "content": "Here's a robust pipeline:\n\nimport pandas as pd\nimport numpy as np\n\ndef clean_dataset(df):\n    # Fill numeric NAs with median\n    num_cols = df.select_dtypes(include=[np.number]).columns\n    df[num_cols] = df[num_cols].fillna(df[num_cols].median())\n    # Fill categorical NAs with mode\n    cat_cols = df.select_dtypes(include=['object']).columns\n    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])\n    return df\n\n# Usage:\n# df = pd.read_csv('data.csv')\n# df_clean = clean_dataset(df)"},
]
result = engine.compress_messages(messages)
print(f"\n  Messages compression:  {result['stats']['total_original_tokens']:>5} → {result['stats']['total_compressed_tokens']:>5} tokens"
      f"  ({result['stats']['total_reduction_pct']:.1f}%)")

# A5: Rewind retrieval
if "markers" in result.get("per_message", [{}])[0] if result.get("per_message") else False:
    pass
# Test rewind independently
rewind_result = engine.compress(tool_results, content_type="json")
if rewind_result.get("markers"):
    mid = rewind_result["markers"][0]["id"]
    original = engine.rewind_store.retrieve(mid)
    print(f"\n  Rewind retrieval:     {len(original)} chars restored from marker {mid[:8]}...")

# ─── Option B: Token Merging ─────────────────────────────────
print("\n" + "─" * 65)
print("  B) TOKEN MERGING — ToMe + CoMe + CrossModal + Routing")
print("─" * 65)

import numpy as np
from token_fusion.merging import ToMeMerger, CoMeMerger, CrossModalFuser, AdaptiveRouter
from token_fusion.merging.base import MergingConfig

np.random.seed(42)

# B1: ToMe
tokens = np.random.randn(100, 128).astype(np.float32)
merger = ToMeMerger(MergingConfig(reduction_ratio=0.5, similarity_threshold=0.0))
result = merger(tokens)
print(f"\n  ToMe merging:         {result.original_count:>4} → {result.merged_count:>4} tokens ({result.reduction_pct:.0f}%)"
      f"  [{result.merge_indices[0][0]}←{result.merge_indices[0][1]}, ...]")

# B2: CoMe
high_conf = np.zeros(100)
high_conf[:20] = 1.0  # First 20 tokens are "important"
tokens_conf = np.random.randn(100, 128).astype(np.float32)
tokens_conf[:20] *= 5  # Higher norm
merger = CoMeMerger(MergingConfig(reduction_ratio=0.5))
result = merger(tokens_conf)
# Check that high-confidence tokens were less likely to be merged
mergee_set = {m for _, m in result.merge_indices}
high_conf_merged = len(set(range(20)) & mergee_set)
low_conf_merged = len(set(range(20, 100)) & mergee_set)
print(f"  CoMe merging:         {result.original_count:>4} → {result.merged_count:>4} tokens")
print(f"    High-confidence merged: {high_conf_merged}, Low-confidence merged: {low_conf_merged}")

# B3: Cross-modal fusion
primary = np.random.randn(16, 64).astype(np.float32)
auxiliary = np.random.randn(8, 64).astype(np.float32)
fuser = CrossModalFuser(MergingConfig(), num_heads=4)
result = fuser(primary, auxiliary_tokens=auxiliary)
print(f"\n  Cross-modal fusion:   {result.original_count:>4} total → {result.merged_count:>4} primary tokens")

# B4: Adaptive routing
routable = np.random.randn(50, 64).astype(np.float32)
router = AdaptiveRouter(MergingConfig(similarity_threshold=0.2), num_routes=4)
result = router(routable)
routes = result.metadata["route_distribution"]
print(f"  Adaptive routing:     {result.original_count:>4} tokens → {len(routes)} routes")
for r, c in sorted(routes.items()):
    print(f"    Route {r}: {c} tokens")

# ─── Option C: Agent Context ─────────────────────────────────
print("\n" + "─" * 65)
print("  C) AGENT CONTEXT — Conversation Compression")
print("─" * 65)

from token_fusion.agent import AgentContextCompressor

# Create compressor with small budget to show effect
compressor = AgentContextCompressor(max_tokens=5000, enable_rewind=True)
print(f"\n  Created compressor:   max_tokens={compressor.buffer.max_tokens}, rewind={compressor._rewind_enabled}")

# Add system prompt
compressor.add_message("system", "You are a coding assistant with access to file system, git, and web search tools.")

# Simulate a long agent conversation with repetitive tool results
import json
tool_json = json.dumps([
    {"file": f"src/module_{i}.py", "issues": [
        {"type": "warning", "line": j, "msg": "unused import"}
        for j in range(3)
    ]} for i in range(25)
], indent=2)

conversation = [
    ("user", "Search the codebase for all linting issues in the Python files."),
    ("tool", tool_json),
    ("assistant", "I found 75 linting issues across 25 files. Most are unused imports. " * 20),
    ("user", "Show me the implementation details of the FusionEngine.compress method with all the stages."),
    ("tool", tool_json.replace('src/', 'lib/')),
    ("assistant", "The FusionEngine orchestrates the 14-stage pipeline " + 
     "including Cortex, Photon, RLE, SemanticDedup, Ionizer, and more. " * 15),
]

for _ in range(5):
    for role, content in conversation:
        compressor.add_message(role, content)

summary = compressor.summary()
print(f"\n  Conversation:         {summary['total_messages']} messages, {summary['total_tokens']} tokens")

# Compress
print(f"  Compressing...")
c_result = compressor.compress(target_token_count=2000, preserve_recent=2, aggressive_old=True)

if c_result["status"] == "compressed":
    print(f"  After compression:    {c_result['tokens_after']} tokens ({c_result['reduction_pct']:.1f}% reduction)")
    print(f"  Messages compressed:  {c_result['messages_compressed']}")
    if c_result.get("stage_stats"):
        top_stage = max(c_result["stage_stats"].items(), key=lambda x: x[1]["applied"])
        print(f"  Most active stage:    {top_stage[0]} ({top_stage[1]['applied']}×)")
else:
    print(f"  No compression needed: {c_result.get('reason', 'unknown')}")

print(f"\n  Rewind store:         {compressor.rewind_store.size} entries")
print(f"  Context ready ({len(compressor.get_context())} messages)")

print("\n" + "=" * 65)
print("  ✅ TOKEN FUSION — All systems operational")
print("  Options A + B + C working in concert")
print("=" * 65)
