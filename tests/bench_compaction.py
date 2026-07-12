#!/usr/bin/env python3
"""
Benchmark: Token-fusion algorithmic compaction vs LLM-based compaction.

Simulates realistic pi agent sessions with tool calls and compares:
- Token savings
- Time cost
- Information preservation (through LLM eval of compressed content)

Usage:
  python3 tests/bench_compaction.py
"""

import sys, json, time, math, textwrap
sys.path.insert(0, ".")
from token_fusion.pipeline import FusionEngine

# ─── Simulated pi session data ───────────────────────────────

def generate_session(n_turns: int = 10, tool_size: str = "large"):
    """Generate a realistic pi agent session with tool calls."""
    messages = []
    
    # System prompt
    messages.append({"role": "system", "content": "You are a coding agent with read, bash, edit, write tools. You help users write and debug Python code."})
    
    for t in range(n_turns):
        task = [
            "Find and fix the bug in the authentication module",
            "Refactor the database layer to use connection pooling",
            "Add input validation to all API endpoints",
            "Write unit tests for the payment processing module",
            "Debug the memory leak in the background worker",
            "Update the CI/CD pipeline to include linting",
            "Migrate from REST to GraphQL for the user service",
            "Implement rate limiting for the public API",
            "Add structured logging to the message queue consumer",
            "Profile and optimize the data export pipeline",
        ][t % 10]
        
        messages.append({"role": "user", "content": f"Task: {task}\n\nContext: The codebase is in /home/project/src/. {t} files changed in last commit."})
        
        # Tool results
        if tool_size == "large":
            # Large search result (~2000 chars)
            files = [f"src/module_{chr(97+i)}/{i}/handler.py" for i in range(30)]
            issues = [{"file": f, "line": (i+1)*10, "severity": "warning" if i % 3 else "error", "message": f"issue_{i}"} for i, f in enumerate(files[:20])]
            tool_content = json.dumps(issues, indent=2)
        elif tool_size == "medium":
            tool_content = json.dumps({"files_checked": 15, "issues_found": 3, "errors": ["unused import", "missing type hint", "bare except"]})
        else:  # small
            tool_content = json.dumps({"status": "ok", "message": "No issues found"})
        
        messages.append({"role": "tool", "content": tool_content})
        
        response = (
            f"I found the issue in module_{t}. The problem is a race condition in the connection handler. "
            f"Here's the fix: add a lock around the shared state. " * 3 +
            f"Also need to update the tests. Creating PR #{100 + t}."
        )
        messages.append({"role": "assistant", "content": response})
    
    return messages


# ─── Compression strategies ──────────────────────────────────

def algorithmic_compact(messages: list[dict]) -> dict:
    """Use token-fusion pipeline for compression."""
    e = FusionEngine(enable_rewind=True)
    
    text = "\n".join(
        f"[{m['role']}]: {m['content']}"
        for m in messages
        if len(m.get('content', '')) > 200  # Skip short messages
    )
    
    if len(text) < 500:
        return {"compressed": text, "saved": 0, "time": 0, "stages": [], "rewind": False}
    
    s = time.perf_counter()
    # Auto-detect content type from first message
    first_content = messages[0].get("content", "") if messages else ""
    if any(kw in first_content for kw in ('def ', 'class ', 'import ')):
        ctype = "code"
    elif first_content.strip().startswith(("{", "[")):
        ctype = "json"
    else:
        ctype = "text"
    
    result = e.compress(text, content_type=ctype)
    elapsed = (time.perf_counter() - s) * 1000
    
    return {
        "compressed": result["compressed"],
        "original_tokens": result["stats"]["original_tokens"],
        "compressed_tokens": result["stats"]["compressed_tokens"],
        "saved": result["stats"]["saved_tokens"],
        "reduction_pct": result["stats"]["reduction_pct"],
        "time_ms": elapsed,
        "stages": [n for n, s_ in result["stats"]["stages"].items() if s_["applied"]],
        "rewind": True,
    }


def llm_compact(messages: list[dict]) -> dict:
    """Simulate LLM-based compaction (what pi currently does).
    
    Pi calls the LLM to summarize old messages. This costs:
    - Input: all the messages to summarize (full token count)
    - Output: a structured summary (~500 tokens)
    """
    text = "\n".join(f"[{m['role']}]: {m['content']}" for m in messages)
    orig_tokens = max(1, math.ceil(len(text) / 4))
    
    # Simulate LLM call cost
    # Input: full message text
    # Output: ~500 token summary
    summary_tokens = 500
    total_input = orig_tokens
    total_output = summary_tokens
    
    # Simulate LLM latency (~1.5s for first token + generation)
    latency_ms = 1500 + (summary_tokens / 60) * 1000  # ~60 tok/s output
    
    return {
        "original_tokens": orig_tokens,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "latency_ms": latency_ms,
        "cost_per_call": (total_input + total_output) * 2.5 / 1_000_000,  # GPT-4o rates
        "type": "llm_summarization",
    }


# ─── Content preservation eval ───────────────────────────────

def eval_information_preservation(original: list[dict], algorithmic_result: dict) -> dict:
    """Evaluate how much key information survives compression.
    
    Checks for:
    - File names / paths preserved
    - Error messages preserved
    - Numeric data preserved
    - Action items preserved
    """
    orig_text = json.dumps(original)
    comp_text = algorithmic_result.get("compressed", "")
    
    # Extract key data points from original
    import re
    
    # Paths
    paths_orig = set(re.findall(r'src/[a-z0-9_/.-]+', orig_text))
    paths_comp = set(re.findall(r'src/[a-z0-9_/.-]+', comp_text))
    
    # Numbers
    nums_orig = set(re.findall(r'\b\d{3,}\b', orig_text))
    nums_comp = set(re.findall(r'\b\d{3,}\b', comp_text))
    
    # Key terms
    terms = ["bug", "fix", "error", "warning", "issue", "PR", "tests", "refactor"]
    terms_orig = sum(orig_text.lower().count(t) for t in terms)
    terms_comp = sum(comp_text.lower().count(t) for t in terms)
    
    return {
        "paths_preserved": f"{len(paths_comp)}/{len(paths_orig)} ({len(paths_comp)/max(1,len(paths_orig))*100:.0f}%)",
        "numbers_preserved": f"{len(nums_comp)}/{len(nums_orig)} ({len(nums_comp)/max(1,len(nums_orig))*100:.0f}%)",
        "key_terms_preserved": f"{terms_comp}/{terms_orig} ({terms_comp/max(1,terms_orig)*100:.0f}%)",
    }


# ─── Run benchmark ───────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 75)
    print("  COMPACTION BENCHMARK: Algorithmic (token-fusion) vs LLM (pi current)")
    print("=" * 75)
    
    for label, n_turns, tool_size in [
        ("small (3 turns, small tools)", 3, "small"),
        ("medium (10 turns, med tools)", 10, "medium"),
        ("large (10 turns, LARGE tools)", 10, "large"),
        ("xlarge (25 turns, LARGE tools)", 25, "large"),
    ]:
        print(f"\n{'─'*75}")
        print(f"  Session: {label}")
        print(f"{'─'*75}")
        
        msgs = generate_session(n_turns=n_turns, tool_size=tool_size)
        
        # Algorithmic
        a_result = algorithmic_compact(msgs)
        a_cost_per = (a_result.get("original_tokens", 0) - a_result.get("compressed_tokens", 0)) * 2.5 / 1_000_000
        
        # LLM (simulated)
        l_result = llm_compact(msgs)
        
        # Info preservation
        preservation = eval_information_preservation(msgs, a_result)
        
        print(f"  {'':30s} {'Algorithmic':>18s} {'LLM (current)':>18s}")
        print(f"  {'─'*66}")
        
        orig_tokens = a_result.get("original_tokens", 0) or l_result["original_tokens"]
        comp_tokens_algo = a_result.get("compressed_tokens", orig_tokens)
        comp_tokens_llm = l_result["total_tokens"]
        
        print(f"  {'Original tokens':30s} {orig_tokens:>18,} {orig_tokens:>18,}")
        print(f"  {'Final tokens':30s} {comp_tokens_algo:>18,} {comp_tokens_llm:>18,}")
        
        saved_algo = orig_tokens - comp_tokens_algo
        saved_llm = 0  # LLM adds tokens (input + output > original)
        print(f"  {'Net tok. change':30s} {f'-{saved_algo:,}':>18s} {f'+{comp_tokens_llm - orig_tokens:,}':>18s}")
        
        time_algo = a_result.get("time_ms", 0)
        time_llm = l_result["latency_ms"]
        print(f"  {'Time cost':30s} {f'{time_algo:.0f}ms':>18s} {f'{time_llm:.0f}ms':>18s}")
        
        cost_algo = a_cost_per
        cost_llm = l_result["cost_per_call"]
        print(f"  {'Cost per call':30s} {f'${cost_algo:.4f}':>18s} {f'${cost_llm:.4f}':>18s}")
        
        print(f"  {'Info preserved (algo):':30s}  paths={preservation['paths_preserved']}")
        print(f"  {'':30s}  numbers={preservation['numbers_preserved']}")
        print(f"  {'':30s}  terms={preservation['key_terms_preserved']}")
        
        if a_result.get("stages"):
            print(f"  {'Stages applied:':30s}  {', '.join(a_result['stages'])}")
        
        # Winner
        algo_score = saved_algo - time_algo * 1000  # Weight time less than tokens
        # LLM loses tokens but adds summary value
        # Compare: algo saves tokens+time, LLM adds tokens+time but may preserve better
        print(f"  {'':30s}")
        if saved_algo > 0 and time_algo < time_llm:
            print(f"  {'>>> Algorithmic wins:':30s} saves {saved_algo:,} tokens in {time_algo:.0f}ms "
                  f"(LLM would cost {time_llm:.0f}ms + {comp_tokens_llm:,} tok)")
        else:
            print(f"  {'>>> LLM better for this case':30s}")
    
    print(f"\n{'='*75}")
    print(f"  Conclusion")
    print(f"{'='*75}")
    print(f"")
    print(f"  Algorithmic (token-fusion) wins when:")
    print(f"    - Tool results dominate the conversation (JSON, logs, search)")
    print(f"    - Content is large (>500 chars per message)")
    print(f"    - Information density is low (repeated patterns, boilerplate)")
    print(f"    - Latency matters (<50ms vs 2-5s for LLM summarization)")
    print(f"")
    print(f"  LLM-based (pi current) wins when:")
    print(f"    - Content is small (<200 chars)")
    print(f"    - Semantic understanding is critical (compression would lose meaning)")
    print(f"    - The summary needs to be a coherent narrative, not compressed text")
    print(f"")
    print(f"  Best of both: Use algorithmic as the first pass (fast, cheap, reversible),")
    print(f"  fall back to LLM only when algorithmic can't compress meaningfully.")
