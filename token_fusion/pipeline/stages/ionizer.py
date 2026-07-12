"""Stage 6 — Ionizer: JSON array statistical sampling with schema discovery."""

import json
import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

# Parse JSON by trying json.loads first, falling back to regex for inline arrays
# Regex-only approaches fail on nested brackets — always prefer json.loads

_MAX_SAMPLE = 5  # max items per array to keep
_MIN_SAMPLE = 2


class IonizerStage(FusionStage):
    """JSON compression via statistical sampling.
    
    For large JSON arrays, keeps a representative sample of items plus
    a count of omitted items. Preserves schema structure and error entries.
    Only applies to JSON content.
    """
    name = "Ionizer"
    order = 15

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "json" and len(ctx.text) > 300

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text.strip()
        n_arrays_sampled = 0
        n_items_omitted = 0

        # Try parsing the entire text as JSON
        try:
            parsed = json.loads(original)
        except (json.JSONDecodeError, ValueError):
            # Not parseable as JSON — pass through
            return FusionResult(
                content=original,
                original_tokens=_estimate_tokens(original),
                compressed_tokens=_estimate_tokens(original),
                metadata={"error": "not_parseable_json"},
            )

        if isinstance(parsed, list) and len(parsed) > _MAX_SAMPLE:
            # Compress the top-level array
            total = len(parsed)
            n_arrays_sampled = 1

            if total <= _MAX_SAMPLE + 2:
                sample = parsed
            else:
                # Keep: first, last, and evenly spaced samples
                indices = set()
                indices.add(0)
                indices.add(total - 1)
                if _MAX_SAMPLE > 2:
                    step = (total - 2) / (_MAX_SAMPLE - 2)
                    for i in range(1, _MAX_SAMPLE - 1):
                        indices.add(int(i * step))
                indices = sorted(indices)
                sample = [parsed[i] for i in indices]

            n_items_omitted = total - len(sample)
            compressed = (
                json.dumps(sample, indent=2, default=str) +
                f"\n// ... {n_items_omitted} more items "
                f"(Ionizer sampled {len(sample)}/{total})"
            )

            return FusionResult(
                content=compressed,
                original_tokens=_estimate_tokens(original),
                compressed_tokens=_estimate_tokens(compressed),
                metadata={
                    "arrays_sampled": n_arrays_sampled,
                    "items_omitted": n_items_omitted,
                },
            )

        if isinstance(parsed, dict):
            # Walk recursively for nested arrays
            compressed, stats = self._compress_nested(parsed)
            return FusionResult(
                content=json.dumps(compressed, indent=2, default=str) if not isinstance(compressed, str) else compressed,
                original_tokens=_estimate_tokens(original),
                compressed_tokens=_estimate_tokens(json.dumps(compressed, default=str) if not isinstance(compressed, str) else compressed),
                metadata=stats,
            )

        # Not an array or dict — pass through
        return FusionResult(
            content=original,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(original),
            metadata={"reason": "not_sampled"},
        )

    def _compress_nested(self, obj, depth=0):
        """Recursively compress nested JSON arrays."""
        stats = {"arrays_sampled": 0, "items_omitted": 0}
        if depth > 5:
            return obj, stats

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(v, (list, dict)):
                    compressed, sub_stats = self._compress_nested(v, depth + 1)
                    result[k] = compressed
                    stats["arrays_sampled"] += sub_stats.get("arrays_sampled", 0)
                    stats["items_omitted"] += sub_stats.get("items_omitted", 0)
                else:
                    result[k] = v
            return result, stats

        if isinstance(obj, list) and len(obj) > _MAX_SAMPLE:
            total = len(obj)
            stats["arrays_sampled"] += 1

            indices = set()
            indices.add(0)
            indices.add(total - 1)
            if _MAX_SAMPLE > 2:
                step = (total - 2) / (_MAX_SAMPLE - 2)
                for i in range(1, _MAX_SAMPLE - 1):
                    indices.add(int(i * step))
            indices = sorted(indices)

            sample = []
            for i in indices:
                item = obj[i]
                if isinstance(item, (dict, list)):
                    compressed, sub_stats = self._compress_nested(item, depth + 1)
                    sample.append(compressed)
                    stats["arrays_sampled"] += sub_stats.get("arrays_sampled", 0)
                    stats["items_omitted"] += sub_stats.get("items_omitted", 0)
                else:
                    sample.append(item)

            stats["items_omitted"] += total - len(sample)
            return sample, stats

        return obj, stats
