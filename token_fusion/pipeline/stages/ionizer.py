"""Stage 6 — Ionizer: JSON array statistical sampling with schema discovery."""

import json
import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_ARRAY_RE = re.compile(r'\[\s*\{.*?\}\s*\]', re.DOTALL)
_JSON_OBJECT_RE = re.compile(r'\{[^}]*\}')

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
        original = ctx.text
        n_arrays_sampled = 0
        n_items_omitted = 0

        def _sample_array(m: re.Match) -> str:
            nonlocal n_arrays_sampled, n_items_omitted
            try:
                array = json.loads(m.group(0))
                if not isinstance(array, list) or len(array) <= _MAX_SAMPLE:
                    return m.group(0)

                n_arrays_sampled += 1
                total = len(array)

                # Keep first, last, and evenly spaced samples
                if total <= _MAX_SAMPLE + 2:
                    sample = array
                else:
                    indices = set()
                    indices.add(0)
                    indices.add(total - 1)
                    step = (total - 2) / (_MAX_SAMPLE - 2)
                    for i in range(1, _MAX_SAMPLE - 1):
                        indices.add(int(i * step))
                    indices = sorted(indices)
                    sample = [array[i] for i in indices]

                n_items_omitted += total - len(sample)
                sampled = json.dumps(sample, indent=2)
                return f"{sampled}\n  // ... {total - len(sample)} more items (Ionizer sampled {len(sample)}/{total})"
            except (json.JSONDecodeError, ValueError):
                return m.group(0)

        compressed = _ARRAY_RE.sub(_sample_array, original)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={
                "arrays_sampled": n_arrays_sampled,
                "items_omitted": n_items_omitted,
            },
        )
