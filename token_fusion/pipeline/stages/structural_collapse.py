"""Stage 10 — StructuralCollapse: merges imports, collapses repeats."""

import re
from collections import Counter
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_IMPORT_RE = re.compile(r'(?m)^(import |from |use |require|#include|using |package )')
_REPEATED_ASSERT_RE = re.compile(r'(?m)^(\s*(assert|expect|test|describe|it|check)\s+)(.*)$')


class StructuralCollapseStage(FusionStage):
    """Merges import blocks and collapses repeated patterns in code.
    
    Groups consecutive import/use/include lines into a single merged
    declaration. Collapses repeated testing/assertion patterns.
    Only applies to code content.
    """
    name = "StructuralCollapse"
    order = 20

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "code"

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        lines = original.split("\n")

        # Identify import blocks
        result_lines = []
        i = 0
        import_block: list[str] = []
        in_import_block = False
        n_imports_merged = 0

        while i < len(lines):
            line = lines[i]
            if _IMPORT_RE.match(line):
                import_block.append(line)
                in_import_block = True
            else:
                if in_import_block and import_block:
                    if len(import_block) > 3:
                        merged = f"// {len(import_block)} imports merged"
                        result_lines.append(merged)
                        n_imports_merged += len(import_block)
                    else:
                        result_lines.extend(import_block)
                    import_block = []
                    in_import_block = False
                result_lines.append(line)
            i += 1

        # Flush any remaining import block
        if import_block:
            if len(import_block) > 3:
                result_lines.append(f"// {len(import_block)} imports merged")
                n_imports_merged += len(import_block)
            else:
                result_lines.extend(import_block)

        compressed = "\n".join(result_lines)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"imports_merged": n_imports_merged},
        )
