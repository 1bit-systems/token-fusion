"""Stage 14 — Abbrev: natural language abbreviation (text only)."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

# Common word -> abbreviation mapping (safe, widely understood)
_ABBREVIATIONS: dict[str, str] = {
    "because": "b/c",
    "with": "w/",
    "without": "w/o",
    "about": "approx",
    "approximately": "approx",
    "especially": "esp",
    "example": "e.g.",
    "for example": "e.g.",
    "that is": "i.e.",
    "information": "info",
    "application": "app",
    "applications": "apps",
    "configuration": "config",
    "configure": "config",
    "configuration": "config",
    "documentation": "docs",
    "document": "doc",
    "implementation": "impl",
    "implement": "impl",
    "initialization": "init",
    "initialize": "init",
    "parameter": "param",
    "parameters": "params",
    "reference": "ref",
    "references": "refs",
    "repository": "repo",
    "repositories": "repos",
    "specification": "spec",
    "specifications": "specs",
    "standard": "std",
    "temporary": "temp",
    "temporary file": "tmp",
    "versus": "vs",
    "version": "v",
}

_ABBREV_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r')\b',
    re.IGNORECASE
)


class AbbrevStage(FusionStage):
    """Natural language abbreviation.
    
    Replaces common English words with standard abbreviations
    (because → b/c, with → w/, configuration → config, etc.).
    ONLY fires on text content — never touches code, JSON, or structured data.
    """
    name = "Abbrev"
    order = 45

    def should_apply(self, ctx: FusionContext) -> bool:
        return ctx.content_profile.content_type.value == "text" and len(ctx.text) > 100

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        n_replaced = 0

        def _replace(m: re.Match) -> str:
            nonlocal n_replaced
            n_replaced += 1
            key = m.group(0).lower()
            return _ABBREVIATIONS[key]

        compressed = _ABBREV_RE.sub(_replace, original)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"words_abbreviated": n_replaced},
        )
