"""Stage 4 — RLE: path shorthand, IP prefix compression, enum compaction."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_PATH_PREFIXES = {
    "/home/": "$HOME/",
    "/Users/": "$HOME/",
    "/tmp/": "$TMP/",
    "/var/log/": "$LOG/",
    "/etc/": "$ETC/",
    "/usr/local/": "$LOCAL/",
    "/opt/": "$OPT/",
}

_IPV4_RE = re.compile(r'\b(\d{1,3}\.){3}\d{1,3}\b')
_LONG_PATH_RE = re.compile(r'(/[a-zA-Z0-9_.-]+){4,}')


class RLEStage(FusionStage):
    """Path shorthand replacement, IP address prefix compression, enum compaction.
    
    Replaces common long path prefixes with short variables ($WS, $HOME, etc.).
    Compresses repeated patterns like IPs.
    """
    name = "RLE"
    order = 10

    def should_apply(self, ctx: FusionContext) -> bool:
        return bool(_LONG_PATH_RE.search(ctx.text)) or bool(_IPV4_RE.search(ctx.text))

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        path_count = 0
        ip_count = 0

        # Path prefix replacement
        compressed = original
        for prefix, replacement in _PATH_PREFIXES.items():
            if prefix in compressed:
                compressed = compressed.replace(prefix, replacement)
                path_count += compressed.count(replacement)

        # IP address -> [IP:n] compaction
        seen_ips: dict[str, str] = {}
        ip_counter = [0]

        def _compact_ip(m: re.Match) -> str:
            ip = m.group(0)
            if ip not in seen_ips:
                seen_ips[ip] = f"[IP:{ip_counter[0]}]"
                ip_counter[0] += 1
            nonlocal ip_count
            ip_count += 1
            return seen_ips[ip]

        compressed = _IPV4_RE.sub(_compact_ip, compressed)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"paths_shortened": path_count, "ips_compacted": ip_count},
        )
