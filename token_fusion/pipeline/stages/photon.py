"""Stage 3 — Photon: detects and compresses base64-encoded images."""

import re
from token_fusion.pipeline.base import FusionStage, FusionContext, FusionResult, _estimate_tokens

_BASE64_IMAGE_RE = re.compile(
    r'!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=]{100,})\)'
)
_BASE64_BLOB_RE = re.compile(
    r'(?:[A-Za-z0-9+/=]{200,})'
)

class PhotonStage(FusionStage):
    """Detects base64-encoded images and replaces them with compact markers.
    
    A marker like "![img:abc123...]" preserves image structure while
    drastically reducing token count.
    """
    name = "Photon"
    order = 8

    def should_apply(self, ctx: FusionContext) -> bool:
        return bool(_BASE64_IMAGE_RE.search(ctx.text)) or bool(_BASE64_BLOB_RE.search(ctx.text))

    def apply(self, ctx: FusionContext) -> FusionResult:
        original = ctx.text
        n_replacements = 0
        n_blobs = 0

        # Replace markdown base64 images
        def _replace_img(m: re.Match) -> str:
            nonlocal n_replacements
            n_replacements += 1
            b64 = m.group(1)
            return f"![img:{hash(b64):x}]"

        compressed = _BASE64_IMAGE_RE.sub(_replace_img, original)

        # Replace bare base64 blobs that look like images
        def _replace_blob(m: re.Match) -> str:
            nonlocal n_blobs
            n_blobs += 1
            return f"[b64:{hash(m.group(0)):x}]"

        compressed = _BASE64_BLOB_RE.sub(_replace_blob, compressed)

        return FusionResult(
            content=compressed,
            original_tokens=_estimate_tokens(original),
            compressed_tokens=_estimate_tokens(compressed),
            metadata={"images_compressed": n_replacements, "blobs_compressed": n_blobs},
        )
