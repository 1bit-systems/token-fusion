# Token Fusion — Unified Library

**Pipeline compression · Learned token merging · Agent context optimization**

Three research lineages, one composable library:

| Module | What | Origin |
|--------|------|--------|
| `token_fusion.pipeline` | 14-stage Fusion Pipeline for LLM token compression | Claw Compactor architecture |
| `token_fusion.merging` | Learned token merging (ToMe, Co-Me, cross-modal fusion) | CVPR/ICML paper implementations |
| `token_fusion.agent` | Multi-turn agent conversation compressor | Agent context management |

## Quick Start

```python
from token_fusion.pipeline import FusionEngine

engine = FusionEngine()

result = engine.compress(
    text="def hello():\n    print('hello world')",
    content_type="code",
    language="python",
)

print(result["compressed"])
print(result["stats"]["reduction_pct"])
```
