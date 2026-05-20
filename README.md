# Z-Space Core

**Content-addressable tensor memory with deterministic synthesis.**

Z-Space Core is a research-grade tensor store that addresses tensors by the
**hash of their content** instead of by name or by file path. It compresses each
tensor with a tournament of codecs, keeps a verifiable Merkle-DAG of immutable
nodes, and lets you mutate models through **reversible deltas** that never
duplicate the underlying weights.

> ⚠️ **License notice** — This repository is **not** open source. It is
> published under a temporary, all-rights-reserved license while authorship,
> patent strategy, and academic publication are evaluated. See [LICENSE](LICENSE)
> and [AUTHORSHIP.md](AUTHORSHIP.md) before any use beyond personal reading.

---

## Why this exists

Modern ML workflows ship the same tensors over and over: identical weight
matrices across checkpoints, embeddings that barely change between fine-tunes,
buffers that are bit-for-bit equal across replicas. Z-Space Core treats tensors
the way Git treats blobs — once a content hash is known, you never store it
twice, and any descriptor can be replayed exactly from its address.

Compared to `torch.save`, `safetensors`, and DVC pipelines, Z-Space Core is
designed to:

- **Deduplicate by content**, not by filename.
- **Verify reconstruction** bit-exactly via XOR residuals on top of a lossy
  approximation.
- **Version models with O(delta) growth** instead of O(full checkpoint) growth.
- **Stay framework-agnostic at the wire level** by serializing through `mscs`
  (no `pickle`, no arbitrary code execution at load time).

---

## Core ideas

| Property | What it gives you |
| --- | --- |
| **Canonical SHA-256 addresses** | Every descriptor and every node has a stable, collision-resistant identity. |
| **Merkle-DAG node store** | Descriptors point to nodes by hash, so identical sub-tensors are shared automatically. |
| **Codec tournament** | At register time the engine races `RAW`, `SPARSE`, `SVD`, and optional TensorLy strategies and picks the smallest faithful encoding. |
| **Hybrid lossy + exact** | A low-rank approximation is stored alongside a **bitwise XOR residual**, so `exact=True` reconstructs the original tensor bit-for-bit. |
| **Progressive loads** | `load(..., exact=False)` returns the cheap approximation; `exact=True` materializes the full tensor and verifies the hash. |
| **Reversible deltas** | `add`, `mul`, and `patch` mutations record the previous values, so any version can be rewound. |
| **Safe serialization** | Components and deltas are written through `mscs`. No `pickle`. |
| **Cache hygiene** | The LRU returns **clones**, so cached entries can never be mutated by a caller. |

---

## Installation

```powershell
pip install -r requirements.txt
```

`mscs` is required (it owns the safe serialization path). The following are
optional at runtime — Z-Space Core degrades gracefully if they are missing:

- `lz4` → falls back to `zlib`.
- `tensorly` → `RAW`, `SPARSE`, and `SVD` still work; Tucker / CP / TT codecs
  are skipped.
- `zstandard`, `safetensors`, `dvc` → only used by benchmark scripts.

Python 3.10+ is recommended.

---

## Quick start

```python
import torch
from z_space_core import DecompType, ZSpace

space = ZSpace(cache_size=1 << 28)

# Build a tensor that has obvious low-rank structure.
a = torch.arange(120, dtype=torch.float32).reshape(20, 6)
b = torch.arange(90, dtype=torch.float32).reshape(6, 15) / 31
tensor = a @ b

# Register it. exact=True keeps a residual so the original can be recovered
# bit-for-bit, even though the primary encoding is lossy.
desc = space.register(
    "low_rank_matrix",
    tensor,
    decomp_type=DecompType.SVD,
    target_ratio=0.4,
    exact=True,
)

approx = space.load("low_rank_matrix", exact=False)   # cheap, low-rank
exact  = space.load("low_rank_matrix", exact=True)    # bit-exact

assert torch.equal(exact, tensor)
print(desc.address_hex)   # canonical SHA-256 address
print(desc.meta_view())   # codec choice, ratios, lineage
```

### Reversible deltas

```python
delta = space.patch(
    "low_rank_matrix",
    indices=[(0, 0), (1, 2)],
    values=torch.tensor([99.0, -1.0]),
)
modified = space.load("low_rank_matrix", exact=True)
space.revert(delta)        # rewinds the patch
restored = space.load("low_rank_matrix", exact=True)
assert torch.equal(restored, tensor)
```

---

## Repository layout

```
z_space_core.py              Core library (≈ 75 kB, single module).
requirements.txt             Pinned runtime + benchmark dependencies.
docs/
  deep_audit_findings_innovations.md
  paper_benchmark_metrics.md
  postmortem_pythia_410m_sft_benchmark.md
scripts/
  smoke_2m_model.py          Smoke test against a ~2M-parameter model.
  smoke_model_versioning.py  Sparse-delta versioning vs. torch.save.
  benchmark_dvc_pythia_410m.py
  benchmark_safetensors_zstd_baseline.py
  benchmark_sft_pythia_410m.py
  audit_pythia_delta_probe.py
tests/
  test_z_space_core.py
  test_z_space_model_2m.py
  test_z_space_model_versioning.py
```

---

## Tests

```powershell
python -m unittest
```

Tests are written against `unittest` and self-skip if `torch` is not available
in the active environment.

---

## Smoke tests and benchmarks

```powershell
.\.venv\Scripts\python.exe scripts\smoke_2m_model.py
.\.venv\Scripts\python.exe scripts\smoke_model_versioning.py
```

`smoke_model_versioning.py` builds a 2,002,058-parameter model, registers a
baseline, applies a modified version with 4,096 sparse weight changes plus a
scalar bias delta, and then compares the storage growth of Z-Space against two
plain `torch.save` checkpoints.

The `docs/` directory contains:

- `paper_benchmark_metrics.md` — the metrics used in the in-progress write-up.
- `postmortem_pythia_410m_sft_benchmark.md` — analysis of the Pythia-410M SFT
  run and the gaps it exposed.
- `deep_audit_findings_innovations.md` — internal audit notes on the codec
  tournament and addressing scheme.

---

## Status

Z-Space Core is **active research code**. Public API surface is intentionally
small, but signatures may still change between versions. Benchmarks beyond the
2M-parameter smoke test should be considered preliminary.

---

## Authorship and contact

Z-Space Core is the work of **Raul Cruz Acosta**. See
[AUTHORSHIP.md](AUTHORSHIP.md) for the full authorship statement, including the
list of technical contributions claimed and the channels for licensing
inquiries.

For any use beyond personal reading and academic citation, written
authorization is required — see [LICENSE](LICENSE).
