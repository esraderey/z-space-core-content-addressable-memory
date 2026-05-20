"""
Z-Space Core: content-addressable tensor memory with deterministic synthesis.

Key properties implemented here:
- Canonical descriptor addressing with SHA-256 namespaces.
- Content-addressed Merkle-style node store.
- Codec tournament: RAW, SPARSE, SVD, and optional TensorLy strategies.
- Hybrid lossy + exact mode using bitwise XOR residuals.
- Progressive loads: approximate first, exact when requested.
- Reversible delta chains with parent lineage.
- Mutable-cache protection by returning cloned tensors.

The tensor backend is PyTorch. Compression uses LZ4 when installed and falls
back to zlib without changing public APIs.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
import sys
import zlib
from collections import Counter, OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import mscs

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a project dependency
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised only on machines without torch
    torch = None  # type: ignore[assignment]

try:
    import lz4.frame as lz4_frame
except ImportError:  # pragma: no cover - optional dependency
    lz4_frame = None

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional for non-checkpoint workflows
    zstd = None

try:
    import tensorly as tl
    from tensorly.decomposition import matrix_product_state, parafac, tucker

    if torch is not None:
        tl.set_backend("pytorch")
except ImportError:  # pragma: no cover - optional dependency
    tl = None
    matrix_product_state = None
    parafac = None
    tucker = None


class DecompType(Enum):
    TT = "tt"
    CP = "cp"
    TUCKER = "tucker"
    SVD = "svd"
    RAW = "raw"
    SPARSE = "sparse"


class LoadMode(Enum):
    APPROX = "approx"
    EXACT = "exact"


class NodeKind(Enum):
    RAW_TENSOR = "raw_tensor"
    COMPONENTS = "components"
    XOR_RESIDUAL = "xor_residual"
    DELTA_CHAIN = "delta_chain"
    TENSOR_PAYLOAD = "tensor_payload"
    TENSOR_BLOCK = "tensor_block"
    TENSOR_DELTA = "tensor_delta"


DEFAULT_TENSOR_BLOCK_SIZE = 64 * 1024
DEFAULT_DELTA_CHUNK_SIZE = 16 * 1024 * 1024
_DELTA_PRECONDITIONERS = {"xor", "u16_sub"}
_TENSOR_BLOCK_MANIFEST_HEADER = b"ZTB1"
_TENSOR_BLOCK_MANIFEST_FORMAT = "tensor_block_manifest_v1"


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "Z-Space requires PyTorch for tensor operations. Install torch to "
            "use register/load/update."
        )
    return torch


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {"__bytes_b64__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in sorted(value.items())}
    if torch is not None and isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    return value


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _stable_json_text(value: Any) -> str:
    return _stable_json_bytes(value).decode("utf-8")


def _freeze_meta(meta: Optional[Mapping[str, Any] | Sequence[Tuple[str, Any]]]) -> Tuple[Tuple[str, str], ...]:
    if not meta:
        return ()
    items = meta.items() if isinstance(meta, Mapping) else meta
    return tuple(sorted((str(k), _stable_json_text(v)) for k, v in items))


def _digest(namespace: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(namespace + b"\0" + payload).digest()


def _hex(data: Optional[bytes]) -> Optional[str]:
    return None if data is None else data.hex()


def _pack_field(payload: bytes) -> bytes:
    return struct.pack("<Q", len(payload)) + payload


def _unpack_field(payload: bytes, offset: int = 0) -> Tuple[bytes, int]:
    if len(payload) < offset + 8:
        raise ValueError("Malformed packed field: missing length prefix")
    size = struct.unpack("<Q", payload[offset : offset + 8])[0]
    start = offset + 8
    end = start + size
    if len(payload) < end:
        raise ValueError("Malformed packed field: declared payload is truncated")
    return payload[start:end], end


def _xor_bytes(left: Any, right: Any) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR residual requires equal byte lengths")
    if np is not None:
        left_view = np.frombuffer(left, dtype=np.uint8)
        right_view = np.frombuffer(right, dtype=np.uint8)
        return np.bitwise_xor(left_view, right_view).tobytes()
    return bytes(a ^ b for a, b in zip(left, right))


def _require_zstd() -> Any:
    if zstd is None:
        raise RuntimeError("zstandard is required for dense XOR checkpoint deltas")
    return zstd


@dataclass(frozen=True)
class ZDescriptor:
    """Immutable descriptor for synthesizable content.

    The descriptor itself is small. Heavy payloads live in the content store and
    are referenced by node digests.
    """

    kind: str
    decomp_type: DecompType
    shape: Tuple[int, ...]
    ranks: Optional[Tuple[int, ...]] = None
    exact: bool = True
    version: int = 0
    raw_node: Optional[bytes] = None
    component_node: Optional[bytes] = None
    residual_node: Optional[bytes] = None
    delta_chain_node: Optional[bytes] = None
    parent_addr: Optional[bytes] = None
    meta: Mapping[str, Any] | Sequence[Tuple[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(
            self,
            "ranks",
            None if self.ranks is None else tuple(int(r) for r in self.ranks),
        )
        object.__setattr__(self, "meta", _freeze_meta(self.meta))

    def canonical_bytes(self) -> bytes:
        payload = {
            "kind": self.kind,
            "decomp_type": self.decomp_type.value,
            "shape": list(self.shape),
            "ranks": None if self.ranks is None else list(self.ranks),
            "exact": self.exact,
            "version": self.version,
            "raw_node": _hex(self.raw_node),
            "component_node": _hex(self.component_node),
            "residual_node": _hex(self.residual_node),
            "delta_chain_node": _hex(self.delta_chain_node),
            "parent_addr": _hex(self.parent_addr),
            "meta": list(self.meta),
        }
        return _stable_json_bytes(payload)

    @property
    def address(self) -> bytes:
        return _digest(b"zspace:descriptor:v1", self.canonical_bytes())

    @property
    def address_hex(self) -> str:
        return self.address.hex()

    def meta_view(self) -> Dict[str, Any]:
        return {k: json.loads(v) for k, v in self.meta}


@dataclass(frozen=True)
class CheckpointDescriptor:
    """Immutable descriptor for a model/state-dict checkpoint."""

    kind: str = "checkpoint"
    version: int = 0
    tensor_descriptors: Mapping[str, ZDescriptor] | Sequence[Tuple[str, ZDescriptor]] = field(default_factory=dict)
    tensor_delta_nodes: Mapping[str, bytes] | Sequence[Tuple[str, bytes]] = field(default_factory=dict)
    parent_addr: Optional[bytes] = None
    meta: Mapping[str, Any] | Sequence[Tuple[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tensor_items = (
            self.tensor_descriptors.items()
            if isinstance(self.tensor_descriptors, Mapping)
            else self.tensor_descriptors
        )
        delta_items = (
            self.tensor_delta_nodes.items()
            if isinstance(self.tensor_delta_nodes, Mapping)
            else self.tensor_delta_nodes
        )
        object.__setattr__(self, "tensor_descriptors", tuple(sorted((str(k), v) for k, v in tensor_items)))
        object.__setattr__(self, "tensor_delta_nodes", tuple(sorted((str(k), v) for k, v in delta_items)))
        object.__setattr__(self, "meta", _freeze_meta(self.meta))

    def canonical_bytes(self) -> bytes:
        payload = {
            "kind": self.kind,
            "version": self.version,
            "tensor_descriptors": [(key, desc.address.hex()) for key, desc in self.tensor_descriptors],
            "tensor_delta_nodes": [(key, node.hex()) for key, node in self.tensor_delta_nodes],
            "parent_addr": _hex(self.parent_addr),
            "meta": list(self.meta),
        }
        return _stable_json_bytes(payload)

    @property
    def address(self) -> bytes:
        return _digest(b"zspace:checkpoint:v1", self.canonical_bytes())

    @property
    def address_hex(self) -> str:
        return self.address.hex()

    def meta_view(self) -> Dict[str, Any]:
        return {k: json.loads(v) for k, v in self.meta}

    def tensor_map(self) -> Dict[str, ZDescriptor]:
        return dict(self.tensor_descriptors)

    def delta_map(self) -> Dict[str, bytes]:
        return dict(self.tensor_delta_nodes)


class ReversibleCompressor:
    """Reversible compressor with an explicit codec header."""

    LZ4_HEADER = b"ZC1L"
    ZLIB_HEADER = b"ZC1Z"

    @staticmethod
    def compress(data: bytes) -> bytes:
        if lz4_frame is not None:
            return ReversibleCompressor.LZ4_HEADER + lz4_frame.compress(
                data,
                compression_level=lz4_frame.COMPRESSIONLEVEL_MAX,
                content_checksum=True,
            )
        return ReversibleCompressor.ZLIB_HEADER + zlib.compress(data, level=9)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        if data.startswith(ReversibleCompressor.LZ4_HEADER):
            if lz4_frame is None:
                raise RuntimeError("This node was compressed with LZ4, but lz4 is not installed")
            return lz4_frame.decompress(data[len(ReversibleCompressor.LZ4_HEADER) :])
        if data.startswith(ReversibleCompressor.ZLIB_HEADER):
            return zlib.decompress(data[len(ReversibleCompressor.ZLIB_HEADER) :])
        raise ValueError("Unknown compression header")

    @staticmethod
    def compressed_size(data: bytes) -> int:
        return len(ReversibleCompressor.compress(data))


class ContentStore:
    """In-memory content-addressed node store."""

    def __init__(self) -> None:
        self._nodes: Dict[bytes, bytes] = {}
        self._kinds: Dict[bytes, NodeKind] = {}
        self._lock = RLock()

    @staticmethod
    def _node_digest(payload: bytes, kind: NodeKind) -> bytes:
        return _digest(b"zspace:node:v1:" + kind.value.encode("ascii"), payload)

    @staticmethod
    def _is_tensor_payload_kind(kind: NodeKind) -> bool:
        return kind in (NodeKind.RAW_TENSOR, NodeKind.XOR_RESIDUAL, NodeKind.TENSOR_PAYLOAD)

    def put(self, payload: bytes, kind: NodeKind) -> bytes:
        digest = self._node_digest(payload, kind)
        packed = ReversibleCompressor.compress(payload)
        with self._lock:
            self._nodes.setdefault(digest, packed)
            self._kinds.setdefault(digest, kind)
        return digest

    def put_tensor_payload(
        self,
        payload: bytes,
        kind: NodeKind,
        *,
        block_size: int = DEFAULT_TENSOR_BLOCK_SIZE,
    ) -> bytes:
        """Store a TensorCodec payload with content-addressed raw-byte blocks."""

        if not self._is_tensor_payload_kind(kind):
            raise ValueError(f"{kind.value} cannot store tensor block manifests")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        info, raw = TensorCodec.split_payload(payload)
        if len(raw) <= block_size:
            return self.put(payload, kind)

        blocks = []
        for offset in range(0, len(raw), block_size):
            block = raw[offset : offset + block_size]
            block_digest = self.put(block, NodeKind.TENSOR_BLOCK)
            blocks.append({"digest": block_digest.hex(), "nbytes": len(block)})

        manifest = {
            "format": _TENSOR_BLOCK_MANIFEST_FORMAT,
            "tensor_info": info,
            "block_size": int(block_size),
            "raw_nbytes": len(raw),
            "blocks": blocks,
        }
        return self.put(_TENSOR_BLOCK_MANIFEST_HEADER + _stable_json_bytes(manifest), kind)

    def _read_stored(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> Tuple[NodeKind, bytes]:
        with self._lock:
            if digest not in self._nodes:
                raise KeyError(f"Unknown content node: {digest.hex()}")
            kind = self._kinds[digest]
            packed = self._nodes[digest]
        if expected_kind is not None and kind != expected_kind:
            raise TypeError(f"Node {digest.hex()} is {kind.value}, expected {expected_kind.value}")
        return kind, ReversibleCompressor.decompress(packed)

    @staticmethod
    def _decode_tensor_block_manifest(payload: bytes) -> Optional[Dict[str, Any]]:
        if not payload.startswith(_TENSOR_BLOCK_MANIFEST_HEADER):
            return None
        manifest = json.loads(payload[len(_TENSOR_BLOCK_MANIFEST_HEADER) :].decode("utf-8"))
        if manifest.get("format") != _TENSOR_BLOCK_MANIFEST_FORMAT:
            raise ValueError("Unknown tensor block manifest format")
        return manifest

    def _materialize_tensor_payload(self, manifest: Mapping[str, Any]) -> bytes:
        info = dict(manifest["tensor_info"])
        raw_parts = []
        for block in manifest["blocks"]:
            block_digest = bytes.fromhex(block["digest"])
            block_payload = self.get(block_digest, NodeKind.TENSOR_BLOCK)
            expected_nbytes = int(block["nbytes"])
            if len(block_payload) != expected_nbytes:
                raise ValueError("Tensor block byte count does not match manifest")
            raw_parts.append(block_payload)
        raw = b"".join(raw_parts)
        if len(raw) != int(manifest["raw_nbytes"]):
            raise ValueError("Tensor block manifest raw byte count does not match blocks")
        return TensorCodec.pack_raw_parts(info, raw)

    def get(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> bytes:
        kind, payload = self._read_stored(digest, expected_kind)
        if self._is_tensor_payload_kind(kind):
            manifest = self._decode_tensor_block_manifest(payload)
            if manifest is not None:
                return self._materialize_tensor_payload(manifest)
        return payload

    def has(self, digest: bytes) -> bool:
        with self._lock:
            return digest in self._nodes

    def compressed_size(self, digest: bytes) -> int:
        with self._lock:
            return len(self._nodes[digest])

    def tree_compressed_size(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> int:
        kind, payload = self._read_stored(digest, expected_kind)
        total = self.compressed_size(digest)
        if not self._is_tensor_payload_kind(kind):
            return total
        manifest = self._decode_tensor_block_manifest(payload)
        if manifest is None:
            return total
        seen = set()
        for block in manifest["blocks"]:
            block_digest = bytes.fromhex(block["digest"])
            if block_digest in seen:
                continue
            seen.add(block_digest)
            total += self.compressed_size(block_digest)
        return total

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            node_kinds = Counter(kind.value for kind in self._kinds.values())
            return {
                "nodes": len(self._nodes),
                "compressed_bytes": sum(len(v) for v in self._nodes.values()),
                "node_kinds": dict(node_kinds),
            }


class PackfileContentStore(ContentStore):
    """Append-only packfile-backed node store.

    This store keeps the digest index in memory, but writes node bytes to one
    append-only file instead of creating one filesystem entry per node. It is
    intended for long-running benchmark/checkpoint sessions where small-file
    fragmentation dominates commit time.
    """

    def __init__(self, root: str | Path, pack_name: str = "nodes.zspack") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.pack_path = self.root / pack_name
        self._index: Dict[bytes, Tuple[int, int]] = {}
        self._kinds: Dict[bytes, NodeKind] = {}
        self._sizes: Dict[bytes, int] = {}
        self._lock = RLock()
        self._pack = self.pack_path.open("a+b")

    def put(self, payload: bytes, kind: NodeKind) -> bytes:
        digest = self._node_digest(payload, kind)
        with self._lock:
            if digest not in self._index:
                packed = ReversibleCompressor.compress(payload)
                self._pack.seek(0, os.SEEK_END)
                offset = self._pack.tell()
                self._pack.write(packed)
                self._index[digest] = (offset, len(packed))
                self._sizes[digest] = len(packed)
            self._kinds.setdefault(digest, kind)
        return digest

    def _read_stored(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> Tuple[NodeKind, bytes]:
        with self._lock:
            if digest not in self._index:
                raise KeyError(f"Unknown content node: {digest.hex()}")
            kind = self._kinds[digest]
            offset, size = self._index[digest]
            self._pack.flush()
            self._pack.seek(offset)
            packed = self._pack.read(size)
        if expected_kind is not None and kind != expected_kind:
            raise TypeError(f"Node {digest.hex()} is {kind.value}, expected {expected_kind.value}")
        return kind, ReversibleCompressor.decompress(packed)

    def get(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> bytes:
        kind, payload = self._read_stored(digest, expected_kind)
        if self._is_tensor_payload_kind(kind):
            manifest = self._decode_tensor_block_manifest(payload)
            if manifest is not None:
                return self._materialize_tensor_payload(manifest)
        return payload

    def has(self, digest: bytes) -> bool:
        with self._lock:
            return digest in self._index

    def compressed_size(self, digest: bytes) -> int:
        with self._lock:
            return self._sizes[digest]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            node_kinds = Counter(kind.value for kind in self._kinds.values())
            return {
                "nodes": len(self._index),
                "compressed_bytes": sum(self._sizes.values()),
                "node_kinds": dict(node_kinds),
                "packfile": str(self.pack_path),
            }

    def close(self) -> None:
        with self._lock:
            self._pack.close()


class TensorCodec:
    """Canonical tensor byte codec used for exact storage and verification."""

    @staticmethod
    def dtype_name(dtype: Any) -> str:
        return str(dtype).replace("torch.", "")

    @staticmethod
    def dtype_from_name(name: str) -> Any:
        t = _require_torch()
        if not hasattr(t, name):
            raise ValueError(f"Unsupported torch dtype: {name}")
        return getattr(t, name)

    @staticmethod
    def raw_parts(tensor: Any) -> Tuple[Dict[str, Any], bytes]:
        t = _require_torch()
        if not isinstance(tensor, t.Tensor):
            raise TypeError("Expected a torch.Tensor")
        cpu = tensor.detach().cpu().contiguous()
        flat = cpu.reshape(-1).contiguous()
        byte_view = flat.view(t.uint8)
        raw = byte_view.numpy().tobytes()
        info = {
            "format": "torch_raw_bytes_v1",
            "byteorder": sys.byteorder,
            "dtype": TensorCodec.dtype_name(cpu.dtype),
            "shape": list(cpu.shape),
            "numel": int(cpu.numel()),
            "nbytes": len(raw),
        }
        return info, raw

    @staticmethod
    def pack_tensor(tensor: Any) -> bytes:
        info, raw = TensorCodec.raw_parts(tensor)
        return _pack_field(_stable_json_bytes(info)) + raw

    @staticmethod
    def split_payload(payload: bytes) -> Tuple[Dict[str, Any], bytes]:
        info_bytes, offset = _unpack_field(payload, 0)
        info = json.loads(info_bytes.decode("utf-8"))
        raw = payload[offset:]
        if len(raw) != info["nbytes"]:
            raise ValueError("Tensor payload byte count does not match metadata")
        return info, raw

    @staticmethod
    def pack_raw_parts(info: Mapping[str, Any], raw: bytes) -> bytes:
        checked = dict(info)
        checked["nbytes"] = len(raw)
        return _pack_field(_stable_json_bytes(checked)) + raw

    @staticmethod
    def unpack_tensor(payload: bytes) -> Any:
        t = _require_torch()
        info, raw = TensorCodec.split_payload(payload)
        dtype = TensorCodec.dtype_from_name(info["dtype"])
        shape = tuple(int(s) for s in info["shape"])
        expected_numel = int(info["numel"])
        tensor = t.frombuffer(bytearray(raw), dtype=dtype).clone()
        if int(tensor.numel()) != expected_numel:
            raise ValueError("Tensor payload element count does not match metadata")
        return tensor.reshape(shape)

    @staticmethod
    def tensor_digest(tensor: Any) -> str:
        return _digest(b"zspace:tensor:v1", TensorCodec.pack_tensor(tensor)).hex()

    @staticmethod
    def pack_xor_residual(original: Any, approximate: Any) -> bytes:
        t = _require_torch()
        original_info, original_raw = TensorCodec.raw_parts(original)
        approx_cast = approximate.detach().to(dtype=original.dtype).reshape(original.shape).cpu().contiguous()
        approx_info, approx_raw = TensorCodec.raw_parts(approx_cast)
        if original_info["shape"] != approx_info["shape"] or original_info["dtype"] != approx_info["dtype"]:
            raise ValueError("Approximation cannot be aligned to original tensor")
        residual_info = dict(original_info)
        residual_info["format"] = "xor_residual_v1"
        residual_info["base_dtype"] = TensorCodec.dtype_name(approx_cast.dtype)
        residual = _xor_bytes(original_raw, approx_raw)
        if original.numel() == 0:
            residual = b""
        return TensorCodec.pack_raw_parts(residual_info, residual)

    @staticmethod
    def apply_xor_residual(approximate: Any, residual_payload: bytes) -> Any:
        residual_info, residual = TensorCodec.split_payload(residual_payload)
        dtype = TensorCodec.dtype_from_name(residual_info["dtype"])
        shape = tuple(int(s) for s in residual_info["shape"])
        approx = approximate.detach().to(dtype=dtype).reshape(shape).cpu().contiguous()
        _, approx_raw = TensorCodec.raw_parts(approx)
        exact_raw = _xor_bytes(approx_raw, residual)
        exact_payload = TensorCodec.pack_raw_parts(
            {
                "format": "torch_raw_bytes_v1",
                "byteorder": residual_info["byteorder"],
                "dtype": residual_info["dtype"],
                "shape": residual_info["shape"],
                "numel": residual_info["numel"],
            },
            exact_raw,
        )
        return TensorCodec.unpack_tensor(exact_payload)


class TensorDeltaCodec:
    """Lossless dense tensor deltas."""

    @staticmethod
    def _normalize_preconditioner(preconditioner: str) -> str:
        normalized = str(preconditioner).replace("-", "_")
        if normalized not in _DELTA_PRECONDITIONERS:
            supported = ", ".join(sorted(_DELTA_PRECONDITIONERS))
            raise ValueError(f"Unsupported delta preconditioner: {preconditioner}. Supported: {supported}")
        return normalized

    @staticmethod
    def _validate_chunk_size(chunk_size: int, *, preconditioner: str = "xor") -> int:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if preconditioner == "u16_sub" and chunk_size % 2:
            raise ValueError("u16_sub delta preconditioner requires an even chunk_size")
        return chunk_size

    @staticmethod
    def _zstd_dict_digest(zstd_dict: Optional[bytes]) -> Optional[str]:
        if zstd_dict is None:
            return None
        return _digest(b"zspace:zstd-dict:v1", bytes(zstd_dict)).hex()

    @staticmethod
    def _zstd_dict_data(zstd_dict: Optional[bytes]) -> Any:
        if zstd_dict is None:
            return None
        return _require_zstd().ZstdCompressionDict(bytes(zstd_dict))

    @staticmethod
    def _u16_dtype(byteorder: str) -> Any:
        if np is None:
            return None
        if byteorder == "big":
            return np.dtype(">u2")
        return np.dtype("<u2")

    @staticmethod
    def _u16_subtract(parent: Any, current: Any, *, byteorder: str) -> bytes:
        if len(parent) != len(current):
            raise ValueError("u16_sub delta requires equal byte lengths")
        if len(parent) % 2:
            raise ValueError("u16_sub delta requires an even byte count")
        if np is not None:
            dtype = TensorDeltaCodec._u16_dtype(byteorder)
            parent_words = np.frombuffer(parent, dtype=dtype)
            current_words = np.frombuffer(current, dtype=dtype)
            return (current_words - parent_words).astype(dtype, copy=False).tobytes()
        out = bytearray(len(parent))
        for offset in range(0, len(parent), 2):
            parent_word = int.from_bytes(parent[offset : offset + 2], byteorder=byteorder, signed=False)
            current_word = int.from_bytes(current[offset : offset + 2], byteorder=byteorder, signed=False)
            out[offset : offset + 2] = ((current_word - parent_word) & 0xFFFF).to_bytes(2, byteorder=byteorder)
        return bytes(out)

    @staticmethod
    def _u16_add(parent: Any, delta: Any, *, byteorder: str) -> bytes:
        if len(parent) != len(delta):
            raise ValueError("u16_sub delta requires equal byte lengths")
        if len(parent) % 2:
            raise ValueError("u16_sub delta requires an even byte count")
        if np is not None:
            dtype = TensorDeltaCodec._u16_dtype(byteorder)
            parent_words = np.frombuffer(parent, dtype=dtype)
            delta_words = np.frombuffer(delta, dtype=dtype)
            return (parent_words + delta_words).astype(dtype, copy=False).tobytes()
        out = bytearray(len(parent))
        for offset in range(0, len(parent), 2):
            parent_word = int.from_bytes(parent[offset : offset + 2], byteorder=byteorder, signed=False)
            delta_word = int.from_bytes(delta[offset : offset + 2], byteorder=byteorder, signed=False)
            out[offset : offset + 2] = ((parent_word + delta_word) & 0xFFFF).to_bytes(2, byteorder=byteorder)
        return bytes(out)

    @staticmethod
    def _precondition_chunk(parent: Any, current: Any, *, preconditioner: str, byteorder: str) -> bytes:
        if preconditioner == "xor":
            return _xor_bytes(parent, current)
        if preconditioner == "u16_sub":
            return TensorDeltaCodec._u16_subtract(parent, current, byteorder=byteorder)
        raise ValueError(f"Unsupported delta preconditioner: {preconditioner}")

    @staticmethod
    def _restore_chunk(parent: Any, delta: Any, *, preconditioner: str, byteorder: str) -> bytes:
        if preconditioner == "xor":
            return _xor_bytes(parent, delta)
        if preconditioner == "u16_sub":
            return TensorDeltaCodec._u16_add(parent, delta, byteorder=byteorder)
        raise ValueError(f"Unsupported delta preconditioner: {preconditioner}")

    @staticmethod
    def _compress_xor_zstd_stream(
        parent_raw: bytes,
        current_raw: bytes,
        *,
        level: int,
        chunk_size: int,
        preconditioner: str,
        byteorder: str,
        zstd_dict: Optional[bytes],
    ) -> bytes:
        zstd_mod = _require_zstd()
        if len(parent_raw) != len(current_raw):
            raise ValueError("XOR zstd delta requires equal byte lengths")
        preconditioner = TensorDeltaCodec._normalize_preconditioner(preconditioner)
        chunk_size = TensorDeltaCodec._validate_chunk_size(chunk_size, preconditioner=preconditioner)
        parent_view = memoryview(parent_raw)
        current_view = memoryview(current_raw)
        dict_data = TensorDeltaCodec._zstd_dict_data(zstd_dict)
        compressor = zstd_mod.ZstdCompressor(level=level, dict_data=dict_data).compressobj()
        parts = []
        for offset in range(0, len(current_raw), chunk_size):
            end = min(offset + chunk_size, len(current_raw))
            delta = TensorDeltaCodec._precondition_chunk(
                parent_view[offset:end],
                current_view[offset:end],
                preconditioner=preconditioner,
                byteorder=byteorder,
            )
            chunk = compressor.compress(delta)
            if chunk:
                parts.append(chunk)
        tail = compressor.flush()
        if tail:
            parts.append(tail)
        return b"".join(parts)

    @staticmethod
    def _decompress_xor_zstd_stream(
        parent_raw: bytes,
        compressed: bytes,
        *,
        expected_nbytes: int,
        chunk_size: int,
        preconditioner: str,
        byteorder: str,
        zstd_dict: Optional[bytes],
    ) -> bytes:
        zstd_mod = _require_zstd()
        preconditioner = TensorDeltaCodec._normalize_preconditioner(preconditioner)
        chunk_size = TensorDeltaCodec._validate_chunk_size(chunk_size, preconditioner=preconditioner)
        if preconditioner == "u16_sub" and expected_nbytes % 2:
            raise ValueError("u16_sub delta requires an even byte count")
        parent_view = memoryview(parent_raw)
        current_raw = bytearray(expected_nbytes)
        offset = 0
        dict_data = TensorDeltaCodec._zstd_dict_data(zstd_dict)
        reader = zstd_mod.ZstdDecompressor(dict_data=dict_data).stream_reader(io.BytesIO(compressed))
        try:
            while offset < expected_nbytes:
                wanted = min(chunk_size, expected_nbytes - offset)
                delta = reader.read(wanted)
                if preconditioner == "u16_sub" and delta and len(delta) % 2:
                    delta += reader.read(1)
                if not delta:
                    break
                if preconditioner == "u16_sub" and len(delta) % 2:
                    raise ValueError("u16_sub delta stream is not aligned to 16-bit words")
                end = offset + len(delta)
                current_raw[offset:end] = TensorDeltaCodec._restore_chunk(
                    parent_view[offset:end],
                    delta,
                    preconditioner=preconditioner,
                    byteorder=byteorder,
                )
                offset = end
            if reader.read(1):
                raise ValueError("XOR zstd delta expands beyond metadata byte count")
        finally:
            reader.close()
        if offset != expected_nbytes:
            raise ValueError("XOR zstd delta byte count does not match metadata")
        return bytes(current_raw)

    @staticmethod
    def pack_xor_zstd(
        parent: Any,
        current: Any,
        *,
        level: int = 3,
        chunk_size: int = DEFAULT_DELTA_CHUNK_SIZE,
        preconditioner: str = "xor",
        zstd_dict: Optional[bytes] = None,
    ) -> bytes:
        preconditioner = TensorDeltaCodec._normalize_preconditioner(preconditioner)
        parent_info, parent_raw = TensorCodec.raw_parts(parent)
        current_info, current_raw = TensorCodec.raw_parts(current)
        if parent_info["shape"] != current_info["shape"] or parent_info["dtype"] != current_info["dtype"]:
            raise ValueError("XOR zstd delta requires matching tensor shape and dtype")
        chunk_size = TensorDeltaCodec._validate_chunk_size(chunk_size, preconditioner=preconditioner)
        compressed = TensorDeltaCodec._compress_xor_zstd_stream(
            parent_raw,
            current_raw,
            level=level,
            chunk_size=chunk_size,
            preconditioner=preconditioner,
            byteorder=current_info["byteorder"],
            zstd_dict=zstd_dict,
        )
        payload = {
            "type": "xor_zstd_v1" if preconditioner == "xor" else "delta_zstd_v1",
            "level": int(level),
            "chunk_size": int(chunk_size),
            "preconditioner": preconditioner,
            "zstd_dict_digest": TensorDeltaCodec._zstd_dict_digest(zstd_dict),
            "tensor_info": current_info,
            "raw_nbytes": len(current_raw),
            "parent_digest": _digest(b"zspace:tensor:v1", TensorCodec.pack_raw_parts(parent_info, parent_raw)).hex(),
            "current_digest": _digest(b"zspace:tensor:v1", TensorCodec.pack_raw_parts(current_info, current_raw)).hex(),
            "zstd": compressed,
        }
        return mscs.dumps(payload)

    @staticmethod
    def apply_xor_zstd(
        parent: Any,
        delta_payload: bytes,
        *,
        verify: bool = True,
        zstd_dict: Optional[bytes] = None,
    ) -> Any:
        payload = mscs.loads(delta_payload)
        if payload.get("type") not in ("xor_zstd_v1", "delta_zstd_v1"):
            raise ValueError(f"Unsupported tensor delta payload: {payload.get('type')}")
        info = dict(payload["tensor_info"])
        preconditioner = TensorDeltaCodec._normalize_preconditioner(payload.get("preconditioner", "xor"))
        expected_dict_digest = payload.get("zstd_dict_digest")
        if expected_dict_digest is not None:
            if TensorDeltaCodec._zstd_dict_digest(zstd_dict) != expected_dict_digest:
                raise ValueError("zstd dictionary does not match tensor delta metadata")
        parent_info, parent_raw = TensorCodec.raw_parts(parent)
        if parent_info["shape"] != info["shape"] or parent_info["dtype"] != info["dtype"]:
            raise ValueError("Parent tensor does not match XOR zstd delta metadata")
        expected_nbytes = int(payload["raw_nbytes"])
        current_raw = TensorDeltaCodec._decompress_xor_zstd_stream(
            parent_raw,
            payload["zstd"],
            expected_nbytes=expected_nbytes,
            chunk_size=int(payload.get("chunk_size", DEFAULT_DELTA_CHUNK_SIZE)),
            preconditioner=preconditioner,
            byteorder=info["byteorder"],
            zstd_dict=zstd_dict,
        )
        current = TensorCodec.unpack_tensor(TensorCodec.pack_raw_parts(info, current_raw))
        if verify:
            expected = payload.get("current_digest")
            if expected is not None and TensorCodec.tensor_digest(current) != expected:
                raise ValueError("XOR zstd delta reconstruction failed digest verification")
        return current


class TensorDecomposer:
    """Tensor decomposition and reconstruction strategies."""

    @staticmethod
    def _ensure_float_matrix(tensor: Any) -> bool:
        t = _require_torch()
        return (
            isinstance(tensor, t.Tensor)
            and tensor.ndim == 2
            and tensor.dtype in (t.float32, t.float64)
            and min(tensor.shape) > 1
        )

    @staticmethod
    def svd_rank(shape: Sequence[int], target_ratio: float) -> int:
        if len(shape) != 2:
            raise ValueError("SVD rank requires a matrix shape")
        max_rank = min(int(shape[0]), int(shape[1]))
        return max(1, min(max_rank, int(max_rank * target_ratio)))

    @staticmethod
    def decompose(tensor: Any, decomp_type: DecompType, **params: Any) -> bytes:
        t = _require_torch()
        if decomp_type == DecompType.RAW:
            return TensorCodec.pack_tensor(tensor)

        if decomp_type == DecompType.SPARSE:
            indices = t.nonzero(tensor, as_tuple=False).to(dtype=t.int64).cpu().contiguous()
            if indices.numel() == 0:
                values = tensor.reshape(-1)[:0].detach().cpu().contiguous()
            else:
                values = tensor[tuple(indices.T)].detach().cpu().contiguous()
            payload = {
                "type": "sparse_v1",
                "shape": list(tensor.shape),
                "dtype": TensorCodec.dtype_name(tensor.dtype),
                "indices": TensorCodec.pack_tensor(indices),
                "values": TensorCodec.pack_tensor(values),
            }
            return mscs.dumps(payload)

        if decomp_type == DecompType.SVD:
            if not TensorDecomposer._ensure_float_matrix(tensor):
                raise ValueError("SVD strategy requires a float32/float64 matrix")
            rank = int(params.get("rank", TensorDecomposer.svd_rank(tensor.shape, 0.1)))
            rank = max(1, min(rank, min(tensor.shape)))
            u, s, vh = t.linalg.svd(tensor.detach().cpu(), full_matrices=False)
            payload = {
                "type": "svd_v1",
                "shape": list(tensor.shape),
                "rank": rank,
                "dtype": TensorCodec.dtype_name(tensor.dtype),
                "u": TensorCodec.pack_tensor(u[:, :rank].contiguous()),
                "s": TensorCodec.pack_tensor(s[:rank].contiguous()),
                "vh": TensorCodec.pack_tensor(vh[:rank, :].contiguous()),
            }
            return mscs.dumps(payload)

        if decomp_type in (DecompType.TT, DecompType.CP, DecompType.TUCKER):
            if tl is None:
                raise RuntimeError("TensorLy is required for TT/CP/TUCKER decomposition")
            return TensorDecomposer._decompose_tensorly(tensor, decomp_type, **params)

        raise ValueError(f"Unsupported decomposition type: {decomp_type}")

    @staticmethod
    def _decompose_tensorly(tensor: Any, decomp_type: DecompType, **params: Any) -> bytes:
        if decomp_type == DecompType.TT:
            ranks = tuple(int(r) for r in params["ranks"])
            factors = matrix_product_state(tensor.detach().cpu(), rank=ranks)
            payload = {
                "type": "tt_v1",
                "shape": list(tensor.shape),
                "ranks": list(ranks),
                "dtype": TensorCodec.dtype_name(tensor.dtype),
                "factors": [TensorCodec.pack_tensor(f.contiguous()) for f in factors],
            }
            return mscs.dumps(payload)

        if decomp_type == DecompType.CP:
            rank = int(params.get("rank", 8))
            weights, factors = parafac(tensor.detach().cpu(), rank=rank, init="random", n_iter_max=100, tol=1e-6)
            payload = {
                "type": "cp_v1",
                "shape": list(tensor.shape),
                "rank": rank,
                "dtype": TensorCodec.dtype_name(tensor.dtype),
                "weights": TensorCodec.pack_tensor(weights.contiguous()),
                "factors": [TensorCodec.pack_tensor(f.contiguous()) for f in factors],
            }
            return mscs.dumps(payload)

        ranks = tuple(int(r) for r in params.get("ranks", [min(int(s), 8) for s in tensor.shape]))
        core, factors = tucker(tensor.detach().cpu(), rank=ranks)
        payload = {
            "type": "tucker_v1",
            "shape": list(tensor.shape),
            "ranks": list(ranks),
            "dtype": TensorCodec.dtype_name(tensor.dtype),
            "core": TensorCodec.pack_tensor(core.contiguous()),
            "factors": [TensorCodec.pack_tensor(f.contiguous()) for f in factors],
        }
        return mscs.dumps(payload)

    @staticmethod
    def reconstruct(component_payload: bytes) -> Any:
        t = _require_torch()
        payload = mscs.loads(component_payload)
        comp_type = payload["type"]

        if comp_type == "sparse_v1":
            shape = tuple(int(s) for s in payload["shape"])
            dtype = TensorCodec.dtype_from_name(payload["dtype"])
            indices = TensorCodec.unpack_tensor(payload["indices"]).to(dtype=t.int64)
            values = TensorCodec.unpack_tensor(payload["values"]).to(dtype=dtype)
            out = t.zeros(shape, dtype=dtype)
            if indices.numel() > 0:
                out[tuple(indices.T)] = values
            return out

        if comp_type == "svd_v1":
            u = TensorCodec.unpack_tensor(payload["u"])
            s = TensorCodec.unpack_tensor(payload["s"])
            vh = TensorCodec.unpack_tensor(payload["vh"])
            return (u * s.unsqueeze(0)) @ vh

        if comp_type == "tt_v1":
            if tl is None:
                raise RuntimeError("TensorLy is required to reconstruct TT components")
            factors = [TensorCodec.unpack_tensor(f) for f in payload["factors"]]
            return tl.tt_to_tensor(factors)

        if comp_type == "cp_v1":
            if tl is None:
                raise RuntimeError("TensorLy is required to reconstruct CP components")
            weights = TensorCodec.unpack_tensor(payload["weights"])
            factors = [TensorCodec.unpack_tensor(f) for f in payload["factors"]]
            return tl.cp_to_tensor((weights, factors))

        if comp_type == "tucker_v1":
            if tl is None:
                raise RuntimeError("TensorLy is required to reconstruct Tucker components")
            core = TensorCodec.unpack_tensor(payload["core"])
            factors = [TensorCodec.unpack_tensor(f) for f in payload["factors"]]
            return tl.tucker_to_tensor((core, factors))

        raise ValueError(f"Unsupported component payload: {comp_type}")


@dataclass
class _Candidate:
    decomp_type: DecompType
    exact: bool
    ranks: Optional[Tuple[int, ...]]
    raw_payload: Optional[bytes] = None
    component_payload: Optional[bytes] = None
    residual_payload: Optional[bytes] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def compressed_size(self) -> int:
        total = 0
        for payload in (self.raw_payload, self.component_payload, self.residual_payload):
            if payload is not None:
                total += ReversibleCompressor.compressed_size(payload)
        return total


class CodecTournament:
    """Build and score storage candidates."""

    @staticmethod
    def build_candidates(
        tensor: Any,
        target_ratio: float,
        exact: bool,
        decomp_type: Optional[DecompType],
    ) -> List[_Candidate]:
        t = _require_torch()
        candidates: List[_Candidate] = []

        raw_payload = TensorDecomposer.decompose(tensor, DecompType.RAW)
        candidates.append(
            _Candidate(
                decomp_type=DecompType.RAW,
                exact=True,
                ranks=None,
                raw_payload=raw_payload,
                meta={"strategy": "raw", "progressive": False},
            )
        )

        requested = {decomp_type} if decomp_type is not None else set(DecompType)

        if DecompType.SPARSE in requested:
            sparsity = float((tensor == 0).to(dtype=t.float32).mean().item()) if tensor.numel() else 1.0
            if decomp_type == DecompType.SPARSE or sparsity >= 0.60:
                component_payload = TensorDecomposer.decompose(tensor, DecompType.SPARSE)
                candidates.append(
                    _Candidate(
                        decomp_type=DecompType.SPARSE,
                        exact=True,
                        ranks=None,
                        component_payload=component_payload,
                        meta={"strategy": "sparse", "sparsity": sparsity, "progressive": False},
                    )
                )

        if DecompType.SVD in requested and TensorDecomposer._ensure_float_matrix(tensor):
            rank = TensorDecomposer.svd_rank(tensor.shape, target_ratio)
            component_payload = TensorDecomposer.decompose(tensor, DecompType.SVD, rank=rank)
            approx = TensorDecomposer.reconstruct(component_payload)
            residual_payload = TensorCodec.pack_xor_residual(tensor, approx) if exact else None
            candidates.append(
                _Candidate(
                    decomp_type=DecompType.SVD,
                    exact=exact,
                    ranks=(rank,),
                    component_payload=component_payload,
                    residual_payload=residual_payload,
                    meta={
                        "strategy": "svd",
                        "rank": rank,
                        "progressive": True,
                        "approx_first": True,
                    },
                )
            )

        if decomp_type in (DecompType.TT, DecompType.CP, DecompType.TUCKER):
            component_payload, ranks = CodecTournament._build_requested_tensorly(tensor, decomp_type, target_ratio)
            approx = TensorDecomposer.reconstruct(component_payload)
            residual_payload = TensorCodec.pack_xor_residual(tensor, approx) if exact else None
            candidates.append(
                _Candidate(
                    decomp_type=decomp_type,
                    exact=exact,
                    ranks=ranks,
                    component_payload=component_payload,
                    residual_payload=residual_payload,
                    meta={
                        "strategy": decomp_type.value,
                        "ranks": ranks,
                        "progressive": True,
                        "approx_first": True,
                    },
                )
            )

        if decomp_type is not None:
            forced = [c for c in candidates if c.decomp_type == decomp_type]
            if not forced:
                raise ValueError(f"Cannot build requested decomposition: {decomp_type.value}")
            return forced

        return candidates

    @staticmethod
    def _build_requested_tensorly(
        tensor: Any,
        decomp_type: DecompType,
        target_ratio: float,
    ) -> Tuple[bytes, Optional[Tuple[int, ...]]]:
        if tl is None:
            raise RuntimeError("TensorLy is required for TT/CP/TUCKER decomposition")
        shape = tuple(int(s) for s in tensor.shape)
        if decomp_type == DecompType.TT:
            ranks = [1]
            for i in range(len(shape) - 1):
                left = 1
                for s in shape[: i + 1]:
                    left *= s
                right = 1
                for s in shape[i + 1 :]:
                    right *= s
                ranks.append(max(1, min(left, right, int(min(left, right) * target_ratio))))
            ranks.append(1)
            ranks_tuple = tuple(ranks)
            return TensorDecomposer.decompose(tensor, DecompType.TT, ranks=ranks_tuple), ranks_tuple
        if decomp_type == DecompType.CP:
            rank = max(1, int(min(shape) * target_ratio))
            return TensorDecomposer.decompose(tensor, DecompType.CP, rank=rank), (rank,)
        ranks_tuple = tuple(max(1, int(s * target_ratio)) for s in shape)
        return TensorDecomposer.decompose(tensor, DecompType.TUCKER, ranks=ranks_tuple), ranks_tuple

    @staticmethod
    def choose(candidates: Iterable[_Candidate], prefer_progressive: bool) -> _Candidate:
        scored: List[Tuple[float, _Candidate]] = []
        for candidate in candidates:
            size = candidate.compressed_size()
            progressive_bonus = 0.94 if prefer_progressive and candidate.meta.get("progressive") else 1.0
            scored.append((size * progressive_bonus, candidate))
        if not scored:
            raise ValueError("No storage candidates were generated")
        return min(scored, key=lambda item: item[0])[1]


class ZCache:
    """LRU tensor cache. Returns clones to preserve content immutability."""

    def __init__(self, capacity_bytes: int = 1 << 30) -> None:
        self.capacity = int(capacity_bytes)
        self.used = 0
        self.cache: OrderedDict[bytes, Any] = OrderedDict()
        self.lock = RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    @staticmethod
    def _tensor_bytes(tensor: Any) -> int:
        return int(tensor.element_size() * tensor.nelement())

    def get(self, key: bytes) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return self.cache[key].clone()
            self.stats["misses"] += 1
            return None

    def put(self, key: bytes, tensor: Any) -> None:
        with self.lock:
            tensor_bytes = self._tensor_bytes(tensor)
            while self.used + tensor_bytes > self.capacity and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.used -= self._tensor_bytes(evicted)
                self.stats["evictions"] += 1
            if self.used + tensor_bytes <= self.capacity:
                self.cache[key] = tensor.detach().cpu().contiguous().clone()
                self.used += tensor_bytes


class MarkovPrefetcher:
    """Second-order Markov predictor."""

    def __init__(self, history_size: int = 1000) -> None:
        self.history = deque(maxlen=history_size)
        self.transitions: Dict[Tuple[bytes, bytes], List[bytes]] = {}
        self.lock = Lock()

    def record_access(self, addr: bytes) -> None:
        with self.lock:
            if len(self.history) >= 2:
                key = (self.history[-2], self.history[-1])
                self.transitions.setdefault(key, []).append(addr)
            self.history.append(addr)

    def predict_next(self, curr: bytes, prev: Optional[bytes]) -> List[bytes]:
        if prev is None:
            return []
        with self.lock:
            next_addrs = self.transitions.get((prev, curr), [])
            return [addr for addr, _ in Counter(next_addrs).most_common(3)]


class ZGen:
    """Synthesis engine and storage coordinator."""

    def __init__(self, cache_size: int = 1 << 30, store: Optional[ContentStore] = None) -> None:
        self.store = store or ContentStore()
        self.cache = ZCache(cache_size)
        self.prefetcher = MarkovPrefetcher()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.pending_synthesis: Dict[bytes, Future[Any]] = {}
        self.descriptor_index: Dict[bytes, ZDescriptor] = {}
        self.lock = Lock()
        self._last_addr: Optional[bytes] = None

    def remember(self, desc: ZDescriptor) -> None:
        self.descriptor_index[desc.address] = desc

    def store_tensor(
        self,
        tensor: Any,
        *,
        target_ratio: float = 0.1,
        exact: bool = True,
        decomp_type: Optional[DecompType] = None,
        prefer_progressive: bool = True,
    ) -> ZDescriptor:
        _require_torch()
        if not (0 < target_ratio <= 1):
            raise ValueError("target_ratio must be in the interval (0, 1]")

        if decomp_type == DecompType.RAW:
            candidate = _Candidate(
                decomp_type=DecompType.RAW,
                exact=True,
                ranks=None,
                raw_payload=TensorDecomposer.decompose(tensor, DecompType.RAW),
                meta={"strategy": "raw", "progressive": False, "fast_path": True},
            )
        else:
            candidates = CodecTournament.build_candidates(tensor, target_ratio, exact, decomp_type)
            candidate = CodecTournament.choose(candidates, prefer_progressive)

        raw_node = (
            self.store.put_tensor_payload(candidate.raw_payload, NodeKind.RAW_TENSOR)
            if candidate.raw_payload is not None
            else None
        )
        component_node = (
            self.store.put(candidate.component_payload, NodeKind.COMPONENTS)
            if candidate.component_payload is not None
            else None
        )
        residual_node = (
            self.store.put_tensor_payload(candidate.residual_payload, NodeKind.XOR_RESIDUAL)
            if candidate.residual_payload is not None
            else None
        )

        raw_bytes = tensor.nelement() * tensor.element_size()
        compressed_bytes = sum(
            self.store.tree_compressed_size(n)
            for n in (raw_node, component_node, residual_node)
            if n is not None
        )
        meta = {
            **candidate.meta,
            "original_dtype": TensorCodec.dtype_name(tensor.dtype),
            "raw_bytes": int(raw_bytes),
            "compressed_bytes": int(compressed_bytes),
            "compression_ratio": compressed_bytes / max(1, raw_bytes),
            "tensor_digest": TensorCodec.tensor_digest(tensor),
            "guarantee": "bitwise_exact" if candidate.exact else "approximate",
        }
        desc = ZDescriptor(
            kind="tensor",
            decomp_type=candidate.decomp_type,
            shape=tuple(int(s) for s in tensor.shape),
            ranks=candidate.ranks,
            exact=candidate.exact,
            version=0,
            raw_node=raw_node,
            component_node=component_node,
            residual_node=residual_node,
            meta=meta,
        )
        self.remember(desc)
        return desc

    def synthesize(
        self,
        desc: ZDescriptor,
        *,
        mode: LoadMode = LoadMode.EXACT,
        verify: bool = True,
        device: Optional[str] = None,
    ) -> Any:
        t = _require_torch()

        if desc.decomp_type == DecompType.RAW:
            if desc.raw_node is None:
                raise ValueError("RAW descriptor is missing raw_node")
            tensor = TensorCodec.unpack_tensor(self.store.get(desc.raw_node, NodeKind.RAW_TENSOR))
        else:
            if desc.component_node is None:
                raise ValueError(f"{desc.decomp_type.value} descriptor is missing component_node")
            approx = TensorDecomposer.reconstruct(self.store.get(desc.component_node, NodeKind.COMPONENTS))
            if mode == LoadMode.APPROX or not desc.exact:
                tensor = approx
            elif desc.residual_node is not None:
                residual_payload = self.store.get(desc.residual_node, NodeKind.XOR_RESIDUAL)
                tensor = TensorCodec.apply_xor_residual(approx, residual_payload)
            else:
                tensor = approx

        tensor = tensor.reshape(desc.shape)
        if desc.delta_chain_node is not None:
            deltas = mscs.loads(self.store.get(desc.delta_chain_node, NodeKind.DELTA_CHAIN))
            for delta in deltas:
                tensor = self._apply_delta(tensor, delta)

        if verify and mode == LoadMode.EXACT and desc.exact:
            expected = desc.meta_view().get("tensor_digest")
            if expected is not None and TensorCodec.tensor_digest(tensor) != expected:
                raise ValueError("Exact reconstruction failed tensor digest verification")

        if device is not None:
            tensor = tensor.to(device)
        elif t.cuda.is_available() and desc.meta_view().get("preferred_device") == "cuda":
            tensor = tensor.to("cuda")
        return tensor

    def _cache_key(self, desc: ZDescriptor, mode: LoadMode, device: Optional[str]) -> bytes:
        return desc.address + b"|" + mode.value.encode("ascii") + b"|" + (device or "cpu").encode("ascii")

    def load(
        self,
        desc: ZDescriptor,
        *,
        mode: LoadMode = LoadMode.EXACT,
        verify: bool = True,
        device: Optional[str] = None,
    ) -> Any:
        self.remember(desc)
        addr = desc.address
        key = self._cache_key(desc, mode, device)
        cached = self.cache.get(key)
        if cached is not None:
            self._record_and_prefetch(addr)
            return cached

        with self.lock:
            future = self.pending_synthesis.get(key)
            if future is None:
                future = self.executor.submit(self.synthesize, desc, mode=mode, verify=verify, device=device)
                self.pending_synthesis[key] = future

        try:
            tensor = future.result()
            self.cache.put(key, tensor)
            return tensor.clone()
        finally:
            with self.lock:
                self.pending_synthesis.pop(key, None)
            self._record_and_prefetch(addr)

    def _record_and_prefetch(self, current_addr: bytes) -> None:
        prev = self._last_addr
        self.prefetcher.record_access(current_addr)
        predicted = self.prefetcher.predict_next(current_addr, prev)
        self._last_addr = current_addr
        for addr in predicted:
            desc = self.descriptor_index.get(addr)
            if desc is None:
                continue
            key = self._cache_key(desc, LoadMode.APPROX, None)
            if self.cache.get(key) is None:
                self.executor.submit(self.load, desc, mode=LoadMode.APPROX, verify=False)

    def _delta_tensor_payload(self, delta: Mapping[str, Any], inline_key: str, node_key: str) -> bytes:
        node = delta.get(node_key)
        if node is not None:
            return self.store.get(node, NodeKind.TENSOR_PAYLOAD)
        return delta[inline_key]

    def _apply_delta(self, tensor: Any, delta: Mapping[str, Any]) -> Any:
        t = _require_torch()
        op_type = delta["type"]

        if op_type == "add":
            return tensor + delta["value"]
        if op_type == "mul":
            value = delta["value"]
            if value == 0:
                raise ValueError("mul delta by zero is not reversible")
            return tensor * value
        if op_type == "patch":
            indices_payload = self._delta_tensor_payload(delta, "indices", "indices_node")
            values_payload = self._delta_tensor_payload(delta, "new_values", "new_values_node")
            indices = TensorCodec.unpack_tensor(indices_payload).to(dtype=t.int64)
            values = TensorCodec.unpack_tensor(values_payload).to(dtype=tensor.dtype)
            out = tensor.clone()
            if indices.numel() > 0:
                out[tuple(indices.T)] = values
            return out
        raise ValueError(f"Unknown delta op: {op_type}")

    def invert_delta(self, delta: Mapping[str, Any]) -> Dict[str, Any]:
        op_type = delta["type"]
        if op_type == "add":
            return {"type": "add", "value": -delta["value"]}
        if op_type == "mul":
            value = delta["value"]
            if value == 0:
                raise ValueError("mul delta by zero is not reversible")
            return {"type": "mul", "value": 1 / value}
        if op_type == "patch":
            if "indices_node" in delta:
                return {
                    "type": "patch",
                    "indices_node": delta["indices_node"],
                    "new_values_node": delta["old_values_node"],
                }
            return {
                "type": "patch",
                "indices": delta["indices"],
                "old_values": delta["new_values"],
                "new_values": delta["old_values"],
            }
        raise ValueError(f"Unknown delta op: {op_type}")


class ZSpace:
    """Main runtime interface."""

    def __init__(self, cache_size: int = 1 << 30, store: Optional[ContentStore] = None) -> None:
        self.gen = ZGen(cache_size, store=store)
        self.descriptor_table: Dict[str, ZDescriptor] = {}
        self.version_graph: Dict[bytes, bytes] = {}
        self.checkpoint_table: Dict[str, CheckpointDescriptor] = {}
        self.checkpoint_index: Dict[bytes, CheckpointDescriptor] = {}
        self.checkpoint_version_graph: Dict[bytes, bytes] = {}

    def register(
        self,
        name: str,
        tensor: Any,
        *,
        target_ratio: float = 0.1,
        exact: bool = True,
        decomp_type: Optional[DecompType] = None,
        prefer_progressive: bool = True,
    ) -> ZDescriptor:
        desc = self.gen.store_tensor(
            tensor,
            target_ratio=target_ratio,
            exact=exact,
            decomp_type=decomp_type,
            prefer_progressive=prefer_progressive,
        )
        self.descriptor_table[name] = desc
        return desc

    def load(
        self,
        name: str,
        *,
        exact: bool = True,
        verify: bool = True,
        device: Optional[str] = None,
    ) -> Any:
        if name not in self.descriptor_table:
            raise KeyError(f"Unknown tensor: {name}")
        mode = LoadMode.EXACT if exact else LoadMode.APPROX
        return self.gen.load(self.descriptor_table[name], mode=mode, verify=verify, device=device)

    def load_desc(
        self,
        desc: ZDescriptor,
        *,
        exact: bool = True,
        verify: bool = True,
        device: Optional[str] = None,
    ) -> Any:
        mode = LoadMode.EXACT if exact else LoadMode.APPROX
        return self.gen.load(desc, mode=mode, verify=verify, device=device)

    def register_checkpoint(
        self,
        name: str,
        state_dict: Mapping[str, Any],
        *,
        target_ratio: float = 0.1,
        exact: bool = True,
        decomp_type: Optional[DecompType] = DecompType.RAW,
        prefer_progressive: bool = False,
    ) -> CheckpointDescriptor:
        tensor_descs = {
            key: self.gen.store_tensor(
                tensor,
                target_ratio=target_ratio,
                exact=exact,
                decomp_type=decomp_type,
                prefer_progressive=prefer_progressive,
            )
            for key, tensor in state_dict.items()
        }
        tensor_digests = {key: desc.meta_view().get("tensor_digest") for key, desc in tensor_descs.items()}
        meta = {
            "strategy": "full",
            "requires_parent": False,
            "checkpoint_policy": "base",
            "tensor_count": len(tensor_descs),
            "tensor_digests": tensor_digests,
        }
        desc = CheckpointDescriptor(version=0, tensor_descriptors=tensor_descs, meta=meta)
        self.checkpoint_table[name] = desc
        self.checkpoint_index[desc.address] = desc
        return desc

    def update_checkpoint(
        self,
        name: str,
        state_dict: Mapping[str, Any],
        *,
        codec: str = "xor_zstd",
        zstd_level: int = 3,
        delta_preconditioner: str = "xor",
        full_every: Optional[int] = None,
        parent_state: Optional[Mapping[str, Any]] = None,
        zstd_dicts: Optional[Mapping[str, bytes]] = None,
    ) -> CheckpointDescriptor:
        if name not in self.checkpoint_table:
            raise KeyError(f"Unknown checkpoint: {name}")
        if codec not in ("xor_zstd", "delta_zstd"):
            raise ValueError(f"Unsupported checkpoint delta codec: {codec}")
        delta_preconditioner = TensorDeltaCodec._normalize_preconditioner(delta_preconditioner)
        if codec == "xor_zstd" and delta_preconditioner != "xor":
            codec = "delta_zstd"
        if full_every is not None:
            full_every = int(full_every)
            if full_every <= 0:
                raise ValueError("full_every must be positive")

        parent = self.checkpoint_table[name]
        next_version = parent.version + 1
        materialize_full = full_every is not None and next_version % full_every == 0
        if materialize_full:
            tensor_descs = {
                key: self.gen.store_tensor(
                    tensor,
                    exact=True,
                    decomp_type=DecompType.RAW,
                    prefer_progressive=False,
                )
                for key, tensor in state_dict.items()
            }
            tensor_digests = {key: desc.meta_view().get("tensor_digest") for key, desc in tensor_descs.items()}
            meta = {
                "strategy": "full",
                "requires_parent": False,
                "checkpoint_policy": "periodic_full",
                "full_every": int(full_every),
                "tensor_count": len(tensor_descs),
                "tensor_digests": tensor_digests,
            }
            desc = CheckpointDescriptor(
                version=next_version,
                tensor_descriptors=tensor_descs,
                parent_addr=parent.address,
                meta=meta,
            )
            self.checkpoint_table[name] = desc
            self.checkpoint_index[desc.address] = desc
            self.checkpoint_version_graph[desc.address] = parent.address
            return desc

        if parent_state is None:
            parent_state = self.load_checkpoint_desc(parent, verify=True, zstd_dicts=zstd_dicts)
        full_tensors: Dict[str, ZDescriptor] = {}
        delta_nodes: Dict[str, bytes] = {}

        for key, tensor in state_dict.items():
            if key in parent_state and tuple(parent_state[key].shape) == tuple(tensor.shape) and parent_state[key].dtype == tensor.dtype:
                zstd_dict = zstd_dicts.get(key) if zstd_dicts is not None else None
                delta_payload = TensorDeltaCodec.pack_xor_zstd(
                    parent_state[key],
                    tensor,
                    level=zstd_level,
                    preconditioner=delta_preconditioner,
                    zstd_dict=zstd_dict,
                )
                delta_nodes[key] = self.gen.store.put(delta_payload, NodeKind.TENSOR_DELTA)
            else:
                full_tensors[key] = self.gen.store_tensor(
                    tensor,
                    exact=True,
                    decomp_type=DecompType.RAW,
                    prefer_progressive=False,
                )

        removed = sorted(set(parent_state) - set(state_dict))
        tensor_digests = {key: TensorCodec.tensor_digest(tensor) for key, tensor in state_dict.items()}
        meta = {
            "strategy": "xor_zstd_delta",
            "requires_parent": True,
            "codec": codec,
            "zstd_level": int(zstd_level),
            "delta_preconditioner": delta_preconditioner,
            "full_every": full_every,
            "zstd_dict_count": len(zstd_dicts) if zstd_dicts is not None else 0,
            "tensor_count": len(state_dict),
            "delta_count": len(delta_nodes),
            "full_tensor_count": len(full_tensors),
            "removed": removed,
            "tensor_digests": tensor_digests,
        }
        desc = CheckpointDescriptor(
            version=next_version,
            tensor_descriptors=full_tensors,
            tensor_delta_nodes=delta_nodes,
            parent_addr=parent.address,
            meta=meta,
        )
        self.checkpoint_table[name] = desc
        self.checkpoint_index[desc.address] = desc
        self.checkpoint_version_graph[desc.address] = parent.address
        return desc

    def load_checkpoint(
        self,
        name: str,
        *,
        verify: bool = True,
        zstd_dicts: Optional[Mapping[str, bytes]] = None,
    ) -> Dict[str, Any]:
        if name not in self.checkpoint_table:
            raise KeyError(f"Unknown checkpoint: {name}")
        return self.load_checkpoint_desc(self.checkpoint_table[name], verify=verify, zstd_dicts=zstd_dicts)

    def load_checkpoint_desc(
        self,
        desc: CheckpointDescriptor,
        *,
        verify: bool = True,
        zstd_dicts: Optional[Mapping[str, bytes]] = None,
    ) -> Dict[str, Any]:
        meta = desc.meta_view()
        requires_parent = desc.parent_addr is not None and bool(meta.get("requires_parent", True))
        if requires_parent:
            parent = self.checkpoint_index.get(desc.parent_addr)
            if parent is None:
                raise KeyError(f"Unknown parent checkpoint: {desc.parent_addr.hex()}")
            state = self.load_checkpoint_desc(parent, verify=verify, zstd_dicts=zstd_dicts)
        else:
            state = {}

        for key, tensor_desc in desc.tensor_map().items():
            state[key] = self.load_desc(tensor_desc, exact=True, verify=verify)
        for key, delta_node in desc.delta_map().items():
            if key not in state:
                raise ValueError(f"Delta references missing parent tensor: {key}")
            delta_payload = self.gen.store.get(delta_node, NodeKind.TENSOR_DELTA)
            zstd_dict = zstd_dicts.get(key) if zstd_dicts is not None else None
            state[key] = TensorDeltaCodec.apply_xor_zstd(
                state[key],
                delta_payload,
                verify=verify,
                zstd_dict=zstd_dict,
            )

        removed = desc.meta_view().get("removed", [])
        for key in removed:
            state.pop(key, None)
        return state

    def checkpoint_history(self, name: str) -> List[CheckpointDescriptor]:
        if name not in self.checkpoint_table:
            raise KeyError(f"Unknown checkpoint: {name}")
        history = []
        seen = set()
        desc = self.checkpoint_table[name]
        while True:
            if desc.address in seen:
                raise ValueError("Checkpoint version graph contains a cycle")
            seen.add(desc.address)
            history.append(desc)
            if desc.parent_addr is None:
                break
            parent = self.checkpoint_index.get(desc.parent_addr)
            if parent is None:
                raise KeyError(f"Unknown parent checkpoint: {desc.parent_addr.hex()}")
            desc = parent
        return list(reversed(history))

    def update(self, name: str, delta_op: Mapping[str, Any]) -> ZDescriptor:
        if name not in self.descriptor_table:
            raise KeyError(f"Unknown tensor: {name}")

        old_desc = self.descriptor_table[name]
        old_tensor = self.load_desc(old_desc, exact=True)
        normalized = self._normalize_delta(old_tensor, delta_op)

        if old_desc.delta_chain_node is not None:
            deltas = mscs.loads(self.gen.store.get(old_desc.delta_chain_node, NodeKind.DELTA_CHAIN))
        else:
            deltas = []
        deltas.append(normalized)
        delta_chain_node = self.gen.store.put(
            mscs.dumps(deltas),
            NodeKind.DELTA_CHAIN,
        )

        new_tensor = self.gen._apply_delta(old_tensor, normalized)
        meta = old_desc.meta_view()
        meta.update(
            {
                "tensor_digest": TensorCodec.tensor_digest(new_tensor),
                "guarantee": "bitwise_exact",
                "delta_count": len(deltas),
            }
        )
        new_desc = ZDescriptor(
            kind=old_desc.kind,
            decomp_type=old_desc.decomp_type,
            shape=tuple(new_tensor.shape),
            ranks=old_desc.ranks,
            exact=True,
            version=old_desc.version + 1,
            raw_node=old_desc.raw_node,
            component_node=old_desc.component_node,
            residual_node=old_desc.residual_node,
            delta_chain_node=delta_chain_node,
            parent_addr=old_desc.address,
            meta=meta,
        )

        self.descriptor_table[name] = new_desc
        self.version_graph[new_desc.address] = old_desc.address
        self.gen.remember(new_desc)
        return new_desc

    def _normalize_delta(self, tensor: Any, delta_op: Mapping[str, Any]) -> Dict[str, Any]:
        t = _require_torch()
        op_type = delta_op["type"]
        if op_type in ("add", "mul"):
            value = delta_op["value"]
            if op_type == "mul" and value == 0:
                raise ValueError("mul delta by zero is not reversible")
            return {"type": op_type, "value": value}

        if op_type in ("patch", "sparse_update"):
            if "new_values" in delta_op or "new_values_node" in delta_op:
                indices_source = (
                    self.gen.store.get(delta_op["indices_node"], NodeKind.TENSOR_PAYLOAD)
                    if "indices_node" in delta_op
                    else delta_op["indices"]
                )
                values_source = (
                    self.gen.store.get(delta_op["new_values_node"], NodeKind.TENSOR_PAYLOAD)
                    if "new_values_node" in delta_op
                    else delta_op["new_values"]
                )
                indices_tensor = (
                    TensorCodec.unpack_tensor(indices_source)
                    if isinstance(indices_source, bytes)
                    else indices_source
                )
                values_tensor = (
                    TensorCodec.unpack_tensor(values_source)
                    if isinstance(values_source, bytes)
                    else values_source
                )
                return self._normalize_delta(
                    tensor,
                    {"type": "patch", "indices": indices_tensor, "values": values_tensor},
                )

            indices = delta_op["indices"]
            values = delta_op["values"]
            if not isinstance(indices, t.Tensor):
                indices = t.tensor(indices, dtype=t.int64)
            else:
                indices = indices.to(dtype=t.int64)
            if indices.ndim == 1:
                indices = indices.reshape(1, -1)
            if not isinstance(values, t.Tensor):
                values = t.tensor(values, dtype=tensor.dtype)
            else:
                values = values.to(dtype=tensor.dtype)
            old_values = tensor[tuple(indices.T)].detach().cpu().contiguous() if indices.numel() else values[:0]
            indices_payload = TensorCodec.pack_tensor(indices.cpu().contiguous())
            old_values_payload = TensorCodec.pack_tensor(old_values)
            new_values_payload = TensorCodec.pack_tensor(values.detach().cpu().contiguous())
            return {
                "type": "patch",
                "indices_node": self.gen.store.put_tensor_payload(indices_payload, NodeKind.TENSOR_PAYLOAD),
                "old_values_node": self.gen.store.put_tensor_payload(old_values_payload, NodeKind.TENSOR_PAYLOAD),
                "new_values_node": self.gen.store.put_tensor_payload(new_values_payload, NodeKind.TENSOR_PAYLOAD),
            }

        raise ValueError(f"Unknown delta op: {op_type}")

    def revert_last_update(self, name: str) -> ZDescriptor:
        if name not in self.descriptor_table:
            raise KeyError(f"Unknown tensor: {name}")
        desc = self.descriptor_table[name]
        if desc.delta_chain_node is None:
            raise ValueError(f"Tensor {name!r} has no delta chain")
        deltas = mscs.loads(self.gen.store.get(desc.delta_chain_node, NodeKind.DELTA_CHAIN))
        if not deltas:
            raise ValueError(f"Tensor {name!r} has no deltas to revert")
        inverse = self.gen.invert_delta(deltas[-1])
        return self.update(name, inverse)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cache": dict(self.gen.cache.stats),
            "cache_used_bytes": self.gen.cache.used,
            "store": self.gen.store.stats(),
            "descriptors": len(self.descriptor_table),
            "versions": len(self.version_graph),
            "checkpoints": len(self.checkpoint_table),
            "checkpoint_versions": len(self.checkpoint_version_graph),
        }


__all__ = [
    "ContentStore",
    "CheckpointDescriptor",
    "DEFAULT_DELTA_CHUNK_SIZE",
    "DEFAULT_TENSOR_BLOCK_SIZE",
    "DecompType",
    "LoadMode",
    "NodeKind",
    "PackfileContentStore",
    "ReversibleCompressor",
    "TensorDeltaCodec",
    "TensorCodec",
    "TensorDecomposer",
    "ZDescriptor",
    "ZGen",
    "ZSpace",
]
