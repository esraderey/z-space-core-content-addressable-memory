import argparse
import csv
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import z_space_core as zcore
from z_space_core import (
    DEFAULT_TENSOR_BLOCK_SIZE,
    DecompType,
    NodeKind,
    ReversibleCompressor,
    TensorCodec,
    TensorDecomposer,
    ZDescriptor,
)


class Timer:
    def __init__(self) -> None:
        self.seconds = Counter()
        self.counts = Counter()
        self.bytes = Counter()

    def add(self, key: str, seconds: float, count: int = 1, byte_count: int = 0) -> None:
        self.seconds[key] += seconds
        self.counts[key] += count
        self.bytes[key] += byte_count

    def snapshot(self) -> Dict[str, Any]:
        return {
            "seconds": dict(self.seconds),
            "counts": dict(self.counts),
            "bytes": dict(self.bytes),
        }


class InstrumentedDiskContentStore:
    def __init__(self, root: Path, timer: Timer) -> None:
        self.root = root
        self.nodes_dir = root / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self._kinds: Dict[bytes, NodeKind] = {}
        self._sizes: Dict[bytes, int] = {}
        self._lock = RLock()
        self.timer = timer

    @staticmethod
    def _node_digest(payload: bytes, kind: NodeKind) -> bytes:
        return zcore._digest(b"zspace:node:v1:" + kind.value.encode("ascii"), payload)

    @staticmethod
    def _is_tensor_payload_kind(kind: NodeKind) -> bool:
        return kind in (NodeKind.RAW_TENSOR, NodeKind.XOR_RESIDUAL, NodeKind.TENSOR_PAYLOAD)

    def _path(self, digest: bytes) -> Path:
        hex_digest = digest.hex()
        return self.nodes_dir / hex_digest[:2] / hex_digest[2:]

    def put(self, payload: bytes, kind: NodeKind) -> bytes:
        started = time.perf_counter()
        digest = self._node_digest(payload, kind)
        self.timer.add("node_hash", time.perf_counter() - started, byte_count=len(payload))

        path = self._path(digest)
        with self._lock:
            if not path.exists():
                started = time.perf_counter()
                packed = ReversibleCompressor.compress(payload)
                self.timer.add("node_compress", time.perf_counter() - started, byte_count=len(payload))

                started = time.perf_counter()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(packed)
                os.replace(tmp, path)
                self.timer.add("node_write", time.perf_counter() - started, byte_count=len(packed))
                self.timer.add(f"write_{kind.value}", 0.0, byte_count=len(packed))
                self._sizes[digest] = len(packed)
            else:
                self._sizes.setdefault(digest, path.stat().st_size)
                self.timer.add("node_dedup_hit", 0.0)
            self._kinds.setdefault(digest, kind)
        return digest

    def put_tensor_payload(
        self,
        payload: bytes,
        kind: NodeKind,
        *,
        block_size: int = DEFAULT_TENSOR_BLOCK_SIZE,
    ) -> bytes:
        total_started = time.perf_counter()
        started = time.perf_counter()
        info, raw = TensorCodec.split_payload(payload)
        self.timer.add("tensor_payload_split", time.perf_counter() - started, byte_count=len(payload))

        if len(raw) <= block_size:
            digest = self.put(payload, kind)
            self.timer.add("put_tensor_payload_total", time.perf_counter() - total_started, byte_count=len(payload))
            return digest

        blocks = []
        block_started = time.perf_counter()
        for offset in range(0, len(raw), block_size):
            block = raw[offset : offset + block_size]
            block_digest = self.put(block, NodeKind.TENSOR_BLOCK)
            blocks.append({"digest": block_digest.hex(), "nbytes": len(block)})
        self.timer.add("tensor_block_loop_total", time.perf_counter() - block_started, len(blocks), len(raw))

        started = time.perf_counter()
        manifest = {
            "format": zcore._TENSOR_BLOCK_MANIFEST_FORMAT,
            "tensor_info": info,
            "block_size": int(block_size),
            "raw_nbytes": len(raw),
            "blocks": blocks,
        }
        manifest_payload = zcore._TENSOR_BLOCK_MANIFEST_HEADER + zcore._stable_json_bytes(manifest)
        self.timer.add("manifest_build", time.perf_counter() - started, byte_count=len(manifest_payload))
        digest = self.put(manifest_payload, kind)
        self.timer.add("put_tensor_payload_total", time.perf_counter() - total_started, byte_count=len(payload))
        return digest

    def _read_stored(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> Tuple[NodeKind, bytes]:
        with self._lock:
            if digest not in self._kinds:
                raise KeyError(f"Unknown content node: {digest.hex()}")
            kind = self._kinds[digest]
            path = self._path(digest)
        if expected_kind is not None and kind != expected_kind:
            raise TypeError(f"Node {digest.hex()} is {kind.value}, expected {expected_kind.value}")
        return kind, ReversibleCompressor.decompress(path.read_bytes())

    @staticmethod
    def _decode_tensor_block_manifest(payload: bytes) -> Optional[Dict[str, Any]]:
        if not payload.startswith(zcore._TENSOR_BLOCK_MANIFEST_HEADER):
            return None
        manifest = json.loads(payload[len(zcore._TENSOR_BLOCK_MANIFEST_HEADER) :].decode("utf-8"))
        if manifest.get("format") != zcore._TENSOR_BLOCK_MANIFEST_FORMAT:
            raise ValueError("Unknown tensor block manifest format")
        return manifest

    def _materialize_tensor_payload(self, manifest: Mapping[str, Any]) -> bytes:
        info = dict(manifest["tensor_info"])
        raw_parts = []
        for block in manifest["blocks"]:
            block_digest = bytes.fromhex(block["digest"])
            block_payload = self.get(block_digest, NodeKind.TENSOR_BLOCK)
            raw_parts.append(block_payload)
        return TensorCodec.pack_raw_parts(info, b"".join(raw_parts))

    def get(self, digest: bytes, expected_kind: Optional[NodeKind] = None) -> bytes:
        kind, payload = self._read_stored(digest, expected_kind)
        if self._is_tensor_payload_kind(kind):
            manifest = self._decode_tensor_block_manifest(payload)
            if manifest is not None:
                return self._materialize_tensor_payload(manifest)
        return payload

    def compressed_size(self, digest: bytes) -> int:
        return self._sizes[digest]

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
            if block_digest not in seen:
                seen.add(block_digest)
                total += self.compressed_size(block_digest)
        return total

    def stats(self) -> Dict[str, Any]:
        node_kinds = Counter(kind.value for kind in self._kinds.values())
        return {
            "nodes": len(self._kinds),
            "compressed_bytes": sum(self._sizes.values()),
            "node_kinds": dict(node_kinds),
            "store_dir": str(self.root),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="EleutherAI/pythia-410m")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--steps-per-checkpoint", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--store-dir", default=".zspace_bench/pythia_delta_probe/store")
    parser.add_argument("--results-json", default=".zspace_bench/pythia_delta_probe/results.json")
    parser.add_argument("--results-csv", default=".zspace_bench/pythia_delta_probe/commits.csv")
    parser.add_argument("--clear-store", action="store_true")
    parser.add_argument("--zstd-level", type=int, default=3)
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def token_batch_iterator(args: argparse.Namespace, tokenizer: Any, load_dataset: Any, device: torch.device):
    token_buffer = []
    needed = args.batch_size * args.block_size
    while True:
        dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, streaming=True)
        for row in dataset:
            text = row.get("text", "")
            if not text or not text.strip():
                continue
            token_buffer.extend(tokenizer.encode(text))
            token_buffer.append(tokenizer.eos_token_id)
            while len(token_buffer) >= needed:
                ids = token_buffer[:needed]
                del token_buffer[:needed]
                input_ids = torch.tensor(ids, dtype=torch.long, device=device).reshape(args.batch_size, args.block_size)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": torch.ones_like(input_ids),
                }


def raw_tensor_bytes(tensor: torch.Tensor) -> bytes:
    _, raw = TensorCodec.raw_parts(tensor.detach().cpu().contiguous())
    return raw


def xor_bytes(left: bytes, right: bytes) -> bytes:
    if len(left) != len(right):
        raise ValueError("XOR inputs must have equal length")
    return bytes(a ^ b for a, b in zip(left, right))


def canonical_xor_zstd(state_a: Mapping[str, torch.Tensor], state_b: Mapping[str, torch.Tensor], level: int) -> Dict[str, Any]:
    compressor = zstd.ZstdCompressor(level=level)
    total_raw = 0
    total_xor_zstd = 0
    per_tensor = []
    started = time.perf_counter()
    for key in state_a:
        raw_a = raw_tensor_bytes(state_a[key])
        raw_b = raw_tensor_bytes(state_b[key])
        xored = xor_bytes(raw_a, raw_b)
        compressed = compressor.compress(xored)
        per_tensor.append(
            {
                "key": key,
                "raw_bytes": len(raw_a),
                "xor_zstd_bytes": len(compressed),
                "ratio": len(compressed) / max(1, len(raw_a)),
            }
        )
        total_raw += len(raw_a)
        total_xor_zstd += len(compressed)
    return {
        "raw_bytes": total_raw,
        "xor_zstd_bytes": total_xor_zstd,
        "ratio": total_xor_zstd / max(1, total_raw),
        "seconds": time.perf_counter() - started,
        "per_tensor": per_tensor,
    }


def file_xor_zstd(path_a: Path, path_b: Path, level: int) -> Dict[str, Any]:
    data_a = path_a.read_bytes()
    data_b = path_b.read_bytes()
    size = max(len(data_a), len(data_b))
    data_a = data_a.ljust(size, b"\0")
    data_b = data_b.ljust(size, b"\0")
    started = time.perf_counter()
    compressed = zstd.ZstdCompressor(level=level).compress(xor_bytes(data_a, data_b))
    return {
        "file_a_bytes": path_a.stat().st_size,
        "file_b_bytes": path_b.stat().st_size,
        "padded_bytes": size,
        "xor_zstd_bytes": len(compressed),
        "ratio": len(compressed) / max(1, size),
        "seconds": time.perf_counter() - started,
    }


def measured_commit(
    checkpoint_idx: int,
    state: Mapping[str, torch.Tensor],
    store: InstrumentedDiskContentStore,
    previous_store_bytes: int,
) -> Tuple[Dict[str, ZDescriptor], Dict[str, Any]]:
    rows = []
    descriptors = {}
    commit_started = time.perf_counter()
    for key, tensor in state.items():
        tensor_started = time.perf_counter()
        raw_started = time.perf_counter()
        raw_payload = TensorDecomposer.decompose(tensor, DecompType.RAW)
        raw_payload_seconds = time.perf_counter() - raw_started

        score_started = time.perf_counter()
        candidate_score_bytes = ReversibleCompressor.compressed_size(raw_payload)
        candidate_score_seconds = time.perf_counter() - score_started

        store_started = time.perf_counter()
        raw_node = store.put_tensor_payload(raw_payload, NodeKind.RAW_TENSOR)
        store_seconds = time.perf_counter() - store_started

        tree_started = time.perf_counter()
        compressed_bytes = store.tree_compressed_size(raw_node)
        tree_seconds = time.perf_counter() - tree_started

        digest_started = time.perf_counter()
        tensor_digest = TensorCodec.tensor_digest(tensor)
        digest_seconds = time.perf_counter() - digest_started

        desc = ZDescriptor(
            kind="tensor",
            decomp_type=DecompType.RAW,
            shape=tuple(int(s) for s in tensor.shape),
            exact=True,
            version=0,
            raw_node=raw_node,
            meta={
                "strategy": "raw",
                "progressive": False,
                "original_dtype": TensorCodec.dtype_name(tensor.dtype),
                "raw_bytes": int(tensor.nelement() * tensor.element_size()),
                "compressed_bytes": int(compressed_bytes),
                "compression_ratio": compressed_bytes / max(1, int(tensor.nelement() * tensor.element_size())),
                "tensor_digest": tensor_digest,
                "guarantee": "bitwise_exact",
            },
        )
        descriptors[key] = desc
        rows.append(
            {
                "checkpoint": checkpoint_idx,
                "key": key,
                "tensor_seconds": time.perf_counter() - tensor_started,
                "raw_payload_seconds": raw_payload_seconds,
                "candidate_score_seconds": candidate_score_seconds,
                "candidate_score_bytes": candidate_score_bytes,
                "store_seconds": store_seconds,
                "tree_size_seconds": tree_seconds,
                "tensor_digest_seconds": digest_seconds,
                "raw_bytes": int(tensor.nelement() * tensor.element_size()),
            }
        )
    stats = store.stats()
    return descriptors, {
        "checkpoint": checkpoint_idx,
        "commit_seconds": time.perf_counter() - commit_started,
        "growth_bytes": stats["compressed_bytes"] - previous_store_bytes,
        "store_bytes": stats["compressed_bytes"],
        "nodes": stats["nodes"],
        "tensor_rows": rows,
    }


def disk_sequential_probe(root: Path) -> Dict[str, Any]:
    path = root / "disk_probe.bin"
    size = 512 * 1024 * 1024
    chunk = b"\0" * (8 * 1024 * 1024)
    started = time.perf_counter()
    with path.open("wb", buffering=0) as handle:
        for _ in range(size // len(chunk)):
            handle.write(chunk)
    write_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with path.open("rb", buffering=0) as handle:
        while handle.read(len(chunk)):
            pass
    read_seconds = time.perf_counter() - started
    path.unlink()
    return {
        "bytes": size,
        "write_seconds": write_seconds,
        "read_seconds": read_seconds,
        "write_mb_s": size / write_seconds / (1024 * 1024),
        "read_mb_s": size / read_seconds / (1024 * 1024),
    }


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(args.store_dir).parents[0]
    if args.clear_store and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "disk_probe_start"}), flush=True)
    disk_probe = disk_sequential_probe(root)
    print(json.dumps({"event": "disk_probe", **disk_probe}), flush=True)

    device = torch.device("cuda")
    dtype = torch_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batches = token_batch_iterator(args, tokenizer, load_dataset, device)

    states = {}
    losses = {}
    for step in range(1, args.steps_per_checkpoint * 2 + 1):
        batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        if step in (args.steps_per_checkpoint, args.steps_per_checkpoint * 2):
            idx = step // args.steps_per_checkpoint
            losses[idx] = float(out.loss.detach().cpu())
            states[idx] = {key: tensor.detach().cpu().contiguous().clone() for key, tensor in model.state_dict().items()}
            print(json.dumps({"event": "captured_state", "checkpoint": idx, "step": step, "loss": losses[idx]}), flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    timer = Timer()
    store = InstrumentedDiskContentStore(store_dir, timer)
    previous_store_bytes = 0
    commit_summaries = []
    all_rows = []
    for checkpoint_idx in (1, 2):
        _, summary = measured_commit(checkpoint_idx, states[checkpoint_idx], store, previous_store_bytes)
        previous_store_bytes = summary["store_bytes"]
        all_rows.extend(summary.pop("tensor_rows"))
        commit_summaries.append(summary)
        print(json.dumps({"event": "commit", **summary}, sort_keys=True), flush=True)

    with tempfile.TemporaryDirectory(dir=root) as tmp:
        tmpdir = Path(tmp)
        paths = {}
        torch_sizes = {}
        for checkpoint_idx in (1, 2):
            path = tmpdir / f"checkpoint_{checkpoint_idx}.pt"
            started = time.perf_counter()
            torch.save(states[checkpoint_idx], path)
            torch_sizes[checkpoint_idx] = {
                "bytes": path.stat().st_size,
                "seconds": time.perf_counter() - started,
            }
            paths[checkpoint_idx] = path
        file_delta = file_xor_zstd(paths[1], paths[2], args.zstd_level)
        print(json.dumps({"event": "file_xor_zstd", **file_delta}), flush=True)

    canonical_delta = canonical_xor_zstd(states[1], states[2], args.zstd_level)
    print(
        json.dumps(
            {
                "event": "canonical_xor_zstd",
                "raw_bytes": canonical_delta["raw_bytes"],
                "xor_zstd_bytes": canonical_delta["xor_zstd_bytes"],
                "ratio": canonical_delta["ratio"],
                "seconds": canonical_delta["seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    with Path(args.results_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    result = {
        "losses": losses,
        "disk_probe": disk_probe,
        "torch_checkpoint_sizes": torch_sizes,
        "commits": commit_summaries,
        "timer": timer.snapshot(),
        "canonical_xor_zstd": canonical_delta,
        "file_xor_zstd": file_delta,
        "store_stats": store.stats(),
    }
    Path(args.results_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", "results_json": args.results_json}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
