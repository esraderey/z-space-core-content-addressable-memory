import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import shutil
import time
from pathlib import Path
from statistics import mean, median
from threading import RLock
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import z_space_core as zcore
from z_space_core import (
    DEFAULT_TENSOR_BLOCK_SIZE,
    DecompType,
    NodeKind,
    PackfileContentStore,
    ReversibleCompressor,
    TensorCodec,
    ZSpace,
)


DEFAULT_HF_HOME = ".zspace_bench/hf_home"
DEFAULT_HF_DATASETS_CACHE = ".zspace_bench/hf_datasets"


class DiskContentStore:
    """Disk-backed ContentStore-compatible node store for large benchmarks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.nodes_dir = root / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self._kinds: Dict[bytes, NodeKind] = {}
        self._sizes: Dict[bytes, int] = {}
        self._lock = RLock()

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
        digest = self._node_digest(payload, kind)
        path = self._path(digest)
        with self._lock:
            if not path.exists():
                packed = ReversibleCompressor.compress(payload)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(packed)
                os.replace(tmp, path)
                self._sizes[digest] = len(packed)
            else:
                self._sizes.setdefault(digest, path.stat().st_size)
            self._kinds.setdefault(digest, kind)
        return digest

    def put_tensor_payload(
        self,
        payload: bytes,
        kind: NodeKind,
        *,
        block_size: int = DEFAULT_TENSOR_BLOCK_SIZE,
    ) -> bytes:
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
            "format": zcore._TENSOR_BLOCK_MANIFEST_FORMAT,
            "tensor_info": info,
            "block_size": int(block_size),
            "raw_nbytes": len(raw),
            "blocks": blocks,
        }
        payload = zcore._TENSOR_BLOCK_MANIFEST_HEADER + zcore._stable_json_bytes(manifest)
        return self.put(payload, kind)

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
            return digest in self._kinds

    def compressed_size(self, digest: bytes) -> int:
        with self._lock:
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
            if block_digest in seen:
                continue
            seen.add(block_digest)
            total += self.compressed_size(block_digest)
        return total

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            node_kinds: Dict[str, int] = {}
            for kind in self._kinds.values():
                node_kinds[kind.value] = node_kinds.get(kind.value, 0) + 1
            return {
                "nodes": len(self._kinds),
                "compressed_bytes": sum(self._sizes.values()),
                "node_kinds": node_kinds,
                "store_dir": str(self.root),
            }


def require_modules(module_names: Iterable[str]) -> None:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        joined = " ".join(missing)
        raise RuntimeError(f"Missing benchmark dependencies: {joined}. Install them with pip first.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Z-Space growth on real SFT snapshots: Pythia-410M on WikiText, "
            "committing every N steps."
        )
    )
    parser.add_argument("--model-id", default="EleutherAI/pythia-410m")
    parser.add_argument("--dataset-name", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--checkpoints", type=int, default=100)
    parser.add_argument("--steps-per-checkpoint", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--eval-probe-batches", type=int, default=0)
    parser.add_argument("--store-dir", default=".zspace_bench/pythia_410m_wikitext/store")
    parser.add_argument("--results-json", default=".zspace_bench/pythia_410m_wikitext/results.json")
    parser.add_argument("--results-csv", default=".zspace_bench/pythia_410m_wikitext/checkpoints.csv")
    parser.add_argument("--checkpoint-mode", choices=("full", "xor-zstd"), default="full")
    parser.add_argument("--zstd-level", type=int, default=3)
    parser.add_argument("--checkpoint-full-every", type=int, default=0)
    parser.add_argument("--delta-preconditioner", choices=("xor", "u16-sub"), default="xor")
    parser.add_argument("--measure-torch-checkpoints", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--clear-store", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-seed", type=int, default=1234)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE)
    parser.add_argument("--hf-hub-cache", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def configure_hf_cache(args: argparse.Namespace) -> None:
    hf_home = Path(args.hf_home).expanduser().resolve()
    hf_hub_cache = Path(args.hf_hub_cache).expanduser().resolve() if args.hf_hub_cache else hf_home / "hub"
    hf_datasets_cache = Path(args.hf_datasets_cache).expanduser().resolve()
    hf_assets_cache = hf_home / "assets"

    for path in (hf_home, hf_hub_cache, hf_datasets_cache, hf_assets_cache):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    os.environ["HF_ASSETS_CACHE"] = str(hf_assets_cache)

    args.hf_home = str(hf_home)
    args.hf_hub_cache = str(hf_hub_cache)
    args.hf_datasets_cache = str(hf_datasets_cache)


def torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def print_preflight(args: argparse.Namespace) -> None:
    deps = ["transformers", "huggingface_hub", "safetensors"]
    if not args.synthetic_data:
        deps.append("datasets")
    found = {name: importlib.util.find_spec(name) is not None for name in deps}
    disk = shutil.disk_usage(".")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_count={torch.cuda.device_count()}")
    print(f"cuda_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    print(f"dependencies={found}")
    print(f"disk_free_bytes={disk.free}")
    print(f"model_id={args.model_id}")
    print(f"dataset={args.dataset_name}/{args.dataset_config}")
    print(f"total_steps={args.checkpoints * args.steps_per_checkpoint}")
    print(f"checkpoint_interval_steps={args.steps_per_checkpoint}")
    print(f"eval_probe_batches={args.eval_probe_batches}")
    print(f"checkpoint_mode={args.checkpoint_mode}")
    print(f"checkpoint_full_every={args.checkpoint_full_every}")
    print(f"delta_preconditioner={args.delta_preconditioner}")
    print(f"synthetic_data={args.synthetic_data}")
    print(f"hf_home={args.hf_home}")
    print(f"hf_hub_cache={args.hf_hub_cache}")
    print(f"hf_datasets_cache={args.hf_datasets_cache}")


def token_batch_iterator(args: argparse.Namespace, tokenizer: Any, load_dataset: Any, device: torch.device) -> Iterable[Dict[str, torch.Tensor]]:
    token_buffer = []
    needed = args.batch_size * args.block_size
    while True:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=True,
            cache_dir=args.hf_datasets_cache,
        )
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


def synthetic_batch_iterator(args: argparse.Namespace, model_config: Any, device: torch.device) -> Iterable[Dict[str, torch.Tensor]]:
    vocab_size = int(getattr(model_config, "vocab_size"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.synthetic_seed)
    while True:
        input_ids = torch.randint(
            0,
            vocab_size,
            (args.batch_size, args.block_size),
            dtype=torch.long,
            generator=generator,
        ).to(device)
        yield {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def batch_to_cpu(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().contiguous() for key, tensor in batch.items()}


def batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in batch.items()}


def evaluate_probe_loss(model: Any, probe_batches: Sequence[Mapping[str, torch.Tensor]], device: torch.device) -> Optional[float]:
    if not probe_batches:
        return None
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in probe_batches:
            output = model(**batch_to_device(batch, device))
            losses.append(float(output.loss.detach().cpu()))
    if was_training:
        model.train()
    return mean(losses)


def commit_checkpoint(space: ZSpace, checkpoint_idx: int, state_dict: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, Any], float]:
    descriptors = {}
    started = time.perf_counter()
    for key, tensor in state_dict.items():
        desc = space.register(
            f"checkpoint_{checkpoint_idx:04d}::{key}",
            tensor.detach().cpu().contiguous(),
            exact=True,
            decomp_type=DecompType.RAW,
            prefer_progressive=False,
        )
        descriptors[key] = desc
    return descriptors, time.perf_counter() - started


def checkpoint_size_bytes(state_dict: Mapping[str, torch.Tensor], work_dir: Path, checkpoint_idx: int) -> Tuple[int, float]:
    path = work_dir / f"torch_checkpoint_{checkpoint_idx:04d}.pt"
    started = time.perf_counter()
    torch.save({key: tensor.detach().cpu() for key, tensor in state_dict.items()}, path)
    elapsed = time.perf_counter() - started
    size = path.stat().st_size
    path.unlink()
    return size, elapsed


def reconstruct_checkpoint(space: ZSpace, descriptors: Mapping[str, Any]) -> Tuple[Dict[str, torch.Tensor], float]:
    started = time.perf_counter()
    restored = {key: space.load_desc(desc, exact=True, verify=True) for key, desc in descriptors.items()}
    return restored, time.perf_counter() - started


def cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().contiguous().clone() for key, tensor in state_dict.items()}


def commit_versioned_checkpoint(
    space: ZSpace,
    name: str,
    checkpoint_idx: int,
    state_dict: Mapping[str, torch.Tensor],
    previous_state: Optional[Mapping[str, torch.Tensor]],
    *,
    zstd_level: int,
    full_every: Optional[int],
    delta_preconditioner: str,
) -> Tuple[Any, Dict[str, torch.Tensor], float]:
    current_state = cpu_state_dict(state_dict)
    started = time.perf_counter()
    if checkpoint_idx == 1:
        desc = space.register_checkpoint(name, current_state)
    else:
        desc = space.update_checkpoint(
            name,
            current_state,
            zstd_level=zstd_level,
            delta_preconditioner=delta_preconditioner,
            full_every=full_every,
            parent_state=previous_state,
        )
    return desc, current_state, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if args.eval_probe_batches < 0:
        raise ValueError("eval-probe-batches must be non-negative")
    configure_hf_cache(args)
    print_preflight(args)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this real SFT benchmark. Use --allow-cpu only for tiny dry runs.")
    deps = ["transformers", "huggingface_hub", "safetensors"]
    if not args.synthetic_data:
        deps.append("datasets")
    require_modules(deps)

    from transformers import AutoModelForCausalLM
    if args.synthetic_data:
        load_dataset = None
        AutoTokenizer = None
    else:
        from datasets import load_dataset
        from transformers import AutoTokenizer

    store_dir = Path(args.store_dir)
    if args.clear_store and store_dir.exists():
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results_csv).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch_dtype(args.dtype)

    tokenizer = None
    if not args.synthetic_data:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.hf_hub_cache)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=args.hf_hub_cache,
    ).to(device)
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    store = PackfileContentStore(store_dir)
    space = ZSpace(cache_size=0, store=store)
    if args.synthetic_data:
        batches = synthetic_batch_iterator(args, model.config, device)
    else:
        batches = token_batch_iterator(args, tokenizer, load_dataset, device)
    probe_batches = [batch_to_cpu(next(batches)) for _ in range(args.eval_probe_batches)]

    total_steps = args.checkpoints * args.steps_per_checkpoint
    rows = []
    checkpoint_descriptors: Dict[int, Any] = {}
    previous_checkpoint_state: Optional[Dict[str, torch.Tensor]] = None
    previous_store_bytes = space.get_stats()["store"]["compressed_bytes"]
    torch_checkpoint_total = 0
    losses = []
    interval_losses = []

    benchmark_started = time.perf_counter()
    for step in range(1, total_steps + 1):
        batch = next(batches)
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = output.loss
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        interval_losses.append(loss_value)

        if step % args.steps_per_checkpoint != 0:
            continue

        checkpoint_idx = step // args.steps_per_checkpoint
        loss_interval_mean = mean(interval_losses)
        loss_interval_min = min(interval_losses)
        loss_interval_max = max(interval_losses)
        loss_interval_first = interval_losses[0]
        loss_interval_last = interval_losses[-1]
        state_dict = model.state_dict()
        if args.checkpoint_mode == "full":
            descriptors, commit_seconds = commit_checkpoint(space, checkpoint_idx, state_dict)
            checkpoint_descriptors[checkpoint_idx] = descriptors
        else:
            checkpoint_desc, previous_checkpoint_state, commit_seconds = commit_versioned_checkpoint(
                space,
                "model",
                checkpoint_idx,
                state_dict,
                previous_checkpoint_state,
                zstd_level=args.zstd_level,
                full_every=args.checkpoint_full_every or None,
                delta_preconditioner=args.delta_preconditioner,
            )
            checkpoint_descriptors[checkpoint_idx] = checkpoint_desc
        stats = space.get_stats()["store"]
        store_bytes = stats["compressed_bytes"]
        growth_bytes = store_bytes - previous_store_bytes
        previous_store_bytes = store_bytes

        torch_checkpoint_bytes = None
        torch_checkpoint_seconds = None
        if args.measure_torch_checkpoints:
            torch_checkpoint_bytes, torch_checkpoint_seconds = checkpoint_size_bytes(state_dict, store_dir, checkpoint_idx)
            torch_checkpoint_total += torch_checkpoint_bytes

        row = {
            "checkpoint": checkpoint_idx,
            "step": step,
            "loss": losses[-1],
            "loss_interval_mean": loss_interval_mean,
            "loss_interval_min": loss_interval_min,
            "loss_interval_max": loss_interval_max,
            "loss_interval_first": loss_interval_first,
            "loss_interval_last": loss_interval_last,
            "loss_interval_delta": loss_interval_last - loss_interval_first,
            "eval_probe_loss": evaluate_probe_loss(model, probe_batches, device),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "commit_seconds": commit_seconds,
            "growth_bytes": growth_bytes,
            "store_bytes": store_bytes,
            "torch_checkpoint_bytes": torch_checkpoint_bytes,
            "torch_checkpoint_seconds": torch_checkpoint_seconds,
            "nodes": stats["nodes"],
        }
        rows.append(row)
        with Path(args.results_csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row, sort_keys=True), flush=True)
        interval_losses = []
        gc.collect()

    final_descriptors = checkpoint_descriptors[args.checkpoints]
    if args.checkpoint_mode == "full":
        final_state, reconstruction_seconds = reconstruct_checkpoint(space, final_descriptors)
    else:
        started = time.perf_counter()
        final_state = space.load_checkpoint_desc(final_descriptors)
        reconstruction_seconds = time.perf_counter() - started
    current_state = {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
    final_exact = all(torch.equal(final_state[key], current_state[key]) for key in current_state)
    final_store_bytes = space.get_stats()["store"]["compressed_bytes"]
    growths = [int(row["growth_bytes"]) for row in rows]
    commits = [float(row["commit_seconds"]) for row in rows]

    summary = {
        "model_id": args.model_id,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "synthetic_data": args.synthetic_data,
        "hf_home": args.hf_home,
        "hf_hub_cache": args.hf_hub_cache,
        "hf_datasets_cache": args.hf_datasets_cache,
        "checkpoints": args.checkpoints,
        "steps_per_checkpoint": args.steps_per_checkpoint,
        "total_steps": total_steps,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "eval_probe_batches": args.eval_probe_batches,
        "dtype": args.dtype,
        "checkpoint_mode": args.checkpoint_mode,
        "zstd_level": args.zstd_level if args.checkpoint_mode == "xor-zstd" else None,
        "checkpoint_full_every": args.checkpoint_full_every or None,
        "delta_preconditioner": args.delta_preconditioner if args.checkpoint_mode == "xor-zstd" else None,
        "final_exact": final_exact,
        "final_reconstruction_seconds": reconstruction_seconds,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "avg_step_loss": mean(losses) if losses else None,
        "min_step_loss": min(losses) if losses else None,
        "max_step_loss": max(losses) if losses else None,
        "final_store_bytes": final_store_bytes,
        "total_growth_bytes": sum(growths),
        "avg_growth_bytes": mean(growths),
        "median_growth_bytes": median(growths),
        "min_growth_bytes": min(growths),
        "max_growth_bytes": max(growths),
        "avg_commit_seconds": mean(commits),
        "median_commit_seconds": median(commits),
        "min_commit_seconds": min(commits),
        "max_commit_seconds": max(commits),
        "torch_checkpoint_total_bytes": torch_checkpoint_total if args.measure_torch_checkpoints else None,
        "store_stats": space.get_stats()["store"],
        "wall_seconds": time.perf_counter() - benchmark_started,
    }
    Path(args.results_json).write_text(json.dumps({"summary": summary, "checkpoints": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    store.close()
    if not final_exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
