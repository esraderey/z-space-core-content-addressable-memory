import argparse
import csv
import gc
import importlib.util
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import zstandard as zstd
from safetensors.torch import save as save_safetensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_sft_pythia_410m import (  # noqa: E402
    DEFAULT_HF_DATASETS_CACHE,
    DEFAULT_HF_HOME,
    batch_to_cpu,
    configure_hf_cache,
    evaluate_probe_loss,
    require_modules,
    synthetic_batch_iterator,
    token_batch_iterator,
    torch_dtype,
)


BASELINE_NAME = "Safetensors + zstd dictionary compression baseline"
DICT100K_BYTES = 100_000
DICT1M_BYTES = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark independent safetensors checkpoints compressed per-file "
            "with zstd using a dictionary trained from the base checkpoint."
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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-seed", type=int, default=1234)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--hf-datasets-cache", default=DEFAULT_HF_DATASETS_CACHE)
    parser.add_argument("--hf-hub-cache", default=None)
    parser.add_argument("--zstd-level", type=int, default=9)
    parser.add_argument("--zstd-threads", type=int, default=-1)
    parser.add_argument("--dict100k-size", type=int, default=DICT100K_BYTES)
    parser.add_argument("--dict1m-size", type=int, default=DICT1M_BYTES)
    parser.add_argument("--dict-sample-chunk-size", type=int, default=1 << 20)
    parser.add_argument(
        "--dict-train-max-bytes",
        type=int,
        default=0,
        help="Maximum bytes from the base checkpoint used to train the dictionary. 0 means all bytes.",
    )
    parser.add_argument("--results-json", default=".zspace_bench/safetensors_zstd_baseline/results.json")
    parser.add_argument("--results-csv", default=".zspace_bench/safetensors_zstd_baseline/checkpoints.csv")
    parser.add_argument("--analysis-json", default=".zspace_bench/safetensors_zstd_baseline/analysis_summary.json")
    parser.add_argument("--write-compressed-dir", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def append_row(path: Path, row: Mapping[str, Any]) -> bool:
    fieldnames = list(row.keys())
    for attempt in range(60):
        try:
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            return True
        except PermissionError:
            time.sleep(1.0)
    return False


def write_csv_with_retries(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return True
    tmp = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    for attempt in range(60):
        try:
            with tmp.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(1.0)
    return False


def write_json_with_retries(path: Path, payload: Mapping[str, Any]) -> bool:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2)
    for attempt in range(60):
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(1.0)
    return False


def cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().contiguous() for key, tensor in state_dict.items()}


def serialize_safetensors(state_dict: Mapping[str, torch.Tensor]) -> tuple[bytes, float]:
    started = time.perf_counter()
    payload = save_safetensors(cpu_state_dict(state_dict))
    return payload, time.perf_counter() - started


def torch_save_size_once(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, float]:
    buffer = io.BytesIO()
    started = time.perf_counter()
    torch.save(cpu_state_dict(state_dict), buffer)
    return buffer.tell(), time.perf_counter() - started


def dictionary_samples(base_bytes: bytes, chunk_size: int, max_bytes: int) -> Sequence[bytes]:
    if chunk_size <= 0:
        raise ValueError("dict-sample-chunk-size must be positive")
    limit = len(base_bytes) if max_bytes <= 0 else min(len(base_bytes), max_bytes)
    samples = [base_bytes[offset : offset + chunk_size] for offset in range(0, limit, chunk_size)]
    if len(samples) < 2:
        raise ValueError("zstd dictionary training needs at least two samples; lower chunk size or use more bytes")
    return samples


def train_base_dictionary(
    samples: Sequence[bytes],
    dict_size: int,
    label: str,
) -> tuple[zstd.ZstdCompressionDict, Dict[str, Any]]:
    started = time.perf_counter()
    trained = zstd.train_dictionary(dict_size, samples)
    elapsed = time.perf_counter() - started
    return trained, {
        f"{label}_dictionary_requested_bytes": dict_size,
        f"{label}_dictionary_actual_bytes": len(trained.as_bytes()),
        f"{label}_dictionary_train_seconds": elapsed,
    }


def print_preflight(args: argparse.Namespace) -> None:
    deps = ["zstandard", "safetensors", "transformers", "huggingface_hub", "safetensors"]
    if not args.synthetic_data:
        deps.append("datasets")
    found = {name: importlib.util.find_spec(name) is not None for name in sorted(set(deps))}
    disk = shutil.disk_usage(".")
    print(f"baseline_name={BASELINE_NAME}")
    print(f"torch_version={torch.__version__}")
    print(f"zstandard_version={zstd.__version__}")
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
    print(f"zstd_level={args.zstd_level}")
    print(f"zstd_threads={args.zstd_threads}")
    print(f"dict100k_size={args.dict100k_size}")
    print(f"dict1m_size={args.dict1m_size}")
    print(f"dict_sample_chunk_size={args.dict_sample_chunk_size}")
    print(f"dict_train_max_bytes={args.dict_train_max_bytes or None}")
    print(f"synthetic_data={args.synthetic_data}")
    print(f"write_compressed_dir={args.write_compressed_dir}")
    print(f"hf_home={args.hf_home}")
    print(f"hf_hub_cache={args.hf_hub_cache}")
    print(f"hf_datasets_cache={args.hf_datasets_cache}")


def main() -> None:
    args = parse_args()
    if args.eval_probe_batches < 0:
        raise ValueError("eval-probe-batches must be non-negative")
    if args.checkpoints <= 0:
        raise ValueError("checkpoints must be positive")
    if args.steps_per_checkpoint <= 0:
        raise ValueError("steps-per-checkpoint must be positive")
    if args.dict100k_size <= 0 or args.dict1m_size <= 0:
        raise ValueError("dictionary sizes must be positive")

    configure_hf_cache(args)
    results_json = Path(args.results_json).resolve()
    results_csv = Path(args.results_csv).resolve()
    analysis_json = Path(args.analysis_json).resolve()
    for path in (results_json, results_csv, analysis_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    compressed_dir = Path(args.write_compressed_dir).resolve() if args.write_compressed_dir else None
    if compressed_dir is not None:
        compressed_dir.mkdir(parents=True, exist_ok=True)

    print_preflight(args)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this real SFT benchmark. Use --allow-cpu only for tiny dry runs.")
    deps = ["zstandard", "safetensors", "transformers", "huggingface_hub"]
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

    if args.synthetic_data:
        batches = synthetic_batch_iterator(args, model.config, device)
    else:
        batches = token_batch_iterator(args, tokenizer, load_dataset, device)
    probe_batches = [batch_to_cpu(next(batches)) for _ in range(args.eval_probe_batches)]

    total_steps = args.checkpoints * args.steps_per_checkpoint
    rows = []
    losses = []
    interval_losses = []
    cctx_none: Optional[zstd.ZstdCompressor] = None
    cctx_dict100k: Optional[zstd.ZstdCompressor] = None
    cctx_dict1m: Optional[zstd.ZstdCompressor] = None
    dictionary_info: Dict[str, Any] = {}
    cumulative_torch_bytes = 0
    safetensors_total = 0
    cumulative_zstd_none_bytes = 0
    cumulative_zstd_dict100k_bytes = 0
    cumulative_zstd_dict1m_bytes = 0

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
        state_dict = model.state_dict()
        torch_save_bytes, torch_save_seconds = torch_save_size_once(state_dict)
        cumulative_torch_bytes += torch_save_bytes

        safetensors_bytes, serialize_seconds = serialize_safetensors(state_dict)
        safetensors_size = len(safetensors_bytes)
        safetensors_total += safetensors_size

        if checkpoint_idx == 1:
            samples = dictionary_samples(
                safetensors_bytes,
                args.dict_sample_chunk_size,
                args.dict_train_max_bytes,
            )
            dict100k, dict100k_info = train_base_dictionary(samples, args.dict100k_size, "dict100k")
            dict1m, dict1m_info = train_base_dictionary(samples, args.dict1m_size, "dict1m")
            dictionary_info = {
                "dictionary_train_samples": len(samples),
                "dictionary_train_sample_bytes": sum(len(sample) for sample in samples),
                "dictionary_sample_chunk_size": args.dict_sample_chunk_size,
                "dictionary_train_max_bytes": args.dict_train_max_bytes or None,
                **dict100k_info,
                **dict1m_info,
            }
            cctx_none = zstd.ZstdCompressor(level=args.zstd_level, threads=args.zstd_threads)
            cctx_dict100k = zstd.ZstdCompressor(dict_data=dict100k, level=args.zstd_level, threads=args.zstd_threads)
            cctx_dict1m = zstd.ZstdCompressor(dict_data=dict1m, level=args.zstd_level, threads=args.zstd_threads)

        assert cctx_none is not None
        assert cctx_dict100k is not None
        assert cctx_dict1m is not None

        started = time.perf_counter()
        zstd_none = cctx_none.compress(safetensors_bytes)
        zstd_none_seconds = time.perf_counter() - started
        zstd_none_bytes = len(zstd_none)
        cumulative_zstd_none_bytes += zstd_none_bytes

        started = time.perf_counter()
        zstd_dict100k = cctx_dict100k.compress(safetensors_bytes)
        zstd_dict100k_seconds = time.perf_counter() - started
        zstd_dict100k_bytes = len(zstd_dict100k)
        cumulative_zstd_dict100k_bytes += zstd_dict100k_bytes

        started = time.perf_counter()
        zstd_dict1m = cctx_dict1m.compress(safetensors_bytes)
        zstd_dict1m_seconds = time.perf_counter() - started
        zstd_dict1m_bytes = len(zstd_dict1m)
        cumulative_zstd_dict1m_bytes += zstd_dict1m_bytes

        if compressed_dir is not None:
            stem = f"checkpoint_{checkpoint_idx:04d}.safetensors"
            (compressed_dir / f"{stem}.zstd-none.zst").write_bytes(zstd_none)
            (compressed_dir / f"{stem}.zstd-dict100k.zst").write_bytes(zstd_dict100k)
            (compressed_dir / f"{stem}.zstd-dict1m.zst").write_bytes(zstd_dict1m)

        row = {
            "checkpoint": checkpoint_idx,
            "step": step,
            "eval_probe_loss": evaluate_probe_loss(model, probe_batches, device),
            "loss": losses[-1],
            "loss_interval_mean": mean(interval_losses),
            "loss_interval_min": min(interval_losses),
            "loss_interval_max": max(interval_losses),
            "loss_interval_first": interval_losses[0],
            "loss_interval_last": interval_losses[-1],
            "loss_interval_delta": interval_losses[-1] - interval_losses[0],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "torch_save_bytes": torch_save_bytes,
            "torch_save_seconds": torch_save_seconds,
            "safetensors_bytes": safetensors_size,
            "serialize_seconds": serialize_seconds,
            "zstd_none_bytes": zstd_none_bytes,
            "zstd_none_ratio_pct": 100.0 * zstd_none_bytes / torch_save_bytes,
            "zstd_none_seconds": zstd_none_seconds,
            "zstd_dict100k_bytes": zstd_dict100k_bytes,
            "zstd_dict100k_ratio_pct": 100.0 * zstd_dict100k_bytes / torch_save_bytes,
            "zstd_dict100k_seconds": zstd_dict100k_seconds,
            "zstd_dict1m_bytes": zstd_dict1m_bytes,
            "zstd_dict1m_ratio_pct": 100.0 * zstd_dict1m_bytes / torch_save_bytes,
            "zstd_dict1m_seconds": zstd_dict1m_seconds,
            "cumulative_torch_bytes": cumulative_torch_bytes,
            "cumulative_safetensors_bytes": safetensors_total,
            "cumulative_zstd_none_bytes": cumulative_zstd_none_bytes,
            "cumulative_zstd_dict100k_bytes": cumulative_zstd_dict100k_bytes,
            "cumulative_zstd_dict1m_bytes": cumulative_zstd_dict1m_bytes,
        }
        rows.append(row)
        if not append_row(results_csv, row):
            print(f"warning: could not append checkpoint {checkpoint_idx} to {results_csv}; final CSV will be written at exit", flush=True)
        print(json.dumps(row, sort_keys=True), flush=True)
        interval_losses = []
        del safetensors_bytes
        del zstd_none
        del zstd_dict100k
        del zstd_dict1m
        gc.collect()

    assert dictionary_info
    serialization_times = [float(row["serialize_seconds"]) for row in rows]
    torch_save_times = [float(row["torch_save_seconds"]) for row in rows]
    safetensors_sizes = [int(row["safetensors_bytes"]) for row in rows]
    zstd_none_sizes = [int(row["zstd_none_bytes"]) for row in rows]
    zstd_dict100k_sizes = [int(row["zstd_dict100k_bytes"]) for row in rows]
    zstd_dict1m_sizes = [int(row["zstd_dict1m_bytes"]) for row in rows]
    zstd_none_times = [float(row["zstd_none_seconds"]) for row in rows]
    zstd_dict100k_times = [float(row["zstd_dict100k_seconds"]) for row in rows]
    zstd_dict1m_times = [float(row["zstd_dict1m_seconds"]) for row in rows]

    summary = {
        "baseline_name": BASELINE_NAME,
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
        "zstd_level": args.zstd_level,
        "zstd_threads": args.zstd_threads,
        "write_compressed_dir": str(compressed_dir) if compressed_dir else None,
        "compressed_payloads_written": compressed_dir is not None,
        **dictionary_info,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "avg_step_loss": mean(losses) if losses else None,
        "min_step_loss": min(losses) if losses else None,
        "max_step_loss": max(losses) if losses else None,
        "torch_total_bytes": cumulative_torch_bytes,
        "safetensors_total_bytes": safetensors_total,
        "zstd_none_total_bytes": cumulative_zstd_none_bytes,
        "zstd_dict100k_total_bytes": cumulative_zstd_dict100k_bytes,
        "zstd_dict1m_total_bytes": cumulative_zstd_dict1m_bytes,
        "zstd_none_vs_torch_ratio": cumulative_zstd_none_bytes / cumulative_torch_bytes,
        "zstd_dict100k_vs_torch_ratio": cumulative_zstd_dict100k_bytes / cumulative_torch_bytes,
        "zstd_dict1m_vs_torch_ratio": cumulative_zstd_dict1m_bytes / cumulative_torch_bytes,
        "zstd_none_space_saved_percent_vs_torch": 100.0 * (1.0 - (cumulative_zstd_none_bytes / cumulative_torch_bytes)),
        "zstd_dict100k_space_saved_percent_vs_torch": 100.0 * (1.0 - (cumulative_zstd_dict100k_bytes / cumulative_torch_bytes)),
        "zstd_dict1m_space_saved_percent_vs_torch": 100.0 * (1.0 - (cumulative_zstd_dict1m_bytes / cumulative_torch_bytes)),
        "zstd_none_vs_safetensors_ratio": cumulative_zstd_none_bytes / safetensors_total,
        "zstd_dict100k_vs_safetensors_ratio": cumulative_zstd_dict100k_bytes / safetensors_total,
        "zstd_dict1m_vs_safetensors_ratio": cumulative_zstd_dict1m_bytes / safetensors_total,
        "avg_torch_save_seconds": mean(torch_save_times),
        "median_torch_save_seconds": median(torch_save_times),
        "avg_safetensors_bytes": mean(safetensors_sizes),
        "median_safetensors_bytes": median(safetensors_sizes),
        "avg_serialize_seconds": mean(serialization_times),
        "median_serialize_seconds": median(serialization_times),
        "avg_zstd_none_bytes": mean(zstd_none_sizes),
        "median_zstd_none_bytes": median(zstd_none_sizes),
        "avg_zstd_dict100k_bytes": mean(zstd_dict100k_sizes),
        "median_zstd_dict100k_bytes": median(zstd_dict100k_sizes),
        "avg_zstd_dict1m_bytes": mean(zstd_dict1m_sizes),
        "median_zstd_dict1m_bytes": median(zstd_dict1m_sizes),
        "avg_zstd_none_seconds": mean(zstd_none_times),
        "median_zstd_none_seconds": median(zstd_none_times),
        "avg_zstd_dict100k_seconds": mean(zstd_dict100k_times),
        "median_zstd_dict100k_seconds": median(zstd_dict100k_times),
        "avg_zstd_dict1m_seconds": mean(zstd_dict1m_times),
        "median_zstd_dict1m_seconds": median(zstd_dict1m_times),
        "wall_seconds": time.perf_counter() - benchmark_started,
    }
    if not write_csv_with_retries(results_csv, rows):
        print(f"warning: could not rewrite final CSV at {results_csv}", flush=True)
    if not write_json_with_retries(results_json, {"summary": summary, "checkpoints": rows}):
        print(f"warning: could not write results JSON at {results_json}", flush=True)

    analysis = {
        "run": Path(results_json).parent.name,
        "baseline_name": BASELINE_NAME,
        "checkpoints": args.checkpoints,
        "steps": total_steps,
        "zstd_level": args.zstd_level,
        "zstd_threads": args.zstd_threads,
        "dict100k_actual_bytes": dictionary_info["dict100k_dictionary_actual_bytes"],
        "dict1m_actual_bytes": dictionary_info["dict1m_dictionary_actual_bytes"],
        "dict100k_train_seconds": round(dictionary_info["dict100k_dictionary_train_seconds"], 3),
        "dict1m_train_seconds": round(dictionary_info["dict1m_dictionary_train_seconds"], 3),
        "dictionary_train_samples": dictionary_info["dictionary_train_samples"],
        "torch_total_gb": round(cumulative_torch_bytes / 1e9, 3),
        "safetensors_total_gb": round(safetensors_total / 1e9, 3),
        "zstd_none_total_gb": round(cumulative_zstd_none_bytes / 1e9, 3),
        "zstd_dict100k_total_gb": round(cumulative_zstd_dict100k_bytes / 1e9, 3),
        "zstd_dict1m_total_gb": round(cumulative_zstd_dict1m_bytes / 1e9, 3),
        "zstd_none_vs_torch_pct": round(100.0 * summary["zstd_none_vs_torch_ratio"], 2),
        "zstd_dict100k_vs_torch_pct": round(100.0 * summary["zstd_dict100k_vs_torch_ratio"], 2),
        "zstd_dict1m_vs_torch_pct": round(100.0 * summary["zstd_dict1m_vs_torch_ratio"], 2),
        "zstd_none_space_saved_pct_vs_torch": round(summary["zstd_none_space_saved_percent_vs_torch"], 2),
        "zstd_dict100k_space_saved_pct_vs_torch": round(summary["zstd_dict100k_space_saved_percent_vs_torch"], 2),
        "zstd_dict1m_space_saved_pct_vs_torch": round(summary["zstd_dict1m_space_saved_percent_vs_torch"], 2),
        "avg_zstd_none_mb": round(mean(zstd_none_sizes) / 1e6, 3),
        "avg_zstd_dict100k_mb": round(mean(zstd_dict100k_sizes) / 1e6, 3),
        "avg_zstd_dict1m_mb": round(mean(zstd_dict1m_sizes) / 1e6, 3),
        "avg_serialize_seconds": round(mean(serialization_times), 3),
        "avg_zstd_none_seconds": round(mean(zstd_none_times), 3),
        "avg_zstd_dict100k_seconds": round(mean(zstd_dict100k_times), 3),
        "avg_zstd_dict1m_seconds": round(mean(zstd_dict1m_times), 3),
        "wall_minutes": round(summary["wall_seconds"] / 60, 2),
        "compressed_payloads_written": compressed_dir is not None,
        "eval_probe_first": rows[0]["eval_probe_loss"] if rows else None,
        "eval_probe_last": rows[-1]["eval_probe_loss"] if rows else None,
    }
    if not write_json_with_retries(analysis_json, analysis):
        print(f"warning: could not write analysis JSON at {analysis_json}", flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
