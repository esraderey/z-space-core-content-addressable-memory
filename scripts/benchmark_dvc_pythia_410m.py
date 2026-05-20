import argparse
import csv
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

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


BASELINE_NAME = "DVC content-addressable storage baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DVC as a content-addressable baseline for real Pythia-410M "
            "SFT checkpoints."
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
    parser.add_argument("--dvc-root", default=".zspace_bench/dvc_baseline/repo")
    parser.add_argument("--results-json", default=".zspace_bench/dvc_baseline/results.json")
    parser.add_argument("--results-csv", default=".zspace_bench/dvc_baseline/checkpoints.csv")
    parser.add_argument("--clear-dvc-root", action="store_true")
    parser.add_argument("--keep-workspace-checkpoints", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def dvc_env(dvc_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["DVC_NO_ANALYTICS"] = "1"
    env["DVC_STUDIO_OFFLINE"] = "1"
    env["DVC_SYSTEM_CONFIG_DIR"] = str(dvc_root / ".dvc_system_config")
    env["DVC_GLOBAL_CONFIG_DIR"] = str(dvc_root / ".dvc_global_config")
    env["DVC_SITE_CACHE_DIR"] = str(dvc_root / ".dvc_site_cache")
    return env


def run_dvc(dvc_root: Path, env: Mapping[str, str], args: Sequence[str], *, check: bool = True) -> Tuple[float, str, str, int]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "dvc", "--cd", str(dvc_root), *args],
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if check and proc.returncode != 0:
        joined = " ".join(args)
        raise RuntimeError(
            f"dvc {joined} failed with exit code {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return elapsed, proc.stdout, proc.stderr, proc.returncode


def dvc_version(env: Mapping[str, str]) -> str:
    proc = subprocess.run(
        [sys.executable, "-m", "dvc", "--version"],
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return f"unknown: {proc.stderr.strip()}"
    return proc.stdout.strip()


def directory_stats(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            total += item.stat().st_size
    return total, count


def save_torch_checkpoint(state_dict: Mapping[str, torch.Tensor], path: Path) -> Tuple[int, float]:
    started = time.perf_counter()
    cpu_state = {key: tensor.detach().cpu() for key, tensor in state_dict.items()}
    torch.save(cpu_state, path)
    elapsed = time.perf_counter() - started
    return path.stat().st_size, elapsed


def print_preflight(args: argparse.Namespace, dvc_root: Path, dvc_cache: Path, env: Mapping[str, str]) -> None:
    deps = ["dvc", "transformers", "huggingface_hub", "safetensors"]
    if not args.synthetic_data:
        deps.append("datasets")
    found = {name: importlib.util.find_spec(name) is not None for name in deps}
    disk = shutil.disk_usage(".")
    print(f"baseline_name={BASELINE_NAME}")
    print(f"dvc_version={dvc_version(env)}")
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
    print(f"synthetic_data={args.synthetic_data}")
    print(f"keep_workspace_checkpoints={args.keep_workspace_checkpoints}")
    print(f"dvc_root={dvc_root}")
    print(f"dvc_cache={dvc_cache}")
    print(f"hf_home={args.hf_home}")
    print(f"hf_hub_cache={args.hf_hub_cache}")
    print(f"hf_datasets_cache={args.hf_datasets_cache}")


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    fieldnames = list(row.keys())
    for attempt in range(10):
        try:
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.25 * (attempt + 1))


def main() -> None:
    args = parse_args()
    if args.eval_probe_batches < 0:
        raise ValueError("eval-probe-batches must be non-negative")
    if args.checkpoints <= 0:
        raise ValueError("checkpoints must be positive")
    if args.steps_per_checkpoint <= 0:
        raise ValueError("steps-per-checkpoint must be positive")

    configure_hf_cache(args)
    dvc_root = Path(args.dvc_root).resolve()
    if args.clear_dvc_root and dvc_root.exists():
        shutil.rmtree(dvc_root)
    dvc_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = dvc_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_json = Path(args.results_json).resolve()
    results_csv = Path(args.results_csv).resolve()
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    env = dvc_env(dvc_root)
    dvc_cache = dvc_root / ".dvc" / "cache"
    print_preflight(args, dvc_root, dvc_cache, env)
    if args.preflight_only:
        return
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for this real SFT benchmark. Use --allow-cpu only for tiny dry runs.")
    deps = ["dvc", "transformers", "huggingface_hub", "safetensors"]
    if not args.synthetic_data:
        deps.append("datasets")
    require_modules(deps)

    if not (dvc_root / ".dvc").exists():
        run_dvc(dvc_root, env, ["init", "--no-scm", "--quiet"])
    run_dvc(dvc_root, env, ["config", "core.analytics", "false", "--local"], check=False)

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
    torch_checkpoint_total = 0
    previous_cache_bytes, previous_cache_files = directory_stats(dvc_cache)

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
        checkpoint_rel = Path("checkpoints") / f"checkpoint_{checkpoint_idx:04d}.pt"
        checkpoint_path = dvc_root / checkpoint_rel
        torch_checkpoint_bytes, torch_save_seconds = save_torch_checkpoint(model.state_dict(), checkpoint_path)
        torch_checkpoint_total += torch_checkpoint_bytes

        add_seconds, add_stdout, add_stderr, add_returncode = run_dvc(
            dvc_root,
            env,
            ["add", checkpoint_rel.as_posix(), "--quiet"],
        )
        commit_seconds, commit_stdout, commit_stderr, commit_returncode = run_dvc(
            dvc_root,
            env,
            ["commit", f"{checkpoint_rel.as_posix()}.dvc", "--quiet"],
        )

        cache_bytes, cache_files = directory_stats(dvc_cache)
        cache_growth_bytes = cache_bytes - previous_cache_bytes
        cache_file_growth = cache_files - previous_cache_files
        previous_cache_bytes = cache_bytes
        previous_cache_files = cache_files

        if not args.keep_workspace_checkpoints:
            checkpoint_path.unlink()

        row = {
            "checkpoint": checkpoint_idx,
            "step": step,
            "loss": losses[-1],
            "loss_interval_mean": mean(interval_losses),
            "loss_interval_min": min(interval_losses),
            "loss_interval_max": max(interval_losses),
            "loss_interval_first": interval_losses[0],
            "loss_interval_last": interval_losses[-1],
            "loss_interval_delta": interval_losses[-1] - interval_losses[0],
            "eval_probe_loss": evaluate_probe_loss(model, probe_batches, device),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "torch_checkpoint_bytes": torch_checkpoint_bytes,
            "torch_save_seconds": torch_save_seconds,
            "dvc_add_seconds": add_seconds,
            "dvc_commit_seconds": commit_seconds,
            "dvc_version_seconds": add_seconds + commit_seconds,
            "checkpoint_pipeline_seconds": torch_save_seconds + add_seconds + commit_seconds,
            "dvc_cache_growth_bytes": cache_growth_bytes,
            "dvc_cache_bytes": cache_bytes,
            "dvc_cache_files": cache_files,
            "dvc_cache_file_growth": cache_file_growth,
            "workspace_checkpoint_retained": args.keep_workspace_checkpoints,
            "dvc_add_returncode": add_returncode,
            "dvc_commit_returncode": commit_returncode,
            "dvc_add_stdout": add_stdout.strip(),
            "dvc_add_stderr": add_stderr.strip(),
            "dvc_commit_stdout": commit_stdout.strip(),
            "dvc_commit_stderr": commit_stderr.strip(),
        }
        rows.append(row)
        append_row(results_csv, row)
        print(json.dumps(row, sort_keys=True), flush=True)
        interval_losses = []
        gc.collect()

    final_rel = Path("checkpoints") / f"checkpoint_{args.checkpoints:04d}.pt"
    checkout_seconds, checkout_stdout, checkout_stderr, checkout_returncode = run_dvc(
        dvc_root,
        env,
        ["checkout", f"{final_rel.as_posix()}.dvc", "--quiet"],
    )
    final_checkout_path = dvc_root / final_rel
    final_checkout_bytes = final_checkout_path.stat().st_size if final_checkout_path.exists() else None
    if final_checkout_path.exists() and not args.keep_workspace_checkpoints:
        final_checkout_path.unlink()

    cache_bytes, cache_files = directory_stats(dvc_cache)
    growths = [int(row["dvc_cache_growth_bytes"]) for row in rows]
    add_times = [float(row["dvc_add_seconds"]) for row in rows]
    commit_times = [float(row["dvc_commit_seconds"]) for row in rows]
    version_times = [float(row["dvc_version_seconds"]) for row in rows]
    save_times = [float(row["torch_save_seconds"]) for row in rows]
    pipeline_times = [float(row["checkpoint_pipeline_seconds"]) for row in rows]

    summary = {
        "baseline_name": BASELINE_NAME,
        "dvc_version": dvc_version(env),
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
        "keep_workspace_checkpoints": args.keep_workspace_checkpoints,
        "dvc_root": str(dvc_root),
        "dvc_cache": str(dvc_cache),
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "avg_step_loss": mean(losses) if losses else None,
        "min_step_loss": min(losses) if losses else None,
        "max_step_loss": max(losses) if losses else None,
        "final_dvc_cache_bytes": cache_bytes,
        "final_dvc_cache_files": cache_files,
        "torch_checkpoint_total_bytes": torch_checkpoint_total,
        "dvc_cache_to_torch_ratio": cache_bytes / torch_checkpoint_total if torch_checkpoint_total else None,
        "dvc_space_saved_percent_vs_torch": (
            100.0 * (1.0 - (cache_bytes / torch_checkpoint_total)) if torch_checkpoint_total else None
        ),
        "avg_cache_growth_bytes": mean(growths),
        "median_cache_growth_bytes": median(growths),
        "min_cache_growth_bytes": min(growths),
        "max_cache_growth_bytes": max(growths),
        "avg_torch_save_seconds": mean(save_times),
        "median_torch_save_seconds": median(save_times),
        "avg_dvc_add_seconds": mean(add_times),
        "median_dvc_add_seconds": median(add_times),
        "avg_dvc_commit_seconds": mean(commit_times),
        "median_dvc_commit_seconds": median(commit_times),
        "avg_dvc_version_seconds": mean(version_times),
        "median_dvc_version_seconds": median(version_times),
        "avg_checkpoint_pipeline_seconds": mean(pipeline_times),
        "median_checkpoint_pipeline_seconds": median(pipeline_times),
        "final_checkout_seconds": checkout_seconds,
        "final_checkout_returncode": checkout_returncode,
        "final_checkout_bytes": final_checkout_bytes,
        "final_checkout_size_matches": final_checkout_bytes == rows[-1]["torch_checkpoint_bytes"] if rows else None,
        "final_checkout_stdout": checkout_stdout.strip(),
        "final_checkout_stderr": checkout_stderr.strip(),
        "wall_seconds": time.perf_counter() - benchmark_started,
    }
    results_json.write_text(json.dumps({"summary": summary, "checkpoints": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["final_checkout_size_matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
