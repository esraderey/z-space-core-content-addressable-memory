import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.model_fixtures import build_2m_model, build_sparse_weight_patch, count_parameters
from z_space_core import DecompType, ZSpace


def main() -> None:
    model = build_2m_model()
    base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    torch.manual_seed(11)
    sample = torch.randn(4, 1024)
    with torch.no_grad():
        base_output = model(sample)

    space = ZSpace(cache_size=96 << 20)
    names_by_key = {}
    base_descriptors = {}

    started = time.perf_counter()
    for key, tensor in base_state.items():
        z_name = f"model_2m::{key}"
        base_descriptors[key] = space.register(
            z_name,
            tensor,
            exact=True,
            decomp_type=DecompType.RAW,
            prefer_progressive=False,
        )
        names_by_key[key] = z_name
    base_register_seconds = time.perf_counter() - started
    base_store_bytes = space.get_stats()["store"]["compressed_bytes"]

    modified_state = {key: value.clone() for key, value in base_state.items()}
    weight_key = "net.0.weight"
    indices, values = build_sparse_weight_patch(modified_state[weight_key], updates=4096)
    modified_state[weight_key][tuple(indices.T)] = values

    started = time.perf_counter()
    space.update(names_by_key[weight_key], {"type": "patch", "indices": indices, "values": values})

    bias_key = "net.6.bias"
    modified_state[bias_key] = modified_state[bias_key] + 0.001
    space.update(names_by_key[bias_key], {"type": "add", "value": 0.001})
    update_seconds = time.perf_counter() - started

    after_update_stats = space.get_stats()
    z_total_bytes = after_update_stats["store"]["compressed_bytes"]
    z_version_growth = z_total_bytes - base_store_bytes

    started = time.perf_counter()
    restored_modified = {key: space.load(z_name) for key, z_name in names_by_key.items()}
    modified_load_seconds = time.perf_counter() - started

    restored_base = {key: space.load_desc(desc) for key, desc in base_descriptors.items()}
    tensors_exact_modified = all(torch.equal(restored_modified[key], tensor) for key, tensor in modified_state.items())
    tensors_exact_base = all(torch.equal(restored_base[key], tensor) for key, tensor in base_state.items())

    base_clone = build_2m_model(seed=999)
    base_clone.load_state_dict(restored_base)
    modified_clone = build_2m_model(seed=999)
    modified_clone.load_state_dict(restored_modified)

    expected_modified_model = build_2m_model(seed=999)
    expected_modified_model.load_state_dict(modified_state)
    with torch.no_grad():
        base_output_exact = torch.equal(base_clone(sample), base_output)
        modified_output_exact = torch.equal(modified_clone(sample), expected_modified_model(sample))

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        base_path = tmpdir / "base.pt"
        modified_path = tmpdir / "modified.pt"
        torch.save(base_state, base_path)
        torch.save(modified_state, modified_path)
        torch_base_bytes = base_path.stat().st_size
        torch_modified_bytes = modified_path.stat().st_size

    torch_two_checkpoint_bytes = torch_base_bytes + torch_modified_bytes

    print(f"parameters={count_parameters(model)}")
    print(f"modified_tensors=2")
    print(f"sparse_weight_updates={indices.shape[0]}")
    print(f"z_base_store_bytes={base_store_bytes}")
    print(f"z_version_growth_bytes={z_version_growth}")
    print(f"z_total_base_plus_version_bytes={z_total_bytes}")
    print(f"torch_base_checkpoint_bytes={torch_base_bytes}")
    print(f"torch_modified_checkpoint_bytes={torch_modified_bytes}")
    print(f"torch_two_checkpoint_bytes={torch_two_checkpoint_bytes}")
    print(f"z_growth_vs_second_checkpoint={z_version_growth / torch_modified_bytes:.6f}")
    print(f"z_total_vs_two_checkpoints={z_total_bytes / torch_two_checkpoint_bytes:.6f}")
    print(f"base_register_seconds={base_register_seconds:.4f}")
    print(f"update_seconds={update_seconds:.4f}")
    print(f"modified_load_seconds={modified_load_seconds:.4f}")
    print(f"base_tensors_exact={tensors_exact_base}")
    print(f"modified_tensors_exact={tensors_exact_modified}")
    print(f"base_output_exact={base_output_exact}")
    print(f"modified_output_exact={modified_output_exact}")
    print(f"versions={after_update_stats['versions']}")
    print(f"store_nodes={after_update_stats['store']['nodes']}")

    if not (
        tensors_exact_base
        and tensors_exact_modified
        and base_output_exact
        and modified_output_exact
        and z_version_growth < torch_modified_bytes
        and z_total_bytes < torch_two_checkpoint_bytes
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
