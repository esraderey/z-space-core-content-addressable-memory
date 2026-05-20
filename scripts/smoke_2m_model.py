import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.model_fixtures import build_2m_model, count_parameters
from z_space_core import DecompType, ZSpace


def main() -> None:
    model = build_2m_model()
    param_count = count_parameters(model)
    tensor_count = len(model.state_dict())
    raw_bytes = sum(t.nelement() * t.element_size() for t in model.state_dict().values())

    torch.manual_seed(7)
    sample = torch.randn(4, 1024)
    with torch.no_grad():
        expected_output = model(sample)

    space = ZSpace(cache_size=64 << 20)
    names_by_key = {}

    started = time.perf_counter()
    for key, tensor in model.state_dict().items():
        z_name = f"model_2m::{key}"
        space.register(
            z_name,
            tensor,
            exact=True,
            decomp_type=DecompType.RAW,
            prefer_progressive=False,
        )
        names_by_key[key] = z_name
    register_seconds = time.perf_counter() - started

    started = time.perf_counter()
    restored_state = {key: space.load(z_name) for key, z_name in names_by_key.items()}
    load_seconds = time.perf_counter() - started

    tensors_exact = all(torch.equal(restored_state[key], tensor) for key, tensor in model.state_dict().items())

    restored_model = build_2m_model(seed=999)
    restored_model.load_state_dict(restored_state)
    restored_model.eval()
    with torch.no_grad():
        restored_output = restored_model(sample)

    output_exact = torch.equal(restored_output, expected_output)

    first_name = next(iter(names_by_key.values()))
    mutated_load = space.load(first_name)
    mutated_load.reshape(-1)[0] += 12345
    cache_immutable = not torch.equal(mutated_load, space.load(first_name))

    stats = space.get_stats()
    print(f"parameters={param_count}")
    print(f"state_tensors={tensor_count}")
    print(f"raw_bytes={raw_bytes}")
    print(f"register_seconds={register_seconds:.4f}")
    print(f"load_seconds={load_seconds:.4f}")
    print(f"tensors_exact={tensors_exact}")
    print(f"output_exact={output_exact}")
    print(f"cache_immutable={cache_immutable}")
    print(f"store_nodes={stats['store']['nodes']}")
    print(f"store_compressed_bytes={stats['store']['compressed_bytes']}")
    print(f"cache_stats={stats['cache']}")

    if not (tensors_exact and output_exact and cache_immutable):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
