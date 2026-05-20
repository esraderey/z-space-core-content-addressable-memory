import tempfile
import unittest
from pathlib import Path

import torch

from tests.model_fixtures import build_2m_model, build_sparse_weight_patch
from z_space_core import DecompType, ZSpace


class ZSpaceModelVersioningTests(unittest.TestCase):
    def test_sparse_model_update_grows_less_than_second_checkpoint(self):
        model = build_2m_model()
        base_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        torch.manual_seed(11)
        sample = torch.randn(4, 1024)
        with torch.no_grad():
            base_output = model(sample)

        space = ZSpace(cache_size=96 << 20)
        names_by_key = {}
        base_descriptors = {}
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

        base_store_bytes = space.get_stats()["store"]["compressed_bytes"]

        modified_state = {key: value.clone() for key, value in base_state.items()}
        weight_key = "net.0.weight"
        indices, values = build_sparse_weight_patch(modified_state[weight_key], updates=4096)
        modified_state[weight_key][tuple(indices.T)] = values
        space.update(names_by_key[weight_key], {"type": "patch", "indices": indices, "values": values})

        bias_key = "net.6.bias"
        modified_state[bias_key] = modified_state[bias_key] + 0.001
        space.update(names_by_key[bias_key], {"type": "add", "value": 0.001})

        expected_modified_model = build_2m_model(seed=999)
        expected_modified_model.load_state_dict(modified_state)
        expected_modified_model.eval()
        with torch.no_grad():
            expected_modified_output = expected_modified_model(sample)

        after_update_stats = space.get_stats()
        z_version_growth = after_update_stats["store"]["compressed_bytes"] - base_store_bytes

        restored_modified = {key: space.load(z_name) for key, z_name in names_by_key.items()}
        for key, tensor in modified_state.items():
            self.assertTrue(torch.equal(restored_modified[key], tensor), key)

        restored_base = {key: space.load_desc(desc) for key, desc in base_descriptors.items()}
        for key, tensor in base_state.items():
            self.assertTrue(torch.equal(restored_base[key], tensor), key)

        base_clone = build_2m_model(seed=999)
        base_clone.load_state_dict(restored_base)
        modified_clone = build_2m_model(seed=999)
        modified_clone.load_state_dict(restored_modified)
        modified_clone.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(base_clone(sample), base_output))
            self.assertTrue(torch.equal(modified_clone(sample), expected_modified_output))

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base_path = tmpdir / "base.pt"
            modified_path = tmpdir / "modified.pt"
            torch.save(base_state, base_path)
            torch.save(modified_state, modified_path)
            torch_two_checkpoint_bytes = base_path.stat().st_size + modified_path.stat().st_size
            torch_second_checkpoint_bytes = modified_path.stat().st_size

        self.assertLess(z_version_growth, torch_second_checkpoint_bytes)
        self.assertLess(after_update_stats["store"]["compressed_bytes"], torch_two_checkpoint_bytes)
        self.assertEqual(after_update_stats["versions"], 2)


if __name__ == "__main__":
    unittest.main()
