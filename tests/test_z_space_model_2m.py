import unittest

import torch

from tests.model_fixtures import build_2m_model, count_parameters
from z_space_core import DecompType, ZSpace


class ZSpaceTwoMillionParamModelTests(unittest.TestCase):
    def test_2m_parameter_model_state_dict_round_trip(self):
        model = build_2m_model()
        param_count = count_parameters(model)
        self.assertGreaterEqual(param_count, 1_950_000)
        self.assertLessEqual(param_count, 2_050_000)

        torch.manual_seed(7)
        sample = torch.randn(4, 1024)
        with torch.no_grad():
            expected_output = model(sample)

        space = ZSpace(cache_size=64 << 20)
        names_by_key = {}
        for key, tensor in model.state_dict().items():
            z_name = f"model_2m::{key}"
            desc = space.register(
                z_name,
                tensor,
                exact=True,
                decomp_type=DecompType.RAW,
                prefer_progressive=False,
            )
            self.assertTrue(desc.exact)
            self.assertEqual(desc.decomp_type, DecompType.RAW)
            names_by_key[key] = z_name

        restored_state = {key: space.load(z_name) for key, z_name in names_by_key.items()}
        for key, original in model.state_dict().items():
            self.assertTrue(torch.equal(restored_state[key], original), key)

        restored_model = build_2m_model(seed=999)
        restored_model.load_state_dict(restored_state)
        restored_model.eval()
        with torch.no_grad():
            restored_output = restored_model(sample)

        self.assertTrue(torch.equal(restored_output, expected_output))

        first_name = next(iter(names_by_key.values()))
        mutated_load = space.load(first_name)
        mutated_load.reshape(-1)[0] += 12345
        self.assertFalse(torch.equal(mutated_load, space.load(first_name)))


if __name__ == "__main__":
    unittest.main()
