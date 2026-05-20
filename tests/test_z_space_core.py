import tempfile
import unittest
from pathlib import Path

import z_space_core as z

try:
    import torch
except ImportError:  # pragma: no cover - local CI may not have torch
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ZSpaceCoreTests(unittest.TestCase):
    def test_svd_exact_round_trip_with_approx_load(self):
        left = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        right = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 17
        tensor = left @ right

        space = z.ZSpace(cache_size=1 << 20)
        desc = space.register(
            "matrix",
            tensor,
            target_ratio=0.5,
            exact=True,
            decomp_type=z.DecompType.SVD,
        )

        self.assertEqual(desc.decomp_type, z.DecompType.SVD)
        self.assertTrue(desc.exact)
        self.assertEqual(space.load("matrix", exact=False).shape, tensor.shape)
        self.assertTrue(torch.equal(space.load("matrix", exact=True), tensor))

    def test_sparse_round_trip(self):
        tensor = torch.zeros((8, 8), dtype=torch.float32)
        tensor[2, 3] = 7
        tensor[6, 1] = -2

        space = z.ZSpace(cache_size=1 << 20)
        desc = space.register("sparse", tensor, decomp_type=z.DecompType.SPARSE)

        self.assertEqual(desc.decomp_type, z.DecompType.SPARSE)
        self.assertTrue(torch.equal(space.load("sparse"), tensor))

    def test_patch_delta_and_revert(self):
        tensor = torch.zeros((4, 4), dtype=torch.float32)
        space = z.ZSpace(cache_size=1 << 20)
        space.register("grid", tensor)

        space.update("grid", {"type": "patch", "indices": [[1, 2]], "values": [9.0]})
        self.assertEqual(float(space.load("grid")[1, 2]), 9.0)
        self.assertEqual(
            space.get_stats()["store"]["node_kinds"].get(z.NodeKind.TENSOR_PAYLOAD.value),
            3,
        )

        space.revert_last_update("grid")
        self.assertEqual(float(space.load("grid")[1, 2]), 0.0)

    def test_raw_tensor_deduplicates_repeated_internal_blocks(self):
        block = torch.arange(z.DEFAULT_TENSOR_BLOCK_SIZE, dtype=torch.int64).remainder(251).to(torch.uint8)
        tensor = block.repeat(4)

        space = z.ZSpace(cache_size=1 << 20)
        desc = space.register(
            "blocked",
            tensor,
            decomp_type=z.DecompType.RAW,
            prefer_progressive=False,
        )

        stats = space.get_stats()["store"]
        self.assertEqual(stats["node_kinds"].get(z.NodeKind.RAW_TENSOR.value), 1)
        self.assertEqual(stats["node_kinds"].get(z.NodeKind.TENSOR_BLOCK.value), 1)
        self.assertTrue(torch.equal(space.load("blocked"), tensor))
        self.assertTrue(torch.equal(z.TensorCodec.unpack_tensor(space.gen.store.get(desc.raw_node)), tensor))

    def test_checkpoint_xor_zstd_delta_round_trip(self):
        base = {
            "w": torch.linspace(-2, 2, steps=8192, dtype=torch.float32).reshape(128, 64),
            "b": torch.arange(64, dtype=torch.float32) / 17,
        }
        updated = {
            "w": base["w"] + 0.001,
            "b": base["b"] - 0.25,
        }

        space = z.ZSpace(cache_size=1 << 20)
        base_desc = space.register_checkpoint("model", base)
        updated_desc = space.update_checkpoint("model", updated)
        restored_base = space.load_checkpoint_desc(base_desc)
        restored_updated = space.load_checkpoint("model")

        self.assertTrue(all(torch.equal(restored_base[key], value) for key, value in base.items()))
        self.assertTrue(all(torch.equal(restored_updated[key], value) for key, value in updated.items()))
        self.assertEqual(updated_desc.parent_addr, base_desc.address)
        self.assertEqual(space.get_stats()["checkpoint_versions"], 1)
        self.assertEqual(space.get_stats()["store"]["node_kinds"].get(z.NodeKind.TENSOR_DELTA.value), 2)

        raw_updated = sum(z.TensorCodec.raw_parts(tensor)[1].__len__() for tensor in updated.values())
        delta_bytes = sum(
            space.gen.store.compressed_size(node)
            for _, node in updated_desc.tensor_delta_nodes
        )
        self.assertLess(delta_bytes, raw_updated)

    def test_tensor_delta_codec_round_trip_with_small_chunks(self):
        base = torch.arange(16384, dtype=torch.int16)
        updated = (base + 17).to(torch.int16)

        payload = z.TensorDeltaCodec.pack_xor_zstd(base, updated, chunk_size=257)
        restored = z.TensorDeltaCodec.apply_xor_zstd(base, payload)

        self.assertTrue(torch.equal(restored, updated))

    def test_tensor_delta_codec_u16_sub_preconditioner_round_trip(self):
        base = (torch.arange(4096, dtype=torch.float16) / 128).reshape(128, 32)
        updated = (base + torch.tensor(0.125, dtype=torch.float16)).to(torch.float16)

        payload = z.TensorDeltaCodec.pack_xor_zstd(
            base,
            updated,
            chunk_size=512,
            preconditioner="u16-sub",
        )
        restored = z.TensorDeltaCodec.apply_xor_zstd(base, payload)

        self.assertTrue(torch.equal(restored, updated))

    def test_checkpoint_periodic_full_keeps_lineage_without_reconstruction_parent(self):
        states = [
            {"w": torch.arange(32, dtype=torch.float32)},
            {"w": torch.arange(32, dtype=torch.float32) + 1},
            {"w": torch.arange(32, dtype=torch.float32) + 2},
        ]

        space = z.ZSpace(cache_size=1 << 20)
        base_desc = space.register_checkpoint("model", states[0])
        delta_desc = space.update_checkpoint("model", states[1], full_every=2)
        full_desc = space.update_checkpoint("model", states[2], full_every=2, parent_state=states[1])

        self.assertEqual([desc.address for desc in space.checkpoint_history("model")], [
            base_desc.address,
            delta_desc.address,
            full_desc.address,
        ])
        self.assertTrue(full_desc.meta_view()["requires_parent"] is False)
        self.assertEqual(full_desc.parent_addr, delta_desc.address)
        self.assertEqual(len(full_desc.tensor_delta_nodes), 0)

        del space.checkpoint_index[delta_desc.address]
        restored = space.load_checkpoint_desc(full_desc)
        self.assertTrue(torch.equal(restored["w"], states[2]["w"]))

    def test_packfile_store_round_trip_without_per_node_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = z.PackfileContentStore(tmp)
            space = z.ZSpace(cache_size=1 << 20, store=store)
            tensor = torch.arange((z.DEFAULT_TENSOR_BLOCK_SIZE * 3) // 4, dtype=torch.int32)

            try:
                space.register("packed", tensor, decomp_type=z.DecompType.RAW, prefer_progressive=False)

                self.assertTrue(torch.equal(space.load("packed"), tensor))
                files = [path for path in Path(tmp).rglob("*") if path.is_file()]
                self.assertEqual(files, [Path(tmp) / "nodes.zspack"])
                self.assertGreater(space.get_stats()["store"]["node_kinds"].get(z.NodeKind.TENSOR_BLOCK.value, 0), 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
