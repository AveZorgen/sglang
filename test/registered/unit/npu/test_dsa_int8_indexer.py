import unittest

from sglang.srt.hardware_backend.npu.dsa_int8_indexer import (
    npu_indexer_bytes_per_token,
    parse_modelslim_int8_indexer_layers,
    resolve_npu_int8_indexer_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _indexer_key(layer_id: int, suffix: str = "quant_type") -> str:
    return f"model.layers.{layer_id}.self_attn.indexer.{suffix}"


def _resolve(**overrides):
    kwargs = dict(
        enable_int8_indexer=True,
        is_npu=True,
        index_head_dim=128,
        kv_cache_is_fp8=False,
        enable_hierarchical_cache=False,
        enable_hisparse=False,
        enable_sparse_kv_offload=False,
        is_draft_worker=False,
        candidate_layer_ids=tuple(range(8)),
        modelslim_quant_description=None,
    )
    kwargs.update(overrides)
    return resolve_npu_int8_indexer_layer_ids(**kwargs)


class TestParseModelSlimInt8IndexerLayers(unittest.TestCase):
    def test_only_int8_dynamic_layers_are_selected(self):
        description = {
            _indexer_key(1): "INT8_DYNAMIC",
            _indexer_key(2): "BF16",
            _indexer_key(5): "INT8_DYNAMIC",
        }
        self.assertEqual(
            parse_modelslim_int8_indexer_layers(description), frozenset({1, 5})
        )

    def test_no_per_layer_keys_means_all_layers(self):
        description = {"indexer_quant_type": "INT8_DYNAMIC"}
        self.assertIsNone(parse_modelslim_int8_indexer_layers(description))

    def test_weight_scheme_keys_alone_select_nothing(self):
        # Only the indexer's weight scheme is described per layer, so no layer
        # was calibrated for an INT8 cache.
        description = {
            "indexer_quant_type": "INT8_DYNAMIC",
            _indexer_key(3, "wq_b.weight"): "W8A8_DYNAMIC",
        }
        self.assertEqual(parse_modelslim_int8_indexer_layers(description), frozenset())

    def test_weight_scheme_keys_do_not_select_layers(self):
        # GLM-5.1 W4A8: wq_b is W8A8 on layers whose indexer cache stays bf16.
        description = {
            _indexer_key(4, "wq_b.weight"): "W8A8_DYNAMIC",
            _indexer_key(5): "INT8_DYNAMIC",
            _indexer_key(5, "wq_b.weight"): "W8A8_DYNAMIC",
            _indexer_key(78, "wq_b.weight"): "FLOAT",
        }
        self.assertEqual(
            parse_modelslim_int8_indexer_layers(description), frozenset({5})
        )

    def test_only_reverted_layers_select_nothing(self):
        self.assertEqual(
            parse_modelslim_int8_indexer_layers({_indexer_key(5): "BF16"}),
            frozenset(),
        )


class TestResolveNpuInt8IndexerLayerIds(unittest.TestCase):
    def test_disabled_returns_empty(self):
        self.assertEqual(
            _resolve(enable_int8_indexer=False, is_npu=False, index_head_dim=None),
            frozenset(),
        )

    def test_all_candidates_without_modelslim_metadata(self):
        self.assertEqual(_resolve(), frozenset(range(8)))

    def test_modelslim_filter_intersects_candidates(self):
        description = {_indexer_key(i): "INT8_DYNAMIC" for i in (2, 3, 9)}
        description[_indexer_key(4)] = "BF16"
        self.assertEqual(
            _resolve(modelslim_quant_description=description), frozenset({2, 3})
        )

    def test_draft_worker_stays_dense(self):
        description = {_indexer_key(78): "INT8_DYNAMIC"}
        self.assertEqual(
            _resolve(
                is_draft_worker=True,
                candidate_layer_ids=(78,),
                modelslim_quant_description=description,
            ),
            frozenset(),
        )

    def test_checkpoint_without_int8_layers_is_rejected(self):
        # Selecting nothing would disable the flag silently; the rank's own
        # candidates stay out of it so pipeline parallelism still works.
        description = {
            "indexer_quant_type": "W8A8_MXFP8",
            _indexer_key(2, "wq_b.weight"): "W8A8_DYNAMIC",
        }
        with self.assertRaises(ValueError):
            _resolve(modelslim_quant_description=description)

    def test_rejected_configurations(self):
        cases = {
            "not npu": dict(is_npu=False),
            "not dsa": dict(index_head_dim=None),
            "head dim": dict(index_head_dim=64),
            "fp8 kv": dict(kv_cache_is_fp8=True),
            "hicache": dict(enable_hierarchical_cache=True),
            "hisparse": dict(enable_hisparse=True),
            "sparse offload": dict(enable_sparse_kv_offload=True),
        }
        for name, overrides in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                _resolve(**overrides)


class TestNpuIndexerBytesPerToken(unittest.TestCase):
    def test_glm51_mixed_layers(self):
        # 78 target indexer layers, 69 of them INT8 (layers 5-73).
        self.assertEqual(
            npu_indexer_bytes_per_token(
                num_indexer_layers=78,
                num_int8_layers=69,
                index_head_dim=128,
                dense_element_size=2,
            ),
            9 * 256 + 69 * 130,
        )


if __name__ == "__main__":
    unittest.main()
