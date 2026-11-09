"""CPU regressions for NPU DSA indexer pool layer ownership."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestNPUIndexerLayerLayout(CustomTestCase):
    def test_disabled_int8_indexer_needs_no_model_or_runtime_configuration(self):
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        with (
            patch.object(kcc.envs.SGLANG_NPU_INDEXER_INT8, "get", return_value=False),
            patch.object(kcc, "get_memory", side_effect=AssertionError("get_memory")),
            patch.object(kcc, "get_disagg", side_effect=AssertionError("get_disagg")),
            patch.object(
                kcc, "get_parallel", side_effect=AssertionError("get_parallel")
            ),
        ):
            result = configurator.npu_int8_indexer_layer_ids(
                candidate_layer_ids=(0, 1, 2), index_head_dim=128
            )
        self.assertEqual(result, frozenset())

    def _resolve_int8_layers(self, description=None, **disagg_overrides):
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.model = SimpleNamespace(
            quant_config=SimpleNamespace(quant_description=description)
        )
        configurator.model_config = SimpleNamespace(
            model_path="org/remote-checkpoint",
            _find_quant_modelslim_config=Mock(return_value=None),
        )
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.is_draft_worker = False
        configurator.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        configurator.pp_size = 1
        disagg = dict(
            disaggregation_mode="decode",
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_retraction_backup="none",
        )
        disagg.update(disagg_overrides)
        with (
            patch.object(kcc.envs.SGLANG_NPU_INDEXER_INT8, "get", return_value=True),
            patch.object(kcc, "_is_npu", True),
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(
                    enable_hierarchical_cache=False, enable_hisparse=False
                ),
            ),
            patch.object(kcc, "get_disagg", return_value=SimpleNamespace(**disagg)),
            patch.object(
                kcc, "get_parallel", return_value=SimpleNamespace(attn_cp_size=1)
            ),
            patch.object(
                kcc.envs.SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD, "get", return_value=False
            ),
        ):
            result = configurator.npu_int8_indexer_layer_ids(
                candidate_layer_ids=(0, 1, 2), index_head_dim=128
            )
        return result, configurator.model_config._find_quant_modelslim_config

    def test_remote_checkpoint_uses_loaded_modelslim_metadata(self):
        result, local_lookup = self._resolve_int8_layers(
            {
                "model.layers.0.self_attn.indexer.quant_type": "BF16",
                "model.layers.1.self_attn.indexer.quant_type": "INT8_DYNAMIC",
                "model.layers.2.self_attn.indexer.quant_type": "BF16",
            }
        )
        self.assertEqual(result, frozenset({1}))
        local_lookup.assert_not_called()

    def test_missing_loaded_metadata_falls_back_to_local_lookup(self):
        result, local_lookup = self._resolve_int8_layers()
        self.assertEqual(result, frozenset({0, 1, 2}))
        local_lookup.assert_called_once_with()

    def test_decode_host_cache_options_are_rejected_without_hicache(self):
        for options, message in (
            (
                {"disaggregation_decode_enable_offload_kvcache": True},
                "--disaggregation-decode-enable-offload-kvcache",
            ),
            (
                {"disaggregation_decode_retraction_backup": "host_pool"},
                "--disaggregation-decode-retraction-backup=host_pool",
            ),
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, message),
            ):
                self._resolve_int8_layers(**options)

    def test_non_arch35_dsa_pool_uses_only_active_indexer_layers(self):
        """Shared layers must not consume indexer slots on non-arch35 NPU."""
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.device = "cpu"
        configurator.is_draft_worker = False
        configurator.kv_cache_dtype = torch.bfloat16
        configurator.layer_info = SimpleNamespace(
            start_layer=10,
            end_layer=14,
            num_effective_layers=4,
        )
        configurator.model_config = SimpleNamespace(
            hf_config=object(),
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            index_head_dim=32,
        )
        pool = object()

        with (
            patch.object(kcc.envs.SGLANG_NPU_INDEXER_INT8, "get", return_value=False),
            patch.object(
                kcc,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=2),
            ),
            patch.object(
                kcc,
                "get_schedule",
                return_value=SimpleNamespace(page_size=16),
            ),
            patch.object(
                kcc,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch.object(kcc, "_should_elide_dsa_index_k", return_value=True),
            patch.object(
                kcc,
                "dsa_layer_skips_topk",
                side_effect=lambda _config, layer_id: layer_id in (11, 13),
            ),
            patch(
                "sglang.srt.hardware_backend.npu.utils.is_npu_arch35",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.NPUMLATokenToKVPool",
                return_value=pool,
            ) as pool_class,
        ):
            result = configurator._build_ascend_mla_kv_pool(
                max_total_num_tokens=64,
                is_dsa_model=True,
            )

        self.assertIs(result, pool)
        self.assertEqual(pool_class.call_args.args, (64,))
        self.assertEqual(pool_class.call_args.kwargs["index_size"], 128)
        self.assertEqual(pool_class.call_args.kwargs["indexer_layer_ids"], (10, 12))
        self.assertEqual(pool_class.call_args.kwargs["int8_indexer_layer_ids"], ())


if __name__ == "__main__":
    import unittest

    unittest.main()
