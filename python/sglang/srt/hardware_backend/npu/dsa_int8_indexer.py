"""Layer selection and validation for the NPU INT8 DSA Lightning Indexer cache.

With ``SGLANG_NPU_ENABLE_INT8_DSA_INDEXER=1`` the indexer K cache of selected
layers is stored as int8 with one fp16 dequant scale per token, and top-k
selection runs through ``npu_quant_lightning_indexer``. Other indexer layers
stay bf16.

ModelSlim checkpoints mark the layers calibrated for this with
``model.layers.N.self_attn.indexer.quant_type: "INT8_DYNAMIC"``; a checkpoint
without per-layer keys quantizes every indexer layer.

The top-level ``indexer_quant_type`` of quant_model_description.json is not a
switch: it describes what the checkpoint was calibrated for, while the cache
layout is chosen by the user. vllm-ascend reads it the same way -- its
``enable_sparse_li_c8`` comes from ``indexer_kv_dtype`` alone, and the
top-level key only gates loading the (unused) ``indexer.q_rot``/``k_rot``
rotations. Here it is used to warn when a checkpoint describes no indexer
quantization at all.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

INT8_INDEXER_QUANT_TYPE = "INT8_DYNAMIC"
_FLAG = "SGLANG_NPU_ENABLE_INT8_DSA_INDEXER"
GLOBAL_INDEXER_QUANT_TYPE_KEY = "indexer_quant_type"
_INDEXER_QUANT_TYPE_SUFFIX = ".indexer.quant_type"
# ``.indexer.wq_b.weight`` describes the indexer's weight scheme, so its value
# never selects a layer: W8A8 checkpoints set it on layers whose indexer cache
# ModelSlim kept in bf16. Its presence still marks the description as
# per-layer, so a checkpoint listing only these keys quantizes nothing rather
# than everything (vllm-ascend AscendConfig._has_sparse_li_layer_config).
_INDEXER_PER_LAYER_SUFFIXES = (_INDEXER_QUANT_TYPE_SUFFIX, ".indexer.wq_b.weight")
_LAYER_ID_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")

# One fp16 dequant scale per token; npu_quant_lightning_indexer on A2/A3
# takes fp16 scales for int8 inputs.
INT8_INDEXER_SCALE_BYTES = 2


def parse_modelslim_int8_indexer_layers(
    quant_description: Mapping[str, Any],
) -> Optional[frozenset[int]]:
    """Layers whose indexer cache ModelSlim calibrated for INT8.

    Returns None when the description has no per-layer indexer keys, meaning
    every indexer layer is eligible.
    """
    layer_ids = set()
    has_per_layer_keys = False
    for key, value in quant_description.items():
        if not isinstance(key, str):
            continue
        suffix = next((s for s in _INDEXER_PER_LAYER_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            continue
        match = _LAYER_ID_RE.search(key)
        if match is None:
            continue
        has_per_layer_keys = True
        if suffix == _INDEXER_QUANT_TYPE_SUFFIX and value == INT8_INDEXER_QUANT_TYPE:
            layer_ids.add(int(match.group(1)))
    return frozenset(layer_ids) if has_per_layer_keys else None


def npu_indexer_checkpoint_layer_offset(hf_config) -> int:
    """NextN pool layer 0 represents the checkpoint's first MTP layer."""
    if set(getattr(hf_config, "architectures", ()) or ()) & {
        "DeepseekV3ForCausalLMNextN",
        "GlmMoeDsaForCausalLMNextN",
        "Glm5NextForConditionalGenerationNextN",
    }:
        return getattr(hf_config, "text_config", hf_config).num_hidden_layers
    return 0


def resolve_npu_int8_indexer_layer_ids(
    *,
    enable_int8_indexer: bool,
    is_npu: bool,
    index_head_dim: Optional[int],
    kv_cache_is_fp8: bool,
    enable_hierarchical_cache: bool,
    enable_hisparse: bool,
    enable_sparse_kv_offload: bool,
    candidate_layer_ids: tuple[int, ...],
    modelslim_quant_description: Optional[Mapping[str, Any]],
    checkpoint_layer_offset: int = 0,
) -> frozenset[int]:
    """Indexer layers on this rank whose K cache is stored as INT8.

    ``candidate_layer_ids`` are the indexer layers the pool allocates on this
    rank. ``checkpoint_layer_offset`` maps pool IDs to checkpoint IDs for
    NextN/MTP workers; selection is returned in the pool's ID space.
    """
    if not enable_int8_indexer:
        return frozenset()

    if not is_npu:
        raise ValueError(f"{_FLAG} is only supported on Ascend NPU.")
    if index_head_dim is None:
        raise ValueError(f"{_FLAG} requires a DSA model.")
    if index_head_dim != 128:
        raise ValueError(
            f"{_FLAG} requires index_head_dim == 128, got {index_head_dim}."
        )
    unsupported = [
        name
        for name, enabled in (
            ("an FP8 KV cache", kv_cache_is_fp8),
            ("--enable-hierarchical-cache", enable_hierarchical_cache),
            ("--enable-hisparse", enable_hisparse),
            ("SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD", enable_sparse_kv_offload),
        )
        if enabled
    ]
    if unsupported:
        raise ValueError(
            f"{_FLAG} is not supported together with " + ", ".join(unsupported) + "."
        )

    if modelslim_quant_description is None:
        logger.warning(
            "%s is set for a model without a ModelSlim quantization "
            "description. Every indexer layer is quantized without "
            "calibration; accuracy may drop.",
            _FLAG,
        )
        return frozenset(candidate_layer_ids)

    selected = parse_modelslim_int8_indexer_layers(modelslim_quant_description)
    global_quant_type = modelslim_quant_description.get(
        GLOBAL_INDEXER_QUANT_TYPE_KEY, ""
    )
    if selected is None:
        if global_quant_type != INT8_INDEXER_QUANT_TYPE:
            logger.warning(
                "%s is set, but this checkpoint describes no indexer "
                "quantization (%s=%r, no per-layer indexer keys). Every "
                "indexer layer is quantized without calibration; accuracy "
                "may drop.",
                _FLAG,
                GLOBAL_INDEXER_QUANT_TYPE_KEY,
                global_quant_type,
            )
        return frozenset(candidate_layer_ids)
    if not selected:
        # Checked against the whole filter, not this rank's candidates: with
        # pipeline parallelism a rank may legitimately own no INT8 layer.
        raise ValueError(
            f"{_FLAG} is set, but this checkpoint marks no indexer layer "
            f"{INT8_INDEXER_QUANT_TYPE} "
            f"({GLOBAL_INDEXER_QUANT_TYPE_KEY}={global_quant_type!r}). Unset "
            f"{_FLAG}, or use a checkpoint calibrated for an INT8 indexer cache."
        )
    return frozenset(
        i for i in candidate_layer_ids if i + checkpoint_layer_offset in selected
    )


def npu_indexer_bytes_per_token(
    *,
    num_indexer_layers: int,
    num_int8_layers: int,
    index_head_dim: int,
    dense_element_size: int,
) -> int:
    """Indexer cache bytes per token when ``num_int8_layers`` of the layers are INT8."""
    assert 0 <= num_int8_layers <= num_indexer_layers
    num_dense_layers = num_indexer_layers - num_int8_layers
    return num_dense_layers * index_head_dim * dense_element_size + num_int8_layers * (
        index_head_dim + INT8_INDEXER_SCALE_BYTES
    )
