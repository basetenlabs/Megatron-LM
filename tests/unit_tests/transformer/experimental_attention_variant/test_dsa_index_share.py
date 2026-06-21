# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for DSA cross-layer top-k sharing ("GLM-5.2 IndexShare").

Covers:
* The pure-Python math helpers ``is_dsa_skip_topk_layer`` / ``source_dsa_compute_layer`` and
  their consistency with the per-layer ``indexer_types`` published in the GLM-5.2 HF config.
* The pipeline-split validator ``_validate_dsa_index_share_pipeline_split`` (accepts PP=1,
  rejects PP splits where a skip layer's computing source is in a different stage).
* ``DSAttention`` construction: computing layers own an indexer; skip layers do not.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    _validate_dsa_index_share_pipeline_split,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAIndexer,
    DSAIndexerSubmodules,
    DSAttention,
    DSAttentionSubmodules,
    is_dsa_skip_topk_layer,
    source_dsa_compute_layer,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils


# Reproduced from the published ``zai-org/GLM-5.2`` ``config.json`` (len=78). The first
# three layers are dense ("full" with no sparse-attention selection); the remaining 75 layers
# cycle [shared, shared, shared, full] for 18 full cycles (layers 7, 11, ..., 75 are full)
# followed by three trailing shared layers (76, 77, 78).
GLM_5_2_INDEXER_TYPES = (
    ['full', 'full', 'full']
    + (['shared', 'shared', 'shared', 'full'] * 18)
    + ['shared', 'shared', 'shared']
)
GLM_5_2_TOPK_FREQ = 4
GLM_5_2_SKIP_TOPK_OFFSET = 3
GLM_5_2_NUM_LAYERS = len(GLM_5_2_INDEXER_TYPES)
assert GLM_5_2_NUM_LAYERS == 78


class TestDSAIndexShareLayerMath:
    """Pure-Python tests of the IndexShare layer math; no CUDA, no parallel state."""

    def test_formula_matches_glm52_indexer_types(self):
        """``is_dsa_skip_topk_layer`` reproduces the GLM-5.2 HF ``indexer_types`` pattern."""
        mismatches = []
        for layer_number in range(1, GLM_5_2_NUM_LAYERS + 1):
            skip = is_dsa_skip_topk_layer(
                layer_number, GLM_5_2_SKIP_TOPK_OFFSET, GLM_5_2_TOPK_FREQ
            )
            expected = GLM_5_2_INDEXER_TYPES[layer_number - 1] == 'shared'
            if skip != expected:
                mismatches.append((layer_number, skip, GLM_5_2_INDEXER_TYPES[layer_number - 1]))
        assert mismatches == [], f"Formula / HF mismatches: {mismatches[:5]}"

    def test_compute_layer_count_matches_hf(self):
        computing = [
            L for L in range(1, GLM_5_2_NUM_LAYERS + 1)
            if not is_dsa_skip_topk_layer(L, GLM_5_2_SKIP_TOPK_OFFSET, GLM_5_2_TOPK_FREQ)
        ]
        hf_full = [
            L for L, t in enumerate(GLM_5_2_INDEXER_TYPES, start=1) if t == 'full'
        ]
        assert computing == hf_full
        assert len(computing) == 21
        assert computing == [1, 2, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75]

    def test_source_layer_for_skip_layers_always_full_and_earlier(self):
        for layer_number in range(1, GLM_5_2_NUM_LAYERS + 1):
            if not is_dsa_skip_topk_layer(
                layer_number, GLM_5_2_SKIP_TOPK_OFFSET, GLM_5_2_TOPK_FREQ
            ):
                continue
            source = source_dsa_compute_layer(
                layer_number, GLM_5_2_SKIP_TOPK_OFFSET, GLM_5_2_TOPK_FREQ
            )
            assert source < layer_number, (layer_number, source)
            assert GLM_5_2_INDEXER_TYPES[source - 1] == 'full', (layer_number, source)

    def test_boundary_layers_own_themselves_as_source(self):
        # Layers [1, skip_topk_offset] are boundary computing layers; their source is themselves.
        for layer_number in range(1, GLM_5_2_SKIP_TOPK_OFFSET + 1):
            assert not is_dsa_skip_topk_layer(
                layer_number, GLM_5_2_SKIP_TOPK_OFFSET, GLM_5_2_TOPK_FREQ
            )
            assert source_dsa_compute_layer(
                layer_number, GLM_5_2_SKIP_TOPK_OFFSET, GLM_5_2_TOPK_FREQ
            ) == layer_number

    def test_freq_one_disables_sharing(self):
        for layer_number in range(1, 16):
            assert not is_dsa_skip_topk_layer(layer_number, skip_topk_offset=0, topk_freq=1)
            assert not is_dsa_skip_topk_layer(layer_number, skip_topk_offset=3, topk_freq=1)

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError, match="layer_number must be 1-indexed"):
            is_dsa_skip_topk_layer(0, skip_topk_offset=0, topk_freq=4)
        with pytest.raises(ValueError, match="topk_freq must be positive"):
            is_dsa_skip_topk_layer(1, skip_topk_offset=0, topk_freq=0)


class TestValidateDSAIndexSharePipelineSplit:
    """Pipeline-split validator tests; mocks the config (no real parallel state needed)."""

    def _config(self, *, variant="dsa", topk_freq=4, skip_topk_offset=3):
        return SimpleNamespace(
            experimental_attention_variant=variant,
            dsa_indexer_topk_freq=topk_freq,
            dsa_indexer_skip_topk_offset=skip_topk_offset,
        )

    def test_pp1_accepts_full_model(self):
        # All 78 layers in one stage -> every skip layer's source is earlier in the same stage.
        local_layer_ids = list(range(GLM_5_2_NUM_LAYERS))
        _validate_dsa_index_share_pipeline_split(self._config(), local_layer_ids)

    def test_no_op_for_non_dsa_variant(self):
        # Linear / dsv4_hybrid variants do not invoke the validator's body.
        local_layer_ids = list(range(GLM_5_2_NUM_LAYERS))
        _validate_dsa_index_share_pipeline_split(
            self._config(variant="gated_delta_net"), local_layer_ids
        )
        _validate_dsa_index_share_pipeline_split(
            self._config(variant="dsv4_hybrid"), local_layer_ids
        )

    def test_no_op_when_freq_is_one(self):
        # Default ``dsa_indexer_topk_freq=1`` disables sharing -> validator returns early.
        local_layer_ids = list(range(GLM_5_2_NUM_LAYERS))
        _validate_dsa_index_share_pipeline_split(
            self._config(topk_freq=1), local_layer_ids
        )

    def test_rejects_stage_starting_on_a_skip_layer(self):
        # Stage owns layers 4..10 (ids 3..9). L=5 / L=6 are skip layers whose source computing
        # layer is L=3 (= id 2), which is in stage 0 -> must reject.
        local_layer_ids = list(range(3, 10))
        with pytest.raises(AssertionError, match="pipeline split is invalid"):
            _validate_dsa_index_share_pipeline_split(self._config(), local_layer_ids)

    def test_accepts_stage_starting_on_computing_layer(self):
        # Stage owns layers 7..10 (ids 6..9). L=7 is the first computing layer in this cycle;
        # L=8/9/10 are skip layers whose source is L=7 inside the same stage -> ok.
        local_layer_ids = list(range(6, 10))
        _validate_dsa_index_share_pipeline_split(self._config(), local_layer_ids)

    def test_rejects_stage_ending_before_its_skip_source(self):
        # Stage owns only the skip layer 4 (id 3); its source (L=3 = id 2) is missing.
        local_layer_ids = [3]
        with pytest.raises(AssertionError, match="pipeline split is invalid"):
            _validate_dsa_index_share_pipeline_split(self._config(), local_layer_ids)


class TestDSAttentionIndexShareConstruction:
    """Construction-time checks: indexer presence matches the skip/compute classification."""

    @pytest.fixture(scope='class', autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        yield
        Utils.destroy_model_parallel()

    def _config(self):
        return MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
            dsa_indexer_topk_freq=GLM_5_2_TOPK_FREQ,
            dsa_indexer_skip_topk_offset=GLM_5_2_SKIP_TOPK_OFFSET,
        )

    def _submodules(self):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm

        return DSAttentionSubmodules(
            indexer=ModuleSpec(
                module=DSAIndexer,
                submodules=DSAIndexerSubmodules(
                    linear_wq_b=ModuleSpec(module=TELinear),
                    linear_wk=ModuleSpec(module=TELinear),
                    k_norm=ModuleSpec(module=TENorm),
                    linear_weights_proj=ModuleSpec(module=TELinear),
                ),
            )
        )

    def _build(self, layer_number):
        return DSAttention(
            config=self._config(),
            submodules=self._submodules(),
            layer_number=layer_number,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp']),
        )

    def test_default_knobs_disable_sharing(self):
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=16,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type='rope',
            rotary_base=10000,
            rotary_percent=1.0,
            dsa_indexer_n_heads=8,
            dsa_indexer_head_dim=64,
            dsa_indexer_topk=32,
        )
        sa = DSAttention(
            config=config,
            submodules=self._submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type='self',
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp']),
        )
        assert sa.index_share is False
        assert sa.skip_topk is False
        assert sa.indexer is not None
        assert sa.source_layer == 1
        assert sa.index_topk == 32

    @pytest.mark.parametrize(
        "layer_number, expect_skip, expected_source",
        [
            (1, False, 1),
            (2, False, 2),
            (3, False, 3),
            (4, True, 3),
            (5, True, 3),
            (6, True, 3),
            (7, False, 7),
            (8, True, 7),
            (11, False, 11),
            (12, True, 11),
        ],
    )
    def test_layer_classification(self, layer_number, expect_skip, expected_source):
        sa = self._build(layer_number)
        assert sa.index_share is True
        assert sa.skip_topk is expect_skip
        assert sa.source_layer == expected_source
        if expect_skip:
            assert sa.indexer is None, (
                f"Skip layer {layer_number} must not build an indexer"
            )
            assert sa.index_topk == 32
            assert sa.index_topk_freq == GLM_5_2_TOPK_FREQ
            assert sa.index_skip_topk_offset == GLM_5_2_SKIP_TOPK_OFFSET
        else:
            assert isinstance(sa.indexer, DSAIndexer), (
                f"Computing layer {layer_number} must own a DSAIndexer instance"
            )
