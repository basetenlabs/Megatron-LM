# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Config-surface tests for the per-layer recompute dial
(``moe_ep_overlap_checkpoint_num_layers``, LPS-1062).

The dial runs the leading K MoE layers of each pipeline stage as opaque
whole-layer checkpoints inside the fine-grained EP-overlap schedule (the
five-node decomposition cannot see inside a checkpoint, which is why the
overlap flag otherwise forbids full recompute). These tests pin the config
validation only; the plan-structure and grad-equivalence tests live with the
validation harness (see the campaign's VALIDATION_LADDER doc).

CPU-runnable (no distributed init, no CUDA); on Darwin it needs the
stubbed-import harness for the megatron.core import chain (triton et al.) —
trainers repo: experiment_artefacts/glm/lps_1062_perf/tools/run_server_tests_mac.py.
"""

import pytest

from megatron.core.models.gpt.fine_grained_callables import build_checkpointed_layer_callables
from megatron.core.pipeline_parallel.utils import NoopScheduleNode
from megatron.core.transformer.transformer_config import TransformerConfig


def _overlap_legal(**overrides):
    """Minimal TransformerConfig kwargs that pass the overlap_moe_expert_parallel_comm
    assert block, so the dial's own checks are what pass/fail."""
    base = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=2,
        expert_model_parallel_size=2,
        num_moe_experts=4,
        moe_token_dispatcher_type='alltoall',
        bf16=True,
        overlap_moe_expert_parallel_comm=True,
        recompute_granularity=None,
    )
    base.update(overrides)
    return base


class TestRecomputeDialConfig:
    def test_default_is_none(self):
        config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=2)
        assert config.moe_ep_overlap_checkpoint_num_layers is None

    def test_set_with_overlap_flag_accepted(self):
        config = TransformerConfig(
            **{**_overlap_legal(), 'moe_ep_overlap_checkpoint_num_layers': 1}
        )
        assert config.moe_ep_overlap_checkpoint_num_layers == 1

    def test_zero_is_a_legal_noop(self):
        config = TransformerConfig(
            **{**_overlap_legal(), 'moe_ep_overlap_checkpoint_num_layers': 0}
        )
        assert config.moe_ep_overlap_checkpoint_num_layers == 0

    def test_rejected_without_overlap_flag(self):
        with pytest.raises(AssertionError, match='moe_ep_overlap_checkpoint_num_layers'):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=2,
                moe_ep_overlap_checkpoint_num_layers=1,
            )

    def test_negative_rejected(self):
        with pytest.raises(AssertionError, match='moe_ep_overlap_checkpoint_num_layers'):
            TransformerConfig(**{**_overlap_legal(), 'moe_ep_overlap_checkpoint_num_layers': -1})

    def test_existing_recompute_asserts_untouched(self):
        # The dial does not relax the four stock recompute exclusions under the
        # overlap flag; full recompute stays illegal. (recompute_method='block'
        # satisfies the earlier cross-field validator so the overlap block's own
        # assert is the one that fires.)
        with pytest.raises(AssertionError, match='full recomputation'):
            TransformerConfig(
                **{
                    **_overlap_legal(),
                    'recompute_granularity': 'full',
                    'recompute_method': 'block',
                    'recompute_num_layers': 1,
                    'moe_ep_overlap_checkpoint_num_layers': 1,
                }
            )


class TestCheckpointedLayerCallablesStructure:
    """The opaque layer's structural contract with the schedule plan.

    Build-time only: ``build_checkpointed_layer_callables`` does not touch the
    layer until the callable runs, so a stand-in object suffices here. The
    on-hardware plan-structure test (real tiny MoE model, K=1: dial layer gets
    this shape, overlapped layers keep the stock five nodes) lives with the
    validation harness.
    """

    def test_forward_funcs_shape(self):
        fwd_callables, bwd_dw = build_checkpointed_layer_callables(object())
        assert len(fwd_callables) == 5
        assert callable(fwd_callables[0])
        # comm/mlp/mtp slots are never built — the plan installs NoopScheduleNodes.
        assert all(f is None for f in fwd_callables[1:])
        # Empty dw map: the checkpoint's monolithic backward computes weight
        # grads inline; every dw-deferral call site in the schedule no-ops.
        assert bwd_dw == {}

    def test_noop_node_supports_backward_dw(self):
        # The schedule calls mlp.backward_dw() unconditionally; a dial layer's
        # mlp slot is a NoopScheduleNode, so the no-op dw hook must exist.
        node = NoopScheduleNode()
        assert node.backward_dw() is None
        assert node.forward("x") == "x"
        assert node.backward("g") == "g"
