# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import inspect
import os

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    finalize_model_grads,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RouterExpertBiasModel(torch.nn.Module):
    def __init__(self, config, local_tokens_per_expert):
        super().__init__()
        self.config = config
        self.ddp_config = DistributedDataParallelConfig()
        self.router = torch.nn.Module()
        self.router.register_buffer("local_tokens_per_expert", local_tokens_per_expert)
        self.router.register_buffer("expert_bias", torch.zeros_like(local_tokens_per_expert))
        self.finish_grad_sync_calls = 0

    def finish_grad_sync(self, force_all_reduce=False):
        del force_all_reduce
        self.finish_grad_sync_calls += 1


def _router_expert_bias_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        moe_router_enable_expert_bias=True,
        moe_router_score_function="sigmoid",
        moe_router_bias_update_rate=0.25,
        moe_router_load_balancing_type="none",
    )


_NO_TP_DP_CP = object()


def _router_bias_pg_collection(tp_dp_cp=_NO_TP_DP_CP):
    kwargs = {
        'tp': dist.group.WORLD,
        'pp': dist.group.WORLD,
        'embd': None,
        'pos_embd': None,
        'dp_cp': dist.group.WORLD,
    }
    if tp_dp_cp is not _NO_TP_DP_CP:
        kwargs['tp_dp_cp'] = tp_dp_cp
    return ProcessGroupCollection(**kwargs)


class TestFinalizeModelGradsMoEExpertBias:
    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()
        Utils.initialize_distributed()
        parallel_state.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_model_grads_updates_router_expert_bias_with_custom_group(self):
        assert not parallel_state.model_parallel_is_initialized()

        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        local_tokens = torch.tensor(
            [0.0, 2.0] if dist.get_rank() == 0 else [0.0, 0.0], device=device
        )
        model = _RouterExpertBiasModel(config, local_tokens)

        finalize_model_grads(
            [model], pg_collection=_router_bias_pg_collection(tp_dp_cp=dist.group.WORLD)
        )

        expected_bias = torch.tensor([0.25, -0.25], device=device)
        torch.testing.assert_close(model.router.expert_bias, expected_bias)
        torch.testing.assert_close(
            model.router.local_tokens_per_expert, torch.zeros_like(local_tokens)
        )
        assert model.finish_grad_sync_calls == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_finalize_model_grads_requires_custom_group_before_grad_sync(self):
        assert not parallel_state.model_parallel_is_initialized()
        config = _router_expert_bias_config()
        device = torch.device("cuda", torch.cuda.current_device())
        pg_collections = [
            _router_bias_pg_collection(),
            _router_bias_pg_collection(tp_dp_cp=dist.group.WORLD),
        ]
        pg_collections[1].tp_dp_cp = None

        for pg_collection in pg_collections:
            model = _RouterExpertBiasModel(config, torch.tensor([1.0, 0.0], device=device))
            with pytest.raises(AssertionError, match="tp_dp_cp"):
                finalize_model_grads([model], pg_collection=pg_collection)
            assert model.finish_grad_sync_calls == 0


class TestAllReduceLNGrads:

    def init_model(self, share_embeddings_and_output_weights: bool = False):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=True,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):
        self.tp_size = tp_size
        self.pp_size = 1
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_non_tensor_model_parallel_grads(
            [self.model], self.transformer_config, parallel_state.get_tensor_model_parallel_group()
        )

    @pytest.mark.parametrize(
        ("freeze_model", "pp_size", "share_embeddings"),
        [(True, 2, True), (False, 2, True), (True, 2, False), (False, 2, False)],
    )
    def test_allreduce_word_embedding_grads(self, freeze_model, pp_size, share_embeddings):
        self.tp_size = 1
        self.pp_size = pp_size
        Utils.initialize_model_parallel(pipeline_model_parallel_size=self.pp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model(share_embeddings)
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group()

        _allreduce_word_embedding_grads([self.model], self.transformer_config, embd_group, pp_group)


class _TPReplicatedMarkerModel(torch.nn.Module):
    """Minimal module exposing parameters with explicit TP-domain markers.

    Deliberately avoids a full GPTModel: this test is about which reduction
    operator ``_allreduce_non_tensor_model_parallel_grads`` applies to a marked
    parameter, not about model construction.
    """

    def __init__(self, marker: str = None):
        super().__init__()
        self.ddp_config = DistributedDataParallelConfig()
        # Parameter under test: carries the marker (if any).
        self.marked = torch.nn.Parameter(torch.zeros(4))
        # Control parameter: never marked, must not be touched.
        self.unmarked = torch.nn.Parameter(torch.zeros(4))
        if marker is not None:
            setattr(self.marked, marker, True)


def _marker_test_config():
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=1,
        use_cpu_initialization=True,
        # Both off, so the only thing that can select a reduction is the
        # explicit per-parameter marker.
        sequence_parallel=False,
        qk_layernorm=False,
    )


class TestTPDomainGradientReductionOperator:
    """Lock the *operator* used for TP-domain gradient reduction.

    Regression coverage for the Kimi-K3 defect where a model marked
    replicated parameters with ``sum_gradients_across_tp_domain`` but MCore
    consumed only the ``average_`` variant, so the marker was silently ignored
    and replicas drifted apart under TP>1 full-weight training.

    These tests give each rank a *distinct* local gradient, which is what makes
    the reduction operator observable. A test that only compared replicas for
    equality would pass under SUM and AVG alike, and would therefore certify a
    wrong operator.
    """

    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _rank_distinct_grad(param, rank):
        # rank 0 -> 1.0, rank 1 -> 10.0. Chosen so SUM (11.0), AVG (5.5) and
        # no-reduction (1.0 or 10.0) are all mutually distinguishable.
        return torch.full_like(param, 1.0 if rank == 0 else 10.0)

    def _run(self, marker):
        tp_size = 2
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        rank = parallel_state.get_tensor_model_parallel_rank()

        model = _TPReplicatedMarkerModel(marker=marker).cuda()
        config = _marker_test_config()
        for param in model.parameters():
            param.grad = self._rank_distinct_grad(param, rank)

        _allreduce_non_tensor_model_parallel_grads(
            [model], config, parallel_state.get_tensor_model_parallel_group()
        )
        return model

    def test_sum_marker_reduces_with_sum_not_average(self):
        """The marked gradient must equal the SUM across TP ranks."""
        model = self._run("sum_gradients_across_tp_domain")

        expected_sum = 11.0  # 1.0 + 10.0
        average = 5.5

        actual = model.marked.grad
        torch.testing.assert_close(
            actual, torch.full_like(actual, expected_sum), rtol=0, atol=1e-6
        )

        # Explicitly pin the failure modes this test exists to catch, so the
        # intent survives future edits.
        assert not torch.allclose(actual, torch.full_like(actual, average)), (
            "gradient was AVG-reduced; sum_gradients_across_tp_domain requires SUM"
        )
        assert not torch.allclose(actual, torch.full_like(actual, 1.0)) and not torch.allclose(
            actual, torch.full_like(actual, 10.0)
        ), "gradient was not reduced at all; the marker was ignored"

    def test_sum_marker_leaves_replicas_equal(self):
        """Separately from the operator, replicas must agree afterwards.

        Asserted independently of the SUM check above: replica equality is
        necessary but *not* sufficient, since it also holds under AVG.
        """
        model = self._run("sum_gradients_across_tp_domain")

        actual = model.marked.grad
        gathered = [torch.empty_like(actual) for _ in range(2)]
        dist.all_gather(gathered, actual, group=parallel_state.get_tensor_model_parallel_group())
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=1e-6)

    def test_average_marker_still_reduces_with_average(self):
        """The pre-existing average contract is unchanged by the sum routing."""
        model = self._run("average_gradients_across_tp_domain")

        actual = model.marked.grad
        torch.testing.assert_close(actual, torch.full_like(actual, 5.5), rtol=0, atol=1e-6)

    def test_unmarked_parameter_is_not_reduced(self):
        """Control: routing the sum marker must not over-reduce other params."""
        rank_value = None
        model = self._run("sum_gradients_across_tp_domain")
        rank = parallel_state.get_tensor_model_parallel_rank()
        rank_value = 1.0 if rank == 0 else 10.0

        actual = model.unmarked.grad
        torch.testing.assert_close(
            actual, torch.full_like(actual, rank_value), rtol=0, atol=1e-6
        )
