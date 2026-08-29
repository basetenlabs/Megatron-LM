# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import weakref
from unittest.mock import patch

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions.transformer_engine import TEQuantizationParams, TEQuantizationRecipe

pytestmark = pytest.mark.skipif(
    not te_ext.HAVE_TE or not te_ext.is_te_min_version("1.9.0.dev0"),
    reason="TE GroupedLinear is only supported in TE 1.9.0.dev0 and later.",
)


class _FakeGroupedCheckpointTensor:
    def __init__(self, members, quantized_tensors=None):
        self._members = members
        self.quantized_tensors = quantized_tensors

    def split_into_quantized_tensors(self):
        return self._members


def _grouped_linear_stub(
    num_gemms, *, use_bias=False, single_grouped_weight=False, single_grouped_bias=False
):
    module = te_ext.TEGroupedLinear.__new__(te_ext.TEGroupedLinear)
    module.num_gemms = num_gemms
    module.use_bias = use_bias
    module.single_grouped_weight = single_grouped_weight
    module.single_grouped_bias = single_grouped_bias
    return module


def _empty_load_args():
    """Standard load_state_dict pre-hook trailing arguments."""
    return {}, True, [], [], []


def _frozen_blockwise_grouped_linear(num_gemms=2, features=128):
    qparams = te_ext.te.common.recipe.QParams(power_2_scale=False)
    recipe = te_ext.te.common.recipe.Float8BlockScaling(
        fp8_format=te_ext.te.common.recipe.Format.E4M3,
        fp8_quant_fwd_inp=qparams,
        fp8_quant_fwd_weight=qparams,
        fp8_quant_bwd_grad=qparams,
    )
    with (
        torch.no_grad(),
        te_ext.te.pytorch.quantized_model_init(
            enabled=True, recipe=recipe, preserve_high_precision_init_val=False
        ),
    ):
        module = te_ext.te.pytorch.GroupedLinear(
            num_gemms=num_gemms,
            in_features=features,
            out_features=features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    module.requires_grad_(False)
    module.te_quant_params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(
            fp8_quantization_recipe=Fp8Recipe.blockwise,
            fp8_format="e4m3",
            fp8_param=True,
            preserve_high_precision_init_val=False,
            fp8_block_scaling_fp32_scales=True,
        ),
        evaluation_recipe=None,
    )
    module.te_return_bias = False
    module.delay_wgrad_compute = False
    module._frozen_blockwise_bf16_fast_path_enabled = True
    return module


def test_materialize_frozen_blockwise_weights_bf16() -> None:
    torch.manual_seed(1234)
    payloads = tuple(
        torch.randn((256, 128), device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
    )
    scales = tuple(torch.rand((2, 1), device="cuda") for _ in range(2))

    actual = te_ext._materialize_frozen_blockwise_weights_bf16(payloads, scales)
    expected = torch.stack(
        [
            (payload.float().view(2, 128, 1, 128) * scale[:, None, :, None])
            .reshape(256, 128)
            .to(torch.bfloat16)
            for payload, scale in zip(payloads, scales)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_frozen_blockwise_bf16_fast_path_eligibility() -> None:
    if not te_ext._TE_GROUPED_LINEAR_EXTERNAL_WEIGHT_V216_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)

    materialization_inputs = (
        te_ext.TEGroupedLinear._get_frozen_blockwise_bf16_materialization_inputs(module, inp)
    )

    assert materialization_inputs is not None
    payloads, scales = materialization_inputs
    assert len(payloads) == module.num_gemms
    assert all(payload.dtype == torch.float8_e4m3fn for payload in payloads)
    assert all(scale.dtype == torch.float32 for scale in scales)

    module.eval()
    assert (
        te_ext.TEGroupedLinear._get_frozen_blockwise_bf16_materialization_inputs(module, inp)
        is None
    )


def test_frozen_blockwise_bf16_fast_path_rejects_calibration() -> None:
    if not te_ext._TE_GROUPED_LINEAR_EXTERNAL_WEIGHT_V216_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)

    with patch.object(te_ext.FP8GlobalStateManager, "is_fp8_calibration", return_value=True):
        assert (
            te_ext.TEGroupedLinear._get_frozen_blockwise_bf16_materialization_inputs(module, inp)
            is None
        )


def test_external_bf16_weights_follow_grad_lifetime() -> None:
    if not te_ext._TE_GROUPED_LINEAR_EXTERNAL_WEIGHT_V216_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    materialization_inputs = (
        te_ext.TEGroupedLinear._get_frozen_blockwise_bf16_materialization_inputs(module, inp)
    )
    assert materialization_inputs is not None
    materializer = te_ext._get_compiled_frozen_blockwise_weight_materializer()
    m_splits = [128, 128]
    reference_weights = [
        getattr(module, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
        for gemm_idx in range(module.num_gemms)
    ]
    reference_output = torch.cat(
        [
            expert_input @ weight.T
            for expert_input, weight in zip(inp.split(m_splits), reference_weights)
        ]
    )

    with torch.no_grad():
        grouped_weights = materializer(*materialization_inputs)
        no_grad_weight_ref = weakref.ref(grouped_weights)
        output = te_ext.TEGroupedLinear._forward_with_external_bf16_weights(
            module, inp, m_splits, grouped_weights, None
        )
        torch.testing.assert_close(output, reference_output, rtol=1e-2, atol=1e-2)
        del grouped_weights
    gc.collect()
    assert no_grad_weight_ref() is None
    del output

    inp = inp.detach().requires_grad_(True)
    grouped_weights = materializer(*materialization_inputs)
    grad_weight_ref = weakref.ref(grouped_weights)
    output = te_ext.TEGroupedLinear._forward_with_external_bf16_weights(
        module, inp, m_splits, grouped_weights, None
    )
    del grouped_weights
    gc.collect()
    assert grad_weight_ref() is not None

    grad_output = torch.randn_like(output)
    reference_dgrad = torch.cat(
        [
            expert_grad @ weight
            for expert_grad, weight in zip(grad_output.split(m_splits), reference_weights)
        ]
    )
    output.backward(grad_output)
    gc.collect()
    assert grad_weight_ref() is None
    assert inp.grad is not None
    torch.testing.assert_close(inp.grad, reference_dgrad, rtol=1e-2, atol=1e-2)
    assert all(weight.grad is None for weight in module.parameters())


@pytest.mark.parametrize(("parallel_mode", "partition_dim"), (("column", 0), ("row", 1)))
def test_expert_parameter_attributes_use_expert_topology(parallel_mode, partition_dim):
    module = torch.nn.Module()
    module.register_parameter("weight0", torch.nn.Parameter(torch.empty(4, 4)))
    module.register_parameter("bias0", torch.nn.Parameter(torch.empty(4)))

    te_ext._set_expert_parameter_attributes(
        module, parallel_mode=parallel_mode, use_expert_pgs=True
    )

    assert module.weight0.allreduce is False
    assert module.weight0.tensor_model_parallel is True
    assert module.weight0.partition_dim == partition_dim
    assert module.bias0.allreduce is False
    assert module.bias0.tensor_model_parallel is (parallel_mode == "column")


@pytest.mark.parametrize(
    ("name", "is_partitioned"),
    (
        ("weight", True),
        ("weight12", True),
        ("bias", True),
        ("bias12", True),
        ("weight_scale", False),
        ("bias_extra", False),
    ),
)
def test_expert_parameter_attributes_match_parameter_names(name, is_partitioned):
    module = torch.nn.Module()
    module.register_parameter(name, torch.nn.Parameter(torch.empty(4)))

    te_ext._set_expert_parameter_attributes(module, parallel_mode="column", use_expert_pgs=True)

    param = module.get_parameter(name)
    assert getattr(param, "tensor_model_parallel", False) is is_partitioned


def test_split_empty_extra_state_for_stateless_recipe():
    module = _grouped_linear_stub(num_gemms=2)
    module.fp8_meta = {"fp8_checkpoint": True}
    module.fp8 = False
    module.fp8_calibration = False

    states = module._split_extra_state(torch.empty(0, dtype=torch.uint8))

    assert len(states) == 2
    assert all(state.dtype == torch.uint8 and state.numel() == 0 for state in states)


def test_split_grouped_checkpoint_tensor_uses_quantized_members():
    module = _grouped_linear_stub(num_gemms=2)
    members = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    tensor = _FakeGroupedCheckpointTensor(members)

    splits = module._split_grouped_checkpoint_tensor(tensor, "weight")

    assert len(splits) == len(members)
    assert all(split is member for split, member in zip(splits, members))


def test_split_grouped_checkpoint_tensor_unbinds_grouped_first_dim():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = torch.arange(12).view(3, 4)

    splits = module._split_grouped_checkpoint_tensor(tensor, "weight")

    assert len(splits) == 3
    for gemm_idx, split in enumerate(splits):
        torch.testing.assert_close(split, tensor[gemm_idx])


def test_split_grouped_checkpoint_tensor_chunks_packed_first_dim():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = torch.arange(18).view(6, 3)

    splits = module._split_grouped_checkpoint_tensor(tensor, "weight")

    assert len(splits) == 3
    for split, expected in zip(splits, torch.chunk(tensor, 3, dim=0)):
        torch.testing.assert_close(split, expected)


def test_split_grouped_checkpoint_tensor_rejects_bad_group_count():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = _FakeGroupedCheckpointTensor([torch.tensor([1]), torch.tensor([2])])

    with pytest.raises(RuntimeError, match="has 2 groups, expected 3"):
        module._split_grouped_checkpoint_tensor(tensor, "weight")


def test_split_grouped_checkpoint_tensor_rejects_unsplittable_first_dim():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = torch.arange(8).view(4, 2)

    with pytest.raises(RuntimeError, match="Cannot split checkpoint tensor"):
        module._split_grouped_checkpoint_tensor(tensor, "weight")


def test_split_grouped_checkpoint_tensor_rejects_zero_dim():
    module = _grouped_linear_stub(num_gemms=2)
    tensor = torch.tensor(7)

    with pytest.raises(RuntimeError, match="Cannot split checkpoint tensor"):
        module._split_grouped_checkpoint_tensor(tensor, "weight")


def test_normalize_grouped_parameter_keys_indexed_to_grouped_weight_only():
    module = _grouped_linear_stub(num_gemms=3, use_bias=False, single_grouped_weight=True)
    indexed = [torch.tensor([float(i), float(i) + 0.5]) for i in range(3)]
    state_dict = {f"layer.weight{i}": indexed[i] for i in range(3)}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight"}
    torch.testing.assert_close(state_dict["layer.weight"], torch.stack(indexed, dim=0))


def test_normalize_grouped_parameter_keys_indexed_to_grouped_with_bias():
    module = _grouped_linear_stub(
        num_gemms=2, use_bias=True, single_grouped_weight=True, single_grouped_bias=True
    )
    weights = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    biases = [torch.tensor([10.0]), torch.tensor([20.0])]
    state_dict = {
        "layer.weight0": weights[0],
        "layer.weight1": weights[1],
        "layer.bias0": biases[0],
        "layer.bias1": biases[1],
    }

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight", "layer.bias"}
    torch.testing.assert_close(state_dict["layer.weight"], torch.stack(weights, dim=0))
    torch.testing.assert_close(state_dict["layer.bias"], torch.stack(biases, dim=0))


def test_normalize_grouped_parameter_keys_grouped_to_indexed_weight_only():
    module = _grouped_linear_stub(num_gemms=3, use_bias=False, single_grouped_weight=False)
    grouped = torch.arange(9, dtype=torch.float32).view(3, 3)
    state_dict = {"layer.weight": grouped}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight0", "layer.weight1", "layer.weight2"}
    for i in range(3):
        torch.testing.assert_close(state_dict[f"layer.weight{i}"], grouped[i])


def test_normalize_grouped_parameter_keys_grouped_to_indexed_with_bias():
    module = _grouped_linear_stub(num_gemms=2, use_bias=True, single_grouped_weight=False)
    grouped_weight = torch.arange(8, dtype=torch.float32).view(2, 4)
    grouped_bias = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    state_dict = {"layer.weight": grouped_weight, "layer.bias": grouped_bias}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {
        "layer.weight0",
        "layer.weight1",
        "layer.bias0",
        "layer.bias1",
    }
    torch.testing.assert_close(state_dict["layer.weight0"], grouped_weight[0])
    torch.testing.assert_close(state_dict["layer.weight1"], grouped_weight[1])
    torch.testing.assert_close(state_dict["layer.bias0"], grouped_bias[0])
    torch.testing.assert_close(state_dict["layer.bias1"], grouped_bias[1])


def test_normalize_grouped_parameter_keys_skips_bias_when_use_bias_false():
    module = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=True)
    state_dict = {
        "layer.weight0": torch.zeros(2),
        "layer.weight1": torch.ones(2),
        "layer.bias0": torch.tensor([99.0]),
        "layer.bias1": torch.tensor([42.0]),
    }

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert "layer.weight" in state_dict
    assert state_dict["layer.bias0"].item() == 99.0
    assert state_dict["layer.bias1"].item() == 42.0


def test_normalize_grouped_parameter_keys_returns_when_target_layout_already_present():
    # single_grouped_weight=True and grouped key already present → no-op
    module = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=True)
    grouped = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state_dict = {"layer.weight": grouped, "layer.weight0": torch.zeros(2)}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    torch.testing.assert_close(state_dict["layer.weight"], grouped)
    torch.testing.assert_close(state_dict["layer.weight0"], torch.zeros(2))

    # single_grouped_weight=False and any indexed key present → no-op
    module = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=False)
    grouped = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    state_dict = {"layer.weight": grouped, "layer.weight0": torch.tensor([99.0])}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    torch.testing.assert_close(state_dict["layer.weight"], grouped)
    torch.testing.assert_close(state_dict["layer.weight0"], torch.tensor([99.0]))


def test_normalize_grouped_parameter_keys_returns_when_indexed_set_incomplete():
    # single_grouped_weight=True needs ALL indexed keys to fold; partial → no-op
    module = _grouped_linear_stub(num_gemms=3, use_bias=False, single_grouped_weight=True)
    state_dict = {"layer.weight0": torch.zeros(2), "layer.weight2": torch.ones(2)}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert "layer.weight" not in state_dict
    assert set(state_dict.keys()) == {"layer.weight0", "layer.weight2"}


def test_normalize_grouped_parameter_keys_round_trips_via_chained_hooks():
    """Save in one layout, load in the other, then back: tensors survive intact."""
    members = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]

    # Start from indexed checkpoint, target a single-grouped model.
    state_dict = {f"layer.weight{i}": members[i] for i in range(2)}
    folder = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=True)
    folder._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    grouped = state_dict["layer.weight"]
    torch.testing.assert_close(grouped, torch.stack(members, dim=0))

    # Now use that grouped checkpoint with an indexed model.
    splitter = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=False)
    splitter._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight0", "layer.weight1"}
    for i in range(2):
        torch.testing.assert_close(state_dict[f"layer.weight{i}"], members[i])
