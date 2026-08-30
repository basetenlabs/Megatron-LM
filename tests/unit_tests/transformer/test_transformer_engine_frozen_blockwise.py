# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import weakref
from unittest.mock import patch

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions import transformer_engine_frozen_blockwise as frozen_blockwise
from megatron.core.extensions.transformer_engine import TEQuantizationParams, TEQuantizationRecipe

pytestmark = pytest.mark.skipif(
    not te_ext.HAVE_TE or not te_ext.is_te_min_version("1.9.0.dev0"),
    reason="TE GroupedLinear is only supported in TE 1.9.0.dev0 and later.",
)


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
    return module


def test_materialize_frozen_blockwise_weights_bf16() -> None:
    torch.manual_seed(1234)
    payloads = tuple(
        torch.randn((256, 128), device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
    )
    scales = tuple(torch.rand((2, 1), device="cuda") for _ in range(2))

    actual = frozen_blockwise._materialize_frozen_blockwise_weights_bf16(payloads, scales)
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
    if not frozen_blockwise._PRIVATE_ABI_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)

    assert frozen_blockwise._is_eligible(module, inp)
    materialization_inputs = frozen_blockwise._get_materialization_inputs(module, inp)
    assert materialization_inputs is not None
    payloads, scales = materialization_inputs
    assert len(payloads) == module.num_gemms
    assert all(payload.dtype == torch.float8_e4m3fn for payload in payloads)
    assert all(scale.dtype == torch.float32 for scale in scales)

    module.eval()
    assert not frozen_blockwise._is_eligible(module, inp)
    assert (
        frozen_blockwise.try_frozen_blockwise_bf16_forward(
            module, inp, [128, 128], is_first_microbatch=None
        )
        is None
    )


def test_frozen_blockwise_bf16_fast_path_rejects_calibration() -> None:
    if not frozen_blockwise._PRIVATE_ABI_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)

    with patch.object(
        frozen_blockwise.FP8GlobalStateManager, "is_fp8_calibration", return_value=True
    ):
        assert not frozen_blockwise._is_eligible(module, inp)


def test_external_bf16_weights_follow_grad_lifetime() -> None:
    if not frozen_blockwise._PRIVATE_ABI_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    materialization_inputs = frozen_blockwise._get_materialization_inputs(module, inp)
    assert materialization_inputs is not None
    materializer = frozen_blockwise._get_compiled_materializer()
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
        output = frozen_blockwise._forward_with_external_weights(
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
    output = frozen_blockwise._forward_with_external_weights(
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


def test_try_frozen_blockwise_bf16_forward_matches_reference() -> None:
    if not frozen_blockwise._PRIVATE_ABI_SUPPORTED:
        pytest.skip("requires Transformer Engine 2.16.0 private grouped-linear ABI")

    module = _frozen_blockwise_grouped_linear()
    inp = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    m_splits = [128, 128]
    reference_weights = [
        getattr(module, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
        for gemm_idx in range(module.num_gemms)
    ]
    expected = torch.cat(
        [
            expert_input @ weight.T
            for expert_input, weight in zip(inp.split(m_splits), reference_weights)
        ]
    )

    with torch.no_grad():
        actual = frozen_blockwise.try_frozen_blockwise_bf16_forward(
            module, inp, m_splits, is_first_microbatch=None
        )

    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
