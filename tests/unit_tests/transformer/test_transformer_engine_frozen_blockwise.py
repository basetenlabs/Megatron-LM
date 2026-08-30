# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import weakref

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


def _frozen_fp8_grouped_linear(num_gemms=2, features=128):
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
        grouped_linear = te_ext.te.pytorch.GroupedLinear(
            num_gemms=num_gemms,
            in_features=features,
            out_features=features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    grouped_linear.requires_grad_(False)
    grouped_linear.te_quant_params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(
            fp8_quantization_recipe=Fp8Recipe.blockwise,
            fp8_format="e4m3",
            fp8_param=True,
            preserve_high_precision_init_val=False,
            fp8_block_scaling_fp32_scales=True,
        ),
        evaluation_recipe=None,
    )
    return grouped_linear


def test_dequantize_fp8_weights_to_bf16_matches_reference() -> None:
    torch.manual_seed(1234)
    fp8_weight_data = tuple(
        torch.randn((256, 128), device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
    )
    dequantization_scales = tuple(torch.rand((2, 1), device="cuda") for _ in range(2))

    actual = frozen_blockwise._dequantize_fp8_weights_to_bf16(
        fp8_weight_data, dequantization_scales
    )
    expected_bf16_weights = torch.stack(
        [
            (fp8_data.float().view(2, 128, 1, 128) * scale[:, None, :, None])
            .reshape(256, 128)
            .to(torch.bfloat16)
            for fp8_data, scale in zip(fp8_weight_data, dequantization_scales)
        ]
    )

    torch.testing.assert_close(actual, expected_bf16_weights, rtol=0, atol=0)


def test_without_frozen_fp8_recipe_defers_to_normal_te() -> None:
    grouped_linear = _frozen_fp8_grouped_linear()
    grouped_linear.te_quant_params = None
    input_tensor = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)

    output = frozen_blockwise.try_frozen_fp8_to_bf16_forward(
        grouped_linear, input_tensor, [128, 128], is_first_microbatch=None
    )

    assert output is None


def test_temporary_bf16_weights_follow_autograd_lifetime() -> None:
    grouped_linear = _frozen_fp8_grouped_linear()
    input_tensor = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    fp8_weight_data, dequantization_scales = frozen_blockwise._get_frozen_fp8_weight_data(
        grouped_linear
    )
    fp8_to_bf16 = frozen_blockwise._get_compiled_fp8_to_bf16()
    tokens_per_expert = [128, 128]
    reference_bf16_weights = [
        getattr(grouped_linear, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
        for gemm_idx in range(grouped_linear.num_gemms)
    ]

    with torch.no_grad():
        grouped_bf16_weights = fp8_to_bf16(fp8_weight_data, dequantization_scales)
        no_grad_weight_ref = weakref.ref(grouped_bf16_weights)
        output = frozen_blockwise._run_grouped_linear_with_bf16_weights(
            grouped_linear, input_tensor, tokens_per_expert, grouped_bf16_weights, None
        )
        del grouped_bf16_weights
    gc.collect()
    assert no_grad_weight_ref() is None
    del output

    input_tensor = input_tensor.detach().requires_grad_(True)
    grouped_bf16_weights = fp8_to_bf16(fp8_weight_data, dequantization_scales)
    grad_weight_ref = weakref.ref(grouped_bf16_weights)
    output = frozen_blockwise._run_grouped_linear_with_bf16_weights(
        grouped_linear, input_tensor, tokens_per_expert, grouped_bf16_weights, None
    )
    del grouped_bf16_weights
    gc.collect()
    assert grad_weight_ref() is not None

    grad_output = torch.randn_like(output)
    reference_dgrad = torch.cat(
        [
            expert_grad @ weight
            for expert_grad, weight in zip(
                grad_output.split(tokens_per_expert), reference_bf16_weights
            )
        ]
    )
    output.backward(grad_output)
    gc.collect()
    assert grad_weight_ref() is None
    assert input_tensor.grad is not None
    torch.testing.assert_close(input_tensor.grad, reference_dgrad, rtol=1e-2, atol=1e-2)
    assert all(weight.grad is None for weight in grouped_linear.parameters())


def test_frozen_fp8_to_bf16_forward_matches_reference() -> None:
    grouped_linear = _frozen_fp8_grouped_linear()
    input_tensor = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = [128, 128]
    reference_bf16_weights = [
        getattr(grouped_linear, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
        for gemm_idx in range(grouped_linear.num_gemms)
    ]
    expected_output = torch.cat(
        [
            expert_input @ weight.T
            for expert_input, weight in zip(
                input_tensor.split(tokens_per_expert), reference_bf16_weights
            )
        ]
    )

    with torch.no_grad():
        actual = frozen_blockwise.try_frozen_fp8_to_bf16_forward(
            grouped_linear, input_tensor, tokens_per_expert, is_first_microbatch=None
        )

    assert actual is not None
    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=1e-2)
