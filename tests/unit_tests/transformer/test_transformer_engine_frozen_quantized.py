# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import weakref

import pytest
import torch

from megatron.core.enums import Fp4Recipe, Fp8Recipe
from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions import transformer_engine_frozen_quantized as frozen_quantized
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


def _frozen_nvfp4_grouped_linear(num_gemms=2, features=128):
    recipe = te_ext.te.common.recipe.NVFP4BlockScaling(disable_2d_quantization=True)
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
            fp4_quantization_recipe=Fp4Recipe.nvfp4,
            fp4_param=True,
            preserve_high_precision_init_val=False,
            nvfp4_disable_2d_quantization=True,
        ),
        evaluation_recipe=None,
    )
    return grouped_linear


@pytest.mark.skipif(
    not te_ext.is_te_min_version("2.7.0.dev0"),
    reason="NVFP4 tensors require Transformer Engine 2.7.0.dev0 or later.",
)
def test_dequantize_nvfp4_weights_to_bf16_matches_te_dequantize() -> None:
    """The kernel must reproduce TE's own dequantize() exactly, not merely closely.

    This path replaces TE's dequantization rather than calling it, so any drift in
    the packing order or in either level of NVFP4's two-level scaling would be a
    silent numerical change to every expert.
    """
    torch.manual_seed(1234)
    grouped_linear = _frozen_nvfp4_grouped_linear(features=512)

    packed_data, block_scales, amaxes = frozen_quantized._get_frozen_nvfp4_weight_data(
        grouped_linear
    )
    actual = frozen_quantized._dequantize_nvfp4_weights_to_bf16(packed_data, block_scales, amaxes)
    expected = torch.stack(
        [
            getattr(grouped_linear, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
            for gemm_idx in range(grouped_linear.num_gemms)
        ]
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    not te_ext.is_te_min_version("2.7.0.dev0"),
    reason="NVFP4 tensors require Transformer Engine 2.7.0.dev0 or later.",
)
def test_frozen_nvfp4_to_bf16_forward_matches_reference() -> None:
    grouped_linear = _frozen_nvfp4_grouped_linear()
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
        actual = frozen_quantized.try_frozen_quantized_to_bf16_forward(
            grouped_linear, input_tensor, tokens_per_expert, is_first_microbatch=None
        )

    assert actual is not None
    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=1e-2)


def test_nvfp4_disable_2d_quantization_rejected_without_nvfp4_recipe() -> None:
    with pytest.raises(ValueError, match="only supported with the nvfp4 recipe"):
        TEQuantizationRecipe.parse_from_config(
            {"fp8_quantization_recipe": Fp8Recipe.blockwise, "nvfp4_disable_2d_quantization": True}
        )


def test_nvfp4_disable_2d_quantization_rejects_a_string_value() -> None:
    with pytest.raises(ValueError, match="nvfp4_disable_2d_quantization must be a bool"):
        TEQuantizationRecipe.parse_from_config(
            {"fp4_quantization_recipe": Fp4Recipe.nvfp4, "nvfp4_disable_2d_quantization": "true"}
        )


def test_nvfp4_recipe_defaults_leave_te_behaviour_untouched() -> None:
    """An existing NVFP4 recipe must not change because this field was added.

    The flag names an override, so unless it is set the recipe has to come back
    exactly as TE builds it by default -- otherwise every FP4 run that predates
    this field silently switches its scale blocking.
    """
    default = TEQuantizationRecipe.parse_from_config({"fp4_quantization_recipe": Fp4Recipe.nvfp4})
    built = te_ext._get_nvfp4_block_scaling_recipe(default)
    reference = te_ext.te.common.recipe.NVFP4BlockScaling()

    assert built.disable_2d_quantization == reference.disable_2d_quantization

    overridden = TEQuantizationRecipe.parse_from_config(
        {"fp4_quantization_recipe": Fp4Recipe.nvfp4, "nvfp4_disable_2d_quantization": True}
    )
    assert te_ext._get_nvfp4_block_scaling_recipe(overridden).disable_2d_quantization


def test_frozen_nvfp4_gate_requires_the_one_dimensional_scale_marker() -> None:
    """A trainable NVFP4 run must not be hijacked into the frozen path.

    Without a marker the gate would match any NVFP4 run with fp4_param set and no
    high-precision copy, and then die on the frozen-weight assertion.
    """
    grouped_linear = _frozen_nvfp4_grouped_linear()
    grouped_linear.te_quant_params = TEQuantizationParams(
        training_recipe=TEQuantizationRecipe(
            fp4_quantization_recipe=Fp4Recipe.nvfp4,
            fp4_param=True,
            preserve_high_precision_init_val=False,
            nvfp4_disable_2d_quantization=False,
        ),
        evaluation_recipe=None,
    )

    assert not frozen_quantized._uses_frozen_nvfp4(grouped_linear)


def test_nvfp4_autocast_uses_the_configured_scale_layout(monkeypatch) -> None:
    built_recipe = object()
    recipe_kwargs = {}

    def build_recipe(**kwargs):
        recipe_kwargs.update(kwargs)
        return built_recipe

    monkeypatch.setattr(te_ext.FP8GlobalStateManager, "is_fp8_enabled", lambda: False)
    monkeypatch.setattr(te_ext.te.common.recipe, "NVFP4BlockScaling", build_recipe)
    monkeypatch.setattr(te_ext, "fp8_autocast", lambda **kwargs: kwargs)
    qrecipe = TEQuantizationRecipe(
        fp4_quantization_recipe=Fp4Recipe.nvfp4,
        nvfp4_disable_2d_quantization=True,
        override_nonquantized_autocast=True,
    )

    context = te_ext._get_fp8_autocast_for_quant_recipe(qrecipe)

    assert recipe_kwargs == {"disable_2d_quantization": True}
    assert context["enabled"] is True
    assert context["fp8_recipe"] is built_recipe


def test_fp4_autocast_error_names_the_unsupported_fp4_recipe(monkeypatch) -> None:
    monkeypatch.setattr(te_ext.FP8GlobalStateManager, "is_fp8_enabled", lambda: False)
    qrecipe = TEQuantizationRecipe(
        fp4_quantization_recipe="unsupported", override_nonquantized_autocast=True
    )

    with pytest.raises(ValueError, match="Unhandled fp4 recipe: unsupported"):
        te_ext._get_fp8_autocast_for_quant_recipe(qrecipe)


def test_dequantize_fp8_weights_to_bf16_matches_reference() -> None:
    torch.manual_seed(1234)
    fp8_weight_data = tuple(
        torch.randn((256, 128), device="cuda").to(torch.float8_e4m3fn) for _ in range(2)
    )
    dequantization_scales = tuple(torch.rand((2, 1), device="cuda") for _ in range(2))

    actual = frozen_quantized._dequantize_fp8_weights_to_bf16(
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

    output = frozen_quantized.try_frozen_quantized_to_bf16_forward(
        grouped_linear, input_tensor, [128, 128], is_first_microbatch=None
    )

    assert output is None


def test_temporary_bf16_weights_follow_autograd_lifetime() -> None:
    grouped_linear = _frozen_fp8_grouped_linear()
    input_tensor = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16)
    fp8_weight_data, dequantization_scales = frozen_quantized._get_frozen_fp8_weight_data(
        grouped_linear
    )
    fp8_to_bf16 = frozen_quantized._get_compiled_fp8_to_bf16()
    tokens_per_expert = [128, 128]
    reference_bf16_weights = [
        getattr(grouped_linear, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
        for gemm_idx in range(grouped_linear.num_gemms)
    ]

    with torch.no_grad():
        grouped_bf16_weights = fp8_to_bf16(fp8_weight_data, dequantization_scales)
        no_grad_weight_ref = weakref.ref(grouped_bf16_weights)
        output = frozen_quantized._run_grouped_linear_with_bf16_weights(
            grouped_linear, input_tensor, tokens_per_expert, grouped_bf16_weights, None
        )
        del grouped_bf16_weights
    gc.collect()
    assert no_grad_weight_ref() is None
    del output

    input_tensor = input_tensor.detach().requires_grad_(True)
    grouped_bf16_weights = fp8_to_bf16(fp8_weight_data, dequantization_scales)
    grad_weight_ref = weakref.ref(grouped_bf16_weights)
    output = frozen_quantized._run_grouped_linear_with_bf16_weights(
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
        actual = frozen_quantized.try_frozen_quantized_to_bf16_forward(
            grouped_linear, input_tensor, tokens_per_expert, is_first_microbatch=None
        )

    assert actual is not None
    torch.testing.assert_close(actual, expected_output, rtol=1e-2, atol=1e-2)


def test_frozen_fp8_forward_rematerializes_weights_for_backward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped_linear = _frozen_fp8_grouped_linear()
    input_tensor = torch.randn((256, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tokens_per_expert = [128, 128]
    reference_bf16_weights = [
        getattr(grouped_linear, f"weight{gemm_idx}").dequantize(dtype=torch.bfloat16)
        for gemm_idx in range(grouped_linear.num_gemms)
    ]
    real_materialize_bf16_weights = frozen_quantized._materialize_bf16_weights
    materialized_weight_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def record_materialization(grouped_linear) -> torch.Tensor | None:
        weights = real_materialize_bf16_weights(grouped_linear)
        assert weights is not None
        materialized_weight_refs.append(weakref.ref(weights))
        return weights

    monkeypatch.setattr(frozen_quantized, "_materialize_bf16_weights", record_materialization)

    output = frozen_quantized.try_frozen_quantized_to_bf16_forward(
        grouped_linear, input_tensor, tokens_per_expert, is_first_microbatch=None
    )
    assert output is not None
    gc.collect()
    assert len(materialized_weight_refs) == 1
    assert materialized_weight_refs[0]() is None

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

    assert len(materialized_weight_refs) == 2
    assert all(weight_ref() is None for weight_ref in materialized_weight_refs)
    assert input_tensor.grad is not None
    torch.testing.assert_close(input_tensor.grad, reference_dgrad, rtol=1e-2, atol=1e-2)
