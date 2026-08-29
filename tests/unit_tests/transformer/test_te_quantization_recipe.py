# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.extensions import transformer_engine as te_extension
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEQuantizationRecipe,
    _get_fp8_model_init_for_quant_recipe,
)


def _blockwise_recipe_config(**overrides) -> dict:
    config = {"fp8_quantization_recipe": "blockwise", "fp8_param": True}
    config.update(overrides)
    return config


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required for recipe construction")
def test_blockwise_recipe_constructs_fp32_scales_without_bf16_master() -> None:
    recipe = TEQuantizationRecipe.parse_from_config(
        _blockwise_recipe_config(
            preserve_high_precision_init_val=False, fp8_block_scaling_fp32_scales=True
        )
    )
    blockwise_recipe = object()
    model_init_context = object()

    with (
        patch.object(
            te_extension.te.common.recipe, "Float8BlockScaling", return_value=blockwise_recipe
        ) as blockwise_constructor,
        patch.object(te_extension, "fp8_model_init", return_value=model_init_context) as model_init,
    ):
        result = _get_fp8_model_init_for_quant_recipe(recipe)

    assert result is model_init_context
    constructor_kwargs = blockwise_constructor.call_args.kwargs
    assert constructor_kwargs["fp8_format"] == te_extension.te.common.recipe.Format.E4M3
    for qparams_name in ("fp8_quant_fwd_inp", "fp8_quant_fwd_weight", "fp8_quant_bwd_grad"):
        assert constructor_kwargs[qparams_name].power_2_scale is False
    model_init.assert_called_once_with(
        enabled=True, recipe=blockwise_recipe, preserve_high_precision_init_val=False
    )


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required for recipe construction")
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_blockwise_recipe_defaults_preserve_existing_model_init_behavior(grad_enabled) -> None:
    recipe = TEQuantizationRecipe.parse_from_config(_blockwise_recipe_config())
    blockwise_recipe = object()

    with (
        torch.set_grad_enabled(grad_enabled),
        patch.object(
            te_extension.te.common.recipe, "Float8BlockScaling", return_value=blockwise_recipe
        ) as blockwise_constructor,
        patch.object(te_extension, "fp8_model_init") as model_init,
    ):
        _get_fp8_model_init_for_quant_recipe(recipe)

    blockwise_constructor.assert_called_once_with(
        fp8_format=te_extension.te.common.recipe.Format.E4M3
    )
    assert model_init.call_args.kwargs["preserve_high_precision_init_val"] is grad_enabled
