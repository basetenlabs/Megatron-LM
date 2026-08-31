# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimized execution for frozen quantized Transformer Engine expert weights.

The public TE path dequantizes each frozen expert weight separately. This module
materializes all local experts into one temporary BF16 tensor, then passes those
weights to TE's grouped BF16 GEMM. The temporary tensor lives only as long as
autograd needs it for the input gradient.

Two storage formats are supported, and the difference between them is confined to
the dequantization kernel and the payload accessor:

* blockwise FP8, E4M3 elements with one FP32 scale per 128x128 block, and
* NVFP4, E2M1 elements packed two per byte with one E4M3 scale per 16 elements
  along a row plus a single per-tensor amax.

Everything downstream is format-agnostic, because the GEMM itself runs in BF16.

``try_frozen_quantized_to_bf16_forward`` is the only integration point. It returns
``None`` when no frozen quantized recipe applies. Once configured, the pinned
trainers stack is expected to satisfy the runtime contract below.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Callable

import torch
from torch import Tensor
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
from transformer_engine.pytorch.module.grouped_linear import (
    _GroupedLinear as _TEGroupedLinearAutograd,
)
from transformer_engine.pytorch.tensor import Float8BlockwiseQTensor, NVFP4Tensor

from megatron.core.enums import Fp4Recipe, Fp8Recipe

_BLOCK_SIZE = 128

# NVFP4 carries one E4M3 scale per 16 elements along a row, and a second, per-tensor
# scaling level derived from the amax. Dequantization is
# ``element * block_scale * amax / (E2M1_MAX * E4M3_MAX)``.
_NVFP4_BLOCK_SIZE = 16
_NVFP4_E2M1_MAX = 6.0
_NVFP4_E4M3_MAX = 448.0

# Three magnitude bits index this table; the fourth bit carries the sign.
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _dequantize_fp8_weights_to_bf16(
    fp8_weight_data: tuple[Tensor, ...], dequantization_scales: tuple[Tensor, ...]
) -> Tensor:
    """Dequantize grouped E4M3 blockwise weights into one BF16 tensor."""
    bf16_weights = []
    for fp8_data, dequantization_scale in zip(fp8_weight_data, dequantization_scales, strict=True):
        block_rows, block_columns = dequantization_scale.shape
        blocked = fp8_data.float().view(block_rows, _BLOCK_SIZE, block_columns, _BLOCK_SIZE)
        bf16_weights.append(
            (blocked * dequantization_scale[:, None, :, None])
            .reshape(block_rows * _BLOCK_SIZE, block_columns * _BLOCK_SIZE)
            .to(torch.bfloat16)
        )
    return torch.stack(bf16_weights)


def _dequantize_nvfp4_weights_to_bf16(
    packed_data: tuple[Tensor, ...],
    block_scales: tuple[Tensor, ...],
    amaxes: tuple[Tensor, ...],
) -> Tensor:
    """Dequantize grouped NVFP4 weights into one BF16 tensor."""
    magnitudes = torch.tensor(
        _E2M1_MAGNITUDES, dtype=torch.float32, device=packed_data[0].device
    )
    bf16_weights = []
    for data, block_scale, amax in zip(packed_data, block_scales, amaxes, strict=True):
        rows, packed_columns = data.shape
        columns = packed_columns * 2

        # Two E2M1 elements per byte, low nibble first.
        nibbles = torch.stack((data & 0x0F, (data >> 4) & 0x0F), dim=-1).reshape(rows, columns)
        magnitude = magnitudes[(nibbles & 0x07).long()]
        elements = torch.where(nibbles & 0x08 != 0, -magnitude, magnitude)

        scale = block_scale.view(torch.float8_e4m3fn).float()[:, : columns // _NVFP4_BLOCK_SIZE]
        scale = scale.repeat_interleave(_NVFP4_BLOCK_SIZE, dim=1)

        global_scale = amax.float() / (_NVFP4_E2M1_MAX * _NVFP4_E4M3_MAX)
        bf16_weights.append((elements * scale * global_scale).to(torch.bfloat16))
    return torch.stack(bf16_weights)


@cache
def _get_compiled_fp8_to_bf16() -> Callable:
    """Share compiled FC1/FC2 specializations across all expert layers."""
    return torch.compile(_dequantize_fp8_weights_to_bf16, fullgraph=True, dynamic=False)


@cache
def _get_compiled_nvfp4_to_bf16() -> Callable:
    """Share compiled FC1/FC2 specializations across all expert layers."""
    return torch.compile(_dequantize_nvfp4_weights_to_bf16, fullgraph=True, dynamic=False)


def _uses_frozen_blockwise_fp8(grouped_linear: Any) -> bool:
    """Return whether this grouped-linear layer uses frozen blockwise-FP8 storage."""
    if grouped_linear.te_quant_params is None:
        return False

    recipe = grouped_linear.te_quant_params.training_recipe
    return (
        recipe.fp8_quantization_recipe == Fp8Recipe.blockwise
        and recipe.fp8_format == "e4m3"
        and recipe.fp8_param
        and recipe.preserve_high_precision_init_val is False
        and recipe.fp8_block_scaling_fp32_scales
    )


def _uses_frozen_nvfp4(grouped_linear: Any) -> bool:
    """Return whether this grouped-linear layer uses frozen NVFP4 storage."""
    if grouped_linear.te_quant_params is None:
        return False

    recipe = grouped_linear.te_quant_params.training_recipe
    return (
        recipe.fp4_quantization_recipe == Fp4Recipe.nvfp4
        and recipe.fp4_param
        and recipe.preserve_high_precision_init_val is False
    )


def _get_frozen_fp8_weight_data(
    grouped_linear: Any,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Collect raw FP8 weight data and dequantization scales for every local expert."""
    fp8_weight_data = []
    dequantization_scales = []
    for gemm_idx in range(grouped_linear.num_gemms):
        weight = getattr(grouped_linear, f"weight{gemm_idx}")
        assert isinstance(weight, Float8BlockwiseQTensor)
        assert not weight.requires_grad
        assert weight.dtype == torch.bfloat16
        assert not hasattr(weight, "materialize_group_for_forward")
        fp8_data = weight._rowwise_data
        dequantization_scale = weight._rowwise_scale_inv
        block_rows = weight.shape[0] // _BLOCK_SIZE
        block_columns = weight.shape[1] // _BLOCK_SIZE
        fp8_weight_data.append(fp8_data.view(torch.float8_e4m3fn))
        dequantization_scales.append(dequantization_scale[:block_rows, :block_columns])

    return tuple(fp8_weight_data), tuple(dequantization_scales)


def _get_frozen_nvfp4_weight_data(
    grouped_linear: Any,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    """Collect packed NVFP4 data, block scales, and amaxes for every local expert."""
    packed_data = []
    block_scales = []
    amaxes = []
    for gemm_idx in range(grouped_linear.num_gemms):
        weight = getattr(grouped_linear, f"weight{gemm_idx}")
        assert isinstance(weight, NVFP4Tensor)
        assert not weight.requires_grad
        assert weight.dtype == torch.bfloat16
        # A swizzled scale layout is laid out for TE's GEMM, not for this kernel.
        assert not weight._with_gemm_swizzled_scales
        assert not weight._row_scaled_nvfp4
        assert weight._columnwise_data is None
        packed_data.append(weight._rowwise_data)
        block_scales.append(weight._rowwise_scale_inv)
        amaxes.append(weight._amax_rowwise)

    return tuple(packed_data), tuple(block_scales), tuple(amaxes)


def _run_grouped_linear_with_bf16_weights(
    grouped_linear: Any,
    input_tensor: Tensor,
    tokens_per_expert: list[int],
    grouped_bf16_weights: Tensor,
    is_first_microbatch: bool | None,
) -> Tensor:
    """Run TE's grouped linear operation with temporary BF16 expert weights."""
    is_grad_enabled = torch.is_grad_enabled()
    input_tensor = grouped_linear.prepare_forward(input_tensor, num_gemms=grouped_linear.num_gemms)
    try:
        assert not grouped_linear.fp8
        assert not grouped_linear.fp8_calibration
        (
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
        ) = grouped_linear._get_quantizers()

        # Positional layout expected by TE 2.16's private _GroupedLinear function.
        non_tensor_args = (
            tokens_per_expert,
            False,  # use_bias
            is_first_microbatch,
            False,  # fp8
            False,  # fp8_calibration
            grouped_linear.wgrad_store,
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
            grouped_linear.fuse_wgrad_accumulation,
            is_cpu_offload_enabled(),
            grouped_linear.sequence_parallel,
            grouped_linear.activation_dtype,
            is_grad_enabled,
            [None] * grouped_linear.num_gemms,  # weight_workspaces; unused when fp8=False
            False,  # cache_weight; unused when fp8=False
            None,  # skip_fp8_weight_update
            grouped_linear.save_original_input,
            False,  # debug
        )
        bf16_weights = list(grouped_bf16_weights.unbind(dim=0))
        biases = [input_tensor.new_empty(0) for _ in bf16_weights]
        if is_grad_enabled:
            output, _ = _TEGroupedLinearAutograd.apply(
                input_tensor, non_tensor_args, *bf16_weights, *biases
            )
        else:
            output, _ = _TEGroupedLinearAutograd.forward(
                None, input_tensor, non_tensor_args, *bf16_weights, *biases
            )
        return output
    finally:
        grouped_linear.end_forward()


def _materialize_bf16_weights(grouped_linear: Any) -> Tensor | None:
    """Materialize every local expert into one BF16 tensor, or ``None`` if not frozen."""
    if _uses_frozen_blockwise_fp8(grouped_linear):
        fp8_weight_data, dequantization_scales = _get_frozen_fp8_weight_data(grouped_linear)
        return _get_compiled_fp8_to_bf16()(fp8_weight_data, dequantization_scales)
    if _uses_frozen_nvfp4(grouped_linear):
        packed_data, block_scales, amaxes = _get_frozen_nvfp4_weight_data(grouped_linear)
        return _get_compiled_nvfp4_to_bf16()(packed_data, block_scales, amaxes)
    return None


def try_frozen_quantized_to_bf16_forward(
    grouped_linear: Any,
    input_tensor: Tensor,
    tokens_per_expert: list[int],
    *,
    is_first_microbatch: bool | None,
) -> Tensor | None:
    """Return the optimized output, or ``None`` to use TE's public forward path."""
    grouped_bf16_weights = _materialize_bf16_weights(grouped_linear)
    if grouped_bf16_weights is None:
        return None

    assert grouped_linear.training
    assert input_tensor.dtype == torch.bfloat16
    assert not grouped_linear.use_bias
    assert not getattr(grouped_linear, "single_grouped_weight", False)
    assert not grouped_linear.is_debug_iter()

    return _run_grouped_linear_with_bf16_weights(
        grouped_linear, input_tensor, tokens_per_expert, grouped_bf16_weights, is_first_microbatch
    )
