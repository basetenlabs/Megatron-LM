# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optimized execution for frozen blockwise-FP8 Transformer Engine weights.

The public TE path dequantizes each frozen expert weight separately. This module
materializes all local experts into one temporary BF16 tensor, then passes those
weights to TE's grouped BF16 GEMM. The temporary tensor lives only as long as
autograd needs it for the input gradient.

``try_frozen_blockwise_bf16_forward`` is the only integration point. It returns
``None`` unless the installed TE private API, execution mode, recipe, and weight
storage all match the narrow configuration implemented here.
"""

from __future__ import annotations

import inspect
from functools import cache
from typing import Any, Callable

import torch
from packaging.version import Version as PkgVersion
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.utils import get_te_version

_BLOCK_SIZE = 128

try:
    from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.module.grouped_linear import (
        _GroupedLinear as _TEGroupedLinearAutograd,
    )
    from transformer_engine.pytorch.tensor import Float8BlockwiseQTensor
except ImportError:
    FP8GlobalStateManager = None
    Float8BlockwiseQTensor = None
    _TEGroupedLinearAutograd = None
    is_cpu_offload_enabled = None


def _has_supported_private_abi() -> bool:
    """Return whether the installed TE exposes the private API used below."""
    if _TEGroupedLinearAutograd is None:
        return False
    parameters = inspect.signature(_TEGroupedLinearAutograd.forward).parameters
    return (
        get_te_version() == PkgVersion("2.16.0")
        and tuple(parameters) == ("ctx", "inp", "non_tensor_args", "weights_and_biases")
        and parameters["weights_and_biases"].kind == inspect.Parameter.VAR_POSITIONAL
    )


_PRIVATE_ABI_SUPPORTED = _has_supported_private_abi()


def _materialize_frozen_blockwise_weights_bf16(
    payloads: tuple[Tensor, ...], scales: tuple[Tensor, ...]
) -> Tensor:
    """Materialize grouped E4M3 blockwise weights into one BF16 tensor."""
    weights = []
    for payload, scale in zip(payloads, scales, strict=True):
        block_rows, block_columns = scale.shape
        blocked = payload.float().view(block_rows, _BLOCK_SIZE, block_columns, _BLOCK_SIZE)
        weights.append(
            (blocked * scale[:, None, :, None])
            .reshape(block_rows * _BLOCK_SIZE, block_columns * _BLOCK_SIZE)
            .to(torch.bfloat16)
        )
    return torch.stack(weights)


@cache
def _get_compiled_materializer() -> Callable:
    """Share compiled FC1/FC2 specializations across all expert layers."""
    return torch.compile(_materialize_frozen_blockwise_weights_bf16, fullgraph=True, dynamic=False)


def _is_eligible(module: Any, inp: Tensor) -> bool:
    """Check the execution mode and recipe before inspecting private weight storage."""
    assert FP8GlobalStateManager is not None
    if (
        not module.training
        or inp.dtype != torch.bfloat16
        or module.use_bias
        or getattr(module, "single_grouped_weight", False)
        or FP8GlobalStateManager.is_fp8_enabled()
        or FP8GlobalStateManager.is_fp8_calibration()
        or module.is_debug_iter()
        or module.te_quant_params is None
    ):
        return False

    recipe = module.te_quant_params.training_recipe
    return (
        recipe.fp8_quantization_recipe == Fp8Recipe.blockwise
        and recipe.fp8_format == "e4m3"
        and recipe.fp8_param
        and recipe.preserve_high_precision_init_val is False
        and recipe.fp8_block_scaling_fp32_scales
    )


def _get_materialization_inputs(
    module: Any, inp: Tensor
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]] | None:
    """Collect the native FP8 payloads and scales for every local expert."""
    assert Float8BlockwiseQTensor is not None
    payloads = []
    scales = []
    for gemm_idx in range(module.num_gemms):
        weight = getattr(module, f"weight{gemm_idx}")
        if (
            not isinstance(weight, Float8BlockwiseQTensor)
            or weight.requires_grad
            or weight.dtype != torch.bfloat16
            or hasattr(weight, "materialize_group_for_forward")
        ):
            return None

        rowwise_data = weight._rowwise_data
        rowwise_scale = weight._rowwise_scale_inv
        if (
            rowwise_data is None
            or rowwise_scale is None
            or weight.ndim != 2
            or rowwise_data.dtype != torch.uint8
            or rowwise_data.numel() != weight.numel()
            or not rowwise_data.is_contiguous()
            or rowwise_scale.dtype != torch.float32
            or weight.shape[0] % _BLOCK_SIZE != 0
            or weight.shape[1] % _BLOCK_SIZE != 0
            or rowwise_data.device != inp.device
            or rowwise_scale.device != inp.device
        ):
            return None

        block_rows = weight.shape[0] // _BLOCK_SIZE
        block_columns = weight.shape[1] // _BLOCK_SIZE
        if rowwise_scale.shape[0] < block_rows or rowwise_scale.shape[1] < block_columns:
            return None
        payloads.append(rowwise_data.view(torch.float8_e4m3fn))
        scales.append(rowwise_scale[:block_rows, :block_columns])

    return tuple(payloads), tuple(scales)


def _forward_with_external_weights(
    module: Any,
    inp: Tensor,
    m_splits: list[int],
    grouped_weights: Tensor,
    is_first_microbatch: bool | None,
) -> Tensor:
    """Run TE's grouped BF16 GEMM with ephemeral externally materialized weights."""
    assert _TEGroupedLinearAutograd is not None
    assert is_cpu_offload_enabled is not None
    if len(m_splits) != module.num_gemms:
        raise ValueError(
            f"Number of splits ({len(m_splits)}) should match number of "
            f"GEMMs ({module.num_gemms})."
        )

    is_grad_enabled = torch.is_grad_enabled()
    inp = module.prepare_forward(inp, num_gemms=module.num_gemms)
    try:
        assert not module.fp8
        assert not module.fp8_calibration
        (
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
        ) = module._get_quantizers()

        # Positional layout expected by TE 2.16's private _GroupedLinear function.
        non_tensor_args = (
            m_splits,
            False,  # use_bias
            is_first_microbatch,
            False,  # fp8
            False,  # fp8_calibration
            module.wgrad_store,
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
            module.fuse_wgrad_accumulation,
            is_cpu_offload_enabled(),
            module.sequence_parallel,
            module.activation_dtype,
            is_grad_enabled,
            [None] * module.num_gemms,  # weight_workspaces; unused when fp8=False
            False,  # cache_weight; unused when fp8=False
            None,  # skip_fp8_weight_update
            module.save_original_input,
            False,  # debug
        )
        weights = list(grouped_weights.unbind(dim=0))
        biases = [inp.new_empty(0) for _ in weights]
        if is_grad_enabled:
            out, _ = _TEGroupedLinearAutograd.apply(inp, non_tensor_args, *weights, *biases)
        else:
            out, _ = _TEGroupedLinearAutograd.forward(None, inp, non_tensor_args, *weights, *biases)
        return out
    finally:
        module.end_forward()


def try_frozen_blockwise_bf16_forward(
    module: Any, inp: Tensor, m_splits: list[int], *, is_first_microbatch: bool | None
) -> Tensor | None:
    """Return the optimized output, or ``None`` to use TE's public forward path."""
    if not _PRIVATE_ABI_SUPPORTED or not _is_eligible(module, inp):
        return None

    materialization_inputs = _get_materialization_inputs(module, inp)
    if materialization_inputs is None:
        return None

    grouped_weights = _get_compiled_materializer()(*materialization_inputs)
    return _forward_with_external_weights(
        module, inp, m_splits, grouped_weights, is_first_microbatch
    )
