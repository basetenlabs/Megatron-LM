# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Experimental zero-copy BF16 grouped GEMMs for frozen FSDP expert weights."""

from typing import Any

import torch


def _packed_view(module: torch.nn.Module, num_gemms: int) -> torch.Tensor:
    weights = [getattr(module, f"weight{i}") for i in range(num_gemms)]
    first = weights[0]
    n, k = first.shape
    stride_bytes = n * k * first.element_size()
    for i, weight in enumerate(weights):
        if (
            weight.requires_grad
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
            or weight.shape != first.shape
            or weight.data_ptr() != first.data_ptr() + i * stride_bytes
        ):
            raise RuntimeError(
                "Frozen grouped-MM requires contiguous unsharded BF16 expert weights"
            )
    required_bytes = (first.storage_offset() + num_gemms * n * k) * first.element_size()
    if first.untyped_storage().nbytes() < required_bytes:
        raise RuntimeError("Expert weights do not share a sufficiently large backing allocation")
    return first.as_strided((num_gemms, n, k), (n * k, k, 1))


class _FrozenGroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, inp: torch.Tensor, offsets: torch.Tensor, module: torch.nn.Module, num_gemms: int
    ) -> torch.Tensor:
        ctx.module = module
        ctx.num_gemms = num_gemms
        ctx.save_for_backward(offsets)
        return torch.nn.functional.grouped_mm(
            inp, _packed_view(module, num_gemms).transpose(1, 2), offs=offsets
        )

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        if torch.is_grad_enabled():
            raise NotImplementedError("Frozen grouped-MM supports first-order gradients only")
        (offsets,) = ctx.saved_tensors
        # FSDP can recycle the original allocation before backward. Rebuild the
        # view from the currently unsharded weights, never a saved weight view.
        result = torch.nn.functional.grouped_mm(
            grad.contiguous(), _packed_view(ctx.module, ctx.num_gemms), offs=offsets
        )
        return result, None, None, None


def try_frozen_bf16_grouped_mm(
    module: Any, inp: torch.Tensor, m_splits: list[int]
) -> torch.Tensor | None:
    """Use packed grouped-MM for eligible frozen experts; leave adapter GEMMs alone."""
    if not module.config.moe_use_torch_grouped_mm or module.num_gemms <= 1:
        return None
    if any(getattr(module, f"weight{i}").requires_grad for i in range(module.num_gemms)):
        return None
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    if (
        module.config.recompute_granularity != "full"
        or module.use_bias
        or module.tp_size != 1
        or inp.ndim != 2
        or inp.dtype != torch.bfloat16
        or FP8GlobalStateManager.is_fp8_enabled()
    ):
        raise RuntimeError("Frozen grouped-MM requires TP1, full recompute, BF16, and no bias/FP8")
    if len(m_splits) != module.num_gemms or sum(m_splits) != inp.shape[0]:
        raise ValueError("Expert row splits do not match the grouped-MM input")
    offsets = torch.tensor(m_splits, device=inp.device, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    return _FrozenGroupedMM.apply(inp, offsets, module, module.num_gemms)
