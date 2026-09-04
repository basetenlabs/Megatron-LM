# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field

import torch

from megatron.core.transformer.experimental_attention_variant.dsa_topk_cache import DSATopKCache


@dataclass(frozen=True)
class PackedAllGatherCPLayout:
    """Token positions and K/V reorder indices for packed all-gather CP."""

    query_positions: torch.Tensor
    key_reorder_indices: torch.Tensor


@dataclass
class DSAForwardContext:
    """Per-microbatch DSA state shared across layers and activation recomputation."""

    topk_cache: DSATopKCache = field(default_factory=DSATopKCache)
    packed_cp_layout: PackedAllGatherCPLayout | None = None
