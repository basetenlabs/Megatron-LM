# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""GLM LoRA-aware absorbed MLA.

The absorbed path consumes ``linear_kv_up_proj`` as a raw weight rather than
calling its ``forward``. When Bridge wraps that projection with a LoRA adapter,
fold the adapter into the effective weight so the adapter participates in the
absorbed K/V path.
"""

import torch

from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)


class GlmAbsorbedMLASelfAttention(AbsorbedMLASelfAttention):
    """Absorbed MLA that folds a LoRA adapter into the KV up-projection weight."""

    def _kv_up_proj_weight(self) -> torch.Tensor:
        module = self.linear_kv_up_proj
        if not hasattr(module, "to_wrap"):
            return module.weight

        weight = module.to_wrap.weight
        if not getattr(module, "_adapter_enabled", True):
            return weight
        if self.config.tensor_model_parallel_size != 1:
            raise NotImplementedError(
                "LoRA on the absorbed KV up-projection is only supported with TP=1."
            )

        adapter = module.adapter
        lora_a = adapter.linear_in.weight
        lora_b = adapter.linear_out.weight
        scale = getattr(adapter, "scale", None)
        if scale is None:
            scale = adapter.alpha / adapter.dim
        return weight + scale * (lora_b @ lora_a)
