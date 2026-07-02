# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""GLM-5.2 fused DSA core attention (Baseten additive module).

This lives in a separate file from the upstream ``dsa.py`` on purpose: GLM-5.2 support
stays *additive* so it never edits the actively-developed upstream DSA module, which
minimizes rebase conflicts against NVIDIA dev. It only imports the shared DSA primitives
from ``dsa`` / ``dsa_kernels``.

The IndexShare knobs (``dsa_indexer_topk_freq`` / ``dsa_indexer_skip_topk_offset``) are read
from the config with ``getattr`` defaults rather than declared as ``TransformerConfig``
fields, so this module requires no edits to the upstream config either; the GLM bridge sets
them as plain attributes on the provider.
"""

import json
import os
import math
from typing import Optional

import torch

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAttentionSubmodules
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    build_flat_topk_idxs,
    dsa_sparse_attn,
    indexer_topk,
)
from megatron.core.transformer.experimental_attention_variant.prime_fp8_indexer import (
    prime_fp8_indexer_topk,
)
from megatron.core.transformer.experimental_attention_variant.glm_absorbed_mla import (
    GlmAbsorbedMLASelfAttention,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


def _parse_int_set_env(name: str) -> Optional[set[int]]:
    value = os.getenv(name)
    if not value:
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def _dump_dsa_topk_if_requested(
    *,
    layer_number: int,
    skip_topk: bool,
    source_layer: int,
    topk_local: torch.Tensor,
    indexer_scores: Optional[torch.Tensor],
    seqlen_kv: int,
    backend: str,
) -> None:
    dump_dir = os.getenv("TRAINERS_DSA_TOPK_DUMP_DIR")
    if not dump_dir:
        return

    layers = _parse_int_set_env("TRAINERS_DSA_TOPK_LAYERS")
    if layers is not None and layer_number not in layers:
        return

    query_last_n = int(os.getenv("TRAINERS_DSA_TOPK_QUERY_LAST_N", "16"))
    k = int(os.getenv("TRAINERS_DSA_TOPK_K", "256"))
    query_start = max(0, topk_local.shape[1] - query_last_n)
    query_positions = list(range(query_start, topk_local.shape[1]))
    topk_slice = topk_local[:, query_start:, : min(k, topk_local.shape[-1])]
    topk_score_slice = None
    score_rows_slice = None
    score_stats = None
    if indexer_scores is not None:
        score_query_slice = indexer_scores[:, query_start:, :]
        if os.getenv("TRAINERS_DSA_SCORE_ROWS_DUMP", "0") == "1":
            score_rows_slice = score_query_slice
        gather_idx = topk_slice.to(torch.long).clamp(min=0, max=indexer_scores.shape[-1] - 1)
        topk_score_slice = torch.gather(score_query_slice, dim=-1, index=gather_idx)
        topk_score_slice = torch.where(
            topk_slice >= 0,
            topk_score_slice,
            torch.full_like(topk_score_slice, float("nan")),
        )
        finite_mask = torch.isfinite(score_query_slice)
        finite_count = finite_mask.sum(dim=-1).clamp(min=1)
        finite_zero = torch.where(finite_mask, score_query_slice, torch.zeros_like(score_query_slice))
        finite_mean = finite_zero.sum(dim=-1) / finite_count
        finite_var = (
            torch.where(
                finite_mask,
                (score_query_slice - finite_mean.unsqueeze(-1)) ** 2,
                torch.zeros_like(score_query_slice),
            ).sum(dim=-1)
            / finite_count
        )
        score_stats = {
            "min": torch.where(
                finite_mask, score_query_slice, torch.full_like(score_query_slice, float("inf"))
            )
            .min(dim=-1)
            .values.detach()
            .cpu()
            .float()
            .tolist(),
            "max": torch.where(
                finite_mask, score_query_slice, torch.full_like(score_query_slice, float("-inf"))
            )
            .max(dim=-1)
            .values.detach()
            .cpu()
            .float()
            .tolist(),
            "mean": finite_mean.detach().cpu().float().tolist(),
            "std": torch.sqrt(finite_var).detach().cpu().float().tolist(),
        }

    os.makedirs(dump_dir, exist_ok=True)
    rank = int(os.getenv("RANK", "0"))
    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    dp_rank = parallel_state.get_data_parallel_rank()
    cp_rank = parallel_state.get_context_parallel_rank()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    path = os.path.join(
        dump_dir,
        f"rank{rank}_pp{pp_rank}_tp{tp_rank}_dp{dp_rank}_cp{cp_rank}_ep{ep_rank}"
        f"_layer{layer_number}_{backend}.json",
    )
    payload = {
        "engine": "megatron",
        "rank": rank,
        "parallel_ranks": {
            "pipeline": pp_rank,
            "tensor": tp_rank,
            "data": dp_rank,
            "context": cp_rank,
            "expert": ep_rank,
        },
        "backend": backend,
        "layer_number": layer_number,
        "skip_topk": skip_topk,
        "source_layer": source_layer,
        "seq_len": seqlen_kv,
        "query_positions": query_positions,
        "topk_k": int(topk_slice.shape[-1]),
        "topk_shape": list(topk_local.shape),
        "topk_checksum": int(topk_slice.detach().to(torch.int64).sum().item()),
        "topk": topk_slice.detach().cpu().to(torch.int32).tolist(),
    }
    if topk_score_slice is not None:
        payload["topk_scores"] = topk_score_slice.detach().cpu().float().tolist()
        payload["score_stats"] = score_stats
    if score_rows_slice is not None:
        payload["score_rows"] = score_rows_slice.detach().cpu().float().tolist()
        payload["score_shape"] = list(score_query_slice.shape)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result.

    Layers are 1-indexed. The first ``skip_topk_offset`` layers always compute their own
    top-k (they own an indexer). After that, every ``topk_freq``-th layer computes its
    own top-k; the layers in between reuse the top-k indices from the most recent
    computing layer (GLM-5.2 "IndexShare").
    """
    if layer_number < 1:
        raise ValueError(f"layer_number must be 1-indexed and positive, got {layer_number}.")
    if topk_freq < 1:
        raise ValueError(f"topk_freq must be positive, got {topk_freq}.")
    return (max(layer_number - skip_topk_offset, 0) % topk_freq) != 0


def source_dsa_compute_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> int:
    """Return the computing layer whose DSA top-k a skip layer reuses."""
    is_dsa_skip_topk_layer(layer_number, skip_topk_offset, topk_freq)
    if layer_number <= skip_topk_offset:
        return layer_number
    return layer_number - ((layer_number - skip_topk_offset) % topk_freq)


class DSAttentionFused(MegatronModule):
    """GLM-5.2 fused DSA core: frozen-indexer top-k + FlashMLA/cuDNN sparse attention.

    Drop-in core_attention for an absorbed-MLA outer (same signature/submodules as
    ``DSAttention``), so it receives the absorbed query ``[sq, b, np, hn]`` and the
    single-head compressed KV ``[sq, b, 1, v_head_dim]`` plus ``x``/``qr`` for the
    indexer. Uses the production ``dsa_kernels`` primitives (no compression, no
    windowing) and supports cross-layer top-k sharing (GLM-5.2 "IndexShare").

    The indexer is frozen for GLM-5.2 (``dsa_indexer_loss_coeff == 0``), so the top-k
    is always selected with the inference kernel ``indexer_topk`` (no loss, no backward);
    the training/score-recompute path is never used.
    """

    _HOLDER_ATTR = "_dsa_index_share_topk_holder"
    """layer_number -> top-k indices map for IndexShare. Packed batches store it on
    PackedSeqParams (isolated per microbatch); non-packed store it on the config for the
    duration of the forward."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        is_mtp_layer: bool = False,
    ):
        super().__init__(config=config)

        if config.tensor_model_parallel_size != 1:
            raise ValueError("Fused DSA (FlashMLA sparse) currently requires TP=1.")
        if config.context_parallel_size != 1:
            raise ValueError("Fused DSA (FlashMLA sparse) currently requires CP=1.")

        self.layer_number = layer_number
        if is_mtp_layer:
            self.layer_number = self.layer_number + self.config.num_layers

        # Cross-layer top-k sharing (IndexShare): computing layers own an indexer and
        # produce fresh top-k; skip layers reuse the most recent computing layer's top-k.
        # The freq/offset knobs are read via getattr with GLM-5 defaults (freq=1 disables
        # sharing, offset=0) so this module needs no upstream TransformerConfig fields; the
        # GLM bridge sets them as plain attrs. Validate here since the config no longer does.
        self.index_topk = config.dsa_indexer_topk
        self.index_topk_freq = getattr(config, "dsa_indexer_topk_freq", 1)
        self.index_skip_topk_offset = getattr(config, "dsa_indexer_skip_topk_offset", 0)
        if self.index_topk_freq < 1:
            raise ValueError(
                f"dsa_indexer_topk_freq must be positive, got {self.index_topk_freq}."
            )
        if self.index_skip_topk_offset < 0:
            raise ValueError(
                "dsa_indexer_skip_topk_offset must be non-negative, got "
                f"{self.index_skip_topk_offset}."
            )
        self.index_share = self.index_topk_freq > 1
        self.skip_topk = self.index_share and is_dsa_skip_topk_layer(
            self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
        )
        self.source_layer = (
            source_dsa_compute_layer(
                self.layer_number, self.index_skip_topk_offset, self.index_topk_freq
            )
            if self.index_share
            else self.layer_number
        )

        self.indexer = None
        if not self.skip_topk:
            self.indexer = build_module(
                submodules.indexer, config=self.config, pg_collection=pg_collection
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

        # GLM-5.2 has no learnable attention sink; dsa_sparse_attn still needs an (np,) bias.
        self.register_buffer(
            "attn_sink",
            torch.zeros(config.num_attention_heads, dtype=torch.float32),
            persistent=False,
        )

    def _get_index_share_topk_holder(
        self, packed_seq_params: Optional[PackedSeqParams]
    ) -> "dict[int, torch.Tensor]":
        holder_owner = packed_seq_params if packed_seq_params is not None else self.config
        holder = getattr(holder_owner, self._HOLDER_ATTR, None)
        if holder is None:
            holder = {}
            setattr(holder_owner, self._HOLDER_ATTR, holder)
        return holder

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        x: torch.Tensor = None,
        qr: torch.Tensor = None,
        up_v_weight: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Fused frozen-indexer sparse attention with IndexShare.

        Keyword names match ``AbsorbedMLASelfAttention``'s core_attention call, which passes
        ``value``/``attention_mask``/``x``/``qr``/``up_v_weight``/``position_ids``/
        ``packed_seq_params``/``attn_mask_type`` by keyword. ``query``: ``[sq, b, np,
        k_channels]`` absorbed query (k_channels = kv_lora_rank + rope); ``key``: ``[sq, b, 1,
        k_channels]`` single-head compressed KV (value == key under MQA). ``up_v_weight`` and
        ``position_ids`` are unused here: the outer absorbed-MLA applies the V up-projection
        after core attention, and the indexer derives RoPE positions internally.
        """
        b = query.size(1)
        kv = key.squeeze(-2) if key.dim() == 4 else key  # [sq, b, k_channels]
        seqlen_kv = kv.size(0)

        holder = (
            self._get_index_share_topk_holder(packed_seq_params) if self.index_share else None
        )

        if self.skip_topk:
            # IndexShare skip layer: reuse the source computing layer's top-k.
            if holder is None or self.source_layer not in holder:
                raise AssertionError(
                    f"DSA IndexShare skip layer (layer_number={self.layer_number}) needs top-k "
                    f"from source computing layer {self.source_layer}, but it did not run before "
                    "this layer in this pipeline stage. Cross-PP top-k sharing is not supported. "
                    f"Holder has layers {sorted(holder or {})}."
            )
            topk_local = holder[self.source_layer]
            indexer_scores = None
            indexer_backend = "indexshare"
        else:
            # Computing layer: frozen-indexer inference top-k (no loss, no backward).
            assert self.indexer is not None
            q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                x.detach(), qr.detach(), packed_seq_params
            )
            indexer_backend = os.getenv("TRAINERS_DSA_INDEXER_BACKEND", "cudnn")
            if indexer_backend == "prime_fp8":
                topk_local, _ = prime_fp8_indexer_topk(
                    q_idx,
                    k_idx,
                    w_idx,
                    min(self.index_topk, seqlen_kv),
                    indexer_softmax_scale=self.indexer.softmax_scale,
                )
                indexer_scores = None
            else:
                indexer_backend = "cudnn"
                topk_local, _, indexer_scores = indexer_topk(
                    q_idx,
                    k_idx,
                    w_idx,
                    min(self.index_topk, seqlen_kv),
                    ratio=1,  # no compression
                    indexer_softmax_scale=self.indexer.softmax_scale,
                    return_scores=True,
                )
            if holder is not None:
                holder[self.layer_number] = topk_local

        _dump_dsa_topk_if_requested(
            layer_number=self.layer_number,
            skip_topk=self.skip_topk,
            source_layer=self.source_layer,
            topk_local=topk_local,
            indexer_scores=indexer_scores,
            seqlen_kv=seqlen_kv,
            backend=indexer_backend,
        )

        flat_idxs, flat_tlen = build_flat_topk_idxs(
            topk_local, batch_size=b, seqlen_kv=seqlen_kv, compact=True
        )
        # dsa_sparse_attn (FlashMLA convention) attends with the full absorbed query/key dim
        # (kv_lora_rank + rope) but returns only the latent value subspace
        # [sq, b, np * kv_lora_rank], which is exactly what the outer absorbed-MLA V
        # up-projection consumes.
        output = dsa_sparse_attn(
            query,
            kv,
            self.attn_sink.float(),
            flat_idxs,
            self.softmax_scale,
            topk_length=flat_tlen,
        )
        return output


def build_glm_dsa_fused_attention_spec(backend, qk_norm, indexer):
    """Build the GLM-5.2 fused-DSA self-attention ModuleSpec (absorbed MLA + fused DSA core).

    Additive Baseten entry point so the GLM fused spec lives here rather than in the upstream
    ``experimental_attention_variant_module_specs`` builder, keeping that file merge-clean
    against NVIDIA dev. ``backend``/``qk_norm``/``indexer`` are supplied by the shared upstream
    builder, so the linear submodules and HF weight mapping stay identical to the unfused path.
    """
    core_attention = ModuleSpec(
        module=DSAttentionFused,
        submodules=DSAttentionSubmodules(indexer=indexer),
    )
    return ModuleSpec(
        module=GlmAbsorbedMLASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=AbsorbedMLASelfAttentionSubmodules(
            linear_q_proj=backend.column_parallel_linear(),
            linear_q_down_proj=backend.linear(),
            linear_q_up_proj=backend.column_parallel_linear(),
            linear_kv_down_proj=backend.linear(),
            linear_kv_up_proj=backend.column_parallel_linear(),
            core_attention=core_attention,
            linear_proj=backend.row_parallel_linear(),
            q_layernorm=qk_norm,
            kv_layernorm=qk_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )
