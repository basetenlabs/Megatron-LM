# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""GLM-5.2 fused DSA core attention (Baseten additive module).

This lives in a separate file from the upstream ``dsa.py`` on purpose: GLM-5.2 support
stays *additive* so it never edits the actively-developed upstream DSA module, which
minimizes rebase conflicts against NVIDIA dev. It only imports the shared DSA primitives
from ``dsa`` / ``dsa_kernels``.

The IndexShare knobs (``dsa_indexer_topk_freq`` / ``dsa_indexer_skip_topk_offset``) are
declared on ``TransformerConfig`` so values set by the GLM bridge survive provider-to-config
conversion.
"""

import math
from typing import Optional

import torch

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils as cp_utils
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttentionSubmodules,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import (
    build_flat_topk_idxs,
    dsa_sparse_attn,
    indexer_topk,
)
from megatron.core.transformer.experimental_attention_variant.glm_absorbed_mla import (
    GlmAbsorbedMLASelfAttention,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


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
        if config.context_parallel_size > 1:
            # CP support: contiguous THD partitioning only (the trainer-side
            # pack_thd_cp_microbatch path). See _forward_thd_cp below.
            if getattr(config, "cp_partition_mode", "zigzag") != "contiguous":
                raise ValueError(
                    "Fused DSA with CP>1 requires cp_partition_mode='contiguous'."
                )
            if pg_collection is None or getattr(pg_collection, "cp", None) is None:
                raise ValueError("Fused DSA with CP>1 requires a cp process group.")
        self.pg_collection = pg_collection

        self.layer_number = layer_number
        if is_mtp_layer:
            self.layer_number = self.layer_number + self.config.num_layers

        # Cross-layer top-k sharing (IndexShare): computing layers own an indexer and
        # produce fresh top-k; skip layers reuse the most recent computing layer's top-k.
        self.index_topk = config.dsa_indexer_topk
        self.index_topk_freq = config.dsa_indexer_topk_freq
        self.index_skip_topk_offset = config.dsa_indexer_skip_topk_offset
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
        cp_group = (
            getattr(self.pg_collection, "cp", None) if self.pg_collection is not None else None
        )
        if cp_group is not None and cp_group.size() > 1:
            return self._forward_thd_cp(query, key, x, qr, packed_seq_params, cp_group)

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
        else:
            # Computing layer: frozen-indexer inference top-k (no loss, no backward).
            assert self.indexer is not None
            q_idx, k_idx, w_idx = self.indexer.forward_before_topk(
                x.detach(), qr.detach(), packed_seq_params
            )
            topk_local, _ = indexer_topk(
                q_idx,
                k_idx,
                w_idx,
                min(self.index_topk, seqlen_kv),
                ratio=1,  # no compression
                indexer_softmax_scale=self.indexer.softmax_scale,
            )
            if holder is not None:
                holder[self.layer_number] = topk_local

        # PR#5087's build_flat_topk_idxs dropped the seqlen_kv arg (out-of-range
        # masking now happens inside indexer_topk); SBHD semantics are unchanged.
        flat_idxs, flat_tlen = build_flat_topk_idxs(
            topk_local, batch_size=b, compact=True
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

    # ------------------------------------------------------------------
    # Context-parallel THD path (contiguous partitioning)
    # ------------------------------------------------------------------

    def _indexer_qkw_cp(self, x, qr, cu_seqlens, max_seqlen_q, global_start, l_local):
        """CP-aware re-derivation of ``Indexer.forward_before_topk``.

        Identical math to the CP=1 path except RoPE positions: each local row's
        frequency row is its position *within its packed sequence*, recovered
        from the GLOBAL ``cu_seqlens`` and this rank's contiguous row interval
        (``Indexer._apply_rope`` would use local 0..l_local-1, wrong for
        cp_rank > 0). The indexer is frozen (inputs detached), so no autograd
        flows through any of this.
        """
        indexer = self.indexer
        x = x.detach()
        qr = qr.detach()

        # Table must cover positions within PADDED sequences (cu_seqlens is the
        # padded layout; capacity-padding rows can sit past max_seqlen_q, the max
        # over REAL datum lengths). The global padded row count upper-bounds any
        # within-sequence position; clamp defensively — overflow rows are padding.
        rotary_len = max(int(max_seqlen_q), l_local * cp_group.size())
        if indexer.config.rope_type == "rope":
            table = indexer.rotary_pos_emb(rotary_len, packed_seq=False)
            mscale = 1.0
        else:
            table, mscale = indexer.rotary_pos_emb(rotary_len, packed_seq=False)
        position_ids = cp_utils._thd_cp_position_ids(cu_seqlens, global_start, l_local)
        position_ids = position_ids.long().clamp_(0, table.shape[0] - 1)
        freqs = torch.index_select(table, 0, position_ids)  # [t, 1, 1, d]

        # THD shapes: x [t, 1, hidden], qr [t, q_lora] (packed squeeze) or [t, 1, q_lora].
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if qr.dim() == 2:
            qr = qr.unsqueeze(1)
        seqlen, bsz = x.shape[0], x.shape[1]

        q, _ = indexer.linear_wq_b(qr)
        q = q.reshape(seqlen, bsz, indexer.index_n_heads, indexer.index_head_dim)
        q = self._indexer_rope_cp(q, freqs, mscale)

        k, _ = indexer.linear_wk(x)
        k = indexer.k_norm(k)
        k = k.reshape(seqlen, bsz, 1, indexer.index_head_dim)
        k = self._indexer_rope_cp(k, freqs, mscale)
        k = k.reshape(seqlen, bsz, indexer.index_head_dim)

        q = rotate_activation(q)
        k = rotate_activation(k)

        weights, _ = indexer.linear_weights_proj(x)
        weights = weights * (indexer.index_n_heads**-0.5) * indexer.softmax_scale

        # Fold bsz=1: q [t, h, d], k [t, d], w [t, h].
        return q.reshape(seqlen, indexer.index_n_heads, indexer.index_head_dim), k.reshape(
            seqlen, indexer.index_head_dim
        ), weights.reshape(seqlen, indexer.index_n_heads)

    def _indexer_rope_cp(self, t, freqs, mscale):
        """``Indexer._apply_rope`` with explicit per-row frequency rows.

        Mirrors the split convention (pe FIRST, then nope) and flags
        (``mla_rotary_interleaved=False``) exactly.
        """
        indexer = self.indexer
        t_pe, t_nope = torch.split(
            t,
            [indexer.qk_pos_emb_head_dim, indexer.index_head_dim - indexer.qk_pos_emb_head_dim],
            dim=-1,
        )
        t_pe = _apply_rotary_pos_emb_bshd(
            t_pe,
            freqs,
            rotary_interleaved=self.config.rotary_interleaved,
            mla_rotary_interleaved=False,
            mscale=mscale,
        )
        return torch.cat([t_pe, t_nope], dim=-1)

    def _forward_thd_cp(self, query, key, x, qr, packed_seq_params, cp_group):
        """THD-packed contiguous context-parallel forward.

        Each CP rank holds contiguous packed rows [r*l_local, (r+1)*l_local).
        The compressed latent KV (~576 dims) is all-gathered over the CP group
        (autograd: backward reduce-scatters dKV); the frozen indexer's K rows
        are gathered without grad; every local query then runs top-k against
        the full global KV with causal offsets, and FlashMLA/cuDNN sparse
        attention consumes flat global indices. IndexShare holders store
        (topk, layout) per computing layer, per-rank consistent.
        """
        psp = packed_seq_params
        if psp is None or psp.qkv_format != "thd":
            raise ValueError(
                "GLM fused DSA with CP>1 requires THD packed inputs "
                "(the trainer's pack_thd_cp_microbatch path)."
            )
        cu_seqlens = (
            psp.cu_seqlens_q_padded if psp.cu_seqlens_q_padded is not None else psp.cu_seqlens_q
        )
        max_seqlen_q = int(psp.max_seqlen_q)
        cp_rank = cp_group.rank()
        l_local = query.shape[0]
        global_start = cp_rank * l_local

        # ---- KV: [t, 1, d] -> [t, d]; autograd all-gather over CP ----------
        # (contiguous partitioning => rank-major concat IS sequence-major).
        kv_local = key.reshape(l_local, key.shape[-1])
        kv_global = gather_from_sequence_parallel_region(kv_local, group=cp_group)

        holder = (
            self._get_index_share_topk_holder(psp) if self.index_share else None
        )
        if self.skip_topk:
            if holder is None or self.source_layer not in holder:
                raise AssertionError(
                    f"DSA IndexShare skip layer (layer_number={self.layer_number}) needs top-k "
                    f"from source computing layer {self.source_layer}, but it did not run "
                    "before this layer in this pipeline stage. Cross-PP top-k sharing is not "
                    f"supported. Holder has layers {sorted(holder or {})}."
                )
            topk_local, layout = holder[self.source_layer]
        else:
            assert self.indexer is not None
            q_idx, k_idx, w_idx = self._indexer_qkw_cp(
                x, qr, cu_seqlens, max_seqlen_q, global_start, l_local
            )
            # Frozen indexer: plain (non-autograd) all-gather of K rows.
            k_idx = k_idx.contiguous()
            k_idx_global = k_idx.new_empty((k_idx.shape[0] * cp_group.size(),) + k_idx.shape[1:])
            torch.distributed.all_gather_into_tensor(k_idx_global, k_idx, group=cp_group)
            topk_local, layout = cp_utils.compute_cp_indexer_topk(
                q_idx,
                w_idx,
                k_idx_global,
                cu_seqlens,
                cu_seqlens,  # ratio=1: compressed rows == token rows
                global_start,
                1,  # ratio: no compression
                min(self.index_topk, max_seqlen_q),
                self.indexer.softmax_scale,
                max_seqlen_q=max_seqlen_q,
                use_fused=True,
            )
            if topk_local is None:
                raise RuntimeError("GLM fused DSA CP top-k returned no indices.")
            if holder is not None:
                holder[self.layer_number] = (topk_local, layout)

        cu_q_topk, cu_k_topk, _q_causal_offsets = layout
        flat_idxs, flat_tlen = build_flat_topk_idxs(
            topk_local,
            batch_size=1,
            compact=True,
            cu_seqlens_q=cu_q_topk,
            cu_seqlens_kv=cu_k_topk,
        )
        output = dsa_sparse_attn(
            query,
            kv_global,
            self.attn_sink.float(),
            flat_idxs,
            self.softmax_scale,
            topk_length=flat_tlen,
            is_thd=True,
        )
        # Outer absorbed-MLA expects [t, 1, np * kv_lora_rank] under packing.
        return output.unsqueeze(1)


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
