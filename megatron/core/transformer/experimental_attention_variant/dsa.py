# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from megatron.core import parallel_state
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    apply_rotary_pos_emb,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    hadamard_transform = None

try:
    from flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

try:
    from cudnn import DSA
except ImportError:
    DSA = None


def is_dsa_skip_topk_layer(layer_number: int, skip_topk_offset: int, topk_freq: int) -> bool:
    """Return whether a 1-indexed layer reuses a previous DSA top-k result.

    Layers are 1-indexed. The first ``skip_topk_offset`` layers always compute their own
    top-k (they own an indexer). After that, every ``topk_freq``-th layer computes its
    own top-k; the layers in between reuse the top-k indices from the most recent
    computing layer.
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


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.
    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    assert (
        x.dtype == torch.bfloat16
    ), f"rotate_activation only support bf16 input, but got {x.dtype}"
    assert hadamard_transform is not None, "fast_hadamard_transform is not installed."
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DSAIndexerLossLoggingHelper:
    """Helper class for logging sparse attention indexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the indexer loss for logging.

        Args:
            loss: The loss tensor.
            layer_number: Layer index of the loss, 1-indexed.
            num_layers: The number of total layers.
            reduce_group: The group for reducing the loss.
            avg_group: The group for averaging the loss.
        """
        # Skip indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]

        torch.distributed.all_reduce(
            values, group=parallel_state.get_pipeline_model_parallel_group()
        )
        # Reduce indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer,
        wandb_writer=None,
        total_loss_dict=None,
        per_layer_logging: bool = False,
    ):
        """Track the sparse attention indexer metrics for logging.

        Args:
            loss_scale: Scale factor for the loss.
            iteration: Current training iteration.
            writer: TensorBoard writer.
            wandb_writer: Weights & Biases writer.
            total_loss_dict: Dictionary to accumulate total losses.
            per_layer_logging: Whether to log per-layer losses.
        """
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        indexer_loss_values = tracker["values"] * loss_scale
        num_layers = indexer_loss_values.shape[0]

        # Average across all layers (assuming all layers have sparse attention)
        avg_indexer_loss = indexer_loss_values.sum() / num_layers

        # Log average loss
        if total_loss_dict is not None:
            if "indexer loss" in total_loss_dict:
                total_loss_dict["indexer loss"] += avg_indexer_loss
            else:
                total_loss_dict["indexer loss"] = avg_indexer_loss

        if writer is not None:
            writer.add_scalar("indexer loss", avg_indexer_loss, iteration)

        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": avg_indexer_loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_collection: ProcessGroupCollection,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.

    This loss trains the indexer to predict which tokens are important by matching the distribution
    of true attention scores.

    Reference: Section 2.1 of
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf

    Args:
        index_scores: Scores predicted by indexer [batch, seqlen_q, seqlen_k].
        topk_indices: Top-k indices [batch, seqlen_q, index_topk].
        query: Query tensor [seqlen_q, batch, heads, dim].
        key: Key tensor [seqlen_k, batch, heads, dim].
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: bool, whether to use sparse indexer loss. If True, only the topk
            indices will be used to compute the loss.
        pg_collection: Process group collection, must have TP process group.

    Returns:
        index_loss: KL divergence loss (scalar).
    """
    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += causal_mask.view(1, 1, sq, sk)
    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores += index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores += index_mask

    # [b, np, sq, sk] -> [b, np, sq, sk]
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # [b, sq, sk] -> [b, sq, sk]
    index_scores = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)

    # Sum attention scores across heads.
    # [batch, heads, seqlen_q, seqlen_k] -> [batch, seqlen_q, seqlen_k]
    attention_scores = attention_scores.sum(dim=1)
    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores.contiguous(), group=pg_collection.tp)
    # L1 normalize target on the last dimension. Doesn't use abs() because attention_scores are
    # obtained from softmax so they are already non-negative.
    attention_scores = attention_scores / attention_scores.sum(dim=-1, keepdim=True)

    # Compute KL divergence: KL(target || index) = target(x) * log(target(x) / index(x))
    # kl_per_element [b, sq, sk]
    kl_per_element = attention_scores * (
        torch.log(attention_scores + 1e-10) - torch.log(index_scores + 1e-10)
    )

    # [b, sq, sk] -> [b, sq] -> [1]
    # Each token has same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()

    # Scale by coefficient.
    indexer_loss = kl_div * loss_coeff

    return indexer_loss


def _compute_index_scores(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py#L254-L274
    This is a BF16 implementation of the `fp8_index` logic:
        1. Compute attention scores: q @ k^T;
        2. Apply ReLU activation;
        3. Weight by attention weights;
        4. Sum across attention heads.

    Args:
        q: BF16 [seqlen_q, batch, index_n_heads, index_head_dim], the query tensor.
        weights: BF16 [seqlen_q, batch, index_n_heads], the attention weights.
        k: BF16 [seqlen_k, batch, index_head_dim], the key tensor.

    Returns:
        index_scores: FP32 [batch, seqlen_q, seqlen_k], the index scores.
    """
    # Compute attention scores: q @ k^T
    # [seqlen_q, batch, index_n_heads, index_head_dim] @ [seqlen_k, batch, index_head_dim]^T
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())

    # Apply ReLU activation.
    index_scores = torch.relu(index_scores)

    # Weight each head by attention weights.
    # [seqlen_q, batch, index_n_heads, seqlen_k] * [seqlen_q, batch, index_n_heads, 1]
    #   -> [seqlen_q, batch, index_n_heads, seqlen_k]
    index_scores = index_scores * weights.unsqueeze(-1)

    # Sum across attention heads.
    # [seqlen_q, batch, index_n_heads, seqlen_k] -> [seqlen_q, batch, seqlen_k]
    index_scores = index_scores.sum(dim=2)

    # Transpose to [batch, seqlen_q, seqlen_k].
    index_scores = index_scores.transpose(0, 1)

    return index_scores


def fused_qk_topk_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    mask: Optional[torch.Tensor] = None,
):
    """Naive implementation of QK Topk."""
    seqlen = q.size(0)
    # =========================================
    # Compute index scores
    # =========================================
    # [batch, seqlen, seqlen]
    index_scores = _compute_index_scores(q, weights, k)
    if mask is not None:
        assert mask.dtype == index_scores.dtype, "Mask dtype must match index scores dtype"
        index_scores = index_scores + mask

    # =========================================
    # Select top-k indices
    # =========================================
    topk_k = min(index_topk, seqlen)
    # [batch, seqlen, index_topk]
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]

    return index_scores, topk_indices


@triton.jit
def _dsa_indexer_score_kernel(
    q_ptr, k_ptr, w_ptr, o_ptr,
    q_start, q_len, seqlen_k,
    stride_qm, stride_qh, stride_qd,
    stride_kn, stride_kd,
    stride_wm, stride_wh,
    stride_om, stride_on,
    H: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Fused lightning-indexer score for one ``[BLOCK_M, BLOCK_N]`` output tile.

    ``score[m, n] = sum_h relu(q[m, h, :] . k[n, :]) * w[m, h]`` with a causal mask. ``k`` is
    shared across heads (multi-query), so it is loaded once per tile; the per-head relu and
    weighted head-sum are accumulated in fp32 registers, so the ``[H, M, N]`` intermediate is
    never materialized.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    m_valid = offs_m < q_len
    n_valid = offs_n < seqlen_k

    # k tile [BLOCK_N, D], shared across all query heads
    k = tl.load(
        k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=n_valid[:, None], other=0.0,
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in range(H):
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_qm + h * stride_qh + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None], other=0.0,
        )
        s = tl.maximum(tl.dot(q, tl.trans(k)), 0.0)  # per-head relu, fp32 accumulate
        w = tl.load(w_ptr + offs_m * stride_wm + h * stride_wh, mask=m_valid, other=0.0)
        acc += s * w[:, None]

    # causal: key n valid for the query at global position (q_start + m) iff n <= q_start + m
    keep = (offs_n[None, :] <= (q_start + offs_m)[:, None]) & n_valid[None, :]
    acc = tl.where(keep, acc, float("-inf"))

    tl.store(
        o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=m_valid[:, None] & n_valid[None, :],
    )


def _chunked_dsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    *,
    mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    query_block_size: int = 512,
) -> torch.Tensor:
    """Compute DSA frozen-indexer top-k with a fused Triton scoring kernel.

    ``score[m, n] = sum_h w[m, h] * relu(q[m, h] . k[n])`` over a causal range.
    ``_dsa_indexer_score_kernel`` fuses the per-head relu + weighted head-sum, accumulating over
    heads in fp32 so the ``[H, S, S]`` intermediate is never materialized. Chunking over the query
    dimension only gives one kernel-launch grid per query block (instead of ``(seqlen / 512) ** 2``
    Python-driven tiles), with dense logits bounded by ``query_block_size``; ``torch.topk`` then
    selects the indices (cheap; non-differentiable).

    Frozen-indexer assumptions (explicit, no fallback): microbatch size 1 and a causal mask.
    Violations raise rather than silently degrade.
    """
    seqlen_q, bsz, n_heads, head_dim = q.shape
    seqlen_k = k.size(0)
    topk_k = min(index_topk, seqlen_k)
    if topk_k < 1:
        raise ValueError(f"index_topk must be positive, got {index_topk}.")
    if bsz != 1:
        raise RuntimeError(f"fused DSA indexer requires microbatch size 1, got {bsz}.")
    if not causal or mask is not None:
        raise RuntimeError(
            "fused DSA indexer supports causal-only selection (mask must be None); "
            f"got causal={causal}, mask={'set' if mask is not None else None}."
        )

    BLOCK_M, BLOCK_N = 64, 128
    # Drop the size-1 batch dim: q -> [Sq, H, D], k -> [Sk, D], weights -> [Sq, H].
    q2 = q[:, 0].contiguous()
    k2 = k[:, 0].contiguous()
    w2 = weights[:, 0].float().contiguous()

    output_chunks = []
    for q_start in range(0, seqlen_q, query_block_size):
        q_end = min(q_start + query_block_size, seqlen_q)
        q_len = q_end - q_start
        qs, ws = q2[q_start:q_end], w2[q_start:q_end]
        logits = torch.empty((q_len, seqlen_k), dtype=torch.float32, device=q.device)
        grid = (triton.cdiv(q_len, BLOCK_M), triton.cdiv(seqlen_k, BLOCK_N))
        _dsa_indexer_score_kernel[grid](
            qs, k2, ws, logits,
            q_start, q_len, seqlen_k,
            qs.stride(0), qs.stride(1), qs.stride(2),
            k2.stride(0), k2.stride(1),
            ws.stride(0), ws.stride(1),
            logits.stride(0), logits.stride(1),
            H=n_heads, D=head_dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        output_chunks.append(logits.topk(topk_k, dim=-1)[1])  # [q_len, topk_k]

    return torch.cat(output_chunks, dim=0).unsqueeze(0)  # [1, Sq, topk_k]


class FlashMLASparseAttentionFunc(torch.autograd.Function):
    """Autograd bridge for FlashMLA sparse forward and cuDNN DSA sparse backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: Optional[torch.Tensor],
        softmax_scale: float,
        d_v: int,
    ) -> torch.Tensor:
        if flash_mla_sparse_fwd is None:
            raise RuntimeError("flash_mla_sparse_fwd is required for FlashMLA sparse attention.")

        output = flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            softmax_scale,
            d_v=d_v,
            topk_length=topk_length,
        )
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise RuntimeError("flash_mla_sparse_fwd must return at least (output, max_logits, lse).")

        out = output[0].contiguous()
        lse = output[2].contiguous()
        ctx.softmax_scale = softmax_scale
        ctx.kv_had_head_dim = kv.dim() == 3
        ctx.has_topk_length = topk_length is not None
        saved_topk_length = topk_length if topk_length is not None else torch.empty(0, device=q.device)
        ctx.save_for_backward(q, kv, out, lse, indices, saved_topk_length)
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        if DSA is None:
            raise RuntimeError("cuDNN DSA sparse attention backward is not importable.")

        q, kv, out, lse, indices, saved_topk_length = ctx.saved_tensors
        topk_length = saved_topk_length if ctx.has_topk_length else None
        kv_for_backward = kv.squeeze(1).contiguous() if kv.dim() == 3 else kv.contiguous()
        topk_idxs = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices.contiguous()
        attn_sink = torch.full((q.size(1),), float("-inf"), dtype=torch.float32, device=q.device)

        result = DSA.sparse_attention_backward_wrapper(
            q.contiguous(),
            kv_for_backward,
            out,
            dout.contiguous(),
            lse,
            attn_sink,
            topk_idxs,
            softmax_scale=ctx.softmax_scale,
            topk_length=topk_length,
        )
        torch.cuda.synchronize()

        dq = result["dq"]
        dkv = result["dkv"]
        if ctx.kv_had_head_dim:
            dkv = dkv.unsqueeze(1)
        return dq, dkv, None, None, None, None


def fwd_fused_indexer_loss_naive(
    q, weights, k, query, key, topk, softmax_scale, loss_coeff, mask, sparse_loss, pg_collection
):
    """Naive implementation of forward pass for indexer loss."""
    index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, topk, mask)

    indexer_loss = compute_dsa_indexer_loss(
        index_scores,
        topk_indices,
        query,
        key,
        softmax_scale,
        loss_coeff,
        sparse_loss,
        pg_collection,
    )

    return topk_indices, indexer_loss


def bwd_fused_indexer_loss_naive(
    q,
    weights,
    k,
    query,
    key,
    topk_indices,
    softmax_scale,
    loss_coeff,
    sparse_loss,
    grad_loss,
    pg_collection,
):
    """Naive implementation of backward pass for indexer loss."""
    index_scores = _compute_index_scores(q, weights, k)  # [B, Sq, Sk]

    sq, b, np, hn = query.size()
    sk = key.size(0)

    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query_reshaped = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [sk, b, np, hn] -> [b, np, hn, sk] -> [b * np, hn, sk]
    key_reshaped = key.permute(1, 2, 3, 0).reshape(b * np, hn, sk)
    # Compute attention scores [b * np, sq, sk]
    attention_scores = torch.bmm(query_reshaped.float(), key_reshaped.float()) * softmax_scale
    # Free reshaped tensors - no longer needed after bmm
    del query_reshaped, key_reshaped

    # Reshape to [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # causal_mask [sq, sk]
    causal_mask = torch.triu(
        torch.full((sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device),
        diagonal=1,
    )
    # index_mask [b, sq, sk]
    index_mask = torch.full(
        (b, sq, sk), float("-inf"), dtype=torch.float32, device=causal_mask.device
    ).scatter_(-1, topk_indices, 0)

    # Apply causal mask to both attention and index scores
    # [b, np, sq, skv] + [1, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores = attention_scores + causal_mask.view(1, 1, sq, sk)
    # [b, sq, sk] + [1, sq, sk] -> [b, sq, sk]
    index_scores = index_scores + causal_mask.unsqueeze(0)
    # Free causal_mask - no longer needed
    del causal_mask

    if sparse_loss:
        # [b, np, sq, sk] + [b, 1, sq, sk] -> [b, np, sq, sk]
        attention_scores = attention_scores + index_mask.view(b, 1, sq, sk)
        # [b, sq, sk] + [b, sq, sk] -> [b, sq, sk]
        index_scores = index_scores + index_mask

    # Compute softmax for both
    attention_scores_softmax = torch.nn.functional.softmax(
        attention_scores, dim=-1, dtype=torch.float32
    )
    # Free attention_scores immediately
    del attention_scores

    index_scores_softmax = torch.nn.functional.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Free index_scores - no longer needed after softmax
    del index_scores

    # Sum attention scores across heads: [b, np, sq, sk] -> [b, sq, sk]
    attention_scores_sum = attention_scores_softmax.sum(dim=1)
    # Free attention_scores_softmax
    del attention_scores_softmax

    if pg_collection.tp.size() > 1:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(attention_scores_sum.contiguous(), group=pg_collection.tp)

    # L1 normalize
    attention_scores_normalized = attention_scores_sum / attention_scores_sum.sum(
        dim=-1, keepdim=True
    )
    # Free attention_scores_sum - no longer needed after normalization
    del attention_scores_sum

    # Backward through loss = kl_div * loss_coeff
    # where kl_div = kl_per_element.sum(dim=-1).mean()
    grad_kl_div = grad_loss * loss_coeff  # scalar

    # Backward through mean: distribute gradient equally
    grad_kl_per_row = grad_kl_div / (b * sq)  # scalar value for each row

    # Backward through sum(dim=-1): broadcast back to [b, sq, sk]
    # Each element in a row contributes to the sum, so gradient is same for all
    grad_kl_per_element = grad_kl_per_row.view(1, 1, 1).expand(b, sq, sk)

    # Backward through kl_per_element = target * (log(target) - log(index))
    # ∂kl/∂index_softmax = -target / index_softmax
    grad_index_scores_softmax = (
        -attention_scores_normalized / (index_scores_softmax + 1e-10) * grad_kl_per_element
    )
    # Free attention_scores_normalized - no longer needed
    del attention_scores_normalized

    # Backward through softmax: ∂L/∂x = softmax * (∂L/∂softmax - sum(∂L/∂softmax * softmax))
    sum_grad = (grad_index_scores_softmax * index_scores_softmax).sum(dim=-1, keepdim=True)
    grad_index_scores_logits = index_scores_softmax * (grad_index_scores_softmax - sum_grad)
    # Free intermediate tensors
    del index_scores_softmax, grad_index_scores_softmax, sum_grad

    # Zero out gradients for masked positions
    # Create a mask for valid (non-masked) positions
    # Causal mask: position (i, j) is valid if j <= i
    causal_valid_mask = torch.tril(
        torch.ones((sq, sk), device=q.device, dtype=torch.bool)
    )  # [sq, sk]
    if sparse_loss:
        # Also apply index mask - only topk positions are valid
        index_valid_mask = index_mask == 0  # [b, sq, sk]
        del index_mask  # Free index_mask immediately after use
        valid_mask = causal_valid_mask.unsqueeze(0) & index_valid_mask  # [b, sq, sk]
        del index_valid_mask
    else:
        del index_mask  # Free index_mask even if not used for sparse_loss
        valid_mask = causal_valid_mask.unsqueeze(0).expand(b, sq, sk)  # [b, sq, sk]
    del causal_valid_mask

    grad_index_scores_logits = grad_index_scores_logits * valid_mask.float()
    del valid_mask

    # Transpose from [b, sq, sk] to [sq, b, sk]
    grad_index_scores = grad_index_scores_logits.transpose(0, 1)  # [sq, b, sk]
    del grad_index_scores_logits

    # Backward through sum over heads: expand gradient
    grad_weighted_scores = grad_index_scores.unsqueeze(2)  # [sq, b, 1, sk]
    del grad_index_scores

    # Compute forward values needed for backward
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k.float())  # [sq, b, h, sk]
    # Compute relu_mask before relu (saves memory vs keeping both scores and relu output)
    relu_mask = scores > 0
    scores_after_relu = torch.relu(scores)
    del scores

    # Backward through multiplication by weights: index_scores_per_head * weights
    # ∂L/∂weights = grad * relu_scores (sum over sk)
    grad_weights = (grad_weighted_scores * scores_after_relu).sum(dim=-1)  # [sq, b, h]

    # ∂L/∂relu_scores = grad * weights
    grad_scores_after_relu = grad_weighted_scores * weights.unsqueeze(-1)  # [sq, b, h, sk]
    del grad_weighted_scores, scores_after_relu

    # Backward through ReLU
    grad_scores = grad_scores_after_relu * relu_mask.float()  # [sq, b, h, sk]
    del grad_scores_after_relu, relu_mask

    # Backward through einsum 'sbhd,tbd->sbht'
    # ∂L/∂q = einsum('sbht,tbd->sbhd', grad_scores, k)
    grad_q = torch.einsum('sbht,tbd->sbhd', grad_scores, k.float())  # [sq, b, h, d]
    # ∂L/∂k = einsum('sbht,sbhd->tbd', grad_scores, q)
    grad_k = torch.einsum('sbht,sbhd->tbd', grad_scores, q.float())  # [sk, b, d]
    del grad_scores

    return grad_q.to(q.dtype), grad_weights.to(weights.dtype), grad_k.to(k.dtype)


class FusedDSAIndexerLoss(torch.autograd.Function):
    """Fused implementation of DSA Indexer Loss."""

    @staticmethod
    def forward(
        ctx,
        q,
        weights,
        k,
        query,
        key,
        softmax_scale,
        topk,
        loss_coeff,
        mask,
        sparse_loss,
        pg_collection,
    ):
        """
        Fused forward: index_scores never materialized in full.
        """
        topk_indices, loss = fwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk,
            softmax_scale,
            loss_coeff,
            mask,
            sparse_loss,
            pg_collection,
        )

        # Save for backward (recomputation strategy)
        ctx.save_for_backward(q, weights, k, query, key, topk_indices)
        ctx.softmax_scale = softmax_scale
        ctx.loss_coeff = loss_coeff
        ctx.sparse_loss = sparse_loss
        ctx.pg_collection = pg_collection

        return topk_indices, loss

    @staticmethod
    def backward(ctx, grad_topk_indices, grad_loss):
        """
        Backward: Recompute what we need.
        """
        q, weights, k, query, key, topk_indices = ctx.saved_tensors

        grad_q, grad_weights, grad_k = bwd_fused_indexer_loss_naive(
            q,
            weights,
            k,
            query,
            key,
            topk_indices,
            ctx.softmax_scale,
            ctx.loss_coeff,
            ctx.sparse_loss,
            grad_loss,
            ctx.pg_collection,
        )

        # query and key are detached in forward, so return None for their gradients
        return grad_q, grad_weights, grad_k, None, None, None, None, None, None, None, None


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for indexer loss.

    This custom autograd function attaches a KL divergence loss to the activation
    to train the indexer to predict attention scores without affecting the forward pass.
    """

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output: The output tensor (activation).
            indexer_loss: The indexer KL divergence loss tensor.

        Returns:
            torch.Tensor: The output tensor unchanged.
        """
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output: The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                gradient.
        """
        (indexer_loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=indexer_loss.device
            )
        indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_indexer_loss_grad = torch.ones_like(indexer_loss) * indexer_loss_backward_scale
        return grad_output, scaled_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the scale of the indexer loss.

        Args:
            scale: The scale value to set.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


@dataclass
class DSAIndexerSubmodules:
    """
    Configuration class for specifying the submodules of an DSA Indexer.

    Args:
        linear_wq_b: Linear projection for query bottleneck expansion.
        linear_wk: Linear projection for key.
        k_norm: Layer normalization for key.
        linear_weights_proj: Linear projection for attention weights.
    """

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


@dataclass
class DSAttentionSubmodules:
    """
    Configuration class for specifying the submodules of DSAttention.

    Args:
        indexer: DSA Indexer module for computing sparse attention indices.
    """

    indexer: Union[ModuleSpec, type] = None


class DSAIndexer(MegatronModule):
    """
    DSA Lightning Indexer for DeepSeek Sparse Attention.

    Computes index scores to identify the top-k most relevant key-value pairs for each query in
    sparse attention.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L431-L480
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DSAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the indexer.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            submodules (DSAIndexerSubmodules): Indexer submodules specification.
            pg_collection (ProcessGroupCollection, optional): Process groups for the indexer.
        """
        super().__init__(config=config)
        self.hidden_size = self.config.hidden_size
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.q_lora_rank = (
            self.config.q_lora_rank
            if self.config.q_lora_rank is not None
            else self.config.hidden_size
        )

        self.index_n_heads = self.config.dsa_indexer_n_heads
        self.index_head_dim = self.config.dsa_indexer_head_dim
        self.index_topk = self.config.dsa_indexer_topk

        self.softmax_scale: float = self.index_head_dim**-0.5

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        self.pg_collection = pg_collection

        # Initialize Position Embedding.
        if self.config.rope_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f'Unsupported RoPE type: {self.config.rope_type}, supported types are "rope" and '
                f'"yarn"'
            )

        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        self.linear_wk = build_module(
            submodules.linear_wk,
            self.hidden_size,
            self.index_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

        k_norm_config = copy.copy(self.config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            config=k_norm_config,
            hidden_size=self.index_head_dim,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )

    def _apply_rope(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor, mscale: float):
        """Apply RoPE to the input tensor."""
        # x_pe   [seqlen, batch, *, qk_pos_emb_head_dim]
        # x_nope [seqlen, batch, *, index_head_dim - qk_pos_emb_head_dim]
        # To align with DeepSeek's implementation,
        # x_pe is placed at the front, and x_nope is placed at the back.
        x_pe, x_nope = torch.split(
            x, [self.qk_pos_emb_head_dim, self.index_head_dim - self.qk_pos_emb_head_dim], dim=-1
        )
        x_pe = apply_rotary_pos_emb(
            x_pe,
            rotary_pos_emb,
            config=self.config,
            cu_seqlens=None,
            mscale=mscale,
            cp_group=self.pg_collection.cp,
            # This flag is for the MLA-style interleaving in RoPE.
            # Set it to False, as indexer does not apply interleaved RoPE.
            mla_rotary_interleaved=False,
        )
        # [seqlen, batch, *, index_head_dim]
        x = torch.cat([x_pe, x_nope], dim=-1)
        return x

    def forward_before_topk(
        self, x: torch.Tensor, qr: torch.Tensor, packed_seq_params: Optional[PackedSeqParams] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """All computations before topk."""
        # =========================================
        # Prepare RoPE params
        # =========================================
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            None, None, x, self.config, packed_seq_params
        )
        if self.config.rope_type == "rope":
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)
            mscale = 1.0
        else:
            rotary_pos_emb, mscale = self.rotary_pos_emb(rotary_seq_len, packed_seq=False)

        # =========================================
        # Gather inputs if sp is enabled
        # =========================================
        if self.config.sequence_parallel and self.pg_collection.tp.size() > 1:
            x = gather_from_sequence_parallel_region(x, group=self.pg_collection.tp)
            qr = gather_from_sequence_parallel_region(qr, group=self.pg_collection.tp)

        # =========================================
        # Get sequence length and batch size
        # =========================================
        seqlen, bsz, _ = x.size()

        # =========================================
        # q linear and apply rope to q
        # =========================================
        # [seqlen, batch, q_lora_rank] -> [seqlen, batch, index_n_heads * index_head_dim]
        q, _ = self.linear_wq_b(qr)
        # [seqlen, batch, index_n_heads * index_head_dim]
        #   -> [seqlen, batch, index_n_heads, index_head_dim]
        q = q.reshape(seqlen, bsz, self.index_n_heads, self.index_head_dim)
        q = self._apply_rope(q, rotary_pos_emb, mscale)

        # =========================================
        # k linear and apply rope to k
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_head_dim]
        k, _ = self.linear_wk(x)
        k = self.k_norm(k)
        # [seqlen, batch, index_head_dim] -> [seqlen, batch, 1, index_head_dim]
        k = k.reshape(seqlen, bsz, 1, self.index_head_dim)
        k = self._apply_rope(k, rotary_pos_emb, mscale)
        # [seqlen, batch, 1, index_head_dim] -> [seqlen, batch, index_head_dim]
        k = k.reshape(seqlen, bsz, self.index_head_dim)

        # =========================================
        # Rotate activation
        # =========================================
        q = rotate_activation(q)
        k = rotate_activation(k)

        # =========================================
        # Prepare weights for index scores
        # =========================================
        # [seqlen, batch, hidden_size] -> [seqlen, batch, index_n_heads]
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.index_n_heads**-0.5) * self.softmax_scale

        return q, k, weights

    def forward_with_scores(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for DSA Indexer that returns both index scores and top-k indices.

        This is used when KL loss is enabled to compare indexer scores with true attention scores.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            index_scores: Index scores [batch, seqlen, seqlen].
            topk_indices: Top-k indices [batch, seqlen, index_topk].
        """
        assert packed_seq_params is None, "Packed sequence is not supported for DSAttention"

        # [seqlen, batch, index_n_heads * index_head_dim]
        # [seqlen, batch, index_head_dim]
        # [seqlen, batch, index_n_heads]
        q, k, weights = self.forward_before_topk(x, qr, packed_seq_params)

        # [batch, seqlen, seqlen], [batch, seqlen, index_topk]
        index_scores, topk_indices = fused_qk_topk_naive(q, k, weights, self.index_topk, mask)

        return index_scores, topk_indices

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """
        Forward pass for DSA Indexer.

        Args:
            x: hidden states [seqlen, batch, hidden_size].
            qr: Low-rank query tensor [seqlen, batch, q_lora_rank].
            mask: Attention mask [batch, seqlen, seqlen].
            packed_seq_params: Packed sequence parameters for variable length sequences.

        Returns:
            topk_indices: Top-k indices for sparse attention [batch, seqlen, index_topk].
        """
        _, topk_indices = self.forward_with_scores(x, qr, mask, packed_seq_params)
        return topk_indices


def unfused_dsa_fn(query, key, value, topk_indices, softmax_scale):
    """
    Unfused sparse attention implementation.
    """
    sq, b, np, hn = query.size()
    skv = key.size(0)
    hnv = value.size(3)

    # ===================================
    # Raw attention scores [b, np, sq, skv]
    # ===================================
    # [sq, b, np, hn] -> [b, np, sq, hn] -> [b * np, sq, hn]
    query = query.permute(1, 2, 0, 3).reshape(b * np, sq, hn)
    # [skv, b, np, hn] -> [b, np, hn, skv] -> [b * np, hn, skv]
    key = key.permute(1, 2, 3, 0).reshape(b * np, hn, skv)
    # Compute attention scores [b * np, sq, skv]
    attention_scores = torch.bmm(query.float(), key.float()) * softmax_scale
    # Reshape to [b, np, sq, skv]
    attention_scores = attention_scores.reshape(b, np, sq, skv)

    # ===================================
    # Apply sparse mask from indexer
    # ===================================
    # index_mask [b, sq, skv]
    index_mask = torch.full((b, sq, skv), float("-inf"), device=attention_scores.device)
    index_mask.scatter_(-1, topk_indices, 0)
    # causal_mask [sq, skv]
    causal_mask = torch.triu(
        torch.full((sq, skv), float('-inf'), dtype=torch.float32, device=index_mask.device),
        diagonal=1,
    )
    # [b, sq, skv] + [1, sq, skv] -> [b, sq, skv]
    index_mask += causal_mask.view(1, sq, skv)
    # [b, np, sq, skv] + [b, 1, sq, skv] -> [b, np, sq, skv]
    attention_scores += index_mask.unsqueeze(1)
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32)

    # ===================================
    # Output
    # ===================================
    # [skv, b, np, hnv] -> [b, np, skv, hnv] -> [b * np, skv, hnv]
    value = value.permute(1, 2, 0, 3).reshape(b * np, skv, hnv)
    # Reshape attention_scores: [b, np, sq, skv] -> [b * np, sq, skv]
    attention_scores = attention_scores.reshape(b * np, sq, skv)
    # Compute output: [b * np, sq, hnv]
    output = torch.bmm(attention_scores.to(value.dtype), value)
    # Reshape output: [b * np, sq, hnv] -> [b, np, sq, hnv] -> [sq, b, np, hnv]
    output = output.reshape(b, np, sq, hnv).permute(2, 0, 1, 3).contiguous()
    # Flatten: [sq, b, np, hnv] -> [sq, b, np * hnv]
    output = output.reshape(sq, b, np * hnv)
    return output


class DSAttention(MegatronModule):
    """
    This module implements sparse attention mechanism using an DSA Indexer to compute top-k
    attention indices for reducing computational complexity.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L491-L597
    """

    _HOLDER_ATTR = "_dsa_index_share_topk_holder"
    """Attribute carrying the DSA top-k holder when cross-layer top-k sharing is enabled.

    Packed batches store this on ``PackedSeqParams`` so each microbatch owns an isolated
    holder. Non-packed batches do not construct ``PackedSeqParams``; they store it on the
    shared transformer config object for the duration of the forward.
    """

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
    ):
        super().__init__(config=config)

        self.layer_number = layer_number

        if self.config.apply_dsa_kernel_fusion and self.config.tensor_model_parallel_size != 1:
            raise ValueError("DSA FlashMLA sparse attention currently requires TP=1.")
        if self.config.apply_dsa_kernel_fusion and self.config.context_parallel_size != 1:
            raise ValueError(
                "DSA FlashMLA sparse attention currently requires CP=1. Context parallelism "
                "needs global KV exchange and global-sequence top-k indices, which this fused "
                "path does not implement."
            )

        # DSA cross-layer top-k sharing (GLM-5.2 "IndexShare"). Computing layers own an
        # indexer; skip layers reuse top-k indices from the most recent computing layer in
        # this pipeline stage via the per-microbatch holder on ``PackedSeqParams``.
        self.index_topk = self.config.dsa_indexer_topk
        self.index_topk_freq = self.config.dsa_indexer_topk_freq
        self.index_skip_topk_offset = self.config.dsa_indexer_skip_topk_offset
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

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                required_pgs=['tp', 'cp']
            )
        self.pg_collection = pg_collection

        self.indexer = None
        if not self.skip_topk:
            self.indexer = build_module(
                submodules.indexer, config=self.config, pg_collection=self.pg_collection
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(
                k_channels if k_channels is not None else config.kv_channels
            )
        self.softmax_scale = softmax_scale

    def _run_sparse_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """Run either the fused FlashMLA sparse path or the unfused PyTorch fallback."""
        if not self.config.apply_dsa_kernel_fusion:
            assert value is not None, "Unfused DSA attention requires explicit value states."
            return unfused_dsa_fn(query, key, value, topk_indices, self.softmax_scale)

        if flash_mla_sparse_fwd is None:
            raise RuntimeError(
                "apply_dsa_kernel_fusion=True requires flash_mla_sparse_fwd, but flash_mla "
                "is not importable. Install FlashMLA from the nv_dev branch."
            )

        # Absorbed MLA passes query as [seqlen, batch, heads, qk_dim] and compressed KV as
        # [seqlen, batch, qk_dim]. FlashMLA sparse prefill expects THD-style tensors:
        # q [seqlen, heads, qk_dim], kv [seqlen, kv_heads, qk_dim], indices
        # [seqlen, kv_heads, topk]. The current trainer uses microbatch size 1.
        if query.size(1) != 1 or key.size(1) != 1 or topk_indices.size(0) != 1:
            raise RuntimeError("DSA FlashMLA sparse attention currently requires batch size 1.")
        q = query.squeeze(1).contiguous()
        kv = key.squeeze(1).contiguous()
        if kv.dim() == 2:
            kv = kv.unsqueeze(1)
        indices = topk_indices.squeeze(0).unsqueeze(1).to(torch.int32).contiguous()
        topk_length = None
        if causal:
            topk_length = torch.arange(1, q.size(0) + 1, device=q.device, dtype=torch.int32)
            topk_length = torch.clamp(topk_length, max=indices.size(-1)).contiguous()

        if torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad):
            output = FlashMLASparseAttentionFunc.apply(
                q,
                kv,
                indices,
                topk_length,
                self.softmax_scale,
                self.config.kv_lora_rank,
            )
        else:
            output = flash_mla_sparse_fwd(
                q,
                kv,
                indices,
                self.softmax_scale,
                d_v=self.config.kv_lora_rank,
                topk_length=topk_length,
            )
            if isinstance(output, (tuple, list)):
                output = output[0]
        return output.reshape(output.size(0), 1, -1).contiguous()

    def _get_index_share_topk_holder(
        self, packed_seq_params: Optional[PackedSeqParams]
    ) -> "dict[int, torch.Tensor]":
        """Return the per-microbatch top-k holder for DSA index sharing.

        The holder is a ``layer_number -> topk_indices`` map so skip layers in the same
        transformer block execution can read the top-k indices produced by the most recent
        computing layer in this pipeline stage. Cross-pipeline sharing is not supported
        (validated upstream in ``_validate_dsa_index_share_pipeline_split``).
        """
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
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        x: torch.Tensor,
        qr: torch.Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: torch.Tensor = None,
        packed_seq_params: PackedSeqParams = None,
        up_v_weight: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        """
        Forward pass for Sparse Attention.

        Args:
            query: Query tensor [sq, b, np, hn].
            key: Key tensor [skv, b, np, hn].
            value: Value tensor [skv, b, np, hnv].
            x: Original hidden states [sq, b, hidden_size].
            qr: Low-rank query representation [sq, b, q_lora_rank].
            attention_mask: Attention mask tensor [b, 1, sq, sk].
            attn_mask_type: Type of attention mask.
            attention_bias: Optional attention bias.
            packed_seq_params: Packed sequence parameters.

        Returns:
            output: Output tensor [sq, b, hidden_size]
        """
        sq, b, np, hn = query.size()
        skv = key.size(0)

        # Detach x and qr to prevent gradients of indexer from flowing back to the main model.
        x = x.detach()
        qr = qr.detach()

        # Resolve the per-microbatch top-k holder for DSA index sharing. Computing layers
        # write their fresh top-k indices into the holder; skip layers read the indices
        # previously produced by ``self.source_layer`` in this stage.
        topk_holder = (
            self._get_index_share_topk_holder(packed_seq_params) if self.index_share else None
        )
        topk_indices = None

        if self.training and torch.is_grad_enabled():
            if self.skip_topk:
                if topk_holder is None or self.source_layer not in topk_holder:
                    raise AssertionError(
                        "DSA index-share skip layer "
                        f"(layer_number={self.layer_number}) needs top-k indices from source "
                        f"computing layer {self.source_layer}, but that layer did not run before "
                        "it in this pipeline stage. Cross-PP top-k sharing is not supported. "
                        "Ensure each pipeline stage starts on a computing layer "
                        f"(dsa_indexer_topk_freq={self.index_topk_freq}, "
                        f"dsa_indexer_skip_topk_offset={self.index_skip_topk_offset}). "
                        f"Holder has layers {sorted(topk_holder or {})}."
                    )
                topk_indices = topk_holder[self.source_layer]

                # ===================================
                # Run sparse attention kernel
                # ===================================
                output = self._run_sparse_attention(
                    query, key, value, topk_indices, causal=attn_mask_type == AttnMaskType.causal
                )
            else:
                # ===================================
                # Prepare inputs for indexer loss
                # ===================================
                assert self.indexer is not None
                indexer_loss_coeff = getattr(self.config, 'dsa_indexer_loss_coeff', 0.0)

                if indexer_loss_coeff == 0:
                    if attn_mask_type is not None:
                        assert (
                            attn_mask_type == AttnMaskType.causal
                        ), 'Only causal mask is supported for now'
                    with torch.no_grad():
                        q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)
                        topk_indices = _chunked_dsa_topk(
                            q,
                            k,
                            weights,
                            self.index_topk,
                            mask=attention_mask if attn_mask_type is None else None,
                            causal=attn_mask_type == AttnMaskType.causal,
                        )

                    output = self._run_sparse_attention(
                        query,
                        key,
                        value,
                        topk_indices,
                        causal=attn_mask_type == AttnMaskType.causal,
                    )
                    indexer_loss = output.new_zeros(())
                else:
                    q, k, weights = self.indexer.forward_before_topk(x, qr, packed_seq_params)

                    # Get a FP32 mask with -inf for masked positions. This is only needed
                    # for indexer loss; frozen-indexer top-k masks each score tile instead.
                    if attn_mask_type is not None:
                        assert (
                            attn_mask_type == AttnMaskType.causal
                        ), 'Only causal mask is supported for now'
                        float_mask = torch.triu(
                            torch.full(
                                (sq, skv), float('-inf'), dtype=torch.float32, device=x.device
                            ),
                            diagonal=1,
                        )
                    else:
                        assert attention_mask.shape == (
                            b,
                            1,
                            sq,
                            skv,
                        ), 'attention_mask shape mismatch'
                        mask = attention_mask.squeeze()
                        float_mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                            mask, float('-inf')
                        )

                    # ===================================
                    # Attach indexer topk and loss
                    # ===================================
                    # Compute KL divergence loss between indexer scores and true attention scores
                    topk_indices, indexer_loss = FusedDSAIndexerLoss.apply(
                        q,
                        weights,
                        k,
                        query.detach(),
                        key.detach(),
                        self.softmax_scale,
                        self.index_topk,
                        indexer_loss_coeff,
                        float_mask,
                        getattr(self.config, "dsa_indexer_use_sparse_loss", False),
                        self.pg_collection,
                        self.config.calculate_per_token_loss,
                    )

                    # ===================================
                    # Run sparse attention kernel
                    # ===================================
                    output = self._run_sparse_attention(
                        query,
                        key,
                        value,
                        topk_indices,
                        causal=attn_mask_type == AttnMaskType.causal,
                    )

                    # Attach loss to output
                    output = DSAIndexerLossAutoScaler.apply(output, indexer_loss)

                # Save indexer loss for logging
                if indexer_loss_coeff > 0:
                    DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                        loss=indexer_loss,
                        layer_number=self.layer_number,
                        num_layers=self.config.num_layers + (self.config.mtp_num_layers or 0),
                    )
        else:
            if self.skip_topk:
                if topk_holder is None or self.source_layer not in topk_holder:
                    raise AssertionError(
                        "DSA index-share skip layer "
                        f"(layer_number={self.layer_number}) needs top-k indices from source "
                        f"computing layer {self.source_layer}, but that layer did not run before "
                        "it in this pipeline stage. Cross-PP top-k sharing is not supported. "
                        "Ensure each pipeline stage starts on a computing layer "
                        f"(dsa_indexer_topk_freq={self.index_topk_freq}, "
                        f"dsa_indexer_skip_topk_offset={self.index_skip_topk_offset}). "
                        f"Holder has layers {sorted(topk_holder or {})}."
                    )
                topk_indices = topk_holder[self.source_layer]

                # ===================================
                # Run sparse attention kernel
                # ===================================
                output = self._run_sparse_attention(
                    query, key, value, topk_indices, causal=attn_mask_type == AttnMaskType.causal
                )
            else:
                # ===================================
                # Get index scores and top-k indices
                # ===================================
                assert self.indexer is not None
                if attn_mask_type is not None:
                    assert (
                        attn_mask_type == AttnMaskType.causal
                    ), 'Only causal mask is supported for now'
                topk_indices = _chunked_dsa_topk(
                    *self.indexer.forward_before_topk(x, qr, packed_seq_params),
                    self.index_topk,
                    mask=attention_mask if attn_mask_type is None else None,
                    causal=attn_mask_type == AttnMaskType.causal,
                )

                # ===================================
                # Run sparse attention kernel
                # ===================================
                output = self._run_sparse_attention(
                    query, key, value, topk_indices, causal=attn_mask_type == AttnMaskType.causal
                )

        if self.index_share and not self.skip_topk:
            assert topk_holder is not None and topk_indices is not None
            topk_holder[self.layer_number] = topk_indices

        return output
