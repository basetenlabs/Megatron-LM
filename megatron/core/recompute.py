# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import os
import threading
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.lookahead_checkpoint import (
    lookahead_check_config_once as _lookahead_check_config_once,
    lookahead_checkpoint,
    lookahead_recompute_enabled as _lookahead_checkpoint_enabled,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer

te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint


# ----------------------------------------------------------------------------
# BT_MOE_DISPATCH_REPLAY_CACHE support — checkpoint-pass frames (env-gated,
# default OFF; the gate is read by the consumer, MoEAlltoAllTokenDispatcher).
#
# Full recompute runs every checkpointed chunk twice: a no-grad first pass
# (inside the checkpoint Function's forward — always no-grad by the torch
# invariant that autograd.Function.forward runs under no-grad) and, later, a
# grad-enabled recompute replay (inside its backward, under enable_grad). The
# replay recomputes the MoE dispatcher's routing-derived split metadata
# bit-identically, so the dispatcher can reuse the first pass's host splits
# instead of re-running the tp_ep all-gather + D2H + event sync — IF it can
# tell the two passes apart and key metadata to the right microbatch.
#
# The machinery below marks each pass. `_CheckpointChunkPassMarker` wraps the
# chunk's run_function before it is handed to the checkpoint implementation
# (mcore's tensor_parallel.checkpoint or TE's te_checkpoint — wrapping here
# covers both, and any future checkpoint backend that re-runs run_function
# under enable_grad in backward). The marker is a plain callable, NOT an
# autograd.Function, so torch.is_grad_enabled() inside it truthfully
# discriminates the passes (False = first pass, True = replay). Each call
# pushes a CheckpointPassFrame onto a thread-local stack for the duration of
# the pass; the first pass runs on the main thread, the replay on the
# autograd worker thread, and the two never share a stack.
#
# The frame's key_obj is the microbatch's packed_seq_params carrier when
# present (the same per-microbatch lifetime pattern as the DSA top-k holder
# and the BT_DSA_CP_LAYOUT_CACHE — the object is closure-captured by the
# chunk's custom_forward and is therefore identical in both passes), falling
# back to the marker instance itself (held alive by the checkpoint context
# from first pass through backward). Consumers key per-layer state on
# (key_obj, id(self)); entries stored on the key_obj die with the microbatch,
# so no stale read can cross a step boundary.
# ----------------------------------------------------------------------------


class CheckpointPassFrame:
    """One checkpointed chunk pass: first pass (is_replay=False) or replay (True).

    scratch is a per-pass, per-thread scratchpad keyed by id(consumer) so a
    consumer can hand state between its own call sites within the pass without
    writing to shared instance attributes (which the main and autograd threads
    may touch concurrently for different microbatches).
    """

    __slots__ = ("is_replay", "key_obj", "scratch")

    def __init__(self, is_replay: bool, key_obj):
        self.is_replay = is_replay
        self.key_obj = key_obj
        self.scratch = {}


_replay_pass_local = threading.local()


def _replay_pass_stack() -> list:
    stack = getattr(_replay_pass_local, "stack", None)
    if stack is None:
        stack = _replay_pass_local.stack = []
    return stack


def current_checkpoint_pass_frame() -> Optional[CheckpointPassFrame]:
    """Return the innermost checkpoint-pass frame on this thread, or None.

    None means the caller is not inside a wrapped checkpointed chunk (eval,
    inference, non-recompute training, or the gate is off) and must follow
    the status-quo path.
    """
    stack = _replay_pass_stack()
    return stack[-1] if stack else None


class _CheckpointChunkPassMarker:
    """Wraps a checkpointed chunk's run_function; executes in both passes.

    Pushes a CheckpointPassFrame for the duration of each call. is_replay is
    torch.is_grad_enabled() at call time: False in the no-grad first pass,
    True in the enable_grad recompute replay.
    """

    def __init__(self, fn, packed_seq_params):
        self.fn = fn
        self.packed_seq_params = packed_seq_params

    def __call__(self, *args, **kwargs):
        key_obj = self.packed_seq_params if self.packed_seq_params is not None else self
        _replay_pass_stack().append(CheckpointPassFrame(torch.is_grad_enabled(), key_obj))
        try:
            return self.fn(*args, **kwargs)
        finally:
            _replay_pass_stack().pop()


def _wrap_checkpoint_chunk_pass(fn, packed_seq_params):
    """Wrap a checkpointed chunk's run_function with the pass marker.

    No-op unless BT_MOE_DISPATCH_REPLAY_CACHE=1 (default OFF): with the gate
    off the checkpoint receives the unwrapped function, no frames exist, and
    the code path is byte-identical to upstream.
    """
    if os.environ.get("BT_MOE_DISPATCH_REPLAY_CACHE", "0") != "1":
        return fn
    return _CheckpointChunkPassMarker(fn, packed_seq_params)


def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    rotary_pos_emb: Tensor,
    attention_bias: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    padding_mask: Optional[Tensor] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Forward method with activation checkpointing.

    Args:
        extract_layer_indices (Set[int], optional): Global layer
            indices (across all pipeline stages) from which to
            extract features.
        layer_offset (int): The global layer offset for the current
            pipeline stage. Used to convert local layer indices to
            global indices when checking extract_layer_indices.

    Returns:
        If extract_layer_indices is empty: hidden_states tensor
        If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask=None
        ):
            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]

                # Get appropriate inner quantization context
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
                        )
                    # TODO: check if fp4 is supported in this case
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                # Build the full TransformerLayer kwarg set; for non-TL
                # layers (currently MambaLayer in HybridStack) pop the kwargs
                # they don't accept and treat the return as a single tensor.
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    else:  # MambaLayer (HybridStack `M` slot)
                        for k in ("context", "context_mask", "attention_bias", "padding_mask"):
                            layer_kwargs.pop(k, None)
                        hidden_states = layer(**layer_kwargs)
                        context = None

                # Some layer paths may still return a tuple (defensive).
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            return hidden_states, context

        return custom_forward

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        cf = custom(start, end)
        args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask)
        if use_checkpoint:
            # Mark the first pass / recompute replay for the gated MoE
            # dispatcher replay-metadata cache (no-op when the gate is off).
            cf = _wrap_checkpoint_chunk_pass(cf, packed_seq_params)
            if _lookahead_checkpoint_enabled():
                # BT_MOE_LOOKAHEAD_RECOMPUTE: cross-layer lookahead recompute.
                # The chunk key is the chunk's global layer index (uniform) or
                # first layer (block); the registry lives on the
                # packed_seq_params carrier. Dropout must be off: the kick's
                # RNG fork/restore is host-atomic on the autograd thread, but
                # production runs dropout=0 and we keep it that way by assert.
                _lookahead_check_config_once(self.config)
                hidden_states, context = lookahead_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    chunk_key=start + layer_offset,
                    carrier=packed_seq_params,
                    *args,
                )
            # Precision-aware activation checkpoint: TE under FP8/FP4,
            # tensor_parallel under BF16/FP16/FP32.
            elif self.config.fp8 or self.config.fp4:
                hidden_states, context = te_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    *args,
                )
            else:
                hidden_states, context = tensor_parallel.checkpoint(
                    cf, self.config.distribute_saved_activations, *args
                )
        else:
            # Note: original block-branch no-checkpoint path omitted padding_mask
            # (relied on its default=None); restored here for consistency.
            hidden_states, context = cf(*args)

        if self.config.recompute_method == "uniform":
            if (end - 1 + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)
        else:
            if (start + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of layers and checkpoint
        # the input activation of each divided chunk.
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            chunk_end = min(
                layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank
            )
            chunk_runner(layer_idx, chunk_end, True)
            layer_idx += self.config.recompute_num_layers
    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # layers and skip the rest. Need at least one input tensor with
        # gradient computation for the re-entrant autograd engine, so under
        # FP8/FP4 we skip checkpointing while hidden_states.requires_grad
        # is False (these slots get pushed past the recompute window).
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            use_checkpoint = (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            )
            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
    else:
        raise ValueError("Invalid activation recompute method.")

    # Return intermediate hidden states if feature extraction was requested
    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states
