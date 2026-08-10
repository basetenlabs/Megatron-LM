# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lookahead recompute for full activation checkpointing (BT_MOE_LOOKAHEAD_RECOMPUTE).

Under full recompute, every checkpointed chunk (one transformer layer at
recompute_num_layers=1) runs twice: a no-grad first pass and, inside the
checkpoint Function's backward, a grad-enabled recompute replay immediately
followed by the chunk's backward. The replay of chunk L-1 depends ONLY on its
own saved inputs (produced in the forward phase) — not on chunk L's backward
— so it can be kicked EARLY: at the boundary between chunk L's recompute and
chunk L's backward, the previous chunk's recompute is issued on a side
stream, overlapping its compute (and its MoE all-to-alls) with chunk L's
backward. Design: experiment_artefacts/glm/lps_1062_perf/overlap_design/
DESIGN_helmholtz.md section 7 (W3).

Gate: BT_MOE_LOOKAHEAD_RECOMPUTE=1 (default OFF). With the gate off, or with
no per-microbatch carrier (packed_seq_params is None), or during CUDA graph
capture/warmup, the code path is the status-quo checkpoint behavior (the
lookahead Function degrades to an inline recompute, and chunk_runner keeps
using te_checkpoint / tensor_parallel.checkpoint unless the gate is on).

Correctness invariants:
* Bitwise-identical to the status-quo order. The kick is pure scheduling:
  the same run_function runs on the same (detached) saved inputs with the
  same restored RNG states (fork/set/restore is host-atomic on the autograd
  thread; production configs run dropout=0.0 — asserted at gate-on — so RNG
  is never even consumed). Dropout>0 is covered by the CPU test as a
  mechanism proof, not a supported production config.
* The kicked graph's leaf tensors are the kick's own detached inputs; the
  chunk's backward collects their .grad (value-identical to the inline
  path's detached inputs — both detach the same saved tensors).
* Stream/allocator safety: each kick starts with side_stream.wait_stream(
  current), ordering the side stream after the compute stream's backlog so
  allocator blocks freed during earlier backwards can never be clobbered
  while still read; the consuming backward waits the kick's done-event and
  record_stream()s the stashed outputs before use.
* Eviction discipline: a chunk's registry entry is popped on consume (its
  backward), the kick's stash is popped by the kicked chunk's backward, and
  the registry is swept when the last remaining chunk's backward completes
  (a non-empty sweep is logged loud — it indicates a structural bug). The
  registry lives on the per-microbatch carrier, so anything left dies with
  the microbatch regardless.

Telemetry (WARNING level; logger.info from megatron.core does not reach
trainer_srun.log): one-time gate-state line, one-time armed line (first
registered chunk), and per-window counters (kicks / stash_hits / stash_misses
/ sweeps / fallbacks) every 300 chunk backwards (~1 step at 78 layers x 4
datums) for the first 100 windows. A present-but-inert patch is never
silent: gate ON with no kicks shows up as misses.
"""

import contextlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.checkpoint import detach_variable

from megatron.core.tensor_parallel import random as tp_random
from megatron.core.tensor_parallel.utils import (
    gather_split_1d_tensor,
    split_tensor_into_1d_equal_chunks,
)
from megatron.core.utils import safely_set_viewless_tensor_data

logger = logging.getLogger(__name__)

# TE fp8 recompute contexts are optional (mirror tensor_parallel/random.py).
try:
    from transformer_engine.pytorch.distributed import activation_recompute_forward
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast

    HAVE_TE = True
except ImportError:
    activation_recompute_forward = None
    FP8GlobalStateManager = None
    fp8_autocast = None
    HAVE_TE = False


# ----------------------------------------------------------------------------
# Gate + telemetry (house style: mirrors BT_MOE_DISPATCH_REPLAY_CACHE)
# ----------------------------------------------------------------------------

_LOOKAHEAD_GATE_LOGGED = [False]
_LOOKAHEAD_ARMED_LOGGED = [False]
_LOOKAHEAD_STATS = {
    "kicks": 0,
    "stash_hits": 0,
    "stash_misses": 0,
    "sweeps": 0,
    "fallbacks": 0,
}
_LOOKAHEAD_WINDOW = [0, 0]  # [chunk backwards this window, windows logged]
_LOOKAHEAD_WINDOW_SIZE = 300
_LOOKAHEAD_WINDOW_LOG_MAX = 100


def _lookahead_enabled() -> bool:
    """Lazy env-gate read (per-call; negligible at ~2 calls/chunk)."""
    enabled = os.environ.get("BT_MOE_LOOKAHEAD_RECOMPUTE", "0") == "1"
    if not _LOOKAHEAD_GATE_LOGGED[0]:
        _LOOKAHEAD_GATE_LOGGED[0] = True
        if enabled:
            logger.warning(
                "BT_MOE_LOOKAHEAD_RECOMPUTE=1: lookahead recompute (cross-layer "
                "recompute/backward overlap) ACTIVE"
            )
        else:
            logger.warning(
                "BT_MOE_LOOKAHEAD_RECOMPUTE present but DISABLED (env unset or != '1'); "
                "status-quo checkpoint recompute order"
            )
    return enabled


def _lookahead_note(kind: str):
    """Per-window lookahead telemetry (WARNING so it reaches trainer logs)."""
    _LOOKAHEAD_STATS[kind] += 1
    _LOOKAHEAD_WINDOW[0] += 1
    if _LOOKAHEAD_WINDOW[0] >= _LOOKAHEAD_WINDOW_SIZE:
        _LOOKAHEAD_WINDOW[0] = 0
        _LOOKAHEAD_WINDOW[1] += 1
        if _LOOKAHEAD_WINDOW[1] <= _LOOKAHEAD_WINDOW_LOG_MAX:
            logger.warning(
                "BT_MOE_LOOKAHEAD_RECOMPUTE window %d (%d chunk backwards): %s",
                _LOOKAHEAD_WINDOW[1],
                _LOOKAHEAD_WINDOW_SIZE,
                dict(_LOOKAHEAD_STATS),
            )


_LOOKAHEAD_CONFIG_CHECKED = [False]


def lookahead_recompute_enabled() -> bool:
    """Public gate read for callers (recompute.py's chunk_runner)."""
    return _lookahead_enabled()


def lookahead_check_config_once(config) -> None:
    """One-time gate-on config validation — loud failure, not silent fallback.

    Production runs dropout=0; the kick's RNG fork/restore is host-atomic on
    the autograd thread, but dropout>0 makes RNG consumption order visible to
    the overlap, so it is rejected at gate-on (the CPU test suite proves the
    fork/restore mechanism itself is dropout-safe; relax only with box
    evidence).
    """
    if _LOOKAHEAD_CONFIG_CHECKED[0]:
        return
    _LOOKAHEAD_CONFIG_CHECKED[0] = True
    if not _lookahead_enabled():
        return
    hidden_dropout = getattr(config, "hidden_dropout", 0.0) or 0.0
    attention_dropout = getattr(config, "attention_dropout", 0.0) or 0.0
    if hidden_dropout != 0.0 or attention_dropout != 0.0:
        raise RuntimeError(
            "BT_MOE_LOOKAHEAD_RECOMPUTE requires dropout=0.0 (got hidden_dropout="
            f"{hidden_dropout}, attention_dropout={attention_dropout})"
        )


# ----------------------------------------------------------------------------
# Registry (per-microbatch, on the packed_seq_params carrier)
# ----------------------------------------------------------------------------

_REGISTRY_ATTR = "_moe_lookahead_recompute_registry"
_side_stream = None


def _get_side_stream():
    """One lazily-created module-level side stream (mirrors the dispatcher's
    cuda_dtoh_stream). None on CPU-only builds (the kick then runs inline on
    the current stream — semantics preserved, overlap absent)."""
    global _side_stream
    if _side_stream is None and torch.cuda.is_available():
        _side_stream = torch.cuda.Stream()
    return _side_stream


class _LookaheadEntry:
    """One checkpointed chunk's registry entry.

    Written at first pass (run_function / rng_states / fp8 snapshot / ctx);
    the kick fills (kicked_inputs, kicked_outputs, done_event); the chunk's
    backward pops the whole entry on consume.
    """

    __slots__ = (
        "run_function",
        "rng_states",
        "fp8",
        "fp8_recipe",
        "ctx",
        "kicked_inputs",
        "kicked_outputs",
        "done_event",
    )

    def __init__(self):
        self.run_function = None
        self.rng_states = None
        self.fp8 = False
        self.fp8_recipe = None
        self.ctx = None
        self.kicked_inputs = None
        self.kicked_outputs = None
        self.done_event = None


def _get_registry(carrier) -> Dict[str, Any]:
    """Return the carrier's lookahead registry, creating it on first use.

    The registry lives and dies with the per-microbatch carrier (the same
    lifetime pattern as the DSA top-k holder and the FIX-B/FIX-C caches), so
    no entry can outlive its microbatch.
    """
    registry = getattr(carrier, _REGISTRY_ATTR, None)
    if registry is None:
        registry = {"order": [], "chunks": {}}
        setattr(carrier, _REGISTRY_ATTR, registry)
    return registry


def _fp8_snapshot():
    """Mirror CheckpointWithoutOutputFunction's fp8 state capture."""
    if FP8GlobalStateManager is not None and FP8GlobalStateManager.is_fp8_enabled():
        return True, FP8GlobalStateManager.get_fp8_recipe()
    return False, None


def _fp8_first_pass_ctx(fp8):
    if fp8:
        return activation_recompute_forward(activation_recompute=True, recompute_phase=False)
    return contextlib.nullcontext()


def _fp8_recompute_ctxs(fp8, fp8_recipe):
    if fp8:
        return (
            fp8_autocast(enabled=True, fp8_recipe=fp8_recipe),
            activation_recompute_forward(activation_recompute=True, recompute_phase=True),
        )
    return contextlib.nullcontext(), contextlib.nullcontext()


def _kick_previous_chunk(registry, chunk_key):
    """Issue the previous chunk's recompute on the side stream (no-op if
    absent, already kicked, or already consumed)."""
    order: List[Any] = registry["order"]
    chunks: Dict[Any, _LookaheadEntry] = registry["chunks"]
    try:
        idx = order.index(chunk_key)
    except ValueError:
        return
    if idx == 0:
        return  # first chunk: nothing to kick
    prev = chunks.get(order[idx - 1])
    if prev is None or prev.kicked_outputs is not None or prev.ctx is None:
        return
    prev_ctx = prev.ctx
    inputs = prev_ctx.saved_tensors
    if prev_ctx.distribute_saved_activations:
        safely_set_viewless_tensor_data(
            inputs[0],
            gather_split_1d_tensor(inputs[0].data).view(prev_ctx.input_0_shape),
        )
    stream = _get_side_stream()
    # Allocator safety: order the side stream after the compute stream's
    # backlog, so blocks freed (host-side) during earlier backwards — which
    # the side stream's allocations may reuse — are never written while still
    # read by the compute stream's queued work. At kick time the next chunk's
    # backward has not been pushed yet, so this costs ~nothing and preserves
    # the overlap.
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        with tp_random._fork_rng():
            tp_random._set_all_rng_states(*prev.rng_states)
            detached_inputs = detach_variable(inputs)
            fp8_ctx, recompute_ctx = _fp8_recompute_ctxs(prev.fp8, prev.fp8_recipe)
            with torch.enable_grad(), fp8_ctx, recompute_ctx:
                outputs = prev.run_function(*detached_inputs)
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    prev.kicked_inputs = detached_inputs
    prev.kicked_outputs = outputs
    prev.done_event = stream.record_event() if stream is not None else None
    _lookahead_note("kicks")


class LookaheadCheckpointFunction(torch.autograd.Function):
    """Checkpoint Function with lookahead recompute (BT_MOE_LOOKAHEAD_RECOMPUTE).

    Mirrors tensor_parallel.random.CheckpointFunction (plus the fp8 snapshot
    of CheckpointWithoutOutputFunction); the backward is split into
    [pop-or-recompute] -> [kick previous chunk] -> [autograd.backward].
    """

    @staticmethod
    def forward(
        ctx: Any,
        run_function: Callable,
        distribute_saved_activations: bool,
        chunk_key: Any,
        carrier: Any,
        *args: Any,
    ) -> Any:
        """First pass: status-quo no-grad forward + registry entry."""
        tp_random._set_checkpointing()

        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations
        ctx.chunk_key = chunk_key
        ctx.carrier = carrier

        ctx.rng_states = tp_random._get_all_rng_states()
        ctx.fp8, ctx.fp8_recipe = _fp8_snapshot()

        with torch.no_grad(), _fp8_first_pass_ctx(ctx.fp8):
            outputs = run_function(*args)

        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            )

        ctx.save_for_backward(*args)
        tp_random._unset_checkpointing()

        if carrier is not None:
            registry = _get_registry(carrier)
            entry = _LookaheadEntry()
            entry.run_function = run_function
            entry.rng_states = ctx.rng_states
            entry.fp8 = ctx.fp8
            entry.fp8_recipe = ctx.fp8_recipe
            entry.ctx = ctx  # strong ref; the entry is popped on consume and
            # the registry dies with the microbatch carrier regardless.
            registry["order"].append(chunk_key)
            registry["chunks"][chunk_key] = entry
            if not _LOOKAHEAD_ARMED_LOGGED[0]:
                _LOOKAHEAD_ARMED_LOGGED[0] = True
                logger.warning(
                    "BT_MOE_LOOKAHEAD_RECOMPUTE: armed — first checkpointed chunk "
                    "registered (key=%s)",
                    chunk_key,
                )
        return outputs

    @staticmethod
    def backward(ctx, *args):
        """Backward: pop the kicked stash (or recompute inline), kick the
        previous chunk, then backward through this chunk's graph."""
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        tp_random._set_checkpointing()

        inputs = ctx.saved_tensors
        carrier = ctx.carrier
        registry = _get_registry(carrier) if carrier is not None else None
        entry = None
        if registry is not None:
            entry = registry["chunks"].pop(ctx.chunk_key, None)
            # NOTE: chunk_key stays in registry["order"] until after the kick
            # below — the kick targets the previous key relative to it.

        if entry is not None and entry.kicked_outputs is not None:
            # Kicked path: the previous backward already recomputed this
            # chunk's graph on the side stream — wait for it and reuse.
            if entry.done_event is not None:
                torch.cuda.current_stream().wait_event(entry.done_event)
                outputs = entry.kicked_outputs
                for t in outputs:
                    if isinstance(t, torch.Tensor):
                        t.record_stream(torch.cuda.current_stream())
            else:
                outputs = entry.kicked_outputs
            detached_inputs = entry.kicked_inputs
            _lookahead_note("stash_hits")
        else:
            # Inline recompute (status quo): last chunk (never kick-targeted),
            # or no registry entry (carrier appeared mid-run, gate flips).
            _lookahead_note("stash_misses" if entry is not None else "fallbacks")
            if ctx.distribute_saved_activations:
                safely_set_viewless_tensor_data(
                    inputs[0],
                    gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape),
                )
            with tp_random._fork_rng():
                tp_random._set_all_rng_states(*ctx.rng_states)
                detached_inputs = detach_variable(inputs)
                fp8_ctx, recompute_ctx = _fp8_recompute_ctxs(ctx.fp8, ctx.fp8_recipe)
                with torch.enable_grad(), fp8_ctx, recompute_ctx:
                    outputs = ctx.run_function(*detached_inputs)

        # Kick the previous chunk's recompute before starting this chunk's
        # backward, so the two overlap. Only then retire this chunk's key
        # from the order list (the kick targets the previous key relative to
        # the current one).
        if registry is not None:
            _kick_previous_chunk(registry, ctx.chunk_key)
            if ctx.chunk_key in registry["order"]:
                registry["order"].remove(ctx.chunk_key)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # filter out non tensor outputs for backward pass
        outputs, args = zip(
            *filter(lambda x: torch.is_tensor(x[0]) and x[0].requires_grad, zip(outputs, args))
        )
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)

        # Eviction sweep: this was the last registered chunk's backward — the
        # registry must be empty now; anything left is a structural bug (and
        # would otherwise pin a stash until the carrier dies).
        if registry is not None and not registry["order"]:
            leftover = len(registry["chunks"])
            registry["chunks"].clear()
            if leftover:
                _lookahead_note("sweeps")
                logger.warning(
                    "BT_MOE_LOOKAHEAD_RECOMPUTE: sweep dropped %d unconsumed registry "
                    "entries at microbatch end — indicates a structural bug; report",
                    leftover,
                )

        tp_random._unset_checkpointing()
        return (None, None, None, None) + grads


def lookahead_checkpoint(
    function: Callable,
    distribute_saved_activations: bool,
    chunk_key: Any,
    carrier: Any,
    *args: Any,
) -> Any:
    """Checkpoint a model or part of the model with lookahead recompute.

    Mirrors tensor_parallel.random.checkpoint's CUDA-graph passthrough: during
    graph warmup/capture the function runs directly (recomputation cannot run
    inside a captured graph).
    """
    from megatron.core.transformer.cuda_graphs import is_graph_capturing, is_graph_warmup

    if is_graph_warmup() or is_graph_capturing():
        return function(*args)
    return LookaheadCheckpointFunction.apply(
        function, distribute_saved_activations, chunk_key, carrier, *args
    )
