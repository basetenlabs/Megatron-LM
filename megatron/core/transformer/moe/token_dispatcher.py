# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch

from megatron.core import utils
from megatron.core.config import is_experimental_enabled
from megatron.core.fusions.fused_indices_converter import fused_indices_to_multihot
from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map
from megatron.core.jit import jit_fuser
from megatron.core.tensor_parallel import (
    all_to_all,
    all_to_all_deferred,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    wait_deferred_a2a,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.fused_a2a import (
    ensure_nccl_ep_bootstrapped,
    fused_combine,
    fused_dispatch,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    nccl_ep_combine,
    nccl_ep_dispatch,
    new_nccl_ep_buffer,
    set_deepep_num_sms,
)
from megatron.core.transformer.moe.moe_utils import (
    ProcessGroupCollection,
    get_align_size_for_quantization,
    get_capacity,
    maybe_move_tensor_to_cpu,
    pad_routing_map,
    permute,
    sort_chunks_by_idxs,
    unpermute,
)
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# BT_MOE_DISPATCH_REPLAY_CACHE (env-gated, default OFF)
#
# Under full recompute, every MoE layer-microbatch runs this dispatcher twice:
# the no-grad first pass and the grad-enabled recompute replay. The replay
# recomputes routing_map bit-identically (recompute is deterministic by
# design: same inputs, same restored RNG state), so the split metadata the
# dispatcher derives from it — input_splits / output_splits / output_splits_tp
# / num_out_tokens / tokens_per_expert / num_global_tokens_per_local_expert —
# is also bit-identical. The first pass produces those values anyway (16-rank
# all-gather over tp_ep in preprocess, then a side-stream D2H + deferred
# d2h_event.synchronize()); the replay re-derives them at full cost
# (measured 300 replay event-syncs = 23.7s CPU per 4x131k step, plus 300
# replay all-gathers; see experiment_artefacts/glm/lps_1062_perf/
# dispatcher_opt/). With this gate on, the first pass stores the metadata and
# the replay reuses it, skipping the replay all-gather, D2H copies, and event
# sync entirely. Skipping the all-gather is collective-consistent: every rank
# replays the same layers in the same order, so no rank waits on a skipped
# collective.
#
# Pass classification and microbatch keying come from the checkpoint-pass
# frames in megatron/core/recompute.py (the marker wraps the checkpointed
# chunk's run_function, so it works for both mcore's CheckpointFunction and
# TE's te_checkpoint). The frame's key_obj is the per-microbatch
# packed_seq_params carrier (identical object in both passes via closure
# capture), falling back to the marker instance. Entries are stored on the
# key_obj keyed by id(dispatcher) — one entry per layer per microbatch — and
# popped on first replay use, so they can never outlive their microbatch and
# no stale entry can cross a step (or grad-accumulation) boundary. Any number
# of microbatches in flight is supported; the first pass (main thread) and
# replays (autograd thread) never share a frame stack or a carrier.
#
# SAFETY:
# * BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1 (requires the main gate): on every
#   replay, recompute the metadata at full cost AND assert bitwise equality
#   with the cached entry (raises on mismatch). Validation soaks only — this
#   mode keeps the replay's all-gather + D2H + event sync and adds compares.
# * Any key miss or routing_map shape mismatch on a replay falls back to the
#   status-quo recompute path with a WARNING + counter.
# * Gate off, or no checkpoint frame (eval / inference / non-recompute
#   training): the code path is byte-identical to upstream.
#
# Telemetry (WARNING level; logger.info from megatron.core does not reach
# trainer_srun.log): one-time gate-state line, one-time armed / first-hit
# lines, an immediate WARNING on every fallback, and per-window counters
# (stores / hits / misses / shape_mismatches / verifies) every 300 replay
# lookups (~1 step at 75 MoE layers x 4 microbatches) for the first 100
# windows. A present-but-inert patch is never silent: gate ON with no replay
# hits shows up as misses.
# ----------------------------------------------------------------------------

_REPLAY_CACHE_ATTR = "_moe_dispatch_replay_cache"
_REPLAY_CACHE_GATE_LOGGED = [False]
_REPLAY_VERIFY_GATE_LOGGED = [False]
_REPLAY_ARMED_LOGGED = [False]
_REPLAY_HIT_LOGGED = [False]
_REPLAY_STATS = {"stores": 0, "hits": 0, "misses": 0, "shape_mismatches": 0, "verifies": 0}
_REPLAY_WINDOW = [0, 0]  # [replay lookups this window, windows logged]
_REPLAY_WINDOW_SIZE = 300
_REPLAY_WINDOW_LOG_MAX = 100


def _replay_cache_enabled() -> bool:
    """Lazy env-gate read (per-call; negligible at ~5 calls/layer-pass).

    Logs the gate state once per process at WARNING level (logger.info from
    megatron.core does not reach trainer_srun.log) so a present-but-inert
    patch is never silent.
    """
    enabled = os.environ.get("BT_MOE_DISPATCH_REPLAY_CACHE", "0") == "1"
    if not _REPLAY_CACHE_GATE_LOGGED[0]:
        _REPLAY_CACHE_GATE_LOGGED[0] = True
        if enabled:
            logger.warning(
                "BT_MOE_DISPATCH_REPLAY_CACHE=1: MoE dispatcher replay-metadata cache ACTIVE"
            )
        else:
            logger.warning(
                "BT_MOE_DISPATCH_REPLAY_CACHE present but DISABLED (env unset or != '1'); "
                "recomputing dispatcher split metadata in the recompute replay"
            )
    return enabled


def _replay_verify_enabled() -> bool:
    """Lazy read of the verify-mode gate (requires the main gate to matter)."""
    verify = os.environ.get("BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY", "0") == "1"
    if not _REPLAY_VERIFY_GATE_LOGGED[0]:
        _REPLAY_VERIFY_GATE_LOGGED[0] = True
        if verify and _replay_cache_enabled():
            logger.warning(
                "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1: replay-metadata verify mode ACTIVE "
                "(every replay also recomputes the metadata and asserts bitwise equality; "
                "validation soaks only — no performance benefit)"
            )
        elif verify:
            logger.warning(
                "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1 ignored: "
                "BT_MOE_DISPATCH_REPLAY_CACHE is not enabled"
            )
    return verify and _replay_cache_enabled()


def _current_pass_frame():
    """Innermost checkpoint-pass frame on this thread, or None.

    Lazy import: megatron.core.recompute imports the transformer layer stack,
    which leads back to this module — a module-level import would cycle.
    """
    from megatron.core.recompute import current_checkpoint_pass_frame

    return current_checkpoint_pass_frame()


class _ReplayEntry:
    """One layer-microbatch of dispatcher split metadata (first-pass values).

    The *_dev fields are the device tensors as produced by preprocess (needed
    to reproduce the exact instance-state transitions of the status-quo
    replay); the *_host fields are the post-D2H host values installed at the
    DtoH point. num_out_tokens_fwd / num_out_tokens_host are python ints in
    the dropless no-capacity path (the same object serves both slots).
    """

    __slots__ = (
        "routing_shape",
        "input_splits_dev",
        "output_splits_dev",
        "output_splits_tp_dev",
        "num_out_tokens_fwd",
        "num_tokens_per_local_expert_dev",
        "num_global_tokens_per_local_expert_dev",
        "input_splits_host",
        "output_splits_host",
        "output_splits_tp_host",
        "num_out_tokens_host",
        "tokens_per_expert_host",
        "num_global_tokens_per_local_expert_host",
        # BT_MOE_A2A_PIPELINE (W2): host count matrices the chunk plan derives
        # from; populated only when the dispatcher's _w2_config is armed.
        "w2_local_counts_host",
        "w2_global_counts_host",
        # Verify mode only: the first pass's (padded) routing_map, for the
        # mismatch forensics' XOR detail (None otherwise — 2 MB per entry).
        "routing_map_dev",
    )

    def __init__(self):
        for field in self.__slots__:
            setattr(self, field, None)


class _ReplayPassRecord:
    """Per-(dispatcher, pass) state, kept on the frame's thread-local scratch.

    mode: "store" (first pass — capture at the sync point), "hit" (replay
    reuse), "fallback" (replay without a usable entry — status-quo path).
    """

    __slots__ = ("mode", "key_obj", "entry", "verify", "device_vals")

    def __init__(self, mode, key_obj, entry=None, verify=False):
        self.mode = mode
        self.key_obj = key_obj
        self.entry = entry
        self.verify = verify
        self.device_vals = None


def _replay_note_lookup():
    """Per-window replay-lookup telemetry (WARNING so it reaches trainer logs)."""
    _REPLAY_WINDOW[0] += 1
    if _REPLAY_WINDOW[0] >= _REPLAY_WINDOW_SIZE:
        _REPLAY_WINDOW[0] = 0
        _REPLAY_WINDOW[1] += 1
        if _REPLAY_WINDOW[1] <= _REPLAY_WINDOW_LOG_MAX:
            logger.warning(
                "BT_MOE_DISPATCH_REPLAY_CACHE window %d (%d replay lookups): %s",
                _REPLAY_WINDOW[1],
                _REPLAY_WINDOW_SIZE,
                dict(_REPLAY_STATS),
            )


def _capture_device_metadata(dispatcher, num_tokens_per_local_expert):
    """Snapshot the device-side split metadata at the end of preprocess."""
    return {
        "input_splits_dev": dispatcher.input_splits,
        "output_splits_dev": dispatcher.output_splits,
        "output_splits_tp_dev": dispatcher.output_splits_tp,
        "num_out_tokens_fwd": dispatcher.num_out_tokens,
        "num_tokens_per_local_expert_dev": num_tokens_per_local_expert,
        "num_global_tokens_per_local_expert_dev": getattr(
            dispatcher, "num_global_tokens_per_local_expert", None
        ),
    }


def _restore_device_metadata(dispatcher, entry):
    """Install the cached first-pass device metadata (pre-DtoH state)."""
    dispatcher.input_splits = entry.input_splits_dev
    dispatcher.output_splits = entry.output_splits_dev
    dispatcher.output_splits_tp = entry.output_splits_tp_dev
    dispatcher.num_out_tokens = entry.num_out_tokens_fwd
    if entry.num_global_tokens_per_local_expert_dev is not None:
        dispatcher.num_global_tokens_per_local_expert = (
            entry.num_global_tokens_per_local_expert_dev
        )


def _values_equal(fresh, cached) -> bool:
    """Bitwise equality across the metadata value types (tensor/ndarray/int)."""
    if fresh is None or cached is None:
        return fresh is None and cached is None
    if torch.is_tensor(fresh):
        return (
            torch.is_tensor(cached)
            and fresh.shape == cached.shape
            and fresh.dtype == cached.dtype
            and torch.equal(fresh, cached)
        )
    if isinstance(fresh, np.ndarray):
        return (
            isinstance(cached, np.ndarray)
            and fresh.shape == cached.shape
            and fresh.dtype == cached.dtype
            and np.array_equal(fresh, cached)
        )
    return type(fresh) is type(cached) and fresh == cached


def _verify_mismatch_detail(name, fresh_val, cached_val) -> str:
    """Boundary-flip forensics for a verify mismatch (bohr's ask, ARM-1 failure).

    Reports mismatch count / total / max |diff| and the first mismatching flat
    indices — for the per-expert counts vector the index IS the expert id, for
    the [ep] splits it is the rank. A boundary-flip signature (few elements,
    small ±k, paired experts) points at replay-vs-fwd ULP divergence in the
    router; wholesale mismatch points at a store/restore bug.
    """
    try:
        if torch.is_tensor(fresh_val) and torch.is_tensor(cached_val):
            f, c = fresh_val.reshape(-1), cached_val.reshape(-1)
            if f.shape != c.shape:
                return f"{name}: shape {tuple(fresh_val.shape)} vs {tuple(cached_val.shape)}"
            neq = f != c
            n_mismatch = int(neq.sum().item())
            if n_mismatch == 0:
                return f"{name}: torch.equal False but elementwise diff empty (dtype/metadata?)"
            idx = neq.nonzero().reshape(-1)[:16].tolist()
            max_abs = int((f - c).abs().max().item()) if f.numel() else 0
            return (
                f"{name}: {n_mismatch}/{f.numel()} elements differ, max|diff|={max_abs}, "
                f"first flat indices={idx}, fresh[idx]={f[neq][:16].tolist()}, "
                f"cached[idx]={c[neq][:16].tolist()}"
            )
    except Exception as exc:  # forensics must never mask the real failure
        return f"{name}: (forensics failed: {exc})"
    return f"{name}: fresh={fresh_val!r} cached={cached_val!r}"


def _verify_routing_map_detail(dispatcher, entry) -> str:
    """routing_map XOR forensics: the entry carries the first pass's (padded)
    routing_map in verify mode; the replay's fresh map is on the dispatcher.
    Returns the flip count + which token rows / experts flipped."""
    stored = getattr(entry, "routing_map_dev", None)
    fresh = getattr(dispatcher, "routing_map", None)
    if stored is None or fresh is None or not (torch.is_tensor(stored) and torch.is_tensor(fresh)):
        return "routing_map XOR: unavailable (not stashed or not tensors)"
    if stored.shape != fresh.shape:
        return f"routing_map XOR: shape {tuple(fresh.shape)} vs {tuple(stored.shape)}"
    flips = fresh != stored
    n_flips = int(flips.sum().item())
    if n_flips == 0:
        return "routing_map XOR: 0 flips (maps identical — divergence is downstream of routing)"
    rows = flips.any(dim=1).nonzero().reshape(-1)
    row_list = rows[:8].tolist()
    per_row = {
        int(r): flips[r].nonzero().reshape(-1).tolist()[:8] for r in rows[:8]
    }
    return (
        f"routing_map XOR: {n_flips} flipped entries across {rows.numel()} token rows "
        f"(first rows={row_list}; per-row flipped experts={per_row})"
    )


def _verify_device_metadata(dispatcher, entry, num_tokens_per_local_expert):
    """Verify mode: assert the replay's fresh device metadata matches the cache."""
    fresh = _capture_device_metadata(dispatcher, num_tokens_per_local_expert)
    for name, fresh_val in fresh.items():
        if not _values_equal(fresh_val, getattr(entry, name)):
            # Boundary-flip forensics BEFORE raising (ARM-1 verify failure):
            # per-field mismatch detail + the routing_map XOR — a few ±k flips
            # on paired experts = replay-vs-fwd ULP divergence in the router;
            # wholesale mismatch = a store/restore bug.
            logger.error(
                "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY: %s",
                _verify_mismatch_detail(name, fresh_val, getattr(entry, name)),
            )
            logger.error(
                "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY: %s",
                _verify_routing_map_detail(dispatcher, entry),
            )
            raise RuntimeError(
                "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY: replay device metadata mismatch in "
                f"{name} on {type(dispatcher).__name__} (routing_shape {entry.routing_shape}): "
                "the recompute replay did not reproduce the first pass bitwise — "
                "do NOT enable BT_MOE_DISPATCH_REPLAY_CACHE on this configuration"
            )


def _verify_host_metadata(dispatcher, entry, tokens_per_expert_host):
    """Verify mode: assert the replay's fresh host metadata matches the cache."""
    pairs = [
        ("input_splits", dispatcher.input_splits, entry.input_splits_host),
        ("output_splits", dispatcher.output_splits, entry.output_splits_host),
        ("output_splits_tp", dispatcher.output_splits_tp, entry.output_splits_tp_host),
        ("num_out_tokens", dispatcher.num_out_tokens, entry.num_out_tokens_host),
        ("tokens_per_expert", tokens_per_expert_host, entry.tokens_per_expert_host),
    ]
    if entry.num_global_tokens_per_local_expert_host is not None:
        pairs.append(
            (
                "num_global_tokens_per_local_expert",
                getattr(dispatcher, "num_global_tokens_per_local_expert", None),
                entry.num_global_tokens_per_local_expert_host,
            )
        )
    if entry.w2_local_counts_host is not None:
        pairs.append(
            (
                "w2_local_counts",
                getattr(dispatcher, "_w2_local_counts_host", None),
                entry.w2_local_counts_host,
            )
        )
    if entry.w2_global_counts_host is not None:
        pairs.append(
            (
                "w2_global_counts",
                getattr(dispatcher, "_w2_global_counts_host", None),
                entry.w2_global_counts_host,
            )
        )
    for name, fresh_val, cached_val in pairs:
        if not _values_equal(fresh_val, cached_val):
            raise RuntimeError(
                "BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY: replay host metadata mismatch in "
                f"{name} on {type(dispatcher).__name__} (routing_shape {entry.routing_shape}): "
                "the recompute replay did not reproduce the first pass bitwise — "
                "do NOT enable BT_MOE_DISPATCH_REPLAY_CACHE on this configuration"
            )
    _REPLAY_STATS["verifies"] += 1


def _store_replay_entry(dispatcher, rec, tokens_per_expert_host):
    """First pass: stash this layer-microbatch's metadata on the frame's key_obj.

    Called at the sync point, right after d2h_event.synchronize(), when all
    host values are valid. The entry is popped by the matching replay (at
    preprocess time) and otherwise dies with the microbatch carrier.
    """
    if rec.device_vals is None:
        # preprocess did not run in this pass (cannot happen in the dispatch
        # flow); refuse to store rather than stash a partial entry.
        return
    entry = _ReplayEntry()
    entry.routing_shape = tuple(dispatcher.routing_map.shape)
    for name, val in rec.device_vals.items():
        setattr(entry, name, val)
    entry.input_splits_host = dispatcher.input_splits
    entry.output_splits_host = dispatcher.output_splits
    entry.output_splits_tp_host = dispatcher.output_splits_tp
    entry.num_out_tokens_host = dispatcher.num_out_tokens
    entry.tokens_per_expert_host = tokens_per_expert_host
    # num_global_tokens_per_local_expert is host-moved at the DtoH point only
    # in the unfused path (moe_permute_fusion=False); record the host form
    # only when the status-quo path would have produced one.
    if dispatcher.num_local_experts > 1 and not dispatcher.config.moe_permute_fusion:
        entry.num_global_tokens_per_local_expert_host = getattr(
            dispatcher, "num_global_tokens_per_local_expert", None
        )
    # BT_MOE_A2A_PIPELINE: cache the chunk-metadata matrices alongside the
    # status-quo host values so a replay hit rebuilds the chunk plan with no
    # D2H and no event sync.
    if getattr(dispatcher, "_w2_config", None) is not None:
        entry.w2_local_counts_host = getattr(dispatcher, "_w2_local_counts_host", None)
        entry.w2_global_counts_host = getattr(dispatcher, "_w2_global_counts_host", None)
    # Verify mode only: keep the first pass's routing_map for the mismatch
    # forensics' XOR detail (boundary flips vs wholesale corruption).
    if _replay_verify_enabled():
        entry.routing_map_dev = dispatcher.routing_map
    store = getattr(rec.key_obj, _REPLAY_CACHE_ATTR, None)
    if store is None:
        store = {}
        setattr(rec.key_obj, _REPLAY_CACHE_ATTR, store)
    store[id(dispatcher)] = entry
    _REPLAY_STATS["stores"] += 1
    if not _REPLAY_ARMED_LOGGED[0]:
        _REPLAY_ARMED_LOGGED[0] = True
        logger.warning(
            "BT_MOE_DISPATCH_REPLAY_CACHE: armed — first replay-metadata entry stored"
        )


def _replay_pass_begin(dispatcher, routing_map):
    """Classify this dispatcher pass and, on a replay hit, restore the cache.

    Returns the per-pass record (also kept on the frame scratch for the
    dispatcher's later call sites in the same pass), or None when the gate is
    off or the pass is not inside a wrapped checkpoint (eval / inference /
    non-recompute training — status-quo path).
    """
    if not _replay_cache_enabled():
        return None
    frame = _current_pass_frame()
    if frame is None:
        return None
    rec = frame.scratch.get(id(dispatcher))
    if rec is not None:
        # Repeat call within one frame (e.g. a nested MLP-level recompute
        # replaying inside a block-level replay): re-install the cached state
        # so the pass is self-consistent.
        if rec.mode == "hit" and not rec.verify:
            _restore_device_metadata(dispatcher, rec.entry)
        return rec
    if not frame.is_replay:
        rec = _ReplayPassRecord("store", frame.key_obj)
        frame.scratch[id(dispatcher)] = rec
        return rec
    # Replay pass: consume the entry the matching first pass stored.
    store = getattr(frame.key_obj, _REPLAY_CACHE_ATTR, None)
    entry = store.pop(id(dispatcher), None) if store is not None else None
    if entry is None:
        rec = _ReplayPassRecord("fallback", frame.key_obj)
        _REPLAY_STATS["misses"] += 1
        logger.warning(
            "BT_MOE_DISPATCH_REPLAY_CACHE: replay cache MISS on %s — falling back to the "
            "recompute path (all-gather + D2H + event sync) for this pass",
            type(dispatcher).__name__,
        )
    elif entry.routing_shape != tuple(routing_map.shape):
        rec = _ReplayPassRecord("fallback", frame.key_obj)
        _REPLAY_STATS["shape_mismatches"] += 1
        logger.warning(
            "BT_MOE_DISPATCH_REPLAY_CACHE: routing_map shape %s != cached %s on %s — "
            "falling back to the recompute path",
            tuple(routing_map.shape),
            entry.routing_shape,
            type(dispatcher).__name__,
        )
    else:
        verify = _replay_verify_enabled()
        rec = _ReplayPassRecord("hit", frame.key_obj, entry=entry, verify=verify)
        _REPLAY_STATS["hits"] += 1
        if not verify:
            _restore_device_metadata(dispatcher, entry)
        if not _REPLAY_HIT_LOGGED[0]:
            _REPLAY_HIT_LOGGED[0] = True
            logger.warning(
                "BT_MOE_DISPATCH_REPLAY_CACHE: first replay cache hit — replay all-gather, "
                "D2H copies, and d2h_event.synchronize() skipped"
            )
    frame.scratch[id(dispatcher)] = rec
    _replay_note_lookup()
    return rec


def _replay_pass_record(dispatcher):
    """Return this pass's record for the dispatcher, or None (gate off / no frame)."""
    if not _replay_cache_enabled():
        return None
    frame = _current_pass_frame()
    if frame is None:
        return None
    return frame.scratch.get(id(dispatcher))


# ----------------------------------------------------------------------------
# BT_MOE_PROBS_A2A_COMM (env-gated, default OFF)
#
# The alltoall dispatcher issues two all-to-alls per dispatch: the tokens A2A
# (bandwidth-bound; 805 MB bf16 at GLM-5.2 EP16/131k, p50 ~10.7 ms) and the
# probs A2A (latency-bound; ~66 KB f32 per peer, ~5.7 ms). Both ride the same
# EP communicator, so they serialize on its internal NCCL stream, and the
# probs call sits fully exposed between the tokens A2A and the expert
# sort/GEMM that consumes both (measured 5.13 s per 4x131k step over 900
# layer-passes; see experiment_artefacts/glm/lps_1062_perf/overlap_design/
# DESIGN_helmholtz.md). With this gate on, the probs A2A rides a SECOND
# communicator over the same ranks (created once per EP group, shared by all
# layers) and both A2As are issued before either wait
# (all_to_all_deferred/wait_deferred_a2a), so the probs call overlaps the
# tokens call. The shared-expert fc1 keeps its existing overlap: it is still
# launched between the issues and the waits (push order matters at
# CUDA_DEVICE_MAX_CONNECTIONS=1 — every independent op is pushed before the
# first compute-stream wait).
#
# Composition with BT_MOE_DISPATCH_REPLAY_CACHE: this gate consumes only
# self.input_splits / self.output_splits, which are valid host values at
# token_dispatch time in every replay-cache mode (normal: D2H'd; hit: restored
# from the cache with the D2H/event sync skipped; verify: freshly D2H'd). The
# issue/issue/wait-both ordering adds no dependency on the D2H event.
#
# Numerics: bitwise-safe. Both collectives move bytes verbatim; the second
# communicator carries the identical messages the EP communicator would.
#
# Backward: each deferred A2A's backward is the status-quo reverse all-to-all
# (issued and waited inline, on its own communicator — the probs reverse no
# longer serializes between the two token reverses on the EP communicator's
# stream when CUDA_DEVICE_MAX_CONNECTIONS > 1; at =1 the backward timing is
# unchanged because the probs-reverse wait head-of-line-blocks the tokens
# reverse issue).
#
# Telemetry (WARNING level; logger.info from megatron.core does not reach
# trainer_srun.log): one-time gate-state line, one-time armed line when the
# second communicator is created, and per-window issue counters every 300
# gated dispatches (~1 step at 75 MoE layers x 4 microbatches) for the first
# 100 windows. A present-but-inert patch is never silent: gate ON with EP=1
# (no all-to-all at all) logs armed=NO once.
# ----------------------------------------------------------------------------

_PROBS_A2A_COMM_GATE_LOGGED = [False]
_PROBS_A2A_COMM_ARMED_LOGGED = [False]
_PROBS_A2A_COMM_STATS = {"token_issues": 0, "probs_issues": 0, "waits": 0}
_PROBS_A2A_COMM_WINDOW = [0, 0]  # [gated dispatches this window, windows logged]
_PROBS_A2A_COMM_WINDOW_SIZE = 300
_PROBS_A2A_COMM_WINDOW_LOG_MAX = 100


def _probs_a2a_comm_enabled() -> bool:
    """Lazy env-gate read (per-call; negligible at ~1 call/layer-pass).

    Logs the gate state once per process at WARNING level (logger.info from
    megatron.core does not reach trainer_srun.log) so a present-but-inert
    patch is never silent.
    """
    enabled = os.environ.get("BT_MOE_PROBS_A2A_COMM", "0") == "1"
    if not _PROBS_A2A_COMM_GATE_LOGGED[0]:
        _PROBS_A2A_COMM_GATE_LOGGED[0] = True
        if enabled:
            logger.warning(
                "BT_MOE_PROBS_A2A_COMM=1: MoE probs all-to-all on a second communicator ACTIVE"
            )
        else:
            logger.warning(
                "BT_MOE_PROBS_A2A_COMM present but DISABLED (env unset or != '1'); "
                "probs all-to-all stays serialized behind the tokens all-to-all"
            )
    return enabled


def _probs_a2a_comm_note_dispatch():
    """Per-window issue telemetry (WARNING so it reaches trainer logs)."""
    _PROBS_A2A_COMM_STATS["token_issues"] += 1
    _PROBS_A2A_COMM_STATS["probs_issues"] += 1
    _PROBS_A2A_COMM_STATS["waits"] += 2
    _PROBS_A2A_COMM_WINDOW[0] += 1
    if _PROBS_A2A_COMM_WINDOW[0] >= _PROBS_A2A_COMM_WINDOW_SIZE:
        _PROBS_A2A_COMM_WINDOW[0] = 0
        _PROBS_A2A_COMM_WINDOW[1] += 1
        if _PROBS_A2A_COMM_WINDOW[1] <= _PROBS_A2A_COMM_WINDOW_LOG_MAX:
            logger.warning(
                "BT_MOE_PROBS_A2A_COMM window %d (%d gated dispatches): %s",
                _PROBS_A2A_COMM_WINDOW[1],
                _PROBS_A2A_COMM_WINDOW_SIZE,
                dict(_PROBS_A2A_COMM_STATS),
            )


# ----------------------------------------------------------------------------
# BT_MOE_A2A_PIPELINE=K (env-gated, default OFF; K must divide
# num_local_experts — K=2 is the validated configuration)
#
# Intra-MoE-layer all-to-all <-> compute pipelining for the alltoall
# dispatcher (design: experiment_artefacts/glm/lps_1062_perf/overlap_design/
# DESIGN_helmholtz.md §3). The 16 local experts are split into K groups of
# L = num_local_experts/K (group g = local experts [gL, (g+1)L) on EVERY
# rank). The dispatch all-to-all is issued as K list-form all-to-alls
# (torch.distributed.all_to_all over per-peer contiguous VIEWS — group g's
# rows for dest rank r' are the contiguous slice [base(r')+off_g(r') :
# +cnt_g(r')] of the expert-major permuted buffer, because group g is the
# g-th slice of every rank's local-expert block). Chunk g+1's dispatch
# overlaps chunk g's expert compute; chunk g's combine overlaps chunk g+1's
# expert compute. Every expert's row block stays whole in one grouped-GEMM
# call (bitwise per-expert), and the combine writes each group's rows into a
# shared combine buffer at the SAME per-source offsets the unchunked path
# produces, so the final unpermute consumes a byte-identical buffer in the
# byte-identical order (bitwise regardless of TE's internal reduction order).
#
# v1 scope: forward pipelining; backward issues+waits each reverse A2A inline
# (status-quo backward timing). Zero-count peers and zero-row experts are
# supported (empty views / zero m_splits entries).
#
# Fallbacks (gate on but constraint violated — loud WARNING + status-quo
# path): K does not divide num_local_experts; drop_and_pad; expert-TP > 1;
# moe_permute_fusion off; cuda graphs; fused TEGroupedMLP impl; activation
# offloading / paged stash / moe_act recompute; expert weights requiring grad
# (wgrad would double-compute across chunk calls);
# overlap_dispatch_backward_with_experts_wgrad; moe_apply_probs_on_input.
#
# Composition:
# * BT_MOE_PROBS_A2A_COMM (W1): with W1 on, the single full probs A2A rides
#   the second communicator; with W1 off it rides the EP communicator and is
#   issued BEFORE the token chunks so it does not trail them. Per-group probs
#   are then selected on-device by one fused sort_chunks kernel (no extra
#   latency-bound A2As).
# * BT_MOE_DISPATCH_REPLAY_CACHE (FIX C): the two count matrices this gate
#   D2H's per pass are cached by FIX C across the recompute replay via two
#   W2-gated slots on its entry (w2_* below); on a replay hit the chunk plan
#   is rebuilt from the cached host values with no D2H and no event sync.
#
# Telemetry (WARNING level): one-time gate-state line; one-time armed line
# with K/L (or the fallback reason); per-window counters every 300 gated
# layer-passes (~1 step) for the first 100 windows.
# ----------------------------------------------------------------------------

_A2A_PIPELINE_GATE_LOGGED = [False]
_A2A_PIPELINE_ARMED_LOGGED = [False]
# Separate one-time flag for the first-forward trainable-experts disarm — the
# shared armed flag would suppress it (a silent disarm after a config-armed
# layer is the v1 inert-gate trap; second-review item 3a).
_A2A_PIPELINE_DISARM_LOGGED = [False]
_A2A_PIPELINE_STATS = {
    "dispatch_issues": 0,
    "combine_issues": 0,
    "waits": 0,
    "fallback_passes": 0,
}
_A2A_PIPELINE_WINDOW = [0, 0]  # [gated layer-passes this window, windows logged]
_A2A_PIPELINE_WINDOW_SIZE = 300
_A2A_PIPELINE_WINDOW_LOG_MAX = 100


def _a2a_pipeline_gate_value() -> int:
    """Lazy env-gate read (per-call; negligible at ~1 call/layer-pass)."""
    raw = os.environ.get("BT_MOE_A2A_PIPELINE", "0")
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if not _A2A_PIPELINE_GATE_LOGGED[0]:
        _A2A_PIPELINE_GATE_LOGGED[0] = True
        if value > 1:
            logger.warning("BT_MOE_A2A_PIPELINE=%s: chunked MoE A2A pipeline ACTIVE", raw)
        else:
            logger.warning(
                "BT_MOE_A2A_PIPELINE present but DISABLED (env unset, '0', or K=1); "
                "MoE dispatch/combine all-to-alls stay monolithic"
            )
    return value


def _a2a_pipeline_note_pass(dispatch_issues, combine_issues, waits, advance_window=False):
    """Per-window pipeline telemetry (WARNING so it reaches trainer logs).

    The window counter advances once per gated layer-pass (at token_dispatch);
    the dispatch_postprocess / token_combine_chunked call sites only add their
    issue/wait counts.
    """
    _A2A_PIPELINE_STATS["dispatch_issues"] += dispatch_issues
    _A2A_PIPELINE_STATS["combine_issues"] += combine_issues
    _A2A_PIPELINE_STATS["waits"] += waits
    if advance_window:
        _A2A_PIPELINE_WINDOW[0] += 1
    if _A2A_PIPELINE_WINDOW[0] >= _A2A_PIPELINE_WINDOW_SIZE:
        _A2A_PIPELINE_WINDOW[0] = 0
        _A2A_PIPELINE_WINDOW[1] += 1
        if _A2A_PIPELINE_WINDOW[1] <= _A2A_PIPELINE_WINDOW_LOG_MAX:
            logger.warning(
                "BT_MOE_A2A_PIPELINE window %d (%d gated layer-passes): %s",
                _A2A_PIPELINE_WINDOW[1],
                _A2A_PIPELINE_WINDOW_SIZE,
                dict(_A2A_PIPELINE_STATS),
            )


def _cumsum_excl(counts):
    """Exclusive prefix sum as python ints (host-side; counts is array-like)."""
    out = [0]
    total = 0
    for c in counts:
        total += int(c)
        out.append(total)
    return out


class _W2ChunkPlan:
    """Per-pass host metadata for the chunked pipeline (pure index math).

    All splits are python-int lists of length ep_size; all view bounds are
    (start, length) python-int pairs into the relevant buffer:

    * dispatch_send_bounds[g][d]: slice of the permuted buffer P (expert-id
      order) holding this rank's rows for dest d's group-g experts.
    * dispatch_recv_bounds[g][s]: slice of recv buffer g holding src s's rows.
    * combine_send_bounds[g][d]: slice of unsorted_g (same (src, expert-in-
      group) layout as recv buffer g) holding rows to send back to d.
    * combine_recv_bounds[g][s]: slice of the SHARED combine buffer holding
      the rows src s returns to this rank. The combine output mirrors the
      permuted buffer P's layout (per-src blocks of the rows THIS rank sent
      to s, expert-id order within a block, group g at offset [gL:(g+1)L)
      within the block) — so the bounds derive from the LOCAL (send-side)
      counts, not the receive-side matrix, and the final unpermute sees a
      byte-identical buffer.
    * tokens_per_expert[g]: host list of length L for the grouped GEMM.
    """

    __slots__ = (
        "K",
        "L",
        "input_splits",
        "output_splits",
        "dispatch_send_bounds",
        "dispatch_recv_bounds",
        "combine_send_bounds",
        "combine_recv_bounds",
        "tokens_per_expert",
        "total_recv_rows",
        "total_rows",
        "combine_works",
    )

    def __init__(self, K, L):
        self.K = K
        self.L = L
        self.input_splits = []
        self.output_splits = []
        self.dispatch_send_bounds = []
        self.dispatch_recv_bounds = []
        self.combine_send_bounds = []
        self.combine_recv_bounds = []
        self.tokens_per_expert = []
        self.total_recv_rows = []
        self.total_rows = 0
        # Async work handles for the K combine A2As. Carried on the plan (a
        # stable-identity Python object), NEVER on the combine buffer tensor:
        # when a custom Function returns its input tensor, autograd returns a
        # NEW alias object and tensor attributes do not propagate — so in
        # every grad-enabled pass the waits would silently never run
        # (second-review BUG A).
        self.combine_works = []


def _w2_compute_chunk_plan(local_counts, global_counts, *, K, ep_size):
    """Compute the per-group A2A splits and view bounds (pure host math).

    Args:
        local_counts: array-like [num_experts] — rows THIS rank sends to each
            global expert (num_local_tokens_per_expert).
        global_counts: array-like [ep_size, num_local_experts] — rows this
            rank receives: [src_ep_rank, local_expert] (requires expert-TP=1).
        K: number of expert groups (must divide num_local_experts).
        ep_size: EP group size.

    Returns:
        _W2ChunkPlan. Bitwise-exact restriction of the unchunked metadata:
        sum over groups of input_splits[g] == today's input_splits, etc.
    """
    import numpy as np

    local = np.asarray(local_counts, dtype=np.int64).reshape(ep_size, -1)
    glob = np.asarray(global_counts, dtype=np.int64)
    num_local_experts = glob.shape[1]
    assert num_local_experts % K == 0
    L = num_local_experts // K
    local_g = local.reshape(ep_size, K, L)  # [dest, group, expert-in-group]
    glob_g = glob.reshape(ep_size, K, L)  # [src, group, expert-in-group]

    plan = _W2ChunkPlan(K, L)
    # Per-rank row totals (today's input_splits / output_splits).
    rank_send = local.sum(axis=1)  # [ep] — rows this rank sends to each dest
    send_base = _cumsum_excl(rank_send)  # into P (and into the combine buffer:
    # the combine output mirrors P's layout — per-src blocks of the rows this
    # rank sent, so it is carved by the SEND-side totals, not the recv ones)

    for g in range(K):
        in_g = [int(x) for x in local_g[:, g, :].sum(axis=1)]
        out_g = [int(x) for x in glob_g[:, g, :].sum(axis=1)]
        plan.input_splits.append(in_g)
        plan.output_splits.append(out_g)
        plan.total_recv_rows.append(sum(out_g))
        plan.tokens_per_expert.append([int(x) for x in glob_g[:, g, :].sum(axis=0)])

        # Dispatch: send views into P (per dest), recv views into recv_g (per src).
        off_in_rank = local_g[:, :g, :].sum(axis=(1, 2)) if g > 0 else np.zeros(ep_size, np.int64)
        plan.dispatch_send_bounds.append(
            [(send_base[d] + int(off_in_rank[d]), in_g[d]) for d in range(ep_size)]
        )
        recv_base_g = _cumsum_excl(out_g)
        plan.dispatch_recv_bounds.append(
            [(recv_base_g[s], out_g[s]) for s in range(ep_size)]
        )
        # Combine: send views into unsorted_g (same layout as recv_g — rows
        # this rank received, sent back), recv views into the shared combine
        # buffer at P's per-src offsets (rows this rank sent, returned).
        plan.combine_send_bounds.append(
            [(recv_base_g[d], out_g[d]) for d in range(ep_size)]
        )
        off_in_src_block = local_g[:, :g, :].sum(axis=(1, 2)) if g > 0 else np.zeros(
            ep_size, np.int64
        )
        plan.combine_recv_bounds.append(
            [(send_base[s] + int(off_in_src_block[s]), in_g[s]) for s in range(ep_size)]
        )
    plan.total_rows = int(rank_send.sum())
    return plan


def _w2_narrow_views(tensor, bounds):
    """Per-peer contiguous views of a [rows, ...] tensor (empty views kept:
    NCCL grouped send/recv with a zero-count peer is a no-op — verified on-box
    by the T2 gate script)."""
    return [tensor.narrow(0, start, length) for (start, length) in bounds]


class _ChunkedDispatchA2A(torch.autograd.Function):
    """BT_MOE_A2A_PIPELINE: K list-form all-to-alls dispatching the permuted
    buffer by expert group, issued back-to-back async (the per-group waits are
    plain `wait_deferred_a2a` calls in dispatch_postprocess, so chunk g+1's
    A2A is in flight while chunk g's expert compute runs).

    One Function covers all K groups so the backward can write every group's
    reverse-A2A rows into views of a single grad buffer for the permuted
    input (no cross-group grad accumulation kernel, exact by disjointness).
    """

    @staticmethod
    def forward(ctx, group, permuted_tokens, plan):
        ctx.group = group
        ctx.plan = plan
        ctx.permuted_shape = permuted_tokens.shape
        recv_bufs = []
        for g in range(plan.K):
            recv_g = permuted_tokens.new_empty(
                [plan.total_recv_rows[g]] + list(permuted_tokens.size()[1:])
            )
            work = torch.distributed.all_to_all(
                _w2_narrow_views(recv_g, plan.dispatch_recv_bounds[g]),
                _w2_narrow_views(permuted_tokens, plan.dispatch_send_bounds[g]),
                group=group,
                async_op=True,
            )
            setattr(recv_g, "_deferred_a2a_work", work)
            recv_bufs.append(recv_g)
        return tuple(recv_bufs)

    @staticmethod
    def backward(ctx, *grad_recvs):
        """Reverse list-A2As into views of one grad buffer for the permuted input.

        v1: all K reverses issued async, then waited inline (status-quo
        backward timing — the K half-size reverses serialize on the
        communicator's stream like the monolithic reverse did).
        """
        plan = ctx.plan
        grad_permuted = None
        works = []
        for g in range(plan.K):
            if grad_permuted is None:
                grad_permuted = grad_recvs[g].new_zeros(ctx.permuted_shape)
            works.append(
                torch.distributed.all_to_all(
                    _w2_narrow_views(grad_permuted, plan.dispatch_send_bounds[g]),
                    _w2_narrow_views(grad_recvs[g].contiguous(), plan.dispatch_recv_bounds[g]),
                    group=ctx.group,
                    async_op=True,
                )
            )
        for work in works:
            work.wait()
        return None, grad_permuted, None


class _ChunkedCombineA2A(torch.autograd.Function):
    """BT_MOE_A2A_PIPELINE: one expert group's combine all-to-all (list form),
    writing into the shared combine buffer at the unchunked path's per-source
    offsets, so the final unpermute sees a byte-identical buffer.

    Group 0's call allocates the combine buffer; later groups receive it as an
    input and return it (the passthrough keeps the autograd chain intact so
    the unpermute's backward reaches every group's reverse A2A).
    """

    @staticmethod
    def forward(ctx, group, unsorted_g, combine_buf, plan, g):
        ctx.had_buf = combine_buf is not None
        if combine_buf is None:
            combine_buf = unsorted_g.new_empty(
                [plan.total_rows] + list(unsorted_g.size()[1:])
            )
        work = torch.distributed.all_to_all(
            _w2_narrow_views(combine_buf, plan.combine_recv_bounds[g]),
            _w2_narrow_views(unsorted_g, plan.combine_send_bounds[g]),
            group=group,
            async_op=True,
        )
        # The work handle rides on the plan (stable identity), never on the
        # combine buffer tensor (autograd aliases lose tensor attrs — BUG A).
        plan.combine_works.append(work)
        ctx.group = group
        ctx.plan = plan
        ctx.g = g
        ctx.unsorted_rows = unsorted_g.shape
        return combine_buf

    @staticmethod
    def backward(ctx, grad_buf):
        """Reverse list-A2A for this group (issued and waited inline, v1)."""
        plan = ctx.plan
        g = ctx.g
        grad_unsorted = grad_buf.new_empty(ctx.unsorted_rows)
        work = torch.distributed.all_to_all(
            _w2_narrow_views(grad_unsorted, plan.combine_send_bounds[g]),
            _w2_narrow_views(grad_buf, plan.combine_recv_bounds[g]),
            group=ctx.group,
            async_op=True,
        )
        work.wait()
        # grad_buf passes through unchanged to the previous group's Function —
        # but only when that input was actually a tensor (group 0 received
        # None); returning a grad for a non-Variable input is a RuntimeError
        # (second-review BUG B).
        return None, grad_unsorted, grad_buf if ctx.had_buf else None, None, None


class _W2PipelineConfig:
    """Init-time (pass-independent) chunked-pipeline state."""

    __slots__ = (
        "K",
        "L",
        "sort_input_chunk",
        "restore_output_chunk",
        "probs_keep_idxs",
        "probs_keep_idxs_host",
    )

    def __init__(self, K, L, tp_size, ep_size, device):
        self.K = K
        self.L = L
        # Per-group chunk re-sort index permutations, same construction as the
        # unchunked sort_input_by_local_experts / restore_output_by_local_experts
        # but for L experts per group instead of num_local_experts.
        chunk_idxs = torch.arange(L * tp_size * ep_size, device=device)
        # [tp*ep, L] -> [L, tp*ep] -> ravel: (src, expert-in-group) -> expert-major.
        self.sort_input_chunk = chunk_idxs.reshape(-1, L).T.ravel()
        # [L, tp*ep] -> [tp*ep, L] -> ravel: expert-major -> (src, expert-in-group).
        self.restore_output_chunk = chunk_idxs.reshape(L, -1).T.ravel()
        # Fixed keep-indices selecting group g's (src, expert) blocks from the
        # full (src, num_local_experts) grid, in (src, expert-in-group) order —
        # used to select per-group probs from the full probs A2A output.
        # [K, tp*ep*L]. The HOST copy feeds the UNFUSED sort_chunks call: the
        # selection is a subset of chunks, not a permutation, and the TE fused
        # kernel's generated backward assumes full coverage (returns a
        # group-sized grad for the full-size input — the T2-gate defect); the
        # unfused split/cat path's autograd scatters group grads into the
        # full-size buffer correctly.
        self.probs_keep_idxs = torch.arange(
            tp_size * ep_size * K * L, device=device
        ).reshape(tp_size * ep_size, K, L).permute(1, 0, 2).reshape(K, -1).contiguous()
        self.probs_keep_idxs_host = [row for row in self.probs_keep_idxs.cpu().numpy()]


class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the MoE Token Dispatcher.

        Args:
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        self.config = config
        self.shared_experts: Optional[SharedExpertMLP] = None
        # Whether to use NCCL stream for A2A communication, otherwise default stream is used.
        self.use_nccl_stream = False  # Will be set to True when shared_experts is set.

        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.tp_group = pg_collection.expt_tp
        self.tp_ep_group = pg_collection.tp_ep

        self.tp_size = utils.get_pg_size(self.tp_group)
        self.tp_rank = utils.get_pg_rank(self.tp_group)
        self.ep_size = utils.get_pg_size(self.ep_group)
        self.ep_rank = utils.get_pg_rank(self.ep_group)

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_modules contains moe_preprocess.
        self.cudagraph_attrs = []
        self.valid_cudagraph_attrs = None

    @abstractmethod
    def dispatch_preprocess(
        self, tokens: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Prepares tokens for dispatch without inter-device communication.

        This method should handle all local computations like tensor rearrangement and
        metadata extraction before the main communication step.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            tokens (torch.Tensor): Input tokens.
            routing_map (torch.Tensor): Token to expert mapping tensor.
            probs (torch.Tensor): The routing probability tensor, [num_tokens, num_experts].

        Returns:
            A tuple of preprocessed tokens and probabilities.
        """
        raise NotImplementedError("dispatch_preprocess function not implemented.")

    @abstractmethod
    def token_dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to expert devices using communication.

        This method performs the main communication (e.g., All-to-All) to send
        tokens to the devices where their assigned experts reside.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched.
            probs (torch.Tensor): Preprocessed probabilities for each token-expert pair.

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        raise NotImplementedError("token_dispatch function not implemented.")

    @abstractmethod
    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Performs local processing after token dispatch communication.

        This method handles post-communication tasks like token reordering and
        preparing metadata for the expert forward pass.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Dispatched hidden states.
            probs (torch.Tensor): Dispatched probabilities.

        Returns:
            A tuple containing the permuted tokens for experts, the number of
            tokens per expert, and the permuted probabilities.
        """
        raise NotImplementedError("dispatch_postprocess function not implemented.")

    @abstractmethod
    def combine_preprocess(self, hidden_states):
        """Prepares expert outputs for the combine step.

        This method performs local computations on expert outputs before the
        communication step for combining them.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): The output tensor from the experts.

        Returns:
            The preprocessed expert output.
        """
        raise NotImplementedError("combine_preprocess function not implemented.")

    @abstractmethod
    def token_combine(self, hidden_states):
        """Combines expert outputs across devices using communication.

        This method aggregates expert outputs from different devices via
        communication (e.g., All-to-All or Reduce-Scatter).

        Args:
            hidden_states (torch.Tensor): Preprocessed output from experts.

        Returns:
            The combined expert outputs.
        """
        raise NotImplementedError("token_combine function not implemented.")

    @abstractmethod
    def combine_postprocess(self, hidden_states):
        """Performs local processing after token combine.

        This method handles post-communication tasks like unpermuting and
        reshaping to restore the original tensor structure.

        Note:
            Try to avoid any communication here to enable optimal computation-communication
            overlapping when enabling communication overlap, since communications in the
            same stream runs sequentially and may get exposed.

        Args:
            hidden_states (torch.Tensor): Combined hidden states from token combination

        Returns:
            The final output tensor.
        """
        raise NotImplementedError("combine_postprocess function not implemented.")

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        assert self.config.moe_shared_expert_overlap
        self.shared_experts = shared_experts
        self.use_nccl_stream = True

    def _clear_forward_state(self, *attr_names: str) -> None:
        """Drop per-forward hand-off references once the dispatcher has consumed them."""
        for attr_name in attr_names:
            if hasattr(self, attr_name):
                setattr(self, attr_name, None)


class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    AllGather Based Token dispatcher.
    Note that this allgather spans the communication domain of TP*EP:
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """Initialize the AllGather based token dispatcher.

        Args:
            num_local_experts (int): Number of local experts.
            local_expert_indices (List[int]): Indices of local experts.
            config (TransformerConfig): Configuration for the MoE layer.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert len(self.local_expert_indices) > 0, "Expected at least one local expert index"
        self.router_topk = config.moe_router_topk
        self.add_bias = config.add_bias_linear

        # self.global_local_map: 2D tensor. A mask of mapping between global and local tokens where
        # each element is True if it's between the local_expert_indices. Only useful when cross
        # device token permutation is enabled and **AllGahter** is performed.
        self.global_local_map = None

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_modules contains moe_preprocess.
        self.cudagraph_attrs = ['routing_map']

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Reshapes hidden states and caches the routing map."""
        self.hidden_shape = hidden_states.shape
        # [S/TP, B, H] -> [S*B/TP, H]
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        self.routing_map = routing_map
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        """Gathers tokens from all TP*EP ranks using AllGather."""

        # Permute the tokens across the expert parallel devices.
        if self.tp_size > 1 or self.ep_size > 1:
            ## local_indices calculation
            with torch.no_grad():
                # [num_local_tokens, num_experts] -> [num_global_tokens, num_experts], where:
                #     num_local_tokens=(S/TP)*B, num_global_tokens=S*B*EP
                self.routing_map = gather_from_sequence_parallel_region(
                    self.routing_map, group=self.tp_ep_group
                )

            ## local_probs calculation
            # max_prob: [S/TP*B, num_experts] -> global_probs: [S*B*EP, num_experts]
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
            # Note that this allgather spans the communication domain of TP*EP.
            #  [(S/TP)*B, H] -> [((S/TP)*B)*(TP*EP), H] = [S*B*EP, H]
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group, use_global_buffer=True
            )

        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        """After gathering in token_dispatch, this method identifies tokens for local experts and
        permutes them for expert processing.
        """
        self.hidden_shape_before_permute = hidden_states.shape

        # The routing map and probs that for local experts.
        self.local_map = self.routing_map[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # probs of global token assignment to local experts.
        self.local_probs = probs[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        tokens_per_expert = self.local_map.sum(dim=0).long().cpu()

        permuted_local_hidden_states, _, self.reversed_local_input_permutation_mapping, _, _ = (
            permute(
                hidden_states,
                self.local_map,
                num_out_tokens=tokens_per_expert.sum().item(),
                fused=self.config.moe_permute_fusion,
            )
        )

        self.local_probs = self.local_probs.T.contiguous().masked_select(
            self.local_map.T.contiguous()
        )
        self.routing_map = None
        return permuted_local_hidden_states, tokens_per_expert, self.local_probs

    def combine_preprocess(self, hidden_states):
        """
        Reverses token permutation to restore original ordering before reduction operations.

        This method unpermutes the expert outputs using the cached permutation mapping
        from the dispatch phase. The unpermutation operation restores tokens to their
        original sequence positions, preparing them for the subsequent reduction scatter
        operation that will aggregate results across ranks.
        """
        unpermuted_local_hidden = unpermute(
            hidden_states,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.local_map,
            fused=self.config.moe_permute_fusion,
        )
        return unpermuted_local_hidden

    def token_combine(self, hidden_states):
        """Combines expert outputs using Reduce-Scatter.

        This method performs the ReduceScatter communication operation to collect expert
        outputs from their processing ranks and redistribute tokens back to the ranks that
        originally held them. This completes the expert processing
        communication pattern and prepares tokens for final unpermutation.
        """
        # Unpermute the tokens across ranks.
        if self.tp_size > 1 or self.ep_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.local_probs.dtype), group=self.tp_ep_group
            ).to(hidden_states.dtype)
        return hidden_states

    def combine_postprocess(self, hidden_states):
        """Restores the original tensor shape."""
        return hidden_states.view(self.hidden_shape)


class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    AlltoAll-based token dispatcher.

    The workflow of AlltoAll token dispatcher is as follows:
    (1) preprocess: calculate necessary metadata for communication and permute
    (2) dispatch process: permute tokens
    (3) token dispatch: A2A(EP)
    (4) dispatch postprocess: AG(TP)->sort_chunk(if num_local_experts>1)
    (5) combine preprocess: sort_chunk(if num_local_experts>1)->RS(TP)
    (6) token combine: A2A(EP)
    (7) combine postprocess: unpermute tokens
    """

    # DtoH copies are performed on this stream for overlapping with the main stream.
    cuda_dtoh_stream = None

    # BT_MOE_PROBS_A2A_COMM: second communicator over the EP ranks carrying the
    # probs all-to-all. Class-level and created once per EP group (the
    # pg_collection EP group is shared by all layers), mirroring
    # cuda_dtoh_stream; torch.distributed.new_group is a world-wide collective,
    # so creation happens at model build on every rank at the same point (the
    # first MoE layer's dispatcher __init__).
    _probs_a2a_group = None
    _probs_a2a_group_parent = None

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        """
        Initialize the AlltoAll token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.num_local_experts = num_local_experts
        assert config.num_moe_experts is not None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        assert (
            len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continuous"

        # [ep_size]. Represents the number of tokens sent by the current rank to other
        # EP ranks.
        self.input_splits = None
        # [ep_size]. Represents the number of tokens received by the current rank from
        # other EP ranks.
        self.output_splits = None
        # [tp_size]. Represents the number of tokens received by the current rank from
        # other TP ranks.
        self.output_splits_tp = None
        self.permute_idx_device = torch.device("cuda") if self.config.moe_permute_fusion else "cpu"
        input_chunk_idxs = torch.arange(
            self.num_experts * self.tp_size, device=self.permute_idx_device
        )
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [tp_size * ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        # Token drop and padding.
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
            self.moe_expert_capacity_factor = self.config.moe_expert_capacity_factor
        self.capacity = None

        # A cuda stream synchronization is needed in during token permutation in some cases,
        # because there are several non-blocking DtoH data transfers called at
        # `self.cuda_dtoh_point`. The synchronization happens at `self.cuda_sync_point`, which is
        # decided based on the MoE and parallel settings. Valid points are "before_permutation_1",
        # "before_ep_alltoall", "before_permutation_2", "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"
        self.cuda_sync_point_priority = {
            "before_permutation_1": 0,
            "before_ep_alltoall": 1,
            "before_permutation_2": 2,
            "before_finish": 3,
            "no_sync": 4,
        }
        self.cuda_dtoh_point = "before_permutation_1"
        if config.cuda_graph_impl != "none" and (
            CudaGraphModule.moe_preprocess in config.cuda_graph_modules
            or not self.config.cuda_graph_modules
        ):
            self.cuda_dtoh_point = "before_ep_alltoall"
        if MoEAlltoAllTokenDispatcher.cuda_dtoh_stream is None:
            MoEAlltoAllTokenDispatcher.cuda_dtoh_stream = torch.cuda.Stream()

        # BT_MOE_PROBS_A2A_COMM: create/reuse the second communicator for the
        # probs all-to-all. EP=1 bypasses the all-to-all entirely, so the gate
        # is armed but pointless there (logged once, no communicator created).
        self._probs_a2a_comm = None
        if _probs_a2a_comm_enabled():
            if self.ep_size > 1:
                if MoEAlltoAllTokenDispatcher._probs_a2a_group is None:
                    MoEAlltoAllTokenDispatcher._probs_a2a_group = torch.distributed.new_group(
                        torch.distributed.get_process_group_ranks(self.ep_group)
                    )
                    MoEAlltoAllTokenDispatcher._probs_a2a_group_parent = self.ep_group
                    if not _PROBS_A2A_COMM_ARMED_LOGGED[0]:
                        _PROBS_A2A_COMM_ARMED_LOGGED[0] = True
                        logger.warning(
                            "BT_MOE_PROBS_A2A_COMM: armed — second communicator created over "
                            "%d EP ranks (shared by all MoE layers)",
                            self.ep_size,
                        )
                assert MoEAlltoAllTokenDispatcher._probs_a2a_group_parent is self.ep_group, (
                    "BT_MOE_PROBS_A2A_COMM: dispatcher EP group changed after the probs "
                    "communicator was created — unsupported"
                )
                self._probs_a2a_comm = MoEAlltoAllTokenDispatcher._probs_a2a_group
            elif not _PROBS_A2A_COMM_ARMED_LOGGED[0]:
                _PROBS_A2A_COMM_ARMED_LOGGED[0] = True
                logger.warning(
                    "BT_MOE_PROBS_A2A_COMM: armed=NO — EP size is 1, no all-to-all to move"
                )

        # BT_MOE_A2A_PIPELINE: set by w2_try_enable_pipeline (called from
        # MoELayer.__init__, which owns the experts module the validation
        # needs). Per-pass chunked state lives in self._w2_pass and is cleared
        # with the rest of the forward state in combine_postprocess.
        self._w2_config = None
        self._w2_pass = None

    def w2_try_enable_pipeline(self, experts):
        """Validate constraints and arm BT_MOE_A2A_PIPELINE (K expert groups).

        Called once per layer from MoELayer.__init__. Any constraint violation
        falls back to the status-quo monolithic path with a one-time WARNING
        naming the reason (a gate that cannot fire is never silent). The
        expert-weights-requires_grad check is NOT done here (LoRA freezing may
        happen after model construction) — it runs once at the top of the
        first MoELayer.forward instead.
        """
        K = _a2a_pipeline_gate_value()
        if K <= 1:
            return
        reason = None
        if self.num_local_experts % K != 0:
            reason = f"K={K} does not divide num_local_experts={self.num_local_experts}"
        elif self.ep_size <= 1:
            reason = "EP size is 1, no all-to-all to pipeline"
        elif self.drop_and_pad:
            reason = "drop_and_pad is on"
        elif self.tp_size > 1:
            reason = "expert-TP > 1"
        elif not self.config.moe_permute_fusion:
            reason = "moe_permute_fusion is off (fused sort_chunks required)"
        elif self.config.cuda_graph_impl != "none":
            reason = "cuda graphs are on"
        elif getattr(experts, "_with_fused_impl", False):
            reason = "fused TEGroupedMLP impl (use_transformer_engine_op_fuser)"
        elif getattr(experts, "offload_expert_fc1", False) or getattr(
            experts, "offload_moe_act", False
        ) or getattr(experts, "offload_fused_group_mlp", False):
            reason = "fine-grained activation offloading"
        elif self.config.moe_paged_stash:
            reason = "moe_paged_stash"
        elif getattr(experts, "activation_recompute", False):
            reason = "moe_act selective recompute"
        elif self.config.overlap_dispatch_backward_with_experts_wgrad:
            reason = "overlap_dispatch_backward_with_experts_wgrad"
        elif self.config.overlap_moe_expert_parallel_comm:
            reason = "overlap_moe_expert_parallel_comm"
        elif self.config.moe_apply_probs_on_input:
            reason = "moe_apply_probs_on_input"
        if reason is not None:
            if not _A2A_PIPELINE_ARMED_LOGGED[0]:
                _A2A_PIPELINE_ARMED_LOGGED[0] = True
                logger.warning(
                    "BT_MOE_A2A_PIPELINE: armed=NO — %s; status-quo monolithic A2A path",
                    reason,
                )
            return
        L = self.num_local_experts // K
        self._w2_config = _W2PipelineConfig(K, L, self.tp_size, self.ep_size, self.permute_idx_device)
        # The "armed" line is emitted by MoELayer's first-forward check once
        # the frozen-experts stage also passes (a config-armed but
        # frozen-disabled pipeline must not log armed).

        # Attributes that need to be captured in cudagraph. These attributes are returned
        # as cudagraph outputs when the cuda_graph_modules contains moe_preprocess.
        self.cudagraph_attrs = [
            'tokens_per_expert',
            'input_splits',
            'output_splits',
            'output_splits_tp',
            'num_out_tokens',
            'num_global_tokens_per_local_expert',
            'reversed_local_input_permutation_mapping',
            'routing_map',
            'hidden_shape',
            'probs',
        ]

        self.shared_experts = None

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        super().set_shared_experts(shared_experts)
        if shared_experts.use_shared_expert_gate:
            self.cudagraph_attrs.append('shared_experts.gate_score')
        self.cudagraph_attrs.append('shared_experts.cached_fc1_input')

    def preprocess(self, routing_map: torch.Tensor) -> torch.Tensor:
        """
        Preprocesses the token routing map for All-to-All communication and token permutation.

        This method computes the number of tokens assigned to each expert based on the routing_map.
        It also initializes necessary data structures for All-to-All communication, such as input
        and output splits, and the mapping between global tokens and local experts. This method
        should not call any DtoH data copying due to performance consideration. The necessary DtoH
        copies are made on the `self.cuda_dtoh_stream` at `self.cuda_dtoh_point`.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts.

        Returns:
            A tensor with the number of tokens for each local expert.
        """
        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = routing_map.size(0) * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            # [num_local_experts], number of tokens processed by each expert.
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,),
                self.capacity * self.tp_size * self.ep_size,
                dtype=torch.long,
            )
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = torch.full(
                (self.num_experts * self.tp_size,),
                self.capacity,
                dtype=torch.long,
                device=self.permute_idx_device,
            )
            return num_tokens_per_local_expert

        # BT_MOE_DISPATCH_REPLAY_CACHE: classify this pass. On a replay hit the
        # first pass's metadata is restored onto the instance and we skip the
        # split recompute below (including the tp_ep all-gather); the replay's
        # D2H copies and event sync are skipped in _maybe_dtoh_and_synchronize.
        replay_rec = _replay_pass_begin(self, routing_map)
        if replay_rec is not None and replay_rec.mode == "hit" and not replay_rec.verify:
            return replay_rec.entry.num_tokens_per_local_expert_dev

        # [num_experts], number of tokens assigned to each expert from the current rank's input.
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()
        if self._w2_config is not None:
            # BT_MOE_A2A_PIPELINE: keep the device counts for the D2H batch
            # (per-chunk host metadata derives from this and
            # num_global_tokens_per_local_expert).
            self._w2_local_counts_dev = num_local_tokens_per_expert

        if (
            self.config.moe_expert_capacity_factor is not None
            or self.config.moe_router_padding_for_quantization
        ):
            # When using token dropping or router padding, output size is dynamic.
            # Need to sync output size GPU->CPU before allocating output buffer
            self.num_out_tokens = num_local_tokens_per_expert.sum()
            self._maybe_update_cuda_sync_point("before_permutation_1")
        else:
            # For dropless training, output size is static (num_tokens * topk)
            # No explicit sync needed
            self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk
        if self.ep_size > 1 or self.tp_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall/allgather in variable size.
            # ===================================================
            # [ep_size]. Represents the number of tokens sent by the current rank to other
            # EP ranks.
            self.input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts
            ).sum(axis=1)
            # Gather the global distribution of tokens across ranks.
            # num_global_tokens_per_expert represents the number of tokens sent to each
            # expert by all ranks.
            # [tp_size, ep_size, num_experts]
            num_global_tokens_per_expert = (
                gather_from_sequence_parallel_region(
                    num_local_tokens_per_expert, group=self.tp_ep_group
                )
                .reshape(self.ep_size, self.tp_size, self.num_experts)
                .transpose(0, 1)
            )
            # [tp_size, ep_size, num_experts] -> [tp_size, ep_size, num_local_experts]
            num_global_tokens_per_local_expert = num_global_tokens_per_expert[
                :, :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
            ].contiguous()
            # [tp_size, ep_size, num_local_experts] -> [tp_size, ep_size]
            num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
            # [tp_size, ep_size] -> [ep_size]
            # self.output_splits represents the number of tokens received by the current rank
            # from other EP rank.
            self.output_splits = num_global_tokens_per_rank[self.tp_rank]
            # [tp_size, ep_size] -> [tp_size]
            # self.output_splits_tp represents the number of tokens received by the current
            # rank from other TP rank.
            self.output_splits_tp = num_global_tokens_per_rank.sum(axis=1)
            # [tp_size, ep_size, num_local_experts] -> [num_local_experts]
            num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))

            # A synchronization is needed before expert parallel AlltoAll communication
            # to get the `input_splits` and `output_splits` CPU values.
            self._maybe_update_cuda_sync_point("before_ep_alltoall")
        else:
            num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

            # A synchronization is needed before the returns
            # to get the `num_tokens_per_local_expert` CPU value.
            self._maybe_update_cuda_sync_point("before_finish")

        if self.num_local_experts > 1:
            # [tp_size * ep_size, num_local_experts]. Represents the number of tokens sent
            # to each local expert by all ranks.
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )
            if not self.config.moe_permute_fusion:
                # A synchronization is needed before permutation 2
                # to get the `num_global_tokens_per_local_expert` CPU value.
                self._maybe_update_cuda_sync_point("before_permutation_2")

        assert (
            self.cuda_sync_point_priority[self.cuda_dtoh_point]
            <= self.cuda_sync_point_priority[self.cuda_sync_point]
        ), "cuda_sync_point must be after cuda_dtoh_point."
        if replay_rec is not None:
            if replay_rec.mode == "store":
                # First pass: keep the device-side metadata on the frame
                # scratch; it is stored on the microbatch carrier at the sync
                # point (see _maybe_dtoh_and_synchronize).
                replay_rec.device_vals = _capture_device_metadata(
                    self, num_tokens_per_local_expert
                )
            elif replay_rec.mode == "hit" and replay_rec.verify:
                # Verify mode: the full recompute ran above; assert it
                # reproduces the cached first-pass metadata bitwise.
                _verify_device_metadata(self, replay_rec.entry, num_tokens_per_local_expert)
        return num_tokens_per_local_expert

    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Prepares hidden states and probabilities for dispatch.

        This method reshapes the hidden states, computes communication metadata,
        and permutes the tokens and probabilities before the All-to-All communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            routing_map (torch.Tensor): The mapping of tokens to experts.
            probs (torch.Tensor): Routing probabilities.

        Returns:
            A tuple of permuted hidden states and probabilities.
        """
        # Preprocess: Get the metadata for communication, permutation and computation operations.
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for token2expert mask"
        assert routing_map.dtype == torch.bool, "Expected bool tensor for mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if self.config.moe_router_padding_for_quantization:
            pad_multiple = get_align_size_for_quantization(self.config)
            if is_experimental_enabled() and self.config.moe_permute_fusion:
                self.routing_map = fused_pad_routing_map(self.routing_map, pad_multiple)
            else:
                self.routing_map = pad_routing_map(self.routing_map, pad_multiple)
        self.tokens_per_expert = self.preprocess(self.routing_map)

        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states.view(self.hidden_shape))

        # Permutation 1: input to AlltoAll input
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_1", self.tokens_per_expert
        )
        self.hidden_shape_before_permute = hidden_states.shape
        (
            permutated_local_input_tokens,
            permuted_probs,
            self.reversed_local_input_permutation_mapping,
            _,
            _,
        ) = permute(
            hidden_states,
            self.routing_map,
            probs=probs,
            num_out_tokens=self.num_out_tokens,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )
        return permutated_local_input_tokens, permuted_probs

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        """
        Perform all-to-all communication for dispatching tokens.

        This method performs the all-to-all communication step to dispatch tokens across
        expert parallel ranks. It synchronizes metadata at the appropriate point before
        performing the communication.

        Args:
            permutated_local_input_tokens (torch.Tensor): Pre-permuted input tokens.
            permuted_probs (torch.Tensor): Pre-permuted probabilities.

        Returns:
            A tuple of tokens and probabilities after All-to-All.
        """
        # Make sure the shared experts fc1 is overlapped with dispatch A2A
        # when CUDA_DEVICE_MAX_CONNECTIONS>1.
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        # Perform expert parallel AlltoAll communication
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_ep_alltoall", self.tokens_per_expert
        )
        if self._w2_config is not None:
            # BT_MOE_A2A_PIPELINE: K expert-group list-A2As issued back-to-back
            # (chunk g+1's dispatch overlaps chunk g's expert compute). The
            # probs A2A stays a single full call: on the W1 second
            # communicator when armed (concurrent with the token chunks), else
            # on the EP communicator issued FIRST so it does not trail the
            # token chunks. Per-group probs are selected on-device in
            # dispatch_postprocess. Push order (matters at
            # CUDA_DEVICE_MAX_CONNECTIONS=1): all issues -> shared fc1 ->
            # probs wait; the per-group token waits happen in
            # dispatch_postprocess right before each group's sort.
            chunk_plan = _w2_compute_chunk_plan(
                self._w2_local_counts_host,
                self._w2_global_counts_host,
                K=self._w2_config.K,
                ep_size=self.ep_size,
            )
            probs_group = (
                self._probs_a2a_comm if self._probs_a2a_comm is not None else self.ep_group
            )
            if probs_group is self.ep_group:
                global_probs = all_to_all_deferred(
                    probs_group, permuted_probs, self.output_splits, self.input_splits
                )
            recv_bufs = _ChunkedDispatchA2A.apply(
                self.ep_group, permutated_local_input_tokens, chunk_plan
            )
            if probs_group is not self.ep_group:
                global_probs = all_to_all_deferred(
                    probs_group, permuted_probs, self.output_splits, self.input_splits
                )
            if self.shared_experts is not None:
                self.shared_experts.linear_fc1_forward_and_act(recv_bufs[0])
            wait_deferred_a2a(global_probs)
            self._w2_pass = {
                "plan": chunk_plan,
                "recv_bufs": recv_bufs,
                "global_probs": global_probs,
            }
            _a2a_pipeline_note_pass(
                dispatch_issues=chunk_plan.K, combine_issues=0, waits=1, advance_window=True
            )
            return recv_bufs, global_probs
        if self._probs_a2a_comm is not None:
            # BT_MOE_PROBS_A2A_COMM: issue both all-to-alls before waiting
            # either. The probs A2A rides the second communicator so its
            # latency-bound call overlaps the tokens A2A instead of
            # serializing behind it on the EP communicator's stream. The
            # shared-expert fc1 stays between the issues and the waits so its
            # launch is not blocked by a compute-stream wait at
            # CUDA_DEVICE_MAX_CONNECTIONS=1.
            # Forward launch order: tokens A2A issue -> probs A2A issue (comm 2)
            #   -> shared experts fc1 -> wait tokens -> wait probs
            # Backward launch order: probs A2A reverse (comm 2) -> tokens A2A
            #   reverse -> shared experts fc1 (as upstream).
            global_input_tokens = all_to_all_deferred(
                self.ep_group,
                permutated_local_input_tokens,
                self.output_splits,
                self.input_splits,
            )
            global_probs = all_to_all_deferred(
                self._probs_a2a_comm,
                permuted_probs,
                self.output_splits,
                self.input_splits,
            )
            if self.shared_experts is not None:
                self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)
            wait_deferred_a2a(global_input_tokens)
            wait_deferred_a2a(global_probs)
            _probs_a2a_comm_note_dispatch()
            return global_input_tokens, global_probs
        global_input_tokens = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            use_nccl_stream=self.use_nccl_stream,
        )
        # Move the shared experts fc1 right after the tokens A2A, to prevent the probs A2A
        # block the launch of fc1 GEMM when CUDA_DEVICE_MAX_CONNECTIONS=1.
        # Forward launch order: tokens A2A -> shared experts fc1 -> probs A2A
        # Backward launch order: probs A2A -> tokens A2A -> shared experts fc1
        if self.shared_experts is not None:
            self.shared_experts.linear_fc1_forward_and_act(global_input_tokens)
        global_probs = all_to_all(
            self.ep_group,
            permuted_probs,
            self.output_splits,
            self.input_splits,
            use_nccl_stream=self.use_nccl_stream,
        )

        return global_input_tokens, global_probs

    def dispatch_postprocess(self, global_input_tokens, global_probs):
        """Post-processes tokens after All-to-All communication.

        This involves an All-Gather in the tensor parallel dimension and sorting
        tokens by expert if there are multiple local experts.

        Args:
            global_input_tokens (torch.Tensor): Tokens after All-to-All.
            global_probs (torch.Tensor): Probabilities after All-to-All.

        Returns:
            A tuple of processed tokens, token counts per expert, and processed probabilities.
        """
        if self._w2_config is not None:
            # BT_MOE_A2A_PIPELINE: per-group wait -> probs selection -> sort.
            # Returns a list of K (sorted_tokens, tokens_per_expert host list,
            # sorted probs) tuples for MoELayer's chunked experts loop.
            cfg = self._w2_config
            state = self._w2_pass
            plan = state["plan"]
            counts_ge = self.num_global_tokens_per_local_expert  # device [tp*ep, K*L]
            probs_full = state["global_probs"]  # already waited in token_dispatch
            chunks = []
            for g in range(cfg.K):
                # Wait chunk g's dispatch A2A (stream wait, host-non-blocking).
                wait_deferred_a2a(state["recv_bufs"][g])
                # Select group g's probs rows ((src, expert-in-group) order)
                # from the full probs buffer. This is a SUBSET of chunks, not
                # a permutation — the TE fused sort's generated backward
                # assumes full coverage and returns a group-sized grad for the
                # full-size input (the T2-gate defect). The UNFUSED split/cat
                # path's autograd (views + cat) scatters group grads into the
                # full-size buffer correctly and is bitwise-exact; host
                # split/index lists mean no new syncs.
                probs_g = sort_chunks_by_idxs(
                    probs_full,
                    self._w2_global_counts_host.ravel(),
                    cfg.probs_keep_idxs_host[g],
                    fused=False,
                )[0]
                # Sort chunk g's received tokens by local expert within the group.
                sorted_g, probs_g_sorted = sort_chunks_by_idxs(
                    state["recv_bufs"][g],
                    counts_ge[:, g * cfg.L : (g + 1) * cfg.L].reshape(-1),
                    cfg.sort_input_chunk,
                    probs=probs_g,
                    fused=self.config.moe_permute_fusion,
                )
                chunks.append((sorted_g, plan.tokens_per_expert[g], probs_g_sorted))
            state["chunks"] = chunks
            self.tokens_per_expert = None
            _a2a_pipeline_note_pass(dispatch_issues=0, combine_issues=0, waits=cfg.K)
            return chunks
        if self.tp_size > 1:
            if self.output_splits_tp is None:
                output_split_sizes = None
            else:
                output_split_sizes = self.output_splits_tp.tolist()
            global_input_tokens = gather_from_sequence_parallel_region(
                global_input_tokens, group=self.tp_group, output_split_sizes=output_split_sizes
            )
            global_probs = gather_from_sequence_parallel_region(
                global_probs, group=self.tp_group, output_split_sizes=output_split_sizes
            )

        # Permutation 2: Sort tokens by local expert.
        self.tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_permutation_2", self.tokens_per_expert
        )
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                global_input_tokens = (
                    global_input_tokens.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_input_tokens.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
                global_probs = (
                    global_probs.view(
                        self.tp_size * self.ep_size,
                        self.num_local_experts,
                        self.capacity,
                        *global_probs.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                global_input_tokens, global_probs = sort_chunks_by_idxs(
                    global_input_tokens,
                    self.num_global_tokens_per_local_expert.ravel(),
                    self.sort_input_by_local_experts,
                    probs=global_probs,
                    fused=self.config.moe_permute_fusion,
                )

        tokens_per_expert = self._maybe_dtoh_and_synchronize(
            "before_finish", self.tokens_per_expert
        )
        self.tokens_per_expert = None
        return global_input_tokens, tokens_per_expert, global_probs

    def combine_preprocess(self, hidden_states):
        """Prepares hidden states for token combination after expert computations.

        This may involve un-sorting tokens and a Reduce-Scatter in the tensor
        parallel dimension.
        """
        # Unpermutation 2: Unsort tokens by local expert.
        if self.num_local_experts > 1:
            if self.drop_and_pad:
                hidden_states = (
                    hidden_states.view(
                        self.num_local_experts,
                        self.tp_size * self.ep_size,
                        self.capacity,
                        *hidden_states.size()[1:],
                    )
                    .transpose(0, 1)
                    .contiguous()
                    .flatten(start_dim=0, end_dim=2)
                )
            else:
                hidden_states, _ = sort_chunks_by_idxs(
                    hidden_states,
                    self.num_global_tokens_per_local_expert.T.ravel(),
                    self.restore_output_by_local_experts,
                    fused=self.config.moe_permute_fusion,
                )

        if self.tp_size > 1:
            if self.output_splits_tp is None:
                input_split_sizes = None
            else:
                input_split_sizes = self.output_splits_tp.tolist()
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states.to(self.probs.dtype),
                group=self.tp_group,
                input_split_sizes=input_split_sizes,
            ).to(hidden_states.dtype)

        return hidden_states

    def combine_preprocess_chunk(self, hidden_states, group_index):
        """BT_MOE_A2A_PIPELINE: unsort one expert group's expert outputs from
        expert-major-within-group order back to (src, expert-in-group) order
        for the combine A2A. Same fused sort_chunks restriction as the
        unchunked combine_preprocess."""
        cfg = self._w2_config
        counts_g = self.num_global_tokens_per_local_expert[
            :, group_index * cfg.L : (group_index + 1) * cfg.L
        ]
        unsorted_g, _ = sort_chunks_by_idxs(
            hidden_states,
            counts_g.T.reshape(-1),
            cfg.restore_output_chunk,
            fused=self.config.moe_permute_fusion,
        )
        return unsorted_g

    def token_combine_chunked(self, unsorted_list):
        """BT_MOE_A2A_PIPELINE: issue the K groups' combine list-A2As into the
        shared combine buffer (at the unchunked path's per-source offsets), then
        wait them all. Returns the combine buffer, which combine_postprocess
        unpermutes exactly as today."""
        state = self._w2_pass
        plan = state["plan"]
        # Make sure the shared experts fc2 is not overlapped with routed experts fc1
        # when CUDA_DEVICE_MAX_CONNECTIONS>1 (same ordering as token_combine).
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        combine_buf = None
        for g, unsorted_g in enumerate(unsorted_list):
            combine_buf = _ChunkedCombineA2A.apply(
                self.ep_group, unsorted_g, combine_buf, plan, g
            )
        # Shared-expert fc2 after all combine issues, before the waits (push
        # order at CUDA_DEVICE_MAX_CONNECTIONS=1).
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(combine_buf)
            self.shared_experts.post_forward_comm()
        for work in plan.combine_works:
            work.wait()
        _a2a_pipeline_note_pass(dispatch_issues=0, combine_issues=plan.K, waits=plan.K)
        return combine_buf

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This method performs the inverse AlltoAll communication operation to collect expert
        outputs from their processing ranks and redistribute them back to the ranks that
        originally held the corresponding tokens. This completes the expert processing
        communication pattern and prepares tokens for final unpermutation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Tokens after the All-to-All communication for combining.
        """
        if self._w2_config is not None:
            # BT_MOE_A2A_PIPELINE: the combine already happened per-group in
            # token_combine_chunked (called from MoELayer.routed_experts_compute).
            return hidden_states
        # Make sure the shared experts fc2 is not overlapped with routed experts fc1
        # when CUDA_DEVICE_MAX_CONNECTIONS>1.
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        permutated_local_input_tokens = all_to_all(
            self.ep_group,
            hidden_states,
            self.input_splits,
            self.output_splits,
            use_nccl_stream=self.use_nccl_stream,
        )
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
            self.shared_experts.post_forward_comm()
        return permutated_local_input_tokens

    def combine_postprocess(self, permutated_local_input_tokens):
        """Finalizes token reconstruction with un-permutation and reshaping.

        This method un-permutes the tokens back to their original order,
        reshapes the tensor to its original shape, and adds the shared
        expert output if enabled.

        Args:
            permutated_local_input_tokens (torch.Tensor): Permuted hidden states from token combine.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """

        # Unpermutation 1: AlltoAll output to output
        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.routing_map,
            fused=self.config.moe_permute_fusion,
            drop_and_pad=self.drop_and_pad,
        )

        # Reshape the output tensor
        output = output.view(self.hidden_shape)

        # Add shared experts output
        if self.shared_experts is not None:
            shared_expert_output = self.shared_experts.get_output()
            output += shared_expert_output

        self._clear_forward_state(
            "hidden_shape",
            "hidden_shape_before_permute",
            "probs",
            "routing_map",
            "reversed_local_input_permutation_mapping",
            "tokens_per_expert",
            "input_splits",
            "output_splits",
            "output_splits_tp",
            "num_out_tokens",
            "num_global_tokens_per_local_expert",
            "capacity",
            "d2h_event",
            "_w2_pass",
            "_w2_local_counts_dev",
            "_w2_local_counts_host",
            "_w2_global_counts_host",
        )
        return output

    def _maybe_update_cuda_sync_point(self, point: str):
        """
        Update the CUDA sync point if the priority of the new point is higher than the current
        sync point, which means the new point is reached earlier than the current sync point.
        """
        if (
            self.cuda_sync_point_priority[point]
            < self.cuda_sync_point_priority[self.cuda_sync_point]
        ):
            self.cuda_sync_point = point

    def _maybe_dtoh_and_synchronize(
        self, point: str, tokens_per_expert: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Move all possible GPU tensors to CPU and make a synchronization at the expected point.
        """
        if not self.drop_and_pad:
            # BT_MOE_DISPATCH_REPLAY_CACHE: on a replay hit, install the cached
            # first-pass host values at the DtoH point and skip the copies, the
            # event record, and the event sync entirely.
            replay_rec = _replay_pass_record(self)
            if replay_rec is not None and replay_rec.mode == "hit" and not replay_rec.verify:
                if point == self.cuda_dtoh_point:
                    entry = replay_rec.entry
                    self.input_splits = entry.input_splits_host
                    self.output_splits = entry.output_splits_host
                    self.output_splits_tp = entry.output_splits_tp_host
                    self.num_out_tokens = entry.num_out_tokens_host
                    if entry.num_global_tokens_per_local_expert_host is not None:
                        self.num_global_tokens_per_local_expert = (
                            entry.num_global_tokens_per_local_expert_host
                        )
                    if self._w2_config is not None:
                        # BT_MOE_A2A_PIPELINE: restore the cached chunk-metadata
                        # matrices (no D2H, no event sync on a replay hit).
                        self._w2_local_counts_host = entry.w2_local_counts_host
                        self._w2_global_counts_host = entry.w2_global_counts_host
                    tokens_per_expert = entry.tokens_per_expert_host
                    self.d2h_event = None
                return tokens_per_expert
            if point == self.cuda_dtoh_point:
                # Move all possible GPU tensors to CPU at self.cuda_dtoh_point.
                on_side_stream = torch.cuda.current_stream() != self.cuda_dtoh_stream
                if on_side_stream:
                    self.cuda_dtoh_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.cuda_dtoh_stream):
                    # TODO: use MemcpyBatchAsync instead.
                    tokens_per_expert = maybe_move_tensor_to_cpu(
                        tokens_per_expert, record_stream=on_side_stream
                    )
                    self.input_splits = maybe_move_tensor_to_cpu(
                        self.input_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits = maybe_move_tensor_to_cpu(
                        self.output_splits, as_numpy=True, record_stream=on_side_stream
                    )
                    self.output_splits_tp = maybe_move_tensor_to_cpu(
                        self.output_splits_tp, as_numpy=True, record_stream=on_side_stream
                    )
                    self.num_out_tokens = maybe_move_tensor_to_cpu(
                        self.num_out_tokens, record_stream=on_side_stream
                    )
                    if self.num_local_experts > 1 and not self.config.moe_permute_fusion:
                        self.num_global_tokens_per_local_expert = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, record_stream=on_side_stream
                        )
                    if self._w2_config is not None:
                        # BT_MOE_A2A_PIPELINE: add the two count matrices the
                        # chunk plan derives from to the SAME D2H batch (one
                        # event, one sync per layer-pass — unchanged).
                        self._w2_local_counts_host = maybe_move_tensor_to_cpu(
                            self._w2_local_counts_dev, as_numpy=True,
                            record_stream=on_side_stream,
                        )
                        self._w2_global_counts_host = maybe_move_tensor_to_cpu(
                            self.num_global_tokens_per_local_expert, as_numpy=True,
                            record_stream=on_side_stream,
                        )
                self.d2h_event = self.cuda_dtoh_stream.record_event()

            if point == self.cuda_sync_point:
                # Synchronize with the DtoH stream at self.cuda_sync_point.
                self.d2h_event.synchronize()
                if replay_rec is not None and replay_rec.mode == "store":
                    # First pass: all host values are valid now — stash them
                    # for this layer-microbatch's recompute replay.
                    _store_replay_entry(self, replay_rec, tokens_per_expert)
                elif replay_rec is not None and replay_rec.mode == "hit" and replay_rec.verify:
                    # Verify mode: the full D2H + sync ran above; assert the
                    # fresh host values match the cached entry bitwise.
                    _verify_host_metadata(self, replay_rec.entry, tokens_per_expert)

        return tokens_per_expert


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _HybridEPManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    HybridEP backend. See https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep for more details.

    The workflow of the HybridEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Permute tokens for communication, perform all-to-all communication,
        and permute tokens for experts in single step
    (3) combine():
        - Unpermute tokens for communication, perform all-to-all communication,
        and unpermute tokens for attention in single step
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        """
        Initialize the HybridEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            num_local_experts (int): The number of local experts.
            num_experts (int): The total number of experts in the group.
            config (TransformerConfig): The configuration for the transformer model.
        """
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.config = config
        self.permute_fusion = config.moe_permute_fusion
        self.capacity_factor = config.moe_expert_capacity_factor
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.capacity_factor is not None
        self.capacity = None
        # Actually the the up-bound for the number of tokens
        # after permute op, None means no up-bound, will cause a CPU sync
        self.num_permuted_tokens = None

        # Metadata
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None
        # Used for padding the output for each expert
        self.pad_multiple = None

        if hybrid_ep_dispatch is None:
            raise ImportError(
                "HybridEP is not installed. Please install HybridEP package from "
                "https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep."
            )

        self.moe_expert_rank_capacity_factor = self.config.moe_expert_rank_capacity_factor
        self.over_budget = torch.zeros(1, dtype=torch.bool, device='cuda')

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(num_tokens, self.num_experts)
        self.token_probs = probs.reshape(num_tokens, self.num_experts)

        if self.moe_expert_rank_capacity_factor is not None:
            pad_multiple = get_align_size_for_quantization(self.config)
            # Static upper bound on permuted tokens passed to HybridEP (dropless EP rank
            # budget). Tokens above this budget are dropped inside HybridEP; dispatch then
            # sets overflow_flag on the handle (accumulated in over_budget in dispatch()).
            budget = int(
                routing_map.shape[0]
                * self.config.moe_router_topk
                * self.moe_expert_rank_capacity_factor
            )
            # Round budget up to pad_multiple (FP8/FP4/CUTLASS alignment for permute buffers).
            budget += -budget % pad_multiple
            self.num_permuted_tokens = budget
        # else: num_permuted_tokens stays None; HybridEP sizes buffers dynamically (CPU sync
        # in dispatch) and does not drop tokens or report overflow.
        # Compute the capacity for each expert at the drop_and_pad mode
        if self.drop_and_pad:
            num_out_tokens = num_tokens * self.config.moe_router_topk
            # Drop and pad the input to capacity.
            self.capacity = get_capacity(
                num_tokens=num_out_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.capacity_factor,
            )
            # In drop_and_pad mode, the number of tokens after the permute op
            # can be computed on the CPU
            self.num_permuted_tokens = self.capacity * self.group.size() * self.num_local_experts
            self.tokens_per_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.group.size(), dtype=torch.long
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        # HybridEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning(
                    "HybridEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )
            self.token_probs = self.token_probs.float()  # downcast or upcast
        if self.config.fp8 or self.config.fp4:
            self.pad_multiple = get_align_size_for_quantization(self.config)
        dispatched_hidden, self.dispatched_probs, _, tokens_per_expert, self.handle = (
            hybrid_ep_dispatch(
                x=hidden_states,
                routing_map=self.routing_map,
                probs=self.token_probs,
                group=self.group,
                num_local_experts=self.num_local_experts,
                num_sms_dispatch_api=self.config.moe_flex_dispatcher_num_sms,
                num_sms_combine_api=self.config.moe_flex_dispatcher_num_sms,
                num_blocks_permute=self.config.moe_hybridep_num_blocks_permute,
                num_blocks_unpermute=self.config.moe_hybridep_num_blocks_unpermute,
                num_permuted_tokens=self.num_permuted_tokens,
                pad_multiple=self.pad_multiple,
                fused=self.config.moe_permute_fusion_into_hybridep,
                num_sms_preprocessing_api=self.config.moe_hybridep_num_sms_preprocessing,
            )
        )
        if self.moe_expert_rank_capacity_factor is not None:
            # Static-budget path only: handle[-1] is HybridEP overflow_flag when tokens were
            # dropped because permuted count exceeded num_permuted_tokens from setup_metadata.
            over_budget = self.handle[-1] != 0
            self.over_budget |= over_budget
        # When capacity factor is None, skip overflow tracking (no token drops). Actual
        # permuted size is resolved below via tokens_per_expert.sum() (CPU sync).

        if self.num_permuted_tokens is None:
            self.tokens_per_expert = tokens_per_expert.to(torch.int64)
            # num_permuted_tokens is necessary to allocate the output tensor for combine.
            self.num_permuted_tokens = self.tokens_per_expert.sum()
        if self.moe_expert_rank_capacity_factor is not None:
            self.tokens_per_expert = tokens_per_expert.to(torch.int64)
        return dispatched_hidden

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        hidden_states = hybrid_ep_combine(
            x=hidden_states,
            handle=self.handle,
            num_permuted_tokens=self.num_permuted_tokens,
            pad_multiple=self.pad_multiple,
            fused=self.config.moe_permute_fusion_into_hybridep,
        )
        # Release the used handle/num_permuted_tokens which could change in each iteration.
        # For drop_and_pad mode, we don't need to reset the num_permuted_tokens and
        # num_dispatched_tokens, because their values never change.
        self.handle = None
        if not self.drop_and_pad:
            self.num_permuted_tokens = None
        self._original_num_tokens = None
        self._padded_num_tokens = None
        self.routing_map = None
        self.token_probs = None
        self.dispatched_probs = None
        self.tokens_per_expert = None
        self.pad_multiple = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states, self.dispatched_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        '''
        Get the number of tokens per expert.
        '''
        return self.tokens_per_expert


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        """
        Initialize the DeepEP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the ETPxEP group.
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts for each token to select.
            num_experts (int): The total number of experts in the group.
            config (TransformerConfig): The configuration for the transformer model.
        """
        self.group = group
        self.num_local_experts = num_local_experts
        self.config = config

        self.router_topk = router_topk
        self.num_experts = num_experts
        self.router_dtype = config.moe_router_dtype
        self.capacity_factor = config.moe_expert_capacity_factor
        self.permute_fusion = config.moe_permute_fusion

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )
        # None -> 20 (DeepEP's historical mcore default when moe_flex_dispatcher_num_sms is unset).
        set_deepep_num_sms(
            config.moe_flex_dispatcher_num_sms
            if config.moe_flex_dispatcher_num_sms is not None
            else 20
        )

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]

        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                logger.warning(
                    "DeepEP only supports float32 probs, please set --moe-router-dtype=fp32"
                )
            self.token_probs = self.token_probs.float()  # downcast or upcast
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states,
                self.token_indices,
                self.token_probs,
                self.num_experts,
                self.group,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            A tuple of (routing_map, probs), where routing_map is the multihot vector
            and probs is the multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        hidden_states, _ = fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        # Manually release the metadata to avoid memory leak.
        self.dispatched_indices = None
        self.dispatched_probs = None
        # These are forward-only hand-off references; autograd Functions own
        # anything needed for backward after combine/restoration has consumed them.
        self.reversed_mapping_for_combine = None
        self.pad_offsets = None
        self.dispatched_routing_map = None
        self.hidden_shape_before_permute = None
        self.token_indices = None
        self.token_probs = None
        self.tokens_per_expert = None
        return hidden_states

    def _pad_routing_map(
        self, routing_map: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the routing map to the nearest multiple of the pad_multiple.
        """
        pad_multiple = get_align_size_for_quantization(self.config)

        num_input_tokens = routing_map.shape[0]
        target_tokens_per_expert = (
            torch.ceil(tokens_per_expert / pad_multiple) * pad_multiple
        ).long()

        # Check if there are enough tokens to pad
        enough_tokens_to_pad = torch.all(target_tokens_per_expert <= num_input_tokens)
        if not enough_tokens_to_pad:
            logger.warning(
                "Not enough tokens to pad. The total number of tokens received in this rank "
                "is smaller than the target number of tokens for each expert. "
                "Falling back to explicit padding within GroupedMLP"
            )
        else:
            if is_experimental_enabled() and self.permute_fusion:
                from megatron.core.fusions.fused_pad_routing_map import fused_pad_routing_map

                routing_map = fused_pad_routing_map(routing_map, pad_multiple)
            else:
                routing_map = pad_routing_map(routing_map, pad_multiple)
            tokens_per_expert = target_tokens_per_expert
        return routing_map, tokens_per_expert

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_experimental_enabled() and self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs, self.num_local_experts
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        if self.config.moe_router_padding_for_quantization:
            self.dispatched_routing_map, self.tokens_per_expert = self._pad_routing_map(
                self.dispatched_routing_map, self.tokens_per_expert
            )

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        (
            hidden_states,
            permuted_probs,
            self.reversed_mapping_for_combine,
            self.pad_offsets,
            self.tokens_per_expert,
        ) = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
            tokens_per_expert=self.tokens_per_expert,
            align_size=get_align_size_for_quantization(self.config),
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
            pad_offsets=self.pad_offsets,
        )
        return hidden_states


class _NCCLEPManager(_DispatchManager):
    """A manager class to handle dispatch/combine for MoE models using the NCCL Expert
    Parallelism backend, via TransformerEngine's transformer_engine.pytorch.ep API
    (wrapped in fused_a2a.py).

    The workflow mirrors the other flex backends:
    (1) setup_metadata(): reconstruct topk indices/probs from the routing map (like DeepEP).
    (2) dispatch(): TE ep_dispatch permutes tokens to expert-major layout and performs the
        all-to-all in one step, returning a packed receive buffer + per-expert counts.
    (3) get_permuted_hidden_states_by_experts(): the receive buffer is already expert-major,
        so this only narrows it to the valid (sum of per-expert counts) rows for the experts.
    (4) get_restored_hidden_states_by_experts(): re-expand the expert output back into the
        static receive-capacity buffer that TE ep_combine writes from.
    (5) combine(): TE ep_combine scatters expert outputs back to the original tokens.

    The TE NCCL EP context (a single EpBuffer) and the process-wide bootstrap are created
    lazily on the first dispatch, when the local token count is known.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        router_topk: int,
        num_experts: int,
        config: TransformerConfig,
    ):
        """
        Initialize the NCCL EP dispatcher.

        Args:
            group (torch.distributed.ProcessGroup): The process group to use for communication.
                This should be the TPxEP group.
            num_local_experts (int): The number of local experts.
            router_topk (int): The number of experts each token selects (TP-folded).
            num_experts (int): The total number of experts in the group (TP-folded).
            config (TransformerConfig): The configuration for the transformer model.
        """
        self.group = group
        self.num_local_experts = num_local_experts
        self.router_topk = router_topk
        self.num_experts = num_experts
        self.config = config
        # With MoE latent projections, the dispatcher operates on latent-dim tensors
        # (fc1_latent_proj runs before dispatch; see moe_layer.py), so the EP buffers must be
        # sized to the latent dim, not hidden_size.
        self.hidden_dim = config.moe_latent_size or config.hidden_size
        # Per-expert packing alignment for the receive buffer (grouped-GEMM tile)
        self.alignment = get_align_size_for_quantization(config)
        self.rank_capacity_factor = config.moe_expert_rank_capacity_factor
        self.static_shape = config.moe_ncclep_static_shape
        if config.moe_ncclep_use_symm_mem:
            raise NotImplementedError(
                "moe_ncclep_use_symm_mem (symm-mem / zero-copy EP payload buffers) is not "
                "supported yet."
            )
        if self.static_shape:
            if torch.cuda.get_device_capability()[0] < 10:
                raise ValueError(
                    "moe_ncclep_static_shape=True requires an sm100+ (Blackwell or later) GPU with "
                    "a CuTe DSL / device-offset grouped GEMM; leave it False (dynamic shape) on "
                    "older GPUs."
                )
            if not (config.use_transformer_engine_op_fuser or config.moe_grouped_gemm):
                raise ValueError(
                    "moe_ncclep_static_shape=True requires the fused grouped GEMM; enable "
                    "use_transformer_engine_op_fuser (or moe_grouped_gemm)."
                )
            if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
                raise ValueError(
                    "moe_ncclep_static_shape=True requires the CuTe DSL grouped GEMM; set "
                    "NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 (the expert grouped GEMM must consume ragged "
                    "per-expert counts on device)."
                )

        if nccl_ep_dispatch is None:
            raise ImportError(
                "TransformerEngine NCCL EP is unavailable. The 'ncclep' backend requires a "
                "TransformerEngine build with NCCL EP support (NVTE_BUILD_WITH_NCCL_EP=1)."
            )
        if self.rank_capacity_factor is None:
            raise ValueError(
                "The 'ncclep' backend requires moe_expert_rank_capacity_factor to be set: it "
                "sizes the per-rank receive buffer. Exceeding the budget hard-traps, so set it "
                "generously."
            )

        # Fresh EpBuffer per dispatch, held until the matching combine consumes it. dispatch
        # and combine share one buffer: handle_mem is the routing table that dispatch writes
        # and combine reads. Safe because dispatch i / combine i strictly alternate.
        self._buffer = None
        self._bootstrapped: bool = False
        self._max_tokens_per_rank: Optional[int] = None

        self._recv_capacity: Optional[int] = None

        # Metadata
        self.token_probs: Optional[torch.Tensor] = None
        self.token_indices: Optional[torch.Tensor] = None
        self.dispatched_probs: Optional[torch.Tensor] = None
        self.tokens_per_expert: Optional[torch.Tensor] = None
        self.num_local_tokens: Optional[int] = None

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        num_tokens = routing_map.shape[0]
        probs = probs.reshape(num_tokens, self.num_experts)
        # Convert the multihot routing map to (topk weights, topk indices), like DeepEP.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        self.num_local_tokens = num_tokens

    def _ensure_bootstrap(self):
        """Bootstrap NCCL EP and size the receive buffer on first use (static shapes)."""
        if self._bootstrapped:
            return
        # NCCL EP's HT backend requires max_dispatch_tokens_per_rank to be a multiple of the HT
        # chunk size (64); ncclEpCreateGroup otherwise fails with "invalid usage".
        # (nccl_ep device/hybridep_adapter.cu).
        _HT_TOKENS_PER_CHUNK = 64
        self._max_tokens_per_rank = (
            (self.num_local_tokens + _HT_TOKENS_PER_CHUNK - 1)
            // _HT_TOKENS_PER_CHUNK
            * _HT_TOKENS_PER_CHUNK
        )
        budget = int(self._max_tokens_per_rank * self.router_topk * self.rank_capacity_factor)
        if self.alignment != 0:
            budget += -budget % self.alignment
        self._recv_capacity = budget

        ensure_nccl_ep_bootstrapped(
            self.group,
            num_experts=self.num_experts,
            max_tokens_per_rank=self._max_tokens_per_rank,
            recv_capacity_per_rank=self._recv_capacity,
            hidden_dim=self.hidden_dim,
            num_sms=(
                self.config.moe_flex_dispatcher_num_sms
                if self.config.moe_flex_dispatcher_num_sms is not None
                else 0
            ),
            zero_copy=False,
        )
        self._bootstrapped = True

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        # Note: this needs to stay out of the torch.compile region because TE's ep_bootstrap does
        # opaque ProcessGroup._get_backend()._comm_ptr() access that dynamo cannot trace.
        self._ensure_bootstrap()
        # Fresh buffer per dispatch; held until the matching combine consumes it.
        self._buffer = new_nccl_ep_buffer(
            top_k=self.router_topk,
            max_tokens_per_rank=self._max_tokens_per_rank,
            recv_capacity_per_rank=self._recv_capacity,
            hidden_dim=self.hidden_dim,
            num_local_experts=self.num_local_experts,
            alignment=self.alignment,
        )
        # TE requires int64 indices and float32 weights.
        # token_indices/token_probs: [num_local_tokens, router_topk]
        topk_idx = self.token_indices
        topk_weights = self.token_probs.float()
        # hidden_states: [num_local_tokens, H] -> recv_tokens: [recv_capacity_per_rank, H]
        #   tokens_per_expert: [num_local_experts]
        #   dispatched_probs: [recv_capacity_per_rank]
        recv_tokens, tokens_per_expert, dispatched_probs = nccl_ep_dispatch(
            self._buffer, hidden_states, topk_idx, topk_weights
        )
        self.tokens_per_expert = tokens_per_expert.to(torch.int64)
        self.dispatched_probs = dispatched_probs
        return recv_tokens

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.static_shape:
            return hidden_states, self.dispatched_probs
        # narrow to the sum(counts) valid (alignment-padded) rows the experts consume.
        num_valid = int(self.tokens_per_expert.sum().item())  # sum(counts) = Σ
        permuted_hidden = hidden_states[:num_valid]  # [recv_capacity_per_rank, H] -> [Σ, H]
        permuted_probs = self.dispatched_probs[:num_valid]  # [recv_capacity_per_rank] -> [Σ]
        return permuted_hidden, permuted_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        '''
        Get the number of tokens per expert.
        '''
        return self.tokens_per_expert

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # TE ep_combine reads from the static [recv_capacity, H] buffer. static_shape=False path the
        # experts ran on the narrowed [Σ, H] slice, so re-expand back to recv_capacity; in
        # static_shape mode the output is already recv_capacity rows (no-op). Rows beyond the valid
        # region map to no token and combine ignores them.
        num_valid = hidden_states.shape[0]
        pad_rows = self._recv_capacity - num_valid
        if pad_rows > 0:
            hidden_states = torch.cat(
                [hidden_states, hidden_states.new_zeros(pad_rows, hidden_states.shape[-1])], dim=0
            )
        return hidden_states

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ) -> torch.Tensor:
        # hidden_states: [recv_capacity_per_rank, H] -> [num_local_tokens, H]
        hidden_states = nccl_ep_combine(
            self._buffer, hidden_states, num_local_tokens=self.num_local_tokens
        )
        # Drop the buffer; backward keeps handle_mem alive via save_for_backward.
        self._buffer = None
        # Release per-iteration metadata.
        self.dispatched_probs = None
        self.tokens_per_expert = None
        return hidden_states


class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """A flexible token dispatcher that abstracts the underlying tensor and expert
    parallelism. It uses a single communication group over all TP and EP ranks,
    making the dispatch logic independent of the specific parallelism strategy.
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TransformerConfig): Configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        if self.config.moe_flex_dispatcher_backend == "deepep":
            assert self.tp_size * self.ep_size > 1, "DeepEP dispatcher requires TPxEP > 1"
            self._comm_manager = _DeepepManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                router_topk=self.tp_size * self.config.moe_router_topk,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.token_indices']
        elif self.config.moe_flex_dispatcher_backend == "hybridep":
            self._comm_manager = _HybridEPManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.routing_map']
        elif self.config.moe_flex_dispatcher_backend == "ncclep":
            assert self.tp_size * self.ep_size > 1, "NCCL EP dispatcher requires TPxEP > 1"
            self._comm_manager = _NCCLEPManager(
                group=self.tp_ep_group,
                num_local_experts=self.num_local_experts,
                router_topk=self.tp_size * self.config.moe_router_topk,
                num_experts=self.tp_size * self.config.num_moe_experts,
                config=self.config,
            )
            self.cudagraph_attrs = ['_comm_manager.token_probs', '_comm_manager.token_indices']
        else:
            raise ValueError(
                f"Invalid backend: {self.config.moe_flex_dispatcher_backend}"
                "Please set --moe-flex-dispatcher-backend to deepep, hybridep, or ncclep"
            )

    def _initialize_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.

        This function expands the routing_map from shape [num_local_tokens, num_experts] to
        [num_local_tokens, world_size, num_local_experts]. Each element in the routing_map
        indicates whether a token should be sent to a specific rank. Specifically, the
        routing_map is replicated across TP group since each TP ranks in a TP group should
        receive the same tokens.
        """
        num_local_tokens = routing_map.shape[0]
        world_size = self.tp_size * self.ep_size
        # Organize routing map and probs to [num_local_tokens, world_size, num_local_experts]
        routing_map = (
            routing_map.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()

        return routing_map, probs

    @jit_fuser
    def dispatch_preprocess(
        self, hidden_states: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor
    ):
        """Initializes routing metadata and prepares tensors for fused dispatch.

        This method reshapes input tensors and processes routing information into a
        unified format, where the routing map is expanded to cover the TPxEP communication domain,
        enabling the token dispatch logic to be agnostic to parallelism strategies.

        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            routing_map (torch.Tensor): Map indicating which expert each token should be routed to
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            A tuple of reshaped hidden states and token probabilities.
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        routing_map, probs = self._initialize_metadata(routing_map, probs)

        self._comm_manager.setup_metadata(routing_map, probs)
        return hidden_states, self._comm_manager.token_probs

    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Execute fused permutation and AlltoAll communication.

        This method currently leverages DeepEP's fused dispatch kernel, which combines token
        permutation and AlltoAll communication into a single optimized operation.
        The fused approach reduces memory bandwidth requirements and enables better
        overlap between computation and communication operations.

        Args:
            hidden_states (torch.Tensor): Preprocessed hidden states to be dispatched
            probs (torch.Tensor): Routing probabilities (unused in current implementation)
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            A tuple of dispatched tokens and probabilities.
        """
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        dispatched_hidden_states = self._comm_manager.dispatch(
            hidden_states, async_finish, allocate_on_comm_stream
        )
        if self.shared_experts is not None:
            self.shared_experts.pre_forward_comm(hidden_states, wait_current_stream=False)
            self.shared_experts.linear_fc1_forward_and_act(dispatched_hidden_states)

        return dispatched_hidden_states, self._comm_manager.dispatched_probs

    def dispatch_postprocess(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Converts dispatched tokens to a per-expert format for expert processing.

        This method transforms the output of the fused dispatch into the tensor
        organization required for the expert computation.

        Args:
            hidden_states (torch.Tensor): Hidden states after fused dispatch
            probs (torch.Tensor): Routing probabilities after fused dispatch

        Returns:
            A tuple of permuted tokens, token counts per expert, and permuted probabilities.
        """
        global_input_tokens, permuted_probs = (
            self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """Pre-processes hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        return hidden_states

    def token_combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """Executes fused un-permutation and communication using DeepEP kernels.

        This is the inverse of the `token_dispatch` operation.

        Args:
            hidden_states (torch.Tensor): Expert outputs ready for combination
            async_finish (bool): Whether to use asynchronous communication completion
            allocate_on_comm_stream (bool): Whether to allocate buffers on communication stream

        Returns:
            Combined tokens after fused un-permutation and communication.
        """
        # Make sure the shared experts fc2 is not overlapped with routed experts GEMM
        # when CUDA_DEVICE_MAX_CONNECTIONS>1.
        if self.shared_experts is not None:
            self.shared_experts.wait_current_stream()
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Restores the original tensor shape and finalizes the MoE layer output.

        This method performs the final step of the MoE token processing pipeline
        by reshaping the combined tokens back to their original input dimensions.

        Args:
            hidden_states (torch.Tensor): Combined tokens.

        Returns:
            The final MoE layer output reshaped to its original dimensions.
        """
        if self.shared_experts is not None:
            self.shared_experts.linear_fc2_forward(hidden_states)
            self.shared_experts.post_forward_comm()
            hidden_states += self.shared_experts.get_output()
        hidden_states = hidden_states.view(self.hidden_shape)
        self._clear_forward_state("hidden_shape")
        return hidden_states

    def check_over_budget(self):
        """Check if the dispatcher has exceeded its budget."""
        if hasattr(self._comm_manager, 'over_budget'):
            return self._comm_manager.over_budget
        else:
            return None

    def reset_over_budget(self):
        """Reset the accumulated over-budget flag on the communication manager."""
        if hasattr(self._comm_manager, 'over_budget'):
            self._comm_manager.over_budget.fill_(0)
