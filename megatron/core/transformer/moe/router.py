# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    apply_biased_logits,
    apply_random_logits,
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    get_tokens_per_expert_and_token_count,
    router_gating_linear,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.moe.router_replay import RouterReplay
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# BT_MOE_ROUTING_REPLAY_FORCE (env-gated, default OFF) — FIX C-prime.
#
# Motivating evidence (ARM-1 verify failure, 2026-08-09, deterministic 2/2):
# BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY failed on
# num_tokens_per_local_expert_dev at routing_shape (8192, 256) in the 131k
# boot warmup replay — the recompute replay's routing_map is NOT bitwise-equal
# to the first pass's on this stack (grad-mode kernel/algo selection upstream
# of the router — TE/FP8 activation-caching variants under enable_grad — drifts
# the replay's hidden states at ULP level, and boundary tokens flip topk).
#
# C-prime makes the replay's routing DECISION bitwise-identical to the first
# pass BY CONSTRUCTION instead of by assumption: the first pass stashes its
# routing_map on the per-microbatch packed_seq_params carrier (via the FIX C
# checkpoint-pass frames) keyed by layer_number; the replay pops it and masks
# its own recomputed logits with -inf on every unsaved (token, expert) pair
# before the score function, so the top-k selection is forced to the saved
# set. The returned routing_map IS the saved map (bitwise); the probs are
# recomputed from the replay's masked logits — grad-connected, so the
# router-path term in dL/d(hidden_states) is preserved exactly as in the
# status-quo replay (consuming detached saved probs would drop it — a
# structural grad change, not ULP; rejected). Rows with no saved entries
# (THD padding) are left unmasked (their routing_map is all-False either way
# and their probs are never gathered), avoiding all-(-inf) NaN rows.
#
# The dispatcher's replay-metadata cache (FIX C) then works unchanged: the
# replay's routing_map bitwise-equals the first pass's, so the cached split
# metadata is exact by construction. The router GEMM still runs in the replay
# (logits are needed for the grad-connected probs) — the win is the metadata
# path (all-gather + D2H + event sync), exactly as FIX C intended.
#
# Fallbacks (loud, counted): aux/z-loss enabled (the -inf mask would perturb
# loss-term computations on the logits — off in the ship config); replay with
# no saved map (miss); saved-map shape mismatch. Gate off or no checkpoint
# frame (eval / non-recompute training): byte-identical status-quo path.
#
# Telemetry (WARNING level): one-time gate-state line, one-time armed line,
# one-time first-force line, per-window counters every 312 replay passes
# (~1 step at 78 layers x 4 mb): stashes / forces / misses / shape_mismatches
# / verify_asserts, plus in-flight stash bytes (2 MB per entry; the peak must
# plateau at layers x in-flight microbatches — entries are popped on replay
# fetch and otherwise die with their microbatch carrier, so growth across
# steps would mean a leak).
# ----------------------------------------------------------------------------

_ROUTING_FORCE_GATE_LOGGED = [False]
_ROUTING_FORCE_ARMED_LOGGED = [False]
_ROUTING_FORCE_HIT_LOGGED = [False]
_ROUTING_FORCE_LOSS_DISABLED = [False]
_ROUTING_FORCE_STATS = {
    "stashes": 0,
    "forces": 0,
    "misses": 0,
    "shape_mismatches": 0,
    "verify_asserts": 0,
}
_ROUTING_FORCE_WINDOW = [0, 0]  # [replay passes this window, windows logged]
_ROUTING_FORCE_WINDOW_SIZE = 312
_ROUTING_FORCE_LOG_MAX = 100
_ROUTING_FORCE_STASH_ATTR = "_moe_routing_replay_force"
_ROUTING_FORCE_BYTES = [0]  # in-flight stash bytes (bool = 1 byte/elem)
_ROUTING_FORCE_BYTES_PEAK = [0]


def _routing_force_enabled() -> bool:
    """Lazy env-gate read (per-call; ~1 call/layer-pass).

    Logs the gate state once per process at WARNING level (logger.info from
    megatron.core does not reach trainer_srun.log) so a present-but-inert
    patch is never silent.
    """
    enabled = os.environ.get("BT_MOE_ROUTING_REPLAY_FORCE", "0") == "1"
    if not _ROUTING_FORCE_GATE_LOGGED[0]:
        _ROUTING_FORCE_GATE_LOGGED[0] = True
        if enabled:
            logger.warning(
                "BT_MOE_ROUTING_REPLAY_FORCE=1: recompute-replay routing forced to the "
                "first-pass decision (FIX C-prime) ACTIVE"
            )
        else:
            logger.warning(
                "BT_MOE_ROUTING_REPLAY_FORCE present but DISABLED (env unset or != '1'); "
                "the replay recomputes routing"
            )
    return enabled


def _routing_force_loss_disabled(router) -> bool:
    """True (once, loud) when z-loss/aux-loss make the -inf mask unsafe."""
    if _ROUTING_FORCE_LOSS_DISABLED[0]:
        return True
    if router.config.moe_z_loss_coeff is not None or router.is_aux_loss_enabled():
        _ROUTING_FORCE_LOSS_DISABLED[0] = True
        logger.warning(
            "BT_MOE_ROUTING_REPLAY_FORCE: armed=NO — moe_z_loss_coeff or an aux loss is "
            "enabled; the -inf logit mask would perturb loss-term computations. "
            "Status-quo replay routing."
        )
        return True
    return False


def _routing_force_note_pass():
    """Per-window replay-pass telemetry (WARNING so it reaches trainer logs)."""
    _ROUTING_FORCE_WINDOW[0] += 1
    if _ROUTING_FORCE_WINDOW[0] >= _ROUTING_FORCE_WINDOW_SIZE:
        _ROUTING_FORCE_WINDOW[0] = 0
        _ROUTING_FORCE_WINDOW[1] += 1
        if _ROUTING_FORCE_WINDOW[1] <= _ROUTING_FORCE_LOG_MAX:
            logger.warning(
                "BT_MOE_ROUTING_REPLAY_FORCE window %d (%d replay passes): %s; "
                "in-flight stash %.1f MB (peak %.1f MB)",
                _ROUTING_FORCE_WINDOW[1],
                _ROUTING_FORCE_WINDOW_SIZE,
                dict(_ROUTING_FORCE_STATS),
                _ROUTING_FORCE_BYTES[0] / 1e6,
                _ROUTING_FORCE_BYTES_PEAK[0] / 1e6,
            )


def _routing_force_frame():
    """Innermost FIX C checkpoint-pass frame on this thread, or None.

    Lazy import: megatron.core.recompute imports the transformer layer stack,
    which leads back here — a module-level import would cycle.
    """
    from megatron.core.recompute import current_checkpoint_pass_frame

    return current_checkpoint_pass_frame()


def _routing_force_replay_saved_map(router):
    """Replay: pop this layer's saved first-pass routing_map, or None.

    None means 'route normally' (gate off, loss-terms on, no frame, first
    pass, or a miss/shape fallback). The pop is one-shot: the entry dies here
    or with the microbatch carrier.
    """
    if not _routing_force_enabled() or _routing_force_loss_disabled(router):
        return None
    frame = _routing_force_frame()
    if frame is None or not frame.is_replay:
        return None
    stash = getattr(frame.key_obj, _ROUTING_FORCE_STASH_ATTR, None)
    saved = stash.pop(router.layer_number, None) if stash is not None else None
    _routing_force_note_pass()
    if saved is None:
        _ROUTING_FORCE_STATS["misses"] += 1
        logger.warning(
            "BT_MOE_ROUTING_REPLAY_FORCE: replay found no saved routing_map for layer "
            "%s — routing freely for this pass (status quo)",
            getattr(router, "layer_number", None),
        )
        return None
    _ROUTING_FORCE_BYTES[0] -= saved.numel()
    _ROUTING_FORCE_STATS["forces"] += 1
    if not _ROUTING_FORCE_HIT_LOGGED[0]:
        _ROUTING_FORCE_HIT_LOGGED[0] = True
        # Informational config check (the hybrid preserves router grads
        # regardless — this documents the ship-config assumption).
        trainable = [n for n, p in router.named_parameters() if p.requires_grad]
        logger.warning(
            "BT_MOE_ROUTING_REPLAY_FORCE: first replay force — saved first-pass "
            "routing_map consumed; router params trainable=%s (assumed frozen under "
            "attention-only LoRA; the hybrid preserves router grads either way)",
            trainable if trainable else "none",
        )
    return saved


def _routing_force_stash(router, routing_map):
    """First pass: stash this layer's routing_map on the microbatch carrier."""
    if not _routing_force_enabled() or _ROUTING_FORCE_LOSS_DISABLED[0]:
        return
    frame = _routing_force_frame()
    if frame is None or frame.is_replay:
        return
    stash = getattr(frame.key_obj, _ROUTING_FORCE_STASH_ATTR, None)
    if stash is None:
        stash = {}
        setattr(frame.key_obj, _ROUTING_FORCE_STASH_ATTR, stash)
    stash[router.layer_number] = routing_map
    _ROUTING_FORCE_BYTES[0] += routing_map.numel()
    if _ROUTING_FORCE_BYTES[0] > _ROUTING_FORCE_BYTES_PEAK[0]:
        _ROUTING_FORCE_BYTES_PEAK[0] = _ROUTING_FORCE_BYTES[0]
    _ROUTING_FORCE_STATS["stashes"] += 1
    if not _ROUTING_FORCE_ARMED_LOGGED[0]:
        _ROUTING_FORCE_ARMED_LOGGED[0] = True
        logger.warning(
            "BT_MOE_ROUTING_REPLAY_FORCE: armed — first routing_map stashed on the carrier"
        )


def _routing_force_verify_set_equality(router, recomputed_map, saved_map):
    """Verify mode (BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY=1): per-token, the
    masked-topk output set must equal the saved set EXACTLY. Ties among the
    selected are impossible (unselected are -inf); assert anyway. Rows with no
    saved entries (padding) are exempt (left unmasked)."""
    if os.environ.get("BT_MOE_DISPATCH_REPLAY_CACHE_VERIFY", "0") != "1":
        return
    has_any = saved_map.any(dim=1)
    per_row_equal = (recomputed_map == saved_map).all(dim=1)
    mismatched = (~per_row_equal) & has_any
    _ROUTING_FORCE_STATS["verify_asserts"] += 1
    if mismatched.any():
        rows = mismatched.nonzero().reshape(-1)[:8].tolist()
        raise RuntimeError(
            "BT_MOE_ROUTING_REPLAY_FORCE verify: the masked-topk selection differs from "
            f"the saved routing_map on layer {getattr(router, 'layer_number', None)} "
            f"in {int(mismatched.sum().item())} token rows (first={rows}) — the forcing "
            "is broken; do NOT enable BT_MOE_ROUTING_REPLAY_FORCE"
        )


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
            is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        self.is_mtp_layer = is_mtp_layer
        self.tp_group = pg_collection.tp
        self.cp_group = pg_collection.cp
        self.tp_cp_group = pg_collection.tp_cp
        self.tp_dp_cp_group = pg_collection.tp_dp_cp

        # Initialize the gate weights.
        # TODO: Add support for GPU initialization, which requires updating the golden values.
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        if self.config.add_bias_linear:
            self.bias = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts), dtype=torch.float32)
            )
        else:
            self.bias = None
        # If calculate per token loss, we need to scale up moe aux loss by the number of tokens.
        # So we need to know if the model is configured to calculate per token loss.
        self.calculate_per_token_loss = self.config.calculate_per_token_loss
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the router parameters."""
        if self.config.perform_initialization:
            self.config.init_method(self.weight)
            if self.bias is not None:
                self.config.init_method(self.bias)
        self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)
        setattr(self.weight, 'sequence_parallel', self.config.sequence_parallel)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(dtype=self.config.params_dtype)
            setattr(self.bias, 'sequence_parallel', self.config.sequence_parallel)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        if self.weight.device.type == 'cpu':
            # move weights to GPU
            self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
        if self.bias is not None and self.bias.device.type == 'cpu':
            self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

        # Convert to specified datatype for routing computation if enabled
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mapping.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the router."""
        self.layer_number = layer_number
        if getattr(self, "router_replay", None) is not None:
            self.router_replay.layer_number = layer_number


class TopKRouter(Router):
    """Route each token to the top-k experts.

    The workflow of TopKRouter is as follows:
    (1) Calculate the logits by the router gating network.
    (2) Calculate the routing probabilities and map for top-k selection with score function.
    (3) [Optional] Apply token dropping to top-k expert selection.
    (4) [Optional] Apply the auxiliary load balancing loss for the given scores and routing map.

    Naming convention:
        logits: The output logits by the router gating network.
        scores: The scores after score function used to select the experts and calculate aux loss.
        probs: The topk weights used to combined the experts' outputs.
        routing_map: The masked routing map between tokens and experts.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
            is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
        """
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None
        self.frozen_expert_bias = False

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

        # Initialize global tokens per expert for global aux loss
        if self.get_aux_loss_coeff("global_aux_loss") > 0:
            self.register_buffer(
                'global_tokens_per_expert',
                torch.zeros(
                    self.config.num_moe_experts,
                    dtype=torch.float32,
                    device=torch.cuda.current_device(),
                ),
                persistent=False,
            )
            self.register_buffer(
                'ga_steps',
                torch.tensor(0, dtype=torch.float32, device=torch.cuda.current_device()),
                persistent=False,
            )
        else:
            self.global_tokens_per_expert = None
            self.ga_steps = None

        self.router_replay = None
        if self.config.moe_enable_routing_replay:
            self.router_replay = RouterReplay()

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.topk, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def get_aux_loss_coeff(self, aux_loss_type: str) -> float:
        """Return the aux loss coeff for the given auxiliary loss type.
        If the auxiliary loss type is not found, return 0.0.
        """
        if isinstance(self.routing_type, str):
            if self.routing_type == aux_loss_type:
                return self.config.moe_aux_loss_coeff
        if isinstance(self.routing_type, list):
            try:
                idx = self.routing_type.index(aux_loss_type)
                return self.config.moe_aux_loss_coeff[idx]
            except ValueError:
                return 0.0
        return 0.0

    def is_aux_loss_enabled(self) -> bool:
        """Check if the auxiliary loss is enabled."""
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            if self.get_aux_loss_coeff(aux_loss_type) > 0:
                return True
        return False

    def _apply_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs

        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=self.tp_cp_group,
                topk=self.topk,
                with_padding_mask=with_padding_mask,
            )
        )

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=global_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            aux_loss_coeff,
            aux_loss,
            "load_balancing_loss",
            self.tp_cp_group,
            valid_token_count=local_num_tokens,
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
        with_padding_mask: bool = False,
    ):
        """Apply the sequence-level auxiliary loss for the given scores and routing map.

        To calculate the sequence-level aux loss, we reshape the batch_size dimension to
        experts dimension. The resulted loss by switch_load_balancing_loss_func is equal
        to the sum of aux loss for each sequence in the batch. And then we divide the aux
        loss by the batch size to get averaged aux loss.
        """
        seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
        if seq_aux_loss_coeff == 0:
            return probs

        scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
        routing_map = routing_map.reshape(seq_length, -1)

        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=self.tp_cp_group,
                with_padding_mask=with_padding_mask,
                topk=self.topk * bsz,
            )
        )

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=global_tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_fusion,
            )
            / bsz
        )

        probs = self.attach_and_log_load_balancing_loss(
            probs,
            seq_aux_loss_coeff,
            aux_loss,
            "seq_load_balancing_loss",
            self.tp_cp_group,
            valid_token_count=local_num_tokens,
        )
        return probs

    def _apply_global_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        with_padding_mask: bool = False,
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        # Use unified function to compute tokens_per_expert and num_tokens
        global_tokens_per_expert, local_num_tokens, total_num_tokens = (
            get_tokens_per_expert_and_token_count(
                routing_map=routing_map,
                reduce_group=self.tp_dp_cp_group,
                with_padding_mask=with_padding_mask,
                topk=self.topk,
            )
        )

        self.global_tokens_per_expert += global_tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        global_aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=averated_tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=global_aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs,
            global_aux_loss_coeff,
            global_aux_loss,
            "global_load_balancing_loss",
            self.tp_dp_cp_group,
            needs_dp_avg=False,
            valid_token_count=local_num_tokens,
        )
        return probs

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        reduce_group: torch.distributed.ProcessGroup,
        needs_dp_avg: bool = True,
        valid_token_count: Optional[Union[int, torch.Tensor]] = None,
    ):
        """Attach aux loss function to activation and add to logging.

        Args:
            activation (torch.Tensor): Activation tensor to attach the aux loss to.
            aux_loss_coeff (float): Coefficient for the aux loss.
            aux_loss (torch.Tensor): Computed aux loss.
            aux_loss_name (str): Name of the aux loss for logging.
            reduce_group (torch.distributed.ProcessGroup): Process group for reduction.
            needs_dp_avg (bool): Whether to average this metric across DP ranks after reduce_group.
            valid_token_count (int or torch.Tensor, optional): Number of valid tokens excluding
                padding tokens. Can be a Python int or a torch.Tensor (typically 0-d tensor).
                If None, uses activation.shape[0]. Defaults to None.
        """
        # When using repeated MTP layers, the loss is counted "mtp_num_layers" times.
        # To avoid accumulating the load balancing loss multiple times, we scale it by
        # 1/mtp_num_layers so the total loss is correct.
        if (
            self.is_mtp_layer
            and self.config.mtp_use_repeated_layer
            and self.config.mtp_num_layers is not None
        ):
            aux_loss = aux_loss / self.config.mtp_num_layers

        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers

        if self.is_mtp_layer:
            layer_number = self.layer_number + self.config.num_layers
        else:
            layer_number = self.layer_number

        get_moe_metrics_tracker().record(
            aux_loss_name,
            aux_loss / aux_loss_coeff,
            layer_number,
            num_layers,
            reduce_group=reduce_group,
            needs_dp_avg=needs_dp_avg,
        )
        if self.calculate_per_token_loss:
            # Target final scaling on aux_loss gradients: 1 / (num_micro_batches * dp_size),
            # matching the !calculate_per_token_loss path.
            #
            # --calculate-per-token-loss already divides every parameter gradient by
            # total_global_tokens (the global non-padded token count summed in
            # finalize_model_grads). The router's `num_local_tokens` (= activation.shape[0])
            # is sequence-parallel sharded — the router weight is marked
            # `sequence_parallel=True` in Router.reset_parameters (see
            # `setattr(self.weight, 'sequence_parallel', ...)` above), so each TP rank
            # computes a partial gradient on the router weight from its local sequence
            # shard, and `_allreduce_non_tensor_model_parallel_grads` SUMS those partial
            # gradients across the TP group. Re-expressing total_global_tokens in terms of the
            # router's `num_local_tokens`:
            #     total_global_tokens
            #         = num_micro_batches * dp_cp_size * loss_func_local_tokens
            #         = num_micro_batches * dp_cp_size * tp_size * num_local_tokens
            #         = num_micro_batches * dp_size * (num_local_tokens * tp_cp_group.size())
            # (using loss_func_local_tokens = tp_size * num_local_tokens, then regrouping
            # dp_cp_size * tp_size as dp_size * tp_cp_group.size()).
            #
            # So pre-multiplying aux_loss by num_local_tokens * tp_cp_group.size() cancels
            # that same factor in total_global_tokens above, leaving 1 / (num_micro_batches *
            # dp_size) as the effective scaling on the aux_loss gradient — the target.
            # Use valid_token_count (excluding padding) if provided, otherwise use total tokens.
            num_local_tokens = (
                valid_token_count if valid_token_count is not None else activation.shape[0]
            )
            activation = MoEAuxLossAutoScaler.apply(
                activation, aux_loss * num_local_tokens * self.tp_cp_group.size()
            )
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits, padding_mask: Optional[torch.Tensor] = None):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape in [num_tokens]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff, padding_mask=padding_mask)
            if self.calculate_per_token_loss:
                # Same derivation as in attach_and_log_load_balancing_loss:
                #   - Target final scaling on z_loss gradients: 1 / (num_micro_batches * dp_size).
                #   - In terms of the router's `num_local_tokens`, the total_global_tokens
                #     divisor that finalize_model_grads applies factors as
                #         num_micro_batches * dp_size * (num_local_tokens * tp_cp_group.size()).
                #   - Pre-multiplying z_loss by num_local_tokens * tp_cp_group.size() cancels
                #     that same factor in total_global_tokens, leaving
                #     1 / (num_micro_batches * dp_size) as the effective scaling — the target.
                # The /tp_cp_group.size() on moe_z_loss_coeff above is a separate forward-side
                # correction: z_loss is computed independently on each TP+CP rank's local
                # logits and must be averaged across TP+CP rather than summed.
                # Count valid tokens: sum of inverted mask (False -> True = valid)
                num_local_tokens = (
                    (~padding_mask).sum() if padding_mask is not None else logits.shape[0]
                )
                logits = MoEAuxLossAutoScaler.apply(
                    logits, z_loss * num_local_tokens * self.tp_cp_group.size()
                )
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            # When using repeated MTP layers, the same MTP layer is called mtp_num_layers times.
            # To avoid accumulating the z_loss multiple times, we scale it by 1/mtp_num_layers
            # so the total loss is correct.
            if (
                self.is_mtp_layer
                and self.config.mtp_use_repeated_layer
                and self.config.mtp_num_layers is not None
            ):
                z_loss = z_loss / self.config.mtp_num_layers

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers

            if self.is_mtp_layer:
                layer_number = self.layer_number + self.config.num_layers
            else:
                layer_number = self.layer_number

            get_moe_metrics_tracker().record(
                "z_loss", z_loss / moe_z_loss_coeff, layer_number, num_layers
            )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, dtype=input.dtype, device=input.device),
                    torch.tensor(1.0 + eps, dtype=input.dtype, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    @jit_fuser
    def _apply_expert_bias(
        self, routing_map: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ):
        """
        Update expert bias and tokens_per_expert
        Prevent extra local tokens accumulation on evaluation or activation recomputation
        """
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                if padding_mask is not None:
                    routing_map = routing_map & (~padding_mask)
                self.local_tokens_per_expert += routing_map.sum(dim=0)

    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Flatten padding_mask to [num_tokens] if provided
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits, padding_mask=padding_mask)

        # Calculate probs and routing_map for token dispatching
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits,
                self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                num_groups=self.config.moe_router_num_groups,
                group_topk=self.config.moe_router_group_topk,
                scaling_factor=self.config.moe_router_topk_scaling_factor,
                score_function=self.score_function,
                expert_bias=self.expert_bias,
                fused=self.config.moe_router_fusion,
                router_replay=self.router_replay,
            )

        # Apply token dropping to probs and routing_map.
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Apply each aux loss type and attach aux loss autograd function to probs
        if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
            # Calculate scores and routing_map for aux loss
            routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
                logits,
                self.topk,
                self.score_function,
                fused=self.config.moe_router_fusion,
                padding_mask=padding_mask,
            )
            probs = self._apply_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=padding_mask is not None,
            )
            probs = self._apply_seq_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                seq_length,
                bsz,
                with_padding_mask=padding_mask is not None,
            )
            probs = self._apply_global_aux_loss(
                probs,
                scores_for_aux_loss,
                routing_map_for_aux_loss,
                with_padding_mask=padding_mask is not None,
            )

        # Optionally apply expert bias
        self._apply_expert_bias(routing_map, padding_mask=padding_mask)

        return probs, routing_map

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
            padding_mask (torch.Tensor, optional): Boolean mask indicating non-padding tokens.
                                                   Shape [seq_length, bsz]. True for valid tokens,
                                                   False for padding tokens. Defaults to None.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        if self.config.moe_router_force_biased is not None:
            # Apply biased logits with shared random bias across all ranks
            logits = apply_biased_logits(
                logits, self.config.moe_router_force_biased, self.layer_number
            )

        # BT_MOE_ROUTING_REPLAY_FORCE (FIX C-prime): in the recompute replay,
        # force the top-k selection to the first pass's saved routing_map by
        # masking unsaved logits with -inf (the probs recompute grad-connected
        # from the replay's logits — the router-path activation grad is
        # preserved). Rows with no saved entries (padding) stay unmasked so
        # their score function never sees an all-(-inf) row.
        saved_map = _routing_force_replay_saved_map(self)
        if saved_map is not None:
            # routing() flattens logits to [-1, num_experts] internally; the
            # saved map is [tokens, experts] — compare on numel/last-dim and
            # reshape for the mask.
            if saved_map.numel() != logits.numel() or saved_map.size(-1) != logits.size(-1):
                _ROUTING_FORCE_STATS["shape_mismatches"] += 1
                logger.warning(
                    "BT_MOE_ROUTING_REPLAY_FORCE: saved routing_map shape %s incompatible "
                    "with logits shape %s on layer %s — routing freely for this pass "
                    "(status quo)",
                    tuple(saved_map.shape),
                    tuple(logits.shape),
                    getattr(self, "layer_number", None),
                )
                saved_map = None
            else:
                # Reshape for the mask only — the returned/stashed map keeps
                # its [tokens, experts] form (routing() flattens logits).
                mask = (~saved_map).reshape(logits.shape)
                has_any = saved_map.any(dim=-1).reshape(
                    logits.shape[:-1] + (1,)
                )  # per token row
                logits = logits.masked_fill(mask & has_any, float("-inf"))

        probs, routing_map = self.routing(logits, padding_mask=padding_mask)

        if saved_map is not None:
            _routing_force_verify_set_equality(self, routing_map, saved_map)
            # Return the saved map itself: the dispatch decision is bitwise the
            # first pass's by construction (the recomputed map must equal it —
            # asserted in verify mode).
            routing_map = saved_map
        else:
            # First pass (or an unforced pass): stash this layer's decision.
            _routing_force_stash(self, routing_map)

        return probs, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)


class InferenceTopKRouter(TopKRouter):
    """Inference-only top-k router that strips out training-specific overhead.

    A stripped-down version of TopKRouter that skips z-loss, auxiliary load
    balancing losses, token dropping, and expert bias updates. The _forward()
    method is @torch.compile()'d and returns dense [num_tokens, topk] tensors
    instead of sparse [num_tokens, num_experts] for compatibility with FlashInfer.

    Falls back to the parent TopKRouter.forward() for training or
    non-CUDA-graphed inference iterations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        """Initialize the specialized inference top-k router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        # Enforce constraints before calling super().__init__
        assert config.moe_router_num_groups is None, (
            f"InferenceTopKRouter requires moe_router_num_groups=None, "
            f"got {config.moe_router_num_groups}"
        )
        assert config.moe_router_score_function in ["sigmoid", "softmax"], (
            f"InferenceTopKRouter requires moe_router_score_function in "
            f"['sigmoid', 'softmax'], got '{config.moe_router_score_function}'"
        )

        super().__init__(config=config, pg_collection=pg_collection)

    @staticmethod
    @torch.compile
    def _compiled_topk_routing(
        logits,
        topk,
        use_pre_softmax,
        num_groups,
        group_topk,
        scaling_factor,
        score_function,
        expert_bias,
        fused,
        router_replay,
        dense_output,
    ):
        return topk_routing_with_score_function(
            logits,
            topk,
            use_pre_softmax=use_pre_softmax,
            num_groups=num_groups,
            group_topk=group_topk,
            scaling_factor=scaling_factor,
            score_function=score_function,
            expert_bias=expert_bias,
            fused=fused,
            router_replay=router_replay,
            dense_output=dense_output,
        )

    def _forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        logits = self.gating(input).squeeze(1)  # [num_tokens, num_experts]

        probs, top_indices = self._compiled_topk_routing(
            logits,
            self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            fused=self.config.moe_router_fusion,
            router_replay=self.router_replay,
            dense_output=True,
        )
        return probs.squeeze(1), top_indices.squeeze(1)

    def forward(self, input: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        """Simplified forward pass for inference - returns dense tensors only.

        Args:
            input (torch.Tensor): Input tensor of shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Not used in inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - probs: Normalized routing probabilities [num_tokens, topk]
                - top_indices: Selected expert indices [num_tokens, topk]
        """

        if not InferenceMode.is_active():
            return super().forward(input, padding_mask)

        return self._forward(input, padding_mask)
