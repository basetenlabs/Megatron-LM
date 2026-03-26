# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Optional

import torch

from megatron.core.tensor_parallel import reduce_from_tensor_model_parallel_region
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    apply_random_logits,
    apply_router_token_dropping,
    compute_routing_scores_for_aux_loss,
    router_gating_linear,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
    z_loss_func,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config)
        self.config = config
        self.num_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
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
        self, config: TransformerConfig, pg_collection: Optional[ProcessGroupCollection] = None
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
            pg_collection (ProcessGroupCollection, optional): Process groups for MoE operations.
        """
        super().__init__(config=config, pg_collection=pg_collection)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.score_function = self.config.moe_router_score_function
        self.input_jitter = None

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
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the auxiliary loss for the given scores and routing map."""
        aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
        if aux_loss_coeff == 0:
            return probs
        tokens_per_expert = routing_map.sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )
        num_tokens = routing_map.shape[0]
        total_num_tokens = num_tokens * self.tp_cp_group.size()

        aux_loss = switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, aux_loss_coeff, aux_loss, "load_balancing_loss", self.tp_cp_group
        )
        return probs

    def _apply_seq_aux_loss(
        self,
        probs: torch.Tensor,
        scores_for_aux_loss: torch.Tensor,
        routing_map: torch.Tensor,
        seq_length: int,
        bsz: int,
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
        tokens_per_expert = routing_map.reshape(seq_length, -1).sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_cp_group
        )

        total_num_tokens = seq_length * self.tp_cp_group.size()

        aux_loss = (
            switch_load_balancing_loss_func(
                probs=scores_for_aux_loss,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens,
                topk=self.topk,
                num_experts=self.config.num_moe_experts,
                moe_aux_loss_coeff=seq_aux_loss_coeff,
                fused=self.config.moe_router_fusion,
            )
            / bsz
        )
        probs = self.attach_and_log_load_balancing_loss(
            probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss", self.tp_cp_group
        )
        return probs

    def _apply_global_aux_loss(
        self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor, routing_map: torch.Tensor
    ):
        """Apply the global auxiliary loss for the given scores and routing map."""
        global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
        if global_aux_loss_coeff == 0:
            return probs

        tokens_per_expert = routing_map.sum(dim=0)
        tokens_per_expert = reduce_from_tensor_model_parallel_region(
            tokens_per_expert, self.tp_dp_cp_group
        )

        self.global_tokens_per_expert += tokens_per_expert
        self.ga_steps += 1
        averated_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

        num_tokens = scores_for_aux_loss.shape[0]
        total_num_tokens = num_tokens * self.tp_dp_cp_group.size()

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
        )
        return probs

    def attach_and_log_load_balancing_loss(
        self,
        activation: torch.Tensor,
        aux_loss_coeff: float,
        aux_loss: torch.Tensor,
        aux_loss_name: str,
        reduce_group: torch.distributed.ProcessGroup,
    ):
        """Attach aux loss function to activation and add to logging."""
        # TODO (zijiey): fix the per_layer_logging for MTP, currently it will incorrectly
        # add the aux loss logging value to other layer's since it is difficult to get the
        # correct layer_number for MTP. It does not affect the correctness of the calculation
        # results and the reduced load_balancing_loss logging value.
        num_layers = self.config.num_layers
        if self.config.mtp_num_layers is not None:
            num_layers += self.config.mtp_num_layers
        save_to_aux_losses_tracker(
            aux_loss_name,
            aux_loss / aux_loss_coeff,
            self.layer_number,
            num_layers,
            reduce_group=reduce_group,
        )
        if self.calculate_per_token_loss:
            # Scale the aux_loss by the number of tokens.
            # The expected final scaling for aux_loss gradients is 1/(num_micro_batches * dp_size).
            # After commit 02648000, Megatron started using the number of total tokens to scale
            # gradients under the argument of calculate_per_token_loss,
            # which scales both the main_loss gradient and aux_loss gradient by
            # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads function.
            # To correct this scaling, we need to scale the aux_loss by num_local_tokens here.
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss * activation.shape[0])
        else:
            activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None and self.training and torch.is_grad_enabled():
            # Skip Z loss calculations when using torch.no_grad() or checkpointing.
            moe_z_loss_coeff = self.config.moe_z_loss_coeff / self.tp_cp_group.size()
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            scale_up = 1.0
            if self.calculate_per_token_loss:
                # The expected final scaling for z_loss gradients is
                # 1/(num_micro_batches * dp_size).
                # After commit 02648000, Megatron started using the number of total tokens
                # to scale gradients under the argument of calculate_per_token_loss,
                # which scales both the main_loss gradient and z_loss gradient by
                # 1/(num_local_tokens * dp_size * num_micro_batches) in finalize_model_grads().
                # To correct this scaling, we need to scale the z_loss by num_local_tokens here.
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss * logits.shape[0])
            else:
                logits = MoEAuxLossAutoScaler.apply(logits, z_loss)

            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            save_to_aux_losses_tracker(
                "z_loss", z_loss / moe_z_loss_coeff, self.layer_number, num_layers
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

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): The probabilities of token to experts assignment.
            routing_map (torch.Tensor): The mapping of token to experts assignment,
                with shape [num_tokens, num_experts].
        """
        seq_length, bsz = logits.shape[:2]
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

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
                logits, self.topk, self.score_function, fused=self.config.moe_router_fusion
            )
            probs = self._apply_aux_loss(probs, scores_for_aux_loss, routing_map_for_aux_loss)
            probs = self._apply_seq_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss, seq_length, bsz
            )
            probs = self._apply_global_aux_loss(
                probs, scores_for_aux_loss, routing_map_for_aux_loss
            )

        # Update expert bias and tokens_per_expert
        # Prevent extra local tokens accumulation on evaluation or activation recomputation
        if self.enable_expert_bias and torch.is_grad_enabled():
            with torch.no_grad():
                self.local_tokens_per_expert += routing_map.sum(dim=0)

        return probs, routing_map

    def reset_global_aux_loss_tracker(self):
        """Reset the global aux loss tracker."""
        if self.global_tokens_per_expert is not None:
            self.global_tokens_per_expert.zero_()
            self.ga_steps.zero_()

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self._maintain_float32_expert_bias()

        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)

        if self.config.moe_router_force_load_balancing:
            # Apply force load balancing with random logits for benchmark
            logits = apply_random_logits(logits)

        # --- Router debug logging ---
        import os
        if (
            os.getenv("WHETSTONE_MOE_ALLOC_DEBUG", "0") == "1"
            and not getattr(self, '_router_debug_done', False)
        ):
            self._router_debug_done = True
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            debug_ranks_env = os.getenv("WHETSTONE_MOE_ALLOC_DEBUG_RANKS", "").strip()
            debug_ranks = {int(r) for r in debug_ranks_env.split(",") if r.strip()} if debug_ranks_env else None
            if debug_ranks is None or rank in debug_ranks:
                w = self.weight.data.float()
                inp = input.float()
                lg = logits.float()
                bias = self.expert_bias if self.expert_bias is not None else None

                def _write(msg):
                    debug_file = os.getenv("WHETSTONE_MOE_ALLOC_DEBUG_FILE", "").strip()
                    if debug_file:
                        with open(debug_file, "a") as f:
                            f.write(msg + "\n")
                    print(f"[WHETSTONE_MOE_ALLOC_DEBUG] {msg}", flush=True)

                # Gate weight: first 5 experts, first 10 hidden dims
                _write(
                    f"ROUTER_WEIGHT rank={rank} layer={self.layer_number} "
                    f"shape={tuple(w.shape)} dtype={self.weight.dtype} "
                    f"weight[0,:10]={w[0,:10].tolist()} "
                    f"weight[1,:10]={w[1,:10].tolist()} "
                    f"weight[383,:10]={w[383,:10].tolist() if w.shape[0] > 383 else 'N/A'} "
                    f"per_expert_norms={w.norm(dim=1).tolist()}"
                )

                # Expert bias values
                if bias is not None:
                    _write(
                        f"ROUTER_BIAS rank={rank} layer={self.layer_number} "
                        f"shape={tuple(bias.shape)} dtype={bias.dtype} "
                        f"values={bias.float().tolist()}"
                    )

                # Input hidden states: first 3 tokens, first 10 dims
                inp_flat = inp.reshape(-1, inp.shape[-1])
                _write(
                    f"ROUTER_INPUT rank={rank} layer={self.layer_number} "
                    f"shape={tuple(input.shape)} num_tokens={inp_flat.shape[0]} "
                    f"token0[:10]={inp_flat[0,:10].tolist()} "
                    f"token1[:10]={inp_flat[1,:10].tolist() if inp_flat.shape[0] > 1 else 'N/A'} "
                    f"token_last[:10]={inp_flat[-1,:10].tolist()} "
                    f"per_token_norms_first20={inp_flat[:20].norm(dim=1).tolist()} "
                    f"has_nan={torch.isnan(inp_flat).any().item()} "
                    f"has_inf={torch.isinf(inp_flat).any().item()}"
                )

                # Logits: first 3 tokens, all 384 expert logits (this is what determines routing)
                lg_flat = lg.reshape(-1, lg.shape[-1])
                _write(
                    f"ROUTER_LOGITS rank={rank} layer={self.layer_number} "
                    f"shape={tuple(logits.shape)} "
                    f"token0_logits={lg_flat[0].tolist()} "
                    f"token1_logits={lg_flat[1].tolist() if lg_flat.shape[0] > 1 else 'N/A'} "
                    f"token_last_logits={lg_flat[-1].tolist()}"
                )

                # Summary stats
                expert_norms = w.norm(dim=1)
                token_logit_std = lg_flat.std(dim=1)
                _write(
                    f"ROUTER_SUMMARY rank={rank} layer={self.layer_number} "
                    f"weight_norm={w.norm().item():.2f} "
                    f"expert_norms_min={expert_norms.min().item():.4f} max={expert_norms.max().item():.4f} "
                    f"input_norm_mean={inp_flat.norm(dim=1).mean().item():.2f} "
                    f"logits_min={lg_flat.min().item():.4f} max={lg_flat.max().item():.4f} "
                    f"mean={lg_flat.mean().item():.4f} std={lg_flat.std().item():.4f} "
                    f"token_logit_diversity_mean={token_logit_std.mean().item():.4f} "
                    f"token_logit_diversity_min={token_logit_std.min().item():.4f}"
                )

        probs, routing_map = self.routing(logits)

        return probs, routing_map

    def _load_from_state_dict(self, *args, **kwargs):
        """Load the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before loading
        return super()._load_from_state_dict(*args, **kwargs)

    def _save_to_state_dict(self, *args, **kwargs):
        """Save the state dict of the router."""
        self._maintain_float32_expert_bias()  # switch to float32 before saving
        return super()._save_to_state_dict(*args, **kwargs)
