# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        assert ep_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % ep_size == 0
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)
        self.token_dispatcher.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        residual = hidden_states
        probs, routing_map = self.router(hidden_states)
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    def routed_experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        assert mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}"
        output = self.token_dispatcher.combine_preprocess(expert_output)

        return output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            # --- MoE layer debug: track hidden state norms ---
            import os
            _moe_debug = (
                os.getenv("WHETSTONE_MOE_ALLOC_DEBUG", "0") == "1"
                and not getattr(self, '_moe_layer_debug_done', False)
            )
            if _moe_debug:
                self._moe_layer_debug_done = True
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                debug_ranks_env = os.getenv("WHETSTONE_MOE_ALLOC_DEBUG_RANKS", "").strip()
                debug_ranks = {int(r) for r in debug_ranks_env.split(",") if r.strip()} if debug_ranks_env else None
                if debug_ranks is None or rank in debug_ranks:
                    def _write(msg):
                        debug_file = os.getenv("WHETSTONE_MOE_ALLOC_DEBUG_FILE", "").strip()
                        if debug_file:
                            with open(debug_file, "a") as f:
                                f.write(msg + "\n")
                        print(f"[WHETSTONE_MOE_ALLOC_DEBUG] {msg}", flush=True)

                    hs_flat = hidden_states.float().reshape(-1, hidden_states.shape[-1])
                    in_norms = hs_flat.norm(dim=1)
                    has_nan = torch.isnan(hs_flat).any().item()
                    has_inf = torch.isinf(hs_flat).any().item()
                    # Pairwise cosine similarity of first 20 tokens
                    n_sample = min(20, hs_flat.shape[0])
                    sample = hs_flat[:n_sample]
                    sample_normed = sample / (sample.norm(dim=1, keepdim=True) + 1e-8)
                    cos_matrix = sample_normed @ sample_normed.T
                    mean_cos = (cos_matrix.sum() - n_sample) / (n_sample * (n_sample - 1)) if n_sample > 1 else 1.0

                    _write(
                        f"MOE_HIDDEN rank={rank} layer={self.layer_number} phase=input "
                        f"shape={tuple(hidden_states.shape)} "
                        f"norm_mean={in_norms.mean().item():.2f} norm_std={in_norms.std().item():.2f} "
                        f"has_nan={has_nan} has_inf={has_inf} "
                        f"mean_pairwise_cos={mean_cos:.4f} "
                        f"token0[:20]={hs_flat[0,:20].tolist()} "
                        f"token1[:20]={hs_flat[1,:20].tolist() if hs_flat.shape[0] > 1 else 'N/A'} "
                        f"token_last[:20]={hs_flat[-1,:20].tolist()} "
                        f"per_token_norms_first20={in_norms[:20].tolist()}"
                    )

            shared_expert_output = self.shared_experts_compute(hidden_states)

            # --- Log shared expert output and expert weights ---
            if _moe_debug and (debug_ranks is None or rank in debug_ranks):
                if shared_expert_output is not None:
                    se = shared_expert_output.float().reshape(-1, shared_expert_output.shape[-1])
                    se_norms = se.norm(dim=1)
                    _write(
                        f"MOE_SHARED_OUTPUT rank={rank} layer={self.layer_number} "
                        f"shape={tuple(shared_expert_output.shape)} "
                        f"norm_mean={se_norms.mean().item():.4f} norm_max={se_norms.max().item():.4f} "
                        f"token0[:20]={se[0,:20].tolist()} "
                        f"per_token_norms={se_norms[:20].tolist()}"
                    )
                else:
                    _write(f"MOE_SHARED_OUTPUT rank={rank} layer={self.layer_number} shared_expert_output=None")

                # Expert weights — enumerate all params to find the actual names
                experts = self.experts
                param_info = []
                all_norms = {}
                sample_values = {}
                for name, param in experts.named_parameters():
                    pf = param.data.float()
                    all_norms[name] = pf.norm().item()
                    if len(sample_values) < 6:  # first 6 params
                        sample_values[name] = {
                            'shape': tuple(param.shape),
                            'dtype': str(param.dtype),
                            'norm': pf.norm().item(),
                            'values': pf.reshape(-1)[:10].tolist(),
                        }
                # Summary: group norms by param type
                fc1_norms = {k: v for k, v in all_norms.items() if 'fc1' in k or 'weight1' in k}
                fc2_norms = {k: v for k, v in all_norms.items() if 'fc2' in k or 'weight2' in k}
                _write(
                    f"MOE_EXPERT_WEIGHTS rank={rank} layer={self.layer_number} "
                    f"num_local_experts={self.num_local_experts} "
                    f"local_expert_indices=[{self.local_expert_indices[0]}..{self.local_expert_indices[-1]}] "
                    f"total_params={len(all_norms)} "
                    f"fc1_norms={fc1_norms} "
                    f"fc2_norms={fc2_norms} "
                    f"sample_params={sample_values}"
                )

            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)

            # --- Log routed expert output BEFORE adding shared expert ---
            if _moe_debug and (debug_ranks is None or rank in debug_ranks):
                # output here is after token_combine + combine_postprocess but BEFORE combine adds shared
                # Actually output is from routed_experts_compute which returns after combine_preprocess
                # We need to look at the combine method... output at this point is PRE-combine (before a2a back)
                # Let me log what combine() will produce
                pass

            output = self.combine(output, shared_expert_output)

            # --- Log routed+shared combined output ---
            if _moe_debug and (debug_ranks is None or rank in debug_ranks):
                out_flat = output.float().reshape(-1, output.shape[-1])
                out_norms = out_flat.norm(dim=1)
                _write(
                    f"MOE_COMBINED_OUTPUT rank={rank} layer={self.layer_number} "
                    f"shape={tuple(output.shape)} "
                    f"norm_mean={out_norms.mean().item():.4f} norm_max={out_norms.max().item():.4f} "
                    f"token0[:20]={out_flat[0,:20].tolist()} "
                    f"token1[:20]={out_flat[1,:20].tolist() if out_flat.shape[0] > 1 else 'N/A'} "
                    f"per_token_norms={out_norms[:20].tolist()}"
                )

            return output, mlp_bias

        if self.moe_layer_recompute:
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
