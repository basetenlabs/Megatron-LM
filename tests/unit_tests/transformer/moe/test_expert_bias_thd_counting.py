# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the aux-loss-free balancing token-count sensor under THD packing.

Covers three input regimes for ``Router._apply_expert_bias``:
  (a) a boot-warmup-shaped synthetic datum (single short doc, all-valid mask),
  (b) a padded THD window (multiple docs, each with a masked pad tail),
  (c) a clean fully-packed window (no padding; ``None`` mask and all-False mask
      must count identically, bit-for-bit).

Also covers:
  * ``TopKRouter._maintain_float32_expert_bias`` must keep the count sensor
    (``local_tokens_per_expert``) in float32 even after a bf16/fp16 module cast
    (e.g. Float16Module). A bf16 count buffer silently loses increments once
    counts grow past the bf16 ulp (e.g. ``512 + 1 == 512``), corrupting the
    counts that drive the expert-bias update.
  * The count-sanity gate in ``get_updated_expert_bias``: non-finite or
    negative token counts must skip the bias tick (bias returned unchanged)
    with one loud warning, instead of feeding garbage into ``torch.sign``.

All tests in this module are CPU-runnable (single-rank gloo for the gate
tests); no CUDA or multi-rank setup is required.
"""

import logging
import os

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
from megatron.core.transformer.moe.router import TopKRouter

NUM_EXPERTS = 64
TOPK = 8


def _ensure_single_rank_dist():
    """Initialize a single-rank gloo process group if none exists (CPU-safe)."""
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29981")
    torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)


class _RouterStub:
    """Minimal stand-in exposing exactly what the sensor methods touch."""

    def __init__(self, num_experts=NUM_EXPERTS, dtype=torch.float32):
        self.enable_expert_bias = True
        self.local_tokens_per_expert = torch.zeros(num_experts, dtype=dtype)
        self.expert_bias = torch.zeros(num_experts, dtype=dtype)


def _random_routing_map(num_tokens, num_experts=NUM_EXPERTS, topk=TOPK, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_tokens, num_experts, generator=g)
    idx = torch.topk(logits, k=topk, dim=1).indices
    return torch.zeros(num_tokens, num_experts, dtype=torch.int).scatter(1, idx, 1).bool()


def _thd_padding_mask(doc_lens, pad_multiple):
    """Per-doc pad-tail mask exactly like the trainers THD packer (True = pad)."""
    flags = []
    for length in doc_lens:
        padded = -(-length // pad_multiple) * pad_multiple
        flags += [False] * length + [True] * (padded - length)
    return torch.tensor(flags, dtype=torch.bool)


def _naive_counts(routing_map, padding_mask):
    """Ground-truth per-expert counts computed token by token (no broadcasting)."""
    flat_mask = None if padding_mask is None else padding_mask.reshape(-1)
    counts = torch.zeros(routing_map.shape[1], dtype=torch.float32)
    for t in range(routing_map.shape[0]):
        if flat_mask is not None and bool(flat_mask[t]):
            continue
        counts += routing_map[t].float()
    return counts


def _apply_sensor(stub, routing_map, padding_mask):
    """Call the real (jit_fuser-decorated) sensor exactly as routing() does."""
    with torch.enable_grad():
        TopKRouter._apply_expert_bias(stub, routing_map, padding_mask=padding_mask)


class TestApplyExpertBiasThdCounting:
    """Exact-count assertions for the three THD input regimes."""

    def test_regime_a_warmup_shaped_datum(self):
        # Boot warmup: one synthetic 64-token doc, pad_multiple=2 -> no pad tail.
        routing_map = _random_routing_map(64, seed=1)
        padding_mask = _thd_padding_mask([64], pad_multiple=2)
        assert padding_mask.shape[0] == 64 and not padding_mask.any()
        stub = _RouterStub()
        _apply_sensor(stub, routing_map, padding_mask)
        assert torch.equal(stub.local_tokens_per_expert, _naive_counts(routing_map, padding_mask))
        assert stub.local_tokens_per_expert.sum().item() == 64 * TOPK

    @pytest.mark.parametrize("mask_shape", ["flat", "seq_bsz"])
    def test_regime_b_padded_thd_window(self, mask_shape):
        # 2048-token-window-like packing: several docs with masked pad tails.
        doc_lens = [1000, 700, 260, 37]
        padding_mask = _thd_padding_mask(doc_lens, pad_multiple=64)
        num_tokens = padding_mask.shape[0]
        routing_map = _random_routing_map(num_tokens, seed=2)
        if mask_shape == "seq_bsz":
            padding_mask = padding_mask.unsqueeze(-1)  # [S, 1] as moe_layer passes
        stub = _RouterStub()
        _apply_sensor(stub, routing_map, padding_mask)
        expected = _naive_counts(routing_map, padding_mask)
        assert torch.equal(stub.local_tokens_per_expert, expected)
        # Masked (pad) tokens must not count.
        assert stub.local_tokens_per_expert.sum().item() == sum(doc_lens) * TOPK

    def test_regime_c_clean_packed_window(self):
        # Fully-packed window: mask=None and all-False mask must match bit-for-bit.
        routing_map = _random_routing_map(2048, seed=3)
        stub_none = _RouterStub()
        _apply_sensor(stub_none, routing_map, None)
        stub_false = _RouterStub()
        _apply_sensor(stub_false, routing_map, torch.zeros(2048, dtype=torch.bool))
        assert torch.equal(
            stub_none.local_tokens_per_expert, stub_false.local_tokens_per_expert
        )
        assert torch.equal(
            stub_none.local_tokens_per_expert, _naive_counts(routing_map, None)
        )

    def test_accumulates_across_microbatches(self):
        stub = _RouterStub()
        rm1 = _random_routing_map(128, seed=4)
        rm2 = _random_routing_map(256, seed=5)
        mask2 = _thd_padding_mask([200], pad_multiple=256)
        _apply_sensor(stub, rm1, None)
        _apply_sensor(stub, rm2, mask2)
        expected = _naive_counts(rm1, None) + _naive_counts(rm2, mask2)
        assert torch.equal(stub.local_tokens_per_expert, expected)


class TestCountBufferDtypeMaintenance:
    """The count sensor must survive a bf16/fp16 module cast in float32."""

    def test_maintain_float32_restores_count_buffer_dtype(self):
        stub = _RouterStub(dtype=torch.bfloat16)
        TopKRouter._maintain_float32_expert_bias(stub)
        assert stub.expert_bias.dtype == torch.float32
        # FAILS on the unfixed code: local_tokens_per_expert was left bf16.
        assert stub.local_tokens_per_expert.dtype == torch.float32

    def test_counts_do_not_saturate_after_module_cast(self):
        # bf16 spacing at 512 is 4: a bf16 count buffer silently drops small
        # increments. After maintenance the buffer must be fp32 and exact.
        stub = _RouterStub(dtype=torch.bfloat16)
        stub.local_tokens_per_expert += 512.0
        TopKRouter._maintain_float32_expert_bias(stub)
        one_token_map = torch.zeros(1, NUM_EXPERTS, dtype=torch.bool)
        one_token_map[0, :] = True
        _apply_sensor(stub, one_token_map, None)
        # FAILS on the unfixed code: bf16 512 + 1 rounds back to 512.
        assert torch.equal(
            stub.local_tokens_per_expert,
            torch.full((NUM_EXPERTS,), 513.0, dtype=torch.float32),
        )


class TestExpertBiasCountSanityGate:
    """Garbage token counts must skip the bias tick instead of corrupting it."""

    RATE = 1e-3

    def setup_method(self, method):
        _ensure_single_rank_dist()
        self.group = torch.distributed.group.WORLD

    def _healthy_counts(self):
        return torch.tensor(
            [[10.0, 20.0, 30.0, 40.0], [5.0, 5.0, 100.0, 2.0]], dtype=torch.float32
        )

    def test_healthy_counts_still_tick(self):
        counts = self._healthy_counts()
        bias = torch.zeros_like(counts)
        updated = get_updated_expert_bias(counts.clone(), bias, self.RATE, self.group)
        avg = counts.sum(dim=-1, keepdim=True) / counts.shape[-1]
        expected = torch.sign(avg - counts) * self.RATE
        assert torch.equal(updated, expected)

    @pytest.mark.parametrize("poison", [float("nan"), float("inf"), -3.0])
    def test_garbage_counts_skip_tick_and_warn(self, poison, caplog):
        counts = self._healthy_counts()
        counts[1, 2] = poison  # one bad layer; layer 0 stays healthy
        bias = torch.full_like(counts, 0.25)
        with caplog.at_level(
            logging.WARNING, logger="megatron.core.transformer.moe.moe_utils"
        ):
            updated = get_updated_expert_bias(counts, bias.clone(), self.RATE, self.group)
        # FAILS on the ungated code: the healthy layer-0 row still ticked.
        assert torch.equal(updated, bias), (
            "bias must be returned unchanged when any count is garbage"
        )
        assert any(
            "SKIPPING expert-bias update" in rec.message for rec in caplog.records
        ), "the skipped tick must be loudly logged"

    def test_gate_is_all_ranks_consistent_shape_1d(self):
        # The tick is also exercised with unstacked [num_experts] tensors.
        counts = torch.tensor([4.0, float("nan"), 8.0, 4.0])
        bias = torch.zeros(4)
        updated = get_updated_expert_bias(counts, bias.clone(), self.RATE, self.group)
        assert torch.equal(updated, bias)


class TestExpertBiasSensorEndToEndCuda:
    """Full TopKRouter forward on GPU, including the production bf16 module cast."""

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def _make_router(self):
        from megatron.core import parallel_state
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
        from megatron.core.transformer.transformer_config import TransformerConfig

        if not torch.distributed.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29981")
            torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel()

        config = TransformerConfig(
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=NUM_EXPERTS,
            moe_router_topk=TOPK,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=1e-3,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
        )
        router = TopKRouter(
            config=config, pg_collection=get_default_pg_collection(), layer_number=1
        )
        # Production casts the whole module (buffers included) to bf16.
        router = router.cuda().bfloat16()
        router.train()
        return router

    def test_count_buffer_stays_fp32_and_counts_exactly(self):
        torch.manual_seed(1234)
        router = self._make_router()
        doc_lens = [1000, 700, 260, 37]
        padding_mask = _thd_padding_mask(doc_lens, pad_multiple=64).cuda().unsqueeze(-1)
        num_tokens = padding_mask.shape[0]
        hidden = torch.randn(
            num_tokens, 1, router.config.hidden_size, device="cuda", dtype=torch.bfloat16
        )
        with torch.enable_grad():
            _, routing_map = router(hidden, padding_mask)
        # FAILS on the unfixed code: the bf16 module cast left the sensor bf16.
        assert router.local_tokens_per_expert.dtype == torch.float32
        expected = _naive_counts(routing_map.cpu(), padding_mask.cpu())
        assert torch.equal(router.local_tokens_per_expert.float().cpu(), expected)
        assert router.local_tokens_per_expert.sum().item() == sum(doc_lens) * TOPK
