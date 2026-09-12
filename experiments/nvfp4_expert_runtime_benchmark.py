"""Measure what NVFP4 expert materialization costs, on one GPU.

The NVFP4 counterpart to the GLM one-GPU expert benchmark. It answers a single
question: against holding the routed experts in BF16, how much time does it cost
to hold them in four bits and materialize BF16 on the fly?

Both arms run the identical grouped BF16 GEMM through
``_run_grouped_linear_with_bf16_weights``; the only difference is where the BF16
weights come from. So the delta is the materialization and nothing else.

Two details that decide whether the number means anything:

* **Full activation recompute, asserted.** The block is wrapped in
  ``torch.utils.checkpoint`` and the forward count is checked, because under
  recompute the materialization is paid twice per step, which is exactly the
  configuration native storage requires.
* **A realistic token count.** Cost is amortized over the GEMM, so overhead is a
  function of how many rows reach an expert. At 131,072 tokens with top-k 16 and
  EP=32, one rank sees about 65,536 expert-token rows per layer. Benchmarking at
  a few thousand would overstate the overhead badly.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import transformer_engine.pytorch as tep
from transformer_engine.common.recipe import NVFP4BlockScaling

from megatron.core.extensions.transformer_engine_frozen_quantized import (
    _get_compiled_nvfp4_to_bf16,
    _get_frozen_nvfp4_weight_data,
    _run_grouped_linear_with_bf16_weights,
)

# Kimi-K3 routed experts, per rank at EP=32.
EXPERTS_PER_RANK = 28
LATENT_SIZE = 3584
INTERMEDIATE_SIZE = 3072


class ExpertBlock(torch.nn.Module):
    """One grouped-linear expert projection, in one of the two storage arms."""

    def __init__(self, grouped_linear, tokens_per_expert: list[int], *, materialize: bool):
        super().__init__()
        self.grouped_linear = grouped_linear
        self.tokens_per_expert = tokens_per_expert
        self.materialize = materialize
        self.forward_calls = 0
        if not materialize:
            self.persistent = torch.stack(
                [
                    getattr(grouped_linear, f"weight{i}").detach().clone()
                    for i in range(grouped_linear.num_gemms)
                ]
            )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if self.materialize:
            payload, scales, amaxes = _get_frozen_nvfp4_weight_data(self.grouped_linear)
            weights = _get_compiled_nvfp4_to_bf16()(payload, scales, amaxes)
        else:
            weights = self.persistent
        return _run_grouped_linear_with_bf16_weights(
            self.grouped_linear, inp, self.tokens_per_expert, weights, None
        )


def build(arm: str, out_features: int, in_features: int, tokens: int):
    recipe = NVFP4BlockScaling(disable_2d_quantization=True)
    with torch.no_grad(), tep.fp8_model_init(enabled=True, recipe=recipe):
        grouped_linear = tep.GroupedLinear(
            EXPERTS_PER_RANK,
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
    grouped_linear.requires_grad_(False)

    per_expert = tokens // EXPERTS_PER_RANK
    tokens_per_expert = [per_expert] * EXPERTS_PER_RANK
    tokens_per_expert[-1] += tokens - per_expert * EXPERTS_PER_RANK

    block = ExpertBlock(grouped_linear, tokens_per_expert, materialize=(arm == "nvfp4"))
    if arm != "nvfp4":
        # The BF16 arm must not also carry the quantized copy, or its memory
        # figure would be the sum of both.
        for i in range(grouped_linear.num_gemms):
            setattr(grouped_linear, f"weight{i}", None)
    return block


def time_arm(block, inp, grad_output, controls: int) -> tuple[float, int]:
    from torch.utils.checkpoint import checkpoint

    def step():
        inp.grad = None
        before = block.forward_calls
        out = checkpoint(block, inp, use_reentrant=False)
        out.backward(grad_output)
        # Under full recompute the block runs twice, so the materialization is
        # paid twice. If it only ran once the number would be optimistic.
        assert block.forward_calls - before == 2, block.forward_calls - before
        del out

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    durations = []
    for _ in range(controls):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        step()
        end.record()
        end.synchronize()
        durations.append(start.elapsed_time(end))
    return statistics.median(durations), torch.cuda.max_memory_allocated()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        default=65536,
        help="expert-token rows per rank per layer; 131K x top-16 / EP32",
    )
    parser.add_argument("--controls", type=int, default=10)
    parser.add_argument("--projection", choices=("fc1", "fc2", "both"), default="both")
    args = parser.parse_args()

    shapes = {"fc1": (2 * INTERMEDIATE_SIZE, LATENT_SIZE), "fc2": (LATENT_SIZE, INTERMEDIATE_SIZE)}
    if args.projection != "both":
        shapes = {args.projection: shapes[args.projection]}

    print(
        f"device={torch.cuda.get_device_name(0)}  experts/rank={EXPERTS_PER_RANK}  "
        f"tokens={args.tokens}  controls={args.controls}"
    )
    print(f"\n{'proj':<5} {'arm':<8} {'median ms':>10} {'peak MiB':>10} {'overhead':>10}")

    for name, (out_features, in_features) in shapes.items():
        results = {}
        for arm in ("bf16", "nvfp4"):
            torch.cuda.empty_cache()
            block = build(arm, out_features, in_features, args.tokens)
            inp = torch.randn(
                args.tokens, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            grad_output = torch.randn(
                args.tokens, out_features, device="cuda", dtype=torch.bfloat16
            )
            results[arm] = time_arm(block, inp, grad_output, args.controls)
            del block, inp, grad_output

        base = results["bf16"][0]
        for arm in ("bf16", "nvfp4"):
            duration, peak = results[arm]
            overhead = "" if arm == "bf16" else f"{(duration / base - 1) * 100:+.1f}%"
            print(f"{name:<5} {arm:<8} {duration:>10.3f} {peak / 2**20:>10.1f} {overhead:>10}")


if __name__ == "__main__":
    main()
