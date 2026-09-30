# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field

import torch


@dataclass
class DSATopKCacheEntry:
    """Top-k tensors and the recompute layers that still consume them."""

    indices: torch.Tensor
    length: torch.Tensor | None
    # Layer numbers are 1-indexed, matching DSAttention.layer_number.
    remaining_recompute_layers: set[int] = field(default_factory=set)


@dataclass
class DSATopKCache:
    """Per-microbatch cache of DSA top-k tensors shared with recompute forwards."""

    _entries: dict[int, DSATopKCacheEntry] = field(default_factory=dict)

    @property
    def source_layers(self) -> set[int]:
        """Return the source layers with live cached top-k tensors."""
        return set(self._entries)

    def get(self, source_layer: int) -> DSATopKCacheEntry | None:
        """Return a cached entry without changing its lifetime."""
        return self._entries.get(source_layer)

    def is_recompute_layer_pending(self, source_layer: int, layer_number: int) -> bool:
        """Return whether a layer is registered to consume this entry during recompute."""
        entry = self._entries.get(source_layer)
        return entry is not None and layer_number in entry.remaining_recompute_layers

    def store(
        self,
        source_layer: int,
        indices: torch.Tensor,
        length: torch.Tensor | None,
        recompute_layers: set[int] | None = None,
    ) -> None:
        """Store one source layer's top-k tensors and recompute consumers."""
        if source_layer in self._entries:
            raise RuntimeError(f"DSA top-k for source layer {source_layer} is already cached.")
        self._entries[source_layer] = DSATopKCacheEntry(
            indices=indices,
            length=length,
            remaining_recompute_layers=set(recompute_layers or ()),
        )

    def release(self, source_layer: int) -> None:
        """Release one source layer's cached tensors."""
        self._entries.pop(source_layer, None)

    def register_recompute_layer(self, source_layer: int, layer_number: int) -> None:
        """Register a layer that will consume this entry during recompute."""
        entry = self._entries.get(source_layer)
        if entry is None:
            raise RuntimeError(f"DSA top-k for source layer {source_layer} is not cached.")
        entry.remaining_recompute_layers.add(layer_number)

    def release_if_recompute_complete(self, source_layer: int) -> bool:
        """Release an entry when it has no pending recompute consumers."""
        entry = self._entries.get(source_layer)
        if entry is None or entry.remaining_recompute_layers:
            return False
        self.release(source_layer)
        return True

    def mark_recompute_layer_complete(self, source_layer: int, layer_number: int) -> bool:
        """Release an entry after its final registered recompute consumer completes."""
        entry = self._entries.get(source_layer)
        if entry is None or layer_number not in entry.remaining_recompute_layers:
            return False
        entry.remaining_recompute_layers.remove(layer_number)
        if entry.remaining_recompute_layers:
            return False
        self.release(source_layer)
        return True
