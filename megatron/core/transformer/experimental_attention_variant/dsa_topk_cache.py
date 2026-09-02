# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch


@dataclass
class DSATopKCacheEntry:
    """Top-k tensors and the recompute layers that still consume them."""

    indices: torch.Tensor
    length: torch.Tensor | None
    remaining_recompute_layers: set[int] = field(default_factory=set)


@dataclass
class DSATopKCache:
    """Per-microbatch cache shared by DSA layers and their recompute forwards."""

    _entries: dict[int, DSATopKCacheEntry] = field(default_factory=dict)
    _activation_recompute_enabled: bool = False
    _in_recompute: bool = False

    @property
    def activation_recompute_enabled(self) -> bool:
        """Return whether the current layer invocation will be recomputed."""
        return self._activation_recompute_enabled

    @property
    def in_recompute(self) -> bool:
        """Return whether the cache is currently serving a recompute forward."""
        return self._in_recompute

    @property
    def source_layers(self) -> set[int]:
        """Return the source layers with live cached top-k tensors."""
        return set(self._entries)

    def get(self, source_layer: int) -> DSATopKCacheEntry | None:
        """Return a cached entry without changing its lifetime."""
        return self._entries.get(source_layer)

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

    @contextmanager
    def checkpoint_phase(self, enabled: bool, recomputing: bool) -> Iterator[None]:
        """Set checkpoint state for one layer invocation and restore it afterward."""
        previous_enabled = self._activation_recompute_enabled
        previous_recomputing = self._in_recompute
        self._activation_recompute_enabled = previous_enabled or enabled
        self._in_recompute = previous_recomputing or (enabled and recomputing)
        try:
            yield
        finally:
            self._activation_recompute_enabled = previous_enabled
            self._in_recompute = previous_recomputing
