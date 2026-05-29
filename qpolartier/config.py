"""Configuration objects for QPolarTier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class QPolarTierConfig:
    """Runtime configuration for QPolarTier tiering and quantization."""

    tier_base: int = 2
    top_tier_bits: int = 8
    min_viable_bits: int = 2
    no_tiering: bool = False
    oracle_tiering: bool = False
    nm_splits: Optional[List[Tuple[int, int]]] = None
    use_prefill_interception: bool = True
    prefill_importance_ema: float = 0.0
    frequency_aware: bool = False
    heuristic_tiering: bool = False
    debug_attention_probe: bool = False
    debug_probe_layer_idx: Optional[int] = None
    debug_probe_head_idx: Optional[int] = None
    debug_probe_decode_step: int = 0
    debug_probe_top_k: int = 10
    freeze_tiers_after_first_assign: bool = False

    def get_nm_split(self, tier: int) -> Tuple[int, int]:
        """Return the (radius_bits, angle_bits) split for a 1-based tier."""

        if tier < 1:
            raise ValueError(f"tier must be >= 1, got {tier}")

        if self.nm_splits is not None:
            if tier - 1 < len(self.nm_splits):
                return self.nm_splits[tier - 1]
            # Fall back to the geometric default if the explicit schedule is shorter.

        total_bits = self.top_tier_bits // (self.tier_base ** (tier - 1))
        if total_bits < self.min_viable_bits:
            return (0, 0)

        radius_bits = max(1, total_bits // 3)
        angle_bits = max(1, total_bits - radius_bits)
        return radius_bits, angle_bits

    def tier_capacity(self, tier: int) -> int:
        """Capacity of a 1-based tier under the geometric schedule."""

        if tier < 1:
            raise ValueError(f"tier must be >= 1, got {tier}")
        return self.tier_base ** tier

    def num_tiers(self, token_count: int) -> int:
        """Number of tiers needed to store token_count items."""

        import math

        if token_count < 0:
            raise ValueError(f"token_count must be >= 0, got {token_count}")
        if token_count == 0:
            return 0
        return math.ceil(math.log(token_count + 2, self.tier_base)) - 1
