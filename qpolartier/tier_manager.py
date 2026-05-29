"""Tier assignment and tiered key cache structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .config import QPolarTierConfig
from .quantize import quantize_angle, quantize_radius, reshape_key_to_pairs, to_polar


@dataclass
class TieredKeyCache:
    """Quantized keys grouped by tier."""

    r_q: List[Optional[torch.Tensor]]
    theta_q: List[Optional[torch.Tensor]]
    r_scale: List[Optional[torch.Tensor]]
    n_bits: List[int]
    m_bits: List[int]
    token_indices: List[List[int]]
    evicted: List[int]


class TierManager:
    """Assign token importance to tiers and quantize keys per tier."""

    def __init__(self, config: QPolarTierConfig):
        self.config = config

    def assign_tiers(self, importance: torch.Tensor) -> torch.Tensor:
        """Return a 1D tensor of 1-based tier ids, with 0 reserved for eviction."""

        if importance.dim() != 1:
            raise ValueError(f"importance must be 1D, got shape {tuple(importance.shape)}")

        token_count = importance.numel()
        if self.config.no_tiering:
            return torch.ones(token_count, dtype=torch.long, device=importance.device)

        assignments = torch.zeros(token_count, dtype=torch.long, device=importance.device)
        sorted_indices = torch.argsort(importance, descending=True)

        filled = 0
        tier = 1
        while filled < token_count:
            capacity = self.config.tier_capacity(tier)
            end = min(filled + capacity, token_count)
            radius_bits, angle_bits = self.config.get_nm_split(tier)
            if radius_bits > 0 and angle_bits > 0:
                assignments[sorted_indices[filled:end]] = tier
            filled = end
            tier += 1

        return assignments

    def quantize_keys(self, k: torch.Tensor, assignments: torch.Tensor) -> TieredKeyCache:
        """Quantize post-RoPE key vectors according to a tier assignment."""

        if k.dim() != 2:
            raise ValueError(f"k must be [T, d], got shape {tuple(k.shape)}")
        if assignments.dim() != 1:
            raise ValueError(f"assignments must be 1D, got shape {tuple(assignments.shape)}")
        if k.shape[0] != assignments.numel():
            raise ValueError("k and assignments must refer to the same token count")

        k_pairs = reshape_key_to_pairs(k)
        radius, theta = to_polar(k_pairs)

        max_tier = int(assignments.max().item())
        cache = TieredKeyCache(
            r_q=[None] * (max_tier + 1),
            theta_q=[None] * (max_tier + 1),
            r_scale=[None] * (max_tier + 1),
            n_bits=[0] * (max_tier + 1),
            m_bits=[0] * (max_tier + 1),
            token_indices=[[] for _ in range(max_tier + 1)],
            evicted=[],
        )

        for tier in range(1, max_tier + 1):
            tier_mask = assignments == tier
            if not bool(tier_mask.any().item()):
                continue

            radius_bits, angle_bits = self.config.get_nm_split(tier)
            if radius_bits <= 0 or angle_bits <= 0:
                continue

            tier_radius = radius[tier_mask]
            tier_theta = theta[tier_mask]
            radius_q, radius_scale = quantize_radius(tier_radius, radius_bits)
            theta_q = quantize_angle(tier_theta, angle_bits)

            cache.r_q[tier] = radius_q
            cache.theta_q[tier] = theta_q
            cache.r_scale[tier] = radius_scale
            cache.n_bits[tier] = radius_bits
            cache.m_bits[tier] = angle_bits
            cache.token_indices[tier] = tier_mask.nonzero(as_tuple=True)[0].tolist()

        cache.evicted = (assignments == 0).nonzero(as_tuple=True)[0].tolist()
        return cache
