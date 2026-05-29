"""Pure PyTorch polar conversion and quantization helpers."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def to_polar(k_pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert paired Cartesian coordinates to polar form."""

    x, y = k_pair[..., 0], k_pair[..., 1]
    radius = torch.sqrt(x.square() + y.square())
    theta = torch.atan2(y, x)
    return radius, theta


def from_polar(radius: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Convert polar coordinates back to paired Cartesian coordinates."""

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    return torch.stack((x, y), dim=-1)


def quantize_radius(
    radius: torch.Tensor,
    n_bits: int,
    scale: torch.Tensor | None = None,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """Quantize non-negative radii using per-channel max scaling."""

    if n_bits <= 0:
        return None, None

    max_val = (1 << n_bits) - 1
    if scale is None:
        scale = radius.max(dim=0).values / max_val
        scale = scale.clamp(min=1e-8)

    radius_q = (radius / scale).round().clamp(0, max_val).to(torch.int32)
    return radius_q, scale


def dequantize_radius(radius_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize radii from integer bins."""

    return radius_q.to(torch.float32) * scale


def quantize_angle(theta: torch.Tensor, m_bits: int) -> torch.Tensor | None:
    """Quantize angles in (-pi, pi] to uniform bins."""

    if m_bits <= 0:
        return None

    num_bins = 1 << m_bits
    theta_normalized = (theta + math.pi) / (2 * math.pi)
    theta_q = torch.floor(theta_normalized * num_bins).long()
    return theta_q.remainder(num_bins)


def dequantize_angle(theta_q: torch.Tensor, m_bits: int) -> torch.Tensor:
    """Dequantize angle indices using bin centers."""

    num_bins = 1 << m_bits
    theta_norm = (theta_q.to(torch.float32) + 0.5) / num_bins
    return theta_norm * (2 * math.pi) - math.pi


def reshape_key_to_pairs(k: torch.Tensor) -> torch.Tensor:
    """Reshape key vectors from [T, d] to [T, d/2, 2]."""

    if k.shape[-1] % 2 != 0:
        raise ValueError(f"head dim must be even, got {k.shape[-1]}")
    return k.view(*k.shape[:-1], k.shape[-1] // 2, 2)


def polar_dot_product(
    q: torch.Tensor,
    radius_q: torch.Tensor,
    theta_q: torch.Tensor,
    radius_scale: torch.Tensor,
    n_bits: int,
    m_bits: int,
) -> torch.Tensor:
    """Compute q · k_i from a quantized polar representation."""

    if n_bits <= 0 or m_bits <= 0:
        raise ValueError("polar_dot_product requires positive radius and angle bit-widths")

    head_dim = q.shape[0]
    if head_dim % 2 != 0:
        raise ValueError(f"query head dim must be even, got {head_dim}")

    q_pairs = q.view(head_dim // 2, 2)
    radius_hat = dequantize_radius(radius_q, radius_scale)
    theta_hat = dequantize_angle(theta_q, m_bits)

    k_x = radius_hat * torch.cos(theta_hat)
    k_y = radius_hat * torch.sin(theta_hat)

    pair_x = q_pairs[:, 0].unsqueeze(0)
    pair_y = q_pairs[:, 1].unsqueeze(0)
    return (k_x * pair_x + k_y * pair_y).sum(dim=-1)
