"""Utilities for patching a loaded Hugging Face Llama model."""

from __future__ import annotations

from .attention import QPolarTierAttention
from .config import QPolarTierConfig


def patch_model(model, config: QPolarTierConfig):
    """Replace each decoder layer's self-attention with QPolarTierAttention."""

    for layer_idx, layer in enumerate(model.model.layers):
        layer.self_attn = QPolarTierAttention(layer.self_attn, config)
        layer.self_attn.layer_idx = layer_idx
    return model


def reset_caches(model, clear_history: bool = False):
    """Clear QPolarTier caches between independent runs."""

    for layer in model.model.layers:
        attn = layer.self_attn
        if isinstance(attn, QPolarTierAttention):
            attn._tiered_key_cache = None
            attn._value_cache = None
            attn._prefill_importance = None
            attn._prefill_seq_len = None
            attn._cached_assignments = None
            attn._tier_update_count = 0
            if hasattr(attn, "_cache_size_history"):
                attn._cache_size_history.clear()
            if hasattr(attn, "_debug_probe_records"):
                attn._debug_probe_records.clear()
            if clear_history:
                attn._prefill_importance_history.clear()
                attn._oracle_importance_history.clear()
                attn._oracle_assignments_history.clear()
                attn._oracle_tier_mass_history.clear()
                attn._oracle_tier_fraction_history.clear()
                if hasattr(attn, "_feature_rows_history"):
                    attn._feature_rows_history.clear()
            attn._is_prefill = True
