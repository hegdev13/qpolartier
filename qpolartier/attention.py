"""QPolarTier attention module for Hugging Face Llama models."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .config import QPolarTierConfig
from .quantize import polar_dot_product, reshape_key_to_pairs, to_polar
from .tier_manager import TierManager, TieredKeyCache


class QPolarTierAttention(nn.Module):
    """Drop-in attention replacement that quantizes prefilling keys into tiers."""

    def __init__(self, original_attn: nn.Module, config: QPolarTierConfig):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.config = getattr(original_attn, "config", None)

        self.num_heads = getattr(original_attn, "num_heads", None)
        if self.num_heads is None:
            self.num_heads = getattr(self.config, "num_attention_heads")

        self.num_key_value_heads = getattr(original_attn, "num_key_value_heads", None)
        if self.num_key_value_heads is None:
            self.num_key_value_heads = getattr(self.config, "num_key_value_heads", self.num_heads)
        self.head_dim = getattr(original_attn, "head_dim")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.layer_idx = getattr(original_attn, "layer_idx", None)
        self.attention_dropout = getattr(original_attn, "attention_dropout", 0.0)
        self.rotary_emb = getattr(original_attn, "rotary_emb", None)

        self.tier_config = config
        self.tier_manager = TierManager(config)

        self._tiered_key_cache: Optional[list[TieredKeyCache]] = None
        self._value_cache: Optional[torch.Tensor] = None
        self._prefill_importance: Optional[torch.Tensor] = None
        self._prefill_seq_len: Optional[int] = None
        self._cached_assignments: Optional[torch.Tensor] = None
        self._tier_update_count: int = 0
        self._cache_size_history: list[int] = []
        self._feature_rows_history: list[list[dict]] = []
        self._prefill_importance_history: list[torch.Tensor] = []
        self._oracle_importance_history: list[torch.Tensor] = []
        self._oracle_assignments_history: list[torch.Tensor] = []
        self._oracle_tier_mass_history: list[Tuple[float, float]] = []
        self._oracle_tier_fraction_history: list[dict[int, float]] = []
        self._debug_probe_records: list[dict] = []
        self._is_prefill: bool = True

    def _importance_from_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        if attn_weights.dim() != 4:
            raise ValueError(f"expected 4D attention weights, got shape {tuple(attn_weights.shape)}")
        first_batch_weights = attn_weights[0].float()
        return first_batch_weights.mean(dim=0).sum(dim=0)

    def _record_assignment_stats(self, importance: torch.Tensor, assignments: torch.Tensor) -> None:
        importance = importance.detach().float().cpu()
        assignments = assignments.detach().cpu()
        self._oracle_importance_history.append(importance)
        self._oracle_assignments_history.append(assignments)

        tier1_mask = assignments == 1
        tier12_mask = (assignments == 1) | (assignments == 2)
        tier1_mass = float(importance[tier1_mask].sum().item()) if bool(tier1_mask.any().item()) else 0.0
        tier12_mass = float(importance[tier12_mask].sum().item()) if bool(tier12_mask.any().item()) else 0.0
        self._oracle_tier_mass_history.append((tier1_mass, tier12_mass))

        total_tokens = max(1, int(assignments.numel()))
        fraction_by_tier: dict[int, float] = {}
        for tier in torch.unique(assignments).tolist():
            fraction_by_tier[int(tier)] = float((assignments == int(tier)).sum().item() / total_tokens)
        self._oracle_tier_fraction_history.append(fraction_by_tier)

    def _record_static_feature_rows(self, key_states: torch.Tensor, assignments: torch.Tensor) -> None:
        """Save per-token static features that are available at prefill time."""

        key_vectors = key_states[0].detach().float().cpu()  # [kv_heads, tokens, head_dim]
        assignments_cpu = assignments.detach().cpu()
        token_count = int(assignments_cpu.numel())

        key_l2_norm = torch.linalg.norm(key_vectors, dim=-1).mean(dim=0)
        mean_abs_key_value = key_vectors.abs().mean(dim=-1).mean(dim=0)

        key_pairs = reshape_key_to_pairs(key_vectors)
        radius, _ = to_polar(key_pairs)
        polar_radius_mean = radius.mean(dim=-1).mean(dim=0)
        polar_radius_std = radius.std(dim=-1, unbiased=False).mean(dim=0)
        polar_radius_max = radius.max(dim=-1).values.mean(dim=0)

        pair_energy = radius.square()
        pair_count = pair_energy.shape[-1]
        band_indices = torch.tensor_split(torch.arange(pair_count), 3)
        band_energy_rows = []
        total_pair_energy = pair_energy.sum(dim=-1).clamp(min=1e-8)
        for band in band_indices:
            if band.numel() == 0:
                band_energy_rows.append(torch.zeros_like(total_pair_energy))
            else:
                band_energy_rows.append(pair_energy.index_select(-1, band).sum(dim=-1) / total_pair_energy)
        rope_band_energy = torch.stack(band_energy_rows, dim=-1).mean(dim=0)

        rows: list[dict] = []
        for position_index in range(token_count):
            rows.append(
                {
                    "layer_index": int(self.layer_idx if self.layer_idx is not None else -1),
                    "position_index": int(position_index),
                    "token_age": int(token_count - 1 - position_index),
                    "oracle_tier": int(assignments_cpu[position_index].item()),
                    "key_l2_norm": float(key_l2_norm[position_index].item()),
                    "polar_radius_mean": float(polar_radius_mean[position_index].item()),
                    "polar_radius_std": float(polar_radius_std[position_index].item()),
                    "polar_radius_max": float(polar_radius_max[position_index].item()),
                    "rope_band_energy_low": float(rope_band_energy[position_index, 0].item()),
                    "rope_band_energy_mid": float(rope_band_energy[position_index, 1].item()),
                    "rope_band_energy_high": float(rope_band_energy[position_index, 2].item()),
                    "mean_abs_key_value": float(mean_abs_key_value[position_index].item()),
                }
            )

        self._feature_rows_history.append(rows)

    def _normalize_feature(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        min_val = values.min()
        max_val = values.max()
        if torch.isclose(max_val, min_val):
            return torch.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def _assign_heuristic_tiers(self, key_states: torch.Tensor) -> torch.Tensor:
        """Assign tiers from static prefill-time features without decode attention."""

        key_vectors = key_states[0].detach().float().cpu()
        token_count = int(key_vectors.shape[1])
        positions = torch.arange(token_count, dtype=torch.float32)
        token_age = (token_count - 1 - positions).float()

        key_l2_norm = torch.linalg.norm(key_vectors, dim=-1).mean(dim=0)
        key_pairs = reshape_key_to_pairs(key_vectors)
        radius, _ = to_polar(key_pairs)
        polar_radius_mean = radius.mean(dim=-1).mean(dim=0)

        pair_count = radius.shape[-1]
        low_band_count = max(1, pair_count // 3)
        low_band_energy = radius.square().index_select(-1, torch.arange(low_band_count)).sum(dim=-1)
        total_energy = radius.square().sum(dim=-1).clamp(min=1e-8)
        rope_band_energy_low = low_band_energy.mean(dim=0) / total_energy.mean(dim=0)

        token_age_n = self._normalize_feature(token_age)
        polar_radius_n = self._normalize_feature(polar_radius_mean)
        rope_low_n = self._normalize_feature(rope_band_energy_low)

        score = 0.4 * token_age_n + 0.3 * polar_radius_n + 0.3 * rope_low_n
        return self.tier_manager.assign_tiers(score.to(key_states.device))

    def _rankdata(self, values: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(values)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
        return ranks

    def _pearsonr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.float()
        y = y.float()
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.linalg.norm(x) * torch.linalg.norm(y)
        if denom == 0:
            return float("nan")
        return float(torch.dot(x, y) / denom)

    def _spearmanr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return self._pearsonr(self._rankdata(x.float()), self._rankdata(y.float()))

    def _maybe_record_debug_probe(
        self,
        *,
        query: torch.Tensor,
        key_states: torch.Tensor,
        qpt_scores: torch.Tensor,
        tier_cache: TieredKeyCache,
        head_idx: int,
        decode_step_idx: int,
    ) -> None:
        if not getattr(self.tier_config, "debug_attention_probe", False):
            return
        debug_layer_idx = getattr(self.tier_config, "debug_probe_layer_idx", None)
        debug_head_idx = getattr(self.tier_config, "debug_probe_head_idx", None)
        debug_decode_step = getattr(self.tier_config, "debug_probe_decode_step", 0)
        if debug_layer_idx is not None and self.layer_idx != debug_layer_idx:
            return
        if debug_head_idx is not None and head_idx != debug_head_idx:
            return
        decode_mismatch = decode_step_idx != debug_decode_step

        fp_scores = torch.matmul(query.unsqueeze(0), key_states[0, head_idx // self.num_key_value_groups].transpose(0, 1)).squeeze(0)
        qpt_scores = qpt_scores.detach().float()
        fp_scores = fp_scores.detach().float()

        valid_mask = torch.isfinite(fp_scores) & torch.isfinite(qpt_scores)
        valid_fp = fp_scores[valid_mask]
        valid_qpt = qpt_scores[valid_mask]

        if valid_fp.numel() == 0:
            mse = float("nan")
            pearson = float("nan")
            spearman = float("nan")
            max_abs_error = float("nan")
            top_fp = []
            top_qpt = []
        else:
            mse = float(torch.mean((valid_fp - valid_qpt) ** 2).item())
            pearson = self._pearsonr(valid_fp, valid_qpt)
            spearman = self._spearmanr(valid_fp, valid_qpt)
            max_abs_error = float(torch.max(torch.abs(valid_fp - valid_qpt)).item())

            top_k = min(getattr(self.tier_config, "debug_probe_top_k", 10), valid_fp.numel())
            top_fp_idx = torch.topk(valid_fp, top_k).indices
            top_qpt_idx = torch.topk(valid_qpt, top_k).indices
            valid_positions = valid_mask.nonzero(as_tuple=True)[0]
            top_fp = valid_positions[top_fp_idx].tolist()
            top_qpt = valid_positions[top_qpt_idx].tolist()

        tier_indices_ok = True
        position_coverage_ok = True
        scatter_exact = True
        seen_positions = []
        for tier in range(1, len(tier_cache.token_indices)):
            indices = tier_cache.token_indices[tier]
            if indices != sorted(indices):
                tier_indices_ok = False
            seen_positions.extend(indices)

        seen_positions.extend(tier_cache.evicted)
        expected_positions = list(range(qpt_scores.numel()))
        if sorted(seen_positions) != expected_positions:
            position_coverage_ok = False

        reconstructed = torch.full_like(qpt_scores, float("nan"))
        for tier in range(1, len(tier_cache.token_indices)):
            indices = tier_cache.token_indices[tier]
            if not indices:
                continue
            token_indices = torch.tensor(indices, device=qpt_scores.device, dtype=torch.long)
            reconstructed[token_indices] = qpt_scores[token_indices]
        if tier_cache.evicted:
            evicted_idx = torch.tensor(tier_cache.evicted, device=qpt_scores.device, dtype=torch.long)
            reconstructed[evicted_idx] = qpt_scores[evicted_idx]
        if not torch.equal(torch.isnan(reconstructed), torch.isnan(qpt_scores)):
            scatter_exact = False
        else:
            finite_mask = torch.isfinite(reconstructed) & torch.isfinite(qpt_scores)
            if finite_mask.any() and not torch.allclose(reconstructed[finite_mask], qpt_scores[finite_mask]):
                scatter_exact = False

        self._debug_probe_records.append(
            {
                "layer": int(self.layer_idx if self.layer_idx is not None else -1),
                "head": int(head_idx),
                "decode_step": int(decode_step_idx),
                "mse": mse,
                "pearson": pearson,
                "spearman": spearman,
                "max_abs_error": max_abs_error,
                "top_fp": top_fp,
                "top_qpt": top_qpt,
                "tier_indices_sorted": tier_indices_ok,
                "position_coverage_ok": position_coverage_ok,
                "scatter_exact": scatter_exact,
                "qpt_length": int(qpt_scores.numel()),
                "valid_positions": int(valid_mask.sum().item()),
                "evicted_count": int(len(tier_cache.evicted)),
            }
        )

    def _build_tier_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        assignments: torch.Tensor,
    ) -> None:
        # Record assignments but optionally avoid re-quantizing if freezing is enabled
        # If freeze flag is set and cache already exists, skip rebuild to preserve tiers
        if getattr(self.tier_config, "freeze_tiers_after_first_assign", False) and self._tiered_key_cache is not None:
            # still record that an update was requested
            self._tier_update_count += 1
            return

        self._cached_assignments = assignments.detach().cpu()
        self._tiered_key_cache = []

        for kv_head in range(key_states.shape[1]):
            key_matrix = key_states[0, kv_head].detach().cpu()
            self._tiered_key_cache.append(self.tier_manager.quantize_keys(key_matrix, self._cached_assignments))
        self._value_cache = value_states[0].detach()
        self._prefill_seq_len = int(key_states.shape[2])
        # track update counts and cache sizes
        self._tier_update_count += 1
        total_cached = 0
        for tier_cache in self._tiered_key_cache:
            # token_indices lists are stored on TieredKeyCache
            total_cached += sum(len(lst) for lst in getattr(tier_cache, "token_indices", []))
        self._cache_size_history.append(int(total_cached))
        self._is_prefill = False

    def _apply_rotary(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        position_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is not None:
            cos, sin = position_embeddings
            return apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.rotary_emb is None:
            raise ValueError("QPolarTierAttention requires position_embeddings or rotary_emb")

        if position_ids is None:
            raise ValueError("position_ids must be provided when position_embeddings is omitted")

        cos, sin = self.rotary_emb(key_states, position_ids=position_ids)
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def _full_precision_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output, attn_weights if output_attentions else None

    def _store_prefill_cache(
        self,
        attn_weights: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Capture prefill importance and build the tiered key cache."""

        importance = self._importance_from_attention(attn_weights)
        attendable = torch.arange(1, importance.numel() + 1, device=importance.device, dtype=torch.float32)
        importance = importance / attendable

        if self._prefill_importance is not None and self.tier_config.prefill_importance_ema > 0:
            ema = self.tier_config.prefill_importance_ema
            importance = ema * self._prefill_importance.to(importance.device) + (1 - ema) * importance

        self._prefill_importance = importance.detach().cpu()
        self._prefill_importance_history.append(self._prefill_importance)

        if getattr(self.tier_config, "heuristic_tiering", False):
            assignments = self._assign_heuristic_tiers(key_states)
            self._record_static_feature_rows(key_states, assignments)
            self._build_tier_cache(key_states, value_states, assignments)
        elif not self.tier_config.oracle_tiering:
            assignments = self.tier_manager.assign_tiers(importance)
            self._build_tier_cache(key_states, value_states, assignments)
        else:
            self._prefill_seq_len = int(key_states.shape[2])
            self._is_prefill = False

    def _oracle_decode_step(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, actual_attn_weights = self._full_precision_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            True,
        )

        importance = self._importance_from_attention(actual_attn_weights)
        freeze_after_first = getattr(self.tier_config, "freeze_tiers_after_first_assign", False)
        if freeze_after_first and self._cached_assignments is not None and self._tiered_key_cache is not None:
            assignments = self._cached_assignments.to(importance.device)
        else:
            assignments = self.tier_manager.assign_tiers(importance)
            self._record_assignment_stats(importance, assignments)
            if not self._feature_rows_history:
                self._record_static_feature_rows(key_states, assignments)
            self._build_tier_cache(key_states, value_states, assignments)

        attn_output, attn_weights = self._decode_with_tiers(
            query_states,
            key_states,
            value_states,
            attention_mask,
            output_attentions,
        )
        batch_size, q_len, _ = query_states.shape[0], query_states.shape[2], query_states.shape[1]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights
        self._is_prefill = False

    def _decode_with_tiers(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._tiered_key_cache is None or self._prefill_seq_len is None:
            return self._full_precision_attention(query_states, key_states, value_states, attention_mask, output_attentions)

        batch_size, num_heads, q_len, _ = query_states.shape
        if batch_size != 1 or q_len != 1:
            return self._full_precision_attention(query_states, key_states, value_states, attention_mask, output_attentions)

        total_seq_len = int(key_states.shape[2])
        prefill_seq_len = int(self._prefill_seq_len)

        attn_outputs = []
        attn_weights_all = []

        for head_idx in range(num_heads):
            kv_head_idx = head_idx // self.num_key_value_groups
            query = query_states[0, head_idx, 0]

            scores = torch.full((total_seq_len,), float("-inf"), device=query.device, dtype=torch.float32)
            tier_cache = self._tiered_key_cache[kv_head_idx]

            for tier in range(1, len(tier_cache.r_q)):
                radius_q = tier_cache.r_q[tier]
                theta_q = tier_cache.theta_q[tier]
                radius_scale = tier_cache.r_scale[tier]
                if radius_q is None or theta_q is None or radius_scale is None:
                    continue

                tier_scores = polar_dot_product(
                    query,
                    radius_q.to(query.device),
                    theta_q.to(query.device),
                    radius_scale.to(query.device),
                    tier_cache.n_bits[tier],
                    tier_cache.m_bits[tier],
                ).to(torch.float32)
                token_indices = torch.tensor(tier_cache.token_indices[tier], device=query.device, dtype=torch.long)
                scores[token_indices] = tier_scores

            if tier_cache.evicted:
                evicted_idx = torch.tensor(tier_cache.evicted, device=query.device, dtype=torch.long)
                scores[evicted_idx] = float("-inf")

            if total_seq_len > prefill_seq_len:
                decode_keys = key_states[0, kv_head_idx, prefill_seq_len:, :]
                scores[prefill_seq_len:] = torch.matmul(decode_keys, query).to(torch.float32)

            decode_step_idx = total_seq_len - prefill_seq_len - 1
            self._maybe_record_debug_probe(
                query=query,
                key_states=key_states,
                qpt_scores=scores,
                tier_cache=tier_cache,
                head_idx=head_idx,
                decode_step_idx=decode_step_idx,
            )

            scores = scores / math.sqrt(self.head_dim)
            if attention_mask is not None:
                mask = attention_mask
                if mask.dim() == 4:
                    scores = scores + mask[0, 0, 0, : total_seq_len].to(scores.dtype)
                elif mask.dim() == 2:
                    scores = scores + mask[0, : total_seq_len].to(scores.dtype)
                else:
                    scores = scores + mask.reshape(-1)[:total_seq_len].to(scores.dtype)

            weights = F.softmax(scores, dim=-1).to(query_states.dtype)
            value_matrix = value_states[0, kv_head_idx, :total_seq_len, :]
            output = torch.matmul(weights.unsqueeze(0), value_matrix).squeeze(0)

            attn_outputs.append(output)
            if output_attentions:
                attn_weights_all.append(weights)

        attn_output = torch.stack(attn_outputs, dim=0).unsqueeze(0).unsqueeze(2)
        attn_weights = None
        if output_attentions:
            attn_weights = torch.stack(attn_weights_all, dim=0).unsqueeze(0).unsqueeze(2)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = self._apply_rotary(query_states, key_states, position_embeddings, position_ids)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if self.tier_config.oracle_tiering and past_key_values is not None and q_len == 1:
            return self._oracle_decode_step(
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
            )

        if self._is_prefill or q_len > 1 or self._tiered_key_cache is None:
            attn_output, attn_weights = self._full_precision_attention(
                query_states,
                key_states,
                value_states,
                attention_mask,
                True,
            )
            self._store_prefill_cache(attn_weights, key_states, value_states)
            if not output_attentions:
                attn_weights = None
        else:
            attn_output, attn_weights = self._decode_with_tiers(
                query_states,
                key_states,
                value_states,
                attention_mask,
                output_attentions,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
