"""Analyze whether static token properties predict frozen oracle tier assignments.

The script runs a frozen-oracle QPolarTier benchmark, captures the per-token
feature rows recorded at the first tier assignment in each layer, writes the
token-level data to disk, and reports correlation / mutual-information scores
for each feature against the oracle tier label.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpolartier.config import QPolarTierConfig
from qpolartier.patch import patch_model, reset_caches


def _load_texts(limit: int, min_chars: int = 100) -> List[str]:
    fallback_texts = [
        " ".join(
            [
                "PolarQuant explores structured KV-cache compression with polar quantization and tiered approximation.",
                "The oracle experiment checks whether the tiering mechanism remains stable when the importance signal is perfect.",
                "A robust prototype should preserve baseline quality while compressing the cache aggressively.",
            ]
            * 3
        ),
        " ".join(
            [
                "This benchmark measures whether the patched attention path preserves perplexity under quantized caches.",
                "We use a fixed warmup prefix and stream the remaining tokens one at a time through the cache.",
                "The goal is to separate quantization error from tier-assignment error.",
            ]
            * 3
        ),
        " ".join(
            [
                "Importance-aware caching can trade off memory savings against quality degradation during long-context inference.",
                "If the oracle-tiered run stays close to baseline, then the tiering framework is sound.",
                "If it still degrades badly, the tiering mechanism itself needs attention.",
            ]
            * 3
        ),
    ]

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts: List[str] = []
        for row in dataset:
            text = row["text"]
            if isinstance(text, str) and len(text) >= min_chars:
                texts.append(text)
            if len(texts) >= limit:
                break
        if texts:
            return texts
    except Exception:
        pass

    return fallback_texts[:limit]


def _bits_to_nm_split(bits: int) -> tuple[int, int]:
    if bits < 2:
        raise ValueError(f"tier bits must be >= 2, got {bits}")
    radius_bits = max(1, bits // 3)
    angle_bits = max(1, bits - radius_bits)
    return radius_bits, angle_bits


def _parse_tier_schedule(raw_schedule: str) -> List[int]:
    bits = [int(item.strip()) for item in raw_schedule.split(",") if item.strip()]
    if not bits:
        raise ValueError("oracle tier schedule cannot be empty")
    return bits


def _prepare_inputs(tokenizer, text: str, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return encoded


def _set_freeze_flag(config: QPolarTierConfig, enabled: bool) -> QPolarTierConfig:
    setattr(config, "freeze_tiers_after_first_assign", enabled)
    return config


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    min_val = float(values.min())
    max_val = float(values.max())
    if math.isclose(max_val, min_val):
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def _heuristic_scores_for_rows(rows: list[dict]) -> np.ndarray:
    token_age = _normalize(np.asarray([row["token_age"] for row in rows], dtype=float))
    polar_radius_mean = _normalize(np.asarray([row["polar_radius_mean"] for row in rows], dtype=float))
    rope_band_energy_low = _normalize(np.asarray([row["rope_band_energy_low"] for row in rows], dtype=float))
    return 0.4 * token_age + 0.3 * polar_radius_mean + 0.3 * rope_band_energy_low


def _assign_tiers_from_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    assignments = np.zeros(scores.shape[0], dtype=int)
    filled = 0
    tier = 1
    while filled < scores.shape[0]:
        capacity = 2**tier
        end = min(filled + capacity, scores.shape[0])
        assignments[order[filled:end]] = tier
        filled = end
        tier += 1
    return assignments


def _group_rows_by_layer(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["layer_index"]), []).append(row)
    for layer_rows in grouped.values():
        layer_rows.sort(key=lambda item: item["position_index"])
    return grouped


def _compare_heuristic_to_oracle(
    rows: list[dict],
    oracle_model,
) -> dict[str, float | list[dict]]:
    grouped_rows = _group_rows_by_layer(rows)
    all_scores = []
    all_oracle = []
    all_heuristic = []
    layer_mass_rows = []

    for layer_idx, layer_rows in sorted(grouped_rows.items()):
        scores = _heuristic_scores_for_rows(layer_rows)
        heuristic_assignments = _assign_tiers_from_scores(scores)
        oracle_assignments = np.asarray([row["oracle_tier"] for row in layer_rows], dtype=int)
        all_scores.append(scores)
        all_oracle.append(oracle_assignments)
        all_heuristic.append(heuristic_assignments)

        layer = oracle_model.model.layers[layer_idx]
        importance_history = getattr(layer.self_attn, "_oracle_importance_history", [])
        if not importance_history:
            continue
        importance = importance_history[0].detach().float().cpu().numpy()
        if importance.shape[0] != heuristic_assignments.shape[0]:
            continue
        total_mass = float(importance.sum()) if importance.size else float("nan")
        tier1_mass = float(importance[heuristic_assignments == 1].sum() / total_mass) if total_mass else float("nan")
        tier12_mass = float(importance[np.isin(heuristic_assignments, [1, 2])].sum() / total_mass) if total_mass else float("nan")
        tier_overlap = float((heuristic_assignments == oracle_assignments).mean())
        layer_mass_rows.append(
            {
                "layer": layer_idx,
                "tier1_mass": tier1_mass,
                "tier12_mass": tier12_mass,
                "tier_overlap": tier_overlap,
            }
        )

    if not all_scores:
        return {
            "spearman": float("nan"),
            "tier_overlap": float("nan"),
            "tier1_mass": float("nan"),
            "tier12_mass": float("nan"),
            "layer_mass_rows": layer_mass_rows,
        }

    flat_scores = np.concatenate(all_scores)
    flat_oracle = np.concatenate(all_oracle)
    flat_heuristic = np.concatenate(all_heuristic)
    return {
        "spearman": _spearmanr(flat_scores, flat_oracle.astype(float)),
        "tier_overlap": float((flat_heuristic == flat_oracle).mean()),
        "tier1_mass": float(np.mean([row["tier1_mass"] for row in layer_mass_rows])) if layer_mass_rows else float("nan"),
        "tier12_mass": float(np.mean([row["tier12_mass"] for row in layer_mass_rows])) if layer_mass_rows else float("nan"),
        "layer_mass_rows": layer_mass_rows,
    }


def _load_model(model_id: str, device: str, patched_config: QPolarTierConfig | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
    )
    model.to(device).eval()
    if patched_config is not None:
        patch_model(model, patched_config)
    return model, tokenizer


def _streaming_perplexity(
    model,
    tokenizer,
    texts: Sequence[str],
    prefill_tokens: int,
    max_length: int,
) -> float:
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                continue

            inputs = _prepare_inputs(tokenizer, text, max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            if input_ids.shape[1] <= prefill_tokens + 1:
                continue

            prefix_ids = input_ids[:, :prefill_tokens]
            prefix_mask = attention_mask[:, :prefill_tokens] if attention_mask is not None else None
            prefix_outputs = model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
                return_dict=True,
            )

            past_key_values = prefix_outputs.past_key_values
            for position in range(prefill_tokens, input_ids.shape[1] - 1):
                step_ids = input_ids[:, position : position + 1]
                step_mask = attention_mask[:, : position + 1] if attention_mask is not None else None
                outputs = model(
                    input_ids=step_ids,
                    attention_mask=step_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values

                logits = outputs.logits[:, -1, :]
                target_token = input_ids[:, position + 1]
                total_nll += float(F.cross_entropy(logits, target_token, reduction="sum").item())
                total_tokens += int(target_token.numel())

            reset_caches(model)

    if total_tokens == 0:
        raise ValueError("No non-empty texts were provided")

    return math.exp(total_nll / total_tokens)


def _collect_feature_rows(model) -> list[dict]:
    rows: list[dict] = []
    for layer in model.model.layers:
        attn = layer.self_attn
        for batch_rows in getattr(attn, "_feature_rows_history", []):
            rows.extend(batch_rows)
    return rows


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size:
        raise ValueError("shape mismatch for pearsonr")
    if x.size < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return _pearsonr(_rankdata(x), _rankdata(y))


def _discretize_feature(values: np.ndarray, max_bins: int = 16) -> np.ndarray:
    values = np.asarray(values)
    unique_count = np.unique(values).size
    if unique_count <= max_bins:
        _, encoded = np.unique(values, return_inverse=True)
        return encoded.astype(int)

    quantiles = np.linspace(0.0, 1.0, max_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    if edges.size <= 2:
        return np.zeros_like(values, dtype=int)
    return np.digitize(values, edges[1:-1], right=True).astype(int)


def _mutual_information(x: np.ndarray, y: np.ndarray, max_bins: int = 16) -> float:
    x_disc = _discretize_feature(x, max_bins=max_bins)
    x_vals, x_inv = np.unique(x_disc, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)
    y_vals = np.unique(y)
    contingency = np.zeros((x_vals.size, y_vals.size), dtype=float)
    for xi, yi in zip(x_inv, y_inv):
        contingency[xi, yi] += 1.0

    total = contingency.sum()
    if total == 0:
        return float("nan")

    mi = 0.0
    for i in range(contingency.shape[0]):
        for j in range(contingency.shape[1]):
            if contingency[i, j] == 0:
                continue
            pxy_ij = contingency[i, j] / total
            px_i = contingency[i, :].sum() / total
            py_j = contingency[:, j].sum() / total
            mi += pxy_ij * math.log(pxy_ij / (px_i * py_j))
    return float(mi)


def _summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    feature_names = [
        "layer_index",
        "position_index",
        "token_age",
        "key_l2_norm",
        "polar_radius_mean",
        "polar_radius_std",
        "polar_radius_max",
        "rope_band_energy_low",
        "rope_band_energy_mid",
        "rope_band_energy_high",
        "mean_abs_key_value",
    ]

    targets = np.asarray([row["oracle_tier"] for row in rows], dtype=float)
    summaries: list[dict] = []
    for feature_name in feature_names:
        values = np.asarray([row[feature_name] for row in rows], dtype=float)
        summaries.append(
            {
                "feature": feature_name,
                "pearson": _pearsonr(values, targets),
                "spearman": _spearmanr(values, targets),
                "mutual_info": _mutual_information(values, targets),
            }
        )

    ranking = sorted(summaries, key=lambda item: item["mutual_info"], reverse=True)
    return summaries, ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--prefill-tokens", type=int, default=8)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--oracle-tier-schedule", default="8,6,4,2")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    oracle_tier_bits = _parse_tier_schedule(args.oracle_tier_schedule)
    oracle_nm_splits = [_bits_to_nm_split(bits) for bits in oracle_tier_bits]

    texts = _load_texts(args.sample_count)

    oracle_config = _set_freeze_flag(
        QPolarTierConfig(top_tier_bits=max(oracle_tier_bits), oracle_tiering=True, nm_splits=oracle_nm_splits),
        True,
    )
    oracle_model, tokenizer = _load_model(args.model_id, args.device, oracle_config)
    oracle_ppl = _streaming_perplexity(oracle_model, tokenizer, texts, args.prefill_tokens, args.max_length)
    print(f"Frozen-oracle benchmark PPL: {oracle_ppl:.4f}")

    rows = _collect_feature_rows(oracle_model)
    if not rows:
        raise RuntimeError("no feature rows were collected; check the frozen oracle configuration")

    heuristic_metrics = _compare_heuristic_to_oracle(rows, oracle_model)

    heuristic_config = _set_freeze_flag(
        QPolarTierConfig(top_tier_bits=max(oracle_tier_bits), nm_splits=oracle_nm_splits),
        True,
    )
    setattr(heuristic_config, "heuristic_tiering", True)
    heuristic_model, heuristic_tokenizer = _load_model(args.model_id, args.device, heuristic_config)
    heuristic_ppl = _streaming_perplexity(heuristic_model, heuristic_tokenizer, texts, args.prefill_tokens, args.max_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "frozen_oracle_token_features.jsonl"
    summary_path = output_dir / "frozen_oracle_feature_summary.json"

    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    summaries, ranking = _summarize(rows)
    summary = {
        "row_count": len(rows),
        "tier_counts": {str(k): int(v) for k, v in Counter(row["oracle_tier"] for row in rows).items()},
        "features": summaries,
        "ranking_by_mutual_information": ranking,
        "heuristic_metrics": heuristic_metrics,
        "rows_path": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved token rows to: {rows_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Row count: {len(rows)}")
    print(f"Frozen heuristic PPL: {heuristic_ppl:.4f}")
    print(f"Spearman(heuristic score, oracle tier): {heuristic_metrics['spearman']:.4f}")
    print(f"Tier overlap with oracle: {heuristic_metrics['tier_overlap']:.4f}")
    print(f"Frozen heuristic Tier-1 mass: {heuristic_metrics['tier1_mass']:.4f}")
    print(f"Frozen heuristic Tier-1+Tier-2 mass: {heuristic_metrics['tier12_mass']:.4f}")
    print("Feature correlations:")
    for item in summaries:
        print(
            f"- {item['feature']}: pearson={item['pearson']:.4f}, spearman={item['spearman']:.4f}, mi={item['mutual_info']:.4f}"
        )

    print("Feature importance ranking by mutual information:")
    for idx, item in enumerate(ranking, start=1):
        print(f"{idx}. {item['feature']} (mi={item['mutual_info']:.4f})")

    del oracle_model
    del heuristic_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()