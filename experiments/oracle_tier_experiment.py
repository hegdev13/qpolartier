"""Oracle tiering experiment for QPolarTier.

This script runs three benchmark modes on a shared streaming-perplexity setup:

- Baseline fp16
- Uniform 8-bit polar (no tiering)
- Oracle-tiered QPolarTier

It also collects oracle attention rankings for a few decode steps and compares
them against prefill-attention rankings using Spearman correlation.
"""

from __future__ import annotations

import argparse
import gc
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpolartier.config import QPolarTierConfig
from qpolartier.patch import patch_model, reset_caches


@dataclass
class BenchmarkResult:
    mode: str
    ppl: float


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


def spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel():
        raise ValueError(f"shape mismatch for spearmanr: {x.shape} vs {y.shape}")
    if x.numel() < 2:
        return float("nan")

    rx = _rankdata(x.float())
    ry = _rankdata(y.float())
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = torch.linalg.norm(rx) * torch.linalg.norm(ry)
    if denom == 0:
        return float("nan")
    return float(torch.dot(rx, ry) / denom)


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
        " ".join(
            [
                "A robust prototype should run both the baseline and the modified model on the same input texts.",
                "Saving the oracle rankings lets us quantify how far the prefill signal is from the oracle signal.",
                "That makes this a clean sanity check for the full QPolarTier pipeline.",
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
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return encoded


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


def _oracle_probe_text() -> str:
    return (
        "PolarQuant treats key-value cache compression as a polar approximation problem. "
        "This probe compares a perfect ranking signal against the prefill signal to see "
        "whether tiering itself is the bottleneck."
    )


def _run_rank_probe(
    model,
    tokenizer,
    text: str,
    prefill_tokens: int,
    decode_steps: int,
    max_length: int,
):
    reset_caches(model, clear_history=True)

    inputs = _prepare_inputs(tokenizer, text, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    if input_ids.shape[1] <= prefill_tokens + decode_steps:
        raise ValueError("probe text is too short for the requested prefill/decode split")

    prefix_ids = input_ids[:, :prefill_tokens]
    prefix_mask = attention_mask[:, :prefill_tokens] if attention_mask is not None else None
    prefix_outputs = model(
        input_ids=prefix_ids,
        attention_mask=prefix_mask,
        use_cache=True,
        return_dict=True,
    )

    past_key_values = prefix_outputs.past_key_values
    for position in range(prefill_tokens, prefill_tokens + decode_steps):
        step_ids = input_ids[:, position : position + 1]
        step_mask = attention_mask[:, : position + 1] if attention_mask is not None else None
        outputs = model(
            input_ids=step_ids,
            attention_mask=step_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

    layer_rows = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        prefill_history = getattr(attn, "_prefill_importance_history", [])
        oracle_history = getattr(attn, "_oracle_importance_history", [])
        if not prefill_history or not oracle_history:
            continue

        prefill_ranking = prefill_history[0].float()
        for step_idx, oracle_ranking in enumerate(oracle_history[:decode_steps]):
            common_len = min(prefill_ranking.numel(), oracle_ranking.numel())
            corr = spearmanr(prefill_ranking[:common_len], oracle_ranking[:common_len])
            layer_rows.append(
                {
                    "layer": layer_idx,
                    "step": step_idx,
                    "spearman": corr,
                    "oracle_ranking": oracle_ranking[:common_len].tolist(),
                }
            )

    return layer_rows


def _summarize_oracle_stats(model):
    tier_mass_samples: list[tuple[float, float]] = []
    tier_counts: Counter[int] = Counter()
    total_tokens = 0

    for layer in model.model.layers:
        attn = layer.self_attn
        tier_mass_samples.extend(getattr(attn, "_oracle_tier_mass_history", []))
        for assignments in getattr(attn, "_oracle_assignments_history", []):
            flat = assignments.tolist()
            tier_counts.update(int(tier) for tier in flat)
            total_tokens += len(flat)

    if tier_mass_samples:
        avg_tier1_mass = sum(item[0] for item in tier_mass_samples) / len(tier_mass_samples)
        avg_tier12_mass = sum(item[1] for item in tier_mass_samples) / len(tier_mass_samples)
    else:
        avg_tier1_mass = float("nan")
        avg_tier12_mass = float("nan")

    fraction_by_tier = {
        tier: (count / total_tokens if total_tokens else float("nan"))
        for tier, count in sorted(tier_counts.items())
    }

    return avg_tier1_mass, avg_tier12_mass, fraction_by_tier


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


def _set_freeze_flag(config: QPolarTierConfig, enabled: bool) -> QPolarTierConfig:
    setattr(config, "freeze_tiers_after_first_assign", enabled)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=64)
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--oracle-probe-decode-steps", type=int, default=4)
    parser.add_argument("--oracle-tier-schedule", default="8,6,4,2")
    parser.add_argument("--freeze-tiers-after-first-assign", action="store_true", help="Freeze tier assignments after the first oracle assignment; do not requantize or move tokens between tiers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    oracle_tier_bits = _parse_tier_schedule(args.oracle_tier_schedule)
    oracle_nm_splits = [_bits_to_nm_split(bits) for bits in oracle_tier_bits]

    texts = _load_texts(args.sample_count)

    baseline_model, tokenizer = _load_model(args.model_id, args.device)
    baseline_ppl = _streaming_perplexity(baseline_model, tokenizer, texts, args.prefill_tokens, args.max_length)
    print(f"Baseline (fp16) PPL: {baseline_ppl:.4f}")
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    uniform_model, tokenizer = _load_model(
        args.model_id,
        args.device,
        QPolarTierConfig(top_tier_bits=8, no_tiering=True),
    )
    uniform_ppl = _streaming_perplexity(uniform_model, tokenizer, texts, args.prefill_tokens, args.max_length)
    print(f"Uniform 8-bit polar PPL: {uniform_ppl:.4f}")
    del uniform_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    oracle_config = _set_freeze_flag(
        QPolarTierConfig(top_tier_bits=max(oracle_tier_bits), oracle_tiering=True, nm_splits=oracle_nm_splits),
        args.freeze_tiers_after_first_assign,
    )
    oracle_model, tokenizer = _load_model(
        args.model_id,
        args.device,
        oracle_config,
    )
    oracle_ppl = _streaming_perplexity(oracle_model, tokenizer, texts, args.prefill_tokens, args.max_length)
    print(f"Oracle-tiered QPolarTier PPL: {oracle_ppl:.4f}")
    print(f"Delta vs baseline: {oracle_ppl - baseline_ppl:+.4f}")

    avg_tier1_mass, avg_tier12_mass, fraction_by_tier = _summarize_oracle_stats(oracle_model)
    print(f"Average attention mass captured by Tier-1 tokens: {avg_tier1_mass:.4f}")
    print(f"Average attention mass captured by Tier-1 + Tier-2 tokens: {avg_tier12_mass:.4f}")
    for tier, fraction in fraction_by_tier.items():
        print(f"Fraction of tokens assigned to tier {tier}: {fraction:.4f}")

    probe_rows = _run_rank_probe(
        oracle_model,
        tokenizer,
        _oracle_probe_text(),
        args.prefill_tokens,
        args.oracle_probe_decode_steps,
        args.max_length,
    )

    # New metrics: count tier updates and average cache size
    total_updates = 0
    total_cache_samples = 0
    total_cache_sum = 0
    for layer in oracle_model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "_tier_update_count"):
            total_updates += int(getattr(attn, "_tier_update_count", 0))
        if hasattr(attn, "_cache_size_history"):
            hist = getattr(attn, "_cache_size_history", [])
            if hist:
                total_cache_samples += len(hist)
                total_cache_sum += sum(hist)

    avg_cache_size = float(total_cache_sum / total_cache_samples) if total_cache_samples else float("nan")
    print(f"Number of tier updates performed: {total_updates}")
    print(f"Average cache size (tokens): {avg_cache_size}")

    if probe_rows:
        print("Oracle-vs-prefill Spearman correlations:")
        for row in probe_rows:
            print(
                f"layer={row['layer']:02d} step={row['step']:02d} spearman={row['spearman']:.4f}"
            )
    else:
        print("No oracle-vs-prefill correlation rows were produced.")

    print("Saved oracle ranking snapshots for the first decode steps in memory.")
    del oracle_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()