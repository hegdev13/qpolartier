"""Inspect full-precision vs tiered attention scores for one layer/head/step."""

from __future__ import annotations

import argparse
import gc
import json
import math
from typing import List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpolartier.config import QPolarTierConfig
from qpolartier.patch import patch_model, reset_caches


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    y = y.float() - y.float().mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(torch.dot(x, y) / denom)


def _spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearsonr(_rankdata(x.float()), _rankdata(y.float()))


def _load_texts(limit: int, min_chars: int = 100) -> List[str]:
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

    return [
        "PolarQuant explores structured KV-cache compression with polar quantization and tiered approximation.",
    ]


def _bits_to_nm_split(bits: int) -> tuple[int, int]:
    if bits < 2:
        raise ValueError(f"tier bits must be >= 2, got {bits}")
    radius_bits = max(1, bits // 3)
    angle_bits = max(1, bits - radius_bits)
    return radius_bits, angle_bits


def _prepare_inputs(tokenizer, text: str, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return encoded


def _configure_probe(config: QPolarTierConfig, layer_idx: int, head_idx: int, decode_step: int) -> QPolarTierConfig:
    config.debug_attention_probe = True
    config.debug_probe_layer_idx = layer_idx
    config.debug_probe_head_idx = head_idx
    config.debug_probe_decode_step = decode_step
    config.debug_probe_top_k = 10
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--prefill-tokens", type=int, default=8)
    parser.add_argument("--decode-step", type=int, default=0)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--head-idx", type=int, default=0)
    parser.add_argument("--oracle-tier-schedule", default="8,6,4,2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    texts = _load_texts(args.sample_index + 1)
    text = texts[args.sample_index]

    oracle_tier_bits = [int(item.strip()) for item in args.oracle_tier_schedule.split(",") if item.strip()]
    oracle_nm_splits = [_bits_to_nm_split(bits) for bits in oracle_tier_bits]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
    )
    model.to(args.device).eval()
    cfg = _configure_probe(
        QPolarTierConfig(
            top_tier_bits=max(oracle_tier_bits),
            oracle_tiering=True,
            nm_splits=oracle_nm_splits,
        ),
        args.layer_idx,
        args.head_idx,
        args.decode_step,
    )
    print("Patching model with debug probe config...", flush=True)
    patch_model(model, cfg)
    print("Patched. Verifying layer attributes:")
    for idx, layer in enumerate(model.model.layers[: min(3, len(model.model.layers))]):
        attn = layer.self_attn
        has_probe = hasattr(attn, "_debug_probe_records")
        print(f"layer {idx} probe_attr={has_probe} type={type(attn)}", flush=True)

    inputs = _prepare_inputs(tokenizer, text, max_length=args.max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    if input_ids.shape[1] <= args.prefill_tokens + args.decode_step + 1:
        raise ValueError("text is too short for the requested prefill/decode split")

    prefix_ids = input_ids[:, : args.prefill_tokens]
    prefix_mask = attention_mask[:, : args.prefill_tokens] if attention_mask is not None else None
    prefix_outputs = model(
        input_ids=prefix_ids,
        attention_mask=prefix_mask,
        use_cache=True,
        return_dict=True,
    )

    past_key_values = prefix_outputs.past_key_values
    for position in range(args.prefill_tokens, args.prefill_tokens + args.decode_step + 1):
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

    layer = model.model.layers[args.layer_idx]
    attn = layer.self_attn
    if not attn._debug_probe_records:
        raise RuntimeError("probe did not fire; check layer/head/decode-step settings")

    record = attn._debug_probe_records[0]
    print(json.dumps(record, indent=2, sort_keys=True))

    print("Verification checks:")
    print(f"- token_indices sorted per tier: {record['tier_indices_sorted']}")
    print(f"- positions covered exactly once: {record['position_coverage_ok']}")
    print(f"- scatter reconstruction exact: {record['scatter_exact']}")
    print(f"- valid positions compared: {record['valid_positions']}")
    print(f"- evicted positions: {record['evicted_count']}")
    print(f"- MSE: {record['mse']:.6f}")
    print(f"- Pearson: {record['pearson']:.6f}")
    print(f"- Spearman: {record['spearman']:.6f}")
    print(f"- Max abs error: {record['max_abs_error']:.6f}")
    print(f"- Top-10 fp16 tokens: {record['top_fp']}")
    print(f"- Top-10 QPolarTier tokens: {record['top_qpt']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()