"""Validate whether prefill attention predicts decode-time importance.

This script follows the phase-1 experiment in Implementation.md.
It loads a causal LM, records attention weights via forward hooks during
prefill and decode, and reports per-layer Spearman rank correlation between
prefill token importance and mean decode-time attention received.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LayerCorrelation:
    layer_idx: int
    spearman: float
    prefill_len: int
    decode_steps: int


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    """Return 0-based ranks for a 1D tensor.

    Ties are not averaged; this is sufficient for a quick exploratory check.
    """

    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
    return ranks


def spearmanr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute a simple Spearman rank correlation on 1D tensors."""

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


def _extract_attn_weights(output) -> Optional[torch.Tensor]:
    """Extract attention weights from a module forward hook output."""

    if isinstance(output, tuple):
        if len(output) >= 2 and output[1] is not None:
            return output[1]
    if hasattr(output, "attentions") and output.attentions is not None:
        return output.attentions
    return None


def _build_prompt(tokenizer, text: str, max_length: int) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return encoded


def _register_hooks(model) -> Tuple[List[torch.utils.hooks.RemovableHandle], Dict[int, torch.Tensor]]:
    prefill_attn_weights: Dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                attn_weights = _extract_attn_weights(output)
                if attn_weights is None:
                    return
                # attn_weights: [batch, heads, q_len, k_len]
                if attn_weights.dim() == 4:
                    prefill_attn_weights[idx] = attn_weights[0].mean(0).detach().float().cpu()
                elif attn_weights.dim() == 3:
                    prefill_attn_weights[idx] = attn_weights.mean(0).detach().float().cpu()
            return hook

        hooks.append(layer.self_attn.register_forward_hook(make_hook(layer_idx)))

    return hooks, prefill_attn_weights


def _column_importance(attn_matrix: torch.Tensor) -> torch.Tensor:
    """Convert a [T, T] causal attention matrix into token importance."""

    seq_len = attn_matrix.shape[-1]
    attendable = torch.arange(1, seq_len + 1, dtype=torch.float32)
    return attn_matrix.sum(dim=0) / attendable


def _decode_attention_series(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, decode_steps: int) -> Dict[int, torch.Tensor]:
    """Run a few decode steps and accumulate mean attention received per token."""

    if decode_steps <= 0:
        return {}

    decode_attention_sums: Dict[int, torch.Tensor] = {}

    prefix_len = max(1, input_ids.shape[1] - decode_steps)
    prefix_ids = input_ids[:, :prefix_len]
    prefix_mask = attention_mask[:, :prefix_len]

    with torch.no_grad():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

    past_key_values = outputs.past_key_values
    next_token_ids = input_ids[:, prefix_len:]

    for step in range(next_token_ids.shape[1]):
        step_ids = next_token_ids[:, step : step + 1]
        step_mask = attention_mask[:, : prefix_len + step + 1]
        with torch.no_grad():
            outputs = model(
                input_ids=step_ids,
                attention_mask=step_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        past_key_values = outputs.past_key_values

        layer_attns = outputs.attentions or ()
        for layer_idx, attn in enumerate(layer_attns):
            if attn is None:
                continue
            # Decode attention is [batch, heads, 1, total_k]
            mean_attn = attn[0].mean(0)[0].detach().float().cpu()
            accumulator = decode_attention_sums.get(layer_idx)
            if accumulator is None or accumulator.numel() < mean_attn.numel():
                expanded = torch.zeros_like(mean_attn)
                if accumulator is not None:
                    expanded[: accumulator.numel()] = accumulator
                decode_attention_sums[layer_idx] = expanded
                accumulator = expanded
            accumulator[: mean_attn.numel()] += mean_attn

    return decode_attention_sums


def run_validation(model_id: str, text: str, max_length: int, decode_steps: int, device: str) -> List[LayerCorrelation]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    inputs = _build_prompt(tokenizer, text, max_length=max_length)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    hooks, prefill_attn_weights = _register_hooks(model)
    try:
        with torch.no_grad():
            model(**inputs, use_cache=True, output_attentions=True, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()

    if not prefill_attn_weights:
        raise RuntimeError(
            "No attention weights were captured. Try attn_implementation='eager' and a model that returns attentions."
        )

    decode_attention_sums = _decode_attention_series(
        model,
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
        decode_steps=min(decode_steps, max(1, inputs["input_ids"].shape[1] - 1)),
    )

    results: List[LayerCorrelation] = []
    for layer_idx in sorted(prefill_attn_weights):
        prefill_matrix = prefill_attn_weights[layer_idx]
        prefill_importance = _column_importance(prefill_matrix)
        decode_importance = decode_attention_sums.get(layer_idx)
        if decode_importance is None:
            continue

        common_len = min(prefill_importance.numel(), decode_importance.numel())
        corr = spearmanr(prefill_importance[:common_len], decode_importance[:common_len])
        results.append(
            LayerCorrelation(
                layer_idx=layer_idx,
                spearman=corr,
                prefill_len=int(prefill_importance.numel()),
                decode_steps=int(decode_steps),
            )
        )

    return results


def _default_text() -> str:
    return (
        "PolarQuant turns key/value cache compression into a polar quantization problem. "
        "This validation pass checks whether attention received during prefill predicts "
        "which tokens remain important during decode."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--text", default=_default_text())
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = run_validation(
        model_id=args.model_id,
        text=args.text,
        max_length=args.max_length,
        decode_steps=args.decode_steps,
        device=args.device,
    )

    if not results:
        print("No per-layer correlations were produced.")
        return

    average = sum(item.spearman for item in results if not math.isnan(item.spearman)) / max(
        1, sum(0 if math.isnan(item.spearman) else 1 for item in results)
    )

    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Decode steps: {args.decode_steps}")
    for item in results:
        print(
            f"layer={item.layer_idx:02d} spearman={item.spearman:.4f} "
            f"prefill_len={item.prefill_len} decode_steps={item.decode_steps}"
        )
    print(f"average_spearman={average:.4f}")


if __name__ == "__main__":
    main()
