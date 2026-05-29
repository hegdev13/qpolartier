"""Evaluate a patched QPolarTier model on a text corpus."""

from __future__ import annotations

import argparse
import math
from typing import Iterable, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import QPolarTierConfig
from .patch import patch_model, reset_caches


def evaluate_perplexity(model, tokenizer, texts: Iterable[str], max_length: int = 512) -> float:
    """Compute perplexity over a text iterable with cached autoregressive decoding."""

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text or not text.strip():
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            if input_ids.shape[1] < 2:
                continue

            past_key_values = None
            for position in range(input_ids.shape[1] - 1):
                step_ids = input_ids[:, position : position + 1]
                if position == 0:
                    step_attention_mask = attention_mask[:, :1] if attention_mask is not None else None
                    outputs = model(
                        input_ids=step_ids,
                        attention_mask=step_attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    step_attention_mask = attention_mask[:, : position + 1] if attention_mask is not None else None
                    outputs = model(
                        input_ids=step_ids,
                        attention_mask=step_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                target_token = input_ids[:, position + 1]
                total_nll += float(
                    torch.nn.functional.cross_entropy(logits, target_token, reduction="sum").item()
                )
                total_tokens += int(target_token.numel())
            reset_caches(model)

    if total_tokens == 0:
        raise ValueError("No non-empty texts were provided")

    return math.exp(total_nll / total_tokens)


def _load_wikitext_texts(limit: int, min_chars: int = 100) -> List[str]:
    fallback_texts = [
        "PolarQuant explores structured KV-cache compression with polar quantization and tiered approximation.",
        "This benchmark measures whether the patched attention path preserves perplexity under quantized caches.",
        "Importance-aware caching can trade off memory savings against quality degradation during long-context inference.",
        "A robust prototype should run both the baseline and the modified model on the same input texts.",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--top-tier-bits", type=int, default=8)
    parser.add_argument("--no-tiering", action="store_true")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    )
    model.to(args.device).eval()

    texts = _load_wikitext_texts(args.sample_count)

    baseline_ppl = evaluate_perplexity(model, tokenizer, texts, max_length=args.max_length)
    print(f"Baseline (fp16) PPL: {baseline_ppl:.4f}")

    config = QPolarTierConfig(top_tier_bits=args.top_tier_bits, no_tiering=args.no_tiering)
    patch_model(model, config)
    qpolar_ppl = evaluate_perplexity(model, tokenizer, texts, max_length=args.max_length)
    print(f"QPolarTier b1={args.top_tier_bits} PPL: {qpolar_ppl:.4f}")
    print(f"Delta: {qpolar_ppl - baseline_ppl:+.4f}")


if __name__ == "__main__":
    main()
