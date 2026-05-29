# Result Checkpoint

This file records benchmark runs and sanity checks for QPolarTier.

## Environment

- Workspace: `/Users/vishruth/Projects/Minor`
- Remote host: Lightning A100 (`ssh s_01kst3syrnmqvak1q4k8mbctka@ssh.lightning.ai`)
- Working remote CUDA stack: `torch 2.6.0+cu124`

## Recorded Experiments

| Date | Model | Mode | Bits | Device | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-05-29 | hf-internal-testing/tiny-random-LlamaForCausalLM | phase-1 validation | n/a | cpu | average_spearman=0.1041 |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | baseline | fp16 | cuda | PPL=216.4616 |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | QPolarTier | 8 | cuda | PPL=558.4414 |

## No-Tiering Sanity Check

These runs use streaming perplexity so the decode-time cache path is exercised.

| Date | Model | Mode | Bits | Device | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | baseline | fp16 | cuda | PPL=216.5932 |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | QPolarTier no-tiering | 8 | cuda | PPL=216.5319 |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | QPolarTier no-tiering | 4 | cuda | PPL=216.6577 |

## Oracle Tiering Sanity Check

Settings: TinyLlama/TinyLlama-1.1B-Chat-v1.0, sample-count=1, max-length=64, prefill-tokens=8.

| Date | Model | Mode | Bits | Device | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | baseline | fp16 | cuda | PPL=58.6099 |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | uniform polar | 8 | cuda | PPL=55.1263 |
| 2026-05-29 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | oracle-tiered QPolarTier | 8 | cuda | PPL=984.0787 |

Oracle-side attention summaries from the same run:

- Average attention mass captured by Tier-1 tokens: 0.7841
- Average attention mass captured by Tier-1 + Tier-2 tokens: 0.8699
- Token fractions: tier 0 = 0.6187, tier 1 = 0.0556, tier 2 = 0.1111, tier 3 = 0.2146
- Oracle-vs-prefill Spearman probe over the first 2 decode steps ranged from -0.9048 to 0.7381 across layers and steps

## Schedule-Only Oracle Rerun

Same model, same benchmark, same oracle ranking. Only the tier schedule changed to `8,6,4,2`.

| Date | Model | Mode | Bits | Device | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | baseline | fp16 | cuda | PPL=58.6099 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | uniform polar | 8 | cuda | PPL=55.1263 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | oracle-tiered QPolarTier | 8,6,4,2 | cuda | PPL=829.2827 |

Extra stats:

- Average attention mass captured by Tier-1 tokens: 0.7891
- Average attention mass captured by Tier-1 + Tier-2 tokens: 0.8777
- Token fractions: tier 0 = 0.2833, tier 1 = 0.0556, tier 2 = 0.1111, tier 3 = 0.2146, tier 4 = 0.3354
- Oracle-vs-prefill Spearman probe over the first 2 decode steps remained mixed, from -0.9048 to 0.7619 across layers and steps

## Frozen-Tier Oracle Rerun

Same model, same benchmark, same oracle schedule `8,6,4,2`. Tier assignments were frozen after the first oracle assignment; cached tokens were not re-tiered or requantized on later decode steps.

| Date | Model | Mode | Bits | Device | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | baseline | fp16 | cuda | PPL=58.6099 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | uniform polar | 8 | cuda | PPL=55.1263 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | frozen-tier oracle | 8,6,4,2 | cuda | PPL=69.8518 |

Extra stats:

- Tier updates performed: 44
- Average cache size: 36.0 tokens
- Average attention mass captured by Tier-1 tokens: 0.8058
- Average attention mass captured by Tier-1 + Tier-2 tokens: 0.9453
- Tier fractions: tier 1 = 0.2222, tier 2 = 0.4444, tier 3 = 0.3333

## Frozen-Oracle Feature Analysis

Analysis script: `experiments/analyze_frozen_oracle_features.py`

Run settings:

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Oracle schedule: `8,6,4,2`
- Frozen tiers enabled: yes
- Rows collected: 198

Saved artifacts:

- `artifacts/frozen_oracle_token_features.jsonl`
- `artifacts/frozen_oracle_feature_summary.json`

Feature summary against oracle tier:

- `position_index`: Pearson -0.0212, Spearman -0.0296, MI 0.5088
- `token_age`: Pearson 0.0212, Spearman 0.0296, MI 0.5088
- `polar_radius_mean`: Pearson 0.4470, Spearman 0.3029, MI 0.2395
- `rope_band_energy_low`: Pearson 0.4036, Spearman 0.2637, MI 0.2305
- `key_l2_norm`: Pearson 0.4387, Spearman 0.2968, MI 0.2151
- `mean_abs_key_value`: Pearson 0.4464, Spearman 0.3002, MI 0.2132
- `polar_radius_std`: Pearson 0.3996, Spearman 0.2993, MI 0.2043
- `polar_radius_max`: Pearson 0.3793, Spearman 0.2636, MI 0.1903
- `rope_band_energy_mid`: Pearson -0.1018, Spearman -0.0923, MI 0.1055
- `rope_band_energy_high`: Pearson -0.1631, Spearman -0.1229, MI 0.0797
- `layer_index`: Pearson -0.0000, Spearman 0.0000, MI 0.0000

Top MI features:

1. `position_index` / `token_age`
2. `polar_radius_mean`
3. `rope_band_energy_low`
4. `key_l2_norm`
5. `mean_abs_key_value`

Interpretation:

- Static position/age are the strongest signal, as expected.
- Energy-magnitude features are modestly predictive.
- High-frequency RoPE energy and layer index are weak predictors in this run.

## Frozen-Heuristic Tiering

Heuristic score:

$$
0.4 \cdot \text{normalized(token_age)} + 0.3 \cdot \text{normalized(polar_radius_mean)} + 0.3 \cdot \text{normalized(rope_band_energy_low)}
$$

Same TinyLlama benchmark and same geometric tier schedule `8,6,4,2`.

| Date | Model | Mode | Bits | Device | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | baseline | fp16 | cuda | PPL=58.6099 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | uniform polar | 8 | cuda | PPL=55.1263 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | frozen oracle | 8,6,4,2 | cuda | PPL=69.8518 |
| 2026-05-30 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | frozen heuristic | 8,6,4,2 | cuda | PPL=56.6501 |

Comparison against frozen oracle tiers:

- Spearman correlation between heuristic score and oracle tier: 0.5467
- Tier overlap with oracle tiers: 0.2071
- Frozen heuristic Tier-1 mass: 0.0500
- Frozen heuristic Tier-1 + Tier-2 mass: 0.1663

Interpretation:

- The heuristic is far better than the dynamic oracle-collapse case, but it does not track the frozen oracle tiering closely.
- Its PPL is close to the uniform 8-bit baseline and still well below the frozen oracle collapse.
- Static position/age plus magnitude / low-band RoPE features are predictive, but not enough by themselves to reproduce oracle tiers accurately.