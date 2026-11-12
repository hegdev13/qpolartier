# QPolarTier

### Importance-Aware Tiered Polar Quantization for Large Language Model KV Caches

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/docs/transformers)
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-orange)]()

**QPolarTier** explores importance-aware precision allocation for Transformer key-value (KV) caches using polar quantization and hierarchical memory tiers.

Instead of assigning every cached token the same numerical precision, QPolarTier investigates whether tokens with different levels of attention relevance can be stored at different precision levels — allocating more bits to potentially important tokens while reducing precision or evicting less useful ones.

The project builds on the polar representation introduced by [PolarQuant](https://github.com/ericshwu/PolarQuant) and investigates an adaptive, tiered approach to KV-cache compression.

> **Research status:** Experimental prototype. Results are preliminary and should not be interpreted as evidence of a production-ready KV-cache compression method.

---

## Table of Contents

- [Motivation](#motivation)
- [Core Idea](#core-idea)
- [Methodology](#methodology)
  - [1. Polar Representation](#1-polar-representation)
  - [2. Precision Tiering](#2-precision-tiering)
  - [3. Token Importance Estimation](#3-token-importance-estimation)
  - [4. Feature-Guided Allocation](#4-feature-guided-allocation)
- [Architecture](#architecture)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running Experiments](#running-experiments)
- [Limitations and Open Questions](#limitations-and-open-questions)
- [Research Direction](#research-direction)
- [Acknowledgements](#acknowledgements)

---

## Motivation

Autoregressive language models maintain a KV cache containing previously computed keys and values during generation.

As context length increases, the cache becomes a significant source of GPU memory consumption.

Conventional quantization methods generally apply a relatively uniform precision policy across cached tokens. However, not every token contributes equally to future attention computations.

Some tokens consistently receive substantial attention, while others may become less relevant as generation progresses.

This motivates three questions:

1. Can token importance be estimated accurately enough to guide cache compression?
2. Can different tokens be assigned different quantization precisions without substantially degrading model quality?
3. Can a hierarchical memory organization combine quantization and token eviction under a unified allocation strategy?

QPolarTier investigates these questions through a combination of polar quantization, token importance estimation, and tiered precision allocation.

## Core Idea

QPolarTier explores a hierarchical KV-cache representation in which tokens are assigned to different precision tiers.

Conceptually:

| Tier | Representation | Intended role |
|---|---|---|
| Tier 1 | High precision | Preserve highly relevant tokens |
| Tier 2 | Medium-high precision | Retain moderately relevant tokens |
| Tier 3 | Low precision | Reduce storage for less relevant tokens |
| Eviction | Not retained | Reclaim cache capacity |

The central idea is to make cache precision allocation dependent on token characteristics rather than treating all cached representations identically.

The design combines two complementary mechanisms:

- **Polar quantization:** Representing post-RoPE key vectors through magnitude and angular components.
- **Importance-aware tiering:** Allocating precision based on estimated token relevance and measurable token features.

The approach is intended to explore the trade-off between cache memory consumption and language-model quality.

---

## Methodology

### 1. Polar Representation

Rotary Position Embeddings (RoPE) encode positional information through rotations of consecutive dimensions.

For a post-RoPE key vector:

\[
\tilde{k}\in\mathbb{R}^{d}
\]

QPolarTier considers pairs of consecutive dimensions and represents each pair in polar coordinates.

For pair \(j\):

\[
r_j=\sqrt{\tilde{k}_{2j}^{2}+\tilde{k}_{2j+1}^{2}}
\]

\[
\theta_j=\operatorname{atan2}
(\tilde{k}_{2j+1},\tilde{k}_{2j})
\]

The original components can be reconstructed as:

\[
\tilde{k}_{2j}=r_j\cos(\theta_j)
\]

\[
\tilde{k}_{2j+1}=r_j\sin(\theta_j)
\]

This representation separates the magnitude and angular information of each pair.

Instead of independently quantizing the original Cartesian components, the magnitude and angle can be quantized separately.

**Magnitude quantization**

For an \(n\)-bit representation:

\[
s_j=\frac{\max_i r_i[j]}{2^n-1}
\]

\[
Q_r(r_j)=
\operatorname{clamp}
\left(
\operatorname{round}\left(\frac{r_j}{s_j}\right),
0,2^n-1
\right)
\]

**Angular quantization**

For an \(m\)-bit representation:

\[
Q_\theta(\theta_j)=
\left\lfloor
\frac{\theta_j+\pi}{2\pi}2^m
\right\rfloor
\bmod 2^m
\]

The quantized magnitude and angle are then used to reconstruct an approximation of the original key vector.

This gives each polar pair a total precision budget of:

\[
b=n+m
\]

where \(n\) represents magnitude precision and \(m\) represents angular precision.

### 2. Precision Tiering

QPolarTier explores allocating different bit budgets to different token tiers.

An example experimental schedule is:

| Tier | Total precision | Magnitude bits | Angle bits |
|---|---:|---:|---:|
| Tier 1 | 8 | 3 | 5 |
| Tier 2 | 6 | 2 | 4 |
| Tier 3 | 4 | 1 | 3 |
| Tier 4 | 2 | 1 | 1 |

The schedule is configurable and is not a claim that these allocations are universally optimal.

The implementation also investigates frozen tier assignments, where a token's tier is determined once and retained rather than repeatedly reassigning and requantizing cached tokens.

This distinction is important because dynamic tier updates can introduce additional numerical error.

### 3. Token Importance Estimation

An important part of the research is determining which tokens deserve higher precision.

The initial approach investigates attention-derived importance during the prefill stage.

Let:

\[
A[t,i]
\]

denote the attention weight assigned to token \(i\) by query position \(t\).

A causal-mask-normalized importance estimate is:

\[
I_i=
\frac{1}{T-i}
\sum_{t=i}^{T-1}A[t,i]
\]

where \(T\) is the prefill sequence length.

This estimates the average attention received by token \(i\) from subsequent positions within the prefill window.

Tokens with larger importance scores are candidates for higher-precision storage.

However, prefill attention is not necessarily a reliable predictor of future decode-time attention. Evaluating this relationship is a central part of the research.

### 4. Feature-Guided Allocation

The project also investigates whether token characteristics can approximate the tier assignments produced by an oracle importance signal.

Features explored include:

- Token age and position index.
- Mean polar radius.
- Polar radius variance.
- Key-vector L2 norm.
- Low-, mid-, and high-frequency RoPE band energy.
- Mean absolute key magnitude.

A heuristic scoring function explored in the experiments is:

\[
S_i =
0.4\,\widehat{\text{Age}}_i
+0.3\,\widehat{\text{Radius}}_i
+0.3\,\widehat{\text{LowBandEnergy}}_i
\]

where the individual features are normalized before combination.

This is an experimental heuristic, not a learned or theoretically optimal allocation policy.

---

## Architecture

The project is organized into two primary components:

**Quantization and cache management**

- Polar representation and reconstruction.
- Magnitude and angular quantization.
- Precision-tier configuration.
- KV-cache evaluation.

**Experimental analysis**

- Importance-signal validation.
- Uniform versus tiered quantization.
- Oracle-tier experiments.
- Frozen-tier experiments.
- Feature analysis and heuristic evaluation.

The implementation prioritizes PyTorch-based experimentation and correctness validation. Custom high-performance kernels and production deployment are outside the current scope.

---

## Experimental Results

Experiments are conducted using:

- **Model:** TinyLlama 1.1B Chat
- **Framework:** PyTorch
- **Hardware:** NVIDIA A100
- **Metric:** Perplexity (PPL)
- **Evaluation:** Streaming perplexity with a short-context sanity-check configuration where specified

Lower perplexity indicates better language-model prediction quality.

### Main comparison

The following results are recorded in `result-checkpoint.md`.

| Configuration | Precision schedule | Perplexity |
|---|---|---:|
| FP16 baseline | Full precision | 58.6099 |
| Uniform polar quantization | 8-bit | 55.1263 |
| Oracle-tiered QPolarTier | 8, 6, 4, 2 | 829.2827 |
| Frozen-tier oracle | 8, 6, 4, 2 | 69.8518 |
| Frozen heuristic | 8, 6, 4, 2 | 56.6501 |

These measurements come from the recorded oracle-tier reruns and heuristic experiments. They should be interpreted within their experimental configuration rather than as a broad benchmark.

### Key observations

**1. Uniform polar quantization provides a useful baseline.**

The recorded 8-bit uniform polar configuration achieves a perplexity of 55.1263, compared with 58.6099 for the FP16 baseline in the same short-context experiment.

**2. Dynamic oracle tiering is unstable.**

The initial oracle-tier implementation produces substantial perplexity degradation. This motivated investigating frozen tier assignments.

**3. Freezing tier assignments improves stability.**

The frozen-tier oracle experiment reduces perplexity from 829.2827 to 69.8518.

This suggests that repeatedly updating and requantizing cached representations is an important source of instability in the tested implementation.

**4. Feature-guided tiering is promising but preliminary.**

The frozen heuristic achieves 56.6501 PPL, considerably closer to uniform polar quantization than the frozen oracle configuration.

Its Spearman correlation with the oracle tier assignments is 0.5467, with a tier overlap of 0.2071.

This indicates that the investigated features contain some useful allocation signal, but do not reproduce oracle assignments reliably.

### Feature analysis

The frozen-oracle feature analysis collected 198 observations.

| Feature | Spearman correlation with oracle tier |
|---|---:|
| Polar radius mean | 0.3029 |
| Low-frequency RoPE band energy | 0.2637 |
| Key L2 norm | 0.2968 |
| Mean absolute key value | 0.3002 |
| Polar radius standard deviation | 0.2993 |
| High-frequency RoPE band energy | -0.1229 |
| Layer index | 0.0000 |

Token position and token age also exhibit substantial mutual information with oracle tier assignments, despite having near-zero linear and rank correlations in this particular analysis.

**Current takeaway:** The experiments suggest that cache tiering, assignment stability, and importance estimation are tightly coupled. More work is required before claiming an advantage over uniform polar quantization.

---

## Repository Structure

```text
qpolartier/
│
├── qpolartier/                 # Core QPolarTier implementation
│
├── experiments/                # Validation and evaluation scripts
│   ├── validate_importance.py
│   └── analyze_frozen_oracle_features.py
│
├── Implementation.md           # Detailed implementation plan
├── result-checkpoint.md        # Experimental results and observations
├── requirements.txt            # Python dependencies
└── README.md
```

Refer to the repository for the current set of implementation and experiment files.

---

## Getting Started

### Requirements

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- CUDA-compatible GPU for GPU experiments

### Installation

Clone the repository:

```bash
git clone https://github.com/hegdev13/qpolartier.git
cd qpolartier
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The repository's dependency list includes:

```text
torch
transformers
datasets
accelerate
```

### Running Experiments

Run the importance validation experiment:

```bash
python experiments/validate_importance.py
```

Run the evaluation module:

```bash
python -m qpolartier.evaluate
```

Additional experimental scripts are located in `experiments/`.

For model-dependent experiments, ensure that the required model weights are accessible through Hugging Face and that the environment has sufficient GPU memory.

---

## Limitations and Open Questions

QPolarTier is an ongoing research prototype. Several questions remain open:

- **Importance prediction:** How accurately can prefill-time information predict future decode-time attention?
- **Tier stability:** How should tier assignments evolve without introducing repeated quantization error?
- **Allocation policy:** Which token features best approximate future relevance?
- **Memory accounting:** How should metadata, quantization scales, and storage overhead be incorporated into effective bits-per-token measurements?
- **Quality versus compression:** Can tiered quantization outperform uniform polar quantization at equivalent effective memory budgets?
- **Runtime efficiency:** What is the performance impact of tier management during autoregressive decoding?

The current results primarily establish experimental behavior and identify failure modes. They do not yet demonstrate end-to-end memory or latency improvements.

---

## Research Direction

The next stages of investigation include:

1. Improving decode-time importance prediction.
2. Refining feature-guided tier assignment.
3. Investigating adaptive and frozen tier-management strategies.
4. Evaluating across longer contexts and additional model architectures.
5. Comparing methods at matched effective KV-cache memory budgets.
6. Investigating optimized kernels and runtime overhead.

The long-term objective is to explore whether importance-aware precision allocation can provide a practical alternative to uniform KV-cache quantization.

---

## Acknowledgements

This project builds upon the ideas and implementation of:

- **[PolarQuant](https://github.com/ericshwu/PolarQuant)** — the underlying polar quantization approach.

QPolarTier is an independent research prototype exploring hierarchical precision allocation on top of this foundation.

---

**Disclaimer:** This repository is intended for research and experimentation. Results are preliminary, and the implementation should not be considered a validated production-ready quantization system.
