# QPolarTier: Implementation Plan
*Built on top of github.com/ericshwu/PolarQuant*

---

## Smallest Viable Model

**Use: `Llama-3.2-1B-Instruct`**

Reasons:
- Smallest modern RoPE-based model with well-documented attention behavior
- 1B parameters fits in ~2GB VRAM at fp16, leaving room for activations
- Same architecture family (Llama) as PolarQuant's tested models (3.1-8B),
  so all layer/head/dimension assumptions carry over directly
- HuggingFace model ID: `meta-llama/Llama-3.2-1B-Instruct`

Llama-3.2-1B architecture facts you need:
- 16 layers
- 32 attention heads (Q), 8 KV heads (GQA)
- Head dimension d = 64
- RoPE base = 500,000
- d/2 = 32 polar pairs per head

**Do NOT start with DeepSeek-R1-Distill-Qwen-1.5B** even though PolarQuant's paper uses it
for reasoning benchmarks. It has a different tokenizer and chat template. Llama-3.2-1B is
cleaner for debugging.

If Lightning AI gives you memory issues even at 1B, fall back to `Qwen2.5-0.5B-Instruct`
(0.5B, 24 layers, 14 attention heads, d=64). It uses RoPE and the same GQA pattern.

---

## Repository Structure (PolarQuant)

```
ericshwu/PolarQuant/
├── models/           ← Modified attention implementations (YOUR MAIN TARGET)
├── utils/            ← Triton kernels, quantization utilities
├── benchmark/        ← Latency/memory benchmarks
├── public/           ← Datasets (LongBench, GSM8K)
├── test4long.py      ← LongBench evaluation entrypoint
├── test4gsm8k.py     ← GSM8K evaluation entrypoint
└── README.md
```

You will primarily work in `models/` and add a new `qpolartier/` directory alongside it.

---

## A Critical Note on PolarQuant's Triton Kernels

PolarQuant requires A100/A800 GPUs for their custom Triton kernels. If you are on a
smaller GPU on Lightning AI (e.g., T4, A10), **the Triton kernels will not run**.

**Solution for your prototype:** Implement QPolarTier in pure PyTorch first. This gives
you correctness validation without needing custom kernels. The PyTorch version will be
slower but mathematically identical. Custom Triton kernels come later if you want to
benchmark speed.

You will fork PolarQuant's quantization logic (not their Triton attention kernel) and
replace the attention computation with a standard PyTorch `torch.einsum` or `F.scaled_dot_product_attention`.

---

## Mathematical Verification

Before implementation, verify these equations are correct.

### 1. PolarQuant Encoding

For a post-RoPE key vector $\tilde{k} \in \mathbb{R}^d$, split into $d/2$ consecutive pairs:

$$r[j] = \sqrt{\tilde{k}[2j]^2 + \tilde{k}[2j+1]^2}, \quad j = 0, \ldots, d/2 - 1$$

$$\theta[j] = \text{atan2}(\tilde{k}[2j+1],\ \tilde{k}[2j]) \in (-\pi, \pi]$$

Reconstruction:
$$\tilde{k}[2j] = r[j]\cos\theta[j], \quad \tilde{k}[2j+1] = r[j]\sin\theta[j]$$

**Sanity check:** $r[j]^2 = \tilde{k}[2j]^2 + \tilde{k}[2j+1]^2$ ✓

### 2. Quantization

Radius quantization (n bits, per-channel scale, no zero-point because r ≥ 0):

$$s_j = \frac{\max_i r_i[j]}{2^n - 1}$$

$$Q(r[j]) = \text{clamp}\left(\text{round}\left(\frac{r[j]}{s_j}\right), 0, 2^n - 1\right)$$

$$\hat{r}[j] = Q(r[j]) \cdot s_j$$

Angle quantization (m bits, uniform over $(-\pi, \pi]$):

$$Q(\theta[j]) = \left\lfloor \frac{\theta[j] + \pi}{2\pi} \cdot 2^m \right\rfloor \mod 2^m$$

$$\hat{\theta}[j] = \frac{Q(\theta[j]) + 0.5}{2^m} \cdot 2\pi - \pi$$

Note: Using $+0.5$ centers the dequantized value in the bin, reducing average error.
The range is $(-\pi, \pi]$ matching atan2's output. Total states per pair: $2^{n+m}$.

**Sanity check:** At m=3 bits, you have 8 angular bins of width $2\pi/8 = 45°$ each. ✓

### 3. Attention Score via Polar Dot Product

$$\mathbf{q} \cdot \tilde{k}_i = \sum_{j=0}^{d/2-1}(q[2j]\tilde{k}_i[2j] + q[2j+1]\tilde{k}_i[2j+1])$$

Substituting polar reconstruction:

$$= \sum_{j=0}^{d/2-1} r_i[j] \cdot (q[2j]\cos\theta_i[j] + q[2j+1]\sin\theta_i[j])$$

$$= \sum_{j=0}^{d/2-1} r_i[j] \cdot |q_j| \cos(\theta_i[j] - \phi_j)$$

where $|q_j| = \sqrt{q[2j]^2+q[2j+1]^2}$ and $\phi_j = \text{atan2}(q[2j+1], q[2j])$.

The lookup table computes $(q[2j]\cos\hat{\theta}_i[j] + q[2j+1]\sin\hat{\theta}_i[j])$ for
each quantized key pair. In PolarQuant's implementation this is:

$$\text{LUT}[j][Q_r, Q_\theta] = \hat{r}[j] \cdot (\cos\hat{\theta}[j],\ \sin\hat{\theta}[j])$$

which is a 2-element vector. The inner product with $(q[2j], q[2j+1])$ gives the per-pair
contribution. This is precomputed offline and the attention forward pass becomes a sequence
of 2D dot products and scalar multiplications.

**Sanity check:** Full reconstruction recovers $q \cdot k$ exactly when (n, m) → ∞. ✓

### 4. Tier Structure

Capacity schedule: $C_\ell = 2^\ell$ for $\ell = 1, 2, \ldots, L$

Total capacity: $\sum_{\ell=1}^{L} 2^\ell = 2^{L+1} - 2$

For $T$ tokens, solve $2^{L+1} - 2 \geq T$:

$$L = \lceil \log_2(T + 2) \rceil - 1$$

For T=512: L=9 (total capacity = 1022). For T=1024: L=10 (total capacity = 2046). ✓

Precision schedule: $b_\ell = b_1 / 2^{\ell-1}$ where $b_1$ is the top-tier total bits.

Memory per tier:

$$M_\ell = C_\ell \cdot b_\ell \cdot \frac{d}{2} = 2^\ell \cdot \frac{b_1}{2^{\ell-1}} \cdot \frac{d}{2} = 2 b_1 \cdot \frac{d}{2} = b_1 d$$

This is constant across all tiers. ✓

The split of $b_\ell$ into $(n_\ell, m_\ell)$ pairs (radius bits, angle bits) at each tier:

| Tier | $b_\ell$ | $(n_\ell, m_\ell)$ | Note |
|------|---------|-------------------|------|
| 1    | 8       | (3, 5) or (4, 4)  | High angle precision matters more |
| 2    | 4       | (2, 2)            | Balanced |
| 3    | 2       | (1, 1)            | Minimum viable |
| 4    | 1       | — evict —         | 1-bit polar is not viable; evict instead |

**Important correction from earlier discussions:** At tier 4 (1 bit), store nothing — treat
as token eviction. The memory budget from this tier is reclaimed. This is not a failure of
the scheme; it means QPolarTier unifies quantization and eviction in one framework.

For the (n, m) split at 8 bits: angle precision is more important than radius precision
for attention score preservation (the score depends on cos(Δθ), and angular error is
multiplicative in the dot product). Use (3, 5) as the default top-tier split.

### 5. Prefill Importance Score

During prefill with $T_p$ tokens, the attention weight matrix
$A \in \mathbb{R}^{T_p \times T_p}$ is computed (causally masked):

$$A[t, i] = \text{softmax}\left(\frac{Q_{t} K^T}{\sqrt{d}}\right)_i, \quad i \leq t$$

Token importance score:

$$I_i = \frac{1}{T_p - i} \sum_{t=i}^{T_p-1} A[t, i]$$

(Normalized by how many queries can attend to token i due to causal masking.)

This is the average attention weight token i receives from all subsequent tokens in
the prefill window. High $I_i$ → high-importance → tier 1.

**Implementation note:** In PolarQuant's code (and all HuggingFace attention
implementations), $A$ is computed internally but not returned. You need to hook into
the attention module and save it during prefill. This is done via a forward hook.

---

## Implementation Phases

---

### Phase 0: Environment and Codebase Orientation (Day 1)

**Step 0.1 — Clone and set up**

```bash
git clone https://github.com/ericshwu/PolarQuant
cd PolarQuant
# Do NOT run init_env.sh if you don't have A100
# Instead, install dependencies manually:
pip install torch transformers datasets accelerate
pip install triton  # standard pip version, not their custom build
```

If Triton custom kernels fail on your GPU, set a flag in your config to fall back to
PyTorch. This is your escape hatch throughout development.

**Step 0.2 — Read these files carefully before writing any code:**

- `models/` — find the file that implements the modified attention forward pass.
  It will likely be named something like `modeling_llama_polar.py` or similar.
  Understand: where polar conversion happens, where the LUT is built, where
  attention scores are computed.
- `utils/` — find the quantization utilities. Look for functions named
  `polar_quantize`, `to_polar`, `from_polar`, or similar.

**Step 0.3 — Run PolarQuant baseline on your model**

Adapt `test4long.py` to use `Llama-3.2-1B-Instruct` instead of whatever model
they default to. Run a small subset of LongBench (e.g., just `triviaqa`, 20 samples).
Record the baseline perplexity/score. This is your comparison target.

---

### Phase 1: Validation Experiment — Does the Importance Signal Work? (Day 1-2)

**This is non-negotiable. Do this before building anything.**

The central empirical question: how well does the prefill-intercepted attention weight
$I_i$ predict which tokens will be important at decode time?

**Step 1.1 — Write `experiments/validate_importance.py`**

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Storage for intercepted attention weights
prefill_attn_weights = {}  # layer_idx -> [T_p, T_p] tensor
decode_attn_weights = {}   # layer_idx -> list of [T_d] tensors, one per decode step

def make_prefill_hook(layer_idx):
    def hook(module, input, output):
        # output is (attn_output, attn_weights, past_kv) in most implementations
        # attn_weights shape: [batch, heads, T_q, T_k]
        if output[1] is not None:
            # Average over heads, take mean column sum (importance per token)
            w = output[1][0]  # [heads, T_q, T_k]
            prefill_attn_weights[layer_idx] = w.mean(0)  # [T_q, T_k]
    return hook

# Register hooks on all attention layers
hooks = []
for i, layer in enumerate(model.model.layers):
    h = layer.self_attn.register_forward_hook(make_prefill_hook(i))
    hooks.append(h)

# Run prefill
text = "..." # use a ~200 token passage from LongBench
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs, output_attentions=True)

# For each layer, compute token importance from prefill
for layer_idx, w in prefill_attn_weights.items():
    T = w.shape[-1]
    # Causal mask correction: normalize by number of tokens that attend to each position
    attendable = torch.arange(1, T+1, dtype=torch.float32)
    importance = w.sum(0) / attendable  # [T]
    # ... store this

# Now run decode steps, capture attention weights
# Compare: does high prefill importance predict high decode-time attention?
# Metric: Spearman rank correlation between I_i and mean decode attention to token i
```

**What you're looking for:** Spearman rank correlation ≥ 0.7 across layers and heads.
If it's consistently below 0.5 on your test passages, you have a problem and need to
reconsider the importance signal before building the tier system.

**Step 1.2 — Also test per-layer variation**

Some layers will show high correlation (semantic layers), others low (positional/sink layers).
Note which layers show weak correlation — these may need special handling
(e.g., fixed uniform precision in those layers).

Remove hooks after this experiment:
```python
for h in hooks:
    h.remove()
```

---

### Phase 2: Core Data Structures (Day 2-3)

Create a new directory: `qpolartier/`

**Step 2.1 — `qpolartier/config.py`**

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class QPolarTierConfig:
    # Tier structure
    tier_base: int = 2           # geometric base for capacity schedule
    top_tier_bits: int = 8       # b_1: total bits for tier-1 tokens
    min_viable_bits: int = 2     # below this, evict instead of quantize
    
    # Per-tier (n, m) split: radius bits, angle bits
    # Automatically derived from top_tier_bits but can be overridden
    nm_splits: List[Tuple[int,int]] = None  # [(3,5), (2,2), (1,1)]
    
    # Importance signal
    use_prefill_interception: bool = True
    prefill_importance_ema: float = 0.0   # 0 = no EMA, use raw prefill weights
    
    # Frequency-aware within-tier allocation (optional, Phase 4 extension)
    frequency_aware: bool = False
    
    def get_nm_split(self, tier: int) -> Tuple[int, int]:
        """Return (n_bits, m_bits) for radius and angle at given tier."""
        if self.nm_splits is not None:
            return self.nm_splits[tier - 1]
        # Default: allocate slightly more bits to angle than radius
        # because cos(Δθ) sensitivity is higher
        total = self.top_tier_bits // (2 ** (tier - 1))
        if total <= 1:
            return (0, 0)  # evict
        n = total // 3        # radius gets 1/3
        m = total - n         # angle gets 2/3
        return (max(1, n), max(1, m))
    
    def tier_capacity(self, tier: int) -> int:
        return self.tier_base ** tier
    
    def num_tiers(self, T: int) -> int:
        """Number of tiers needed to cover T tokens."""
        import math
        return math.ceil(math.log(T + 2, self.tier_base)) - 1 + 1
```

**Step 2.2 — `qpolartier/quantize.py`**

This is the core mathematical module. Implement it in pure PyTorch first.

```python
import torch
import math
from typing import Tuple

def to_polar(k_pair: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Cartesian key pair to polar.
    
    Args:
        k_pair: [batch_or_tokens, d/2, 2] — paired key dimensions
    Returns:
        r: [batch_or_tokens, d/2] — radii, non-negative
        theta: [batch_or_tokens, d/2] — angles in (-pi, pi]
    """
    x, y = k_pair[..., 0], k_pair[..., 1]
    r = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)  # returns (-pi, pi]
    return r, theta

def from_polar(r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Convert polar back to Cartesian.
    Returns: [..., d/2, 2]
    """
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=-1)

def quantize_radius(r: torch.Tensor, n_bits: int, 
                    scale: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize radius to n_bits.
    
    Args:
        r: [T, d/2] radii
        n_bits: number of bits
        scale: [d/2] per-channel scale; computed from r if None
    Returns:
        r_q: [T, d/2] quantized indices (integer)
        scale: [d/2] scale factors (save these — needed for dequant)
    """
    if n_bits == 0:
        return None, None
    max_val = 2**n_bits - 1
    if scale is None:
        # Per-channel: max radius across all tokens for each pair j
        scale = r.max(dim=0).values / max_val  # [d/2]
        scale = scale.clamp(min=1e-8)
    r_q = (r / scale).round().clamp(0, max_val).to(torch.int32)
    return r_q, scale

def dequantize_radius(r_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize radius. r_q: [T, d/2], scale: [d/2]"""
    return r_q.float() * scale

def quantize_angle(theta: torch.Tensor, m_bits: int) -> torch.Tensor:
    """
    Quantize angle in (-pi, pi] to m_bits.
    Uses uniform bins centered on bin midpoints.
    
    Args:
        theta: [T, d/2] angles in (-pi, pi]
        m_bits: number of bits
    Returns:
        theta_q: [T, d/2] quantized indices (integer, 0 to 2^m - 1)
    """
    if m_bits == 0:
        return None
    n_bins = 2**m_bits
    # Map (-pi, pi] -> [0, 1) then to [0, n_bins)
    theta_normalized = (theta + math.pi) / (2 * math.pi)  # [0, 1)
    theta_q = torch.floor(theta_normalized * n_bins).long()
    theta_q = theta_q % n_bins  # handle edge case at exactly pi
    return theta_q

def dequantize_angle(theta_q: torch.Tensor, m_bits: int) -> torch.Tensor:
    """Dequantize angle. Returns value at bin center."""
    n_bins = 2**m_bits
    # Bin center: (q + 0.5) / n_bins maps back to [0, 1), then to (-pi, pi]
    theta_norm = (theta_q.float() + 0.5) / n_bins
    return theta_norm * 2 * math.pi - math.pi

def reshape_key_to_pairs(k: torch.Tensor) -> torch.Tensor:
    """
    Reshape key from [T, d] to [T, d/2, 2].
    This groups RoPE-coupled dimension pairs.
    """
    T, d = k.shape
    assert d % 2 == 0, f"Head dim must be even, got {d}"
    return k.view(T, d//2, 2)

def polar_dot_product(q: torch.Tensor, 
                      r_q: torch.Tensor, theta_q: torch.Tensor,
                      r_scale: torch.Tensor,
                      n_bits: int, m_bits: int) -> torch.Tensor:
    """
    Compute q · k_i for all cached tokens i using quantized polar representation.
    No explicit dequantization to Cartesian.
    
    Args:
        q: [d] current query vector
        r_q: [T, d/2] quantized radii (integer indices)
        theta_q: [T, d/2] quantized angles (integer indices)
        r_scale: [d/2] radius scale factors
        n_bits, m_bits: precision
    Returns:
        scores: [T] attention scores (unnormalized, before /sqrt(d))
    """
    d = q.shape[0]
    # Reshape query to pairs
    q_pairs = q.view(d//2, 2)  # [d/2, 2]
    
    # Dequantize keys to 2D vectors
    r_hat = dequantize_radius(r_q, r_scale)          # [T, d/2]
    theta_hat = dequantize_angle(theta_q, m_bits)     # [T, d/2]
    
    # k_hat[i, j, :] = (r_hat[i,j]*cos(theta_hat[i,j]), r_hat[i,j]*sin(theta_hat[i,j]))
    k_x = r_hat * torch.cos(theta_hat)  # [T, d/2]
    k_y = r_hat * torch.sin(theta_hat)  # [T, d/2]
    
    # Inner product: sum over j of (q[2j]*k_x[i,j] + q[2j+1]*k_y[i,j])
    scores = (k_x * q_pairs[:, 0].unsqueeze(0) +
              k_y * q_pairs[:, 1].unsqueeze(0)).sum(dim=1)  # [T]
    return scores
```

**Mathematical verification of `polar_dot_product`:**

Let $q_j = (q[2j], q[2j+1])$ and $\hat{k}_j^{(i)} = (\hat{r}_i[j]\cos\hat{\theta}_i[j],\ \hat{r}_i[j]\sin\hat{\theta}_i[j])$.

Then: $q \cdot \hat{k}_i = \sum_j q_j \cdot \hat{k}_j^{(i)}$
$= \sum_j (q[2j]\hat{r}_i[j]\cos\hat{\theta}_i[j] + q[2j+1]\hat{r}_i[j]\sin\hat{\theta}_i[j])$

This is exactly what the code computes via `k_x * q_pairs[:,0] + k_y * q_pairs[:,1]`. ✓

---

**Step 2.3 — `qpolartier/tier_manager.py`**

```python
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from .config import QPolarTierConfig
from .quantize import (to_polar, quantize_radius, quantize_angle, 
                        reshape_key_to_pairs)

@dataclass
class TieredKeyCache:
    """Stores quantized key vectors organized by tier."""
    # Per tier: list indexed by tier number (1-based)
    r_q: List[Optional[torch.Tensor]]      # [T_ell, d/2] int tensors
    theta_q: List[Optional[torch.Tensor]]  # [T_ell, d/2] int tensors
    r_scale: List[Optional[torch.Tensor]]  # [d/2] float tensors
    n_bits: List[int]                       # per-tier radius bits
    m_bits: List[int]                       # per-tier angle bits
    token_indices: List[List[int]]          # original token indices per tier
    evicted: List[int]                      # indices of evicted tokens

class TierManager:
    def __init__(self, config: QPolarTierConfig):
        self.config = config
    
    def assign_tiers(self, importance: torch.Tensor) -> List[int]:
        """
        Given importance scores [T], return tier assignment [T].
        Tier 1 = most important, tier L = least important, tier 0 = evict.
        
        Args:
            importance: [T] scores, higher = more important
        Returns:
            assignments: [T] tier indices (1-based, 0 = evict)
        """
        T = importance.shape[0]
        config = self.config
        
        # Sort descending: highest importance gets tier 1
        sorted_idx = torch.argsort(importance, descending=True)
        assignments = torch.zeros(T, dtype=torch.long)
        
        filled = 0
        tier = 1
        while filled < T:
            capacity = config.tier_capacity(tier)
            end = min(filled + capacity, T)
            
            n, m = config.get_nm_split(tier)
            if n == 0 or m == 0:
                # Below min viable bits: evict
                assignments[sorted_idx[filled:end]] = 0
            else:
                assignments[sorted_idx[filled:end]] = tier
            
            filled = end
            tier += 1
            if filled >= T:
                break
        
        return assignments
    
    def quantize_keys(self, k: torch.Tensor, 
                       assignments: torch.Tensor) -> TieredKeyCache:
        """
        Quantize all key vectors according to tier assignments.
        
        Args:
            k: [T, d] post-RoPE key vectors (float16)
            assignments: [T] tier assignments (0 = evict)
        Returns:
            TieredKeyCache
        """
        T, d = k.shape
        k_pairs = reshape_key_to_pairs(k)  # [T, d/2, 2]
        r, theta = to_polar(k_pairs)        # [T, d/2] each
        
        max_tier = assignments.max().item()
        cache = TieredKeyCache(
            r_q=[None] * (max_tier + 1),
            theta_q=[None] * (max_tier + 1),
            r_scale=[None] * (max_tier + 1),
            n_bits=[0] * (max_tier + 1),
            m_bits=[0] * (max_tier + 1),
            token_indices=[[] for _ in range(max_tier + 1)],
            evicted=[]
        )
        
        for tier in range(1, max_tier + 1):
            mask = (assignments == tier)
            if not mask.any():
                continue
            
            n, m = self.config.get_nm_split(tier)
            tier_r = r[mask]       # [T_ell, d/2]
            tier_theta = theta[mask]  # [T_ell, d/2]
            
            r_q, r_scale = quantize_radius(tier_r, n)
            theta_q = quantize_angle(tier_theta, m)
            
            cache.r_q[tier] = r_q
            cache.theta_q[tier] = theta_q
            cache.r_scale[tier] = r_scale
            cache.n_bits[tier] = n
            cache.m_bits[tier] = m
            cache.token_indices[tier] = mask.nonzero(as_tuple=True)[0].tolist()
        
        cache.evicted = (assignments == 0).nonzero(as_tuple=True)[0].tolist()
        return cache
```

---

### Phase 3: Modified Attention Module (Day 3-4)

Create `qpolartier/attention.py`. This replaces PolarQuant's attention module with one
that also handles prefill weight interception and tier assignment.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import QPolarTierConfig
from .tier_manager import TierManager, TieredKeyCache
from .quantize import polar_dot_product

class QPolarTierAttention(nn.Module):
    """
    Drop-in replacement for LlamaAttention that uses QPolarTier
    KV cache quantization.
    
    Usage: replace model.model.layers[i].self_attn with this class.
    """
    
    def __init__(self, original_attn: nn.Module, config: QPolarTierConfig):
        super().__init__()
        # Copy projection weights from original attention
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        self.rotary_emb = original_attn.rotary_emb
        
        # Copy architecture config
        self.num_heads = original_attn.num_heads
        self.num_kv_heads = original_attn.num_key_value_heads
        self.head_dim = original_attn.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        # QPolarTier components
        self.tier_config = config
        self.tier_manager = TierManager(config)
        
        # Cache (populated during prefill, used during decode)
        self._tiered_key_cache: Optional[TieredKeyCache] = None
        self._value_cache: Optional[torch.Tensor] = None  # [T, num_kv_heads, d]
        self._prefill_importance: Optional[torch.Tensor] = None
        self._is_prefill: bool = True
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = True,
                **kwargs):
        
        B, T_q, C = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)   # [B, T_q, num_heads * head_dim]
        k = self.k_proj(hidden_states)   # [B, T_q, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)   # [B, T_q, num_kv_heads * head_dim]
        
        # Reshape to [B, heads, T_q, head_dim]
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if self._is_prefill or T_q > 1:
            # PREFILL PATH
            output, attn_weights = self._prefill_forward(q, k, v, attention_mask)
            # Save importance from prefill attention weights
            # attn_weights: [B, num_heads, T_q, T_q]
            if attn_weights is not None:
                self._store_prefill_importance(attn_weights, k, v)
            self._is_prefill = False
        else:
            # DECODE PATH — single token
            output = self._decode_forward(q, k, v)
        
        output = output.transpose(1, 2).contiguous().view(B, T_q, -1)
        output = self.o_proj(output)
        return output, None, None  # match HuggingFace interface
    
    def _prefill_forward(self, q, k, v, attention_mask):
        """Standard full-precision attention during prefill. Intercepts weights."""
        # GQA: repeat KV heads to match Q heads
        k_exp = k.repeat_interleave(self.num_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(attn_weights, v_exp)
        return output, attn_weights
    
    def _store_prefill_importance(self, attn_weights, k, v):
        """
        Compute and store per-token importance from prefill attention weights.
        Also initializes the tiered key cache.
        
        attn_weights: [B, num_heads, T_p, T_p]
        k: [B, num_kv_heads, T_p, head_dim] (post-RoPE)
        v: [B, num_kv_heads, T_p, head_dim]
        """
        B, H, T_p, _ = attn_weights.shape
        
        # Compute importance: mean attention received, causal-normalized
        # attn_weights[b, h, t, i] = attention from position t to position i
        # Sum over query positions t for each key position i
        # Normalize by (T_p - i) since only T_p - i queries can attend to token i
        
        # Sum over heads and batch, then normalize by attendable count
        importance_raw = attn_weights[0].mean(0)  # [T_p, T_p], mean over heads
        
        # Column sum (how much each token i was attended to)
        col_sums = importance_raw.sum(0)  # [T_p]
        
        # Normalize by attendable count (causal: token i attended to by T_p - i queries)
        attendable = torch.arange(1, T_p + 1, device=col_sums.device, 
                                   dtype=torch.float32)
        importance = col_sums / attendable  # [T_p]
        self._prefill_importance = importance
        
        # Assign tiers and build tiered cache
        assignments = self.tier_manager.assign_tiers(importance)
        
        # Flatten KV for caching: take first batch, process each KV head
        # [num_kv_heads, T_p, head_dim] -> tiered cache per head
        k_flat = k[0].transpose(0, 1)  # [T_p, num_kv_heads, head_dim]
        v_flat = v[0].transpose(0, 1)  # [T_p, num_kv_heads, head_dim]
        
        # Build tiered key cache (one per KV head)
        self._tiered_key_cache = []
        for h in range(self.num_kv_heads):
            k_h = k_flat[:, h, :]  # [T_p, head_dim]
            tier_cache = self.tier_manager.quantize_keys(k_h, assignments)
            self._tiered_key_cache.append(tier_cache)
        
        # Store values at full precision for now (value quantization is separate)
        self._value_cache = v_flat  # [T_p, num_kv_heads, head_dim]
        self._cached_assignments = assignments
        self._cached_T = T_p
    
    def _decode_forward(self, q, k_new, v_new):
        """
        Decode step: compute attention using tiered key cache.
        q: [B, num_heads, 1, head_dim]
        k_new, v_new: [B, num_kv_heads, 1, head_dim] — new token's KV
        """
        B = q.shape[0]
        T_cached = self._cached_T
        
        # For each head, compute attention scores against tiered cache
        # Then compute weighted sum of values
        
        all_scores = []
        
        for q_head_idx in range(self.num_heads):
            kv_head_idx = q_head_idx // self.num_kv_groups
            q_h = q[0, q_head_idx, 0, :]  # [head_dim]
            
            tier_cache = self._tiered_key_cache[kv_head_idx]
            
            # Compute scores for each tier
            scores_by_token = torch.zeros(T_cached, device=q.device, dtype=q.dtype)
            
            for tier in range(1, len(tier_cache.r_q)):
                if tier_cache.r_q[tier] is None:
                    continue
                n, m = tier_cache.n_bits[tier], tier_cache.m_bits[tier]
                tier_scores = polar_dot_product(
                    q_h,
                    tier_cache.r_q[tier].to(q.device),
                    tier_cache.theta_q[tier].to(q.device),
                    tier_cache.r_scale[tier].to(q.device),
                    n, m
                )  # [T_ell]
                
                # Place scores back at original token positions
                indices = torch.tensor(tier_cache.token_indices[tier], 
                                        device=q.device)
                scores_by_token.scatter_(0, indices, tier_scores)
            
            # Evicted tokens get -inf (excluded from softmax)
            if tier_cache.evicted:
                evict_idx = torch.tensor(tier_cache.evicted, device=q.device)
                scores_by_token.scatter_(0, evict_idx, 
                                          torch.full_like(evict_idx, float('-inf'), 
                                                          dtype=torch.float32))
            
            # Scale and softmax
            scores_by_token = scores_by_token / (self.head_dim ** 0.5)
            attn_weights_h = F.softmax(scores_by_token, dim=0)  # [T_cached]
            
            # Weighted sum of values
            # self._value_cache: [T_cached, num_kv_heads, head_dim]
            v_h = self._value_cache[:, kv_head_idx, :]  # [T_cached, head_dim]
            output_h = (attn_weights_h.unsqueeze(-1) * v_h).sum(0)  # [head_dim]
            all_scores.append(output_h)
        
        # Stack: [num_heads, head_dim] -> [B, num_heads, 1, head_dim]
        output = torch.stack(all_scores, dim=0).unsqueeze(0).unsqueeze(2)
        return output
```

**Note on GQA (Grouped Query Attention):** Llama-3.2-1B uses GQA with 32 Q heads and
8 KV heads. The tier cache is built per KV head (8 caches). During decode, Q head `h`
uses KV head `h // 4`. The code above handles this via `kv_head_idx = q_head_idx // self.num_kv_groups`.

---

### Phase 4: Model Patching (Day 4)

Create `qpolartier/patch.py`:

```python
import torch
from transformers import AutoModelForCausalLM
from .config import QPolarTierConfig
from .attention import QPolarTierAttention

def patch_model(model, config: QPolarTierConfig):
    """
    Replace all attention modules in the model with QPolarTierAttention.
    Modifies model in-place. Returns model.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        original_attn = layer.self_attn
        new_attn = QPolarTierAttention(original_attn, config)
        layer.self_attn = new_attn
        print(f"Patched layer {layer_idx}")
    return model

def reset_caches(model):
    """Reset all tier caches (call between independent inference runs)."""
    for layer in model.model.layers:
        if isinstance(layer.self_attn, QPolarTierAttention):
            layer.self_attn._tiered_key_cache = None
            layer.self_attn._value_cache = None
            layer.self_attn._prefill_importance = None
            layer.self_attn._is_prefill = True
```

---

### Phase 5: Evaluation Script (Day 5)

Create `qpolartier/evaluate.py`:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .config import QPolarTierConfig
from .patch import patch_model, reset_caches

def evaluate_perplexity(model, tokenizer, texts, max_length=512):
    """
    Compute perplexity on a list of texts.
    Lower = better. Baseline (fp16) is your reference.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", 
                                max_length=max_length, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Prefill
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            n_tokens = inputs["input_ids"].shape[1]
            
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
            reset_caches(model)
    
    import math
    return math.exp(total_loss / total_tokens)

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model.cuda().eval()
    
    # Baseline
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in dataset if len(x["text"]) > 100][:50]
    
    baseline_ppl = evaluate_perplexity(model, tokenizer, texts)
    print(f"Baseline (fp16) PPL: {baseline_ppl:.2f}")
    
    # QPolarTier variants
    configs = [
        QPolarTierConfig(top_tier_bits=8),   # b1=8: tiers (3,5), (2,2), evict
        QPolarTierConfig(top_tier_bits=6),   # b1=6: tiers (2,4), (1,2), evict
    ]
    
    for cfg in configs:
        patch_model(model, cfg)
        qpt_ppl = evaluate_perplexity(model, tokenizer, texts)
        print(f"QPolarTier b1={cfg.top_tier_bits} PPL: {qpt_ppl:.2f}  "
              f"(+{qpt_ppl - baseline_ppl:.2f})")
```

---

### Phase 6: Ablations (Day 6-7)

Run these experiments in order. Each answers one of the open questions.

**Ablation 1: Tier structure baseline**
- Compare uniform PolarQuant (all tokens same bits) vs QPolarTier
- Expected: QPolarTier matches or beats PolarQuant at lower average bits per token

**Ablation 2: Importance signal**
- Compare prefill-interception importance vs random tier assignment vs norm-based importance
- Expected: prefill-interception > norm-based > random

**Ablation 3: Geometric base**
- Try base-2 (default), base-3, and base-4 capacity schedules
- Expected: base-2 may be too aggressive for short contexts

**Ablation 4: (n, m) split**
- For top tier (8 bits total): compare (4,4), (3,5), (2,6)
- Expected: more angle bits helps at high bit count

**Ablation 5: Eviction threshold**
- Compare evicting at 2-bit floor vs 1-bit floor
- Expected: eviction at 2-bit is better than trying to use 1-bit polar

---

## Important Implementation Warnings

### Warning 1: `apply_rotary_pos_emb` Signature

In HuggingFace's Llama implementation, `apply_rotary_pos_emb` may have different
signatures across versions. Check the exact function signature in the version you install:

```python
# In some versions:
q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
# In others:
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

Import it from the same file as the original Llama implementation:
```python
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
```

### Warning 2: Scale Computation Scope

The radius scale `s_j = max_i(r_i[j]) / (2^n - 1)` is computed across ALL tokens
at prefill time. This is correct and necessary — you need a global scale so that
quantized values from different tiers are comparable when assembling scores. Do NOT
compute per-tier scales independently.

### Warning 3: Angle Range Convention

`torch.atan2` returns values in $(-\pi, \pi]$. The quantization formula maps this range
to $[0, 2^m)$. Verify this is consistent throughout: `to_polar`, `quantize_angle`, and
`dequantize_angle` must all agree on the $(-\pi, \pi]$ convention. A mismatch here
will produce silent errors (wrong scores that still look plausible).

### Warning 4: GQA Token Position Tracking

When you have 8 KV heads but 32 Q heads, the tier assignment (which tokens go to which
tier) is **shared across all Q heads that share a KV head**. The `assignments` tensor
is one-dimensional over tokens, not per-head. This is correct — tier assignment is
about the token's importance, not the head.

### Warning 5: fp16 Numerical Stability

Do precision-sensitive operations (softmax, importance score normalization) in fp32,
then cast back to fp16. Specifically:
```python
# Use float32 for softmax to avoid NaN/Inf
attn_weights = F.softmax(scores_by_token.float(), dim=0).to(q.dtype)
```

---

## Expected Results and Calibration

On `Llama-3.2-1B-Instruct` with WikiText-2:

| Method | Avg bits/token | Expected PPL delta vs fp16 |
|--------|---------------|---------------------------|
| fp16 baseline | 16 | 0 |
| PolarQuant (4+2 bits) | ~4.16 | < 0.5 |
| QPolarTier b1=8 | ~1.5–2.5 avg | < 1.5 (estimate) |
| QPolarTier b1=6 | ~1.0–2.0 avg | < 3.0 (estimate) |

If your PPL delta is > 5 at b1=8, something is wrong. Debug in this order:
1. Check that prefill full-precision path gives same PPL as unmodified model (it must)
2. Check that decode scores match within 1% of full-precision scores on a single step
3. Check angle convention consistency
4. Check GQA head mapping

---

## File Structure Summary

```
PolarQuant/
├── qpolartier/
│   ├── __init__.py
│   ├── config.py          ← QPolarTierConfig
│   ├── quantize.py        ← to_polar, quantize_*, polar_dot_product
│   ├── tier_manager.py    ← TierManager, TieredKeyCache
│   ├── attention.py       ← QPolarTierAttention
│   └── patch.py           ← patch_model, reset_caches
├── experiments/
│   ├── validate_importance.py   ← Phase 1 validation (run first)
│   └── evaluate.py              ← Perplexity evaluation
├── models/                      ← Original PolarQuant code (don't modify)
├── utils/                       ← Original PolarQuant utilities
└── test_qpolartier.py           ← Quick smoke test
```

---

*Implementation plan complete. Start with Phase 1 (validate_importance.py) before
writing any quantization code. The entire plan is contingent on the importance
signal being empirically valid.*
