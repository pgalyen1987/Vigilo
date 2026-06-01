"""
Model configuration for CodingSSM.

Three preset configs are provided:
  ModelConfigCPU()  — ~150M params, fits on CPU with 4 GB RAM
  ModelConfig700M() — ~700M params, for architecture validation on Kaggle T4
  ModelConfig3B()   — ~6.38B total / ~3B active params, primary training target

All three share the same architecture (Mamba-2 + sparse attention + MoE + shared
attention weights + LoRA); they differ only in hidden dimensions and expert counts.
"""
from dataclasses import dataclass, field
from typing import Optional


def ModelConfigTiny() -> "ModelConfig":
    """
    ~20M parameter config for local smoke / integration tests.
    Runs a full SFT+GRPO cycle in under 5 minutes on CPU.
    Same architecture shape as 700M — validates pipeline end-to-end.
    """
    return ModelConfig(
        d_model=256, d_inner=512, n_layers=6,
        n_heads=8, d_head=64, d_state=16, n_groups=2, chunk_size=64,
        n_attn_heads=4, d_attn_head=64, attn_every_n=3, attn_window=128,
        n_shared_attn=1, lora_rank=8,
        n_experts=4, max_active_experts=2, min_active_experts=1, d_ffn=512,
        vocab_size=152064, max_seq_len=512,
        recursion_depth=1,
    )


def ModelConfigCPU() -> "ModelConfig":
    """
    ~150M parameter config for CPU training.
    Fits in ~4 GB RAM (fp32 weights + gradients + Adafactor).
    Same architecture as 700M — just smaller dims.
    Trains at ~1-3 min/step on CPU; full SFT in a few hours.
    """
    return ModelConfig(
        d_model=384,
        d_inner=768,
        n_layers=12,
        n_heads=12,
        d_head=64,
        d_state=32,
        n_groups=2,
        chunk_size=128,
        n_attn_heads=6,
        d_attn_head=64,
        attn_every_n=4,
        attn_window=256,
        n_shared_attn=1,
        lora_rank=16,
        n_experts=4,
        max_active_experts=2,
        min_active_experts=1,
        d_ffn=1536,
        vocab_size=152064,
        max_seq_len=4096,
        recursion_depth=1,   # no recursion on CPU — saves 2x activation memory
    )


def ModelConfig700M() -> "ModelConfig":
    """
    700M parameter config for architecture bringup (Stage 0).
    Fits comfortably in ~14GB RAM (weights + Adafactor + activations).
    Same architecture as 3B — just smaller dims.

    Memory estimate:
      weights:    700M * 4B = 2.8GB
      grads:      2.8GB
      adafactor:  ~0.7GB (factored)
      activations (seq512, grad_ckpt): ~2GB
      total: ~9GB  ← fits with VS Code + browser on 64GB
    """
    return ModelConfig(
        d_model=1024,
        d_inner=2048,
        n_layers=24,
        n_heads=32,
        d_head=64,
        d_state=64,
        n_groups=4,
        chunk_size=256,
        n_attn_heads=16,
        d_attn_head=64,
        attn_every_n=4,
        attn_window=512,
        n_shared_attn=2,
        lora_rank=32,
        n_experts=8,
        max_active_experts=4,
        min_active_experts=1,
        d_ffn=4096,
        vocab_size=152064,
        max_seq_len=32768,
        recursion_depth=2,   # TRM-style depth=2; pair with seq_len=256 in SFT to keep activations in budget
    )


def ModelConfig3B() -> "ModelConfig":
    """Full 3B parameter config. Use after Stage 0 validates the architecture."""
    return ModelConfig()  # all defaults are 3B


@dataclass
class ModelConfig:
    """
    Full architecture hyperparameters for CodingSSM.

    Fields are grouped by subsystem. All three preset configs (CPU / 700M / 3B)
    instantiate this class with different values; the defaults here are the 3B config.
    """
    # Core dimensions
    d_model: int = 2048
    d_inner: int = 4096       # expand = 2 * d_model
    n_layers: int = 24

    # Mamba-2 / SSD
    n_heads: int = 64         # d_inner / d_head
    d_head: int = 64
    d_state: int = 128        # SSM state size N
    n_groups: int = 8         # number of SSM groups
    chunk_size: int = 256     # SSD chunk size

    # Sparse attention
    attn_every_n: int = 6     # 1 attention layer every N layers
    attn_window: int = 512    # sliding window size (tokens)
    n_attn_heads: int = 32
    d_attn_head: int = 64     # d_model / n_attn_heads

    # Shared attention + LoRA (Zamba2-style)
    n_shared_attn: int = 2    # number of shared attention weight sets (ABAB...)
    lora_rank: int = 64       # per-layer LoRA adapter rank for attention

    # MoE FFN
    n_experts: int = 8
    max_active_experts: int = 4  # dynamic top-k cap
    min_active_experts: int = 1  # dynamic top-k floor
    d_ffn: int = 8192         # FFN hidden dim (4 * d_model)
    moe_on_even_layers: bool = True  # MoE on even layers, dense FFN on odd

    # Vocabulary & sequence
    vocab_size: int = 152064  # Qwen2.5 tokenizer
    max_seq_len: int = 32768
    pad_token_id: int = 0

    # Recursive reasoning (TRM-style: apply layer stack N times, zero extra params)
    # depth=1 → standard single-pass (default); depth=2+ → recursive inference
    # Increases effective compute depth without adding parameters.
    recursion_depth: int = 1

    # Training
    dropout: float = 0.0
    tie_embeddings: bool = True

    # Distillation
    kd_alpha: float = 0.7     # weight on reverse KLD
    kd_beta: float = 0.3      # weight on CE loss
    kd_temperature: float = 1.0

    def attn_layer_indices(self) -> list[int]:
        """Return 0-based layer indices that use sparse attention instead of Mamba-2."""
        return [i for i in range(self.n_layers) if (i + 1) % self.attn_every_n == 0]

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return True if this layer uses MoE FFN (even layers by default)."""
        return self.moe_on_even_layers and (layer_idx % 2 == 0)

    def expert_budget(self, layer_idx: int) -> int:
        """
        Return the number of active experts for this layer.

        Uses a descending capacity schedule: earlier layers activate more experts
        (up to max_active_experts) and later layers fewer (down to min_active_experts).
        This allocates more compute to early layers where token routing is most uncertain.
        """
        frac = layer_idx / max(self.n_layers - 1, 1)
        budget = self.max_active_experts - frac * (self.max_active_experts - self.min_active_experts)
        return max(self.min_active_experts, round(budget))
