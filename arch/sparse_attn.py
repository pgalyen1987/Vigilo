"""
Sparse local (sliding window) attention with shared weights + per-layer LoRA.

Design (Zamba2-style, adapted):
  - n_shared_attn=2 sets of attention weights (ABAB... interleaving)
  - Each layer that uses attention gets its own LoRA adapter (rank=64)
    applied on top of the shared Q/K/V/O projections
  - Attention is LOCAL: each token attends only to its window of `attn_window`
    past tokens (no global attention — preserves Mamba-2 state advantage)

Window masking:
  For position t, valid keys are [t - window + 1, t] (inclusive).
  Implemented via additive mask: -inf outside window.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mamba2 import RMSNorm


class LoRALinear(nn.Module):
    """Linear layer with a LoRA adapter: y = W*x + (B@A)*x * scale."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.rank = rank
        # Shared weights (frozen during LoRA-only training)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA adapter
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = 1.0 / rank

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias_param)
        out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return out

    def freeze_base(self):
        self.weight.requires_grad_(False)
        if self.bias_param is not None:
            self.bias_param.requires_grad_(False)

    def unfreeze_base(self):
        self.weight.requires_grad_(True)
        if self.bias_param is not None:
            self.bias_param.requires_grad_(True)


class SharedAttentionWeights(nn.Module):
    """
    Shared QKV + output projection weights (used by multiple attention layers via LoRA).
    This module holds only the base weights; each layer wraps them with LoRA adapters.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, lora_rank: int = 0):
        super().__init__()
        d_qkv = n_heads * d_head
        self.n_heads = n_heads
        self.d_head = d_head

        # Plain linear — per-layer LoRA adapters live in SharedAttentionBlock, not here.
        # (Previously used LoRALinear but those lora_A/B were never reached in _lora_proj.)
        self.q_proj = nn.Linear(d_model, d_qkv, bias=False)
        self.k_proj = nn.Linear(d_model, d_qkv, bias=False)
        self.v_proj = nn.Linear(d_model, d_qkv, bias=False)
        self.o_proj = nn.Linear(d_qkv, d_model, bias=False)


def build_sliding_window_mask(seq_len: int, window: int, device: torch.device) -> torch.Tensor:
    """
    Returns additive attention mask (0 for valid, -inf for masked).
    Shape: (seq_len, seq_len).
    token t can attend to tokens s where s in [t-window+1, t].
    """
    # causal mask
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, L)
    # valid: j <= i and j >= i - window + 1
    valid = (j <= i) & (j >= i - window + 1)
    mask = torch.where(valid, torch.zeros(1, device=device), torch.full((1,), float('-inf'), device=device))
    return mask   # (L, L)


class SparseAttention(nn.Module):
    """
    Sliding window self-attention layer using shared weights + LoRA.

    Each instance is tied to a particular SharedAttentionWeights object
    (identified by shared_idx) but has its OWN LoRA adapters.
    """

    def __init__(self, config: ModelConfig, shared_weights: SharedAttentionWeights, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        d = config.d_model
        self.n_heads = config.n_attn_heads
        self.d_head = config.d_attn_head
        self.window = config.attn_window

        # Layer norm before attention
        self.norm = RMSNorm(d)

        # Reference shared weights (not owned by this module — don't double-count params)
        self.shared = shared_weights

        # Per-layer LoRA adapters on top of shared weights
        # We add separate LoRA A/B pairs per layer (the shared weight's lora_A/B are
        # replaced per-layer by registering them as new params here)
        lora_r = config.lora_rank
        d_qkv = self.n_heads * self.d_head

        self.q_lora_A = nn.Parameter(torch.randn(lora_r, d) * 0.02)
        self.q_lora_B = nn.Parameter(torch.zeros(d_qkv, lora_r))
        self.k_lora_A = nn.Parameter(torch.randn(lora_r, d) * 0.02)
        self.k_lora_B = nn.Parameter(torch.zeros(d_qkv, lora_r))
        self.v_lora_A = nn.Parameter(torch.randn(lora_r, d) * 0.02)
        self.v_lora_B = nn.Parameter(torch.zeros(d_qkv, lora_r))
        self.o_lora_A = nn.Parameter(torch.randn(lora_r, d_qkv) * 0.02)
        self.o_lora_B = nn.Parameter(torch.zeros(d, lora_r))
        self.lora_scale = 1.0 / lora_r

        # RoPE frequencies
        self._rope_freqs = None

    def _lora_proj(self, x, base_weight, lora_A, lora_B):
        """Apply base weight + per-layer LoRA."""
        base_out = F.linear(x, base_weight)
        lora_out = F.linear(F.linear(x, lora_A), lora_B) * self.lora_scale
        return base_out + lora_out

    def _get_rope(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Rotary position embeddings, cached."""
        if self._rope_freqs is not None and self._rope_freqs.shape[0] >= seq_len:
            return self._rope_freqs[:seq_len].to(device)

        d = self.d_head
        half = d // 2
        theta = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, theta)       # (L, d/2)
        emb = torch.cat([freqs, freqs], dim=-1)     # (L, d)
        self._rope_freqs = emb
        return emb

    @staticmethod
    def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_heads, L, d_head)
        freqs: (L, d_head)
        """
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, L, d)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        d = x.shape[-1]
        half = d // 2
        x1, x2 = x[..., :half], x[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos + rotated * sin

    def forward(
        self,
        x: torch.Tensor,              # (B, L, d_model)
        kv_cache: dict = None,        # {'k': (B, H, past, dh), 'v': ...} for generation
    ) -> tuple[torch.Tensor, dict]:
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)

        # --- Projections with per-layer LoRA ---
        Q = self._lora_proj(x, self.shared.q_proj.weight, self.q_lora_A, self.q_lora_B)
        K = self._lora_proj(x, self.shared.k_proj.weight, self.k_lora_A, self.k_lora_B)
        V = self._lora_proj(x, self.shared.v_proj.weight, self.v_lora_A, self.v_lora_B)

        n_heads, d_head = self.n_heads, self.d_head
        Q = Q.view(B, L, n_heads, d_head).permute(0, 2, 1, 3)  # (B, H, L, dh)
        K = K.view(B, L, n_heads, d_head).permute(0, 2, 1, 3)
        V = V.view(B, L, n_heads, d_head).permute(0, 2, 1, 3)

        # --- RoPE ---
        past_len = kv_cache['k'].shape[2] if kv_cache else 0
        freqs = self._get_rope(past_len + L, x.device)
        Q = self._apply_rope(Q, freqs[past_len:past_len + L])
        K = self._apply_rope(K, freqs[past_len:past_len + L])

        # --- KV cache (generation mode) ---
        if kv_cache is not None:
            K = torch.cat([kv_cache['k'], K], dim=2)
            V = torch.cat([kv_cache['v'], V], dim=2)
        new_kv = {'k': K, 'v': V}

        # --- Sliding window mask ---
        total_kv = K.shape[2]
        mask = build_sliding_window_mask(total_kv, self.window, x.device)
        # For queries that start at past_len, slice mask appropriately
        mask = mask[past_len:past_len + L, :]   # (L, total_kv)

        # --- Scaled dot-product attention ---
        scale = math.sqrt(d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (B, H, L, total_kv)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn.float(), dim=-1).to(V.dtype)     # fp32 softmax, cast back

        out = torch.matmul(attn, V)                             # (B, H, L, dh)
        out = out.permute(0, 2, 1, 3).reshape(B, L, n_heads * d_head)

        # --- Output projection ---
        out = self._lora_proj(out, self.shared.o_proj.weight, self.o_lora_A, self.o_lora_B)

        return residual + out, new_kv


if __name__ == '__main__':
    cfg = ModelConfig(d_model=256, n_attn_heads=4, d_attn_head=64, attn_window=16, lora_rank=8)
    shared = SharedAttentionWeights(cfg.d_model, cfg.n_attn_heads, cfg.d_attn_head, cfg.lora_rank)
    attn = SparseAttention(cfg, shared, layer_idx=5)
    attn.eval()

    x = torch.randn(2, 32, 256)
    with torch.no_grad():
        y, kv = attn(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"K cache: {kv['k'].shape}")
    assert y.shape == x.shape
    print("OK")
