"""
MoE FFN with dynamic top-k routing (DynaMoE-inspired).

Features:
  - n_experts=8 total, max_active=4 / min_active=1 per token
  - top-k determined dynamically per token by a router confidence score
  - descending capacity schedule: earlier layers use more experts
  - load-balancing auxiliary loss for training stability
  - SwiGLU activation within each expert
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mamba2 import RMSNorm


class Expert(nn.Module):
    """Single SwiGLU expert: gate * up -> SiLU -> down."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj   = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DynamicRouter(nn.Module):
    """
    Dynamic top-k router.

    For each token, computes expert logits. Then determines k dynamically:
      - If router confidence (max softmax score) is high -> fewer experts needed
      - Always clamps k in [min_k, max_k]

    Threshold: use top-1 softmax score to decide. If score >= high_conf_thresh,
    use k=1. If score >= mid_conf_thresh, use k=2. Otherwise use k=max_k.
    (This mimics DynaMoE's adaptive mechanism without the full complexity.)
    """

    def __init__(self, d_model: int, n_experts: int,
                 max_k: int = 4, min_k: int = 1,
                 high_conf: float = 0.6, mid_conf: float = 0.4):
        super().__init__()
        self.n_experts = n_experts
        self.max_k = max_k
        self.min_k = min_k
        self.high_conf = high_conf
        self.mid_conf = mid_conf
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor, fixed_k: int = None):
        """
        Args:
            x: (B*L, d_model) -- flattened tokens
            fixed_k: override dynamic selection (for inference or ablation)

        Returns:
            indices:  (B*L, max_k) -- expert indices (padded with -1)
            weights:  (B*L, max_k) -- softmax weights (0 for padded)
            k_used:   (B*L,)       -- actual k per token
            aux_loss: scalar load-balancing loss
        """
        logits = self.gate(x)           # (T, n_experts)
        probs = F.softmax(logits, dim=-1)

        if fixed_k is not None:
            k_used = torch.full((x.shape[0],), fixed_k, device=x.device, dtype=torch.long)
        else:
            # Dynamic k: high confidence → 1 expert, mid → 2, low → max_k
            top1_score = probs.max(dim=-1).values   # (T,)
            k_used = torch.full((x.shape[0],), self.max_k, device=x.device, dtype=torch.long)
            k_used[top1_score >= self.mid_conf]  = 2
            k_used[top1_score >= self.high_conf] = 1
            k_used = k_used.clamp(self.min_k, self.max_k)

        # Always select top max_k, then zero-out beyond k_used
        top_vals, top_idx = probs.topk(self.max_k, dim=-1)   # (T, max_k)

        # Mask: for each token, positions >= k_used[t] are invalid
        positions = torch.arange(self.max_k, device=x.device).unsqueeze(0)  # (1, max_k)
        valid = positions < k_used.unsqueeze(1)                              # (T, max_k)

        weights = top_vals * valid.float()
        # Renormalize
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        weights = weights / weight_sum

        indices = top_idx.masked_fill(~valid, -1)

        # Load-balancing loss (encourage uniform expert usage)
        # Auxiliary loss = n_experts * sum_e(f_e * P_e) where:
        #   f_e = fraction of tokens routed to expert e (using hard top-1)
        #   P_e = mean router probability for expert e
        with torch.no_grad():
            top1_idx = probs.argmax(dim=-1)   # (T,)
            counts = torch.bincount(top1_idx, minlength=self.n_experts).float()
            f_e = counts / x.shape[0]
        P_e = probs.mean(dim=0)
        aux_loss = self.n_experts * (f_e * P_e).sum()

        return indices, weights, k_used, aux_loss


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN with dynamic top-k routing.

    Token dispatch is done in a simple loop over experts (CPU-friendly).
    For GPU training a scatter/gather approach would be faster, but the
    simple loop is clearer and works on CPU.
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_experts = config.n_experts

        self.norm = RMSNorm(config.d_model)

        self.experts = nn.ModuleList([
            Expert(config.d_model, config.d_ffn)
            for _ in range(config.n_experts)
        ])

        max_k = config.expert_budget(layer_idx)
        self.router = DynamicRouter(
            d_model=config.d_model,
            n_experts=config.n_experts,
            max_k=config.max_active_experts,
            min_k=config.min_active_experts,
        )
        self.layer_max_k = max_k   # budget for this layer

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            out:      (B, L, d_model)
            aux_loss: scalar
        """
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        x_flat = x.reshape(B * L, D)  # (T, D)
        indices, weights, k_used, aux_loss = self.router(x_flat, fixed_k=self.layer_max_k)

        out_flat = torch.zeros_like(x_flat)

        for e_idx in range(self.n_experts):
            # Find tokens that route to this expert
            token_mask = (indices == e_idx).any(dim=-1)   # (T,)
            if not token_mask.any():
                continue

            tokens = x_flat[token_mask]                    # (n_tokens, D)
            expert_out = self.experts[e_idx](tokens)       # (n_tokens, D)

            # Find weight for this expert per token
            # weights shape: (T, max_k), indices shape: (T, max_k)
            e_weights = torch.zeros(x_flat.shape[0], device=x.device)
            for k in range(indices.shape[1]):
                match = indices[:, k] == e_idx
                e_weights[match] = weights[match, k]

            e_weights_masked = e_weights[token_mask].unsqueeze(-1)   # (n_tokens, 1)
            out_flat[token_mask] += expert_out * e_weights_masked

        out = out_flat.reshape(B, L, D)
        return residual + out, aux_loss


class DenseFFN(nn.Module):
    """Standard SwiGLU FFN for non-MoE layers."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.gate_proj = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.down_proj = nn.Linear(config.d_ffn, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        out = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return residual + out, torch.tensor(0.0)


if __name__ == '__main__':
    cfg = ModelConfig(d_model=256, d_ffn=1024, n_experts=4, max_active_experts=2, min_active_experts=1)
    moe = MoEFFN(cfg, layer_idx=0)
    moe.eval()

    x = torch.randn(2, 32, 256)
    with torch.no_grad():
        y, loss = moe(x)
    print(f"Input:    {x.shape}")
    print(f"Output:   {y.shape}")
    print(f"Aux loss: {loss.item():.4f}")
    assert y.shape == x.shape
    print("OK")
