"""
PdMForecaster — the CodingSSM Mamba-2 layer stack with continuous I/O.

Instead of token-embedding → vocab-logits (the code model), this projects a
continuous sensor vector in and predicts the next cycle's sensor vector out.
The anomaly score is forecast error: on healthy steady-state the trajectory is
predictable (low error); as degradation accelerates, dynamics drift away from
what the model learned (error climbs).

Reuses arch.CodingSSMLayer / SharedAttentionWeights / RMSNorm verbatim — the
shared backbone is not modified.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from arch.config import ModelConfig
from arch.mamba2 import RMSNorm
from arch.model import CodingSSMLayer
from arch.sparse_attn import SharedAttentionWeights


class PdMForecaster(nn.Module):
    def __init__(self, config: ModelConfig, n_features: int):
        super().__init__()
        self.config = config
        self.n_features = n_features

        # Continuous input projection (replaces nn.Embedding).
        self.input_proj = nn.Linear(n_features, config.d_model)

        # Same shared-attention construction as CodingSSM.
        self.shared_attn = nn.ModuleList([
            SharedAttentionWeights(
                config.d_model, config.n_attn_heads,
                config.d_attn_head, config.lora_rank)
            for _ in range(config.n_shared_attn)
        ])
        attn_indices = config.attn_layer_indices()
        attn_shared_map = {
            layer_idx: self.shared_attn[i % config.n_shared_attn]
            for i, layer_idx in enumerate(attn_indices)
        }
        self.layers = nn.ModuleList([
            CodingSSMLayer(config, layer_idx=i, shared_attn=attn_shared_map.get(i))
            for i in range(config.n_layers)
        ])

        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, n_features)   # regression output
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor):
        """
        feats: (B, L, n_features) continuous, time-ordered cycles.
        Returns:
            pred:     (B, L, n_features) — pred[:, t] forecasts feats[:, t+1]
            aux_loss: scalar MoE load-balancing loss
        """
        x = self.input_proj(feats)
        total_aux = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, aux = layer(x, state=None)
            total_aux = total_aux + aux
        x = self.norm_f(x)
        return self.head(x), total_aux

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
