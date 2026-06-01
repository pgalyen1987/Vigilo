"""
CodingSSM: Full model assembly.

Layer stack (24 layers):
  - Every 6th layer (indices 5, 11, 17, 23): SparseAttention
  - Remaining 20 layers: Mamba-2

  FFN:
  - Even layer indices: MoEFFN (12 MoE layers)
  - Odd layer indices:  DenseFFN (12 dense layers)

  Shared attention: 2 weight sets (A and B), interleaved:
    layer 5  -> shared set A
    layer 11 -> shared set B
    layer 17 -> shared set A
    layer 23 -> shared set B

Total parameters (approximate):
  Embeddings:  2048 * 152064 ≈ 311M
  Per Mamba2 block (20):  ~30M each ≈ 600M
  Per attn block (4):     ~10M each ≈ 40M
  MoE FFN (12):           ~50M each ≈ 600M (but 8 experts, mostly inactive)
  Dense FFN (12):         ~67M each ≈ 800M
  Total ≈ 3B params, ~800M active
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .config import ModelConfig
from .mamba2 import Mamba2Block, RMSNorm
from .sparse_attn import SparseAttention, SharedAttentionWeights
from .moe import MoEFFN, DenseFFN


class CodingSSMLayer(nn.Module):
    """One transformer-style layer: (Mamba2 or SparseAttn) + FFN."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        shared_attn: SharedAttentionWeights = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attn = layer_idx in config.attn_layer_indices()

        if self.is_attn:
            assert shared_attn is not None
            self.mixer = SparseAttention(config, shared_attn, layer_idx=layer_idx)
        else:
            self.mixer = Mamba2Block(config, layer_idx=layer_idx)

        if config.is_moe_layer(layer_idx):
            self.ffn = MoEFFN(config, layer_idx=layer_idx)
        else:
            self.ffn = DenseFFN(config)

    def forward(self, x, state=None):
        """
        Args:
            x: (B, L, d_model)
            state: dict with 'ssm_state' or 'kv' depending on layer type

        Returns:
            x: (B, L, d_model)
            new_state: updated state dict
            aux_loss: MoE load-balancing loss (0 for dense layers)
        """
        if self.is_attn:
            kv_cache = state.get('kv') if state else None
            x, new_kv = self.mixer(x, kv_cache=kv_cache)
            new_state = {'kv': new_kv}
        else:
            x, new_ssm = self.mixer(x, state=state)
            new_state = new_ssm

        x, aux_loss = self.ffn(x)
        return x, new_state, aux_loss


class CodingSSM(nn.Module):
    """
    Full CodingSSM model.

    Forward pass returns (logits, aux_loss) during training.
    For generation, use generate() which handles KV/SSM state caching.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Shared attention weight sets (ABAB interleaving)
        self.shared_attn = nn.ModuleList([
            SharedAttentionWeights(
                config.d_model, config.n_attn_heads,
                config.d_attn_head, config.lora_rank
            )
            for _ in range(config.n_shared_attn)
        ])

        # Map each attention layer to a shared weight set
        attn_indices = config.attn_layer_indices()
        self._attn_shared_map = {
            layer_idx: self.shared_attn[i % config.n_shared_attn]
            for i, layer_idx in enumerate(attn_indices)
        }

        # Build layer stack
        self.layers = nn.ModuleList([
            CodingSSMLayer(
                config,
                layer_idx=i,
                shared_attn=self._attn_shared_map.get(i),
            )
            for i in range(config.n_layers)
        ])

        # Final norm + LM head
        self.norm_f = RMSNorm(config.d_model)

        if config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight   # tied
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.gradient_checkpointing = False
        self._init_weights()

    def enable_gradient_checkpointing(self):
        """Recompute activations during backward pass to save memory (~30% slower, ~60% less RAM)."""
        self.gradient_checkpointing = True

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,     # (B, L)
        states: list[dict] = None,   # per-layer state dicts (generation)
        return_states: bool = False,
    ) -> tuple:
        """
        Returns:
            logits:    (B, L, vocab_size)
            aux_loss:  scalar -- sum of MoE load-balancing losses
            new_states (optional): list of per-layer state dicts
        """
        x = self.embed(input_ids)   # (B, L, d_model)

        # Recursive inference: apply layer stack recursion_depth times.
        # With recursion_depth=2 (TRM-style) the model makes 2 full passes through
        # the same weights — doubling effective depth at zero extra parameters.
        # When state caching is active (generation), we only do one pass to keep
        # the KV/SSM state layout simple and unambiguous.
        recursion = self.config.recursion_depth if states is None else 1

        if states is None:
            states = [None] * self.config.n_layers

        total_aux_loss = torch.tensor(0.0, device=x.device)
        new_states = []

        for rec in range(recursion):
            new_states = []
            for i, layer in enumerate(self.layers):
                # First recursion pass uses the caller-provided states (generation KV/SSM cache).
                # Subsequent passes (recursion_depth>1, training only) always pass None — the
                # hidden state x carries the information forward, not the SSM state.
                layer_state = states[i] if rec == 0 else None
                if self.gradient_checkpointing and self.training and layer_state is None:
                    # Wrap layer forward in gradient checkpoint.
                    # aux_loss and new_state are not checkpointable (non-tensor outputs),
                    # so we split: checkpoint only the hidden state, collect aux separately.
                    def make_fn(layer_):
                        def fn(x_):
                            out, _, aux = layer_(x_, state=None)
                            return out, aux
                        return fn
                    x, aux_loss = grad_checkpoint(make_fn(layer), x, use_reentrant=False)
                    new_state = None
                else:
                    x, new_state, aux_loss = layer(x, state=layer_state)
                total_aux_loss = total_aux_loss + aux_loss
                new_states.append(new_state)

        x = self.norm_f(x)
        logits = self.lm_head(x)   # (B, L, vocab_size)

        if return_states:
            return logits, total_aux_loss, new_states
        return logits, total_aux_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,   # (B, prompt_len)
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        """Simple autoregressive generation with state caching."""
        self.eval()
        B = input_ids.shape[0]
        states = [None] * self.config.n_layers

        # Process prompt
        logits, _, states = self.forward(input_ids, states=states, return_states=True)
        next_token_logits = logits[:, -1, :]   # (B, vocab)

        generated = []
        for _ in range(max_new_tokens):
            # Sample
            next_token = self._sample(next_token_logits, temperature, top_p)  # (B,)
            generated.append(next_token)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            # Forward one token
            logits, _, states = self.forward(
                next_token.unsqueeze(1), states=states, return_states=True
            )
            next_token_logits = logits[:, -1, :]

        if not generated:
            return input_ids
        return torch.stack(generated, dim=1)   # (B, new_tokens)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Top-p (nucleus) sampling."""
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-p filtering
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        remove = cumsum - sorted_probs > top_p
        sorted_probs[remove] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_idx.gather(-1, next_token).squeeze(-1)   # (B,)

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    import time

    # Small config for smoke test
    cfg = ModelConfig(
        d_model=256,
        d_inner=512,
        n_layers=6,
        n_heads=8,
        d_head=64,
        d_state=16,
        n_groups=2,
        chunk_size=64,
        n_attn_heads=4,
        d_attn_head=64,
        attn_every_n=3,
        attn_window=16,
        n_shared_attn=2,
        lora_rank=8,
        n_experts=4,
        max_active_experts=2,
        min_active_experts=1,
        d_ffn=512,
        vocab_size=256,
        max_seq_len=128,
        tie_embeddings=True,
    )

    model = CodingSSM(cfg)
    model.eval()

    n_params = model.num_parameters()
    print(f"Parameters: {n_params:,}")

    input_ids = torch.randint(0, 256, (2, 64))

    t0 = time.time()
    with torch.no_grad():
        logits, aux = model(input_ids)
    elapsed = time.time() - t0

    print(f"Input:    {input_ids.shape}")
    print(f"Logits:   {logits.shape}")
    print(f"Aux loss: {aux.item():.4f}")
    print(f"Time:     {elapsed*1000:.1f}ms")

    assert logits.shape == (2, 64, 256)
    print("Forward pass OK")

    # Test generation
    prompt = torch.randint(0, 256, (1, 8))
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=16, temperature=0.8)
    elapsed = time.time() - t0
    print(f"Generated: {out.shape} tokens in {elapsed*1000:.1f}ms")
    print("Generation OK")
