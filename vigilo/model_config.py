"""
Small CodingSSM config for predictive-maintenance telemetry.

The PdM task tokenizes sensor readings into a tiny vocabulary (a few hundred
tokens), so the model is far smaller than the code model — it fits and trains
on CPU. Same Mamba-2 + attention + MoE architecture, just scaled down.
"""
from arch.config import ModelConfig


def ModelConfigPdM(max_seq_len: int = 1024) -> ModelConfig:
    """Small telemetry config for the forecaster. vocab_size is unused (the
    forecaster has continuous I/O) but ModelConfig requires the field."""
    return ModelConfig(
        d_model=128,
        d_inner=256,
        n_layers=4,
        n_heads=4,
        d_head=64,          # n_heads * d_head must equal d_inner (4*64=256)
        d_state=16,
        n_groups=1,
        chunk_size=64,
        n_attn_heads=4,
        d_attn_head=32,
        attn_every_n=2,        # 1 attention layer every 2 → layers 1 and 3
        attn_window=256,
        n_shared_attn=1,
        lora_rank=8,
        n_experts=4,
        max_active_experts=2,
        min_active_experts=1,
        d_ffn=256,
        vocab_size=16,          # unused by the forecaster (continuous I/O)
        max_seq_len=max_seq_len,
        recursion_depth=1,
        tie_embeddings=True,
        pad_token_id=0,
    )
