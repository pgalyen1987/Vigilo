"""
Verifier model for CodingSSM.

A small model that scores how likely a solution is correct given a problem.
Used for best-of-N selection without test execution.

Architecture: Lightweight transformer encoder (not SSM — simpler for classification)
~100M params, takes concatenated (problem + solution) as input, outputs correctness score.

Usage:
    from arch.verifier import VerifierModel, VerifierConfig100M
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
    cfg = VerifierConfig100M()
    model = VerifierModel(cfg)

    # Load trained weights
    import torch
    ckpt = torch.load('checkpoints/verifier/verifier_best.pt', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    score = model.score(
        problem_text="Write a function that reverses a string",
        solution_text="def reverse(s): return s[::-1]",
        tokenizer=tokenizer,
    )
    print(f"Correctness probability: {score:.3f}")
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class VerifierConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ffn: int = 2048
    vocab_size: int = 152064    # Qwen2.5 tokenizer
    max_seq_len: int = 4096
    dropout: float = 0.1
    pad_token_id: int = 0


def VerifierConfig100M() -> VerifierConfig:
    """
    ~100M parameter verifier config.

    Parameter count estimate:
      Embeddings: 512 * 152064 ≈ 78M
      6 encoder layers * ~3.6M each ≈ 22M
      Total: ~100M
    """
    return VerifierConfig(
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ffn=2048,
        vocab_size=152064,
        max_seq_len=4096,
        dropout=0.1,
    )


# ── Positional encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal position embeddings (non-learned)."""

    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── Verifier model ────────────────────────────────────────────────────────────

class VerifierModel(nn.Module):
    """
    Transformer encoder that scores (problem, solution) pairs.

    Input:  concatenated token ids  [<im_start>user\n{problem}<im_end>\n<im_start>assistant\n{solution}<im_end>]
    Output: scalar in [0, 1] — probability that the solution is correct.

    Architecture:
      - Token embedding (shared with output head is NOT done — classification only)
      - Sinusoidal positional encoding
      - N × nn.TransformerEncoderLayer (pre-norm variant)
      - Mean pooling over non-padding tokens
      - Linear head → sigmoid
    """

    def __init__(self, config: VerifierConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,   # (B, L, d_model) convention
            norm_first=True,    # pre-norm (more stable training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Classification head
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) — token ids, padded with pad_token_id

        Returns:
            score: (B,) — correctness probability in [0, 1]
        """
        B, L = input_ids.shape
        pad_id = self.config.pad_token_id

        # Build padding mask: True where tokens are padding (TransformerEncoder convention)
        key_padding_mask = (input_ids == pad_id)  # (B, L)

        # Embed + positional encoding
        x = self.embed(input_ids)        # (B, L, d_model)
        x = self.pos_enc(x)              # (B, L, d_model)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        # Mean pool over non-padding positions
        non_pad_mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
        x = (x * non_pad_mask).sum(dim=1) / non_pad_mask.sum(dim=1).clamp(min=1)  # (B, d_model)

        x = self.norm(x)
        logit = self.head(x).squeeze(-1)  # (B,)
        score = torch.sigmoid(logit)
        return score

    def score(
        self,
        problem_text: str,
        solution_text: str,
        tokenizer,
        max_length: int = 2048,
        device: torch.device = None,
    ) -> float:
        """
        Convenience method: tokenize (problem, solution) pair and return a correctness score.

        Args:
            problem_text:  The coding problem description.
            solution_text: The proposed solution code.
            tokenizer:     Hugging Face tokenizer (Qwen2.5 or compatible).
            max_length:    Maximum sequence length (truncated if longer).
            device:        Device to run on. Uses model's current device if None.

        Returns:
            float in [0, 1] — probability the solution is correct.
        """
        if device is None:
            device = next(self.parameters()).device

        # Format as chat template matching training format
        text = (
            f"<|im_start|>user\n{problem_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{solution_text}<|im_end|>"
        )

        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)  # (1, L)

        self.eval()
        with torch.no_grad():
            prob = self(input_ids)  # (1,)

        return prob.item()

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = VerifierConfig100M()
    model = VerifierModel(cfg)
    n = model.num_parameters()
    print(f"VerifierModel params: {n:,}  ({n/1e6:.1f}M)")

    # Sanity forward pass
    B, L = 2, 128
    ids = torch.randint(0, cfg.vocab_size, (B, L))
    ids[:, -10:] = cfg.pad_token_id  # some padding

    scores = model(ids)
    print(f"Input:  {ids.shape}")
    print(f"Scores: {scores.shape}  values={scores.tolist()}")
    assert scores.shape == (B,)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    print("Forward pass OK")
