"""
Mamba-2 / SSD (Structured State Space Duality) block.

Key equations (from Dao & Gu, 2024):
  h_t = A_t * h_{t-1} + B_t * x_t          (recurrent form)
  y_t = C_t^T * h_t                          (output)

In matrix / SSD form over a chunk of T tokens:
  Y = (L ⊙ (C @ B^T)) @ X    where L[t,s] = exp(A_cumsum[t] - A_cumsum[s]) for t≥s

We use the chunked algorithm:
  1. Intra-chunk: Y_diag[c] = L_c ⊙ (C_c @ B_c^T) @ X_c
  2. Inter-chunk: carry h passed forward across chunks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# Optional fast CUDA kernels (pip install mamba-ssm causal-conv1d)
# Falls back to pure PyTorch automatically when unavailable (e.g. Kaggle CPU/T4).
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_fast_fn
    _HAS_MAMBA_FAST = True
except Exception:
    _HAS_MAMBA_FAST = False

try:
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
    _HAS_CAUSAL_CONV = True
except Exception:
    _HAS_CAUSAL_CONV = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction, learnable scale)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def compute_decay_matrix(A_cumsum: torch.Tensor) -> torch.Tensor:
    """
    Build lower-triangular decay matrix L for one chunk.

    Args:
        A_cumsum: (batch, chunk, heads) -- cumulative sum of log-decays within chunk

    Returns:
        L: (batch, heads, chunk, chunk)  L[t,s] = exp(A_cs[t] - A_cs[s]) for t>=s, else 0
    """
    # A_cumsum: (B, T, H) -> work in (B, H, T)
    A = A_cumsum.permute(0, 2, 1)          # (B, H, T)
    diff = A.unsqueeze(-1) - A.unsqueeze(-2)  # (B, H, T, T): diff[t, s] = A[t] - A[s]
    mask = torch.tril(torch.ones(A.shape[-1], A.shape[-1], device=A.device, dtype=torch.bool))
    diff = diff.masked_fill(~mask, float('-inf'))
    return torch.exp(diff)                 # (B, H, T, T)


def ssd_chunk_scan(
    X: torch.Tensor,      # (B, L, H, D)  -- projected input
    A_log: torch.Tensor,  # (B, L, H)     -- log-space decay
    B: torch.Tensor,      # (B, L, H, N)  -- input projection
    C: torch.Tensor,      # (B, L, H, N)  -- output projection
    chunk_size: int = 256,
    initial_state: torch.Tensor = None,  # (B, H, N, D)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked SSD scan. Returns (Y, final_state).
      Y:           (B, L, H, D)
      final_state: (B, H, N, D)
    """
    B_batch, L, H, D = X.shape
    N = B.shape[-1]

    # Pad to multiple of chunk_size
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad > 0:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A_log = F.pad(A_log, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    total_len = X.shape[1]
    n_chunks = total_len // chunk_size

    # Reshape into chunks
    Xc = X.view(B_batch, n_chunks, chunk_size, H, D)        # (B, nC, T, H, D)
    A_c = A_log.view(B_batch, n_chunks, chunk_size, H)      # (B, nC, T, H)
    Bc = B.view(B_batch, n_chunks, chunk_size, H, N)        # (B, nC, T, H, N)
    Cc = C.view(B_batch, n_chunks, chunk_size, H, N)        # (B, nC, T, H, N)

    # Cumulative sum within each chunk (for decay matrix)
    A_cs = A_c.cumsum(dim=2)   # (B, nC, T, H)

    # Chunk-level cumulative decay (last token of each chunk)
    A_chunk_last = A_cs[:, :, -1, :]   # (B, nC, H) -- total decay per chunk

    Y_out = torch.zeros_like(Xc)

    if initial_state is None:
        h = torch.zeros(B_batch, H, N, D, device=X.device, dtype=X.dtype)
    else:
        h = initial_state

    for c in range(n_chunks):
        # --- Intra-chunk (diagonal block) ---
        L_mat = compute_decay_matrix(A_cs[:, c])   # (B, H, T, T)

        # Y_diag[b, t, h, d] = sum_s L[b,h,t,s] * sum_n C[b,t,h,n] * B[b,s,h,n] * X[b,s,h,d]
        # Factored via (C @ B^T) then multiply by L:
        # CB: (B, H, T, T) where CB[b,h,t,s] = sum_n C[b,c,t,h,n] * B[b,c,s,h,n]
        Cc_c = Cc[:, c]   # (B, T, H, N)
        Bc_c = Bc[:, c]
        Xc_c = Xc[:, c]  # (B, T, H, D)

        # (B, H, T, N) @ (B, H, N, T) -> (B, H, T, T)
        CB = torch.einsum('bthN,bshN->bhts', Cc_c, Bc_c)   # sum over N
        LCB = L_mat * CB                                      # (B, H, T, T)
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D) -> (B, T, H, D)
        Y_diag = torch.einsum('bhts,bshd->bthd', LCB, Xc_c)

        # --- Inter-chunk: contribution from carry h ---
        # Decay from start of chunk to each position t
        # A_cs[:, c] shape: (B, T, H)
        # decay_to_t[b, t, h] = exp(A_cs[b, t, h]) -- relative to start of chunk
        decay_to_t = torch.exp(A_cs[:, c])   # (B, T, H)

        # Y_carry[b, t, h, d] = sum_n C[b,t,h,n] * h[b,h,n,d] * decay_to_t[b,t,h]
        # h: (B, H, N, D)
        # C: (B, T, H, N)
        Y_carry = torch.einsum('bthN,bhNd->bthd', Cc_c, h)  # (B, T, H, D)
        Y_carry = Y_carry * decay_to_t.unsqueeze(-1)         # scale by decay

        Y_out[:, c] = Y_diag + Y_carry

        # --- Update carry state ---
        # h_new = exp(A_chunk_last) * h + sum_t B[t] * X[t] * exp(A_cs_end - A_cs[t])
        chunk_decay = torch.exp(A_chunk_last[:, c])          # (B, H)
        h = h * chunk_decay.unsqueeze(-1).unsqueeze(-1)      # (B, H, N, D)

        # Residual: sum_t exp(A_cs[-1] - A_cs[t]) * B[t] * X[t]
        # decay_from_t: (B, T, H)  decay_from_t[t] = exp(A_cs[-1] - A_cs[t])
        A_cs_last = A_cs[:, c, -1, :]    # (B, H)
        decay_from_t = torch.exp(A_cs_last.unsqueeze(1) - A_cs[:, c])  # (B, T, H)

        # Fused: sum_t decay[t] * B[t] ⊗ X[t]  →  (B, H, N, D)
        # Avoids materializing (B, T, H, N, D) which is 128 MiB at seq=512.
        # Equivalent to: einsum(decay * B, X -> bthNd).sum(t)
        B_decayed = Bc_c * decay_from_t.unsqueeze(-1)        # (B, T, H, N)
        h = h + torch.einsum('bthN,bthd->bhNd', B_decayed, Xc_c)  # (B, H, N, D)

    # Remove padding
    Y_out = Y_out.reshape(B_batch, total_len, H, D)
    if pad > 0:
        Y_out = Y_out[:, :L]

    return Y_out, h


class Mamba2Block(nn.Module):
    """
    Full Mamba-2 block:
      x -> RMSNorm -> in_proj -> conv1d -> SSD -> gate -> out_proj -> residual
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        d = config.d_model
        d_inner = config.d_inner
        H = config.n_heads        # number of SSM heads
        N = config.d_state        # state size
        G = config.n_groups       # number of groups (B and C are grouped)

        self.norm = RMSNorm(d)

        # Projection sizes:
        #   X:    H * d_head  = d_inner
        #   A:    H           (one decay per head)
        #   B:    G * N
        #   C:    G * N
        #   gate: d_inner
        d_head = config.d_head
        assert H * d_head == d_inner

        self.d_head = d_head
        self.H = H
        self.N = N
        self.G = G

        in_proj_dim = d_inner + H + 2 * G * N + d_inner  # X, A, B, C, gate
        self.in_proj = nn.Linear(d, in_proj_dim, bias=False)

        # Short depthwise conv on X+B+C before SSM (dt-free Mamba-2)
        conv_dim = d_inner + 2 * G * N
        self.conv1d = nn.Conv1d(
            conv_dim, conv_dim,
            kernel_size=4, padding=3, groups=conv_dim, bias=True
        )
        self.conv_norm = RMSNorm(conv_dim)

        # A log-scale parameter (learnable base decay)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, H + 1, dtype=torch.float32)))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d, bias=False)
        self.out_norm = RMSNorm(d_inner)

        # dt (input-dependent time-step bias added to A)
        self.dt_proj = nn.Linear(H, H, bias=True)
        nn.init.constant_(self.dt_proj.bias, math.log(0.001))

    def forward(
        self,
        x: torch.Tensor,            # (B, L, d_model)
        state: dict = None,         # cached recurrent state for generation
    ) -> tuple[torch.Tensor, dict]:
        B, L, _ = x.shape
        residual = x

        x = self.norm(x)

        # --- Input projection ---
        proj = self.in_proj(x)     # (B, L, in_proj_dim)

        d_inner = self.config.d_inner
        H, N, G = self.H, self.N, self.G
        d_head = self.d_head

        # Split: X, A_dt, B_raw, C_raw, gate
        X_raw = proj[..., :d_inner]                         # (B, L, d_inner)
        A_dt  = proj[..., d_inner:d_inner+H]                # (B, L, H)
        B_raw = proj[..., d_inner+H:d_inner+H+G*N]         # (B, L, G*N)
        C_raw = proj[..., d_inner+H+G*N:d_inner+H+2*G*N]   # (B, L, G*N)
        gate  = proj[..., d_inner+H+2*G*N:]                 # (B, L, d_inner)

        # --- Conv1d on X, B, C ---
        conv_in = torch.cat([X_raw, B_raw, C_raw], dim=-1)  # (B, L, conv_dim)
        conv_in_t = conv_in.permute(0, 2, 1)                 # (B, conv_dim, L)

        if _HAS_CAUSAL_CONV and x.is_cuda:
            # causal_conv1d_fn: no padding needed, handles causality natively
            conv_weight = self.conv1d.weight.squeeze(1)      # (conv_dim, kernel_size)
            conv_out_t = _causal_conv1d_fn(conv_in_t, conv_weight, self.conv1d.bias, activation=None)
            conv_out = conv_out_t.permute(0, 2, 1)           # (B, L, conv_dim)
        else:
            conv_out = self.conv1d(conv_in_t)[..., :L].permute(0, 2, 1)

        conv_out = F.silu(self.conv_norm(conv_out))

        X_conv = conv_out[..., :d_inner]
        B_conv = conv_out[..., d_inner:d_inner+G*N]
        C_conv = conv_out[..., d_inner+G*N:]

        # Reshape for SSM heads
        X_ssm = X_conv.view(B, L, H, d_head)               # (B, L, H, d_head)
        B_ssm = B_conv.view(B, L, G, N)                    # (B, L, G, N)
        C_ssm = C_conv.view(B, L, G, N)                    # (B, L, G, N)

        # Expand B/C from G groups to H heads
        heads_per_group = H // G
        B_ssm = B_ssm.repeat_interleave(heads_per_group, dim=2)  # (B, L, H, N)
        C_ssm = C_ssm.repeat_interleave(heads_per_group, dim=2)

        # --- Compute A (log-decay) ---
        # A_log is learnable base; A_dt is input-dependent delta
        A_log_base = -torch.exp(self.A_log).unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        dt = F.softplus(self.dt_proj(A_dt))                            # (B, L, H)
        A = A_log_base * dt                                            # (B, L, H)

        # --- SSD chunked scan (or fast recurrent step for L=1 generation) ---
        initial_state = state.get('ssm_state') if state else None

        if L == 1 and initial_state is not None:
            # Fast recurrent path for single-token generation steps.
            # Avoids the chunk_size=256 padding overhead of ssd_chunk_scan.
            # h_new = decay * h + B * x,  y = C * h_new
            decay = torch.exp(A[:, 0, :])                          # (B, H)
            h = initial_state                                       # (B, H, N, D)
            h = h * decay[:, :, None, None]
            h = h + torch.einsum('bhN,bhd->bhNd', B_ssm[:, 0], X_ssm[:, 0])
            Y_step = torch.einsum('bhN,bhNd->bhd', C_ssm[:, 0], h)  # (B, H, D)
            Y = Y_step.unsqueeze(1).reshape(B, 1, d_inner)
            new_state = {'ssm_state': h}
        elif _HAS_MAMBA_FAST and x.is_cuda:
            try:
                # Fast Triton kernel: pass grouped B/C (before repeat_interleave),
                # let the kernel handle head-expansion internally.
                B_grp = B_conv.view(B, L, G, N)   # (B, L, G, N)
                C_grp = C_conv.view(B, L, G, N)
                # dt without softplus — kernel applies softplus when dt_softplus=True
                dt_raw = F.linear(A_dt, self.dt_proj.weight)  # (B, L, H), no bias yet
                Y_4d, new_ssm = _mamba_fast_fn(
                    X_ssm,                          # (B, L, H, d_head)
                    dt_raw,                         # (B, L, H)
                    -torch.exp(self.A_log),         # (H,) negative log-decay
                    B_grp,                          # (B, L, G, N)
                    C_grp,                          # (B, L, G, N)
                    chunk_size=self.config.chunk_size,
                    dt_bias=self.dt_proj.bias,      # (H,)
                    dt_softplus=True,
                    initial_states=initial_state,   # (B, H, N, d_head) or None
                    return_final_states=True,
                )
                Y = Y_4d.reshape(B, L, d_inner)
                new_state = {'ssm_state': new_ssm}
            except Exception:
                # Kernel API mismatch — fall back silently
                Y, new_ssm = ssd_chunk_scan(X_ssm, A, B_ssm, C_ssm, self.config.chunk_size, initial_state)
                Y = Y.reshape(B, L, d_inner)
                new_state = {'ssm_state': new_ssm}
        else:
            Y, new_ssm = ssd_chunk_scan(X_ssm, A, B_ssm, C_ssm, self.config.chunk_size, initial_state)
            Y = Y.reshape(B, L, d_inner)
            new_state = {'ssm_state': new_ssm}

        # --- Output ---
        Y = self.out_norm(Y)
        Y = Y * F.silu(gate)
        Y = self.out_proj(Y)

        out = residual + Y

        return out, new_state


if __name__ == '__main__':
    import time
    cfg = ModelConfig(d_model=256, d_inner=512, n_heads=8, d_head=64, d_state=16, n_groups=2, chunk_size=64)
    block = Mamba2Block(cfg, layer_idx=0)
    block.eval()

    x = torch.randn(2, 128, 256)
    t0 = time.time()
    with torch.no_grad():
        y, state = block(x)
    elapsed = time.time() - t0

    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"State:  {state['ssm_state'].shape}")
    print(f"Time:   {elapsed*1000:.1f}ms")
    assert y.shape == x.shape, "Output shape mismatch"
    print("OK")
