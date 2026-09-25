"""Rotary Position Embeddings (RoPE) compatible with Rust blocks::trasformer::rope.

Supports:
  - Standard full RoPE (100%)
  - Partial RoPE (rotary_pct < 1.0) — e.g. Kimi K2, Phi-2
  - NTK-aware scaling
  - Precomputed cos/sin cache
"""

import math
import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Precomputed Rotary Position Embeddings.

    Matches Rust RoPEConfig / RoPE struct.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"RoPE head_dim must be even, got {head_dim}"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        half_dim = head_dim // 2
        scaled_base = base * scaling_factor

        # theta_k = base^(-2k/d) for k = 0..half_dim
        inv_freq = torch.tensor(
            [1.0 / (scaled_base ** (2.0 * k / head_dim)) for k in range(half_dim)],
            dtype=torch.float32,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache: (max_seq_len, half_dim)
        positions = torch.arange(0, max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, half_dim)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to Q and K tensors.

        Args:
            q: (batch, seq_len, num_heads, head_dim)
            k: (batch, seq_len, num_kv_groups, head_dim)
            offset: starting position index

        Returns:
            (q_rotated, k_rotated) same shapes
        """
        seq_len = q.shape[1]
        half_dim = self.head_dim // 2

        if offset + seq_len <= self.max_seq_len:
            cos = self.cos_cache[offset:offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cache[offset:offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
        else:
            positions = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=q.device)
            freqs = torch.outer(positions, self.inv_freq[:half_dim])
            cos = freqs.cos().unsqueeze(0).unsqueeze(2)
            sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        return q_rot, k_rot

    def apply_to_single(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply RoPE to a single tensor."""
        seq_len = x.shape[1]
        half_dim = self.head_dim // 2
        if offset + seq_len <= self.max_seq_len:
            cos = self.cos_cache[offset:offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
            sin = self.sin_cache[offset:offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
        else:
            positions = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=x.device)
            freqs = torch.outer(positions, self.inv_freq[:half_dim])
            cos = freqs.cos().unsqueeze(0).unsqueeze(2)
            sin = freqs.sin().unsqueeze(0).unsqueeze(2)
        return self._apply_rotation(x, cos, sin)

    @staticmethod
    def _apply_rotation(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Core rotation: split into even/odd pairs, rotate.

        x_rot[..., 2k]   = x[..., 2k] * cos[k] - x[..., 2k+1] * sin[k]
        x_rot[..., 2k+1] = x[..., 2k] * sin[k] + x[..., 2k+1] * cos[k]
        """
        half_dim = x.shape[-1] // 2
        x_first = x[..., :half_dim]
        x_second = x[..., half_dim:]

        out_first = x_first * cos - x_second * sin
        out_second = x_first * sin + x_second * cos
        return torch.cat([out_first, out_second], dim=-1)


def apply_rope_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: int,
    rotary_pct: float,
    inv_freq: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to only `rotary_pct` of head dimensions.

    Compatible with Rust apply_rope_partial.
    First rotary_dim dimensions are rotated; rest pass through unchanged.

    Args:
        q: (batch, seq_len, num_heads, head_dim)
        k: (batch, seq_len, num_kv_groups, head_dim)
        offset: starting position
        rotary_pct: fraction of head_dim to rotate (0.0 - 1.0)
        inv_freq, cos_cache, sin_cache: from RoPE module buffers
        head_dim, max_seq_len: configuration
    """
    rotary_dim = int(head_dim * rotary_pct)
    rotary_dim = rotary_dim - (rotary_dim % 2)  # round to even

    if rotary_dim == 0:
        return q, k
    if rotary_dim >= head_dim:
        # Full rotation
        seq_len = q.shape[1]
        half_dim = head_dim // 2
        if offset + seq_len <= max_seq_len:
            cos = cos_cache[offset:offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
            sin = sin_cache[offset:offset + seq_len, :half_dim].unsqueeze(0).unsqueeze(2)
        else:
            positions = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=q.device)
            freqs = torch.outer(positions, inv_freq[:half_dim])
            cos = freqs.cos().unsqueeze(0).unsqueeze(2)
            sin = freqs.sin().unsqueeze(0).unsqueeze(2)
        return RoPE._apply_rotation(q, cos, sin), RoPE._apply_rotation(k, cos, sin)

    seq_len = q.shape[1]
    hh = rotary_dim // 2

    positions = torch.arange(offset, offset + seq_len, dtype=torch.float32, device=q.device)
    freqs = torch.outer(positions, inv_freq[:hh])  # (seq_len, hh)
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, hh)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)

    # Split into rotated and pass-through parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    qr_first = q_rot[..., :hh]
    qr_second = q_rot[..., hh:rotary_dim]
    q_rotated = torch.cat([
        qr_first * cos - qr_second * sin,
        qr_first * sin + qr_second * cos,
    ], dim=-1)
    q_out = torch.cat([q_rotated, q_pass], dim=-1)

    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]
    kr_first = k_rot[..., :hh]
    kr_second = k_rot[..., hh:rotary_dim]
    k_rotated = torch.cat([
        kr_first * cos - kr_second * sin,
        kr_first * sin + kr_second * cos,
    ], dim=-1)
    k_out = torch.cat([k_rotated, k_pass], dim=-1)

    return q_out, k_out
