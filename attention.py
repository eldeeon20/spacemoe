"""Grouped Query Attention (GQA) compatible with Rust blocks::trasformer::attention.

Supports:
  - Multi-Head Attention (num_kv_groups == num_heads)
  - Multi-Query Attention (num_kv_groups == 1)
  - Grouped Query Attention (1 < num_kv_groups < num_heads)
  - Causal masking
  - Attention logit soft-capping (Gemma2 style)
  - KV cache for autoregressive generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial
from cache_kv import KVCache


def repeat_kv(x: torch.Tensor, num_heads: int, num_kv_groups: int) -> torch.Tensor:
    """Repeat KV groups to match the number of query heads.

    Compatible with Rust repeat_kv.

    Args:
        x: (batch, seq_len, num_kv_groups, head_dim)
    Returns:
        (batch, seq_len, num_heads, head_dim)
    """
    if num_kv_groups == num_heads:
        return x
    repeats = num_heads // num_kv_groups
    return x.repeat_interleave(repeats, dim=2)


class QKVProjection(nn.Module):
    """QKV projection with per-head reshaping.

    Compatible with Rust QKVProjection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_groups * head_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            q: (batch, seq_len, num_heads, head_dim)
            k: (batch, seq_len, num_kv_groups, head_dim)
            v: (batch, seq_len, num_kv_groups, head_dim)
        """
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_groups, self.head_dim)
        return q, k, v


class OutputProjection(nn.Module):
    """Merge heads and project output.

    Compatible with Rust OutputProjection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        bias: bool = False,
    ):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_heads, head_dim)
        Returns:
            (batch, seq_len, d_model)
        """
        B, S, NH, HD = x.shape
        return self.o_proj(x.reshape(B, S, NH * HD))


class Attention(nn.Module):
    """Grouped Query Attention with RoPE and causal masking.

    Compatible with Rust Attention struct.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap

        self.qkv = QKVProjection(d_model, num_heads, num_kv_groups, head_dim, bias)
        self.o_proj = OutputProjection(d_model, num_heads, head_dim, bias)
        self.rope = RoPE(head_dim, max_seq_len, rope_base, rope_scaling)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Full attention forward (no cache, for training).

        Args:
            x: (batch, seq_len, d_model)
            offset: position offset for RoPE
        Returns:
            (batch, seq_len, d_model)
        """
        q, k, v = self.qkv(x)
        q, k = self.rope(q, k, offset)

        k = repeat_kv(k, self.num_heads, self.num_kv_groups)
        v = repeat_kv(v, self.num_heads, self.num_kv_groups)

        # Transpose: (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        seq_len = q.shape[2]
        if self.causal and seq_len > 1:
            scores = self._apply_causal_mask(scores, seq_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back: (B, H, S, D) -> (B, S, H, D)
        attn_output = attn_output.transpose(1, 2)
        return self.o_proj(attn_output)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        offset: int,
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Attention with KV cache for autoregressive generation.

        Args:
            x: (batch, new_seq_len, d_model)
            offset: position offset
            cache: previous KV cache (None for prefill)
        Returns:
            (output, new_cache)
        """
        q, k_new, v_new = self.qkv(x)
        q, k_new = self.rope(q, k_new, offset)

        if cache is not None:
            k_full = torch.cat([cache.cached_k, k_new], dim=1)
            v_full = torch.cat([cache.cached_v, v_new], dim=1)
        else:
            k_full = k_new
            v_full = v_new

        new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

        k_expanded = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_expanded = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k_expanded.transpose(1, 2)
        v = v_expanded.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        q_len = q.shape[2]
        kv_len = k.shape[2]
        if self.causal and q_len > 1:
            scores = self._apply_causal_mask_with_offset(scores, q_len, kv_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)
        output = self.o_proj(attn_output)
        return output, new_cache

    def forward_with_cache_partial(
        self,
        x: torch.Tensor,
        offset: int,
        cache: KVCache | None,
        rotary_pct: float,
    ) -> tuple[torch.Tensor, KVCache]:
        """Attention with KV cache + partial RoPE.

        Compatible with Rust forward_with_cache_partial.
        """
        q, k_new, v_new = self.qkv(x)

        # Apply partial RoPE
        q, k_new = apply_rope_partial(
            q, k_new, offset, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.head_dim, self.rope.max_seq_len,
        )

        if cache is not None:
            k_full = torch.cat([cache.cached_k, k_new], dim=1)
            v_full = torch.cat([cache.cached_v, v_new], dim=1)
        else:
            k_full = k_new
            v_full = v_new

        new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

        k_expanded = repeat_kv(k_full, self.num_heads, self.num_kv_groups)
        v_expanded = repeat_kv(v_full, self.num_heads, self.num_kv_groups)

        q = q.transpose(1, 2)
        k = k_expanded.transpose(1, 2)
        v = v_expanded.transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        q_len = q.shape[2]
        kv_len = k.shape[2]
        if self.causal and q_len > 1:
            scores = self._apply_causal_mask_with_offset(scores, q_len, kv_len)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2)
        output = self.o_proj(attn_output)
        return output, new_cache

    def _apply_causal_mask(
        self, scores: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Lower-triangular causal mask."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
            diagonal=1,
        )
        return scores + mask.unsqueeze(0).unsqueeze(0)

    def _apply_causal_mask_with_offset(
        self, scores: torch.Tensor, q_len: int, kv_len: int
    ) -> torch.Tensor:
        """Causal mask with offset for cached generation."""
        offset = kv_len - q_len
        mask = torch.triu(
            torch.full((q_len, kv_len), float("-inf"), device=scores.device),
            diagonal=offset + 1,
        )
        return scores + mask.unsqueeze(0).unsqueeze(0)


class SparseAttnWrapper(Attention):
    """Wraps Native Sparse Attention to match Attention interface.

    Replaces standard GQA attention with NSA (compressed + fine + sliding window).
    Falls back to standard Attention if native-sparse-attention not installed.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        sliding_window_size: int = 64,
        compress_block_size: int = 32,
        compress_block_sliding_stride: int = 16,
        selection_block_size: int = 32,
        num_selected_blocks: int = 4,
    ):
        if not HAS_SPARSE:
            super().__init__(d_model, num_heads, num_kv_groups, head_dim,
                             max_seq_len, rope_base, rope_scaling, causal,
                             dropout, attn_logit_cap, bias)
            return

        nn.Module.__init__(self)
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self._fallback = False

        self.rope = RoPE(head_dim, max_seq_len, rope_base, rope_scaling)
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.sparse_attn = NativeSparseAttention(
            dim=d_model,
            dim_head=head_dim,
            heads=num_heads,
            kv_heads=num_kv_groups,
            causal=causal,
            sliding_window_size=sliding_window_size,
            compress_block_size=compress_block_size,
            compress_block_sliding_stride=compress_block_sliding_stride,
            selection_block_size=selection_block_size,
            num_selected_blocks=num_selected_blocks,
        )

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if not HAS_SPARSE or getattr(self, '_fallback', False):
            return super().forward(x, offset)
        return self.sparse_attn(x)

    def _apply_causal_mask(self, scores, seq_len):
        return super()._apply_causal_mask(scores, seq_len)

    def _apply_causal_mask_with_offset(self, scores, q_len, kv_len):
        return super()._apply_causal_mask_with_offset(scores, q_len, kv_len)

    def forward_with_cache(self, x, offset, cache):
        return super().forward_with_cache(x, offset, cache)

    def forward_with_cache_partial(self, x, offset, cache, rotary_pct):
        return super().forward_with_cache_partial(x, offset, cache, rotary_pct)
