"""Transformer Block compatible with Rust blocks::trasformer::layer.

Architecture per layer:
  x -> RMSNorm -> Attention(GQA + RoPE) -> +residual
    -> RMSNorm -> FeedForward(SwiGLU) -> +residual -> output

Also includes the full Transformer stack with final RMSNorm.
"""

import math
import torch
import torch.nn as nn

from attention import Attention
from mla_attention import MultiHeadLatentAttentionGQA
from cache_kv import KVCache


class RMSNorm(nn.Module):
    """RMSNorm compatible with burn::nn::RmsNorm.

    x -> x * weight / sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Compatible with Rust SwiGLUFeedForward.
    x -> SwiGlu(gate+up) -> dropout -> down_proj

    SwiGLU intermediate_dim = round_to_multiple(expansion * d_model * 2/3, 64)
    """

    def __init__(
        self,
        d_model: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        # SwiGLU has gate_proj + up_proj combined in one Linear (2 * intermediate_dim output)
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = gate * up
        h = self.dropout(h)
        return self.down_proj(h)


class StandardFFN(nn.Module):
    """Standard FFN: x -> up -> GELU -> dropout -> down"""

    def __init__(
        self,
        d_model: int,
        intermediate_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.up_proj(x))
        h = self.dropout(h)
        return self.down_proj(h)


import torch.nn.functional as F


def compute_intermediate_dim(
    d_model: int,
    expansion_factor: float = 4.0,
    use_swiglu: bool = True,
    round_to: int = 64,
) -> int:
    """Compute FFN intermediate dimension matching Rust FeedForwardConfig::intermediate_dim."""
    if use_swiglu:
        raw = int(expansion_factor * d_model * 2.0 / 3.0)
    else:
        raw = int(expansion_factor * d_model)
    return ((raw + round_to - 1) // round_to) * round_to


class TransformerLayer(nn.Module):
    """Single Transformer decoder layer.

    Compatible with Rust TransformerLayer.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_groups: int = 0,
        head_dim: int | None = None,
        ffn_expansion: float = 4.0,
        use_swiglu: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        norm_eps: float = 1e-5,
        ffn_round_to: int = 64,
        use_sparse_attn: bool = False,
        num_selected_blocks: int = 16,
        use_mla: bool = False,
        mla_d_c: int | None = None,
        mla_d_c1: int | None = None,
        mla_d_rotate: int | None = None,
        mla_block_size: int = 128,
    ):
        super().__init__()

        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads

        # Pre-attention norm
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        # MLA + GQA or standard GQA (no sparse in MLA project)
        self.use_mla = use_mla
        if use_mla:
            self.attention = MultiHeadLatentAttentionGQA(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
                d_c=mla_d_c, d_c1=mla_d_c1, d_rotate=mla_d_rotate,
                block_size=mla_block_size,
            )
        else:
            self.attention = Attention(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
            )
        self.use_sparse_attn = False
        # Pre-FFN norm
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)
        # FFN
        inter_dim = compute_intermediate_dim(d_model, ffn_expansion, use_swiglu, ffn_round_to)
        if use_swiglu:
            self.ffn = SwiGLUFFN(d_model, inter_dim, ffn_dropout, bias)
        else:
            self.ffn = StandardFFN(d_model, inter_dim, ffn_dropout, bias)
        # Residual dropout
        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()

        # Store config for reference
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Forward pass with Pre-Norm residual connections."""
        # 1. Pre-Norm -> Attention -> Residual
        residual = x
        h = self.attn_norm(x)
        h = self.attention(h, offset)
        h = self.residual_dropout(h)
        x = residual + h

        # 2. Pre-Norm -> FFN -> Residual
        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.residual_dropout(h)
        return residual + h

    def forward_with_cache(
        self,
        x: torch.Tensor,
        offset: int,
        cache: KVCache | None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Forward with KV cache."""
        # 1. Pre-Norm -> Attention with cache -> Residual
        residual = x
        h = self.attn_norm(x)
        h, new_cache = self.attention.forward_with_cache(h, offset, cache)
        h = self.residual_dropout(h)
        x = residual + h

        # 2. Pre-Norm -> FFN -> Residual
        residual = x
        h = self.ffn_norm(x)
        h = self.ffn(h)
        h = self.residual_dropout(h)
        return residual + h, new_cache


class Transformer(nn.Module):
    """Transformer stack with N identical layers + final RMSNorm.

    Compatible with Rust Transformer struct.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        num_kv_groups: int = 0,
        head_dim: int | None = None,
        ffn_expansion: float = 4.0,
        use_swiglu: bool = True,
        max_seq_len: int = 2048,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        causal: bool = True,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        attn_logit_cap: float | None = None,
        bias: bool = False,
        norm_eps: float = 1e-5,
        ffn_round_to: int = 64,
        use_sparse_attn: bool = False,
        num_selected_blocks: int = 16,
        use_mla: bool = False,
        mla_d_c: int | None = None,
        mla_d_c1: int | None = None,
        mla_d_rotate: int | None = None,
        mla_block_size: int = 128,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_groups=num_kv_groups,
                head_dim=head_dim,
                ffn_expansion=ffn_expansion,
                use_swiglu=use_swiglu,
                max_seq_len=max_seq_len,
                rope_base=rope_base,
                rope_scaling=rope_scaling,
                causal=causal,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                attn_logit_cap=attn_logit_cap,
                bias=bias,
                norm_eps=norm_eps,
                ffn_round_to=ffn_round_to,
                use_sparse_attn=use_sparse_attn,
                num_selected_blocks=num_selected_blocks,
                use_mla=use_mla,
                mla_d_c=mla_d_c,
                mla_d_c1=mla_d_c1,
                mla_d_rotate=mla_d_rotate,
                mla_block_size=mla_block_size,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model, eps=norm_eps)
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_sparse_attn = use_sparse_attn
        self.use_mla = use_mla

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Forward through all layers + final norm."""
        for layer in self.layers:
            x = layer(x, offset)
        return self.final_norm(x)

    def forward_with_cache(
        self,
        x: torch.Tensor,
        offset: int,
        caches: list[KVCache | None],
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Forward through all layers with per-layer KV caches."""
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            x, new_cache = layer.forward_with_cache(x, offset, cache)
            new_caches.append(new_cache)
        return self.final_norm(x), new_caches
