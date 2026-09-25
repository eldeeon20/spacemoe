"""Transformer decoder block with MoE support + hybrid layer dispatch.

Layer types:
  - Dense:  PreNorm → Attention → Residual → PreNorm → DenseFFN → Residual
  - MoE:    PreNorm → Attention → Residual → PreNorm → MoE → Residual

Supports hybrid architectures (first N / last N dense, middle MoE).

Depth-scaling init on output projections for stability at any depth.
"""

import math
import torch
import torch.nn as nn

from attention import Attention
from mla_attention import MultiHeadLatentAttentionGQA
from block import RMSNorm, compute_intermediate_dim
from moe import MoELayer, DenseFFN
from rope import RoPE, apply_rope_partial
from attention import repeat_kv, KVCache


class LayerWithMoE(nn.Module):
    """Single transformer layer with MoE or Dense FFN.

    Same interface as TransformerLayer in block.py, but replaces
    the hardcoded FFN with MoELayer when use_moe=True.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        num_kv_groups=0,
        head_dim=None,
        ffn_expansion=4.0,
        use_swiglu=True,
        max_seq_len=2048,
        rope_base=10000.0,
        rope_scaling=1.0,
        causal=True,
        attn_dropout=0.0,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        attn_logit_cap=None,
        bias=False,
        norm_eps=1e-5,
        ffn_round_to=64,
        use_mla=False,
        use_xsa=False,
        qk_norm=True,
        use_sandwich_norm=False,
        mla_d_c=None,
        mla_d_c1=None,
        mla_d_rotate=None,
        mla_block_size=128,
        use_moe=True,
        n_experts=8,
        top_k=2,
        n_shared=1,
        expert_dim=None,
        capacity_factor=1.35,
        z_loss_gamma=0.001,
        bias_decay=0.1,
        noise_std=0.005,
        layer_idx=0,
        num_layers=1,
    ):
        super().__init__()

        if num_kv_groups == 0:
            num_kv_groups = num_heads
        if head_dim is None:
            head_dim = d_model // num_heads

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.causal = causal
        self.use_moe = use_moe
        self.use_mla = use_mla
        self.use_xsa = use_xsa
        self.use_sandwich_norm = use_sandwich_norm
        self.attn_logit_cap = attn_logit_cap
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Pre-norms
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps)

        # Attention (MLA or standard GQA)
        if use_mla:
            self.attention = MultiHeadLatentAttentionGQA(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
                d_c=mla_d_c, d_c1=mla_d_c1, d_rotate=mla_d_rotate,
                block_size=mla_block_size,
                use_xsa=use_xsa, qk_norm=qk_norm,
            )
        else:
            self.attention = Attention(
                d_model=d_model, num_heads=num_heads, num_kv_groups=num_kv_groups,
                head_dim=head_dim, max_seq_len=max_seq_len, rope_base=rope_base,
                rope_scaling=rope_scaling, causal=causal, dropout=attn_dropout,
                attn_logit_cap=attn_logit_cap, bias=bias,
            )

        # FFN or MoE
        inter_dim = compute_intermediate_dim(d_model, ffn_expansion, use_swiglu, ffn_round_to)
        if use_moe:
            self.ffn = MoELayer(
                d_model=d_model, n_experts=n_experts, top_k=top_k,
                n_shared=n_shared, expert_dim=expert_dim or inter_dim,
                capacity_factor=capacity_factor, z_loss_gamma=z_loss_gamma,
                bias_decay=bias_decay, bias=bias, noise_std=noise_std,
            )
        else:
            self.ffn = DenseFFN(d_model, inter_dim, ffn_dropout, bias)

        self.residual_dropout = nn.Dropout(residual_dropout) if residual_dropout > 0.0 else nn.Identity()

        # Sandwich norm: extra RMSNorm after each sub-layer
        if use_sandwich_norm:
            self.post_attn_norm = RMSNorm(d_model, eps=norm_eps)
            self.post_ffn_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, offset=0, width=None):
        residual = x
        h = self.attn_norm(x)
        h = self.attention(h, offset)
        h = self.residual_dropout(h)
        x = residual + h
        if self.use_sandwich_norm:
            x = self.post_attn_norm(x)

        residual = x
        h = self.ffn_norm(x)
        if self.use_moe:
            h, aux_loss = self.ffn(h, width)
        else:
            h = self.ffn(h)
            aux_loss = 0.0
        h = self.residual_dropout(h)
        x = residual + h
        if self.use_sandwich_norm:
            x = self.post_ffn_norm(x)
        return x, aux_loss

    def forward_with_cache(self, x, offset, cache, width=None):
        residual = x
        h = self.attn_norm(x)
        h, new_cache = self.attention.forward_with_cache(h, offset, cache)
        h = self.residual_dropout(h)
        x = residual + h
        if self.use_sandwich_norm:
            x = self.post_attn_norm(x)

        residual = x
        h = self.ffn_norm(x)
        if self.use_moe:
            h, aux_loss = self.ffn(h, width)
        else:
            h = self.ffn(h)
        h = self.residual_dropout(h)
        x = residual + h
        if self.use_sandwich_norm:
            x = self.post_ffn_norm(x)
        return x, new_cache


def _resolve_per_layer(val, num_layers, default=0):
    """Convert int or list to per-layer list. None → [default]*num_layers."""
    if val is None:
        return [default] * num_layers
    if isinstance(val, (int, float)):
        return [val] * num_layers
    return list(val)


class MoETransformer(nn.Module):
    """Transformer stack with hybrid dense/MoE layer support.

    Architecture:
      - First `n_dense_start` layers: Dense
      - Middle layers: MoE (if use_moe is True)
      - Last `n_dense_end` layers: Dense

    Per-layer config: `n_experts` and `expert_dim` can be int (same for all
    MoE layers) or list[int] (one per layer). Dense layers ignore these.

    This hybrid design is critical: first layers process raw embeddings,
    last layers prepare output logits; both benefit from dense computation.
    """

    def __init__(self, num_layers, d_model, num_heads, num_kv_groups=0,
                 head_dim=None, ffn_expansion=4.0, use_swiglu=True,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, attn_dropout=0.0, ffn_dropout=0.0,
                 residual_dropout=0.0, attn_logit_cap=None, bias=False,
                 norm_eps=1e-5, ffn_round_to=64,
                 use_mla=False, use_xsa=False, qk_norm=True, use_sandwich_norm=False,
                 mla_d_c=None, mla_d_c1=None,
                 mla_d_rotate=None, mla_block_size=128,
                 use_moe=True, n_experts=8, top_k=2, n_shared=1,
                 expert_dim=None, capacity_factor=1.35, z_loss_gamma=0.001,
                 bias_decay=1e-3, noise_std=0.005,
                 n_dense_start=3, n_dense_end=3):
        super().__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        # Resolve per-layer configs
        n_experts_list = _resolve_per_layer(n_experts, num_layers, 8)
        expert_dim_list = _resolve_per_layer(expert_dim, num_layers)

        layers = []
        for i in range(num_layers):
            is_dense = i < n_dense_start or i >= num_layers - n_dense_end
            use_moe_this = use_moe and not is_dense

            layers.append(LayerWithMoE(
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
                use_mla=use_mla,
                use_xsa=use_xsa,
                qk_norm=qk_norm,
                use_sandwich_norm=use_sandwich_norm,
                mla_d_c=mla_d_c, mla_d_c1=mla_d_c1,
                mla_d_rotate=mla_d_rotate, mla_block_size=mla_block_size,
                use_moe=use_moe_this,
                n_experts=n_experts_list[i], top_k=top_k, n_shared=n_shared,
                expert_dim=expert_dim_list[i] if expert_dim_list[i] is not None else None,
                capacity_factor=capacity_factor,
                z_loss_gamma=z_loss_gamma, bias_decay=bias_decay,
                noise_std=noise_std,
                layer_idx=i, num_layers=num_layers,
            ))
        self.layers = nn.ModuleList(layers)
        self.final_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x, offset=0, width=None):
        aux_losses = []
        for layer in self.layers:
            x, aux = layer(x, offset, width)
            aux_losses.append(aux)
        return self.final_norm(x), sum(aux_losses)

    def forward_with_cache(self, x, offset, caches, width=None):
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            x, new_cache = layer.forward_with_cache(x, offset, cache, width)
            new_caches.append(new_cache)
        return self.final_norm(x), new_caches
