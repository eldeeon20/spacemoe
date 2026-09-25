"""Multi-head Latent Attention (MLA) with GQA.

Based on DeepSeek MLA adapted for GQA.
- Latent KV compression, shared latent across KV groups
- RoPE on separate rotation dimension
- Latent cache (C_KV, K_rotate_raw)
- RMSNorm on latent latents (C_Q, C_KV)
- Decoupled scoring: content/sqrt(hd) + rope/sqrt(dr)
- N(0, 0.02) init (matching nano-moe-mla)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE, apply_rope_partial
from attention import repeat_kv


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        msq = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * msq).type_as(x) * self.weight


class QKVProjectionMLA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups, head_dim, d_c, d_c1, d_rotate, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_c = d_c
        self.d_c1 = d_c1
        self.d_rotate = d_rotate

        self.W_down = nn.Linear(d_model, d_c1 + d_c + d_rotate, bias=bias)
        self.norm_cq = RMSNorm(d_c1)
        self.norm_ckv = RMSNorm(d_c)
        self.W_up_q = nn.Linear(d_c1, num_heads * (head_dim + d_rotate), bias=bias)
        self.W_up_kv = nn.Linear(d_c, 2 * num_kv_groups * head_dim, bias=bias)

    def forward(self, x):
        B, S, _ = x.shape
        down = self.W_down(x)
        C_Q, C_KV, K_rotate = down.split([self.d_c1, self.d_c, self.d_rotate], dim=-1)

        C_Q = self.norm_cq(C_Q)
        C_KV = self.norm_ckv(C_KV)

        q_up = self.W_up_q(C_Q)
        Q_state, Q_rotate = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S, self.num_heads, self.head_dim)
        Q_rotate = Q_rotate.reshape(B, S, self.num_heads, self.d_rotate)

        kv_up = self.W_up_kv(C_KV)
        K, V = kv_up.chunk(2, dim=-1)
        K = K.reshape(B, S, self.num_kv_groups, self.head_dim)
        V = V.reshape(B, S, self.num_kv_groups, self.head_dim)
        K_rotate = K_rotate.reshape(B, S, 1, self.d_rotate)

        return Q_state, Q_rotate, K, V, K_rotate


class OutputProjectionMLA(nn.Module):
    def __init__(self, d_model, num_heads, head_dim, bias=False):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=bias)

    def forward(self, x):
        B, S, NH, QK = x.shape
        return self.o_proj(x.reshape(B, S, NH * QK))


class MultiHeadLatentAttentionGQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups=0, head_dim=None,
                 max_seq_len=2048, rope_base=10000.0, rope_scaling=1.0,
                 causal=True, dropout=0.0, attn_logit_cap=None, bias=False,
                 d_c=None, d_c1=None, d_rotate=None, block_size=128,
                 use_xsa=False, qk_norm=True):
        super().__init__()
        if num_kv_groups == 0: num_kv_groups = num_heads
        if head_dim is None: head_dim = d_model // num_heads
        if d_c is None: d_c = max(32, d_model // 6)
        if d_c1 is None: d_c1 = max(32, d_model // 6)
        if d_rotate is None: d_rotate = max(16, d_model // 12)

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = head_dim
        self.d_rotate = d_rotate
        self.causal = causal
        self.attn_logit_cap = attn_logit_cap
        self.block_size = block_size
        self.use_xsa = use_xsa
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

        self.qkv = QKVProjectionMLA(d_model, num_heads, num_kv_groups, head_dim,
                                     d_c, d_c1, d_rotate, bias)
        self.o_proj = OutputProjectionMLA(d_model, num_heads, head_dim, bias)
        self.rope = RoPE(head_dim=d_rotate, max_seq_len=max_seq_len, base=rope_base, scaling_factor=rope_scaling)
        self.rope.head_dim = d_rotate
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if causal:
            mask = torch.triu(torch.full((max_seq_len, max_seq_len), float("-inf")), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, RMSNorm):
            nn.init.ones_(m.weight)

    def _decoupled_scores(self, Q_state, Q_rot, K_state, K_rot, seq_len, kv_len=None, causal=None):
        """Decoupled content + RoPE scoring.
        
        With QK-Norm, Q and K have RMS=1 so the dot product is bounded by the
        true information dimension d_c (KV latent), not the projected head_dim.
        Using 1/sqrt(d_c) keeps variance ~hd/d_c ≈ 1.3 — slightly sharper than
        standard MHA, accounting for QK correlation via shared latent.
        Without QK-Norm, fall back to 1/sqrt(head_dim).
        """
        nh, nkv, hd, dr = self.num_heads, self.num_kv_groups, self.head_dim, self.d_rotate
        scale_c = 1.0 / math.sqrt(self.qkv.d_c) if self.qk_norm else 1.0 / math.sqrt(hd)
        scale_r = 1.0 if self.qk_norm else 1.0 / math.sqrt(dr)

        k_c = repeat_kv(K_state, nh, nkv).transpose(1, 2)
        q_c = Q_state.transpose(1, 2)
        s_c = torch.matmul(q_c, k_c.transpose(-2, -1)) * scale_c

        q_r = Q_rot.transpose(1, 2)
        k_r = K_rot.transpose(1, 2).expand(-1, nh, -1, -1)
        s_r = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale_r

        scores = s_c + s_r
        if self.attn_logit_cap is not None:
            scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        is_causal = self.causal if causal is None else causal
        if is_causal:
            if kv_len is None:
                kv_len = K_state.shape[1]
            if seq_len > 1:
                if seq_len == kv_len:
                    if seq_len <= self.causal_mask.shape[0]:
                        scores = scores + self.causal_mask[:seq_len, :seq_len]
                    else:
                        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
                        scores = scores + mask
                else:
                    mask = torch.triu(
                        torch.full((seq_len, kv_len), float("-inf"), device=scores.device),
                        diagonal=kv_len - seq_len + 1
                    )
                    scores = scores + mask
        return scores

    def forward(self, x, offset=0):
        Q_state, Q_rotate, K, V, K_rotate = self.qkv(x)
        Q_rotate, K_rotate = self.rope(Q_rotate, K_rotate, offset)
        B, T = x.shape[0], x.shape[1]

        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            K = self.k_norm(K)
        scores = self._decoupled_scores(Q_state, Q_rotate, K, K_rotate, T)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        v = repeat_kv(V, self.num_heads, self.num_kv_groups).transpose(1, 2)
        attn_out = torch.matmul(attn_w, v)  # (B, nh, T, hd)
        if self.use_xsa:
            Vn = F.normalize(v, dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn
        return self.o_proj(attn_out.transpose(1, 2))

    def _attention_from_components(self, Q_state, Q_rot, K_state, V_state, K_rot, q_len, kv_len, causal_mask):
        """Compute attention output from pre-projected components with decoupled scoring."""
        if self.qk_norm:
            Q_state = self.q_norm(Q_state)
            K_state = self.k_norm(K_state)
        scores = self._decoupled_scores(Q_state, Q_rot, K_state, K_rot, q_len, kv_len, causal=causal_mask)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = self.attn_dropout(attn_w)
        v = repeat_kv(V_state, self.num_heads, self.num_kv_groups).transpose(1, 2)
        attn_out = torch.matmul(attn_w, v)  # (B, nh, T, hd)
        if self.use_xsa:
            v_new = v[:, :, -q_len:, :]
            Vn = F.normalize(v_new, dim=-1)
            attn_out = attn_out - (attn_out * Vn).sum(dim=-1, keepdim=True) * Vn
        return attn_out.transpose(1, 2)

    def forward_with_cache(self, x, offset, cache):
        B, S_new, _ = x.shape
        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        C_Q_new = self.qkv.norm_cq(C_Q_new)
        C_KV_new = self.qkv.norm_ckv(C_KV_new)

        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.d_rotate)
        Q_rot = self.rope.apply_to_single(Q_rot_raw, offset=offset)

        if cache is not None:
            C_KV_full = torch.cat([cache[0], C_KV_new], dim=1)
            K_rot_full = torch.cat([cache[1], K_rot_raw], dim=1)
        else:
            C_KV_full = C_KV_new
            K_rot_full = K_rot_raw
        S_full = C_KV_full.shape[1]

        kv_up = self.qkv.W_up_kv(C_KV_full)
        K_state, V_state = kv_up.chunk(2, dim=-1)
        K_state = K_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)
        V_state = V_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)

        K_rot = self.rope.apply_to_single(K_rot_full.unsqueeze(2), offset=0)

        attn_out = self._attention_from_components(
            Q_state, Q_rot, K_state, V_state, K_rot, S_new, S_full, S_new > 1)
        return self.o_proj(attn_out), (C_KV_full, K_rot_full)

    def forward_with_cache_partial(self, x, offset, cache, rotary_pct):
        B, S_new, _ = x.shape
        down = self.qkv.W_down(x)
        C_Q_new, C_KV_new, K_rot_raw = down.split([self.qkv.d_c1, self.qkv.d_c, self.qkv.d_rotate], dim=-1)
        C_Q_new = self.qkv.norm_cq(C_Q_new)
        C_KV_new = self.qkv.norm_ckv(C_KV_new)

        q_up = self.qkv.W_up_q(C_Q_new)
        Q_state, Q_rot_raw = q_up.split([self.num_heads * self.head_dim, self.num_heads * self.d_rotate], dim=-1)
        Q_state = Q_state.reshape(B, S_new, self.num_heads, self.head_dim)
        Q_rot_raw = Q_rot_raw.reshape(B, S_new, self.num_heads, self.d_rotate)

        Q_rot, _ = apply_rope_partial(Q_rot_raw, Q_rot_raw, offset, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)

        if cache is not None:
            C_KV_full = torch.cat([cache[0], C_KV_new], dim=1)
            K_rot_full = torch.cat([cache[1], K_rot_raw], dim=1)
        else:
            C_KV_full = C_KV_new
            K_rot_full = K_rot_raw
        S_full = C_KV_full.shape[1]

        kv_up = self.qkv.W_up_kv(C_KV_full)
        K_state, V_state = kv_up.chunk(2, dim=-1)
        K_state = K_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)
        V_state = V_state.reshape(B, S_full, self.num_kv_groups, self.head_dim)

        _, K_rot = apply_rope_partial(K_rot_full.unsqueeze(2), K_rot_full.unsqueeze(2), 0, rotary_pct,
            self.rope.inv_freq, self.rope.cos_cache, self.rope.sin_cache,
            self.rope.head_dim, self.rope.max_seq_len)

        attn_out = self._attention_from_components(
            Q_state, Q_rot, K_state, V_state, K_rot, S_new, S_full, S_new > 1)
        return self.o_proj(attn_out), (C_KV_full, K_rot_full)
