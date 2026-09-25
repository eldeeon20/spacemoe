"""Transformer Language Model — supports GQA, MLA, MoE, hybrid layers.

Architecture:
  Embedding -> Transformer(N layers) -> Linear -> logits

Features:
  - GQA with configurable num_heads / num_kv_groups
  - MLA with Q/KV compression, RMSNorm latents, decoupled scoring
  - MoE with bmm experts, shared experts, bias trick, z-loss, capacity
  - Hybrid layers (first/last N dense, middle MoE)
  - RoPE with partial rotation support
  - x0 injection (learned scalar per layer)
  - Depth-scaled init for stability at any depth
  - KV cache for autoregressive generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from block import Transformer
from moe_block import MoETransformer
from cache_kv import KVCache
from attention import repeat_kv
from rope import apply_rope_partial


class TiedHead(nn.Module):
    """Head anclado al embedding — weight tying forzado, no se puede sacar."""
    def __init__(self, embedding):
        super().__init__()
        # Bypass nn.Module tracking — solo referencia, no parámetro ni submódulo
        object.__setattr__(self, "_emb_mod", embedding)
    def forward(self, x):
        return x @ self._emb_mod.weight.T


class TransformerLM(nn.Module):
    """Full Transformer Language Model.

    Compatible with Rust TransformerLM.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
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
        use_x0: bool = False,
        use_sparse_attn: bool = False,
        num_selected_blocks: int = 16,
        use_mla: bool = False,
        use_xsa: bool = False,
        qk_norm: bool = True,
        use_sandwich_norm: bool = False,
        mla_d_c: int | None = None,
        mla_d_c1: int | None = None,
        mla_d_rotate: int | None = None,
        mla_block_size: int = 128,
        # MoE options
        use_moe: bool = True,
        n_experts: int | list[int] = 8,
        top_k: int = 2,
        n_shared: int = 1,
        expert_dim: int | list[int] | None = None,
        capacity_factor: float = 1.25,
        z_loss_gamma: float = 0.001,
        bias_decay: float = 1e-3,
        noise_std: float = 0.01,
        n_dense_start: int = 3,
        n_dense_end: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_moe = use_moe

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.use_moe = use_moe

        # Depth-scaled init on embedding
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        if use_moe:
            self.transformer = MoETransformer(
                num_layers=num_layers,
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
                use_moe=use_moe,
                n_experts=n_experts, top_k=top_k, n_shared=n_shared,
                expert_dim=expert_dim,
                capacity_factor=capacity_factor,
                z_loss_gamma=z_loss_gamma, bias_decay=bias_decay,
                noise_std=noise_std,
                n_dense_start=n_dense_start, n_dense_end=n_dense_end,
            )
        else:
            self.transformer = Transformer(
            num_layers=num_layers,
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
            mla_d_c=mla_d_c, mla_d_c1=mla_d_c1,
            mla_d_rotate=mla_d_rotate, mla_block_size=mla_block_size,
        )
        # Head anclado al embedding — no se puede desactivar
        self.head = TiedHead(self.embedding)

        # x0 injection: learned scalar per layer
        if use_x0:
            self.x0_lambdas = nn.Parameter(torch.zeros(1, num_layers))
        else:
            self.x0_lambdas = None

    def forward(self, input_ids: torch.Tensor, width=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard forward (for training, no cache).

        Args:
            input_ids: (batch, seq_len) - token indices
            width: ancho global MoSE (None = router libre, inferencia)
        Returns:
            (logits, aux_loss): (batch, seq_len, vocab_size), scalar tensor
            aux_loss is the z-loss from MoE routers (0 if no MoE).
        """
        x = self.embedding(input_ids)
        if self.use_moe:
            x, aux_loss = self.transformer(x, 0, width)
        else:
            x = self.transformer(x, 0)
            aux_loss = 0.0
        return self.head(x), aux_loss

    def forward_train_partial_rope(
        self,
        input_ids: torch.Tensor,
        rotary_pct: float = 0.5,
        width=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward with partial RoPE + x0 injection."""
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        x0 = x.clone()
        h = x
        aux_loss = torch.tensor(0.0, device=h.device)

        for i, layer in enumerate(self.transformer.layers):
            residual = h
            h_norm = layer.attn_norm(h)

            if layer.use_mla:
                h_attn = layer.attention(h_norm)
                h = residual + h_attn
            elif layer.use_sparse_attn:
                q, k, v = layer.attention.qkv(h_norm)

                q, k = apply_rope_partial(
                    q, k, 0, rotary_pct,
                    layer.attention.rope.inv_freq,
                    layer.attention.rope.cos_cache,
                    layer.attention.rope.sin_cache,
                    layer.attention.head_dim,
                    layer.attention.rope.max_seq_len,
                )

                k = repeat_kv(k, layer.num_heads, layer.num_kv_groups)
                v = repeat_kv(v, layer.num_heads, layer.num_kv_groups)

                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

                scale = math.sqrt(layer.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale

                if seq_len > 1:
                    mask = torch.triu(
                        torch.full((seq_len, seq_len), float("-inf"), device=h.device),
                        diagonal=1,
                    )
                    scores = scores + mask.unsqueeze(0).unsqueeze(0)

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = layer.attention.attn_dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)

                attn_output = attn_output.transpose(1, 2)
                h_attn = layer.attention.o_proj(attn_output)
                h = residual + h_attn

            # FFN / MoE
            residual = h
            h_norm = layer.ffn_norm(h)
            if layer.use_moe:
                ffn_out, moe_aux = layer.ffn(h_norm, width)
                aux_loss = aux_loss + moe_aux
            else:
                ffn_out = layer.ffn(h_norm)
                ffn_out = ffn_out[0] if isinstance(ffn_out, tuple) else ffn_out
            h = residual + ffn_out

            # x0 injection
            if self.x0_lambdas is not None:
                lam = self.x0_lambdas[0, i]
                h = h + lam * x0

        h = self.transformer.final_norm(h)
        return self.head(h), aux_loss

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        offset: int,
        caches: list[KVCache | None],
        width=None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Forward with KV cache for autoregressive generation."""
        x = self.embedding(input_ids)
        x, new_caches = self.transformer.forward_with_cache(x, offset, caches, width)
        return self.head(x), new_caches

    def forward_with_cache_partial(
        self,
        input_ids: torch.Tensor,
        offset: int,
        caches: list[KVCache | None],
        rotary_pct: float = 0.5,
        width=None,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Forward with KV cache + partial RoPE."""
        x = self.embedding(input_ids)
        h = x
        new_caches = []

        for i, (layer, cache) in enumerate(zip(self.transformer.layers, caches)):
            residual = h
            h_norm = layer.attn_norm(h)

            if layer.use_mla:
                h_attn, new_cache = layer.attention.forward_with_cache_partial(
                    h_norm, offset, cache, rotary_pct)
            else:
                q, k_new, v_new = layer.attention.qkv(h_norm)
                q, k_new = apply_rope_partial(
                    q, k_new, offset, rotary_pct,
                    layer.attention.rope.inv_freq,
                    layer.attention.rope.cos_cache,
                    layer.attention.rope.sin_cache,
                    layer.attention.head_dim,
                    layer.attention.rope.max_seq_len,
                )

                if cache is not None:
                    k_full = torch.cat([cache.cached_k, k_new], dim=1)
                    v_full = torch.cat([cache.cached_v, v_new], dim=1)
                else:
                    k_full = k_new
                    v_full = v_new

                new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())

                k_expanded = repeat_kv(k_full, layer.num_heads, layer.num_kv_groups)
                v_expanded = repeat_kv(v_full, layer.num_heads, layer.num_kv_groups)

                q = q.transpose(1, 2)
                k = k_expanded.transpose(1, 2)
                v = v_expanded.transpose(1, 2)

                scale = math.sqrt(layer.head_dim)
                scores = torch.matmul(q, k.transpose(-2, -1)) / scale

                if layer.attn_logit_cap is not None:
                    scores = torch.tanh(scores / layer.attn_logit_cap) * layer.attn_logit_cap

                q_len = q.shape[2]
                kv_len = k.shape[2]
                if layer.causal and q_len > 1:
                    mask = torch.triu(
                        torch.full((q_len, kv_len), float("-inf"), device=h.device),
                        diagonal=kv_len - q_len + 1,
                    )
                    scores = scores + mask.unsqueeze(0).unsqueeze(0)

                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = layer.attention.attn_dropout(attn_weights)
                attn_output = torch.matmul(attn_weights, v)

                attn_output = attn_output.transpose(1, 2)
                h_attn = layer.attention.o_proj(attn_output)
            h = residual + h_attn

            residual = h
            h_norm = layer.ffn_norm(h)
            if layer.use_moe:
                ffn_out = layer.ffn(h_norm, width)
            else:
                ffn_out = layer.ffn(h_norm)
            h_ffn = ffn_out[0] if isinstance(ffn_out, tuple) else ffn_out
            h = residual + h_ffn

            new_caches.append(new_cache)

        h = self.transformer.final_norm(h)
        return self.head(h), new_caches

    # ─── Weight conversion utilities ─────────────────────────────────────

    def state_dict_to_safetensors(self, path: str):
        """Save model weights in safetensors format (for HF compatibility)."""
        from safetensors.torch import save_file
        sd = self.state_dict()
        sd.pop("head.emb_weight", None)  # shared with embedding.weight
        save_file(sd, path)

    @staticmethod
    def load_from_safetensors(path: str, **model_kwargs) -> "TransformerLM":
        """Load model from safetensors file."""
        from safetensors.torch import load_file
        state = load_file(path)
        model = TransformerLM(**model_kwargs)
        # head.emb_weight excluded from save (shared with embedding.weight);
        # re-linked via TiedHead.__init__ so strict=False is safe.
        model.load_state_dict(state, strict=False)
        return model

    def export_for_rust(self, path: str, mapping_path: str = None):
        """Export weights as safetensors + mapping.json for Rust loading.

        Args:
            path: output .safetensors file
            mapping_path: output mapping.json (default: same dir, _mapping.json suffix)
        """
        import json
        from safetensors.torch import save_file

        state = self.state_dict()
        save_file(state, path)

        if mapping_path is None:
            mapping_path = path.replace(".safetensors", "_mapping.json")

        mapping = {}
        for name, tensor in state.items():
            mapping[name] = {
                "shape": list(tensor.shape),
                "dtype": "f32",
                "burn_name": name,
            }

        with open(mapping_path, "w") as f:
            json.dump({"parameters": mapping}, f, indent=2)

        print(f"Exported {len(state)} tensors to {path}")
        print(f"Mapping saved to {mapping_path}")
        print("Load in Rust with: PyTorchLoader::load_safetensors(\"{path}\", \"{mapping_path}\", &device)")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        use_partial_rope: bool = False,
        rotary_pct: float = 0.5,
        width=None,
    ) -> torch.Tensor:
        """Autoregressive text generation with KV cache.

        Args:
            input_ids: (batch, prompt_len)
            max_new_tokens: number of tokens to generate
            width: ancho MoSE forzado (None = router libre, inferencia)
        Returns:
            (batch, prompt_len + max_new_tokens)
        """
        B = input_ids.shape[0]
        device = input_ids.device
        num_layers = self.num_layers

        # Get head_dim from first layer
        head_dim = self.transformer.layers[0].head_dim
        num_kv_groups = self.transformer.layers[0].num_kv_groups

        # Initialize empty caches
        caches = [None] * num_layers
        generated = input_ids.clone()

        # Prefill: codifica el prompt completo (posiciones 0..L-1) una vez.
        # Sin esto, el cache quedaba vacio y se ignoraba el prompt.
        prompt_len = generated.shape[1]
        if use_partial_rope:
            logits, caches = self.forward_with_cache_partial(
                generated, 0, caches, rotary_pct, width
            )
        else:
            logits, caches = self.forward_with_cache(
                generated, 0, caches, width
            )
        offset = prompt_len

        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    prev_tokens = generated[b].tolist()
                    for t in set(prev_tokens):
                        next_logits[b, t] /= repetition_penalty

            # Temperature
            next_logits = next_logits / max(temperature, 1e-6)

            # Top-K
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
                next_logits[indices_to_remove] = float("-inf")

            # Top-P
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            # Forwardea solo el token nuevo en su posicion absoluta.
            if use_partial_rope:
                logits, caches = self.forward_with_cache_partial(
                    generated[:, -1:], offset, caches, rotary_pct, width
                )
            else:
                logits, caches = self.forward_with_cache(
                    generated[:, -1:], offset, caches, width
                )
            offset += 1

        return generated
