"""KV Cache for autoregressive generation (compatible with Rust KVCache)."""

import torch
from dataclasses import dataclass


@dataclass
class KVCache:
    """Accumulated Key and Value tensors from previous positions.

    Shape: (batch, accumulated_seq_len, num_kv_groups, head_dim)
    Matches Rust: blocks::trasformer::attention::KVCache
    """
    cached_k: torch.Tensor  # (B, S, num_kv_groups, head_dim)
    cached_v: torch.Tensor  # (B, S, num_kv_groups, head_dim)

    @staticmethod
    def new(batch: int, num_kv_groups: int, head_dim: int, device: torch.device) -> "KVCache":
        return KVCache(
            cached_k=torch.empty(batch, 0, num_kv_groups, head_dim, device=device),
            cached_v=torch.empty(batch, 0, num_kv_groups, head_dim, device=device),
        )

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> "KVCache":
        """Concatenate new K/V with cached K/V along sequence dim (dim=1)."""
        if self.cached_k.shape[1] == 0:
            return KVCache(cached_k=k_new, cached_v=v_new)
        return KVCache(
            cached_k=torch.cat([self.cached_k, k_new], dim=1),
            cached_v=torch.cat([self.cached_v, v_new], dim=1),
        )

    def trim_prefix(self, remove: int) -> "KVCache":
        """Remove the first `remove` positions."""
        if remove == 0:
            return self
        seq = self.cached_k.shape[1]
        if remove >= seq:
            return self
        return KVCache(
            cached_k=self.cached_k[:, remove:seq, :, :],
            cached_v=self.cached_v[:, remove:seq, :, :],
        )

    def keep_last(self, keep: int) -> "KVCache":
        """Keep only the last `keep` positions."""
        seq = self.cached_k.shape[1]
        if keep >= seq:
            return self
        start = seq - keep
        return KVCache(
            cached_k=self.cached_k[:, start:seq, :, :],
            cached_v=self.cached_v[:, start:seq, :, :],
        )

    @property
    def seq_len(self) -> int:
        return self.cached_k.shape[1]

    def clone(self) -> "KVCache":
        return KVCache(
            cached_k=self.cached_k.clone(),
            cached_v=self.cached_v.clone(),
        )
