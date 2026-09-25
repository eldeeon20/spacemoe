"""Benchmark: SparseAttentionMio vs GQA — velocidad y memoria."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))

torch.manual_seed(0)

from sparse_attn_mio import SparseAttentionMio
from attention import Attention

device = "cpu"
seq_lens = [1024, 2048, 4096, 8192]
batch = 1
d_model = 768
nh, nk, hd = 12, 4, 64

def mem():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() // 1024 // 1024
    import psutil
    return psutil.Process().memory_info().rss // 1024 // 1024

print(f"{'seq':>6} | {'GQA ms':>8} {'GQA MB':>8} | {'MIO ms':>8} {'MIO MB':>8} | {'ratio':>6}")
print("-" * 55)

try:
    import psutil
except ImportError:
    psutil = None

for S in seq_lens:
    x = torch.randn(batch, S, d_model)

    m0 = mem() if psutil else 0
    t0 = time.perf_counter()
    gqa = Attention(d_model=d_model, num_heads=nh, num_kv_groups=nk, head_dim=hd)(x)
    tg = time.perf_counter() - t0
    mg = (mem() - m0) if psutil else 0

    m0 = mem() if psutil else 0
    t0 = time.perf_counter()
    mio = SparseAttentionMio(d_model=d_model, num_heads=nh, num_kv_groups=nk, head_dim=hd)(x)
    tm = time.perf_counter() - t0
    mm = (mem() - m0) if psutil else 0

    gqa_str = f"{tg*1000:8.1f} {mg:8d}" if psutil else f"{tg*1000:8.1f} {'?':>8}"
    mio_str = f"{tm*1000:8.1f} {mm:8d}" if psutil else f"{tm*1000:8.1f} {'?':>8}"
    print(f"{S:6d} | {gqa_str} | {mio_str} | {tm/tg:6.2f}x")

print("\nOutput difference (MIO vs GQA, se espera diferente — mecanismos distintos):")
x = torch.randn(1, 256, d_model)
with torch.no_grad():
    o1 = Attention(d_model=d_model, num_heads=nh, num_kv_groups=nk, head_dim=hd)(x)
    o2 = SparseAttentionMio(d_model=d_model, num_heads=nh, num_kv_groups=nk, head_dim=hd)(x)
print(f"GQA mean={o1.mean().item():.4f} std={o1.std().item():.4f}")
print(f"MIO mean={o2.mean().item():.4f} std={o2.std().item():.4f}")
print(f"diff max={(o1-o2).abs().max().item():.4f}")

p1 = sum(p.numel() for p in Attention(d_model, nh, nk, hd).parameters())
p2 = sum(p.numel() for p in SparseAttentionMio(d_model, nh, nk, hd).parameters())
print(f"\nParams: GQA={p1:,} MIO={p2:,} ratio={p2/p1:.2f}x")
