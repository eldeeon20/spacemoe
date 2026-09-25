"""Mide memoria real de 1 capa MoE en training.
Uso: python measure_moe_mem.py
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe import MoELayer

d_model, expert_dim, n_experts, top_k, n_shared = 512, 1024, 4, 2, 1
seq_len, batch_size = 512, 4
device = torch.device("cuda")

# Crear MoE y optimizador
moe = MoELayer(d_model, n_experts, top_k, n_shared, expert_dim,
               capacity_factor=1.25, z_loss_gamma=0.0001, noise_std=0.01).to(device)
opt = torch.optim.AdamW(moe.parameters(), lr=1e-4)

torch.cuda.reset_peak_memory_stats(device)
base = torch.cuda.memory_allocated(device)

# Warmup
x = torch.randn(2, 64, d_model, device=device); moe(x)

# --- Medir forward ---
torch.cuda.reset_peak_memory_stats(device)
x = torch.randn(batch_size, seq_len, d_model, device=device)
y, aux = moe(x)
loss = y.sum() + aux
fwd_cur = torch.cuda.memory_allocated(device) - base
fwd_peak = torch.cuda.max_memory_allocated(device) - base

# --- Medir backward ---
torch.cuda.reset_peak_memory_stats(device)
loss.backward()
bwd_cur = torch.cuda.memory_allocated(device) - base
bwd_peak = torch.cuda.max_memory_allocated(device) - base

# --- Medir optimizer ---
torch.cuda.reset_peak_memory_stats(device)
opt.step(); opt.zero_grad()
opt_cur = torch.cuda.memory_allocated(device) - base

params_mem = sum(p.numel() * p.element_size() for p in moe.parameters())
grads_mem = sum(p.grad.numel() * p.grad.element_size() for p in moe.parameters() if p.grad is not None)
act_mem = fwd_cur - params_mem
adam_mem = opt_cur - params_mem - grads_mem

print(f"Params:      {params_mem/1024**2:7.1f} MB")
print(f"Gradients:   {grads_mem/1024**2:7.1f} MB")
print(f"AdamW:       {adam_mem/1024**2:7.1f} MB")
print(f"Activations: {act_mem/1024**2:7.1f} MB  (fwd: {fwd_cur/1024**2:.1f} - params: {params_mem/1024**2:.1f})")
print(f"TOTAL:       {(params_mem+grads_mem+adam_mem+act_mem)/1024**2:7.1f} MB")
print(f"Peak bwd:    {bwd_peak/1024**2:7.1f} MB")
