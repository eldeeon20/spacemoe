# MLA: `mla/` vs `LLM_D3` + MoE Routing

## MLA — Diferencias

Ambos implementan Multi-Head Latent Attention con latentes compartidos. La diferencia es estructural:

| Componente | `mla/` | `LLM_D3` |
|---|---|---|
| Down-projection | `W_c1` (Q), `W_c` (KV), `W_rotate` — **3 matrices separadas** | `W_down` — **1 matriz fusionada** `d_model → d_c1 + d_c + d_rotate` |
| Q up-project | `W_up_q`: `d_c1 → num_heads × (head_dim + d_rotate)` | `W_up_q`: `d_c1 → d_model + num_heads × d_rotate` |
| KV up-project | `W_up_kv`: `d_c → 2 × kv_groups × head_dim` **(GQA)** | `W_up_kv`: `d_c → d_model + d_model` **(MHA, full heads)** |
| K_rotate | Compartido entre grupos: `d_rotate` | Por head: `num_heads × d_rotate` |
| Cache latente | `(C_KV, K_rotate_raw)` — **sí**, implementado | No, expande todo cada forward |
| Weight tying | `TiedHead` con `@ emb_weight.T` | `lm_head.weight = wte.weight` (PyTorch reference) |

### ¿Cuál es mejor?

Son equivalentes en cómputo — `W_down` fusionada vs separada da exactamente los mismos pesos y FLOPs. La diferencia real es **GQA vs MHA**:

- **LLM_D3** (MHA): más parámetros, mejor calidad por head, cache más grande
- **mla/** (GQA): menos parámetros, cache 4× más chico, compresión por grupos

### Ejemplo de cache por token (`d_model=768, h=12, kv=4, d_rot=64`)

| | MHA (LLM_D3) | GQA (mla/) |
|---|---|---|
| K_state | 768 | 256 |
| V_state | 768 | 256 |
| K_rotate | 768 | 64 |
| **Total** | **2304** | **576** |

---

## MoE — Routing y Arquitectura (LLM_D3)

### Qué capas usan MoE

24 capas totales. `stride=3`:

```
Capas  0-2:  MLP denso (SwiGLU)
Capas  3-20: MOELayer (top-2, 6 expertos)
Capas 21-23: MLP denso (SwiGLU)
```

Las primeras 3 y últimas 3 son densas por estabilidad. Las 18 intermedias son MoE.

### Router (top-2 con noisy gating)

1. **Proyección**: `w_g: d_model → n_exp` (logits por experto)
2. **Ruido** (solo training): `w_noise → softplus → × randn` se suma a logits
3. **Top-2**: selecciona los 2 expertos con mayor logit
4. **Softmax** solo sobre los 2 elegidos (no todos los expertos)
5. **Capacity**: `floor(top_k × capacity_factor × tokens / n_exp)`, mínimo 8. Tokens que exceden capacity se dropean.

### Dispatch — sin loops

```
tokens: [B×T, d_model]
exp_mask: [n_exp, capacity, B×T]  (one-hot de experto asignado + posición en capacity)
exp_batches = exp_mask @ tokens    → [n_exp, capacity, d_model]
```

Cada experto recibe sus tokens en un solo `bmm`. No hay loops por experto.

### Expertos — MLPExperts

Parámetros como tensores 3D `[n_exp, dim_in, dim_out]`:

```
c_fc:   [n_exp, d_model, 2 × expert_dim]    # SwiGLU up
c_proj: [n_exp, expert_dim, d_model]         # down
```

Forward: `x @ c_fc` → chunk → `swish(gate)` → `bmm @ c_proj`

Un solo `bmm` computa todos los expertos en paralelo.

### Agregación

```
exp_weight: [B×T, n_exp × capacity]  (router probs en posición correcta)
exp_out:    [n_exp × capacity, d_model]
output = exp_weight @ exp_out → [B×T, d_model]
```

Cada token recibe combinación ponderada de sus 2 expertos.

### Pérdidas auxiliares

Ambas se acumulan en `MANAGER` (singleton) y se suman al loss total:

- **Load balancing loss** (`aux_loss_weight=0.01`): `n_exp × sum(prob_per_expert × tokens_per_expert)`. Penaliza distribución desigual de tokens entre expertos.
- **Router z-loss** (`router_z_loss_weight=0.001`): `logsumexp(logits)².mean()`. Penaliza logits muy grandes para estabilidad numérica.

### Config

```python
n_exp = 6
top_k = 2
expert_dim = d_model × 2    # 1280
train_capacity = 1.25       # factor de capacidad en training
eval_capacity = 2.0         # más capacidad en eval (menos dropout)
use_aux_loss = True
use_router_z_loss = True
use_noisy_top_k = True
```
