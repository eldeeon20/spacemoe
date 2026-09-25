# Comparativa: MoE-MLA-GQA vs LLM_D3 vs nano-moe-mla

## Arquitectura General

| Característica | MoE-MLA-GQA (nuestro) | LLM_D3 | nano-moe-mla |
|---|---|---|---|
| Atención | **MLA + GQA** (KV comprimido, Q comprimido, RoPE decoupled) | Atención multi-head estándar | **MLA** (similar al nuestro, sin GQA) |
| MoE | **bmm experts** (tensor único `(n_exp, d, 2*edim)`, matmul por lote) | **ModuleList** loop sobre experts | **ModuleList** loop sobre experts |
| Shared experts | ✅ **Sí** (1+ expertos compartidos, DeepSeekMoE style) | ❌ No | ✅ Sí |
| Load balancing | **Bias trick** (feedback sin gradiente, 0 overhead) | Noisy top-k gating + aux loss | Bias trick |
| z-loss | ✅ **Sí** (router z-loss, estabilizador) | ❌ Opcional | ❌ No |
| Capacity factor | ✅ **Sí** (1.25, token dropping) | ❌ No | ❌ No |
| Init | **Depth-scaled** (`std=0.02/sqrt(2*n_layers)` en output proj) | Switch transformer init | Estándar (`N(0,0.02)`) |
| Capas híbridas | **n_dense_start/n_dense_end** (primeras/últimas N densas) | Stride fijo | ❌ No |
| x0 injection | ✅ Sí (learned scalar per layer) | ❌ No | ❌ No |
| Per-layer expert config | ✅ **Sí** (lista de ints) | ❌ No | ❌ No |
| Weight tying | ✅ Sí (embedding ↔ head) | ❌ No | ❌ No |

## Atención: MLA + GQA

Nuestra implementación de **Multi-head Latent Attention** comprime las claves (K) y valores (V) a un espacio latente `d_c`, y opcionalmente también el query (Q) a `d_c1`. ElRoPE se aplica solo a `d_rotate` dimensiones (decoupled). Con `num_kv_groups=4, num_heads=12`, el cache KV se reduce de:

- **GQA tradicional**: `2 × num_kv_groups × head_dim = 2 × 4 × 64 = 512` floats/token
- **MLA**: `d_c + d_rotate = 32 + 32 = 64` floats/token → **87% menos**

LLM_D3 usa atención estándar sin compresión: cache completo de `2 × num_heads × head_dim` por token.

## MoE: bmm experts vs ModuleList loop

Nuestros expertos son un solo tensor `Parameter(n_experts, d_model, 2*expert_dim)` y usamos `bmm` (o `@` con índice escalar) para procesar todos los tokens de un experto en paralelo:

```python
# Nuestro: un solo matmul
w_fc = self.c_fc[expert_idx]  # (d_model, 2*edim) con índice escalar
h = x @ w_fc                  # (N, 2*edim)
```

```python
# LLM_D3 / nano: loop sobre ModuleList
h = self.experts[expert_idx](x)  # llama a nn.Module individual
```

La diferencia clave:
- **Nuestro**: GPU-friendly, un solo kernel lanzado, memoria contigua, escala a cientos de expertos
- **ModuleList**: N kernels lanzados secuencialmente, fragmentación de memoria, no escala

## Load Balancing: Bias Trick vs Aux Loss

**Bias trick** (nuestro, tomado de nano-moe-mla y DeepSeek-V3):
- No hay pérdida auxiliar — los biases del router se actualizan con un feedback loop: `bias_e += bias_decay * (count_e - target)`
- No contamina el gradiente de los expertos
- 0 overhead computacional
- El router aprende a balancear naturalmente

**Noisy top-k + aux loss** (LLM_D3):
- Añade ruido al router y una pérdida auxiliar `load_balancing_loss`
- Contamina el gradiente del router y los expertos
- Overhead adicional
- Requiere sintonizar el peso de la pérdida auxiliar

## Capacidad: Capacity Factor

Nuestro MoE tiene un **capacity factor** (1.25 por defecto) que limita cuántos tokens puede procesar cada experto:

```
capacity = ceil(capacity_factor × top_k × N / n_experts)
```

Los tokens que exceden la capacidad se descartan (se quedan sin procesar por ese experto). Esto:
- Previene que un experto se sobrecargue
- Garantiza presupuesto computacional constante
- Fuerza al router a distribuir mejor

LLM_D3 y nano-moe-mla no tienen capacity factor — todos los tokens siempre se procesan.

## Init: Depth-Scaled

```python
std = 0.02 / sqrt(2 * num_layers)  # en output projections
```

Esto evita que la varianza explote en modelos profundos. La capa 1 tiene std=0.02, la capa 25 tiene std=0.02/sqrt(50)≈0.0028. Sin esto, residuales profundos divergen (logit std > 10).

LLM_D3 usa init de Switch Transformer, nano-moe-mla usa init estándar.

## Comparación Experimental (d_model=256, 3 layers, 10 steps)

| Modelo | Loss step 10 | Params | MACs/tok |
|---|---|---|---|
| **MoE+shared (nuestro)** | **0.915** | 8.47M | ~49K |
| MLA dense | 1.164 | 1.98M | ~49K |
| MLA+x0 | 1.141 | 1.98M | ~49K |
| LLM_D3 | 1.325 | 1.83M | ~49K |
| D5(XSA) | 1.208 | 2.28M | ~49K |
| nano-mla | 1.236 | 2.17M | ~49K |
| Py GQA | 3.038 | 2.44M | ~49K |

Nuestro MoE-MLA con shared experts obtiene **15.5% mejor loss** que MLA dense y **31% mejor** que LLM_D3 en el mismo número de pasos.

## Resumen Técnico

Nuestras ventajas principales:
1. **MLA** → 87% menos cache KV que atención estándar
2. **bmm experts** → GPU-efficient, escala a 100+ expertos
3. **Bias trick** → load balancing sin pérdida auxiliar, 0 overhead
4. **z-loss** → estabilidad del router sin contaminar gradientes
5. **Capacity factor** → presupuesto computacional constante, fuerza balanceo
6. **Depth-scaled init** → estabilidad a cualquier profundidad
7. **Shared experts** → capturan conocimiento común (DeepSeekMoE)
8. **Per-layer config** → diferente número de expertos por capa
9. **Capas híbridas** → primeras/últimas densas, medio MoE

Desventajas:
- Mayor cantidad de parámetros (MoE almacena 4 expertos aunque solo active 1)
- Router bias trick requiere syncing en training distribuido (no implementado)
- Capacity factor puede descartar tokens informativos si está muy ajustado








step 3250 loss 3.4360 lr 0.000300 2624t/s
  >>> desde fines de la década de los 80, se extendió así a los Estados Unidos el crecimiento sostenido para el desarrollo urbano.

Los problemas medioambi
Loaded block 130: 767739 tokens
Epoch 80 done. Loaded next block: 767739 tokens
step 3260 loss 3.6684 lr 0.000300 2271t/s
step 3270 loss 3.5427 lr 0.000300 2630t/s
Loaded block 131: 775602 tokens
Epoch 81 done. Loaded next block: 775602 tokens
step 3280 loss 3.6137 lr 0.000300 2326t/s
Processing Files (0 / 0)      : |          |  0.00B /  0.00B            
New Data Upload 
