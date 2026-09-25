# MiniMax Sparse Attention (MSA) — Report

**Repo**: `MSA-main/` (fork de `github.com/MiniMax-AI/MSA`)  
**Paper**: `docs/MiniMaxSparseAttention.pdf`  
**Stack**: CuTe-DSL + CUDA C++, NVIDIA SM100 (Blackwell, CC 10.0)

## Algoritmo

1. **Block division**: K/V sequence se divide en bloques de `blk_kv=128` tokens
2. **Block selection**: Para cada query, se seleccionan top-K bloques de KV (K=4/8/16/32). Esta selección viene de un "proxy pass" previo o de un índice externo (`q2k_indices` shape `[Hkv, total_q, topK]`)
3. **CSR metadata**: La selección se codifica como CSR sparse matrix:
   - `k2q_row_ptr[Hkv, total_rows + 1]` — filas = bloques KV
   - `k2q_q_indices[Hkv, total_q * topK]` — qué queries atienden a cada bloque KV
4. **Sparse FlashAttention**: Cada SM procesa un subset de queries + sus bloques KV seleccionados, compute block-sparse QK^T + softmax + PV

## Claves de eficiencia

| Aspecto | Detalle |
|---|---|
| Block size fijo | `blk_kv=128` — tamaño de bloque óptimo para Tensor Cores SM100 |
| CSR schedule | Precomputado, permite trabajo balanceado entre SMs |
| CuTe-DSL | DSL Python → compile-time tiling + TMA descriptors → kernel CUDA |
| Causal mask | Se maneja vía CSR: cada query solo atiende a bloques KV ≤ su posición |
| Top-K fijo | Mismo K para todas las queries, permite layouts fijos de memoria |
| Quantization | FP8 e4m3, NVFP4, FP4 con escalas por bloque |

## Layout de memoria

```
Q: [total_q, Hq, D]        — D=128, Hq = Hkv * qhead_per_kv
K, V: [total_k, Hkv, D]
q2k_indices: [Hkv, total_q, topK]  — índices de bloques KV (int32)
k2q_row_ptr: [Hkv, total_rows + 1]  — CSR pointer
k2q_q_indices: [Hkv, total_q * topK]  — CSR indices
```

## Performance

- Dense prefill: FlashAttention con tiling CuTe-DSL, same API que sparse
- Sparse prefill: block-sparse con CSR, ~2-4x más rápido que dense según sparsity
- Decode: paged KV con FP8, atención a bloque completo + sliding window
- NVFP4: K/V en 4 bits packed, descomprime on-the-fly con TMA

## Limitaciones

- Solo SM100 (Blackwell). Requiere `compute_cap == 10.0`
- D=128 fijo (head_dim)
- top-K fijo, no adaptativo
- No hay CPU fallback — requiere GPU NVIDIA
- Proxy pass para generación de índices no incluido (se pasa externamente)

## Block Selection — cómo elige cada query sus bloques

### Mean-pool compression (MSA default)

Cada bloque KV de BS=128 tokens se comprime a **un solo vector** vía mean-pool:

```python
k_comp = k_b.mean(dim=2)  # (B, NK, NB, HD)
```

Cada query hace dot-product contra ese vector comprimido:

```python
scores = q @ k_comp^T / sqrt(HD)  # similaridad bloque-query
_, topk = scores.topk(K)           # elegir K bloques
```

Intuición: el vector promedio representa el "tema" del bloque (como un centroide). Una query sobre "perros" tendrá alto score contra bloques cuyo contenido promedio sea cercano a "perros". La atención fina **dentro** del bloque usa los 128 tokens completos con sus posiciones — allí no hay pérdida.

### Alternativas de compresión

| Método | Descripción | Pros | Contras |
|--------|-------------|------|---------|
| **Mean-pool** | Promedio aritmético | Simple, barato | No captura varianza intra-bloque |
| **Max-pool** | Máximo por dimensión | Captura features extremos | Ruidoso |
| **Learned MLP** | MLP pequeño comprime bloque | Más preciso | Params extra, forward adicional |
| **Query-aware** | Pooling ponderado por query | Máxima precisión | Costo O(S²) — inviable |

Mean-pool es suficiente para selección gruesa (MSA lo usa). La atención fina dentro del bloque seleccionado recupera cualquier pérdida de información.

### GQA y selección por grupo

Con GQA (NK grupos, NH cabezas, HPG = NH/NK cabezas por grupo), la selección se hace **por grupo KV**, no por cabeza individual:

```python
q_group = q.reshape(B, NK, HPG, C, HD).mean(dim=2)  # Q promedio del grupo
scores = q_group @ k_comp^T / sqrt(HD)
_, topk = scores.topk(K)  # (B, NK, C, K) — un set de bloques por grupo
```

Las HPG cabezas del grupo **comparten el mismo conjunto de bloques**. Esto reduce el gather 3× (para NH=12, NK=4) y es consistente con GQA (cabezas del grupo ya comparten K/V). La pérdida de precisión vs selección por cabeza individual es mínima en la práctica.

## Relevancia para CPU implementation

La idea de **bloques de tamaño fijo + CSR metadata** es portable a CPU:
- Dividir K/V en bloques (BS=128 o más pequeño para CPU)
- Elegir top-K bloques por query
- Atender solo a esos bloques con SDPA estándar (aprovechando mask)
- GPU lo hace con kernels especializados; en CPU se puede emular con chunked SDPA

### Benchmark CPU (PyTorch, AMD EPYC, fp32, attention-only)

Modelo: d_model=768, NH=12, NK=4, HD=64, BS=128, RoPE aplicado.

| Config | S | Tokens/query | Tiempo | Delta vs GQA |
|--------|---|-------------|--------|-------------|
| GQA | 1024 | 1024 (100%) | 129ms | — |
| Mio3 K=1 | 1024 | 128 (12.5%) | 726ms | **5.6× peor** |
| GQA | 4096 | 4096 (100%) | 2332ms | — |
| Mio3 K=1 | 4096 | 128 (**3.1%**) | 3091ms | **1.3× peor** |
| Mio3 K=2 | 4096 | 256 (**6.2%**) | 5198ms | 2.2× peor |

En CPU con PyTorch puro, Mio3 **nunca es más rápido** que GQA. El overhead del gather `(B, NH, C, K, BS, HD)` + loop de chunks + checkpoint domina sobre el ahorro de cómputo. La brecha se cierra con S grande (5.6× → 1.3×) pero no se invierte sin kernel fusionado.

La ganancia real de memoria:
- GQA: scores intermedios de 805MB a S=4096 que SDPA maneja vía tiling (no materializa completo)
- Mio3: gather de 32MB por chunk, pico ~50MB

En TPU/GPU con **kernel fusionado** (CuTe/Triton), el gather desaparece como operación separada y el bloque seleccionado se carga directo a registros/SRAM. Ahí la ventaja de cómputo (2048 tokens vs 4096 a 4k) se traduce en velocidad real. A 64k con K=2 → 256/65536 = **0.4%** → ~10× más rápido que GQA.





Scoring	Desacoplado: (q_c@k_c + q_r@k_r) * scale	Concatenado: sdpa(Q, K, V) sobre [content, rope]
Cache KV	d_latent + d_rope (ej: 32+8=40)	d_c + d_rotate (ej: 128+64=192)
RMSNorm latentes	❌ No	❌ No
QK-Norm	✅ Opcional	❌ No
Partial RoPE	❌ No	❌ No
FFN / MoE: DIFERENCIA FUNDAMENTAL
Aspecto	nano-moe-mla	LLM-D3
MoE disponible	✅ Sí	✅ Sí
Expertos	ModuleList con loop for e in range(n_experts)	Parámetros batch (n_exp, ...) con bmm
Shared experts	✅ Sí (n_shared)	❌ No
Capacidad / Token dropping	❌ No	✅ Sí (capacity_factor)
Balanceo de carga	Bias trick (DeepSeek, sin loss)	Aux-loss (Switch Transformer) + z-loss
Ruido en routing	Optional	✅ Sí (W_noise + gaussiano)
Capas híbridas	❌ Todas MoE o todas dense	✅ Primeras 3 y últimas 2 dense, medias MoE
Init
Aspecto	nano-moe-mla	LLM-D3
Esquema	N(0, 0.02) plano en todo	Switch Transformer: truncated normal con scaling por profundidad y por tipo de capa
Output proj scaling	std=0.02 igual al resto	std = 0.02 / sqrt(2*n_layer)









Evaluación técnica: nano-moe-mla vs LLM-D3
MLA
Aspecto	Mejor	Por qué
Scoring desacoplado vs concatenado	Desacoplado (nano)	q_c@k_c/sqrt(hd) + q_r@k_r/sqrt(dr) da control independiente sobre escalas. El concatenado fuerza a sqrt(hd+dr) que penaliza la señal de contenido cuando dr es grande. DeepSeek-V3 usa desacoplado.
Compresión de Q	LLM-D3	Ahorra parámetros en Q (W_down pequeña → W_up_q). nano sin compresión gasta nh*(hd+dr)*d_model en w_q al pedo.
RMSNorm en latentes	Ninguno (ambos mal)	Deberían tenerlo. Estabiliza C_Q y C_KV antes de up-proyectar. Nuestra impl en mla_attention.py lo tiene y es mejor.
QK-Norm	nano (opcional)	Útil en modelos grandes para evitar logits de atención explosivos. DeepSeek-V3 lo usa.
Partial RoPE	Ninguno	No relevante a esta escala, pero nuestra impl lo soporta y es útil para compatibilidad con otros modelos.
MoE
Aspecto	Mejor	Por qué
Expertos: ModuleList loop vs bmm batch	LLM-D3 (bmm)	ModuleList con loop no escala a cientos de expertos. bmm con tensor batch es GPU-friendly.
Shared experts	nano	Concepto DeepSeekMoE: un experto siempre-on captura conocimiento común, los ruteados se especializan. Mejor calidad por parámetro.
Capacity / Token dropping	LLM-D3	Necesario en producción: evita que un experto se sature y reciba >capacity tokens (colapsa GPU memory).
Balanceo: bias trick vs aux-loss	nano (bias trick)	DeepSeek-V3 demostró que el bias trick (sin loss) funciona igual o mejor que aux-loss, sin contaminar el gradiente del router. Es más simple y no requiere hiperparámetro de weight.
Router z-loss	LLM-D3	Complementa al bias trick o aux-loss. Penaliza router logits grandes, estabiliza entrenamiento. Ambos deberían tenerlo.
Ruido en routing	LLM-D3	W_noise con softplus permite que el ruido se ajuste durante training, mejor que un noise_std fijo.
Capas híbridas	LLM-D3	Las primeras y últimas capas procesan tokens más "importantes" (embedding → primeros conceptos, últimas → salida). Hacerlas densas evita pérdida de información por routing. DeepSeek-V3 y Qwen2.5-MoE lo hacen.
Init
Aspecto	Mejor	Por qué
N(0,0.02) vs Switch Transformer init	LLM-D3	Depth-scaling (/sqrt(2*n_layer)) en output projections es crítico para modelos profundos (>12 layers). Sin eso, la varianza del residual stream crece linealmente con depth → logits explotan. El truncated normal además evita outliers. El init plano N(0,0.02) solo funciona en modelos chicos (<100M).
Veredicto general
Para modelos chicos (<100M params): gana nano — simple, funciona, bias trick suficiente.

Para modelos grandes (>100M params): gana LLM-D3 — depth-scaling init, bmm experts, capacity factor, capas híbridas, z-loss. Todo necesario para estabilidad a escala.

Lo ideal: combinación de ambos. Tomar la compresión de Q y QK-Norm de D3, el scoring desacoplado y bias trick de nano, y sumarle depth-scaling init + shared experts + z-loss. Eso es básicamente DeepSeek-V3.




Modelo	Loss	vs MLA dense
MLA+MoE+sh	0.917	-19%
MLA+MoE	1.090	-3%
MLA+x0	1.110	-1%
MLA dense	1.126	—
D5(XSA)	1.149	+2%
nano-mla	1.288	+14%
D3	1.283	+14%
Py GQA	3.046	+170%