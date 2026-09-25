# Cambios MLA vs Standard GQA (python/)

## Atención

| Componente | `python/` (Standard GQA) | `mla/` (MLA + GQA) |
|---|---|---|
| Q | `W_q`: `d_model → d_model` | `W_down(d_model → d_c1+NK*d_c+d_rot) + W_up_q(d_c1 → d_model+NH*d_rot)` |
| K | `W_k`: `d_model → NK*hd` | `W_down + W_up_kv(NK*d_c → 2*NK*hd)` |
| V | `W_v`: `d_model → NK*hd` | Compartido con K via `W_up_kv` |
| RoPE | `head_dim` completo | Solo `d_rotate` parcial |
| qk_dim | `head_dim` | `head_dim + d_rotate` |
| Params Q | `d_model²` | `d_model*(d_c1+NK*d_c+d_rot) + d_c1*(d_model+NH*d_rot)` |
| Params K+V | `2*d_model*NK*hd` | `(d_model*NK*d_c) + (NK*d_c)*(2*NK*hd)` |

## Latentes (MLA)

- `d_c = max(32, d_model//6)` — KV compression dim
- `d_c1 = max(32, d_model//6)` — Q compression dim
- `d_rotate = max(16, d_model//12)` — RoPE dim

## Archivos

| Archivo | `python/` | `mla/` |
|---|---|---|
| `attention.py` | `Attention` (GQA + RoPE) | — |
| `mla_attention.py` | — | `MultiHeadLatentAttentionGQA` + `QKVProjectionMLA` + `OutputProjectionMLA` |
| `block.py` | `TransformerLayer` usa `Attention` | `TransformerLayer` con `use_mla=True` crea `MultiHeadLatentAttentionGQA` |
| `model.py` | `TransformerLM` standard forward | `forward_train_partial_rope` con branch MLA |
| `rope.py` | `RoPE`, `apply_rope_partial` | Sin cambios (reutilizado) |
| `cache_kv.py` | `KVCache` | Sin cambios |
| `train.py` | Original con HF Hub + streaming | Copia con `revision="gens0mla"`, MLA por defecto |

## Forward `MultiHeadLatentAttentionGQA`

```
x → W_down → C_Q (d_c1) + C_KV (NK*d_c) + K_rotate (d_rot)
              ↓                  ↓
          W_up_q              W_up_kv
              ↓                  ↓
         Q_state (d_model)    K_state (NK*hd)
         Q_rotate (NH*d_rot)  V_state (NK*hd)

Q = cat(Q_state, Q_rotate).reshape(NH, hd+d_rot)
K = cat(K_state, expand(K_rotate, NK)).reshape(NK, hd+d_rot)
V = V_state.reshape(NK, hd)

K = repeat_kv(K, NH, NK)  # expand NK → NH
V = repeat_kv(V, NH, NK)

scores = Q @ K.T / sqrt(hd+d_rot)
out = softmax(scores, causal) @ V
out = o_proj(out.reshape(NH*hd))  # solo head_dim, no qk_dim
```

## OutputProjection

- `python/`: `nn.Linear(NH*hd, d_model)` (qk_dim = hd)
- `mla/`: **`nn.Linear(NH*hd, d_model)`** (usa `head_dim`, **NO** `qk_dim`)

Esto es porque `scaled_dot_product_attention(Q,K,V)` produce salida con la última dim de `V` = `head_dim`, no `qk_dim`.

## Weight tying — Una sola tabla entrada/salida

`embedding.weight` es **una** tabla de `(vocab_size, d_model)`. Sirve pa'las dos:

- **Entrada**: `embedding(token_id)` → lookup por ID, devuelve la fila
- **Salida**: `h @ weight.T` → producto punto contra todas las filas → logits

`head.weight = embedding.weight` hace que ambos punten al **mismo** tensor. Ni copia ni referencia distinta — es el mismo `Parameter`.

No se duplica nada. El modelo aprende un único vector por token, usado tanto para representarlo como para predecirlo.

### ⚠️ Inicialización del embedding

`nn.Embedding(vocab_size, d_model)` en PyTorch inicia los pesos con **N(0, 1)** (no con uniform chico como suele creerse). Con weight tying el head hereda esos pesos enormes, y los logits explotan:

- `h` después de RMSNorm tiene `σ ≈ 1`
- `head.weight` con `σ ≈ 1` → logits `σ ≈ √d_model ≈ 11.3` → loss inicial ~120+ en vez de `ln(vocab) ≈ 9.68`

**Fix:** en `model.py` se reinit el embedding con `nn.init.normal_(self.embedding.weight, mean=0, std=1/math.sqrt(d_model))`.

Con `std=1/√d_model`:
- `h` σ ≈ 1, `head.weight` σ ≈ 1/√128 ≈ 0.088
- logits σ ≈ √d_model * 1/√d_model = 1 → loss inicial ≈ ln(vocab) ≈ 9.68 ✓

## Train

- `test_mode` activo cuando se pasa `input.txt` como argumento
- En test mode: no pide HF token, no sube checkpoints, single pass
- `mla_block_size=128` pasado al constructor
- Sin flag `use_mla` — en `mla/` es el default
