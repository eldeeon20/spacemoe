# Granite — FFN de 5 posiciones fijas

Denso de 10 capas (`granite/train.py`) con FFN Granite (`granite/granite.py`).
Sin MoSE y sin escalera: un solo forward + CE por step, siempre 10/10.

## Diseño

La FF base es 100% = 5 bloques de 20%::

                      ┌─ fijo compartido (siempre activo)
                      │
    [ F ][ R1 ][ R2 ][ R3 ][ R4 ]
     20%   40%   40%   40%   40%

    F  = 1 bloque de 20%, siempre activo (R0 = 20% siempre).
    Rk = 2 bloques de 20% = 40%, según router (top-2 módulos).

Posiciones virtuales::

    C1  = 20%  fijo
    C2  = 20%  → M1 + M2
    C3  = 20%  → M3 + M4
    C4  = 20%  → M5 + M6
    C5  = 20%  → M7 + M8

## Qué pasa por token

Si el router elige m1 y m2 (casilla C2), ambas trabajan en paralelo
sobre x y COMPARTEN EL MISMO LUGAR: se combinan en la posición C2::

                 ┌─────┼─────┐
                 ↓     ↓     ↓
                C1     M1     M2      (20% c/u, en paralelo)
                │     └──┬───┘
                │        ↓
                │   c2 = w1*M1 + w2*M2   (sigue siendo 20%)
                ↓        ↓
               C1        C2              (dos celdas físicas de 20%)
                └───┬───┘
                    ↓
            h = cat([c1, c2])  →  40% físico
                    ↓
               down único → y

## Ancho físico vs virtual

- **Físico** (lo que vive): C1 20% + C2 20% = **40%**.
- **Virtual** (lo que se calcula): C1 20% + M1 20% + M2 20% = **60%**.

Es decir: la posición C2 es 20%; el cálculo dentro de C2 es 40%
(M1 + M2). M1/M2 no ocupan dos posiciones físicas: ambos están
asociados a la misma posición C2. La parte de la posición 2 es
siempre esa posición: nada se reordena.

## Router

- `Linear(dim → 8)`, una score por módulo (todos menos el fijo;
  el fijo no lleva router).
- `softmax` + `topk(k=2)` + renormalización (w1 + w2 = 1).
- Ruido gaussiano en train (NoisyTopK) + bias por módulo con feedback
  (balanceo loss-free, como MoELayer).
- Aux: z-loss sobre logits SIN bias + load-balance hacia uniforme;
  se suma a la CE (`last_aux_live`). Stats `last_counts`/`balance_str()`
  para el log. El gradiente también le llega por la CE vía w1/w2.

## Módulos (`granite.py`, clase `GraniteFF`)

| Pieza | Qué es |
|---|---|
| `h20 = base // 5` | 20% de la base (409 con base 2048) |
| `up_fixed` | `Linear(dim, h20)`: C1, siempre activo |
| `up_moe` ×8 | `Linear(dim, h20)`: M1–M8, candidatos del router |
| `router` | `Linear(dim, 8)`: top-2 |
| `down` | `Linear(h20*2, dim)`: down único sobre `[c1, c2]` |
| `forward(x)` | `(…, dim) → (…, dim)` (drop-in de capa densa) |

Sin bias en ningún Linear. Activación silu (no SwiGLU).
`h = cat([c1, c2])` = 40% físico → `down(h)`.

NOTA DE IMPLEMENTACIÓN: `ModuleList` no acepta un tensor como índice
(`up_moe[indices]` revienta con batch), así que el forward despacha
por módulo (los tokens que lo eligieron). La matemática es la misma:
`c2 = w1·silu(m1) + w2·silu(m2)`.

## Train (`train.py`, 10 capas)

- Copia de `spacemoe/train.py`: mismos datos (tuit/bloques), tokenizer
  compartido del padre, MLA, mismos LR/batch/seq/optimizador.
- Sin MoSE (un forward + CE, sin anchos) ni escalera (sin máscara/KL).
- Construye `TransformerLM(use_moe=False)` e injerta
  `GraniteFF(d_model, 4*d_model)` por capa (todo dentro de `granite/`).
- `d_model = 512`, `num_layers = 10`, `base = 2048`, `h20 = 409`.
- Loss = CE (sin aux). Reporte cada 10 steps + generate cada 50.
- Ckpt `checkpoint_granite.pt`, rama `granite` (no pisa spacemoe).

## Correr

```bash
cd spacemoe
python granite/train.py      # precisión: a | bloque: el que toque
python train-data.py ...     # .txt suelto = test_mode (como spacemoe)
```

## Números (dim=512, base=2048)

- Params FF/capa ≈ 2.1M (~1.0× del denso SwiGLU 1408).
- Activo/token ≈ C1 + 2 módulos + down ≈ 60% virtual / 40% físico.
