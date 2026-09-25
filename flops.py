"""flops.py — FLOPs por token de spacemoe, por profundidad (escalera) y ancho MoSE.

Cuenta FLOPs de FORWARD (2 por MAC), con las proyecciones reales de MLA y el
MoE slimmable. Las cuentas de atencion dependen de la longitud de secuencia, asi
que se reportan decode (S=1) y prefill.

Verificado contra el log de train.py: 16 capas = 171,932,048 params.

    python flops.py            # tabla
    python flops.py 8 0.5      # una celda: 8 capas al 50%
"""

from __future__ import annotations

from ladder import Escalera

# ── Config (la de train.py; cambiala si cambias el modelo) ──
D = 512                 # d_model
L = 16                  # num_layers
H = 12                  # num_heads
G = 4                   # num_kv_groups
HEAD_DIM = D // H       # 42 (division entera)
D_C = 64                # latente KV
D_C1 = 85               # latente Q
D_ROT = 42              # parte rope
E = 4                   # n_experts
TOP_K = 2               # top_k
N_SHARED = 1            # experts compartidos (densos, siempre)
WIDTHS = (0.25, 0.50, 0.75, 1.00)
FFN_EXP = 4.0
ROUND_TO = 64
VOCAB = 32000
N_DENSE_START = 1       # capas 0..N_DENSE_START-1 densas
PISO = 2

# intermediate_dim: mismo calculo que block.compute_intermediate_dim
_raw = int(FFN_EXP * D * 2.0 / 3.0)
INTER = ((_raw + ROUND_TO - 1) // ROUND_TO) * ROUND_TO   # 1408


# ── Parametros por capa ──
def params_mla() -> int:
    w_down = D * (D_C1 + D_C + D_ROT)
    w_up_q = D_C1 * H * (HEAD_DIM + D_ROT)
    w_up_kv = D_C * 2 * G * HEAD_DIM
    o_proj = H * HEAD_DIM * D
    norms = (D_C1 + D_C) + 2 * HEAD_DIM
    return w_down + w_up_q + w_up_kv + o_proj + norms


def params_dense_ffn() -> int:
    return 3 * D * INTER          # SwiGLU: (D,2I) + (I,D)


def params_moe_ffn() -> int:
    router = D * (E * len(WIDTHS))
    experts = E * params_dense_ffn()
    shared = N_SHARED * params_dense_ffn()
    return router + experts + shared


def params_capa(idx: int) -> int:
    base = params_mla() + 4 * D          # + norms del bloque (sandwich)
    if idx < N_DENSE_START:
        return base + params_dense_ffn()
    return base + params_moe_ffn()


# ── FLOPs por token (2 por MAC) ──
def macs_mla(seq: int) -> float:
    """Proyecciones (no dependen de S) + scores y AV (dependen de S)."""
    proy = D * (D_C1 + D_C + D_ROT) \
        + D_C1 * H * (HEAD_DIM + D_ROT) \
        + D_C * 2 * G * HEAD_DIM \
        + H * HEAD_DIM * D
    # por token: QK^T y A·V sobre S posiciones, en las dos partes (estado + rope)
    scores = 2 * H * seq * (HEAD_DIM + D_ROT)
    return proy + scores


def macs_dense_ffn() -> float:
    return 3.0 * D * INTER


def macs_moe_ffn(width: float) -> float:
    """top_k expertos al ancho pedido + los compartidos a ancho completo."""
    d = max(8, int(INTER * width))
    por_experto = 2.0 * D * d          # (D,2d) + (d,D)
    router = D * (E * len(WIDTHS))
    return TOP_K * por_experto + N_SHARED * params_dense_ffn() * 1.0 + router


def macs_capa(idx: int, seq: int, width: float) -> float:
    m = macs_mla(seq)
    if idx < N_DENSE_START:
        m += macs_dense_ffn()           # la densa siempre completa
    else:
        m += macs_moe_ffn(width)
    return m


def macs_head() -> float:
    return D * VOCAB


def medir(profundidad: int, width: float = 1.0, seq: int = 1,
          escalera: Escalera | None = None) -> dict:
    esc = escalera or Escalera(n_capas=L, minima=PISO)
    capas = esc.indices(profundidad)
    p = sum(params_capa(i) for i in capas)
    macs = sum(macs_capa(i, seq, width) for i in capas) + macs_head()
    return {
        "capas": capas,
        "params": p,
        "params_pct": 100.0 * p / sum(params_capa(i) for i in range(L)),
        "macs": macs,
        "flops": 2 * macs,
    }


def _fmt(n: float) -> str:
    return f"{n/1e6:.2f}M" if n < 1e9 else f"{n/1e9:.2f}G"


if __name__ == "__main__":
    import sys

    esc = Escalera(n_capas=L, minima=PISO)
    total_params = sum(params_capa(i) for i in range(L))
    print(f"d_model={D} L={L} H={H} G={G} head_dim={HEAD_DIM} "
          f"d_c={D_C} d_c1={D_C1} d_rot={D_ROT} intermediate={INTER}")
    print(f"experts={E} top_k={TOP_K} shared={N_SHARED} widths={WIDTHS} "
          f"vocab={VOCAB} densas={N_DENSE_START}")
    print(f"params 16 capas = {total_params:,}  (train.py dice 171,932,048)")
    print(f"orden escalera  = {esc.orden}\n")

    if len(sys.argv) > 2:
        d, w = int(sys.argv[1]), float(sys.argv[2])
        r = medir(d, w, seq=1, escalera=esc)
        print(f"{d}L @ {w:.0%}: params {r['params']:,} "
              f"({r['params_pct']:.0f}%)  {r['flops']/1e6:.2f} MFLOPs/token (decode)")
    else:
        print(f"{'cfg':>12} {'params':>14} {'%':>5} "
              f"{'decode':>12} {'prefill S=800':>14}")
        print("-" * 64)
        for d in (2, 4, 6, 8, 10, 12, 16):
            for w in (0.25, 0.50, 1.00):
                r1 = medir(d, w, seq=1, escalera=esc)
                r8 = medir(d, w, seq=800, escalera=esc)
                print(f"{f'{d}L@{w:.0%}':>12} {r1['params']:>14,} "
                      f"{r1['params_pct']:>4.0f}% {_fmt(r1['flops']):>12} "
                      f"{_fmt(r8['flops']):>14}")
        print("\n(desde la primera capa densa: toda subred incluye la capa 0,")
        print(" asi que el minimo de una subred es 1 densa + (d-1) MoE)")
