"""flops.py — FLOPs por token y presupuesto, por profundidad (escalera) y ancho MoSE.

Dos configs:
  CFG_SPACEMOE  — la de spacemoe/train.py: 16 capas, dim 512, 4 experts.
                  Verificada: 16 capas = 171,932,048 params (lo que imprime train.py).
  CFG_GRANDE    — 32 capas, dim 1024, 64 experts, top-4.

Cuenta FLOPs de FORWARD (2 por MAC). Las cuentas de atencion dependen de la
longitud de secuencia, asi que se reportan decode (S=1) y prefill.

    python flops.py                 # las dos configs
    python flops.py grande 2 0.25   # una celda: 2 capas al 25%
"""

from __future__ import annotations

from dataclasses import dataclass

from ladder import Escalera


@dataclass
class Config:
    nombre: str
    d_model: int
    num_layers: int
    num_heads: int
    num_kv_groups: int
    d_c: int
    d_c1: int
    d_rotate: int
    n_experts: int
    top_k: int
    n_shared: int
    widths: tuple
    ffn_expansion: float = 4.0
    round_to: int = 64
    vocab: int = 32000
    n_dense_start: int = 1
    piso: int = 2
    seq_prefill: int = 800

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def intermediate(self) -> int:
        raw = int(self.ffn_expansion * self.d_model * 2.0 / 3.0)
        return ((raw + self.round_to - 1) // self.round_to) * self.round_to

    @property
    def n_routes(self) -> int:
        return self.n_experts * len(self.widths)


CFG_SPACEMOE = Config(
    nombre="spacemoe 16L d512 E4",
    d_model=512, num_layers=16, num_heads=12, num_kv_groups=4,
    d_c=64, d_c1=85, d_rotate=42,
    n_experts=4, top_k=2, n_shared=1, widths=(0.25, 0.50, 0.75, 1.00),
)

# 32 capas, dim 1024, 64 experts, top-4. Lo que NO dijo el pedido, y asumo:
#   num_heads=16 (head_dim 64), G=4, d_c=128, d_c1=170, d_rot=64,
#   expert_dim = intermediate (como el default del codigo), 1 capa dense,
#   vocab 32000, expansion 4.0, piso de escalera 2.
CFG_GRANDE = Config(
    nombre="32L d1024 E64 top4",
    d_model=1024, num_layers=32, num_heads=16, num_kv_groups=4,
    d_c=128, d_c1=170, d_rotate=64,
    n_experts=64, top_k=4, n_shared=1, widths=(0.25, 0.50, 0.75, 1.00),
)


# ── Parametros ──
def params_mla(c: Config) -> int:
    d, h, g, hd = c.d_model, c.num_heads, c.num_kv_groups, c.head_dim
    return (d * (c.d_c1 + c.d_c + c.d_rotate)
            + c.d_c1 * h * (hd + c.d_rotate)
            + c.d_c * 2 * g * hd
            + h * hd * d
            + (c.d_c1 + c.d_c) + 2 * hd)


def params_dense_ffn(c: Config) -> int:
    return 3 * c.d_model * c.intermediate


def params_moe_ffn(c: Config) -> int:
    return (c.d_model * c.n_routes
            + c.n_experts * params_dense_ffn(c)
            + c.n_shared * params_dense_ffn(c))


def params_capa(c: Config, idx: int) -> int:
    base = params_mla(c) + 4 * c.d_model        # + norms del bloque
    if idx < c.n_dense_start:
        return base + params_dense_ffn(c)
    return base + params_moe_ffn(c)


# ── FLOPs por token (2 por MAC) ──
def macs_mla(c: Config, seq: int) -> float:
    d, h, g, hd = c.d_model, c.num_heads, c.num_kv_groups, c.head_dim
    proy = (d * (c.d_c1 + c.d_c + c.d_rotate)
            + c.d_c1 * h * (hd + c.d_rotate)
            + c.d_c * 2 * g * hd
            + h * hd * d)
    scores = 2 * h * seq * (hd + c.d_rotate)   # QK^T y A·V, dos partes
    return proy + scores


def macs_dense_ffn(c: Config) -> float:
    return 3.0 * c.d_model * c.intermediate


def macs_moe_ffn(c: Config, width: float) -> float:
    d = max(8, int(c.intermediate * width))
    por_experto = 2.0 * c.d_model * d          # (D,2d) + (d,D)
    router = c.d_model * c.n_routes
    return c.top_k * por_experto + c.n_shared * params_dense_ffn(c) + router


def macs_capa(c: Config, idx: int, seq: int, width: float) -> float:
    m = macs_mla(c, seq)
    if idx < c.n_dense_start:
        m += macs_dense_ffn(c)                 # la densa siempre completa
    else:
        m += macs_moe_ffn(c, width)
    return m


def macs_head(c: Config) -> float:
    return c.d_model * c.vocab


def medir(c: Config, profundidad: int, width: float = 1.0, seq: int = 1) -> dict:
    esc = Escalera(n_capas=c.num_layers, minima=c.piso)
    capas = esc.indices(profundidad)
    p = sum(params_capa(c, i) for i in capas)
    total = sum(params_capa(c, i) for i in range(c.num_layers))
    macs = sum(macs_capa(c, i, seq, width) for i in capas) + macs_head(c)
    return {"capas": capas, "params": p, "params_pct": 100.0 * p / total,
            "macs": macs, "flops": 2 * macs}


def _fmt(n: float) -> str:
    return f"{n/1e6:.2f}M" if n < 1e9 else f"{n/1e9:.2f}G"


# Los indices por defecto: mismas filas que usaste para la de 16 capas.
FILAS_BASE = [(2, 0.25), (2, 1.00), (4, 0.50), (6, 0.25),
              (8, 0.50), (8, 1.00), (10, 0.25), (16, 1.00)]


def _caja(cabeceras: list[str], filas: list[list[str]]) -> str:
    """Caja con bordes, como la tabla que me pasaste."""
    anchos = [max(len(cabeceras[i]),
                  max((len(f[i]) for f in filas), default=0)) for i in range(len(cabeceras))]

    def sep(izq, mid, der):
        return izq + mid.join("─" * (w + 2) for w in anchos) + der

    out = [sep("┌", "┬", "┐"),
           "│ " + " │ ".join(h.ljust(anchos[i]) for i, h in enumerate(cabeceras)) + " │",
           sep("├", "┼", "┤")]
    for f in filas:
        out.append("│ " + " │ ".join(f[i].ljust(anchos[i]) for i in range(len(f))) + " │")
    out.append(sep("└", "┴", "┘"))
    return "\n".join(out)


def tabla(c: Config, filas=None) -> None:
    esc = Escalera(n_capas=c.num_layers, minima=c.piso)
    total = sum(params_capa(c, i) for i in range(c.num_layers))
    filas = filas or FILAS_BASE
    filas = [(d, w) for d, w in filas if d <= c.num_layers]
    if filas and filas[-1][0] != c.num_layers:
        filas.append((c.num_layers, 1.00))

    print(f"\n=== {c.nombre} ===")
    print(f"dim={c.d_model} L={c.num_layers} H={c.num_heads} G={c.num_kv_groups} "
          f"head_dim={c.head_dim}")
    print(f"d_c={c.d_c} d_c1={c.d_c1} d_rot={c.d_rotate} "
          f"intermediate={c.intermediate} expert_dim={c.intermediate}")
    print(f"experts={c.n_experts} top_k={c.top_k} shared={c.n_shared} "
          f"widths={c.widths} rutas={c.n_routes}")
    print(f"vocab={c.vocab} densas={c.n_dense_start} piso_escalera={c.piso}")
    print(f"params total = {total:,}")
    print(f"cabeza (lm_head) = {2*macs_head(c)/1e6:.1f} MFLOPs/token (fijo)")

    filas_txt = []
    for d, w in filas:
        r1 = medir(c, d, w, seq=1)
        r8 = medir(c, d, w, seq=c.seq_prefill)
        filas_txt.append([
            f"{d}L @{w:.0%}",
            _params(c, r1["params"]),
            f"{r1['params_pct']:.0f}%",
            f"{r1['flops']/1e6:.1f} MFLOPs",
            f"{_corto(r8['flops'])}",
        ])
    print()
    print(_caja(["cfg", "params", "%", f"decode (S=1)",
                 f"prefill S={c.seq_prefill}"], filas_txt))


def _params(c: Config, n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return str(n)


def _corto(n: float) -> str:
    return f"{n/1e6:.1f}M" if n < 1e9 else f"{n/1e9:.2f}G"


def presupuesto(c: Config) -> None:
    """El presupuesto: que cuesta el piso de la escalera y cuanto se paga."""
    esc = Escalera(n_capas=c.num_layers, minima=c.piso)
    completo = medir(c, c.num_layers, 1.0, seq=1)
    print(f"\n--- presupuesto ({c.nombre}) ---")
    for d in (c.piso, 4, 8, 16, c.num_layers):
        if d > c.num_layers:
            continue
        peor = medir(c, d, 1.0, seq=1)
        mejor = medir(c, d, 0.25, seq=1)
        print(f"  {d:2d}L: {mejor['flops']/1e6:8.2f} MFLOPs (25%) .. "
              f"{peor['flops']/1e6:8.2f} MFLOPs (100%)   "
              f"params {peor['params']:>15,} ({peor['params_pct']:3.0f}%)")
    piso = medir(c, c.piso, 1.0, seq=1)
    print(f"  piso {c.piso}L@100% = {piso['flops']/1e6:.2f} MFLOPs/token = "
          f"{100*piso['flops']/completo['flops']:.1f}% del modelo completo")
    print(f"  16L@25%            = {medir(c,16,0.25)['flops']/1e6:.2f} MFLOPs/token = "
          f"{100*medir(c,16,0.25)['flops']/completo['flops']:.1f}% del modelo completo")
    print(f"  orden de la escalera = {esc.orden}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "grande":
        c = CFG_GRANDE
        if len(sys.argv) > 3:
            d, w = int(sys.argv[2]), float(sys.argv[3])
            r = medir(c, d, w, seq=1)
            print(f"{c.nombre}: {d}L @ {w:.0%} -> {r['params']:,} params "
                  f"({r['params_pct']:.0f}%), {r['flops']/1e6:.2f} MFLOPs/token")
        else:
            tabla(c)
            presupuesto(c)
    else:
        tabla(CFG_SPACEMOE)
        print("\nparams de 16 capas deben dar 171,932,048")
        if "--grande" in sys.argv:
            tabla(CFG_GRANDE)
