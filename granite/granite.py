"""granite.py — GraniteFF: FFN de 5 posiciones fijas (SwiGLU).

Denso de referencia (dim=512)::

         x: 512
         │
         ├── up: 512 → 2048
         └── gate: 512 → 2048
                     ↓
               SiLU(gate) × up
                     ↓
                    2048
                     ↓
               down: 2048 → 512

Granite: cada posición es ~20% de esa expansión total::

    C1  = 20%  fijo (siempre activo)
    C2  = 20%  → M1 + M2
    C3  = 20%  → M3 + M4
    C4  = 20%  → M5 + M6
    C5  = 20%  → M7 + M8

Cada parte (fija o módulo) es un SwiGLU completo a su ancho: up + gate
dim->w, SiLU(gate)×up. El router hace TOP-2 SOBRE LOS 8 MÓDULOS M1–M8
(todos menos el fijo). Por token::

    C1 20% + M_a 20% + M_b 20% = 60% activo

Antes del down, la expansión física completa::

    C1 + C2 + C3 + C4 + C5 = 409+409+410+410+410 = 2048

Las dos ramas seleccionadas comparten su posición física mediante suma
ponderada (M1/M2 no ocupan dos posiciones: ambas van a C2). Down
IDÉNTICO al denso (2048→512).

Ancho
-----
Físico por token: C1 (20%) + posiciones tocadas (20-40%) = 40-60%
  (mismo slot → 40%; slots distintos → 60%).
Virtual (calculado): C1 + 2 módulos = 60% del oculto.
El down es completo (2048) como en el denso: el cómputo FF total
activo/token ≈ 73% del denso (el down pesa 1/3).

Router: Linear(dim→8) + bias APRENDIDO (`route_bias` Parameter) +
softmax + top-2 renormalizado (w1+w2=1). Sin ruido ni auxiliares: el
gradiente le llega por la CE a través de los pesos.

Interfaz: ``forward(x) -> Tensor`` (drop-in de capa densa: la capa hace
``h = self.ffn(h)``). Para observar el router sin romper la interfaz se
guarda ``last_info`` (detach: probs, índices, pesos) más ``last_counts``
y ``balance_str()``.

Posicional: cada módulo cae siempre en su celda (M1/M2→C2…),
nada se reordena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraniteFF(nn.Module):
    """
    Granite FF

    Denso de referencia:
        512 -> 2048 (up + gate)
        2048 -> 512 (down)

    Granite:
        C1 = posición fija
        C2 = M1/M2
        C3 = M3/M4
        C4 = M5/M6
        C5 = M7/M8

    Cada posición representa ~20% de la expansión total.
    El router selecciona top-2 entre M1..M8.

    Por token:
        C1 20% + M_a 20% + M_b 20% = 60% activo

    Antes del down:
        5 posiciones = 2048 de ancho total.

    Las dos ramas seleccionadas comparten su posición física
    mediante suma ponderada.
    """

    def __init__(self, dim=512, base=2048):
        super().__init__()

        # 5 posiciones que suman exactamente 2048.
        widths = [409, 409, 410, 410, 410]
        assert sum(widths) == base
        self.widths = widths

        # C1: fija, siempre activa (SwiGLU a su ancho).
        self.up_fixed = nn.Linear(dim, widths[0], bias=False)
        self.gate_fixed = nn.Linear(dim, widths[0], bias=False)

        # M1..M8 (SwiGLU a su ancho):
        # M1/M2 -> C2, M3/M4 -> C3, M5/M6 -> C4, M7/M8 -> C5.
        self.up_moe = nn.ModuleList()
        self.gate_moe = nn.ModuleList()
        for i in range(8):
            slot = 1 + i // 2
            h = widths[slot]
            self.up_moe.append(nn.Linear(dim, h, bias=False))
            self.gate_moe.append(nn.Linear(dim, h, bias=False))

        # Router top-2 sobre los 8 (todos menos el fijo).
        # Bias APRENDIDO (va al grupo nodecay por ser dim<2).
        self.router = nn.Linear(dim, 8, bias=False)
        self.route_bias = nn.Parameter(torch.zeros(8))

        # MISMO down que el denso: 2048 -> dim.
        self.down = nn.Linear(base, dim, bias=False)

        # Observación del router (detach; no afecta el forward).
        self.register_buffer("last_counts", torch.zeros(8, dtype=torch.long))
        self.last_total = 0
        self.last_info = {}

    @staticmethod
    def swiglu(up, gate):
        return F.silu(gate) * up

    def forward(self, x):
        # x: [B, T, dim]
        B, T, D = x.shape

        # C1 FIJA (SwiGLU a 409).
        c1 = self.swiglu(self.up_fixed(x), self.gate_fixed(x))

        # ROUTER top-2 por TOKEN (todos menos el fijo).
        logits = self.router(x)
        logits = logits + self.route_bias.to(logits.dtype)
        probs = F.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=2, dim=-1)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

        # C2-C5: solo se calculan las posiciones cuyos módulos salieron.
        # Cada posición tiene su ancho exacto (409/410/410/410).
        cells = [c1] + [
            torch.zeros(B, T, w, device=x.device, dtype=x.dtype)
            for w in self.widths[1:]
        ]

        # TOP-2: cada módulo se procesa solo para sus tokens.
        for k in range(2):
            selected = indices[..., k]
            weight = weights[..., k]
            for expert_id in range(8):
                mask = selected == expert_id
                if not mask.any():
                    continue
                x_e = x[mask]
                h_e = self.swiglu(
                    self.up_moe[expert_id](x_e),
                    self.gate_moe[expert_id](x_e),
                )
                # Peso del router.
                h_e = weight[mask].unsqueeze(-1).to(h_e.dtype) * h_e
                # Posición física del módulo:
                # 0 -> M1/M2 -> C2, 1 -> M3/M4 -> C3,
                # 2 -> M5/M6 -> C4, 3 -> M7/M8 -> C5.
                slot = 1 + expert_id // 2
                cells[slot][mask] += h_e.to(cells[slot].dtype)

        # EXPANSIÓN COMPLETA: 409+409+410+410+410 = 2048.
        h = torch.cat(cells, dim=-1)
        assert h.shape[-1] == sum(self.widths)

        # DOWN idéntico al denso.
        y = self.down(h)

        # Observación (detach) para el reporte del train.
        with torch.no_grad():
            self.last_counts = torch.bincount(
                indices.flatten(), minlength=8).clone()
            self.last_total = B * T
            self.last_info = {
                "router_probs": probs.detach(),
                "top2_indices": indices.detach(),
                "top2_weights": weights.detach(),
            }

        return y

    def balance_str(self):
        """'Fija 100% | Mod: M1 ..'. Votos por módulo (top-2 por token)."""
        total = int(self.last_counts.sum().item()) or 1
        return "Fija 100% | Mod: " + " ".join(
            f"M{i + 1} {self.last_counts[i].item() * 100 // total}%" for i in range(8))
