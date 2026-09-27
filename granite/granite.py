"""granite.py — GraniteFF: FFN de 5 posiciones fijas.

Diseño
------
La FF base es 100% = 5 bloques de 20%. Granite la reorganiza en 5
POSICIONES virtuales::

    C1  = 20%  fijo (siempre activo)
    C2  = 20%  → M1 + M2
    C3  = 20%  → M3 + M4
    C4  = 20%  → M5 + M6
    C5  = 20%  → M7 + M8

El router hace TOP-2 SOBRE LOS 8 MÓDULOS M1–M8 (no elige una celda ni
activa su par automáticamente). El fijo (C1) va siempre, sin router.
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

Ancho
-----
Físico (lo que vive):      C1 20% + C2 20% = 40%.
Virtual (lo que se calcula): C1 20% + M1 20% + M2 20% = 60%.

Es decir: la posición C2 es 20%; el cálculo dentro de C2 es 40%
(M1 + M2). M1/M2 no ocupan dos posiciones físicas: ambos están
asociados a la misma posición C2.

Implementación: UNA sola FF por celda (up dim->h20, silu, down),
h20 = base // 5 (20% de la base). Sin bias (como el sketch). El router
es un Linear dim->8 con softmax + top-2 renormalizado; entrena con
ruido + bias-feedback + z-loss + balance (ver abajo) además del
gradiente de la CE vía w1/w2.

Interfaz drop-in para capas densas: ``forward(x) -> Tensor`` (la capa
densa hace ``h = self.ffn(h)`` y espera tensor).

Posicional: la parte de la posición 2 es siempre esa posición; cada
módulo cae siempre en su celda, nada se reordena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraniteFF(nn.Module):
    def __init__(self, dim=512, base=2048, noise_std=0.005,
                 z_loss_gamma=0.001, load_balance_gamma=0.0001,
                 bias_decay=0.1):
        super().__init__()

        # 20% del FF
        h20 = base // 5  # 409
        self.h20 = h20
        # Entrenamiento del router (no cambian la arquitectura).
        self.noise_std = noise_std
        self.z_loss_gamma = z_loss_gamma
        self.load_balance_gamma = load_balance_gamma
        self.bias_decay = bias_decay
        # Bias por módulo (no aprendido, feedback) + stats de observ.
        self.register_buffer("route_bias", torch.zeros(8))
        self.register_buffer("last_counts", torch.zeros(8, dtype=torch.long))
        self.last_total = 0
        self.last_aux = torch.tensor(0.0)
        self.last_aux_live = torch.tensor(0.0)
        self.last_z_loss = torch.tensor(0.0)
        self.last_load_balance_loss = torch.tensor(0.0)

        # C1: fijo, siempre activo (sin router).
        self.up_fixed = nn.Linear(dim, h20, bias=False)

        # 8 módulos MoE, cada uno = 20%. Candidatos del router
        # (todos menos el fijo).
        self.up_moe = nn.ModuleList([
            nn.Linear(dim, h20, bias=False)
            for _ in range(8)
        ])

        # Router: selecciona 2 de los 8.
        self.router = nn.Linear(dim, 8, bias=False)

        # Un único down para el ancho físico C1 + C2 = 40%.
        self.down = nn.Linear(h20 * 2, dim, bias=False)

    def forward(self, x):
        # --------------------------------
        # C1: 20%, siempre activo
        # --------------------------------
        c1 = F.silu(self.up_fixed(x))

        # --------------------------------
        # Router top-2 entre M1...M8
        # --------------------------------
        logits = self.router(x.float())
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        biased = logits + self.route_bias.to(logits.dtype)
        probs = F.softmax(biased, dim=-1)

        # Balanceo hacia uniforme (aux; no cambia el forward).
        if self.load_balance_gamma > 0:
            p_mean = probs.mean(dim=0)
            tgt = torch.full_like(p_mean, 1.0 / 8.0)
            lb_loss = self.load_balance_gamma * ((p_mean - tgt) ** 2).sum()
        else:
            lb_loss = torch.tensor(0.0, device=probs.device)

        weights, indices = torch.topk(
            probs,
            k=2,
            dim=-1
        )

        # Renormalizar los dos pesos (w1 + w2 = 1).
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # --------------------------------
        # Dos módulos MoE, 20% cada uno.
        # NOTA: ModuleList no acepta un tensor como índice, así que se
        # despacha por módulo (los tokens que lo eligieron). La
        # matemática es la misma: c2 = w1*silu(m1) + w2*silu(m2).
        # --------------------------------
        xf = x.reshape(-1, x.shape[-1])
        idx = indices.reshape(-1, 2)
        w = weights.reshape(-1, 2)
        c2 = torch.zeros(xf.shape[0], self.h20, device=xf.device, dtype=xf.dtype)
        for m, up in enumerate(self.up_moe):
            for j in (0, 1):
                sel = (idx[:, j] == m)
                if sel.any():
                    got = sel.nonzero(as_tuple=True)[0]
                    c2[got] = c2[got] + w[got, j:j + 1].to(c2.dtype) * F.silu(up(xf[got]))
        c2 = c2.reshape(x.shape[:-1] + (self.h20,))

        # --------------------------------
        # Combinación dentro del espacio C2.
        # c2 = w1 * m1 + w2 * m2  →  sigue siendo 20%
        #
        # Físicamente:
        # C1 = 20%
        # C2 = 20%
        # total = 40%
        #
        # Virtualmente/calculado:
        # C1 20% + M1 20% + M2 20% = 60%
        # --------------------------------
        h = torch.cat([c1, c2], dim=-1)       # 40% físico

        out = self.down(h)

        # Aux del router (z-loss sobre logits SIN bias + balanceo).
        # Va a `last_aux_live` (con grafo: el train lo suma a la CE).
        logsumexp = torch.logsumexp(logits, dim=-1)
        z_loss = self.z_loss_gamma * (logsumexp ** 2).mean() if self.z_loss_gamma > 0 else torch.tensor(0.0, device=logits.device)
        aux_loss = z_loss + lb_loss.to(z_loss.dtype)
        self.last_aux_live = aux_loss
        with torch.no_grad():
            counts = torch.bincount(indices.flatten(), minlength=8)
            self.last_counts = counts.clone()
            self.last_total = xf.shape[0]
            tgt = (self.last_total * 2) / 8
            delta = self.bias_decay * (tgt - counts.float()) / max(self.last_total, 1)
            self.route_bias.add_(delta.to(self.route_bias.dtype))
            self.last_aux = aux_loss.detach()
            self.last_z_loss = z_loss.detach()
            self.last_load_balance_loss = lb_loss.detach()

        return out

    def balance_str(self):
        """'Fija 100% | Mod: M1 ..'. Votos por módulo (top-2 por token)."""
        total = int(self.last_counts.sum().item()) or 1
        return "Fija 100% | Mod: " + " ".join(
            f"M{i + 1} {self.last_counts[i].item() * 100 // total}%" for i in range(8))
