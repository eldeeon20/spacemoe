"""MoSE: Mixture of Slimmable Experts con top-k sobre rutas experto x ancho.

Arquitectura por token:
  1. Router scores = x @ W_router -> [E x W] rutas (no bias)
  2. Bias trick: scores += route_bias (no aprendido, feedback de carga)
  3. softmax sobre rutas -> top-k rutas -> pesos renormalizados (ej: 30% + 1%)
  4. Cada experto es slimmable: D -> width*I -> D (prefijo de matrices,
     el resto no se calcula). La dim expandida I tambien es slimmable.
  5. Shared experts = siempre activos (conocimiento comun)
  6. Final = ruteado + compartido

Estructura:
             token
               |
             Router (E x W scores)
               |
    +----------+----------+
    |          |          |
  Expert     Expert     Expert ...
    |
  +-+---+----+----+
  |   |   |    |
 25% 50% 75% 100%
  |   |   |    |
  +-+---+----+----+
          |
         FFN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _swiglu(x, gate):
    return F.silu(gate) * x


class ExpertSwiGLU(nn.Module):
    """Single dense SwiGLU FFN, used standalone or as shared expert."""

    def __init__(self, d_model, expert_dim, bias=False):
        super().__init__()
        self.w_gate = nn.Linear(d_model, expert_dim, bias=bias)
        self.w_up = nn.Linear(d_model, expert_dim, bias=bias)
        self.w_down = nn.Linear(expert_dim, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(_swiglu(self.w_up(x), self.w_gate(x)))


class MoELayer(nn.Module):
    """MoSE: router top-k sobre rutas (experto, ancho), expertos slimmable.

    Args:
        d_model: Model dimension
        n_experts: Total routed experts
        top_k: Rutas (experto x ancho) activadas por token
        n_shared: Always-on shared experts (default 1)
        expert_dim: Intermediate dim FULL por experto (default: 2*d_model)
        widths: Anchos slimmable por experto (default 25/50/75/100%)
        capacity_factor: Max capacity multiplier por ruta (default 1.35)
        z_loss_gamma: Router z-loss weight (0 to disable, default 0.0001)
        bias: Use bias in Linear layers (default False)
    """

    def __init__(self, d_model, n_experts, top_k, n_shared=1, expert_dim=None,
                 widths=(0.25, 0.50, 0.75, 1.0),
                 capacity_factor=1.35, z_loss_gamma=0.0001, bias_decay=0.1,
                 bias=False, noise_std=0.005, load_balance_gamma=0.0001):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_shared = n_shared
        self.widths = tuple(widths)
        self.n_widths = len(self.widths)
        self.n_routes = n_experts * self.n_widths
        self.capacity_factor = capacity_factor
        self.z_loss_gamma = z_loss_gamma
        self.load_balance_gamma = load_balance_gamma
        self.bias_decay = bias_decay
        self.noise_std = noise_std

        if expert_dim is None:
            expert_dim = 2 * d_model
        self.expert_dim = expert_dim

        # Router: una score por ruta (experto, ancho)
        self.router = nn.Linear(d_model, self.n_routes, bias=False)

        # Bias por ruta para balanceo (no aprendido, feedback)
        self.register_buffer("route_bias", torch.zeros(self.n_routes))

        # Expertos slimmable — parametros agrupados para bmm
        # c_fc: (E, D, 2*I) gate+up ; c_proj: (E, I, D)
        self.c_fc = nn.Parameter(torch.randn(n_experts, d_model, 2 * expert_dim) * 0.02)
        self.c_proj = nn.Parameter(torch.randn(n_experts, expert_dim, d_model) * 0.02)
        self.b_fc = nn.Parameter(torch.zeros(n_experts, 2 * expert_dim)) if bias else None
        self.b_proj = nn.Parameter(torch.zeros(n_experts, d_model)) if bias else None

        # Shared experts (densos, siempre activos)
        self.shared = nn.ModuleList([ExpertSwiGLU(d_model, expert_dim, bias) for _ in range(n_shared)])

        # Stats de routing
        self.register_buffer("last_counts", torch.zeros(self.n_routes, dtype=torch.long))
        self.last_total = 0

    def _slimmable_expert_forward(self, x, expert_idx, width):
        """Un grupo de tokens por su experto a un ancho dado (SwiGLU slimmable).

        Solo se calcula el prefijo: up/gate [D -> d], down [d -> D].
        """
        if x.shape[0] == 0:
            return x
        d = max(8, int(self.expert_dim * width))

        w_fc = self.c_fc[expert_idx][:, :2 * d]      # (D, 2d)
        w_proj = self.c_proj[expert_idx][:d, :]      # (d, D)

        h = x @ w_fc                                 # (N, 2d)
        if self.b_fc is not None:
            b = self.b_fc[expert_idx]
            h = h + torch.cat([b[:d], b[self.expert_dim:self.expert_dim + d]], dim=-1)

        gate, up = h.chunk(2, dim=-1)
        h = _swiglu(up, gate)                        # (N, d)

        out = h @ w_proj                             # (N, D)
        if self.b_proj is not None:
            out = out + self.b_proj[expert_idx]
        return out

    def _update_route_bias(self, counts, n_tokens):
        target = n_tokens / self.n_routes
        load = counts.float()
        delta = self.bias_decay * (target - load) / max(n_tokens, 1)
        self.route_bias.add_(delta.to(self.route_bias.dtype))

    def _router_z_loss(self, logits):
        if self.z_loss_gamma <= 0:
            return torch.tensor(0.0, device=logits.device)
        logsumexp = torch.logsumexp(logits, dim=-1)
        return self.z_loss_gamma * (logsumexp ** 2).mean()

    def forward(self, x, width=None):
        """Forward MoSE.

        Args:
            x: (B, T, d_model)
            width: ancho global forzado (entrenamiento Eq.6: w_max o w~U).
                Top-k expertos dentro del bucket discreto mas cercano,
                ejecutados al ancho CONTINUO dado. None = inferencia:
                el router elige (experto, ancho) por token.
        Returns:
            output: (B, T, d_model)
            aux_loss: z-loss + load-balance (scalar)
        """
        orig_ndim = x.ndim
        if orig_ndim == 2:
            x = x.unsqueeze(1)
        B, T, C = x.shape
        N = B * T
        xf = x.reshape(N, C)

        # 1) Router scores por ruta (experto, ancho)
        logits = self.router(xf).view(N, self.n_experts, self.n_widths)  # (N, E, W)

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # 2) Bias trick por ruta
        biased = logits + self.route_bias.view(1, self.n_experts, self.n_widths)

        if width is None:
            # 3a) Inferencia: softmax + top-k rutas -> pesos por token
            probs = F.softmax(biased.flatten(-2), dim=-1)
            if self.load_balance_gamma and self.load_balance_gamma > 0.0:
                p_mean = probs.mean(dim=0)
                target = torch.full_like(p_mean, 1.0 / float(self.n_routes))
                lb_loss = self.load_balance_gamma * ((p_mean - target) ** 2).sum()
            else:
                lb_loss = torch.tensor(0.0, device=probs.device)
            topk_w, topk_i = probs.topk(self.top_k, dim=-1)  # (N, top_k)
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
            # z-loss sobre logits SIN bias trick (el bias es balanceo loss-free;
            # con bias, el forward full arrastra bias añejos y el z diverge).
            z_in = logits.flatten(-2)
        else:
            # 3b) Entrenamiento Eq.(6): ancho global; top-k expertos en el
            # bucket discreto mas cercano, ejecucion al ancho continuo.
            wi = min(range(self.n_widths), key=lambda i: abs(self.widths[i] - width))
            scores_e = biased[..., wi]  # (N, E)
            probs_e = F.softmax(scores_e, dim=-1)
            if self.load_balance_gamma and self.load_balance_gamma > 0.0:
                p_mean = probs_e.mean(dim=0)
                target = torch.full_like(p_mean, 1.0 / float(self.n_experts))
                lb_loss = self.load_balance_gamma * ((p_mean - target) ** 2).sum()
            else:
                lb_loss = torch.tensor(0.0, device=probs_e.device)
            topk_w, topk_e = probs_e.topk(self.top_k, dim=-1)
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-9)
            # z-loss sobre logits SIN bias (ver arriba).
            z_in = logits[..., wi]
            # Re-mapeo a indices de ruta para reutilizar el mismo despacho
            topk_i = topk_e * self.n_widths + wi
            probs = torch.zeros(N, self.n_routes, device=xf.device)
            probs.scatter_(-1, topk_e * self.n_widths + wi,
                            probs_e.gather(-1, topk_e))

        # 4) Capacity por ruta
        cap = max(1, int(math.ceil(self.top_k * N / self.n_routes * self.capacity_factor)))

        # 5) Ejecutar rutas (una sola pasada por ruta, sin doble computo)
        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            for wii in range(self.n_widths):
                route = e * self.n_widths + wii
                sel = (topk_i == route)  # (N, top_k) bool
                tok_mask = sel.any(dim=-1)
                tok_idx = tok_mask.nonzero(as_tuple=True)[0]
                if tok_idx.numel() == 0:
                    continue
                if tok_idx.numel() > cap:
                    order = probs[tok_idx, route].argsort(descending=True)
                    tok_idx = tok_idx[order[:cap]]
                w = (topk_w[tok_idx] * sel[tok_idx]).sum(dim=-1)  # (n_sel,)
                run_w = width if width is not None else self.widths[wii]
                expert_out = self._slimmable_expert_forward(xf[tok_idx], e, run_w)
                out[tok_idx] += w.unsqueeze(-1) * expert_out

        # 6) Shared experts
        for sh in self.shared:
            out += sh(xf)

        # 7) Bias feedback
        with torch.no_grad():
            counts = torch.bincount(topk_i.flatten(), minlength=self.n_routes)
            self.last_counts = counts.clone()
            self.last_total = N
            self._update_route_bias(counts, N)

        # 8) z-loss + load-balance
        z_loss = self._router_z_loss(z_in)
        aux_loss = z_loss + lb_loss
        try:
            self.last_z_loss = z_loss.detach()
        except Exception:
            self.last_z_loss = torch.tensor(0.0, device=out.device)
        try:
            self.last_load_balance_loss = lb_loss.detach()
        except Exception:
            self.last_load_balance_loss = torch.tensor(0.0, device=out.device)

        out = out.reshape(B, T, C)
        if orig_ndim == 2:
            out = out.squeeze(1)
        return out, aux_loss

    def balance_str(self):
        """'Expert: E0 .. | Width: 25% .. (avg ..)'. Reparto suma 100%."""
        counts = self.last_counts.reshape(self.n_experts, self.n_widths)
        total = int(counts.sum().item()) or 1
        exp = " ".join(f"E{e} {counts[e].sum().item()*100//total}%"
                       for e in range(self.n_experts))
        w = " ".join(f"{int(w*100)}% {counts[:, i].sum().item()*100//total}%"
                     for i, w in enumerate(self.widths))
        return f"Expert: {exp} | Width: {w}"


class DenseFFN(nn.Module):
    """Dense SwiGLU FFN, used as fallback when MoE is disabled."""

    def __init__(self, d_model, intermediate_dim, dropout=0.0, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        return self.down_proj(self.dropout(_swiglu(self.up_proj(x), self.gate_proj(x))))
