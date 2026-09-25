"""Chat con atención dispersa sparse MIO v3 desde model_test.safetensors."""
import os, sys, math, time, torch, re
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TransformerLM
from sparse import SparseAttentionMio3
from safetensors.torch import load_file
from tokenizers import Tokenizer
from cache_kv import KVCache
from attention import repeat_kv
from rope import apply_rope_partial

device = torch.device("cpu")
_DIR = os.path.dirname(os.path.abspath(__file__))
_RUST = os.path.dirname(_DIR)
tok = Tokenizer.from_file(os.path.join(_RUST, "tokenizer.json"))
vocab = tok.get_vocab_size()

MAX_BLOCKS = 320  # 320 * 128 = 40960 tokens máximo
model = TransformerLM(
    vocab_size=vocab, d_model=768, num_layers=24, num_heads=12, num_kv_groups=4,
    use_swiglu=True, use_x0=True, max_seq_len=MAX_BLOCKS * 128,
    use_sparse_attn=True, num_selected_blocks=4,
).to(device).eval()
state = load_file(os.path.join(_RUST, "model_test.safetensors"))
model.load_state_dict(state, strict=False)
print(f"Cargado sparse model_test.safetensors ({len(state)} tensors)")

# Patch SparseAttentionMio3._sparse_forward - q_len != kv_len, gather único sin loops
_sparse_global_nb = [0]
_sparse_stats = {}  # bloque -> contador de selecciones
_orig_sparse_fn = SparseAttentionMio3._sparse_forward
def _cached_sparse_forward(self, q, k, v):
    B, NH, S_q, HD = q.shape
    NK = self.num_kv_groups; HPG = NH // NK
    BSZ = self.block_size
    _, _, S_kv, _ = k.shape
    NB = max(1, (S_kv + BSZ - 1) // BSZ)
    K = min(self.num_selected_blocks, NB)
    if NB <= K:
        if NB > _sparse_global_nb[0]:
            _sparse_global_nb[0] = NB
        for b in range(NB):
            _sparse_stats[b] = _sparse_stats.get(b, 0) + 1
        s = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HD)
        if self.causal and S_q > 1:
            s = s + torch.triu(torch.full((S_q, S_kv), float("-inf"), device=q.device), diagonal=S_kv - S_q + 1).unsqueeze(0).unsqueeze(0)
        return torch.matmul(F.softmax(s, dim=-1), v)
    if NB > _sparse_global_nb[0]:
        _sparse_global_nb[0] = NB
        print(f"bloque {NB-1} despachado, cache {NB} bloques ({S_kv} tokens)")
    pad = NB * BSZ - S_kv
    kp = F.pad(k, (0, 0, 0, pad)); vp = F.pad(v, (0, 0, 0, pad))
    last_real = S_kv - (NB - 1) * BSZ
    knk = kp[:, ::HPG]; vnk = vp[:, ::HPG]
    k_b = knk.reshape(B, NK, NB, BSZ, HD); v_b = vnk.reshape(B, NK, NB, BSZ, HD)
    k_comp = k_b.mean(dim=3)
    if last_real < BSZ:
        k_comp[:, :, -1, :] = k_b[:, :, -1, :last_real, :].sum(dim=2) / last_real
    q_g = q.reshape(B, NK, HPG, S_q, HD).mean(dim=2)
    scores = (q_g @ k_comp.transpose(-2, -1)) * (HD ** -0.5)
    if self.causal and S_q > 1:
        scores.masked_fill_((torch.arange(S_q, device=q.device)[:, None] < (torch.arange(NB, device=q.device) * BSZ)[None, :]).unsqueeze(0).unsqueeze(0), float('-inf'))
    always = [i for i in (0, NB-1) if i < NB]
    n_a = len(always)
    s2 = scores.clone(); s2[..., always] = float('-inf')
    _, topk_rest = s2.topk(K - n_a, dim=-1)
    a_t = torch.tensor(always, device=q.device).view(1,1,1,-1).expand(B, NK, S_q, -1)
    topk = torch.cat([a_t, topk_rest], dim=-1)
    for b in topk[0, :, 0].flatten().tolist():
        _sparse_stats[b] = _sparse_stats.get(b, 0) + 1
    topk_nh = topk.repeat_interleave(HPG, dim=1)
    k_b_nh = k_b.repeat_interleave(HPG, dim=1); v_b_nh = v_b.repeat_interleave(HPG, dim=1)
    idx_b = torch.arange(B, device=q.device)[:, None, None, None]
    idx_h = torch.arange(NH, device=q.device)[None, :, None, None]
    k_sel = k_b_nh[idx_b, idx_h, topk_nh]
    v_sel = v_b_nh[idx_b, idx_h, topk_nh]
    # atender_mask: posiciones padding del bloque NB-1 -> -inf
    pos = torch.arange(BSZ, device=q.device).view(1, 1, 1, 1, BSZ)
    amask = ((topk_nh.unsqueeze(-1) == NB - 1) & (pos >= last_real)).float() * float('-inf')
    kf = k_sel.reshape(B * NH * S_q, K * BSZ, HD)
    vf = v_sel.reshape(B * NH * S_q, K * BSZ, HD)
    qf = q.reshape(B * NH * S_q, 1, HD)
    return F.scaled_dot_product_attention(qf, kf, vf, attn_mask=amask.reshape(B * NH * S_q, 1, K * BSZ)).reshape(B, NH, S_q, HD)
SparseAttentionMio3._sparse_forward = _cached_sparse_forward

# Patch SparseAttentionMio3.forward para que SIEMPRE use sparse (incluso seq_len < 2048)
def _always_sparse_forward(self, x, offset=0):
    q, k, v = self.qkv(x)
    q, k = self.rope(q, k, offset)
    k = repeat_kv(k, self.num_heads, self.num_kv_groups)
    v = repeat_kv(v, self.num_heads, self.num_kv_groups)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_output = self._sparse_forward(q, k, v)
    attn_output = attn_output.transpose(1, 2)
    return self.o_proj(attn_output)
SparseAttentionMio3.forward = _always_sparse_forward

# Patch forward_with_cache_partial para usar sparse + x0
def _patched_fwcp(input_ids, offset, caches, rotary_pct=0.5):
    x = model.embedding(input_ids)
    h = x
    x0 = x.clone()
    new_caches = []
    for i, (layer, cache) in enumerate(zip(model.transformer.layers, caches)):
        residual = h
        h_norm = layer.attn_norm(h)

        if layer.use_sparse_attn:
            q, k_new, v_new = layer.attention.qkv(h_norm)
            q, k_new = apply_rope_partial(
                q, k_new, offset, rotary_pct,
                layer.attention.rope.inv_freq, layer.attention.rope.cos_cache,
                layer.attention.rope.sin_cache, layer.attention.head_dim,
                layer.attention.rope.max_seq_len,
            )
            if cache is not None:
                k_full = torch.cat([cache.cached_k, k_new], dim=1)
                v_full = torch.cat([cache.cached_v, v_new], dim=1)
            else:
                k_full = k_new
                v_full = v_new
            new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())
            # repeat KV + transpose → (B, NH, S, HD)
            k_exp = repeat_kv(k_full, layer.num_heads, layer.num_kv_groups)
            v_exp = repeat_kv(v_full, layer.num_heads, layer.num_kv_groups)
            q_t = q.transpose(1, 2)
            k_t = k_exp.transpose(1, 2)
            v_t = v_exp.transpose(1, 2)
            attn_out = layer.attention._sparse_forward(q_t, k_t, v_t)
            attn_out = attn_out.transpose(1, 2)
            h_attn = layer.attention.o_proj(attn_out)
        else:
            q, k_new, v_new = layer.attention.qkv(h_norm)
            q, k_new = apply_rope_partial(
                q, k_new, offset, rotary_pct,
                layer.attention.rope.inv_freq, layer.attention.rope.cos_cache,
                layer.attention.rope.sin_cache, layer.attention.head_dim,
                layer.attention.rope.max_seq_len,
            )
            if cache is not None:
                k_full = torch.cat([cache.cached_k, k_new], dim=1)
                v_full = torch.cat([cache.cached_v, v_new], dim=1)
            else:
                k_full = k_new
                v_full = v_new
            new_cache = KVCache(cached_k=k_full.clone(), cached_v=v_full.clone())
            k_exp = repeat_kv(k_full, layer.num_heads, layer.num_kv_groups)
            v_exp = repeat_kv(v_full, layer.num_heads, layer.num_kv_groups)
            q_t = q.transpose(1, 2)
            k_t = k_exp.transpose(1, 2)
            v_t = v_exp.transpose(1, 2)
            scale = math.sqrt(layer.head_dim)
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) / scale
            if layer.attn_logit_cap is not None:
                scores = torch.tanh(scores / layer.attn_logit_cap) * layer.attn_logit_cap
            if layer.causal and q_t.shape[2] > 1:
                mask = torch.triu(torch.full((q_t.shape[2], k_t.shape[2]), float("-inf"), device=h.device), diagonal=k_t.shape[2] - q_t.shape[2] + 1)
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
            attn_w = F.softmax(scores, dim=-1)
            attn_w = layer.attention.attn_dropout(attn_w)
            attn_out = torch.matmul(attn_w, v_t)
            attn_out = attn_out.transpose(1, 2)
            h_attn = layer.attention.o_proj(attn_out)

        h = residual + h_attn
        residual = h
        h_norm = layer.ffn_norm(h)
        h_ffn = layer.ffn(h_norm)
        h = residual + h_ffn
        if model.x0_lambdas is not None:
            h = h + model.x0_lambdas[0, i] * x0
        new_caches.append(new_cache)
    h = model.transformer.final_norm(h)
    return model.head(h), new_caches
model.forward_with_cache_partial = _patched_fwcp

# Patch generate para hacer prefill + sparse loop
_caches = [None] * model.num_layers
_offset = 0
@torch.no_grad()
def gen(prompt, max_new, temp, top_k, top_p, rep_penalty=1.1):
    global _sparse_stats, _caches, _sparse_global_nb, _offset
    _sparse_stats = {}
    _sparse_global_nb = [0]
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    generated = x.clone()
    # Prefill con cache persistente y offset global
    logits, _caches = _patched_fwcp(generated, _offset, _caches, 0.25)
    _offset = _offset + generated.shape[1]
    next_logits = logits[:, -1, :]
    for _ in range(max_new):
        l2 = torch.nan_to_num(next_logits.clone(), nan=0.0, posinf=1e9, neginf=-1e9) / max(temp, 1e-6)
        if rep_penalty != 1.0:
            for t in set(generated[0].tolist()):
                l2[0, t] /= rep_penalty
        if top_k > 0:
            l2[torch.topk(l2, top_k)[0][:, -1:] > l2] = float("-inf")
        if top_p < 1.0:
            s, si = torch.sort(l2, descending=True)
            cp = torch.cumsum(F.softmax(s, dim=-1), dim=-1)
            sr = cp > top_p
            sr[:, 1:] = sr[:, :-1].clone()
            sr[:, 0] = False
            l2[sr.scatter(1, si, sr)] = float("-inf")
        if l2[0].isfinite().sum() == 0:
            l2 = torch.nan_to_num(next_logits.clone(), nan=0.0) / max(temp, 1e-6)
        p = F.softmax(l2, dim=-1)
        nt = torch.multinomial(p, 1)
        generated = torch.cat([generated, nt], dim=1)
        logits, _caches = _patched_fwcp(nt, _offset, _caches, 0.25)
        _offset += 1
        next_logits = logits[:, -1, :]
    if _sparse_stats:
        top3 = sorted(_sparse_stats.items(), key=lambda x: -x[1])[:3]
        nb_total = max(_sparse_stats.keys()) + 1
        total_tok = generated.shape[1]
        print(f"  {nb_total} bloques ({total_tok} tok) | top: " + ", ".join(f"blk{b}({c})" for b,c in top3))
    return generated

settings = {"max_new": 50, "temp": 1.0, "top_k": 50, "top_p": 0.95}
print("Sparse Chat MIO v3: block_size=128, num_selected_blocks=16, cache 4096")
print("  /help  /len N  /temp N  /top N  /top_p N")

while True:
    prompt = input("\n> ").strip()
    if not prompt:
        continue
    parts = prompt.split()
    cmd = parts[0].lower()
    if cmd in ("help", "ayuda"):
        print(f"max={settings['max_new']} temp={settings['temp']} top_k={settings['top_k']} top_p={settings['top_p']}")
        continue
    if cmd in ("max", "len") and len(parts) > 1 and parts[1].isdigit():
        settings["max_new"] = max(1, int(parts[1]))
        continue
    if cmd == "temp" and len(parts) > 1:
        settings["temp"] = max(0.01, float(parts[1]))
        continue
    if cmd in ("top_k", "top") and len(parts) > 1 and parts[1].isdigit():
        settings["top_k"] = max(1, int(parts[1]))
        continue
    if cmd == "top_p" and len(parts) > 1:
        settings["top_p"] = max(0.01, min(1.0, float(parts[1])))
        continue
    t0 = time.time()
    out = gen(prompt, settings["max_new"], settings["temp"], settings["top_k"], settings["top_p"])
    dt = time.time() - t0
    n = out.shape[1] - len(tok.encode(prompt).ids)
    print(tok.decode(out[0].tolist()))
    if n > 0:
        print(f"[{n} tokens en {dt:.2f}s = {n/dt:.1f} tok/s]")
