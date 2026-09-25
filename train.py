import sys, os, time, math, random, inspect, torch
import torch.nn.functional as F
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.join(_DIR, ".."))
from model import TransformerLM
import importlib
train_data = importlib.import_module("train-data")
from wikipedia import download_wikipedia_50mb
from huggingface import HFManager, PeriodicPusher
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from plot import PlotManager


class BPEWrapper:
    def __init__(self, tok):
        self.tokenizer = tok
        self.vocab_size = tok.get_vocab_size()
    def encode(self, text):
        return self.tokenizer.encode(text).ids
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="hola", max_new=30, width=None):
    model.eval()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new, temperature=0.7, top_k=40, top_p=0.9,
                         repetition_penalty=1.2, use_partial_rope=use_partial_rope, rotary_pct=rotary_pct,
                         width=width)
    model.train()
    return tokenizer.decode(out[0].tolist())


def train_tokenizer_from_wiki(vocab_size, output_path):
    wiki = download_wikipedia_50mb()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["eos_token"])
    with open(wiki, "r", encoding="utf-8") as f:
        tok.train_from_iterator([f.read()], trainer=trainer)
    tok.save(output_path)
    return output_path


def get_lr(step, total, warmup, lr):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (0.2 + 0.8 * (1.0 + math.cos(math.pi * t)) / 2.0)


# ─── Config ──────────────────────────────────────────────────────────────────
d_model = 512
num_layers = 16
num_heads = 12
num_kv_groups = 4
head_dim = d_model // num_heads
seq_len = 800
batch_size = 4
grad_accum = 16
lr = 3e-4
num_epochs = 200000
warmup_steps = 50
bpe_vocab = 32000
rotary_pct = 0.25
use_mla = True
use_xsa = False
qk_norm = True
use_partial_rope = False
use_sandwich_norm = True
noise_std = 0.01
mla_d_c = 64
mla_d_c1 = None
mla_d_rotate = None
tok_path = os.path.join(_DIR, "tokenizer.json")

# ─── FFN dimensions ──────────────────────────────────────────────────────────
# Dense FFN intermediate dim. None = computed from d_model*ffn_expansion.
dense_dim = None
# MoE expert intermediate dim. None = same as dense_dim.
moe_dim = None

# ─── MoE Config ──────────────────────────────────────────────────────────────
use_moe = True
n_dense_start = 1
n_dense_end = 0
n_experts = 4
top_k = 2
n_shared = 1
capacity_factor = 1.25
z_loss_gamma = 0.001
bias_decay = 0.1
# Per-layer expert counts: list or int (same for all MoE layers)
# n_experts = [4, 4, 4, 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 4, 4, 4, 4]

# ─── MoSE Eq.(6): L = 1/2 [L(w_max) + L(w)], w ~ U(w_min, w_max) ───
mose_w_min = 0.25
mose_w_max = 1.0

plot_interval = 256


def main():
    test_mode = len(sys.argv) > 1 and sys.argv[1].endswith(".txt")
    txt_path = sys.argv[1] if test_mode else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── HF ──────────────────────────────────────────────────────────────────
    repo_id = "ScortexIA/laurelia"
    revision = "moe-plus"
    hf = pusher = None
    if not test_mode:
        hf = HFManager(repo_id=repo_id, revision=revision)
        hf._get_token()
        pusher = PeriodicPusher(hf, interval_minutes=20)
    pm = PlotManager(hf if not test_mode else None, save_dir=_DIR, plot_interval=plot_interval)

    # ── Precision ──────────────────────────────────────────────────────────
    amp = False
    if test_mode:
        dtype = torch.float32
    else:
        prec = input("Precision (n=f32, f=f16, b=bf16, a=amp(f16+master-f32)): ").strip().lower()
        if prec == "b":
            dtype = torch.bfloat16
        elif prec == "f":
            dtype = torch.float16
        elif prec == "a":
            dtype = torch.float16
            amp = True
        else:
            dtype = torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=amp)
    master = "f32 (master)" if amp else str(dtype)
    print(f"  Compute: {dtype}  |  Weights: {master}  |  AMP: {amp}  |  Scaler: {scaler.get_scale() if amp else 'off'}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = None
    if os.path.exists(tok_path):
        tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
    elif hf and hf.tokenizer_exists():
        try:
            local_tok = hf.download_tokenizer(tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(local_tok))
        except:
            pass
    if tokenizer is None:
        if hf:
            train_tokenizer_from_wiki(bpe_vocab, tok_path)
            tokenizer = BPEWrapper(Tokenizer.from_file(tok_path))
            hf.upload_tokenizer(tok_path, os.path.join(_DIR, "tokenizer_config.json"))
        else:
            sys.exit("No tokenizer found")
    print(f"Vocab: {tokenizer.vocab_size}")

    # ── Model ───────────────────────────────────────────────────────────────
    if dense_dim is not None:
        ffn_expansion = dense_dim * 3.0 / 2.0 / d_model
    else:
        ffn_expansion = 4.0
    exp_dim = moe_dim or dense_dim
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size, d_model=d_model, num_layers=num_layers,
        num_heads=num_heads, num_kv_groups=num_kv_groups, head_dim=head_dim,
        use_swiglu=True, use_x0=False, max_seq_len=seq_len,
        ffn_expansion=ffn_expansion,
        residual_dropout=0.0, attn_dropout=0.0, ffn_dropout=0.0,
        use_mla=True, use_xsa=use_xsa, qk_norm=qk_norm,
        attn_logit_cap=30, use_sandwich_norm=use_sandwich_norm, noise_std=noise_std,
        mla_block_size=128,
        mla_d_c=mla_d_c, mla_d_c1=mla_d_c1, mla_d_rotate=mla_d_rotate,
        use_moe=use_moe, n_experts=n_experts, top_k=top_k, n_shared=n_shared,
        expert_dim=exp_dim,
        capacity_factor=capacity_factor, z_loss_gamma=z_loss_gamma,
        bias_decay=bias_decay,
        n_dense_start=n_dense_start, n_dense_end=n_dense_end,
    ).to(device).to(dtype=dtype)

    # Weight decay groups: biases + norms no decay, weights sí
    # Embedding gets 1/4 LR (doubles gradient from tied head + lookup)
    emb_params = [model.embedding.weight]
    other_decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad and 'embedding' not in n]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    optim_groups = [
        {"params": emb_params, "weight_decay": 0.01, "lr_scale": 0.25},
        {"params": other_decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device.type == "cuda"
    opt = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)
    print(f"AdamW fused={use_fused} | decay={len(other_decay_params)} param tensors, emb_lr=lr/4, nodecay={len(nodecay_params)}")

    # ── Checkpoint ─────────────────────────────────────────────────────────
    step = 0
    epoch = 0
    ckpt_block = 0
    ckpt_path = os.path.join(_DIR, "checkpoint.pt")
    safe_path = os.path.join(_DIR, "model_test.safetensors")

    if not test_mode:
        loaded = False
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("head.emb_weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded checkpoint: step {step} epoch {epoch} block {ckpt_block}")
            loaded = True
        elif hf and hf.download_checkpoint(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            ckpt["model"].pop("head.emb_weight", None)
            model.load_state_dict(ckpt["model"], strict=False)
            step = ckpt.get("step", 0)
            epoch = ckpt.get("epoch", 0)
            ckpt_block = ckpt.get("block", 0)
            del ckpt
            torch.cuda.empty_cache()
            print(f"Loaded HF checkpoint: step {step} epoch {epoch} block {ckpt_block}")
            loaded = True

        if loaded:
            print("\n── Generation test (MoSE anchos) ──")
            modes = [(None, "libre"), (1.0, "full"), (0.75, "75%"), (0.50, "50%")]
            for p in ["hola", "que es la inteligencia artificial", "en un lugar de la mancha", "hoy hace mucho calor"]:
                for w, tag in modes:
                    sample = generate_sample(model, tokenizer, device, prompt=p, max_new=50, width=w)
                    print(f"  [{p}][{tag}] → {sample}")
            print("── End test ──\n")

    # ── Data ────────────────────────────────────────────────────────────────
    if test_mode:
        with open(txt_path, "r", encoding="utf-8") as f:
            all_tokens = tokenizer.encode(f.read())
        n = len(all_tokens)
        epochs_do = 10
        total_steps = ((n - seq_len - 1) // (batch_size * seq_len)) * epochs_do
    else:
        bi = input(f"Block [{ckpt_block}]: ").strip()
        block_idx = int(bi) if bi else ckpt_block
        sd = train_data.TrainData(block_idx=block_idx)
        sd.load_tokens(tokenizer)
        n = len(sd.get_tokens())
        tokens_per_epoch = (n - seq_len - 1) // seq_len
        total_steps = (tokens_per_epoch // batch_size) * num_epochs
        epochs_do = num_epochs

    # ── Stats ──────────────────────────────────────────────
    emb_p = model.embedding.weight.numel()
    layer_p = sum(p.numel() for l in model.transformer.layers for p in l.parameters())
    norm_p = model.transformer.final_norm.weight.numel()
    total_p = emb_p + layer_p + norm_p
    print(f"Params: emb={emb_p:,} + {num_layers}capas={layer_p:,} + norm={norm_p} = {total_p:,}")
    print(f"dim={d_model} lay={num_layers} heads={num_heads} kv={num_kv_groups} seq={seq_len} bs={batch_size} ga={grad_accum} lr={lr}")
    gqa_cpt = 2 * num_kv_groups * head_dim
    if use_mla:
        a = model.transformer.layers[0].attention.qkv
        d_c_real = a.d_c; d_rot_real = a.d_rotate; d_c1_real = a.d_c1
        cpt = d_c_real + d_rot_real
        pct = 100 * (1 - cpt / gqa_cpt)
        xsa_tag = " + XSA" if use_xsa else ""
        qkn_tag = " + QK-Norm" if qk_norm else ""
        sn_tag = " + SandwichNorm" if use_sandwich_norm else ""
        print(f"MLA{xsa_tag}{qkn_tag}{sn_tag}: d_c={d_c_real} d_c1={d_c1_real} d_rot={d_rot_real} | cache: {gqa_cpt}→{cpt}B/tok ({pct:.0f}%)")
    else:
        cpt = gqa_cpt
    if use_moe:
        moe_layers = sum(1 for l in model.transformer.layers if l.use_moe)
        ntk_tag = " + NoisyTopK" if noise_std > 0 else ""
        print(f"MoE{ntk_tag}: {moe_layers}/{num_layers} MoE layers | {n_dense_start} dense start / {n_dense_end} dense end | n_exp={n_experts} top_k={top_k} n_shared={n_shared}")
    print(f"Tokens: {n:,} | Steps total: {total_steps}")

    # ── Train loop ──────────────────────────────────────────────────────────
    model.train()
    t0 = time.time()
    last_rpt_time = t0
    last_rpt_step = 0

    epoch = 0
    torch.cuda.empty_cache()
    while True:
        if test_mode:
            tokens = all_tokens
        else:
            tokens = sd.get_tokens()

        n_seq = (len(tokens) - seq_len - 1) // seq_len
        if n_seq <= 0:
            print(f"Block too small ({len(tokens)} tokens), skipping epoch {epoch}")
            epoch += 1
            if epoch >= epochs_do:
                break
            continue

        micro = 0
        w_last = 1.0  # ultimo ancho aleatorio Eq.(6), para el log
        for batch_start in range(0, n_seq, batch_size):
            if step >= total_steps:
                break
            batch_end = min(batch_start + batch_size, n_seq)
            x_list, y_list = [], []
            for i in range(batch_start, batch_end):
                idx = i * seq_len
                x = torch.tensor([tokens[idx + j] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor([tokens[idx + j + 1] for j in range(seq_len)], dtype=torch.long, device=device).unsqueeze(0)
                x_list.append(x)
                y_list.append(y)
            x = torch.cat(x_list, dim=0)
            y = torch.cat(y_list, dim=0)

            if micro == 0:
                lr_curr = get_lr(step, total_steps, warmup_steps, lr)
                for pg in opt.param_groups:
                    pg["lr"] = lr_curr * pg.get("lr_scale", 1.0)
                opt.zero_grad()

            if use_moe:
                # MoSE Eq.(6): dos forwards por minibatch (w_max + w aleatorio).
                # Un backward por forward (0.5x cada uno): misma matematica,
                # mitad de pico de memoria que retener los dos grafos.
                w_random = random.uniform(mose_w_min, mose_w_max)
                w_last = w_random
                if use_partial_rope:
                    logits_f, aux_f = model.forward_train_partial_rope(x, rotary_pct=rotary_pct, width=mose_w_max)
                else:
                    logits_f, aux_f = model(x, width=mose_w_max)
                loss_f = F.cross_entropy(logits_f.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                (0.5 * (loss_f + aux_f) / grad_accum).backward()
                loss_f_log = float(loss_f.detach())
                aux_f_log = float(aux_f.detach()) if isinstance(aux_f, torch.Tensor) else float(aux_f)
                del logits_f, loss_f, aux_f
                if use_partial_rope:
                    logits_r, aux_r = model.forward_train_partial_rope(x, rotary_pct=rotary_pct, width=w_random)
                else:
                    logits_r, aux_r = model(x, width=w_random)
                loss_r = F.cross_entropy(logits_r.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                (0.5 * (loss_r + aux_r) / grad_accum).backward()
                loss_r_log = float(loss_r.detach())
                aux_r_log = float(aux_r.detach()) if isinstance(aux_r, torch.Tensor) else float(aux_r)
                del logits_r, loss_r, aux_r
                # Escalares para el log (promedio Eq.6, sin grafo)
                loss = torch.tensor(0.5 * (loss_f_log + loss_r_log))
                aux_loss = torch.tensor(0.5 * (aux_f_log + aux_r_log))
            elif use_partial_rope:
                logits, aux_loss = model.forward_train_partial_rope(x, rotary_pct=rotary_pct)
                loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                loss = loss + aux_loss  # add MoE z-loss
                (loss / grad_accum).backward()
            else:
                logits, aux_loss = model(x)
                loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
                loss = loss + aux_loss  # add MoE z-loss
                (loss / grad_accum).backward()
            micro += 1

            if micro >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

                if step % 10 == 0:
                    # Per-parameter top gradients (as before)
                    grad_stats = []
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            continue
                        g = param.grad
                        grad_stats.append((name, g.norm().item(), g.abs().max().item()))
                    grad_stats.sort(key=lambda x: x[1], reverse=True)
                    top_stats = grad_stats[:6]
                    grad_report = ", ".join(
                        f"{name.split('.')[-1]} norm={norm:.4g} max={mx:.4g}"
                        for name, norm, mx in top_stats
                    )
                    print(f"  Gradients (top params): {grad_report}")

                    # Per-layer gradient norms (aggregate over parameters in each layer)
                    try:
                        layer_reports = []
                        for li, layer in enumerate(model.transformer.layers):
                            sq = 0.0
                            for p in layer.parameters():
                                if p.grad is None:
                                    continue
                                g = p.grad
                                # accumulate squared norms
                                ng = float(g.norm().item())
                                sq += ng * ng
                            ln = math.sqrt(sq) if sq > 0.0 else 0.0
                            layer_reports.append(f"L{li}={ln:.4g}")

                        # Embedding grad
                        emb_norm = 0.0
                        if hasattr(model, 'embedding') and getattr(model.embedding, 'weight', None) is not None:
                            ew = model.embedding.weight
                            if ew.grad is not None:
                                emb_norm = float(ew.grad.norm().item())

                        # Print embedding + first/last few layers to keep line manageable
                        if len(layer_reports) <= 12:
                            layer_str = " ".join(layer_reports)
                        else:
                            layer_str = " ".join(layer_reports[:6]) + " ... " + " ".join(layer_reports[-6:])
                        print(f"  Layer grads: Emb={emb_norm:.4g} | {layer_str}")
                    except Exception as e:
                        print(f"  Layer grad reporting failed: {e}")

                opt.step()
                opt.zero_grad()
                step += 1
                micro = 0

                if step % 10 == 0:
                    now = time.time()
                    tok = (step - last_rpt_step) * batch_size * grad_accum * seq_len
                    tps = tok / max(now - last_rpt_time, 0.001)
                    balance_strs = []
                    moe_dist = {}
                    total_z_loss = 0.0
                    total_lb_loss = 0.0
                    for li, layer in enumerate(model.transformer.layers):
                        if getattr(layer, 'use_moe', False) and hasattr(layer.ffn, 'last_counts'):
                            ffn = layer.ffn
                            csum = float(ffn.last_counts.sum().item()) or 1.0
                            pcts = (ffn.last_counts.float() / csum * 100).tolist()
                            moe_dist[f"L{li}"] = pcts
                            balance_strs.append(f"L{li}:{ffn.balance_str()}")
                            # collect z-loss / load-balance from MoE layer if available
                            try:
                                if hasattr(ffn, 'last_z_loss'):
                                    total_z_loss += float(ffn.last_z_loss.item()) if isinstance(ffn.last_z_loss, torch.Tensor) else float(ffn.last_z_loss)
                            except Exception:
                                pass
                            try:
                                if hasattr(ffn, 'last_load_balance_loss'):
                                    total_lb_loss += float(ffn.last_load_balance_loss.item()) if isinstance(ffn.last_load_balance_loss, torch.Tensor) else float(ffn.last_load_balance_loss)
                            except Exception:
                                pass
                    bal = " | ".join(balance_strs[:3])  # first 3 MoE layers only
                    print(f"e{epoch} s{step} loss {loss.item():.4f} lr {lr_curr:.6f} {tps:.0f}t/s z={total_z_loss:.6f} lb={total_lb_loss:.6f} w={w_last:.2f}")
                    if use_moe:
                        print(f"  MoSE fwd: full_loss={loss_f_log:.4f} random_loss={loss_r_log:.4f} (w={w_last:.2f}) full_aux={aux_f_log:.6g} random_aux={aux_r_log:.6g}")
                    if bal:
                        print(f"  MoE balance: {bal}")
                    if total_z_loss or total_lb_loss:
                        print(f"  MoE aux: z_loss={total_z_loss:.6g} load_balance={total_lb_loss:.6g}")
                    last_rpt_time = now
                    last_rpt_step = step
                    pm.log(step, loss.item(), lr_curr, tps, aux_loss.item() if isinstance(aux_loss, torch.Tensor) else None,
                           grad_norm=grad_norm.item(), moe_dist=moe_dist, z_loss=total_z_loss, load_balance_loss=total_lb_loss)

                if not test_mode and step % 50 == 0:
                    t_gen = time.time()
                    sample = generate_sample(model, tokenizer, device)
                    gen_tps = 100 / (time.time() - t_gen)
                    print(f"  >>> {sample}  [{gen_tps:.0f} tok/s]")

                if not test_mode and pusher and (time.time() - pusher.last_push) >= pusher.interval:
                    state = model.state_dict()
                    state.pop("head.emb_weight", None)
                    ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx if not test_mode else 0, "model": state}
                    torch.save(ckpt, ckpt_path)
                    pusher.maybe_push(ckpt_path, None, tok_path, step)
                    pm.plot(step)
                    pm.plot_grad_moe(step)
                    pm.upload(step)

        if micro > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            opt.zero_grad()
            step += 1
            micro = 0

        epoch += 1
        print(f"── Epoch {epoch} done: {step} steps ──")
        if epoch >= epochs_do:
            break

        if not test_mode:
            tokens = None
            sd.next_block()
            total_tokens = len(sd.get_tokens())
            n_seq = (total_tokens - seq_len - 1) // seq_len
            # NOTA: NO recalcular total_steps aqui — el LR coseno usa el total
            # original del inicio. Recalcularlo por bloque hace que el coseno
            # piense que estamos al final y cae el LR al minimo (0.2*lr).

    if not test_mode and hf:
        ckpt = {"step": step, "epoch": epoch, "block": sd.block_idx, "model": model.state_dict()}
        torch.save(ckpt, ckpt_path)
        hf.upload_checkpoint(ckpt_path, tok_path, step)

    print(f"Done! {step} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
