"""Training history logging + plotting + HF sync.

Usage:
    from plot import PlotManager
    pm = PlotManager(hf, save_dir)
    pm.log(step, loss, lr, tps)
    pm.plot(step)    # saves plot_train_step_NNNN.png
    pm.upload(step)  # pushes plot + history JSON to HF
"""

import os, json, time
from pathlib import Path


class PlotManager:
    """Logs training metrics to JSON, generates loss plots, syncs to HuggingFace.

    JSON is updated locally after each log. Plot is generated + uploaded
    periodically (every plot_interval steps).
    """

    def __init__(self, hf, save_dir=".", history_file="train_history.json",
                 plot_interval=256):
        self.hf = hf
        self.save_dir = Path(save_dir)
        self.history_file = self.save_dir / history_file
        self.plot_interval = plot_interval
        self.last_uploaded_step = 0
        self.history = self._load_or_download()

    def _load_or_download(self):
        """Load local JSON, or fall back to HF download, or start fresh."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        if self.hf:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(
                    repo_id=self.hf.repo_id,
                    filename=self.history_file.name,
                    revision=self.hf.revision,
                    token=self.hf._get_token(),
                )
                import shutil
                shutil.copy2(path, self.history_file)
                print(f"Downloaded {self.history_file.name} from HF")
                with open(self.history_file) as f:
                    return json.load(f)
            except Exception as e:
                print(f"No remote history ({e}), starting fresh")
        return []

    def log(self, step, loss, lr=None, tps=None, aux_loss=None,
            grad_norm=None, moe_dist=None, z_loss=None, load_balance_loss=None):
        """Append one row to history and save JSON."""
        entry = {"step": step, "loss": loss, "time": time.time()}
        if lr is not None:
            entry["lr"] = lr
        if tps is not None:
            entry["tps"] = tps
        if aux_loss is not None:
            entry["aux_loss"] = aux_loss
        if grad_norm is not None:
            entry["grad_norm"] = grad_norm
        if moe_dist is not None:
            entry["moe_dist"] = moe_dist
        if z_loss is not None:
            entry["z_loss"] = z_loss
        if load_balance_loss is not None:
            entry["load_balance_loss"] = load_balance_loss
        self.history.append(entry)
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def plot(self, step):
        """Generate loss plot as plot_train_step_{step}.png. No-op if no matplotlib."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if len(self.history) < 2:
            return
        steps = [e["step"] for e in self.history]
        losses = [e["loss"] for e in self.history]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(steps, losses, label="loss", color="tab:blue")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss", color="tab:blue")
        has_aux = any("aux_loss" in e for e in self.history)
        has_z = any("z_loss" in e for e in self.history)
        has_lb = any("load_balance_loss" in e for e in self.history)
        if has_aux or has_z or has_lb:
            ax2 = ax1.twinx()
            if has_aux:
                aux = [e.get("aux_loss", 0) for e in self.history]
                ax2.plot(steps, aux, label="aux_loss", alpha=0.6, color="tab:orange")
            if has_z:
                z = [e.get("z_loss", 0) for e in self.history]
                ax2.plot(steps, z, label="z_loss", alpha=0.6, color="tab:green")
            if has_lb:
                lb = [e.get("load_balance_loss", 0) for e in self.history]
                ax2.plot(steps, lb, label="load_balance_loss", alpha=0.6, color="tab:purple", linestyle="--")
            ax2.set_ylabel("aux/z/lb", color="tab:orange")
        ax1.set_title(f"Training loss (step {step})")
        fig.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        path = self.save_dir / f"plot_train_step_{step}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def plot_grad_moe(self, step):
        """Generate gradient + MoE distribution plot. Skips entries lacking data."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        grad_entries = [(e["step"], e["grad_norm"]) for e in self.history if "grad_norm" in e]
        moe_entries = [e for e in self.history if "moe_dist" in e]
        if not grad_entries and not moe_entries:
            return
        moe_layers = list(moe_entries[0]["moe_dist"].keys()) if moe_entries else []
        n_moe = len(moe_layers)
        n_rows = (n_moe + 3) // 4  # up to 4 cols
        fig = plt.figure(figsize=(14, 3 + 2.5 * n_rows))
        gs = fig.add_gridspec(n_rows + 1, 4, height_ratios=[1] + [0.6] * n_rows,
                               hspace=0.3, wspace=0.3)
        ax_top = fig.add_subplot(gs[0, :])
        if grad_entries:
            gs_, gn = zip(*grad_entries)
            ax_top.plot(gs_, gn, color="tab:red", alpha=0.7)
            ax_top.set_ylabel("grad norm")
            ax_top.grid(True, alpha=0.3)
            ax_top.set_title(f"Gradient norm (step {step})")
        else:
            ax_top.set_visible(False)
        if moe_entries:
            ms = [e["step"] for e in moe_entries]
            for idx, layer in enumerate(moe_layers):
                r, c = divmod(idx, 4)
                ax = fig.add_subplot(gs[r + 1, c])
                normalized_series = []
                n_experts = len(moe_entries[0]["moe_dist"][layer])
                for ei in range(n_experts):
                    series = []
                    for e in moe_entries:
                        layer_vals = e["moe_dist"][layer]
                        total = sum(layer_vals) or 1.0
                        series.append(float(layer_vals[ei]) / total * 100.0)
                    normalized_series.append(series)
                ax.stackplot(ms, *normalized_series, alpha=0.6)
                ax.set_title(layer, fontsize=9)
                ax.set_ylim(0, 100)
                if r < n_rows - 1:
                    ax.tick_params(labelbottom=False)
        fig.suptitle(f"MoE expert distribution (step {step})", fontsize=12)
        path = self.save_dir / f"train_grad_moe_{step}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    def upload(self, step):
        """Upload JSON (overwrite) + both plots to HF."""
        if not self.hf:
            return
        api = self.hf._get_api()
        self.hf.ensure_repo()
        self.hf.ensure_revision()
        msg = f"Training step {step}"

        if self.history_file.exists():
            api.upload_file(
                path_or_fileobj=str(self.history_file),
                path_in_repo=self.history_file.name,
                repo_id=self.hf.repo_id,
                revision=self.hf.revision,
                token=self.hf._get_token(),
                commit_message=msg,
            )
            print(f"  [plot] uploaded {self.history_file.name} to {self.hf.repo_id}@{self.hf.revision}")

        for name in [f"plot_train_step_{step}.png", f"train_grad_moe_{step}.png"]:
            p = self.save_dir / name
            if p.exists():
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo=p.name,
                    repo_id=self.hf.repo_id,
                    revision=self.hf.revision,
                    token=self.hf._get_token(),
                    commit_message=msg,
                )
                print(f"  [plot] uploaded {p.name}")
        self.last_uploaded_step = step
