"""Weight conversion between Rust (Burn) and PyTorch formats.

Burn CompactRecorder saves .mpk files with flat tensor data.
This module maps parameter names between the two frameworks.
"""

import os
import json
import torch
import numpy as np


# ─── PyTorch -> Burn name mapping ───────────────────────────────────────────

PYTORCH_TO_BURN = {
    # Embedding
    "embedding.weight": "embedding.weight",
    # Transformer layers
    "transformer.layers.{i}.attn_norm.weight": "transformer.layers.{i}.attn_norm.weight",
    "transformer.layers.{i}.attention.qkv.q_proj.weight": "transformer.layers.{i}.attention.qkv.q_proj.weight",
    "transformer.layers.{i}.attention.qkv.k_proj.weight": "transformer.layers.{i}.attention.qkv.k_proj.weight",
    "transformer.layers.{i}.attention.qkv.v_proj.weight": "transformer.layers.{i}.attention.qkv.v_proj.weight",
    "transformer.layers.{i}.attention.o_proj.o_proj.weight": "transformer.layers.{i}.attention.o_proj.o_proj.weight",
    "transformer.layers.{i}.ffn_norm.weight": "transformer.layers.{i}.ffn_norm.weight",
    # SwiGLU FFN
    "transformer.layers.{i}.ffn.gate_proj.weight": "transformer.layers.{i}.ffn.gate_proj.weight",
    "transformer.layers.{i}.ffn.up_proj.weight": "transformer.layers.{i}.ffn.up_proj.weight",
    "transformer.layers.{i}.ffn.down_proj.weight": "transformer.layers.{i}.ffn.down_proj.weight",
    # Final norm + head
    "transformer.final_norm.weight": "transformer.final_norm.weight",
    "head.weight": "head.weight",
    # x0
    "x0_lambdas": "x0_lambdas",
}


def export_state_dict(state_dict: dict, path: str):
    """Export PyTorch state_dict to safetensors (names are already compatible)."""
    from safetensors.torch import save_file
    save_file(state_dict, path)
    print(f"Exported {len(state_dict)} tensors to {path}")


def print_param_shapes(model):
    """Print all parameter names and shapes (useful for debugging name mismatches)."""
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)}")


def load_burn_checkpoint(model, burn_state_dir: str):
    """Load weights from a Burn state directory.

    Burn CompactRecorder saves tensors as individual .bin files
    with a metadata.json describing the structure.
    This is a placeholder - actual implementation depends on Burn's
    serialization format.
    """
    meta_path = os.path.join(burn_state_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.json not found in {burn_state_dir}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    state_dict = {}
    for name, info in metadata.items():
        tensor_path = os.path.join(burn_state_dir, info.get("file", f"{name}.bin"))
        if os.path.exists(tensor_path):
            data = np.fromfile(tensor_path, dtype=np.float32)
            shape = info.get("shape", [len(data)])
            state_dict[name] = torch.from_numpy(data).reshape(shape)

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(state_dict)} tensors from Burn checkpoint")
    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from python.model import TransformerLM

    # Example: create model and print param mapping
    model = TransformerLM(
        vocab_size=16000,
        d_model=768,
        num_layers=24,
        num_heads=12,
        num_kv_groups=4,
        use_swiglu=True,
    )

    print("PyTorch parameter names:")
    print_param_shapes(model)

    print(f"\nTotal params: {sum(p.numel() for p in model.parameters()):,}")
