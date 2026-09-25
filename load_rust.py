"""Load Rust (Burn) exported models into PyTorch.

Usage:
    from python.load_rust import load_rust_model

    model = load_rust_model(
        safetensors_path="model_rust.safetensors",
        mapping_path="model_rust_mapping.json",
        vocab_size=16000,
        d_model=768,
        num_layers=24,
        num_heads=12,
        num_kv_groups=4,
    )
"""

import json
import os
import torch
import torch.nn as nn


def load_safetensors_with_mapping(safetensors_path: str, mapping_path: str):
    """Load safetensors file using mapping for metadata."""
    from safetensors.torch import load_file

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    state_dict = load_file(safetensors_path)
    return state_dict, mapping["parameters"]


def convert_burn_to_pytorch(burn_state: dict, mapping: dict, model):
    """Convert Burn parameter names to PyTorch state_dict format.

    Burn uses the same naming as our Python model, so this is mostly a passthrough.
    But we need to handle any naming differences.
    """
    pytorch_state = {}

    for burn_name, metadata in mapping.items():
        if burn_name in burn_state:
            tensor = burn_state[burn_name]

            if tensor.dtype == torch.float16 or tensor.dtype == torch.bfloat16:
                tensor = tensor.float()

            if burn_name in model.state_dict():
                expected_shape = model.state_dict()[burn_name].shape
                if list(tensor.shape) == list(expected_shape):
                    pytorch_state[burn_name] = tensor
                else:
                    print(f"  Shape mismatch for {burn_name}: {list(tensor.shape)} vs {list(expected_shape)}, skipping")
            else:
                print(f"  Unknown parameter in model: {burn_name}")

    return pytorch_state


def load_rust_model(
    safetensors_path: str,
    mapping_path: str,
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    num_kv_groups: int = 4,
    use_swiglu: bool = True,
    use_x0: bool = True,
    max_seq_len: int = 2048,
    device: torch.device = None,
):
    """Load a Rust-exported model into PyTorch TransformerLM.

    Args:
        safetensors_path: path to .safetensors file exported from Rust
        mapping_path: path to mapping.json
        vocab_size, d_model, num_layers, num_heads, num_kv_groups: model config
        use_swiglu: whether model uses SwiGLU FFN
        use_x0: whether model has x0 injection
        device: target device
    """
    from python.model import TransformerLM

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict, mapping = load_safetensors_with_mapping(safetensors_path, mapping_path)

    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_groups=num_kv_groups,
        use_swiglu=use_swiglu,
        use_x0=use_x0,
        max_seq_len=max_seq_len,
    ).to(device)

    converted = convert_burn_to_pytorch(state_dict, mapping, model)
    missing, unexpected = model.load_state_dict(converted, strict=False)

    print(f"Loaded {len(converted)} parameters")
    if missing:
        print(f"  Missing: {missing}")
    if unexpected:
        print(f"  Unexpected: {unexpected}")

    return model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    if len(sys.argv) < 2:
        print("Usage: python -m python.load_rust <safetensors_path> [mapping_path]")
        print("Example: python -m python.load_rust model_test.safetensors model_test_mapping.json")
        sys.exit(1)

    safetensors_path = sys.argv[1]
    mapping_path = sys.argv[2] if len(sys.argv) > 2 else safetensors_path.replace(".safetensors", "_mapping.json")

    if not os.path.exists(safetensors_path):
        print(f"Error: {safetensors_path} not found")
        sys.exit(1)
    if not os.path.exists(mapping_path):
        print(f"Error: {mapping_path} not found")
        sys.exit(1)

    print(f"Loading from {safetensors_path}...")
    model = load_rust_model(safetensors_path, mapping_path, vocab_size=256, d_model=256, num_layers=6, num_heads=8, num_kv_groups=4)
    print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")
