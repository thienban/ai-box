"""Convertit un fichier GGUF (ex. Bonsai 27B q1_0) vers le format SafeTensors AI Box.

Usage:
    python scripts/convert_gguf_to_aibox.py --gguf path/to/model.gguf --out ./bonsai-aibox
    python scripts/convert_gguf_to_aibox.py --gguf path/to/model.gguf --out ./bonsai-test --max-layers 2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

GROUP_SIZE = 128

TERNARY_SUFFIXES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)


def build_hadamard_128(device: torch.device) -> torch.Tensor:
    """Construit la matrice orthonormale de Walsh-Hadamard 128x128."""
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < GROUP_SIZE:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(GROUP_SIZE)


def pack_ternary_2bit_rows(w_ternary: torch.Tensor) -> np.ndarray:
    """Compacte [N, K] de {-1, 0, +1} en [N, K // 4] uint8 (4 poids/octet LSB-first)."""
    n_rows, k_cols = w_ternary.shape
    codes = torch.zeros((n_rows, k_cols), device=w_ternary.device, dtype=torch.uint8)
    codes[w_ternary == 1] = 1
    codes[w_ternary == -1] = 3
    codes = codes.reshape(n_rows, k_cols // 4, 4)
    packed = codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)
    return packed.cpu().numpy()


def quantize_hadamard_ternary(
    weight_f32: np.ndarray,
    h_128: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique H_128, quantifie en {-1, 0, +1} par groupes de 128 et packe en 2-bit."""
    w = torch.from_numpy(weight_f32).to(device=device, dtype=torch.float32)
    n_rows, k_cols = w.shape
    if k_cols % GROUP_SIZE != 0:
        raise ValueError(f"k_cols={k_cols} doit être multiple de {GROUP_SIZE}")

    w_groups = w.reshape(n_rows, k_cols // GROUP_SIZE, GROUP_SIZE)
    w_rot = w_groups @ h_128
    scales = w_rot.abs().mean(dim=-1).clamp(min=1e-6)
    w_ternary = torch.clamp(torch.round(w_rot / scales.unsqueeze(-1)), -1.0, 1.0)

    packed_u8 = pack_ternary_2bit_rows(w_ternary.reshape(n_rows, k_cols))
    scales_f16 = scales.to(torch.float16).cpu().numpy()
    return packed_u8, scales_f16


def get_gguf_scalar(reader: Any, keys: list[str], default: Any = None) -> Any:
    for key in keys:
        field = reader.fields.get(key)
        if field is not None and len(field.data) > 0:
            val = field.parts[field.data[0]]
            if hasattr(val, "tolist"):
                raw = val.tolist()
                return raw[0] if isinstance(raw, list) and len(raw) == 1 else raw
            return val
    return default


def extract_model_config(reader: Any, max_layers: int | None) -> dict[str, Any]:
    arch = get_gguf_scalar(reader, ["general.architecture"], "llama")
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8")

    num_layers = int(get_gguf_scalar(reader, [f"{arch}.block_count"], 64))
    if max_layers is not None:
        num_layers = min(num_layers, max_layers)

    hidden_size = int(get_gguf_scalar(reader, [f"{arch}.embedding_length"], 5120))
    intermediate_size = int(get_gguf_scalar(reader, [f"{arch}.feed_forward_length"], 17408))
    num_heads = int(get_gguf_scalar(reader, [f"{arch}.attention.head_count"], 40))
    num_kv_heads = int(get_gguf_scalar(reader, [f"{arch}.attention.head_count_kv"], num_heads))
    eps = float(get_gguf_scalar(reader, [f"{arch}.attention.layer_norm_rms_epsilon"], 1e-6))
    rope_theta = float(get_gguf_scalar(reader, [f"{arch}.rope.freq_base"], 10000.0))

    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    vocab_size = len(tokens_field.data) if tokens_field is not None else 151936

    return {
        "architecture": arch,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "rms_norm_eps": eps,
        "rope_theta": rope_theta,
    }


def extract_minimal_tokenizer(reader: Any) -> dict[str, Any]:
    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    vocab: dict[str, int] = {}
    if tokens_field is not None:
        for idx, part_idx in enumerate(tokens_field.data):
            raw_tok = bytes(reader.fields["tokenizer.ggml.tokens"].parts[part_idx])
            vocab[raw_tok.decode("utf-8", errors="replace")] = idx
    return {"model": {"type": "BPE", "vocab": vocab}}


def normalize_tensor_name(gguf_name: str) -> str:
    return gguf_name.replace("attn_output.weight", "attn_o.weight")


def should_include_tensor(name: str, max_layers: int | None) -> bool:
    if max_layers is None or not name.startswith("blk."):
        return True
    parts = name.split(".")
    if len(parts) >= 2 and parts[1].isdigit():
        return int(parts[1]) < max_layers
    return True


def convert_gguf(gguf_path: Path, out_dir: Path, max_layers: int | None) -> None:
    import gguf
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_128 = build_hadamard_128(device)
    print(f"Lecture GGUF : {gguf_path} (device={device})")

    reader = gguf.GGUFReader(str(gguf_path), mode="r")
    config = extract_model_config(reader, max_layers)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "quant_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "bits": 2,
                "format": "ternary_2bit",
                "group_size": GROUP_SIZE,
                "hadamard": True,
            },
            f,
            indent=2,
        )
    with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(extract_minimal_tokenizer(reader), f)

    tensors_out: dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if not should_include_tensor(tensor.name, max_layers):
            continue

        dequant = gguf.quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
        target_name = normalize_tensor_name(tensor.name)

        if tensor.name.endswith(TERNARY_SUFFIXES) and dequant.ndim == 2:
            packed_u8, scales_f16 = quantize_hadamard_ternary(dequant, h_128, device)
            scale_name = target_name.removesuffix(".weight") + ".scales"
            tensors_out[target_name] = packed_u8
            tensors_out[scale_name] = scales_f16
            print(f"  [2-bit] {target_name:32s} -> {packed_u8.shape} + scales {scales_f16.shape}")
        else:
            tensors_out[target_name] = dequant.astype(np.float16)
            print(f"  [FP16 ] {target_name:32s} -> {dequant.shape}")

    save_file(tensors_out, str(out_dir / "model.safetensors"))
    print(f"Export terminé dans {out_dir} ({len(tensors_out)} tenseurs).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convertit un GGUF en SafeTensors ternaire AI Box")
    parser.add_argument("--gguf", type=Path, required=True, help="Chemin vers le fichier .gguf")
    parser.add_argument("--out", type=Path, required=True, help="Dossier de sortie SafeTensors")
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Limiter le nombre de couches exportées (utile pour test rapide)",
    )
    args = parser.parse_args()
    convert_gguf(args.gguf, args.out, args.max_layers)


if __name__ == "__main__":
    main()
