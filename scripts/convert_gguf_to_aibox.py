"""Convertit un fichier GGUF (ex. Bonsai 27B q1_0) vers le format SafeTensors AI Box.

Usage:
    python scripts/convert_gguf_to_aibox.py --gguf path/to/model.gguf --out ./bonsai-aibox
    python scripts/convert_gguf_to_aibox.py --gguf path/to/model.gguf --out ./bonsai-test --max-layers 2
"""

from __future__ import annotations

import argparse
import gc
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
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_out.weight",
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


def pack_binary_1bit_rows(w_binary: torch.Tensor) -> np.ndarray:
    """Compacte [N, K] de {-1, +1} en [N, K // 8] uint8 (8 poids/octet LSB-first: 1=+1, 0=-1)."""
    n_rows, k_cols = w_binary.shape
    bits = (w_binary > 0).to(torch.uint8).reshape(n_rows, k_cols // 8, 8)
    packed = bits[..., 0]
    for b in range(1, 8):
        packed = packed | (bits[..., b] << b)
    return packed.cpu().numpy()


def quantize_hadamard(
    weight_f32: np.ndarray,
    h_128: torch.Tensor,
    device: torch.device,
    bits: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Applique H_128, quantifie en 1-bit {-1, +1} ou 2-bit {-1, 0, +1} par groupes de 128."""
    w = torch.from_numpy(weight_f32).to(device=device, dtype=torch.float32)
    n_rows, k_cols = w.shape
    if k_cols % GROUP_SIZE != 0:
        raise ValueError(f"k_cols={k_cols} doit être multiple de {GROUP_SIZE}")

    w_groups = w.reshape(n_rows, k_cols // GROUP_SIZE, GROUP_SIZE)
    w_rot = w_groups @ h_128
    scales = w_rot.abs().mean(dim=-1).clamp(min=1e-6)

    if bits == 1:
        w_binary = torch.where(w_rot >= 0.0, 1.0, -1.0)
        packed_u8 = pack_binary_1bit_rows(w_binary.reshape(n_rows, k_cols))
    else:
        w_ternary = torch.clamp(torch.round(w_rot / scales.unsqueeze(-1)), -1.0, 1.0)
        packed_u8 = pack_ternary_2bit_rows(w_ternary.reshape(n_rows, k_cols))

    scales_f16 = scales.to(torch.float16).cpu().numpy()
    return packed_u8, scales_f16


def quantize_grouped_linear(weight_f32: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantifie [N, K] sans Hadamard par groupes de 128 en Q8 (int8->uint8) ou Q4 (offset +8, 2/octet)."""
    if bits not in (4, 8):
        raise ValueError(f"bits={bits} non supporté (attendu 4 ou 8)")
    n_rows, k_cols = weight_f32.shape
    if k_cols % GROUP_SIZE != 0:
        raise ValueError(f"k_cols={k_cols} doit être multiple de {GROUP_SIZE}")
    n_groups = k_cols // GROUP_SIZE
    out_cols = k_cols if bits == 8 else k_cols // 2
    packed_out = np.empty((n_rows, out_cols), dtype=np.uint8)
    scales_out = np.empty((n_rows, n_groups), dtype=np.float16)
    qmax = 127.0 if bits == 8 else 7.0

    for start in range(0, n_rows, 4096):
        end = min(start + 4096, n_rows)
        chunk = weight_f32[start:end].reshape(-1, n_groups, GROUP_SIZE)
        scales = np.maximum(np.abs(chunk).max(axis=-1), 1e-8) / qmax
        q = np.clip(np.round(chunk / scales[..., None]), -qmax, qmax).astype(np.int16)
        scales_out[start:end] = scales.astype(np.float16)
        if bits == 8:
            packed_out[start:end] = q.astype(np.int8).view(np.uint8).reshape(-1, k_cols)
        else:
            u4 = (q + 8).astype(np.uint8).reshape(-1, k_cols // 2, 2)
            packed_out[start:end] = u4[..., 0] | (u4[..., 1] << 4)
    return packed_out, scales_out


Q2_GROUP_SIZE = 64


def quantize_affine(weight_f32: np.ndarray, bits: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantifie [n_rows, k_cols] en 2-bit ou 4-bit affine par groupes de 64."""
    n_rows, k_cols = weight_f32.shape
    assert k_cols % Q2_GROUP_SIZE == 0, f"k_cols={k_cols} non multiple de {Q2_GROUP_SIZE}"
    groups = k_cols // Q2_GROUP_SIZE
    grouped = weight_f32.astype(np.float32).reshape(n_rows, groups, Q2_GROUP_SIZE)
    min_w = grouped.min(axis=-1, keepdims=True)
    max_w = grouped.max(axis=-1, keepdims=True)
    levels = 15 if bits == 4 else 3
    scales_f16 = ((max_w - min_w) / float(levels)).astype(np.float16)
    zeros_f16 = min_w.astype(np.float16)
    s_f32 = scales_f16.astype(np.float32)
    z_f32 = zeros_f16.astype(np.float32)
    safe_s = np.where(s_f32 > 1e-12, s_f32, 1.0)
    codes = np.where(s_f32 > 1e-12, np.clip(np.round((grouped - z_f32) / safe_s), 0, levels), 0).astype(np.uint8)
    if bits == 4:
        codes_2 = codes.reshape(n_rows, k_cols // 2, 2)
        packed = (codes_2[..., 0] | (codes_2[..., 1] << 4)).astype(np.uint8)
    else:
        codes_4 = codes.reshape(n_rows, k_cols // 4, 4)
        packed = (
            codes_4[..., 0]
            | (codes_4[..., 1] << 2)
            | (codes_4[..., 2] << 4)
            | (codes_4[..., 3] << 6)
        ).astype(np.uint8)
    return packed, scales_f16.reshape(n_rows, groups), zeros_f16.reshape(n_rows, groups)


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
    if isinstance(arch, list):
        arch = bytes(arch).decode("utf-8")
    elif isinstance(arch, bytes):
        arch = arch.decode("utf-8")

    num_layers = int(get_gguf_scalar(reader, [f"{arch}.block_count"], 64))
    if max_layers is not None:
        num_layers = min(num_layers, max_layers)

    hidden_size = int(get_gguf_scalar(reader, [f"{arch}.embedding_length"], 5120))
    intermediate_size = int(get_gguf_scalar(reader, [f"{arch}.feed_forward_length"], 17408))
    num_heads = int(get_gguf_scalar(reader, [f"{arch}.attention.head_count"], 40))
    num_kv_heads = int(get_gguf_scalar(reader, [f"{arch}.attention.head_count_kv"], num_heads))
    head_dim = get_gguf_scalar(reader, [f"{arch}.attention.key_length"], None)
    rope_dim = get_gguf_scalar(reader, [f"{arch}.rope.dimension_count"], None)
    full_attn_interval = get_gguf_scalar(reader, [f"{arch}.full_attention_interval"], None)
    eps = float(get_gguf_scalar(reader, [f"{arch}.attention.layer_norm_rms_epsilon"], 1e-6))
    rope_theta = float(get_gguf_scalar(reader, [f"{arch}.rope.freq_base"], 10000.0))

    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    vocab_size = len(tokens_field.data) if tokens_field is not None else 151936

    cfg: dict[str, Any] = {
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
    if head_dim is not None:
        cfg["head_dim"] = int(head_dim)
    if rope_dim is not None:
        cfg["rope_dim"] = int(rope_dim)
    if full_attn_interval is not None:
        cfg["full_attention_interval"] = int(full_attn_interval)

    ssm_fields = {
        "ssm_conv_kernel": f"{arch}.ssm.conv_kernel",
        "ssm_group_count": f"{arch}.ssm.group_count",
        "ssm_inner_size": f"{arch}.ssm.inner_size",
        "ssm_state_size": f"{arch}.ssm.state_size",
        "ssm_time_step_rank": f"{arch}.ssm.time_step_rank",
    }
    for out_key, gguf_key in ssm_fields.items():
        val = get_gguf_scalar(reader, [gguf_key], None)
        if val is not None:
            cfg[out_key] = int(val)

    moe_fields = {
        "sliding_window": f"{arch}.attention.sliding_window",
        "num_experts": f"{arch}.expert_count",
        "num_experts_per_tok": f"{arch}.expert_used_count",
        "moe_intermediate_size": f"{arch}.expert_feed_forward_length",
        "head_dim_swa": f"{arch}.attention.key_length_swa",
        "rope_dim_swa": f"{arch}.rope.dimension_count_swa",
    }
    for out_key, gguf_key in moe_fields.items():
        val = get_gguf_scalar(reader, [gguf_key], None)
        if val is not None:
            cfg[out_key] = int(val)
    rope_theta_swa = get_gguf_scalar(reader, [f"{arch}.rope.freq_base_swa"], None)
    if rope_theta_swa is not None:
        cfg["rope_theta_swa"] = float(rope_theta_swa)

    return cfg


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


def dequantize_q1_0(data: np.ndarray, target_dtype: Any = np.float32) -> np.ndarray:
    """Dequantize GGUF Q1_0 par tranches de lignes pour limiter la RAM (128 poids/bloc)."""
    raw = np.ascontiguousarray(data.view(np.uint8))
    out_shape = (*raw.shape[:-1], (raw.shape[-1] // 18) * GROUP_SIZE)
    if raw.ndim == 2 and raw.shape[0] > 4096:
        out = np.empty(out_shape, dtype=target_dtype)
        for start in range(0, raw.shape[0], 4096):
            end = min(start + 4096, raw.shape[0])
            out[start:end] = dequantize_q1_0(raw[start:end], target_dtype)
        return out

    blocks = raw.reshape(-1, 18)
    scales = blocks[:, :2].view(np.float16).astype(np.float32)
    bits = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little")
    signs = bits.astype(np.float32) * 2.0 - 1.0
    return (signs * scales).astype(target_dtype).reshape(out_shape)


def q1_0_to_binary_1bit(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copie directe GGUF Q1_0 vers le format natif 1-bit AI Box (16 octets/groupe) + scales FP16."""
    raw = np.ascontiguousarray(data.view(np.uint8))
    n_rows = raw.shape[0]
    n_groups = raw.shape[1] // 18
    blocks = raw.reshape(n_rows, n_groups, 18)

    scales_f16 = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(n_rows, n_groups)
    packed_u8 = np.ascontiguousarray(blocks[:, :, 2:]).reshape(n_rows, n_groups * 16)
    return packed_u8, scales_f16


def q1_0_to_ternary_2bit(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convertit GGUF Q1_0 sans perte vers le format 2-bit AI Box (1->+1, 3->-1) + scales FP16."""
    raw = np.ascontiguousarray(data.view(np.uint8))
    n_rows = raw.shape[0]
    n_groups = raw.shape[1] // 18
    k_cols = n_groups * GROUP_SIZE
    blocks = raw.reshape(n_rows, n_groups, 18)

    scales_f16 = np.ascontiguousarray(blocks[:, :, :2]).view(np.float16).reshape(n_rows, n_groups)
    bits = np.unpackbits(blocks[:, :, 2:], axis=2, bitorder="little").reshape(n_rows, k_cols)
    codes = np.where(bits == 1, np.uint8(1), np.uint8(3)).reshape(n_rows, k_cols // 4, 4)
    packed_u8 = codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)
    return np.ascontiguousarray(packed_u8), np.ascontiguousarray(scales_f16)


def dequantize_tensor(tensor: Any, gguf_mod: Any, target_dtype: Any = np.float32) -> np.ndarray:
    if getattr(tensor.tensor_type, "name", "") == "Q1_0":
        return dequantize_q1_0(tensor.data, target_dtype=target_dtype)
    return gguf_mod.quants.dequantize(tensor.data, tensor.tensor_type).astype(target_dtype)


def convert_gguf(
    gguf_path: Path,
    out_dir: Path,
    max_layers: int | None,
    force_hadamard: bool = False,
    bits: int = 2,
    lm_head_bits: int = 16,
    quant_format: str = "auto",
    embed_bits: int = 16,
    shard_mb: int = 0,
    moe_bits: int | None = None,
) -> None:
    import gguf
    from safetensors.numpy import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_128 = build_hadamard_128(device)
    print(f"Lecture GGUF : {gguf_path} (device={device})")

    reader = gguf.GGUFReader(str(gguf_path), mode="r")
    config = extract_model_config(reader, max_layers)
    has_q1_0 = any(getattr(t.tensor_type, "name", "") == "Q1_0" for t in reader.tensors)
    use_hadamard = force_hadamard or not has_q1_0
    print(f"Mode quantification : hadamard={use_hadamard} (Q1_0 détecté={has_q1_0})")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "quant_config.json", "w", encoding="utf-8") as f:
        is_affine = quant_format in ("q2_affine", "q4_affine")
        fmt = quant_format if is_affine else ("binary_1bit" if bits == 1 else "ternary_2bit")
        eff_moe_bits = moe_bits if moe_bits is not None else (4 if quant_format == "q4_affine" else 2)
        moe_fmt = f"q{eff_moe_bits}_affine" if is_affine else fmt
        json.dump(
            {
                "bits": 4 if quant_format == "q4_affine" else bits,
                "format": fmt,
                "moe_format": moe_fmt,
                "group_size": Q2_GROUP_SIZE if is_affine else GROUP_SIZE,
                "hadamard": False if is_affine else use_hadamard,
            },
            f,
            indent=2,
        )
    with open(out_dir / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(extract_minimal_tokenizer(reader), f)

    tensors_out: dict[str, np.ndarray] = {}
    shard_idx = 1
    total_tensors = 0

    def flush_shard() -> None:
        nonlocal shard_idx, total_tensors
        if not tensors_out:
            return
        if shard_mb > 0:
            shard_path = out_dir / f"model-{shard_idx:05d}.safetensors"
            save_file(tensors_out, str(shard_path))
            print(f"  [SHARD] {shard_path.name} écrit ({len(tensors_out)} tenseurs).", flush=True)
            shard_idx += 1
            total_tensors += len(tensors_out)
            tensors_out.clear()
            gc.collect()

    for tensor in reader.tensors:
        if not should_include_tensor(tensor.name, max_layers):
            continue

        target_name = normalize_tensor_name(tensor.name)
        is_ternary = tensor.name.endswith(TERNARY_SUFFIXES) and len(tensor.shape) == 2

        if is_ternary:
            if quant_format in ("q2_affine", "q4_affine"):
                affine_bits = 4 if quant_format == "q4_affine" else 2
                dequant = dequantize_tensor(tensor, gguf, target_dtype=np.float32)
                packed_u8, scales_f16, zeros_f16 = quantize_affine(dequant, bits=affine_bits)
                prefix = target_name.removesuffix(".weight")
                tensors_out[target_name] = packed_u8
                tensors_out[prefix + ".scales"] = scales_f16
                tensors_out[prefix + ".zeros"] = zeros_f16
                print(
                    f"  [Q{affine_bits}_AF] {target_name:32s} -> {packed_u8.shape} + scales/zeros {scales_f16.shape}",
                    flush=True,
                )
            else:
                if not use_hadamard and getattr(tensor.tensor_type, "name", "") == "Q1_0":
                    if bits == 1:
                        packed_u8, scales_f16 = q1_0_to_binary_1bit(tensor.data)
                    else:
                        packed_u8, scales_f16 = q1_0_to_ternary_2bit(tensor.data)
                else:
                    dequant = dequantize_tensor(tensor, gguf, target_dtype=np.float32)
                    packed_u8, scales_f16 = quantize_hadamard(dequant, h_128, device, bits=bits)
                scale_name = target_name.removesuffix(".weight") + ".scales"
                tensors_out[target_name] = packed_u8
                tensors_out[scale_name] = scales_f16
                print(f"  [{bits}-bit] {target_name:32s} -> {packed_u8.shape} + scales {scales_f16.shape}", flush=True)
        elif quant_format in ("q2_affine", "q4_affine") and len(tensor.shape) == 3 and tensor.name.endswith("ffn_gate_up_exps.weight"):
            affine_bits = moe_bits if moe_bits is not None else (4 if quant_format == "q4_affine" else 2)
            dequant = dequantize_tensor(tensor, gguf, target_dtype=np.float32)
            num_e, two_inter, hidden = dequant.shape
            inter = two_inter // 2
            gate_flat = dequant[:, :inter, :].reshape(num_e * inter, hidden)
            up_flat = dequant[:, inter:, :].reshape(num_e * inter, hidden)
            base_prefix = target_name.removesuffix("ffn_gate_up_exps.weight")
            for tag, mat in (("ffn_gate_exps", gate_flat), ("ffn_up_exps", up_flat)):
                packed_u8, scales_f16, zeros_f16 = quantize_affine(mat, bits=affine_bits)
                tensors_out[f"{base_prefix}{tag}.weight"] = packed_u8
                tensors_out[f"{base_prefix}{tag}.scales"] = scales_f16
                tensors_out[f"{base_prefix}{tag}.zeros"] = zeros_f16
                print(f"  [Q{affine_bits}_EX] {base_prefix}{tag}.weight -> {packed_u8.shape}", flush=True)
        elif quant_format in ("q2_affine", "q4_affine") and len(tensor.shape) == 3 and tensor.name.endswith("ffn_down_exps.weight"):
            affine_bits = moe_bits if moe_bits is not None else (4 if quant_format == "q4_affine" else 2)
            dequant = dequantize_tensor(tensor, gguf, target_dtype=np.float32)
            num_e, hidden, inter = dequant.shape
            packed_u8, scales_f16, zeros_f16 = quantize_affine(dequant.reshape(num_e * hidden, inter), bits=affine_bits)
            prefix = target_name.removesuffix(".weight")
            tensors_out[target_name] = packed_u8
            tensors_out[prefix + ".scales"] = scales_f16
            tensors_out[prefix + ".zeros"] = zeros_f16
            print(f"  [Q{affine_bits}_EX] {target_name:32s} -> {packed_u8.shape}", flush=True)
        elif target_name == "output.weight" and lm_head_bits in (4, 8):
            dequant = dequantize_tensor(tensor, gguf, target_dtype=np.float32)
            packed_u8, scales_f16 = quantize_grouped_linear(dequant, lm_head_bits)
            tensors_out["output.weight"] = packed_u8
            tensors_out["output.scales"] = scales_f16
            print(f"  [Q{lm_head_bits}   ] {target_name:32s} -> {packed_u8.shape} + scales {scales_f16.shape}", flush=True)
        elif target_name == "token_embd.weight" and embed_bits in (4, 8):
            dequant = dequantize_tensor(tensor, gguf, target_dtype=np.float32)
            packed_u8, scales_f16 = quantize_grouped_linear(dequant, embed_bits)
            tensors_out["token_embd.weight"] = packed_u8
            tensors_out["token_embd.scales"] = scales_f16
            print(f"  [Q{embed_bits}   ] {target_name:32s} -> {packed_u8.shape} + scales {scales_f16.shape}", flush=True)
        else:
            dequant_f16 = dequantize_tensor(tensor, gguf, target_dtype=np.float16)
            tensors_out[target_name] = dequant_f16
            print(f"  [FP16 ] {target_name:32s} -> {dequant_f16.shape}", flush=True)

        if shard_mb > 0:
            cur_bytes = sum(arr.nbytes for arr in tensors_out.values())
            if cur_bytes >= shard_mb * 1024 * 1024:
                flush_shard()

    if shard_mb > 0:
        flush_shard()
        print(f"Export terminé dans {out_dir} ({total_tensors} tenseurs en {shard_idx - 1} shards).")
    else:
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
    parser.add_argument(
        "--force-hadamard",
        action="store_true",
        help="Forcer la rotation Hadamard + re-quantification même sur Q1_0",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[1, 2],
        default=2,
        help="Largeur des poids quantifiés (1=binaire {-1,+1}, 2=ternaire {-1,0,+1})",
    )
    parser.add_argument(
        "--lm-head-bits",
        type=int,
        choices=[4, 8, 16],
        default=16,
        help="Largeur de quantification de output.weight / lm_head (4=Q4, 8=Q8, 16=FP16)",
    )
    parser.add_argument(
        "--quant-format",
        type=str,
        choices=["auto", "q2_affine", "q4_affine"],
        default="auto",
        help="Format de quantification (auto=ternaire/1-bit Hadamard, q2_affine=2-bit affine, q4_affine=4-bit affine)",
    )
    parser.add_argument(
        "--embed-bits",
        type=int,
        choices=[4, 8, 16],
        default=16,
        help="Largeur de quantification de token_embd.weight (4=Q4, 8=Q8, 16=FP16)",
    )
    parser.add_argument(
        "--shard-mb",
        type=int,
        default=0,
        help="Taille max par shard SafeTensors en Mo (0=fichier unique model.safetensors)",
    )
    parser.add_argument(
        "--moe-bits",
        type=int,
        choices=[2, 4],
        default=None,
        help="Largeur affine spécifique des experts MoE (ex: 2 pour mixte Q4 attention + Q2 experts)",
    )
    args = parser.parse_args()
    convert_gguf(
        args.gguf,
        args.out,
        args.max_layers,
        force_hadamard=args.force_hadamard,
        bits=args.bits,
        lm_head_bits=args.lm_head_bits,
        quant_format=args.quant_format,
        embed_bits=args.embed_bits,
        shard_mb=args.shard_mb,
        moe_bits=args.moe_bits,
    )


if __name__ == "__main__":
    main()
