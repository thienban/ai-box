"""Teste l'inférence réelle (sorties texte) de BtbAgent/bonsai-27b-aibox sur Colab GPU.

Garde les poids packés en 2-bit + scales FP16 en VRAM (~7-8 Go) et déquantifie
chaque projection à la volée pour tenir dans les 16 Go d'une T4 Colab.

Usage:
    python scripts/test_bonsai_inference.py
    python scripts/test_bonsai_inference.py --repo BtbAgent/bonsai-27b-aibox --max-new-tokens 48
    python scripts/test_bonsai_inference.py --inspect-only
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open

GROUP_SIZE = 128


def build_hadamard_128(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < GROUP_SIZE:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(GROUP_SIZE)).to(dtype=dtype)


def unpack_ternary_2bit(packed: torch.Tensor) -> torch.Tensor:
    """Dépacke [N, K // 4] uint8 (0->0, 1->+1, 3->-1) vers [N, K] float16."""
    b0 = packed & 0x03
    b1 = (packed >> 2) & 0x03
    b2 = (packed >> 4) & 0x03
    b3 = (packed >> 6) & 0x03
    codes = torch.stack([b0, b1, b2, b3], dim=-1).reshape(packed.shape[0], packed.shape[1] * 4)
    out = torch.zeros_like(codes, dtype=torch.float16)
    out[codes == 1] = 1.0
    out[codes == 3] = -1.0
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.float()
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * inv_rms).to(weight.dtype) * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(pos_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(q.dtype)[None, None, :, :]
    sin = emb.sin().to(q.dtype)[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class TernaryLinear:
    def __init__(self, packed_u8: torch.Tensor, scales_f16: torch.Tensor, h_128: torch.Tensor) -> None:
        self.packed_u8 = packed_u8
        self.scales_f16 = scales_f16
        self.h_128 = h_128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        m, k = x_2d.shape
        x_rot = (x_2d.reshape(m, k // GROUP_SIZE, GROUP_SIZE) @ self.h_128).reshape(m, k)

        w_tern = unpack_ternary_2bit(self.packed_u8)
        n = w_tern.shape[0]
        w_scaled = (
            w_tern.reshape(n, k // GROUP_SIZE, GROUP_SIZE) * self.scales_f16.unsqueeze(-1)
        ).reshape(n, k)
        out = F.linear(x_rot, w_scaled)
        return out.reshape(*orig_shape[:-1], n)


class BonsaiAiboxModel:
    def __init__(self, model_dir: Path, device: torch.device, max_layers: int | None = None) -> None:
        self.device = device
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.quant_config = json.loads((model_dir / "quant_config.json").read_text(encoding="utf-8"))

        self.num_layers = int(self.config["num_hidden_layers"])
        if max_layers is not None:
            self.num_layers = min(self.num_layers, max_layers)
        self.hidden_size = int(self.config["hidden_size"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.eps = float(self.config.get("rms_norm_eps", 1e-6))
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))

        self.h_128 = build_hadamard_128(device, torch.float16)
        self.tensors: dict[str, torch.Tensor] = {}

        safetensors_path = model_dir / "model.safetensors"
        print(f"Chargement de {safetensors_path} sur {device}...")
        with safe_open(str(safetensors_path), framework="pt", device=str(device)) as f:
            keys = sorted(f.keys())
            print(f"Total tenseurs dans SafeTensors : {len(keys)}")
            blk0 = [k for k in keys if k.startswith("blk.0.")]
            print(f"Exemple tenseurs couche 0 ({len(blk0)}) : {blk0}")

            for key in keys:
                if key.startswith("blk."):
                    layer_idx = int(key.split(".")[1])
                    if layer_idx >= self.num_layers:
                        continue
                self.tensors[key] = f.get_tensor(key)

        if "blk.0.attn_q.weight" in self.tensors:
            q_shape = self.tensors["blk.0.attn_q.weight"].shape
            self.head_dim = q_shape[0] // self.num_heads

        vram_gb = torch.cuda.memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        print(f"Modèle chargé ({self.num_layers} couches, head_dim={self.head_dim}, VRAM={vram_gb:.2f} Go).")

    def _linear(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        w_key = f"{prefix}.weight"
        s_key = f"{prefix}.scales"
        if s_key in self.tensors:
            return TernaryLinear(self.tensors[w_key], self.tensors[s_key], self.h_128).forward(x)
        return F.linear(x, self.tensors[w_key].to(x.dtype))

    @torch.no_grad()
    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[int]:
        tokens = list(input_ids)
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * self.num_layers
        emb_table = self.tensors["token_embd.weight"]

        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = torch.tensor([tokens], device=self.device, dtype=torch.long)
                pos_ids = torch.arange(0, len(tokens), device=self.device, dtype=torch.long)
            else:
                cur_ids = torch.tensor([[tokens[-1]]], device=self.device, dtype=torch.long)
                pos_ids = torch.tensor([len(tokens) - 1], device=self.device, dtype=torch.long)

            h = F.embedding(cur_ids, emb_table).to(torch.float16)

            for layer_idx in range(self.num_layers):
                p = f"blk.{layer_idx}"
                norm_x = rms_norm(h, self.tensors[f"{p}.attn_norm.weight"], self.eps)

                q = self._linear(f"{p}.attn_q", norm_x)
                k = self._linear(f"{p}.attn_k", norm_x)
                v = self._linear(f"{p}.attn_v", norm_x)

                bsz, seq_len, _ = q.shape
                q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

                if f"{p}.attn_q_norm.weight" in self.tensors:
                    q = rms_norm(q, self.tensors[f"{p}.attn_q_norm.weight"], self.eps)
                if f"{p}.attn_k_norm.weight" in self.tensors:
                    k = rms_norm(k, self.tensors[f"{p}.attn_k_norm.weight"], self.eps)

                q, k = apply_rope(q, k, pos_ids, self.head_dim, self.rope_theta)

                if kv_cache[layer_idx] is not None:
                    prev_k, prev_v = kv_cache[layer_idx]  # type: ignore[misc]
                    k = torch.cat([prev_k, k], dim=2)
                    v = torch.cat([prev_v, v], dim=2)
                kv_cache[layer_idx] = (k, v)

                groups = self.num_heads // self.num_kv_heads
                k_rep = k.repeat_interleave(groups, dim=1)
                v_rep = v.repeat_interleave(groups, dim=1)

                attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=(seq_len > 1))
                attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
                h = h + self._linear(f"{p}.attn_o", attn_out)

                norm_ffn = rms_norm(h, self.tensors[f"{p}.ffn_norm.weight"], self.eps)
                gate = F.silu(self._linear(f"{p}.ffn_gate", norm_ffn))
                up = self._linear(f"{p}.ffn_up", norm_ffn)
                h = h + self._linear(f"{p}.ffn_down", gate * up)

            h = rms_norm(h[:, -1:, :], self.tensors["output_norm.weight"], self.eps)
            head_weight = self.tensors.get("output.weight", emb_table)
            logits = F.linear(h, head_weight.to(h.dtype))
            next_id = int(torch.argmax(logits[0, 0]).item())
            tokens.append(next_id)

        return tokens[len(input_ids):]


def load_tokenizer(model_dir: Path, fallback_repo: str) -> Any:
    from transformers import AutoTokenizer

    tok_file = model_dir / "tokenizer.json"
    if tok_file.exists():
        data = json.loads(tok_file.read_text(encoding="utf-8"))
        if data.get("model", {}).get("merges"):
            return AutoTokenizer.from_pretrained(str(model_dir))
    print(f"Tokenizer minimal sans merges détecté -> chargement depuis {fallback_repo}")
    return AutoTokenizer.from_pretrained(fallback_repo)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test d'inférence réel sur BtbAgent/bonsai-27b-aibox")
    parser.add_argument("--repo", default="BtbAgent/bonsai-27b-aibox", help="Dépôt Hugging Face")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-32B", help="Tokenizer complet de secours")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()

    local_dir = Path(snapshot_download(repo_id=args.repo))
    print("config.json :", json.loads((local_dir / "config.json").read_text(encoding="utf-8")))
    print("quant_config.json :", json.loads((local_dir / "quant_config.json").read_text(encoding="utf-8")))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BonsaiAiboxModel(local_dir, device=device, max_layers=args.max_layers)
    if args.inspect_only:
        return

    tokenizer = load_tokenizer(local_dir, args.tokenizer)
    prompts = [
        "Quelle est la capitale de la France ? Réponds en une phrase :",
        "Explain in two short sentences what a GPU is:",
    ]

    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        t0 = time.perf_counter()
        out_ids = model.generate(ids, max_new_tokens=args.max_new_tokens)
        dt = time.perf_counter() - t0
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print("\n" + "=" * 60)
        print(f"PROMPT : {prompt}")
        print(f"SORTIE ({len(out_ids)} tok en {dt:.2f}s = {len(out_ids)/dt:.2f} tok/s) :")
        print(text)


if __name__ == "__main__":
    main()
