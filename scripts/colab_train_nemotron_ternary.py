# %% [markdown]
# # Pipeline Colab Optimisé : Fine-Tuning QLoRA (Unsloth) + Hadamard GPU + Ternaire
#
# Version optimisée pour **Google Colab** (T4 15 Go, L4 24 Go ou A100) :
# - **2x à 5x plus rapide** grâce à **Unsloth** + **sequence packing** + **BF16/FP16 auto**
# - **Compression 20x+ plus rapide** : matrice de Hadamard précalculée et exécutée couche par couche sur **GPU**
#   (sans saturer la VRAM) avant rapatriement sur CPU.

# %%
# ==============================================================================
# CELLULE 1 : Installation des dépendances optimisées (Colab)
# ==============================================================================
# !pip install -q unsloth datasets trl accelerate safetensors huggingface_hub

import json
import math
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# %%
# ==============================================================================
# CELLULE 2 : Configuration
# ==============================================================================
MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"  # Ou tout modèle HF compatible
DATASET_ID = "timdettmers/openassistant-guanaco"  # Remplacer par votre dataset métier JSONL
OUTPUT_DIR = "./outputs/nemotron_sft"
MERGED_DIR = "./outputs/nemotron_merged_bf16"
HF_EXPORT_DIR = "./outputs/hf_nemotron_ternary"
HF_REPO_ID = "votre-organisation/nemotron-ternary-hadamard-2bit"
HF_PRIVATE = True  # Mettre False pour un dépôt public open-source

HADAMARD_BLOCK_SIZE = 128
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
MAX_TRAIN_STEPS = 100

# %%
# ==============================================================================
# CELLULE 3 : Étape 1 — Fine-Tuning QLoRA Rapide avec Unsloth + Packing
# ==============================================================================
def run_fast_qlora_finetuning() -> None:
    """Entraîne et fusionne le modèle avec Unsloth (2x-5x plus rapide sur Colab)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-détection BF16 (L4/A100) ou FP16 (T4)
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.0,  # 0.0 active les kernels rapides Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset(DATASET_ID, split="train[:1000]")
    use_bf16 = is_bfloat16_supported()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,  # Plus élevé grâce à la réduction VRAM d'Unsloth
        gradient_accumulation_steps=2,
        warmup_steps=10,
        max_steps=MAX_TRAIN_STEPS,
        learning_rate=2e-4,
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=10,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # Regroupe les séquences courtes -> zéro calcul perdu sur le padding
        args=training_args,
    )
    trainer.train()

    # Fusion rapide directe des poids en 16-bit pour l'étape de compression
    model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")


# Décommenter pour lancer l'entraînement + fusion rapide :
# run_fast_qlora_finetuning()

# %%
# ==============================================================================
# CELLULE 4 : Étape 2 — Matrice de Hadamard Précalculée & Quantification GPU
# ==============================================================================
def build_hadamard_matrix(block_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construit la matrice de Walsh-Hadamard orthogonale (K x K) une seule fois."""
    if block_size & (block_size - 1) != 0:
        raise ValueError(f"block_size doit être une puissance de 2, reçu: {block_size}")

    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < block_size:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)

    return (h / math.sqrt(block_size)).to(dtype)


def rotate_blocks_with_hadamard(weights: torch.Tensor, h_matrix: torch.Tensor) -> torch.Tensor:
    """Applique la rotation orthogonale par blocs via une seule matmul GPU."""
    out_features, in_features = weights.shape
    block_size = h_matrix.shape[0]
    if in_features % block_size != 0:
        raise ValueError(f"in_features ({in_features}) doit être divisible par {block_size}")

    w_blocks = weights.reshape(out_features, in_features // block_size, block_size)
    rotated = torch.matmul(w_blocks, h_matrix)
    return rotated.reshape(out_features, in_features)


def quantize_ternary_groupwise(
    weights: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantifie une matrice de poids en {-1, 0, +1} par groupes de 128 avec échelle FP16."""
    out_features, in_features = weights.shape
    w_grouped = weights.reshape(out_features, in_features // group_size, group_size).float()
    scales = w_grouped.abs().mean(dim=-1, keepdim=True).clamp(min=1e-6)
    threshold = 0.5 * scales

    ternary = torch.zeros_like(w_grouped, dtype=torch.int8)
    ternary[w_grouped > threshold] = 1
    ternary[w_grouped < -threshold] = -1

    return ternary.reshape(out_features, in_features), scales.squeeze(-1).half()


def pack_ternary_2bit(ternary_weights: torch.Tensor) -> torch.Tensor:
    """Compacte 4 poids ternaires {-1, 0, +1} par octet (00=0, 01=+1, 11=-1)."""
    flat = ternary_weights.flatten()
    if flat.numel() % 4 != 0:
        raise ValueError("Le nombre de poids doit être multiple de 4 pour le packing 2-bit.")

    codes = torch.zeros_like(flat, dtype=torch.uint8)
    codes[flat == 1] = 1
    codes[flat == -1] = 3

    codes = codes.reshape(-1, 4)
    return codes[:, 0] | (codes[:, 1] << 2) | (codes[:, 2] << 4) | (codes[:, 3] << 6)


# %%
# ==============================================================================
# CELLULE 5 : Étape 3 — Compression Streamée sur GPU (Rapide & Zéro OOM)
# ==============================================================================
@torch.no_grad()
def compress_model_hadamard_ternary_fast(model_path: str, export_dir: str) -> None:
    """Charge le modèle sur CPU, traite chaque couche sur GPU et exporte en SafeTensors."""
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_matrix = build_hadamard_matrix(HADAMARD_BLOCK_SIZE, compute_device, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    compressed_state: dict[str, torch.Tensor] = {}
    skip_keywords = ("embed", "lm_head", "norm", "router", "gate_network")

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(key in name.lower() for key in skip_keywords) or module.weight.shape[1] % HADAMARD_BLOCK_SIZE != 0:
            compressed_state[f"{name}.weight_fp16"] = module.weight.data.half().contiguous()
            continue

        # Déplace uniquement cette couche sur GPU, calcule, puis rapatrie sur CPU
        w_gpu = module.weight.data.to(compute_device, dtype=torch.float16)
        rotated_w = rotate_blocks_with_hadamard(w_gpu, h_matrix)
        ternary_w, scales = quantize_ternary_groupwise(rotated_w, HADAMARD_BLOCK_SIZE)
        packed_w = pack_ternary_2bit(ternary_w)

        compressed_state[f"{name}.packed_2bit"] = packed_w.cpu().contiguous()
        compressed_state[f"{name}.scales_fp16"] = scales.cpu().contiguous()

    os.makedirs(export_dir, exist_ok=True)
    save_file(compressed_state, os.path.join(export_dir, "model.safetensors"))
    tokenizer.save_pretrained(export_dir)
    model.config.save_pretrained(export_dir)

    quant_meta = {
        "quant_method": "ternary_hadamard",
        "values": [-1, 0, 1],
        "packing": "2bit",
        "group_size": HADAMARD_BLOCK_SIZE,
        "hadamard_block_size": HADAMARD_BLOCK_SIZE,
    }
    with open(os.path.join(export_dir, "quant_config.json"), "w", encoding="utf-8") as f:
        json.dump(quant_meta, f, indent=2)

    print(f"Modèle SafeTensors + config exporté vers : {export_dir}")


# %%
# ==============================================================================
# CELLULE 6 : Étape 4 — Upload sur Hugging Face Hub
# ==============================================================================
def upload_model_to_huggingface(export_dir: str, repo_id: str, private: bool = True) -> None:
    """Envoie le dossier complet (safetensors, tokenizer, quant_config) sur Hugging Face."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata  # type: ignore
            token = userdata.get("HF_TOKEN")
        except Exception as exc:
            raise RuntimeError("Définissez HF_TOKEN dans les Secrets Colab ou os.environ.") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload modèle compressé Ternaire + Hadamard (SafeTensors)",
    )
    print(f"Modèle publié sur : https://huggingface.co/{repo_id}")


# Exemple d'exécution :
# compress_model_hadamard_ternary_fast(MERGED_DIR, HF_EXPORT_DIR)
# upload_model_to_huggingface(HF_EXPORT_DIR, HF_REPO_ID, private=HF_PRIVATE)
