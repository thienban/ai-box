# FEAT-008: Support de l'architecture hybride `qwen35` (Gated DeltaNet / SSM) pour `Bonsai-27B` 1-bit sur 6 Go VRAM

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
`Bonsai-8B` (`architecture: "qwen3"`) tourne déjà à **54,46 tok/s** (pic **59,16 tok/s**) sur RTX 4050 (6 Go VRAM) avec `lm_head` Q4 (`2,42 Go`).
La conversion de `Bonsai-27B` (`bonsai-27b-q1_0`, `--bits 1 --lm-head-bits 4`) a montré que `Bonsai-27B` utilise l'architecture hybride **`qwen35`** (64 couches, `hidden_size = 5120`, `vocab_size = 248 320`) :
- **16 couches Attention standard** (`blk.3, 7, ..., 63`) : `attn_q`, `attn_k`, `attn_v`, `attn_o`, `attn_q_norm`, `attn_k_norm`, `ffn_*` + `post_attention_norm`.
- **48 couches Gated DeltaNet / SSM** (`blk.0, 1, 2, 4, ...`) : `attn_qkv`, `attn_gate`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`, `ffn_*` + `post_attention_norm`.

Ajouter le support de `LayerBlock::GatedDeltaNet` dans le moteur `aibox` permet d'exécuter **à la fois `Bonsai-8B` (`qwen3`) et `Bonsai-27B` (`qwen35`)** dans le même binaire Rust + CUDA, avec une empreinte VRAM estimée à **~4,1 Go** (compatible RTX 4050 6 Go).

## 📋 Critères d'Acceptation
- [x] Détection de `architecture == "qwen35"` dans [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) via un enum `LayerBlock::Attention` / `LayerBlock::GatedDeltaNet` (rétrocompatibilité totale avec `qwen3` / `Bonsai-8B`).
- [x] Chargement des tenseurs `qwen35` depuis SafeTensors (`post_attention_norm`, `attn_qkv`, `attn_gate`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm`, `ssm_out`) et état récurrent par couche (`conv_state` + `ssm_state` au lieu d'un KV cache complet sur 48 couches).
- [x] Implémentation CPU (référence + test unitaire `qwen35_hybrid_batched_prefill_matches_sequential`) et GPU CUDA (`aibox_ternary_gpu.cu`) du pas **Gated DeltaNet / conv1d + récurrence SSM**.
- [x] Vérification que `Bonsai-8B` (`qwen3`) passe toujours **24/24 tests workspace** et répond correctement sur RTX 4050 (`66.69–72.10 tok/s`).
- [x] Chargement et inférence de `bonsai-27b-1bit-q4head` en VRAM sur RTX 4050 (`5114 MiB / 6141 MiB`), à **16.36–19.16 tok/s** (avec résolution math `27 * 14 + 9 = 387` ; note : `repeat_penalty` non encore implémenté dans `SamplingParams`, ce qui peut provoquer des boucles dans `<think>` sur certains prompts).

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Vérifier les métadonnées exactes du GGUF `bonsai-27b-q1_0` (`ssm_*` dimensions, `head_dim`, `full_attention_interval`, `rope_dim`, `conv_kernel`, `state_size`) et les exporter dans `config.json` via [convert_gguf_to_aibox.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/convert_gguf_to_aibox.py).
- [x] Étape 2 : Introduire `enum LayerBlock` (`Attention` | `GatedDeltaNet`) + `conv_states`/`ssm_states` dans [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) sans dupliquer la logique FFN commune.
- [x] Étape 3 : Implémenter l'opérateur Gated DeltaNet sur CPU (avec test unitaire) et en kernel CUDA dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu).
- [x] Étape 4 : Tester `bonsai-27b-1bit-q4head` sur RTX 4050 (VRAM `5114 MiB`, `16.36–19.16 tok/s`) et vérifier la non-régression de `Bonsai-8B`.

## 📝 Notes & Décisions Techniques
- `bonsai-27b-1bit-q4head/model.safetensors` pèse `6362,9 MB` sur disque car `token_embd.weight` (`248 320 × 5120`) est en FP16 (`~2,48 Go`), mais reste en RAM CPU (non uploadé en VRAM).
- Sur GPU, les couches 1-bit (`~3,2 Go`) + `lm_head` Q4 (`~0,64 Go`) + états SSM/KV (`~0,2 Go`) représentent **~4,1 Go VRAM**, ce qui passe sous la limite des 6 Go de la RTX 4050.
- Option future (si besoin d'alléger encore le disque) : quantifier `token_embd.weight` en Q8/Q4 dans le convertisseur.

## ✅ Vérification & Tests
- [ ] `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS
- [ ] `nvcc -O3 -arch=sm_89` : compilation sans erreur (`libaibox_gpu.so`)
- [ ] `Bonsai-8B` (`bonsai-8b-1bit-q4head`) : non-régression **5/5 PASS** (`~54+ tok/s`)
- [ ] `Bonsai-27B` (`bonsai-27b-1bit-q4head`) : chargement VRAM OK sur RTX 4050 + réponses Q&A correctes

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 19:43 | - | TODO | Antigravity | Création du ticket suite à la conversion de `bonsai-27b-q1_0` (`qwen35` hybride SSM + Attention) |
