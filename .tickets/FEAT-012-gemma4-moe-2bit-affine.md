# FEAT-012: Support `gemma4` (MoE 26B-A4B) & Quantification 2-bit Affine Asynchrone

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-011](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-011-refactor-engine-model-isolation.md)

---

## 🎯 Objectif & Contexte
Permettre à `aibox` d'exécuter des modèles **Mixture-of-Experts (MoE)** quantifiés en **2-bit affine** (~7–8 Go pour `google/gemma-4-26B-A4B`, avec seulement ~4B paramètres / ~1,2 Go actifs par token), sans cloner le dossier `app/`.

## 📋 Critères d'Acceptation
- [x] **Kernel CUDA 2-bit Affine (`aibox-kernels`)** :
  - Support des poids `uint2` + `scale_fp16` + `zero_fp16` par groupe (64 ou 128) dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu), avec pipeline asynchrone (inspiré de [arXiv:2311.16442](https://arxiv.org/abs/2311.16442)).
- [x] **Module isolé `engine/gemma4.rs`** :
  - Détection `"architecture": "gemma4"` dans `config.json`.
  - **Routeur MoE Top-K** (`128` experts, `Top-8` actifs + `1` shared expert) : ne calcule que les experts sélectionnés.
  - Primitives Gemma 4 : `GeGLU` (GELU), `Sliding-Window Attention (1024)` + `Global Attention`, et Per-Layer Embeddings (`PLE`).
- [x] **Tiering Serveur 6 Go vs 8–12 Go+** :
  - **8–12 Go+ VRAM** : tout le modèle (~7–8 Go) réside en VRAM (débit niveau 4B avec qualité 26B).
  - **6 Go VRAM** : Attention + Shared Experts + cache LRU d'experts chauds en VRAM, experts froids en Pinned RAM (`cudaHostAlloc`).
- [x] **Convertisseur** : extension de [convert_gguf_to_aibox.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/convert_gguf_to_aibox.py) pour exporter les tenseurs `gemma4` et `q2_affine`.

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Finaliser [FEAT-011](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-011-refactor-engine-model-isolation.md) (isolation `engine/mod.rs`).
- [x] Étape 2 : Ajouter `matmul_q2_affine_f16_dev` dans `aibox_ternary_gpu.cu` et `gpu.rs` + tests unitaires CPU/GPU.
- [x] Étape 3 : Implémenter `engine/gemma4.rs` (routeur Top-K + GeGLU + SWA).
- [ ] Étape 4 : Tester et benchmarker `google/gemma-4-26B-A4B` en 2-bit (VRAM, `tok/s`, Q&A).

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS
- [x] `nvcc -O3 -arch=sm_89` : compilation sans warning
- [ ] Test E2E `Gemma-4-26B-A4B` (2-bit) sur serveur local `/v1/chat/completions`

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 23:42 | - | TODO | Antigravity | Création du ticket pour le support isolé de `gemma4` (MoE 26B-A4B) et kernel 2-bit affine asynchrone |
| 2026-11-16 00:04 | TODO | IN_PROGRESS | Antigravity | Étapes 1-3 terminées : kernel CUDA `q2_affine` + pinned async (`aibox_ternary_gpu.cu`/`gpu.rs`), module `engine/gemma4.rs` (MoE Top-K, GeGLU, SWA, PLE, `ExpertLruCache`), `convert_gguf_to_aibox.py` (`--quant-format q2_affine`). Windows 30/30 PASS (0 warning), `nvcc -O3 -arch=sm_89` 0 warning, GPU test `gpu_q2_affine_matches_scalar_when_available` PASS. Reste Étape 4 (modèle réel `gemma-4-26B-A4B`). |
| 2026-11-16 10:02 | IN_PROGRESS | IN_PROGRESS | Antigravity | Ajout `Q2_GROUP_SIZE=64`, `EmbeddingTable::Quantized`, tokenizer SentencePiece (`▁` + `<|turn>`), `Gemma4GpuScratch`, `rope_freqs.weight`, et loader MoE empilé (`ffn_*_exps`). Smoke tests GPU validés : `gemma-4-12b-q2` 48 couches (`3.34 GiB`, `9.54 tok/s`), `gemma-4-26b-q2` 2 couches MoE (`81.65 tok/s`). Reste à libérer ~8 Go sur `C:` pour exporter les 30 couches complètes de `gemma-4-26b-q2`. |
| 2026-11-16 11:03 | IN_PROGRESS | IN_PROGRESS | Antigravity | Ajout kernel `q4_affine` (`Q4AffineTensor`, `q4_affine_gemv_f16_kernel`, `matmul_q4_affine_f16_streaming_dev`), staging RAM pinnée (`cudaHostAlloc` + `cudaMemcpyAsync`), normes parallèles MoE (`pre_ffw_norm_2`, `post_ffw_norm_1`, `post_ffw_norm_2`), et support multi-shards SafeTensors (`open_dir` + `--shard-mb`). Windows 33/33 PASS (0 warning), GPU test `gpu_q4_affine_matches_scalar_when_available` PASS, `gemma-4-26b-q4` (4 couches MoE streaming) validé à `19.21 tok/s`. |
| 2026-11-16 11:42 | IN_PROGRESS | IN_PROGRESS | Antigravity | Remplacement du cache RAM par `ExpertRamLru(1024)` + ajout `v_norm` unitaire + routage MoE `rms_norm_unweighted * router_scale` dans `gemma4.rs`. Windows 33/33 PASS (0 warning), Docker release CUDA 0 warning. Test GPU 30 couches `/models/gemma-4-26b-q4` (`Hello --max-tokens 16 --gpu`) : `Prefill 10 tok: 34755.7 ms \| Decode 16 tok: 21287.4 ms (0.75 tok/s)` (x4 vs `0.19 tok/s`), sortie `AI!booknet_the_cut_the_the_the_the` (cohérence encore à diagnostiquer). |
| 2026-11-16 12:01 | IN_PROGRESS | IN_PROGRESS | Antigravity | Diagnostic & résolution de la cohérence sur `gemma-4-26b-q4` (30 couches) : `attention_scale = 1.0` (au lieu de `1/sqrt(head_dim)`) et `scalar_root_size = 1/sqrt(hidden_size)` dans le routeur MoE (`gemma4.rs`). Top logit passé de `0.93` à `42.72`, sortie cohérente `<\|channel>thought\n<channel\|>Hello! How can I help you today?` (`Decode 14 tok`, `0.66 tok/s`). Windows 33/33 PASS (0 warning). |
| 2026-11-16 12:12 | IN_PROGRESS | IN_PROGRESS | Antigravity | Optimisations vitesse MoE (`gemma4.rs` + `gpu.rs`) : attention + shared experts résidents en VRAM (`forward_resident_with_scratch`), `ExpertRamLru(1280)` zéro-copie via `Arc`, et `VramExpertLru(900)` (~2,85 Go VRAM) avec réutilisation des `GpuBuffer`s. Windows 33/33 PASS (0 warning), Docker release CUDA 0 warning. Test GPU `/models/gemma-4-26b-q4` (`Hello --max-tokens 16 --gpu`) : `Prefill 10 tok: 38000.8 ms \| Decode 14 tok: 13865.2 ms (1.01 tok/s)`. |
