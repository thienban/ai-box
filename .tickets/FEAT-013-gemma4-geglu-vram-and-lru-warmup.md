# FEAT-013: Optimisation Gemma 4 MoE — Kernel `geglu_inplace_f16` & Chauffe du Cache LRU

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-16
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-012](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-012-gemma4-moe-2bit-affine.md)

---

## 🎯 Objectif & Contexte
Sur `gemma-4-26b-q4` (30 couches, 8 experts actifs/couche = 240 experts/token), la boucle de décodage atteint `1.01 tok/s` (`~990 ms/token`) sur RTX 4050 Laptop (6 Go VRAM) après l'ajout de `VramExpertLru(900)` et la résidence VRAM de l'attention/shared experts.

Deux goulots d'étranglement principaux restent à lever :
1. **~720 synchronisations GPU$\rightarrow$CPU par token** : dans `forward_vram_geglu_expert` ([gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs)), chaque expert actif effectue `download_as_f32()` après `gate`, `up` et `down` (soit `240 × 3 = 720` copies D2H synchrones + re-upload `upload_f32_as_f16` pour `down`), uniquement pour calculer `gelu_tanh(g) * u` et l'accumulation pondérée côté CPU.
2. **Cache misses disque à froid sur les premiers tokens** : pendant le *prefill* et les premiers tokens de *decode*, la chauffe de `LruMap` (`expert_ram_cache(1280)` et `vram_experts(900)`) déclenche des lectures disque séquentielles via `read_tensor_slice()` sur les shards SafeTensors.

---

## 📋 Critères d'Acceptation
- [x] **Kernel CUDA `geglu_inplace_f16_kernel`** ([aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu), [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs)) :
  - Calcule `gate[i] = gelu_tanh(gate[i]) * up[i]` directement en VRAM sur des buffers `GpuBuffer<u16>` (`__half`), sans `download_as_f32()` ni `upload_f32_as_f16()` intermédiaires.
  - Optionnel : accumulation pondérée `moe_acc_dev[i] += weight * down[i]` en VRAM (réduisant les 240 `download_as_f32` de `down` à **1 seul** `download_as_f32` par couche MoE).
- [x] **Intégration `gemma4.rs`** :
  - Ajout de `gate_dev` / `up_dev` / `moe_acc_dev` dans `Gemma4GpuScratch`.
  - Suppression des 720 synchronisations D2H/H2D par token dans `forward_vram_geglu_expert`.
- [ ] **Chauffe / Préchargement du cache LRU (`LruMap`)** :
  - Pré-remplissage opportuniste ou asynchrone des experts les plus probables (ou lecture groupée par couche / *prefetch* pendant le chargement du modèle jusqu'à remplir le budget RAM/VRAM disponible) afin de réduire les *cache misses* disque sur les tout premiers tokens.

---

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Ajouter `geglu_inplace_f16_kernel` (et `axpy_f16_kernel` pour l'accumulation pondérée) dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) + wrappers `GpuBackend` dans [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs) + tests unitaires GPU.
- [x] Étape 2 : Brancher le pipeline 100 % VRAM dans `Gemma4GpuScratch` / `forward_vram_geglu_expert` ([gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs)).
- [ ] Étape 3 : Implémenter la chauffe du cache LRU (préchargement initial des premiers experts dans `expert_ram_cache` / `vram_experts` ou prefetch asynchrone).
- [ ] Étape 4 : Benchmarker `gemma-4-26b-q4` (`Hello --max-tokens 16 --gpu`) et mesurer le gain en `tok/s` vs `1.01 tok/s`.

---

## ✅ Vérification & Tests
- [ ] `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS (`0` warning)
- [ ] `nvcc -O3 -arch=sm_89` : compilation sans warning
- [ ] Benchmark GPU `gemma-4-26b-q4` : sortie identique (`<|channel>thought\n<channel|>Hello! How can I help you today?`) avec gain mesuré sur `Prefill` et `Decode tok/s`.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-16 12:30 | - | TODO | Antigravity | Création du ticket pour éliminer les ~720 syncs GPU->CPU/token (`geglu_inplace_f16`) et accélérer la chauffe du cache LRU |
| 2026-11-16 12:41 | TODO | IN_PROGRESS | Antigravity | Étapes 1-2 terminées : `geglu_inplace_f16_kernel` + `weighted_acc_f16_to_f32_kernel` (`aibox_ternary_gpu.cu` / `gpu.rs`) + `forward_moe_lazy_vram` (`gemma4.rs`, 30 syncs/token au lieu de 720). Windows 33/33 PASS (0 warning), `nvcc` 0 warning. Benchmark GPU `gemma-4-26b-q4` (`Hello --max-tokens 16`) : sortie identique, `Prefill 10 tok: 41499.2 ms \| Decode 14 tok: 16529.3 ms (0.85 tok/s)` — prouve que le goulot dominant sur les 14 premiers tokens reste les lectures disque à froid (Étape 3 : warmup LRU). |
| 2026-11-16 12:54 | IN_PROGRESS | IN_PROGRESS | Antigravity | Ajout compteurs `AIBOX_MOE_STATS=1` + élargissement des caches LRU à `vram_experts(1050)` (~3,3 Go VRAM) et `expert_ram_cache(1500)` (~4,7 Go RAM). Benchmark GPU `Photosynthesis --max-tokens 64` : `vram_hits=14725 (82.9%)`, `ram_hits=791 (4.5%)`, `disk_misses=2244`, `Prefill 11 tok: 46838.3 ms \| Decode 64 tok: 51219.7 ms (1.25 tok/s)`. Windows 33/33 PASS (0 warning). |
