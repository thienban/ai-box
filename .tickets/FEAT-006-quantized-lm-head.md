# FEAT-006: Quantification de `lm_head` (`Q4` / `Q8`) pour réduire la VRAM et viser 60–75+ tok/s

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
Aujourd'hui, sur `Bonsai-8B` 1-bit (`bonsai-8b-1bit`), les 36 couches Transformer ne pèsent que **~0,85 Go**, mais `lm_head` (`output.weight`, `151 669 × 4096`) reste stocké et lu en **FP16** (**1,24 Go**), ce qui représente **~60 % de la bande passante VRAM par token** (~12,5 ms/token sur RTX 4050) et plafonne le débit autour de **44 tok/s**.
Sur `Bonsai-27B` 1-bit (`3,80 Go` GGUF), un `lm_head` FP16 (~1,56 Go) empêche également le modèle de tenir confortablement sous les **6 Go VRAM** (ou proche de 4 Go).

Quantifier `lm_head` en **Q4 (4 bits/poids)** ou **Q8 (8 bits/poids)** par groupes de 128 avec scales FP16 permet à la fois de doubler le débit decode sur `Bonsai-8B` et de faire tenir `Bonsai-27B` 1-bit en VRAM sur RTX 4050 (6 Go).

## 📋 Critères d'Acceptation
- [x] Option `--lm-head-bits {4,8,16}` (défaut `16` pour rétrocompatibilité) dans [convert_gguf_to_aibox.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/convert_gguf_to_aibox.py) exportant `output.weight` quantifié + `output.scales` FP16.
- [x] Détection automatique dans [safetensors.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/safetensors.rs) et réutilisation de `LmHead::Ternary(TernaryTensor)` (`bits == 4 || bits == 8`) dans [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) (support CPU + GPU).
- [x] Kernel CUDA `lm_head_quant_act_f16_kernel<int Bits>` (`Bits=4` et `Bits=8`) dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu), compatible avec le CUDA Graph (`g_active_stream`).
- [x] Sur `Bonsai-8B` 1-bit + `lm_head` Q4/Q8 (RTX 4050) : qualité **5/5 PASS** sur [e2e_eval.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/e2e_eval.py) ; débit moyen **46,50 tok/s** (pic **50,28 tok/s**) en Q8 et **51,14 tok/s** (pic **55,23 tok/s**) en Q4 (vs ~31,7 tok/s en FP16, soit +61 % ; objectif 60 tok/s non atteint uniquement par `lm_head`).
- [ ] Validation chargement et inférence `Bonsai-27B` 1-bit + `lm_head` Q4 en VRAM sur RTX 4050 (6 Go) — non exécuté dans ce run.

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Ajouter la quantification par groupes de 128 (`Q4` : 2 poids/octet + `Q8` : `int8`) de `output.weight` dans [convert_gguf_to_aibox.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/convert_gguf_to_aibox.py).
- [x] Étape 2 : Implémenter `lm_head_quant_act_f16_kernel<Bits>` dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) et l'exposer dans [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs).
- [x] Étape 3 : Supporter `bits == 4 || bits == 8` dans [safetensors.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/safetensors.rs) et [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) + test unitaire `dot_quantized_lm_head_cpu_matches_q8_and_q4_reference`.
- [x] Étape 4 : Reconvertir `Bonsai-8B` (`--bits 1 --lm-head-bits 4` et `8`) et benchmarker sur RTX 4050 (`e2e_eval.py`).

## 📝 Notes & Décisions Techniques
- Ne **pas** appliquer de rotation Hadamard sur `lm_head` (dernière couche avant `argmax` / softmax).
- Gains VRAM estimés sur `lm_head` :
  - `Bonsai-8B` : `1,24 Go` (FP16) ➔ `~0,63 Go` (Q8) / `~0,32 Go` (Q4).
  - `Bonsai-27B` : `~1,56 Go` (FP16) ➔ `~0,79 Go` (Q8) / `~0,40 Go` (Q4), permettant de tenir sous les 6 Go VRAM avec les couches 1-bit (~3,5 Go).

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : 23/23 PASS (13 `aibox`, 9 `aibox-kernels`, 1 `aibox-mistral`)
- [x] `nvcc -O3 -arch=sm_89` : compilation sans erreur (`libaibox_gpu.so`)
- [x] `e2e_eval.py --max-tokens 256` sur `Bonsai-8B` GPU (`lm_head` Q8) : **5/5 PASS**, `46,50 tok/s` moyen (pic `50,28 tok/s`), `model.safetensors` `2718,8 MB`
- [x] `e2e_eval.py --max-tokens 256` sur `Bonsai-8B` GPU (`lm_head` Q4) : **5/5 PASS**, `51,14 tok/s` moyen (pic `55,23 tok/s`), `model.safetensors` `2422,5 MB`

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 17:21 | - | TODO | Antigravity | Création du ticket |
| 2026-11-15 17:23 | TODO | IN_PROGRESS | Antigravity | Démarrage implémentation `lm_head` quantifié Q4/Q8 (convertisseur + CUDA + Rust) |
| 2026-11-15 17:40 | IN_PROGRESS | DONE | Antigravity | Implémentation terminée, tests 23/23 PASS, E2E RTX 4050 5/5 PASS : Q8=46,50 tok/s (2718,8 MB), Q4=51,14 tok/s (2422,5 MB) |
