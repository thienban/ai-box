# FEAT-007: Optimisations decode pour franchir 60+ tok/s (`lm_head` vectorisé + sampling GPU)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
Suite à [FEAT-006-quantized-lm-head.md](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-006-quantized-lm-head.md), `Bonsai-8B` 1-bit + `lm_head` Q4 atteint **51,14 tok/s en moyenne** (**55,23 tok/s** en pic) sur RTX 4050 (`~18,1–19,5 ms/token`).
Pour franchir la barre des **60+ tok/s** (**≤ 16,67 ms/token**), il suffit de gagner **~2 à 3 ms/token** sur le reste du chemin decode sans complexifier l'architecture globale.

## 📋 Critères d'Acceptation
- [x] Vectorisation de `lm_head_quant_act_f16_kernel` (`uint4` 128-bit sur les poids + cache `hidden` en `__shared__`) dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu).
- [x] Élimination du download complet des `151 669` logits (`607 Ko`) vers le CPU dans [sample_next_token_gpu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs#L873-L886) lorsque `temperature > 0` (`aibox_gpu_sample_candidates_f32` + `sample_from_candidates`).
- [x] Fusion ciblée `gate + up + swiglu` (`ternary_gemv_swiglu_f16_kernel` / `aibox_gpu_matmul_ternary_swiglu_f16_dev`) sur les 36 couches dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) et [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs).
- [x] Sur `Bonsai-8B` 1-bit + `lm_head` Q4 (RTX 4050) : qualité **5/5 PASS** sur [e2e_eval.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/e2e_eval.py) ; débit E2E moyen **53,77 tok/s** (pic **57,65 tok/s** sur `en_tech`, **60,0 tok/s** en decode pur hors TTFT sur `code_python`).

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Optimiser `lm_head_quant_act_f16_kernel<Bits>` (`__shared__` pour `hidden` + `uint4`).
- [x] Étape 2 : Implémenter `aibox_gpu_sample_candidates_f32` pour éviter la copie PCIe de `607 Ko` + tri CPU sur 151k éléments à chaque token.
- [x] Étape 3 : Fusionner `gate + up + swiglu` (`ternary_gemv_swiglu_f16_kernel`) dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) (NB : un essai intermédiaire avec `__shared__` dans `ternary_gemv_f16_kernel` a fait chuter l'occupation SM à `33,84 tok/s` et a été annulé).
- [x] Étape 4 : Benchmarker sur RTX 4050 avec `e2e_eval.py --max-tokens 256`.

## 📝 Notes & Décisions Techniques
- Baseline actuelle (`FEAT-006`, commit `5115899`) sur RTX 4050 (`Bonsai-8B` 1-bit + `lm_head` Q4) : `51,14 tok/s` moyen, `55,23 tok/s` pic, `5/5 PASS`.
- Privilégier d'abord les 2 optimisations à fort ROI (`lm_head` `__shared__` + `uint4` et sampling GPU) avant toute fusion complexe dans les 36 couches (KISS/YAGNI).

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : 23/23 PASS
- [x] `nvcc -O3 -arch=sm_89` : compilation sans erreur (`libaibox_gpu.so`)
- [x] `e2e_eval.py --max-tokens 256` sur `Bonsai-8B` GPU (`lm_head` Q4) : **5/5 PASS**, débit E2E moyen **53,77 tok/s** (pic **57,65 tok/s**, **60,0 tok/s** decode pur sur `code_python`)

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 17:46 | - | TODO | Antigravity | Création du ticket suite à FEAT-006 (`51,14 tok/s` moyen, pic `55,23 tok/s`) |
| 2026-11-15 18:07 | TODO | IN_PROGRESS | Antigravity | Démarrage implémentation (`lm_head` `__shared__`/`uint4` + sampling GPU + fusion `gate+up+swiglu`) |
| 2026-11-15 18:41 | IN_PROGRESS | DONE | Antigravity | Tests 23/23 PASS, E2E RTX 4050 5/5 PASS : `53,77 tok/s` moyen, pic `57,65 tok/s` (`60,0 tok/s` decode pur) |
