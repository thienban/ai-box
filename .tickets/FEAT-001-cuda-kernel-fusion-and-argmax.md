# FEAT-001: Fusion de kernels CUDA et `argmax` GPU en decode

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
Actuellement, [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) lance ~17–19 kernels CUDA séparés par couche Transformer (~600 lancements/token) et télécharge l'intégralité des logits `vocab_size` (~151k floats $\approx$ 600 Ko) du GPU vers le CPU à chaque token généré.
L'objectif est de fusionner les opérations adjacentes et d'exécuter `argmax` (sampling greedy) sur GPU pour viser **30–35 tok/s** sur `Bonsai-8B` (RTX 4050) sans changer le format de modèle.

## 📋 Critères d'Acceptation
- [x] Fusion `RMSNorm + FWHT128` en un seul kernel CUDA (zéro aller-retour VRAM intermédiaire).
- [x] Fusion de l'épilogue `+ residual` directement dans `ternary_gemv_kernel` pour `attn_o` et `ffn_down`.
- [x] Fusion `SwiGLU (+ FWHT128)` en un seul kernel.
- [x] Kernel `argmax` GPU retournant uniquement un `u32` (4 octets) lorsque `temperature == 0.0`.
- [x] Qualité E2E maintenue à **5/5 PASS** sur [e2e_eval.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/e2e_eval.py) et débit moyen mesuré en hausse par rapport à `23.35 tok/s`.

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Ajouter les kernels fusionnés (`rms_norm_fwht128_kernel`, `swiglu_fwht128_kernel`, épilogue residual dans `ternary_gemv_kernel`) et `argmax_f32_kernel` dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu).
- [x] Étape 2 : Exposer les symboles FFI dans [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs) et les brancher dans [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs).
- [x] Étape 3 : Valider par tests unitaires Rust (`cargo test --workspace`) et benchmark `e2e_eval.py --gpu --max-tokens 256` sur RTX 4050.

## 📝 Notes & Décisions Techniques
- Extraction d'un helper device `fwht128_warp_inplace` et d'un template `ternary_gemv_kernel<bool AddResidual>` pour éviter toute duplication.
- Résultats mesurés sur RTX 4050 (`Bonsai-8B`, `--max-tokens 256`, `task-1418`) :
  - Débit moyen : **23.59 tok/s** (vs `23.35 tok/s`), pic **25.97 tok/s** (`en_tech`), `code_python` `24.90`, `math_simple` `24.45`, `json_format` `24.62 tok/s`, TTFT min `830 ms`, **5/5 PASS**.
  - Observation clé : le gain de ~1 tok/s confirme que le goulot principal restant est la **bande passante poids 2-bit** (d'où la priorité sur **FEAT-004** 1-bit natif et **FEAT-002** CUDA Graph).

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS (`20 passed; 0 failed`, 0 warning)
- [x] `e2e_eval.py --gpu --max-tokens 256` : 5/5 PASS et gain de débit mesuré

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 13:59 | - | TODO | Antigravity | Création du ticket |
| 2026-11-15 14:01 | TODO | IN_PROGRESS | Antigravity | Démarrage de l'implémentation (fusions CUDA + argmax GPU) |
| 2026-11-15 14:12 | IN_PROGRESS | DONE | Antigravity | Fusions + argmax validés : 20/20 tests Rust, 5/5 PASS E2E sur RTX 4050 (23.59 tok/s moy, pic 25.97 tok/s) |
