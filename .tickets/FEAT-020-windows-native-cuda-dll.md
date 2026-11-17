# FEAT-020: Support CUDA Natif Windows (`aibox_gpu.dll`) sans Docker

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-17
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-019](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-019-aibox-core-unified-engine-and-gemma4-serve.md)

---

## 🎯 Objectif & Contexte
Dans [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs), le backend CUDA est actuellement compilé uniquement sous `#[cfg(target_os = "linux")]` (chargement de `libaibox_gpu.so`), ce qui impose l'usage de Docker + WSL2 sous Windows pour l'accélération GPU.
Pour distribuer `aibox` comme une véritable application Desktop Windows (façon Ollama), le binaire natif Windows doit charger `aibox_gpu.dll` directement.

---

## 📋 Critères d'Acceptation
- [x] **Portabilité `gpu.rs`** : `#[cfg(any(target_os = "linux", target_os = "windows"))]` activé avec `os_open_lib` / `os_find_sym` / `os_close_lib` (`LoadLibraryA` / `GetProcAddress` / `FreeLibrary` via `kernel32` sous Windows, `dlopen` / `dlsym` / `dlclose` sous Linux), recherche automatique de `aibox_gpu.dll` à côté de l'exécutable.
- [x] **Exports `__declspec(dllexport)`** dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) via la macro `AIBOX_EXPORT` sur les 44 fonctions `aibox_gpu_*`.
- [x] **Script de build Windows CUDA** : [build_windows_cuda.ps1](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/build_windows_cuda.ps1) (`nvcc --shared ... -o aibox_gpu.dll` + `cargo build --release`).
- [ ] **Tests GPU natifs Windows** : compilation `aibox_gpu.dll` et exécution de `gpu_q4_affine_matches_scalar_when_available` / `gpu_q2_affine_matches_scalar_when_available` sur un poste équipé du CUDA Toolkit Windows (`nvcc.exe` absent sur cette machine).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 08:54 | - | TODO | Antigravity | Création du ticket pour supprimer la dépendance Docker sur poste client Windows. |
| 2026-11-17 10:24 | TODO | IN_PROGRESS | Antigravity | Portabilité `gpu.rs` (`kernel32` `LoadLibraryA`/`GetProcAddress`/`FreeLibrary`), macro `AIBOX_EXPORT` sur les 44 exports dans `aibox_ternary_gpu.cu`, et `scripts/build_windows_cuda.ps1`. Compilation & tests Windows : 36/36 PASS (0 warning). Reste à compiler `aibox_gpu.dll` une fois CUDA Toolkit Windows installé. |
