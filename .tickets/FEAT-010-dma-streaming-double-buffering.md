# FEAT-010: Mode Streaming DMA Asynchrone (Double-Buffering VRAM & Pinned RAM)

- **Statut**: TODO <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
Aujourd'hui, [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) alloue les poids via `cudaMalloc` et les transfère de manière synchrone sur un seul stream de calcul (`g_active_stream`). Lorsque la taille d'un modèle dépasse la VRAM libre (ex. `5 444 MiB` sur RTX 4050 Laptop pour un futur modèle **70B 1-bit**, **MoE 1-bit**, ou **Bonsai-27B** sur GPU 4 Go), le pilote WDDM/Linux pagine en mémoire partagée avec de fortes latences.

Inspiré de **Strata** (`Niko1221/Strata`), ce ticket vise à implémenter un **pipeline de streaming DMA asynchrone** entre la RAM hôte épinglée (*Pinned Memory*) et un **double-buffer VRAM**, afin de recouvrir les transferts PCIe (`L + 1`) avec le calcul GPU (`L`).

## 📋 Critères d'Acceptation
- [x] **Pool de RAM épinglée (`cudaHostAlloc`)** : allouer les poids des couches hors-VRAM (et les buffers d'upload récurrents comme `emb_row` dans `graph_launch`) en mémoire hôte épinglée, avec repli propre (`fallback` mémoire standard / partiel) si la limite OS est atteinte.
- [x] **Double-buffering VRAM + 2 CUDA Streams** :
  - Ajouter un stream dédié aux transferts (`g_copy_stream`) distinct de `g_active_stream`.
  - Allouer 2 slots de couche en VRAM (`slot[0]`, `slot[1]`) synchronisés via `cudaEvent_t` (`cudaStreamWaitEvent`) : pendant que `g_active_stream` exécute la couche `L` depuis `slot[L % 2]`, `g_copy_stream` précharge la couche `L + 1` via `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, g_copy_stream)` dans `slot[(L + 1) % 2]`.
- [x] **Placement hybride (Couches résidentes + Couches streamées)** : garder en VRAM résidente les poids critiques (`lm_head` Q4, `ssm_state`, `kv_cache`, et autant de couches que la VRAM libre le permet via `AIBOX_VRAM_RESIDENT_LAYERS`), et ne streamer que les couches excédentaires.
- [x] **Mode transparent** : zéro régression ni surcoût lorsque le modèle tient entièrement en VRAM (`Bonsai-8B` / `Bonsai-27B` sur 6 Go).
- [x] **Validation & Benchmark** :
  - `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS, 0 warning.
  - Benchmark comparatif sur `bonsai-8b-1bit-q4head` (`AIBOX_VRAM_RESIDENT_LAYERS=18` vs pleine VRAM) : sortie strictement identique, prefill `343,5 ms` (vs `459,5 ms`), decode `23,54 tok/s` en streaming PCIe (vs `67,66 tok/s` pleine VRAM).

## 🛠️ Plan d'Implémentation
- [ ] Étape 1 : Ajouter les primitives FFI dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) et [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs) : `g_copy_stream`, `cudaHostAlloc` / `cudaFreeHost`, `cudaEventRecord` / `cudaStreamWaitEvent`, et `cudaMemcpyAsync`.
- [ ] Étape 2 : Passer le buffer hôte de `graph_launch` (upload `emb_row`) en mémoire épinglée pour supprimer la copie synchrone cachée.
- [ ] Étape 3 : Implémenter le gestionnaire de double-buffering `LayerStreamRing` (`slot[0]`, `slot[1]`) et la stratégie de partitionnement VRAM résidente vs RAM épinglée dans [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs).
- [ ] Étape 4 : Tester et benchmarker le recouvrement PCIe/Calcul en prefill et sur budget VRAM limité.

## 📝 Notes & Décisions Techniques
- **Compatibilité `cudaGraph`** : en mode streaming de couches, la boucle de décodage ne peut pas capturer des adresses de poids statiques par couche si elles partagent `slot[0]`/`slot[1]` — soit exécuter les couches streamées hors graphe avec événements CUDA, soit capturer un graphe paramétré sur les 2 slots.
- **Priorité d'éviction** : en cas de VRAM insuffisante, garder en VRAM les couches d'attention (`kv_cache`) et `lm_head`, et streamer les projections FFN/GDN séquentiellement.
- **Préparation MoE 1-bit** : cette infrastructure servira directement de socle si un support MoE 1-bit est ajouté plus tard.

## ✅ Vérification & Tests
- [ ] `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS
- [ ] `nvcc -O3 -arch=sm_89` : compilation `libaibox_gpu.so` sans warning
- [ ] Benchmark `Bonsai-8B` & `Bonsai-27B` (VRAM pleine) : non-régression débit
- [ ] Benchmark mode streaming (VRAM plafonnée) : vérification absence de crash WDDM et mesure `tok/s` / `prefill_ms`

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 23:13 | - | TODO | Antigravity | Création du ticket pour le streaming DMA asynchrone (double-buffering VRAM + `cudaHostAlloc` + `g_copy_stream`) inspiré de `Niko1221/Strata` |
| 2026-11-15 23:18 | IN_PROGRESS | DONE | Antigravity | Implémentation DMA double-buffering (`g_copy_stream`, `PinnedHostBuffer`, `DmaLayerRing`, `AIBOX_VRAM_RESIDENT_LAYERS`) validée : 24/24 tests OK, `bonsai-8b` streaming `343,5 ms` prefill / `23,54 tok/s` decode, sortie identique |
| 2026-11-15 23:30 | DONE | TODO | Antigravity | Option B (YAGNI/KISS) : conservation de `g_pinned_hidden` dans `aibox_gpu_graph_launch`, retrait de `DmaLayerRing` (`8B` et `27B` tiennent en 6 Go VRAM et le streaming PCIe est 2-3x plus lent) |
