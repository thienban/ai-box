# FEAT-002: Capture CUDA Graph de la boucle decode GPU

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
Même après fusion partielle (FEAT-001), la boucle decode `forward_token_gpu` dans [engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) soumet plusieurs centaines de kernels CUDA par token via FFI Rust $\to$ C++ $\to$ driver CUDA.
Capturer le graphe d'exécution d'un step decode dans un **CUDA Graph** supprime l'overhead CPU/driver de lancement à chaque token.

## 📋 Critères d'Acceptation
- [x] Capture d'un `cudaGraph_t` / `cudaGraphExec_t` sur un stream dédié pour le graphe decode (avec pointeurs fixes dans `GpuSession` et `pos` / `emb_row` mis à jour par `cudaMemcpyAsync`).
- [x] Rejeu du graphe en 1 seul appel `cudaGraphLaunch` par token généré.
- [x] Sortie identique et `5/5 PASS` sur `e2e_eval.py`.

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Rendre `pos` lisible depuis un buffer device `g_dev_pos` côté CUDA et garder `emb_row` (8 Ko H2D async) hors graphe pour économiser 1,24 Go de VRAM.
- [x] Étape 2 : Ajouter `aibox_gpu_graph_begin_capture` / `aibox_gpu_graph_end_capture` / `aibox_gpu_graph_launch` dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) et [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs).
- [x] Étape 3 : Intégrer dans `generate_tokens_streaming_gpu` ([engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs)) et mesurer sur RTX 4050.

## 📝 Notes & Décisions Techniques
- Cause du blocage initial résolue : l'upload de `token_embeddings` (1,24 Go) sur 2-bit (4,03 Go) dépassait les 6 Go VRAM de la RTX 4050, et les kernels utilisaient le stream par défaut `0` au lieu du stream capturé.
- Résultats mesurés sur RTX 4050 (`bonsai-8b-1bit`, `--max-tokens 256`) : **5/5 PASS**, débit moyen **41.12 tok/s**, pic **44.05 tok/s** (`json_format`), TTFT **273–274 ms**.

## ✅ Vérification & Tests
- [x] Compilation / Build sans erreur (`nvcc -O3 -arch=sm_89` + `cargo test --workspace` : 22/22 PASS)
- [x] Validation `e2e_eval.py --gpu --max-tokens 256` : **5/5 PASS**

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 13:59 | - | TODO | Antigravity | Création du ticket |
| 2026-11-15 15:17 | TODO | IN_PROGRESS | Antigravity | Démarrage implémentation CUDA Graph (embed lookup GPU + dev_pos + stream capture) |
| 2026-11-15 15:50 | IN_PROGRESS | TODO | Antigravity | Revert FEAT-002 sur demande utilisateur (tests Docker trop longs) ; FEAT-001 conservé |
| 2026-11-15 15:56 | TODO | TODO | Antigravity | Flag `ERROR_TO_TEST_AND_RUN` ajouté : erreur/blocage à investiguer lors du test & run Docker RTX 4050 |
| 2026-11-15 17:05 | TODO | IN_PROGRESS | Antigravity | Reprise avec architecture corrigée (`g_active_stream`, `g_dev_pos`, `emb_row` 8 Ko H2D hors graphe) |
| 2026-11-15 17:10 | IN_PROGRESS | DONE | Antigravity | CUDA Graph validé : 22/22 `cargo test` + E2E RTX 4050 (`bonsai-8b-1bit`) **5/5 PASS**, 41.12 tok/s moy, pic 44.05 tok/s. |
