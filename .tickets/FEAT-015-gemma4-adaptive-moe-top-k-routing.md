# FEAT-015: Routage MoE Adaptatif (`AIBOX_MOE_TOP_K` & Top-P) sur Gemma 4

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-16
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-014](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-014-gemma4-mmap-async-prefetch-and-mixed-precision.md)

---

## 🎯 Objectif & Contexte
Sur Gemma-4-26B-A4B (128 experts/couche, `Top-8` par défaut), l'analyse des poids du routeur montre que les 4–6 premiers experts concentrent l'essentiel de la masse softmax, tandis que `Top-8` active ~`2 146` experts distincts sur 75 tokens (`240` appels/token).

Réduire dynamiquement le nombre d'experts actifs par couche (`AIBOX_MOE_TOP_K` ou seuil cumulatif *Top-P*) permet de diminuer les *disk misses* et de multiplier par **2×+** le débit sur GPU 6 Go.

---

## 📋 Critères d'Acceptation
- [x] **Variable `AIBOX_MOE_TOP_K`** dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs) :
  - Surcharge `1..=num_experts_per_tok` via `OnceLock`, sans coût sur le chemin chaud.
  - Renormalisation automatique des poids dans `route_top_k`.
- [ ] **Routage adaptatif `AIBOX_MOE_TOP_P`** (optionnel) :
  - Sélection par masse cumulée (ex. `0.85`, borné entre `min_k = 2` et `max_k = 8`) pour n'activer 6–8 experts que sur les tokens ambigus et 2–4 sur les tokens certains.
- [ ] **Exposition CLI** : option `--moe-top-k` sur `aibox run` / `aibox serve`.

---

## 📊 Benchmarks Mesurés (`gemma-4-26b-q4-moe2`, `Photosynthesis --max-tokens 64 --gpu`)
| Configuration | `Prefill (11 tok)` | `Decode (64 tok)` | `disk_misses` | `vram_hits` |
|---|---|---|---|---|
| **Top-8** (défaut) | `46–207 s` | `0.38–1.25 tok/s` | `2 146` | `15 614 (87,9 %)` |
| **`AIBOX_MOE_TOP_K=6`** | **`15.27 s`** (`15270.0 ms`) | **`2.54 tok/s`** (`25167.1 ms`) | `1 907` | `11 413 / 13 320 (85,7 %)` |
| **`AIBOX_MOE_TOP_K=4`** | `13.11 s` (`13110.0 ms`) | `2.73 tok/s` (`23439.3 ms`) | `1 560` | `7 320 / 8 880 (82,4 %)` |

---

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : **33/33 PASS** (`0` warning)
- [x] Benchmarks GPU `AIBOX_MOE_TOP_K=4` (`2.73 tok/s`) et `AIBOX_MOE_TOP_K=6` (`2.54 tok/s`) validés.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-16 15:12 | - | IN_PROGRESS | Antigravity | Création du ticket et validation de `AIBOX_MOE_TOP_K` sur `gemma-4-26b-q4-moe2` : `Top-6` atteint `2.54 tok/s` (`15.27s` prefill, `85.7%` VRAM hits) et `Top-4` atteint `2.73 tok/s` (`13.11s` prefill). Reste optionnellement `AIBOX_MOE_TOP_P` et le flag CLI `--moe-top-k`. |
