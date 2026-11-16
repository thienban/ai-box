# FEAT-011: Refactoring — Isolation de la logique par architecture modèle (`engine/` modulaire)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Refactor <!-- Options: Feature | Bug | Refactor | Docs | Chore -->
- **Date de création**: 2026-11-15
- **Assigné à**: Antigravity

---

## 🎯 Objectif & Contexte
[engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs) dépasse aujourd'hui **1 940 lignes** et mélange dans un seul fichier :
- Le cœur partagé (`ModelConfig`, `GenerationStats`, `KvCache`, `GpuSession`, boucle de génération, sampling, CUDA Graph).
- La logique spécifique **`qwen3`** (Attention GQA + SwiGLU).
- La logique spécifique **`qwen35`** (Gated DeltaNet SSM + Partial RoPE + `q_gate`).

Avant d'ajouter de nouvelles architectures comme **`gemma4`** (MoE + GeGLU + Sliding-Window + PLE), il faut isoler proprement chaque architecture sans dupliquer le serveur HTTP, le loader SafeTensors, le tokenizer ni le backend CUDA (`aibox-kernels`).

## 📋 Critères d'Acceptation
- [x] **Découpage modulaire KISS dans `app/crates/aibox-cli/src/`** (sans sur-ingénierie ni duplication de `app/`) :
  - `engine/mod.rs` (ou `engine.rs` allégé) : cœur partagé (`ModelConfig`, `KvCache`, `GpuSession`, `generate`, `generate_gpu`, `compute_logits_gpu`).
  - `engine/attention.rs` : logique GQA Attention + RoPE partiel/complet (CPU & GPU).
  - `engine/gdn.rs` : logique spécifique `qwen35` / Gated DeltaNet SSM (CPU `run_gdn_sequence_cpu` + `forward_gdn_gpu`).
  - Point d'extension clair pour `engine/gemma4.rs` (MoE + GeGLU) sans toucher au cœur du serveur ni au sampling.
- [x] **Zéro duplication** : conserver les helpers communs (`forward_ffn_gpu`, `compute_logits_gpu`, `GpuSession`).
- [x] **Non-régression stricte** :
  - `cargo test --workspace --manifest-path app/Cargo.toml` : **24/24 PASS**, **0 warning**.
  - Build release CUDA (`nvcc -O3 -arch=sm_89`) : **0 warning**.
  - Smoke tests `bonsai-8b-1bit-q4head` & `bonsai-27b-1bit-q4head` inchangés.

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Convertir `src/engine.rs` en module `src/engine/mod.rs` (0 changement de API publique).
- [x] Étape 2 : Extraire `src/engine/gdn.rs` (poids `GatedDeltaNetWeights`, `run_gdn_sequence_cpu`, `forward_gdn_block`, `forward_gdn_gpu`).
- [x] Étape 3 : Extraire `src/engine/attention.rs` (`AttentionBlockWeights`, `forward_attention_block`, `forward_full_attention_gpu`, RoPE).
- [x] Étape 4 : Vérifier `cargo test --workspace` (Windows) + build CUDA & smoke tests GPU.

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : 24/24 PASS, 0 warning
- [x] `cargo build --release` (Linux CUDA) : 0 warning
- [x] Non-régression `bonsai-8b-1bit-q4head` & `bonsai-27b-1bit-q4head`

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-15 23:42 | - | TODO | Antigravity | Création du ticket de refactoring pour isoler chaque architecture modèle avant l'ajout de `gemma4` (MoE 2-bit) |
| 2026-11-15 23:55 | IN_PROGRESS | DONE | Antigravity | Refactoring `engine/` terminé (`mod.rs`, `attention.rs`, `gdn.rs`). Tests Windows 24/24 PASS (0 warning), build CUDA release 0 warning, GPU smoke tests OK (`bonsai-8b` 71.41 tok/s, `bonsai-27b` 17.76 tok/s) |
