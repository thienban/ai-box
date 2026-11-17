# FEAT-023: Accélération du Decode Gemma 4 (hidden VRAM, Top-P MoE dynamique, scratch prefill)

- **Statut**: TODO <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-17
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-022](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-022-gemma4-kv-cache-2048-mom-pipeline.md)

---

## 🎯 Objectif & Contexte
Débit actuel mesuré sur RTX 4050 6 Go (`gemma-4-26b-q4-moe2`) : `6.6 – 11.0 tok/s` en decode, et `~95–105 s` par appel complet dans le pipeline financier (prefill inclus). Trois goulots restent dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs).

---

## 📋 Critères d'Acceptation

### 1. `hidden` résident en VRAM sur les 30 couches
- **Constat** : à chaque couche, `rms_norm`, `post_attention_norm`, `add_residual`, `pre_ffw_norm_2`, `post_ffw_norm_1/2`, `layer_output_scale` et `route_top_k` (`128 × 2816`) s'exécutent sur CPU, ce qui impose des centaines de `upload_f32_as_f16` / `download_as_f32` par token.
- [ ] Ajouter `hidden_dev` (FP16) dans `Gemma4GpuScratch` et des kernels `rms_norm_f16`, `add_residual_f16`, `scale_inplace_f16` (certains existent déjà pour Qwen dans `aibox_ternary_gpu.cu`).
- [ ] Porter le routeur MoE sur GPU (GEMV `128 × 2816` + top-k) et ne télécharger que les `K` indices/poids sélectionnés.
- [ ] Test d'équivalence : logits GPU-résident vs chemin actuel `< 1e-3`.

### 2. Routage MoE dynamique (Top-P sur la masse softmax)
- **Constat** : `AIBOX_MOE_TOP_K=8` offre la meilleure qualité (références : `docs/benchmark/gemma4-long-scenarios-topk8-2026-11-16.log`) mais coûte ~33 % de calcul et plus de *cache misses* que `top_k=6`.
- [ ] Garder `K_max = 8` et tronquer à `4..6` experts dès que la masse softmax cumulée atteint un seuil configurable (`AIBOX_MOE_TOP_P`, défaut `0.85`).
- [ ] Benchmark qualité/vitesse sur les 3 scénarios longs (`top_k=8` fixe vs `Top-P` dynamique) et sur le pipeline financier `MVP NEMOTRON`.

### 3. Buffers scratch pré-alloués pour le prefill par lot
- **Constat** : `forward_vram_expert_batch` et `Gemma4Linear::forward_batch` allouent/libèrent des `GpuBuffer` temporaires (`alloc_f16`) à chaque appel pendant le prefill.
- [ ] Réutiliser des buffers batch dimensionnés une fois dans `Gemma4GpuScratch` (croissance à la demande).
- [ ] Mesurer le gain de `Prefill` sur prompt de `~700` tokens (système + tableau financier).

---

## 📊 Objectif chiffré
| Métrique | Actuel | Cible |
|---|---:|---:|
| Decode (`top_k=8`, 420 tok) | `~10.6 tok/s` | `≥ 14 tok/s` |
| Appel pipeline financier (planification) | `~96 s` | `≤ 60 s` |

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 14:22 | - | TODO | Antigravity | Création du ticket suite à l'analyse des goulots CPU↔GPU restants de Gemma 4. |
