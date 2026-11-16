# FEAT-018: Prefill par Lot (`M > 1`) & Regroupement des Experts MoE (Gemma 4)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-16
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-017](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-017-gemma4-gpu-attention-kv-and-fused-geglu.md)

---

## 🎯 Objectif & Contexte
Jusqu'à présent, `Gemma4Model::generate_tokens` exécutait le prefill token par token (`M = 1`), ce qui :
1. Relisait les poids d'Attention (`attn_q`, `attn_k`, `attn_v`, `attn_o`) et `shared_expert` / `Dense` FFN `M` fois.
2. Chargeait potentiellement le même expert MoE depuis le disque/RAM/VRAM jusqu'à `M` fois par couche.

---

## 📋 Critères d'Acceptation
- [x] **MatMul par lot (`M > 1`)** : `Gemma4Linear::forward_batch` et `VramLinearBufs::forward_batch_into_dev` dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs).
- [x] **Attention & KV Cache par lot** : `forward_attention_batch_with_scratch` projette `Q/K/V` et `attn_o` en un seul batch `M` et peuple à la fois `kv_cache` CPU et les tampons VRAM `kv_k_dev` / `kv_v_dev` pour la phase de decode.
- [x] **Regroupement MoE par `expert_idx`** : `forward_moe_batch_with_scratch` regroupe les tokens routés par `expert_idx` sur l'ensemble du prompt `M`, ne chargeant chaque expert actif qu'une seule fois par couche avant d'exécuter `forward_vram_expert_batch` / `forward_geglu_expert_batch` sur le sous-batch `m_e`.
- [x] **Test unitaire d'équivalence** : `gemma4_batched_prefill_matches_sequential` vérifie l'équivalence stricte des logits prefill et decode suivant (`< 1e-4`).

---

## 📊 Benchmarks Validés (`gemma-4-26b-q4-moe2`, `Photosynthesis --max-tokens 160 --gpu`, `AIBOX_MOE_TOP_K=6`)
| Étape | `Prefill (11 tok)` | `Decode (160 tok)` | `vram_hits` | `ram_hits` | `disk_misses` |
|---|---|---|---|---|---|
| Baseline ([FEAT-017](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-017-gemma4-gpu-attention-kv-and-fused-geglu.md) + grouped decode) | `16 803.3 ms` | `4.67 tok/s` (`34 295.0 ms`) | `28 228 (92,2 %)` | `52` | `2 320` |
| **FEAT-018 (Batched Prefill `M > 1`)** | **`10 626.1 ms`** (**-36,8 %**) | **`6.55 tok/s`** (`24 433.4 ms`, **+40,3 %**) | `27 171 (88,8 %)` | `40` | `3 389` |

---

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : **34/34 PASS** (`0` warning)
- [x] `nvcc -O3 -arch=sm_89` + `cargo build --release` : `0` warning
- [x] Benchmark GPU 160 tokens validé : prefill **`10.63 s`**, decode **`6.55 tok/s`**, sortie cohérente.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-16 16:17 | - | DONE | Antigravity | Implémentation du prefill par lot (`forward_prefill_batch_with_scratch`, `forward_attention_batch_with_scratch`, `forward_moe_batch_with_scratch`), 34/34 tests PASS (0 warning), benchmark GPU validé à `10.63 s` prefill et `6.55 tok/s` decode. |
