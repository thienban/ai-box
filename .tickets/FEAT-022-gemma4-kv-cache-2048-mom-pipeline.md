# FEAT-022: Cache KV VRAM 2048 tokens pour le pipeline Mix of Models (Gemma 4)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-17
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-018](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-018-gemma4-batched-prefill.md), [FEAT-021](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-021-tauri-desktop-app-and-ollama-packaging.md)

---

## 🎯 Pourquoi ce changement ?
Le pipeline d'analyse financière `NativeMoMPipeline` (projet `MVP NEMOTRON`, dossier `Mix of Models`) envoie à Gemma 4 des prompts bien plus longs que de simples questions de chat : prompt système (`~2 100` caractères pour le Spécialiste) + tableau comptable + historique de dialogue ReAct + bloc d'indicateurs certifiés pour la synthèse DAF.

Avec l'ancien `max_kv_seq = 512` dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs) :
1. Dès que `pos >= 512`, `try_forward_attention_vram` renvoyait `None` et **chaque token de decode retombait sur l'attention CPU** (`forward_attention_cpu_fallback`), très lente.
2. Résultat observé : timeout HTTP (`TimeoutError`, 120 s) sur le premier run `MVP NEMOTRON` (`task-5471`) et sur le second (`task-5487`).

## ✅ Changements
- `max_kv_seq : 512 → 2048` tokens par couche (`kv_k_dev` / `kv_v_dev` en VRAM).
- `reserved_bytes : 1520 MiB → 1850 MiB` pour réserver la VRAM du KV cache agrandi et conserver `vram_cap` stable sur RTX 4050 6 Go.

## 📊 Résultats validés (RTX 4050 6 Go, `gemma-4-26b-q4-moe2`, `AIBOX_MOE_TOP_K=8`)
Logs complets : [gemma4-mom-planner-2026-11-17.log](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/docs/benchmark/gemma4-mom-planner-2026-11-17.log).

| Étape du pipeline | Rôle | Durée | Résultat |
|---|---|---:|---|
| Planification | `GeneralistQueryPlanner` | `95.9 s` | JSON `{"tasks": [...]}` valide du 1er coup |
| Extraction DSL (run `task-5515`, 100 % Gemma 4) | `Specialist` | `74.8 s` – `139.1 s` | `compute_pct_change` OK, `compute_ratio` OK après 1 retry (`denominator=` → `denominateur=`) |
| Synthèse ComEx | `ExecutiveSynthesizer` | `104.5 s` | Note de `1 891` caractères, statut GenRM **`CERTIFIED`** |
| Calculs certifiés | `FinancialRuntime` (Python IEEE-754) | `< 0.1 s` | `+38.15 %` (CA net), `+14.74 %` (marge d'exploitation) |

## ⚖️ Comparatif Gemma 4 26B vs Nemotron 3.5 Bolt E4B V3
| Critère | Nemotron 3.5 Bolt E4B V3 | Gemma 4 26B-A4B (`Q4+Q2`, `aibox`) |
|---|---|---|
| Entraînement | LoRA financier (`7 827` ex. TAT-QA + FLIPPER) | Généraliste, zero-shot sur le DSL |
| Tool Calling DSL (Spécialiste) | 100 % exact du 1er coup, `0` retry | 1 retry observé sur `denominator=` |
| Planification / Synthèse (Généraliste) | Plan en prose | JSON strict + note ComEx structurée |
| Poids sur disque | `5.06 Go` (`7.5B Q4_K_M`) | `~7.5 Go` (4 shards) |

**Recommandation** : Nemotron V3 pour le rôle `Specialist` (DSL), Gemma 4 pour `Generalist` (planification, synthèse, critique GenRM).

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 14:22 | - | DONE | Antigravity | `max_kv_seq = 2048` + `reserved_bytes = 1850 MiB` dans `gemma4.rs`. 36/36 tests PASS, pipeline MoM complet validé sur GPU. |
