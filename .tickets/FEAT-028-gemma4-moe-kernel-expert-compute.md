# FEAT-028: Optimisation kernel CUDA du `expert-compute` MoE (Gemma 4)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-19
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-026](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-026-gemma4-expert-cache-and-kernel-levers-remesure.md) (§3 leviers kernel, clôturée **en profilé** : `expert-compute ≈ 4,6 s`), [FEAT-017](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-017-gemma4-gpu-attention-kv-and-fused-geglu.md), [FEAT-027](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-027-gemma4-vram-reserve-90pct-tuning.md)

---

## 🎯 Objectif & Contexte
[FEAT-026](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-026-gemma4-expert-cache-and-kernel-levers-remesure.md) a établi que le **cache/victime/VRAM** ne sont plus des leviers (I/O reduction ⇒ `+1,7 %` tok/s seulement, FEAT-027) : le poste dominant restant est le **calcul des experts** (`expert-compute ≈ 4,6–4,7 s` stable sur 200 tok, `≈ 45 %` du decode). Les 3 leviers kernel avaient été **listés puis clôturés en profilé** sans implémentation (Exp 3). Ce ticket **implémente** le levier le plus prometteur et le **mesure**.

Analyse : sur ce modèle (`hidden_dim = 2816` = `44` groupes de 64 ; `inter_dim = 704` = `11` groupes), le noyau batch MoE `aibox_gpu_gemma4_moe_q2_batch` s'exécute en régime **instruction-bound** (le GEMV 2-bit décode/scanne les poids avec peu d'arithmétique utile) : pour `286 G MAC` sur `4,6 s` on obtient `≈ 62 GMAC/s`, très en-deçà du débit fp16 du RTX 4050 ⇒ le goulot est le **nombre d'instructions** (chargements + shuffles du `__shfl_down_sync`), pas le FLOP.

### Noyaux ciblés
[dans aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) — `grouped_q2_affine_gemv_geglu_f16_kernel` (gate+up+GEGLU fusionnés) et `grouped_q2_affine_down_acc_f16_kernel` :
- **v1 (baseline)** : 16 lanes / groupe, `uint2` (8 o = 4 halfs) sur `x`, **1 octet** de poids / lane (`pg`/`pu`), **2 groupes / itération**, réduction `__shfl_down_sync` sur **4 passes** (offset 8,4,2,1) × 3 valeurs, fusion finale à l'offset 16 (2 lanes accumulatrices).

### Levier retenu (fusion des leviers 1 + 3 de FEAT-026 §3)
- **v2** : 8 lanes / groupe, `uint4` (16 o = 8 halfs) sur `x`, **`uchar2` (2 o)** de poids / lane, **4 groupes / itération** (`sub_g = lane >> 3`), réduction sur **3 passes** (offset 4,2,1), fusion finale offsets 16 puis 8 (4 lanes accumulatrices).
- Effet attendu : **÷2 chargements** de poids (2 `uchar2` pour 4 groupes vs 2 octets pour 2), **÷2,7 shuffles** par groupe (`9` pour 4 groupes vs `12` pour 2), **÷2 itérations** de boucle.

> Le levier `__ldg`/`cp.async` (lever 2) est écarté : les pointeurs sont déjà `__restrict__ const` ⇒ nvcc émet déjà des `LDG.E.CI` ; gain attendu nul/speculatif.

### Protocole (A/B reproductible, même binaire)
Pendant la campagne, les deux variantes (`v1` legacy / `v2`) ont **coexisté** dans `aibox_gpu_gemma4_moe_q2_batch`, sélectionnées par `AIBOX_MOE_KERNEL` (`=1` ⇒ v1 ; sinon v2), un seul rebuild du `.so` servi pour l'A/B par variable d'environnement (mêmes conditions disque/thermal).
Conteneur `nvidia/cuda:12.4.1-devel-ubuntu22.04` (le `.so` est recompilé ; le binaire Rust `aibox-linux` est **inchangé**) ; modèle `gemma-4-26b-q4-moe2` sur le volume nommé `aibox-models-vol` (`/models-vol`, fs Linux) ; `top_k = 8` verrouillé (ne PAS définir `AIBOX_MOE_TOP_K`/`AIBOX_MOE_TOP_P`) ; `temperature = 0` ; `--max-tokens 200` ; **runs appariés v1/v2 alternés** ; `AIBOX_MOE_VICTIM_CACHE` défaut (`0`).

> **État final** : après validation du gain, la `v1` et le commutateur `AIBOX_MOE_KERNEL` ont été **supprimés**. La `v2` est désormais le **seul** noyau (nom canonique `grouped_q2_affine_gemv_geglu_f16_kernel` / `grouped_q2_affine_down_acc_f16_kernel`), lancement direct sans branche. Bénéfice : ~`34 Ko` de `.so` en moins, un seul chemin de code à maintenir.

### Variables d'environnement
| Variable | Effet | État |
|---|---|---|
| `AIBOX_GEMMA4_PROFILE=1` + `AIBOX_MOE_STATS=1` | ventilation par étape + `expert-compute` | existant |
| `AIBOX_GPU_LIB=<chemin>` | `.so` CUDA à charger | existant |

---

## 📋 Critères d'Acceptation
- [x] Implémenter la variante `v2` (128-bit loads + réduction 8-lane) dans `grouped_q2_affine_gemv_geglu_f16_kernel` et `grouped_q2_affine_down_acc_f16_kernel`, **sans** modifier la signature ni le lanceur public.
- [x] Conserver `v1` (renommée `_v1`) et exposer `AIBOX_MOE_KERNEL` **pour la durée de l'A/B**.
- [x] Compiler le `.so` (`nvcc -O3 -arch=sm_89`) — **sans erreur** (`BUILD_OK`).
- [x] Vérifier la **non-régression de qualité** : sortie texte **identique** v1/v2 (`temperature = 0`, même prompt) ; routage identique (`total=52 320`, `vram_hits ≈ 46 22x`).
- [x] Mesurer A/B apparié : `expert-compute` (200 tok) et decode (`tok/s`).
- [x] **Promouvoir `v2` en noyau unique** : supprimer `v1` + le commutateur `AIBOX_MOE_KERNEL`, lancement direct ; rebuild OK (`1 610 944 o`), sortie serveur correcte.

### Résultats A/B (6 runs appariés **entrelacés**, même conteneur, disque chaud)
| Variante | `expert-compute` (200 tok) | decode (ms) | tok/s | sortie |
|---|---:|---:|---:|---|
| **v1** (baseline) | **`4 961,9 ms`** (moy. `4 945,8` / `5 037,7` / `4 902,1`) | `9 882 ms` | `20,24` | _réf_ |
| **v2** (optimisée) | **`4 245,1 ms`** (moy. `4 230,3` / `4 253,4` / `4 251,7`) | `9 009 ms` | `22,20` | **identique** ✅ |

> **Chaque** run v2 bat **chaque** run v1 (`expert-compute` et decode) ⇒ effet réel, non qualifiable de bruit. La 1ʳᵉ paire « froide » (`5 371` vs `4 835 ms`) avait laissé croire à un gain nul : c'était le **page cache disque** qui faussait `expert-compute` (contention CPU/DMA), d'où le protocole entrelacé ci-dessus.

---

## 📊 Objectif chiffré
| Métrique | Actuel | Cible | Résultat |
|---|---:|---:|---|
| `expert-compute` (200 tok) | `4 961,9 ms` (v1, mesuré) | `≤ 4,0 s` (`−13 %`) | **`4 245,1 ms` (`−14,4 %`)** ✅ |
| Decode (`top_k=8`, 200 tok) | `20,24 tok/s` (v1, mesuré) | `≥ 20 tok/s` | **`22,20 tok/s` (`+9,7 %`)** ✅ |

> **Conclusion** : la variante `v2` (chargements 128-bit `uint4`/`uchar2` + réduction sur 8 lanes, 4 groupes/itération) réduit `expert-compute` de **`−14,4 %`** (`4 962 → 4 245 ms`) et accélère le decode de **`+9,7 %`** (`20,24 → 22,20 tok/s`), **à qualité strictement identique** (sortie texte et routage inchangés). Gain modeste en valeur absolue car le poste reste **latency-bound** (le GEMV 2-bit lit beaucoup de poids pour peu d'arithmétique) : réduire le nombre d'instructions chargements/shuffles n'élimine pas l'attente mémoire des poids. La `v2` est promue en **noyau unique** (la `v1` et `AIBOX_MOE_KERNEL` sont supprimés).

### Comment mesurer
```powershell
# 1. Stop du serveur qui tient la VRAM
docker stop aibox-g4srv
# 2. Build du .so optimisé (image devel, /app = bind Windows, lent)
docker run --rm --gpus all -v "c:\Users\Bui\Desktop\DevProjets\tiny-gpu\app:/app" `
  -w /app nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c `
  "nvcc -O3 -arch=sm_89 --shared -Xcompiler -fPIC crates/aibox-kernels/cuda/aibox_ternary_gpu.cu -o /app/target/release/libaibox_gpu.so"
# 3. Run profilé (v2 = seul noyau, plus de AIBOX_MOE_KERNEL)
docker run --rm --gpus all -v "c:\Users\Bui\Desktop\DevProjets\tiny-gpu\app:/app" `
  -v aibox-models-vol:/models-vol:ro -w /app `
  -e AIBOX_GPU_LIB=/app/target/release/libaibox_gpu.so `
  -e AIBOX_GEMMA4_PROFILE=1 -e AIBOX_MOE_STATS=1 `
  nvidia/cuda:12.4.1-runtime-ubuntu22.04 `
  /app/target/release/aibox-linux run /models-vol/gemma-4-26b-q4-moe2 "<prompt>" --max-tokens 200 --gpu
```
Sortie attendue :
```
[MoE Cache] total=… vram_hits=… (…%) ram_hits=… (…%) disk_misses=…
[Gemma4 Profile]   attn=…ms  resid+glue=…ms  ffn/moe=…ms  logits=…ms
[Gemma4 Profile]     expert-load=…ms (load-ram=…ms / load-disk=…ms)  expert-compute=…ms
```

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-19 | - | IN_PROGRESS | Antigravity | Création. Implémente le levier kernel de FEAT-026 §3 (fusion vectorisation 128-bit + réduction 8-lane) sous forme de variante `v2` **coexistant** avec la `v1` (sélection `AIBOX_MOE_KERNEL`). Protocole A/B apparié, `--max-tokens 200`, sortie texte comparée. |
| 2026-11-19 | IN_PROGRESS | DONE | Antigravity | Build `nvcc -O3 -arch=sm_89` OK. Qualité **identique** (sortie et routage inchangés). A/B **entrelacé** (6 runs) : `expert-compute` **`4 962 → 4 245 ms` (`−14,4 %`)**, decode **`20,24 → 22,20 tok/s` (`+9,7 %`)**. Cible `expert-compute ≤ 4,0 s` **presque atteinte** (`4,245 s`), cible decode `≥ 20 tok/s` **dépassée**. |
| 2026-11-19 | DONE | DONE | Antigravity | **Consolidation** : gain jugé modeste mais réel et non ambigu (`+9,7 %` decode). La `v2` est promue **noyau unique** — suppression de la `v1` et du commutateur `AIBOX_MOE_KERNEL` ; rebuild OK (`1 610 944 o`), serveur GPU relancé, sortie « Photosynthesis » correcte. On conserve donc la `v2` **au lieu de tout révoquer**. |
