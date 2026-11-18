# FEAT-026: Re-mesure des leviers cache experts & kernel (Gemma 4 MoE) sur hôte stabilisé

- **Statut**: IN_REVIEW <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-19
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-023](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-023-gemma4-vram-resident-hidden-and-dynamic-topk.md), [FEAT-024](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-024-gemma4-router-driven-expert-prefetch.md), [FEAT-025](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-025-gemma4-dissection-router-experts-tuned-lens.md)

---

## 🎯 Objectif & Contexte
[FEAT-023](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-023-gemma4-vram-resident-hidden-and-dynamic-topk.md) a localisé le goulot du decode Gemma 4 : `ffn/moe ≈ 89 %` du temps, dont `expert-load ≈ 73 %`, dominé par la **lecture disque des experts froids** (`load-disk = 96.7 %` du load ; `≈ 4 032–4 106` misses **compulsives**, `~2–3 ms`/miss selon le fs). Les leviers déjà essayés (`vram_cap` ↑, Top-P, parallélisation disque) ont été mesurés **sur un hôte instable** (`.wslconfig memory=12GB` → OOM-kill du conteneur, page cache non déterministe).

L'hôte est désormais **stabilisé** : `.wslconfig memory=10GB`, WSL2 + Docker Desktop relancés, conteneur **`aibox-g4`**, GPU (RTX 4050) repassé en passthrough. Ce ticket **re-mesure proprement** — même protocole, conteneur et volume que [FEAT-023](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-023-gemma4-vram-resident-hidden-and-dynamic-topk.md) — les **3 candidats restants** :

1. **agrandir `AIBOX_MOE_RAM_CACHE`** (capacité du cache RAM d'experts) ;
2. **désactiver le cache victime** (repli VRAM→RAM à l'éviction) ;
3. **leviers kernel** CUDA (coût `expert-compute`, co-dominant à chaud).

### Protocole de comparaison (commun aux 3 expériences)
Conteneur `aibox-g4` ; modèle `gemma-4-26b-q4-moe2` sur le **volume nommé** `/models-vol` (fs Linux, `~1.92 ms`/miss) ; `top_k = 8` verrouillé (**ne PAS** définir `AIBOX_MOE_TOP_K` / `AIBOX_MOE_TOP_P`) ; `temperature = 0` ; `--max-tokens 200` ; **1 run froid puis 1 run chaud** par configuration ; chaque chiffre = **moyenne de ≥ 3 runs** (variance I/O disque documentée dans FEAT-023 §9).

### Variables d'environnement
| Variable | Effet | État |
|---|---|---|
| `AIBOX_MOE_RAM_CACHE=<n>` | cap du cache RAM d'experts (défaut `1024` q4 / `2048` q2) | **existant** |
| `AIBOX_MOE_VICTIM_CACHE=<0\|1>` | **défaut `0`** = ne pas recopier l'expert évincé de la VRAM vers le LRU RAM (`to_expert_weights`) ; `1`/`true` = cache victime actif (comportement historique) | **implémenté** (2026-11-19) |
| `AIBOX_MOE_RESERVE_MB=<mib>` | réserve VRAM (défaut `1850`) | existant |
| `AIBOX_GEMMA4_PROFILE=1` + `AIBOX_MOE_STATS=1` | ventilation par étape + compteurs `vram_hits`/`ram_hits`/`disk_misses`, `load-ram`/`load-disk`, `expert-compute` | existant |

> Points d'ancrage dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs) : `ram_cache_cap` (l.~793), `victim_cache_enabled` (l.~808), `insert_or_reuse_vram_expert` (l.~817, bloc victime **conditionné** l.~829-835), `load_or_cache_ram_expert` (l.~2329), `forward_moe_lazy_vram` (l.~2369).

---

## 📋 Critères d'Acceptation

### 1. Agrandir `AIBOX_MOE_RAM_CACHE`
- **Constat** : défaut `1024` (q4) / `2048` (q2). Ancienne mesure (hôte instable) : `disk_misses = 4106` / `ram_hits = 711` **identiques** à `2048` **et** `3072` ⇒ misses *compulsives* (premières visites) ⇒ cap sans effet **sur ce run court**.
- **Hypothèse à re-tester sur hôte stabilisé (10 Go)** : le jeu complet (`30 × 128 = 3840` experts, `≈ 1.8 MiB`/expert d'après `vram_cap = 1817` pour `~3.1 GiB`) pèse `≈ 6.9 GiB` ⇒ **pourrait tenir en RAM**, contrairement au `vram_cap = 1817` (FEAT-023 §10 / FEAT-025 §A : le jeu actif d'une question = `2 100–2 500` experts, **> cap VRAM**). L'intérêt se mesure sur runs **longs / multi-tours**, où un expert est revu plusieurs fois.
- [x] Re-mesurer avec `AIBOX_MOE_RAM_CACHE ∈ {défaut, 3840, 4096}` (+ contrôle `64`) : `disk_misses`, `ram_hits`, `load-ram`, `load-disk`, decode froid/chaud.

| Config | `vram_hits` | `ram_hits` | `disk_misses` | decode (ms) | tok/s |
|---|---:|---:|---:|---:|---:|
| défaut (`2048`) | `45 583` | `711` | `4 106` | `22 395` / `27 134` | `8.93` / `7.37` |
| `3840` | `45 583` | `711` | `4 106` | `14 312` / `16 771` | `13.97` / `11.93` |
| `4096` | `45 583` | `711` | `4 106` | `45 073` | `4.44` |
| **contrôle `64`** | `45 583` | **`121`** | **`4 696`** | `13 929` | `14.36` |
| **victime OFF** (`2048`) | `45 583` | **`0`** | **`4 817`** | `10 490` (moy.) | `19.1` |

- [x] Conclure : **`2048` sature le cache RAM** → `3840`/`4096` donnent des compteurs **strictement identiques** (`711`/`4 106`). Le contrôle `64` prouve que le cache est bien **exercé** (hits `711 → 121`, misses `4 106 → 4 696`) mais que le jeu des victimes **réutilisées tient dans `≤ 2048`**. ⇒ **`AIBOX_MOE_RAM_CACHE ↑` = non-levier** ; le decode varie (`4.44–13.97 tok/s` à compteurs égaux) = **bruit I/O disque**, pas effet du cap.
- [x] Vérifier la **pression mémoire** hôte à cap élevé : `free -h` avant `9.7Gi/8.6Gi libres`, après `buff/cache=6.2Gi`, **`swap` quasi vide** (`22 MiB`) ⇒ aucun OOM/swap (le conteneur tient dans les `10 Go`).

### 2. Désactiver le cache victime
- **Constat** : à **chaque** éviction VRAM, `insert_or_reuse_vram_expert` reconvertit l'expert évincé depuis ses buffers GPU f16 (`to_expert_weights()` — recopie GPU→hôte sur CPU) puis l'insère dans le LRU RAM (`expert_ram_cache`, passé en paramètre sous le nom `ram_victim_cache`). Ce repli a un **coût sur le chemin critique** de l'éviction et **pollue** le LRU RAM par des experts évincés possiblement jamais réutilisés, au risque d'**évincer** des experts utiles du cache principal.
- **Hypothèse** : pour une charge où `disk_misses` sont déjà compulsives, le cache victime coûte plus qu'il ne rapporte ⇒ le désactiver réduit `ram_hits` inutiles et le surcoût CPU, sans dégrader le débit.
- [x] Implémenter le knob `AIBOX_MOE_VICTIM_CACHE` (défaut **`0`** = cache victime désactivé ; `1`/`true` réactive le comportement historique, rétro-compatible), lu via `OnceLock` (comme `AIBOX_MOE_TOP_K`) ; désactivé, il court-circuite la recopie (`to_expert_weights` + `guard.insert`) dans `insert_or_reuse_vram_expert`.
- [x] Mesurer A/B (`=1` vs `=0`) à `top_k=8`, cap RAM par défaut (**5 runs appariés**, ON/OFF alternés) :

| Config | `vram_hits` | `ram_hits` | `disk_misses` | `expert-load` (moy.) | decode (moy.) | tok/s |
|---|---:|---:|---:|---:|---:|---:|
| victime **ON** (`=1`) | `45 583` | `711` | `4 106` | `5 295 ms` | `12 936 ms` | `15.5` |
| victime **OFF** (`=0`) | `45 583` | **`0`** | **`4 817`** | **`2 763 ms`** | **`10 490 ms`** | **`19.1`** |

- **Résultat : gain net `+23 %` en decode** (`12,94 → 10,49 s` ; `15.5 → 19.1 tok/s`), **malgré `+711` misses disque** (`4 106 → 4 817`). Le désactiver supprime `4 817` recopies GPU→hôte `to_expert_weights` (9 D2H + allocs par éviction) sur le chemin critique : `expert-load` **−48 %** (`5 295 → 2 763 ms`), `load-ram` tombe à `0 ms`. La qualité est inchangée (sortie texte identique, même routage : `vram_hits` identiques).
- [x] Vérifier l'absence de **régression** : les `711` hits RAM sont **entièrement perdus** (`ram_hits = 0`) ⇒ ce cache ne « rapporte » que `1.4 %` de hits, très en-deçà de son coût. Aucune régression de débit (`+23 %`), seul le compteur de misses disque augmente — ce qui **confirme** que le repli victime est un **coût pur** sur cette charge.
- [x] **Décision** : passer le **défaut à OFF** (comportement `=0`), rétro-compatibilité conservée via `AIBOX_MOE_VICTIM_CACHE=1`/`true`. Confirmé par smoke test 64 tok sans variable : `ram_hits=0 (0.0 %)`, `load-ram=0.0 ms` ; avec `=1` : `ram_hits=34 (0.2 %)`, `load-ram=50.2 ms` (repli réactivé).

### 3. Leviers kernel
- **Constat** : à **chaud** (`disk_misses = 0`), `expert-load ≈ expert-compute` (`≈ 2.2 s` / `≈ 2.2 s`, FEAT-023 §7) ⇒ le **calcul** devient co-dominant ; à froid il pèse `16 %` de `ffn/moe`. Kernels concernés : `aibox_gpu_gemma4_moe_q2_batch` (`grouped_q2_affine_gemv_geglu_f16_kernel`, `grouped_q2_affine` + GEGLU fusionnés) et `rms_norm_f16`, dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu).
- [x] Isoler `expert-compute` **à chaud** : `≈ 4,6–4,7 s` **stable** sur 200 tok (`≈ 45 %` du decode victime OFF) ⇒ **seul poste optimisable restant**.
- [x] Lister et hiérarchiser **≤ 3 leviers kernel** mesurables (analyse statique de [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu), **sans** réécriture — décision : **clôturer Exp 3 en profilé**) :
  1. **Vectorisation des chargements 128-bit** (`uchar4`/`uint4`) sur `packed` + `scale`/`zero` dans `grouped_q2_affine_gemv_geglu_f16_kernel` (l.~2751, `bytes_per_group=16`) ;
  2. **`__ldg`/`cp.async`** pour recouvrir la latence des poids froids dans `grouped_q2_affine_down_acc_f16_kernel` (l.~2862) ;
  3. **Réduction de passes** (GEGLU fusionné) : réduction `__shfl_xor_sync` par groupes de 16 lanes plutôt qu'un compteur partagé.
- [ ] Implémenter le levier le plus prometteur et mesurer `expert-compute` + decode chaud — **DIFFÉRÉ (hors périmètre)** : aucun levier kernel n'a été implémenté ; `expert-compute ≈ 4,6 s` reste la ligne de base, gain potentiel **non chiffré** (les 3 leviers ci-dessus sont des hypothèses statiques non mesurées). À reprendre dans un ticket dédié si le débit devient critique.

---

## 📊 Objectif chiffré
| Métrique | Actuel (hôte instable) | Cible | Résultat mesuré (2026-11-19, hôte stabilisé) |
|---|---:|---:|---|
| Decode (`top_k=8`, 200 tok) | `13.2 tok/s` (FEAT-023 §7) | `≥ 14 tok/s` | **`19.1 tok/s`** (victime OFF) ✅ |
| `disk_misses` (cap RAM ↑) | `≈ 4 032` | à établir / ↓ si non compulsives | **`4 106` identiques à `2048`/`3840`/`4096`** ⇒ cap saturé (non-levier) |
| `expert-compute` (200 tok) | `≈ 4.6 s` | ↓ (levier kernel) | `4.6–4.7 s` (stable) — Exp 3 **clôturée en profilé** (leviers listés, non implémentés) |
| `load-ram` (cache victime `=0`) | — | ≤ contrôle | **`0 ms`** (victime OFF) ✅ |

> **Statut au 2026-11-19** : **Exp 1** (`AIBOX_MOE_RAM_CACHE`) = **non-levier** (compteurs identiques dès `2048` ; le contrôle `64` prouve que le cache est exercé mais saturé). **Exp 2** (désactiver le cache victime) = **gagnant** : nouvelle knob `AIBOX_MOE_VICTIM_CACHE` (**défaut `0`** = désactivé ; `1`/`true` = historique), ⇒ **decode `+23 %` (`15.5 → 19.1 tok/s`)**, `expert-load −48 %`, au prix de `+711` misses disque — la **cible `≥ 14 tok/s` est dépassée**. **Exp 3** (leviers kernel) : `expert-compute` ≈ `4.6 s` (~`45 %` du decode victime OFF) ⇒ seul poste optimisable restant, **clôturée en profilé** (3 leviers listés, aucun implémenté).


### Comment mesurer
```powershell
# Conteneur GPU stabilisé : aibox-g4 ; modèle sur volume nommé /models-vol
docker exec -e AIBOX_GEMMA4_PROFILE=1 -e AIBOX_MOE_STATS=1 `
  -e AIBOX_MOE_RAM_CACHE=3840 `                # Exp 1 : varier {2048, 3840, 4096}
  -e AIBOX_MOE_VICTIM_CACHE=0 `                # Exp 2 : A/B {1, 0}
  aibox-g4 /app/target/release/aibox run /models-vol/gemma-4-26b-q4-moe2 "<prompt>" --max-tokens 200 --gpu
# run 2× (froid puis chaud) par configuration ; répéter ≥ 3 fois ; ne PAS définir AIBOX_MOE_TOP_K/AIBOX_MOE_TOP_P
```
Sortie attendue en fin de run :
```
[MoE Cache] total=… vram_hits=… (…%) ram_hits=… (…%) disk_misses=…
[Gemma4 Profile]   attn=…ms (…%)  resid+glue=…ms (…%)  ffn/moe=…ms (…%)  logits=…ms (…%)
[Gemma4 Profile]     expert-load=…ms (load-ram=…ms / load-disk=…ms / per-miss=…ms)  expert-compute=…ms
```
Contrôles d'environnement : `free -h` et `/proc/meminfo` (Exp 1, écarter le swap) ; sortie texte identique au contrôle (non-régression qualité, Exp 3).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-19 | - | TODO | Antigravity | Création. Planifie la re-mesure, sur l'**hôte stabilisé** (`.wslconfig memory=10GB`, WSL2/Docker relancés, conteneur `aibox-g4`), des 3 candidats restants : (1) agrandir `AIBOX_MOE_RAM_CACHE`, (2) désactiver le cache victime (`AIBOX_MOE_VICTIM_CACHE=0`, knob à créer), (3) leviers kernel CUDA. Protocole commun : `top_k=8` verrouillé, volume nommé Linux, runs froid/chaud, `≥ 3` runs par config. |
| 2026-11-19 | TODO | IN_PROGRESS | Antigravity | **Exp 1** : `AIBOX_MOE_RAM_CACHE` = **non-levier** (`ram_hits=711`/`disk_misses=4106` **identiques** à `2048`/`3840`/`4096`) ; contrôle `64` ⇒ `121`/`4696` (cache exercé mais saturé). **Exp 2** : knob `AIBOX_MOE_VICTIM_CACHE` implémenté ; A/B 5 runs ⇒ **victime OFF = decode `+23 %`** (`15.5 → 19.1 tok/s`), `expert-load −48 %`, `ram_hits 711→0`, `disk_misses 4106→4817`. **Qualité inchangée** (même routage, sortie identique). **Cible `≥14 tok/s` dépassée.** |
| 2026-11-19 | IN_PROGRESS | IN_REVIEW | Antigravity | **Décision** : défaut de `AIBOX_MOE_VICTIM_CACHE` passé à **`0` (OFF)** (rétro-compat `1`/`true`) ; smoke test 64 tok confirme `ram_hits=0`/`load-ram=0.0 ms` sans variable. **Exp 3 clôturée en profilé** : 3 leviers kernel listés (vectorisation 128-bit, `__ldg`/`cp.async`, réduction `__shfl_xor_sync`), **aucun implémenté**. Ticket en revue avant commit. |
