# FEAT-023: Accélération du Decode Gemma 4 (hidden VRAM, Top-P MoE dynamique, scratch prefill)

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
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
- [x] Ajouter `hidden_dev` (FP16) dans `Gemma4GpuScratch` et exécuter `rms_norm_f16` sur GPU pour `attn_norm`, `post_attention_norm`, `pre_ffw_norm_2`, `ffn_norm`, `post_ffw_norm_1` et `final_norm`.
- [x] Test d'équivalence : `cpu_rms_norm_stable_under_f16_input_rounding` (`< 1e-3`, CPU-only — n'exerce pas le kernel GPU).

### 2. Routage MoE dynamique (Top-P sur la masse softmax)
- **Constat** : `AIBOX_MOE_TOP_K=8` offre la meilleure qualité (références : `docs/benchmark/gemma4-long-scenarios-topk8-2026-11-16.log`) mais coûte ~33 % de calcul et plus de *cache misses* que `top_k=6`.
- [x] Garder `K_max = 8` (`MIN_DYNAMIC_EXPERTS = 4`) et tronquer à `4..8` experts dès que la masse softmax cumulée atteint un seuil configurable (`AIBOX_MOE_TOP_P`, via `truncate_by_cumulative_mass` + re-normalisation).
- [x] **Décision** : `top_k = 8` reste verrouillé pour préserver la qualité (la baisse de `top_k` dégrade la sortie) — le gain de vitesse doit venir d'ailleurs, pas d'une réduction du nombre d'experts.
- [x] Benchmark qualité/vitesse sur les 3 scénarios longs (`top_k=8` fixe vs `Top-P` dynamique) → section 9 (**Top-P rejeté : dégradation qualité**).
- [x] Benchmark sur le pipeline financier `MVP NEMOTRON` (**planificateur seul**) → section 7.

### 3. Buffers scratch pré-alloués pour le prefill par lot
- **Constat** : `forward_vram_expert_batch` et `Gemma4Linear::forward_batch` allouent/libèrent des `GpuBuffer` temporaires (`alloc_f16`) à chaque appel pendant le prefill.
- [x] Réutiliser des buffers batch dimensionnés une fois dans `Gemma4GpuScratch` (`batch_x_dev`, `batch_gate_dev`, `batch_up_dev`, `batch_out_dev`, croissance à la demande via `ensure_batch_cap`).
- [ ] Mesurer le gain de `Prefill` sur prompt de `~700` tokens (système + tableau financier).

### 4. Diagnostic de performance (instrumentation)
- **Constat (2026-11-18)** : les 3 axes ci-dessus (normes GPU, `hidden_dev` résident, scratch batch) **n'ont produit aucun gain mesurable** — le débit reste `~10.6 tok/s`. L'hypothèse initiale (« décodage orchestration CPU / latence de lancement ») est **invalidée par la mesure** ci-dessous.
- [x] Ajouter un profileur par étape activé par `AIBOX_GEMMA4_PROFILE` : temps `attn` / `resid+glue` / `ffn/moe` / `logits` + compteurs `h2d` / `d2h`, puis ventilation de `ffn/moe` en `expert-load` vs `expert-compute`, puis de `expert-load` en `load-ram` vs `load-disk`.
- [x] **Mesure** (`AIBOX_MOE_STATS=1` + `AIBOX_GEMMA4_PROFILE=1`, `top_k=8` verrouillé, 200 tok, Docker GPU) :

| Étape | Temps | Part du total |
|---|---:|---:|
| `ffn/moe` | `45.65 s` | **89.1 %** |
| ├─ expert-load | `37.41 s` | **73.0 %** |
| │ ├─ load-ram (711 hits) | `1.25 s` | `1.76 ms`/hit |
| │ └─ load-disk (4106 misses) | `36.16 s` | **`8.81 ms`/miss** (96.7 % du load) |
| └─ expert-compute | `8.24 s` | 16.1 % |
| `attn` | `4.39 s` | 8.6 % |
| `logits` | `0.86 s` | 1.7 % |
| `resid+glue` (round-trips CPU) | `0.33 s` | **0.6 %** |

- **Conclusion** : le decode est **I/O-disque-bound sur le streaming des experts** (73 % du temps ; 96.7 % du chargement = lecture disque à `~200 Mo/s`). `resid+glue` = 0.6 % → **explique pourquoi les axes 1–3 (round-trips CPU, normes GPU, scratch) n'ont rien apporté**. `disk_misses ≈ 4 100 ≈ 3 840` experts uniques → **pas de localité temporelle exploitable** (le prefetch par historique est inutile) ; `vram_hits` est déjà à 90 %.
- [x] Choisir le correctif selon le résultat → `ffn/moe` dominé par le **chargement disque** ⇒ axe « streaming des experts » (section 5).
- [x] Ajouter une trace d'activation des experts par question (`AIBOX_MOE_TRACE` + `AIBOX_MOE_TRACE_FILE`) et l'exploiter → **section 10**.

### 5. Lecture disque parallèle des experts ✅
- **Correctif** : les 9 tranches (`gate`/`up`/`down` × poids/scales/zeros) d'un expert étaient lues **séquentiellement** (`seek` + `read_exact`). Remplacées par des lectures **positionnées concurrentes** (`read_exact_at` / `seek_read`, curseur non partagé) dans [safetensors.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/safetensors.rs) (`read_tensor_slices_parallel`, jusqu'à 8 threads) ; [load_expert_weights_from_st](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs) charge les 3 tenseurs en une seule passe parallèle.
- [x] **Résultat** (200 tok, Docker GPU, `top_k=8`) :

| Métrique | Avant | Après | Gain |
|---|---:|---:|---:|
| decode total | `51.2 s` | **`28.9 s`** | **1.77×** |
| débit | `3.9 tok/s` | **`6.9 tok/s`** | **1.77×** |
| `ffn/moe` | `45.65 s` | `22.26 s` | 2.05× |
| `expert-load` | `37.41 s` | **`13.19 s`** | **2.84×** |
| `load-disk` | `36.16 s` | `12.07 s` | 3.0× |
| par miss disque | `8.81 ms` | **`2.99 ms`** | **2.95×** |

- **Interprétation** : les lectures étaient **latency-bound** (bind mount / virtiofs, ~200 Mo/s), **pas** bandwidth-bound → la concurrence I/O débloque le gain. Les stats de cache sont quasi identiques (`disk_misses` 4106 → 4032) : le gain vient du parallélisme, pas d'un meilleur taux de hit.
- **⚠️ Régression qualité introduite puis corrigée (2026-11-18)** : `read_tensor_slices_parallel` calculait l'offset fichier comme `data_offsets.0 + byte_offset` **sans ajouter** `data_start` du shard (fin de l'en-tête JSON), alors que [read_tensor_slice](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/safetensors.rs#L159-L191) fait bien `data_start + (start + byte_offset)` (ligne 183). Conséquence : **tous** les experts chargés depuis le disque étaient décalés de la taille de l'en-tête → **poids corrompus → sorties incohérentes** (gibberish multi-script, `valid_plan=false` sur le pipeline financier, aucun EOS émis). Correctif : lecture à `data_start + offset` (même convention que `read_tensor_slice`).

  | Vérification | Avant correctif | Après correctif |
  |---|---|---|
  | `aibox run … "Photosynthesis" --max-tokens 64` | `Development \어 $\ho \vis lack …` (incohérent) | **anglais cohérent** (« Photosynthesis is the process used by plants… ») |
  | Planificateur financier (`GeneralistQueryPlanner`) | `valid_plan=false`, 2 048 tok sans EOS | **`valid_plan=true`**, 2 tâches, EOS à 82 tok |


### 6. Expérimentations complémentaires (2026-11-18)

#### Exp 1 — Abaisser la réserve VRAM (`AIBOX_MOE_RESERVE_MB`) pour agrandir `vram_cap`
- **Hypothèse** : un cache VRAM plus grand ⇒ moins de `disk_misses` ⇒ decode plus rapide. Nouveau levier `AIBOX_MOE_RESERVE_MB` (défaut `1850` MiB), qui pilote `vram_cap = (vram_libre − réserve) / taille_expert`.
- [x] **Mesure** (200 tok, Docker GPU, `top_k=8`, bind mount) :

| Config | `vram_cap` | `vram_hits` | `ram_hits` | `disk_misses` | Decode | Débit |
|---|---:|---:|---:|---:|---:|---:|
| défaut (`1850`) | `1817` | `45717` | `651` | `4032` | `28.94 s` | `6.91 tok/s` |
| `AIBOX_MOE_RESERVE_MB=1200` | `2184` | `46259` | `109` | **`4032`** | `28.88 s` | `6.93 tok/s` |

- **Résultat : négatif.** `+367` slots VRAM (`+20 %`) ⇒ `disk_misses` **strictement identiques** (`4032`). Les misses sont donc **compulsives** (experts froids, aucun réemploi), **pas** des misses de capacité. Le surplus ne fait que déplacer des hits de la RAM vers la VRAM (latences quasi égales : `load-ram` ≈ `load-disk` après parallélisation). Débit inchangé (`+0.3 %`, bruit de mesure). ⇒ **`vram_cap` n'est pas le levier** ; conserver la réserve par défaut `1850` pour la marge anti-OOM du prefill.

#### Exp 2 — Modèle sur volume Docker nommé (fs Linux natif) vs bind mount Windows
- **Hypothèse** : le bind mount Windows (`virtiofs`) ajoute de la latence par lecture. Modèle copié sur le volume nommé `aibox-models-vol` (ext4 Linux, monté `/models-vol`).
- [x] **Mesure** (200 tok, Docker GPU, `top_k=8`) :

| Emplacement modèle | Decode | Débit | `load-disk` | par miss | Prefill (11 tok) |
|---|---:|---:|---:|---:|---:|
| bind mount `C:\…\.aibox\models` | `28.94 s` | `6.91 tok/s` | `12.30 s` | `3.05 ms` | `11.69 s` |
| volume nommé (fs Linux) | **`26.33 s`** | **`7.60 tok/s`** | `7.76 s` | **`1.92 ms`** | **`5.88 s`** |

- **Résultat : gain partiel mais réel.** Le fs Linux natif réduit la latence par miss de `3.05` → `1.92 ms` (`−37 %`) et **divise le prefill par 2** (`11.69` → `5.88 s`) ; decode `+10 %` (`6.91` → `7.60 tok/s`). Les `disk_misses` restent `4032` ⇒ le coût **structurel** demeure dominant : même à `1.92 ms`/miss, `≈ 4 032 × 1.92 ms ≈ 7.7 s` sur `26.3 s`. Le bind mount n'expliquait donc qu'`~10 %` du decode (mais `~50 %` du prefill).

> **Bilan des 2 expérimentations** : `vram_cap` (Exp 1) est un faux levier (misses compulsives) ; le surcoût filesystem (`virtiofs`, Exp 2) est réel mais secondaire sur le decode. Le poste dominant reste la **lecture disque obligatoire des 4 032 experts froids** ⇒ seul un changement de *représentation* (quantification plus compacte) ou de *matériel* (plus de VRAM pour tout cacher) débloquerait un gain structurel.

### 7. Benchmark du pipeline financier `MVP NEMOTRON` — planificateur seul (2026-11-18)

- **Protocole** : seul l'étage `GeneralistQueryPlanner` est exercé (`build_crew_manager_prompts` → prompt réel du pipeline : système 664 car. + utilisateur 264 car. ; `temperature=0.01`). Aucun document/tableau n'est injecté ; le LLM est appelé uniquement comme **planificateur**. Modèle `gemma-4-26b-q4-moe2`, `top_k=8` verrouillé, Docker GPU (volume nommé Linux), `AIBOX_MOE_STATS=1` + `AIBOX_GEMMA4_PROFILE=1`. Deux appels successifs : A = 1er appel (**cache froid**), B = 2e (**cache chaud**).
- Question : « Calcule la variation du chiffre d'affaires net entre N-1 et N, et la marge d'exploitation en pourcentage. »

| Appel | `max_tokens` demandé | tokens prompt | tokens générés | Wall | Engine | Plan valide |
|---|---:|---:|---:|---:|---:|:--:|
| A. 1er appel (froid) | `384` | `236` | `82` | `45.6 s` | `45.0 s` | ✅ 2 tâches |
| B. 2e appel (chaud) | `2048` | `236` | `82` | **`21.3 s`** | **`21.2 s`** | ✅ 2 tâches |

- **Résultat** : le planificateur émet un **JSON valide** (`{"tasks":[…]}`) et **s'arrête à l'EOS après 82 tokens** — la génération ne consomme donc jamais le plafond. Sortie identique pour A et B (déterminisme `temperature≈0`).
- **Effet du plancher `max_tokens`** : `hf_backend.py::_fallback_openai_payload` impose `max_tokens = max(max_tokens, 2048)` (le `384` du planificateur devient `2048` sur le fil). **Sans conséquence** ici : l'EOS survient à 82 tokens ≪ 2 048. (Avant le correctif qualité, le modèle corrompu n'émettait jamais d'EOS → il générait les 2 048 tokens, d'où les `249 s` observées : c'était un **symptôme du bug**, pas le coût réel.)
- **Ventilation (profileur, 81 pas de decode)** :

| Appel | `vram_hits` | `ram_hits` | `disk_misses` | `attn` | `ffn/moe` (dont load / compute) | `logits` | Decode |
|---|---:|---:|---:|---:|---|---:|---:|
| A (froid) | `23.6 %` | `1.8 %` | **`56 804`** | `1.35 s` | `5.76 s` (`3.55 s` / `2.21 s`) | `0.22 s` | `~7.4 s` → `10.9 tok/s` |
| B (chaud) | `51.6 %` | `48.4 %` | **`0`** | `1.26 s` | `4.52 s` (`2.29 s` / `2.24 s`) | `0.21 s` | `6.1 s` → **`13.2 tok/s`** |

- **Lecture** : à froid, le coût est dominé par le **streaming disque des experts** (`56 804` misses ≈ `236 + 81` tokens × `30` couches × `8` experts = `76 080` slots) ; à chaud (`0` miss), le decode atteint **`13.2 tok/s`** et le poste `expert-load` (RAM seule) reste ≈ `expert-compute`. L'appel planificateur **de bout en bout** passe de `95.9 s` (référence FEAT-022) à **`21.3 s` chaud** / `45.6 s` froid ⇒ **objectif `≤ 60 s` atteint dans les deux cas** (`≈ 4.5×` à chaud). Ce gain n'est **pas** attribuable à un nouveau correctif de vitesse : il résulte de la **fin de la corruption** (EOS émis à 82 tokens au lieu de 2 048).

### 8. Court test du Top-P dynamique (2026-11-18)

- **Protocole** : serveur `aibox serve` (Docker GPU, volume nommé Linux), **un processus par configuration** (le seuil est lu une seule fois via `OnceLock`). Par configuration : `1` requête « froide » puis `1` requête « chaude » (même prompt, `num_predict=128`, `temperature=0.0`). Prompt : marge d'exploitation (`1 200 000 − 900 000`).
- Comparaison `top_k=8` **fixe** (`AIBOX_MOE_TOP_P` non défini) vs **dynamique** (`AIBOX_MOE_TOP_P=0.9` / `0.85`).

| Configuration | experts/token | slots MoE | Wall chaud | Decode chaud | `ffn/moe` | └ load | └ compute | `disk_misses` (froid) | Sortie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `top_p=1.0` (8 fixe) | `8.00` | `41 280` | `9.69 s` | `13.2 tok/s` | `4 931 ms` | `1 951 ms` | `2 981 ms` | `11 484` | cohérente |
| `top_p=0.9` | `6.81` | `35 140` | `8.77 s` | **`14.6 tok/s`** | `4 406 ms` | `1 497 ms` | `2 910 ms` | `9 960` | cohérente |
| `top_p=0.85` | `6.30` | `32 521` | `8.49 s` | **`15.1 tok/s`** | `4 347 ms` | `1 455 ms` | `2 892 ms` | `9 253` | cohérente |

- **Résultats** : la troncature s'active réellement (`8 → 6.8 → 6.3` experts/token). Gain en **decode chaud** : **`+10.5 %`** (`0.9`) et **`+14.4 %`** (`0.85`). Le gain provient **uniquement de `expert-load`** (`1 951 → 1 455 ms`, `−25 %`) ; `expert-compute` reste **constant** (`≈ 2.9 s`) ⇒ le levier agit sur le *streaming* des experts, pas sur le calcul.
- **Qualité (indicatif)** : les 3 sorties restent cohérentes et exactes (`1 200 000 − 900 000 = 300 000`). Test sur **un seul prompt** ⇒ **ne remplace pas** le benchmark qualité sur les 3 scénarios longs (critère 2, toujours ouvert).
- **Réserve** : les temps « froids » (`27.6 / 19.5 / 17.7 s`) sont **non comparables** — ils incluent le réchauffement progressif du *page cache* de la VM Docker (le volume nommé survit aux redémarrages de conteneur) ; seuls les temps **chauds** (intra-processus, `disk_misses=0`) comparent réellement les configurations.

### 9. Benchmark des 3 scénarios longs — `top_k=8` fixe vs `Top-P` dynamique (2026-11-18)

- **Protocole** : exécutions CLI `aibox run … --gpu` (Docker GPU, volume nommé Linux), prompts et longueurs **identiques** à la référence `docs/benchmark/gemma4-long-scenarios-topk8-2026-11-16.log` : **S1** GPU/warps (`280` tok), **S2** respiration cellulaire (`420` tok), **S3** cache LRU Rust (`300` tok). Deux passes pour contrôler la dérive du *page cache* : **passe 1** = `top_k=8` fixe vs `top_p=0.85` (ordre alterné) ; **passe 2** = `top_k=8` fixe vs `top_p=0.9` (ordre alterné).
- **Experts/token réellement routés** (déterministe) : `8.00` (`top_k=8`) → `6.86` (`0.9`) → `6.28` (`0.85`).

| Passe | Scénario | Config | Decode | `expert-load` | `disk_misses` |
|---|---|---:|---:|---:|---:|
| 1 | S1 (280) | `top_k=8` | `12.65 tok/s` | `10 588 ms` | `10 118` |
| 1 | S1 (280) | `top_p=0.85` | `16.75 tok/s` | `6 207 ms` | `8 136` |
| 1 | S2 (420) | `top_k=8` | `15.91 tok/s` | `9 004 ms` | `10 109` |
| 1 | S2 (420) | `top_p=0.85` | `19.41 tok/s` | `6 043 ms` | `8 006` |
| 1 | S3 (300) | `top_k=8` | `13.81 tok/s` | `8 323 ms` | `10 403` |
| 1 | S3 (300) | `top_p=0.85` | `15.97 tok/s` | `6 427 ms` | `8 464` |
| 2 | S1 (280) | `top_k=8` | `13.67 tok/s` | `9 220 ms` | `10 118` |
| 2 | S1 (280) | `top_p=0.9` | `17.09 tok/s` | `5 913 ms` | `8 748` |
| 2 | S2 (420) | `top_k=8` | `17.13 tok/s` | `7 675 ms` | `10 109` |
| 2 | S2 (420) | `top_p=0.9` | `18.57 tok/s` | `6 017 ms` | `8 705` |
| 2 | S3 (300) | `top_k=8` | `16.54 tok/s` | `6 264 ms` | `10 403` |
| 2 | S3 (300) | `top_p=0.9` | `15.27 tok/s` | `6 605 ms` | `9 067` |

- **Vitesse** : `expert-load` et `disk_misses` baissent de `~15–25 %` (moins d'experts à *streamer*), mais le **decode mesuré est bruité** — en passe 2, S3 est même **plus lent** avec `0.9` qu'avec `8` fixe (`15.27` vs `16.54 tok/s`) : la variance I/O disque domine. Signal robuste = **nombre d'experts/token** (`8.00 → 6.86 / 6.28`).

**Qualité — régression constatée (déterministe, `temperature=0.0`, sorties identiques sur les 2 passes) :**

| Scénario | `top_k=8` fixe | `top_p=0.85` | `top_p=0.9` |
|---|---|---|---|
| S1 GPU/warps | cohérent | cohérent | cohérent |
| S2 respiration | propre | jeton étranger (`**로-Splitting:**`) | **boucle dégénérée** (`…pyruvate-derived-two-ly-ket-ket…` ×`72`) |
| S3 cache LRU | Rust valide | **Rust invalide** (`use std::hash::{Hash, HashRet, Hasset;`, `use std::sync even-though-it-is-not-needed-here-for-simple-logic;`) | **Rust invalide** (`use std_macro;`, `use std::util::Lru;`, `$O(ert-order)$`) |

- **Conclusion** : la troncature Top-P dynamique (`0.85` comme `0.9`) **dégrade la qualité** des générations longues (code invalide, boucles de répétition, fuite de jetons) ⇒ elle **ne doit pas** être activée par défaut. Cela **confirme** la décision du critère 2 : **`top_k = 8` reste verrouillé** ; le gain de vitesse (`expert-load`) est **rejeté** au profit de la qualité.

### 10. Sélection des experts « par question » — qui est réellement actif ? (2026-11-18)

- **Hypothèse testée** : « selon la question, seuls certains experts s'activent ⇒ on pourrait ne garder **que ceux-là** en VRAM ».
- **Instrumentation (opt-in)** : nouvelle trace `AIBOX_MOE_TRACE=1` (+ `AIBOX_MOE_TRACE_FILE=<chemin>`) dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs). Un histogramme `(couche, expert) → nombre de routages` est incrémenté au point de passage unique `route_experts_for_token` (**prefill _et_ decode**), puis résumé en fin de run et exporté en TSV (`layer\texpert\tcount`). **Coût nul** sans la variable d'environnement.
- **Protocole** : `8` questions de domaines distincts (GPU/warps, biologie, Rust, finance, maths, poème français, physique, salutation-courte), `--max-tokens 64`, `top_k = 8` verrouillé, Docker GPU. Routage **déterministe** (`temperature = 0.0`) ⇒ résultats reproductibles.

| Question | routages | experts distincts | 50 % | 80 % | 90 % | 95 % |
|---|---:|---:|---:|---:|---:|---:|
| q1 GPU / warps | `22 560` | **`2 496`** / 3840 | 370 | 975 | 1 375 | 1 705 |
| q2 biologie | `23 520` | `2 340` | 293 | 823 | 1 206 | 1 545 |
| q3 Rust / LRU | `23 280` | `2 447` | 308 | 902 | 1 307 | 1 642 |
| q4 finance | `22 320` | `2 327` | 316 | 812 | 1 190 | 1 516 |
| q5 maths | `21 120` | `2 297` | 329 | 860 | 1 215 | 1 527 |
| q6 poème FR | `22 560` | `2 113` | 233 | 662 | 989 | 1 304 |
| q7 physique | `19 920` | `2 283` | 330 | 866 | 1 235 | 1 544 |
| q8 « Hi, how are you? » | `14 400` | `1 763` | 209 | 584 | 866 | 1 151 |

- **Agrégat inter-questions** : union des experts actifs sur les **8** questions = **`3 459` / `3 840` (90 %)**. Recouvrement deux-à-deux **élevé** (`≈ 1 450 – 2 130` experts partagés, soit **~60–85 %** de chaque jeu). Classement **global** par fréquence : top `582` → 50 %, top `1 427` → 80 %, top `1 935` → 90 %.

- **Enseignements** :
  1. **Une seule question active déjà `2 100–2 500` experts** (≈ `55–65 %` du pool) pour **64** tokens générés ; même « Hi, how are you? » en active **`1 763`**. L'idée d'un petit sous-ensemble propre à une question est **démente** : le routeur brasse presque la moitié du modèle dès un prompt court.
  2. Le jeu actif d'une question **dépasse** le `vram_cap` actuel (`1817`) ⇒ **impossible** de résider 100 % des experts d'une question.
  3. Mais pour couvrir **80 % des routages** il ne faut que **`584–975`** experts (`≈ 1.0–1.7 GiB`) — largement dans le budget `~3.1 GiB` (`1817` experts) ; **90 %** ⇒ **`866–1 375`**.
  4. Les questions différentes **ne sélectionnent PAS des experts différents** : les jeux se recouvrent massivement (union de 8 domaines = **90 %** du modèle). Il n'existe **pas** de partition « par thème ».
  5. En revanche la **fréquence** est, elle, propre à la question : le top-N d'une question est **plus petit** que le top-N global pour la même couverture (50 % : `~300` vs `582` ; 80 % : `~800` vs `1427`) ⇒ c'est le **classement**, pas l'**identité** des experts, qui dépend de la question.

- **Verdict** : « mettre en VRAM seulement les experts de la question » **ne réduit pas** le jeu résident de façon exploitable (le jeu est trop large et trop recouvrant). Le levier réel est un **ensemble chaud classé par fréquence**, pas par thème :
  - précharger les **`~1 000 – 1 400`** experts les plus fréquents couvre **80–90 %** des routages (`≈ 1.8 – 2.5 GiB`) ;
  - laisser la **queue froide** (`~2 000` experts < 20 % du temps) au LRU RAM/disque existant.
  - **Piste concrète** : *prefetch piloté par le routeur* — le **prefill** route déjà tous les tokens du prompt ; on peut y collecter l'histogramme de la question puis précharger son top-N (ex. `1000`) **avant** le decode, ce qui supprime la majorité des *cold misses* sans dépendre du thème. → **implémentation suivie dans [FEAT-024](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-024-gemma4-router-driven-expert-prefetch.md)**.

---

## 📊 Objectif chiffré
| Métrique | Actuel | Cible |
|---|---:|---:|
| Decode (`top_k=8`, 420 tok) | `~10.6 tok/s` | `≥ 14 tok/s` |
| Appel pipeline financier (planification) | `21.3 s` chaud / `45.6 s` froid ✅ | `≤ 60 s` |

> **Statut au 2026-11-18** : les axes 1–3 n'ont pas modifié le débit (ils optimisaient `resid+glue` = 0.6 %). L'axe 4 (diagnostic) a localisé le goulot : **chargement disque des experts = 73 %** du decode. L'axe 5 (**lecture disque parallèle**) apporte **`3.9` → `6.9 tok/s`** en Docker (1.77×) — mais a introduit une **régression qualité** (offset manquant `data_start`) **corrigée** en fin de journée. Exp 1 (`vram_cap` ↑) = **aucun effet** (misses compulsives) ; Exp 2 (modèle sur volume nommé Linux) = **`+10 %` decode** (`7.60 tok/s`) et **prefill ÷ 2**. **Pipeline financier (planificateur seul)** : `95.9 s` → **`21.3 s`** à chaud (`13.2 tok/s` en decode, `0` miss disque) ⇒ **cible `≤ 60 s` atteinte**. Cible decode `≥ 14 tok/s` non encore atteinte sur les scénarios longs ; le coût résiduel y est **structurel** (lecture des experts froids).

### Comment mesurer
```powershell
$env:AIBOX_GPU_LIB = "…\app\target\release\aibox_gpu.dll"
$env:AIBOX_MOE_STATS = "1"
$env:AIBOX_GEMMA4_PROFILE = "1"
# (optionnel) trace d'activation des experts par question
$env:AIBOX_MOE_TRACE = "1"
$env:AIBOX_MOE_TRACE_FILE = "…\trace.tsv"
# Ne PAS définir AIBOX_MOE_TOP_K / AIBOX_MOE_TOP_P (conserve top_k = 8 → qualité)
.\app\target\release\aibox.exe run C:\Users\Bui\.aibox\models\gemma-4-26b-q4-moe2 "Photosynthesis" --max-tokens 160 --gpu
```
Sortie attendue en fin de run :
```
[MoE Cache] total=… vram_hits=… (…%) ram_hits=… (…%) disk_misses=…
[Gemma4 Profile] decode: tokens=… layers=… h2d=… d2h=…
[Gemma4 Profile]   attn=…ms (…%)  resid+glue=…ms (…%)  ffn/moe=…ms (…%)  logits=…ms (…%)
[MoE Trace] tokens=… layers=… routing_events=… distinct_experts=…/…
[MoE Trace]   experts to cover 50%->… 80%->… 90%->… 95%->… 99%->… (of … active)
```

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 14:22 | - | TODO | Antigravity | Création du ticket suite à l'analyse des goulots CPU↔GPU restants de Gemma 4. |
| 2026-11-17 18:17 | TODO | IN_REVIEW | Antigravity | Implémentation des 3 axes dans `gemma4.rs` (`hidden_dev` + `rms_norm_f16` GPU, routage MoE `AIBOX_MOE_TOP_P` dynamique avec `truncate_by_cumulative_mass`, buffers batch `batch_*_dev` pré-alloués avec `ensure_batch_cap`). 40/40 tests unitaires PASS (0 warning). En attente du démarrage de Docker Desktop pour le benchmark GPU. |
| 2026-11-18 | IN_REVIEW | IN_PROGRESS | Antigravity | Revue de code : 4 correctifs appliqués (branche morte `compute_logits_from_final_hidden`, test `cpu_rms_norm_stable_under_f16_input_rounding`, déduplication des normes/fallbacks via `apply_post_norm` + `forward_attention_cpu_path`, auto-suffisance de `compute_logits_from_hidden_dev`). Constat de performance : les 3 axes n'apportent **aucun gain** (`~10.6 tok/s`). Décision : `top_k = 8` verrouillé (qualité). Ajout du profileur par étape `AIBOX_GEMMA4_PROFILE` (axe 4) ; prochaine étape = mesurer pour localiser l'étape dominante. |
| 2026-11-18 | IN_PROGRESS | IN_PROGRESS | Antigravity | Diagnostic (axe 4) : decode **I/O-disque-bound** (expert-load = 73 % ; `load-disk` = 96.7 % du load). Correctif axe 5 (**lecture disque parallèle positionnée**, `read_tensor_slices_parallel` + `load_expert_weights_from_st`) : **`3.9` → `6.9 tok/s`** (1.77×), `expert-load` ×2.84. Ajout du levier `AIBOX_MOE_RESERVE_MB`. |
| 2026-11-18 | IN_PROGRESS | IN_PROGRESS | Antigravity | Exp 1 (`AIBOX_MOE_RESERVE_MB=1200`, `vram_cap` `1817`→`2184`) : `disk_misses` **inchangés** (`4032`) ⇒ misses **compulsives**, aucun gain (`6.93 tok/s`). Exp 2 (modèle sur volume Docker nommé, fs Linux) : per-miss `3.05`→`1.92 ms`, decode `7.60 tok/s` (`+10 %`), **prefill ÷ 2** (`11.69`→`5.88 s`). Conclusion : goulot résiduel **structurel** (lecture des experts froids). |
| 2026-11-18 | IN_PROGRESS | IN_PROGRESS | Antigravity | Court test **Top-P dynamique** (section 8) : `top_p=0.9`/`0.85` ⇒ decode chaud `13.2`→`14.6`/`15.1 tok/s` (`+10.5 %`/`+14.4 %`), gain porté à **100 % par `expert-load`** (`expert-compute` constant). Qualité OK sur **1 prompt** ⇒ critère 2 (3 scénarios longs) **toujours ouvert**. |
| 2026-11-18 | IN_PROGRESS | IN_PROGRESS | Antigravity | **Benchmark des 3 scénarios longs** (section 9, critère 2 **soldé**) : `top_k=8` fixe vs Top-P `0.85`/`0.9` (2 passes à ordre alterné). `expert-load`/`disk_misses` `−15–25 %` mais decode **bruité** (I/O disque). **Qualité dégradée** aux deux seuils : S2 boucle dégénérée (`ket-ket-…` ×72) / jeton étranger ; S3 **Rust invalide** (`use std_macro;`, `use std::util::Lru;`). ⇒ **Top-P rejeté**, `top_k = 8` verrouillé confirmé. |
| 2026-11-18 | IN_PROGRESS | IN_PROGRESS | Antigravity | **Expérience « experts par question »** (section 10) : instrumentation opt-in `AIBOX_MOE_TRACE` + 8 questions de domaines distincts. Une **seule** question active **`2 100–2 500`** experts (`55–65 %` du pool) et l'union de 8 domaines = **`3 459`/`3 840` (90 %)** ; recouvrement par paires `60–85 %`. ⇒ la **sélection par thème** ne réduit pas le jeu résident ; le levier est un **ensemble chaud classé par fréquence** (top `~1 000–1 400` = 80–90 % des routages) et un **prefetch piloté par le prefill**. |
