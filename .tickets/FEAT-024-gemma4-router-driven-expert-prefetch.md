# FEAT-024: Prefetch d'experts piloté par le routeur (hot set par fréquence)

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-18
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-023](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-023-gemma4-vram-resident-hidden-and-dynamic-topk.md)

---

## 🎯 Objectif & Contexte
Le decode de `gemma-4-26b-q4-moe2` (30 couches × 128 experts = **3840** experts, `≈ 1.77 MiB`/expert) est **I/O-disque-bound** : `expert-load` = **73 %** du temps de decode, dont **96.7 %** en `load-disk` (FEAT-023, axe 4). Le `ffn/moe` domine donc la latence, et les `disk_misses` sont majoritairement **compulsives** (l'augmentation de `vram_cap` n'a rien changé, FEAT-023 Exp 1).

L'expérience FEAT-023 §10 (« experts par question ») a montré que :
- une **seule** question active déjà `2 100–2 500` experts (`55–65 %` du pool) pour ~64 tokens ; l'union de 8 domaines = `3 459`/`3840` (`90 %`) ;
- il n'existe **pas** de sous-ensemble « par thème » (recouvrement par paires `60–85 %`) ;
- **en revanche**, le **classement par fréquence** est, lui, fortement concentré : **top `~1 000–1 400` experts couvrent `80–90 %` des routages** (`≈ 1.8–2.5 GiB`, largement dans le budget `~3.1 GiB` = `1817` experts).

⇒ **Levier** : remplacer la ségrégation *par thème* (rejetée) par
1. un **hot set chaud classé par fréquence** (préchargé/pré-rempli) ;
2. un **prefetch piloté par le routeur** : le **prefill** route déjà tous les tokens du prompt, on y collecte l'histogramme de la question puis on précharge son top-N **avant** le decode, ce qui supprime la majorité des *cold misses* **sans modélisation de thème**.

> Pré-requis déjà en place (FEAT-023) : instrumentation `AIBOX_MOE_TRACE` + `AIBOX_MOE_TRACE_FILE` (histogramme `(couche, expert)` au point unique `route_experts_for_token`, prefill **et** decode) et le LRU `vram_experts` (`VramExpertEntry`, clé `(layer_idx, expert_idx)`).

---

## 📋 Critères d'Acceptation

### 1. Hot set persistant classé par fréquence
- **Constat** : le LRU `vram_experts` est rempli **réactivement** (cold misses). Aucune notion de « experts les plus fréquents » n'est conservée entre les runs.
- [ ] Construire/maintenir un **histogramme `(couche, expert)` global** (accumulé sur l'ensemble des runs) et le **persister** (ex. `.aibox/moe_hotset.json` ou TSV binaire compact).
- [ ] Au démarrage du modèle, **pré-remplir** `vram_experts` avec le **top-N** du hot set (N borné par le budget VRAM, cf. critère 3) **avant** le premier token.
- [ ] Le hot set doit être **réutilisable hors ligne** : on doit pouvoir l'inspecter/figer (bootstrap sur un corpus de questions, puis gel).

### 2. Prefetch piloté par le routeur (par requête)
- **Constat** : le **prefill** route déjà 100 % des tokens du prompt ; cette information est actuellement jetée.
- [x] Pendant le **prefill**, collecter l'histogramme `(couche, expert)` du **prompt courant** (réutiliser la trace FEAT-023, sans I/O).
- [x] **À la fin du prefill**, **précharger** en VRAM le **top-N de la question** (ex. `N = 1000`) avant de démarrer le decode.
- [ ] Préchargement **parallélisé / asynchrone** (recouvrement I/O ↔ compute), réutilisant la lecture disque positionnée parallèle de FEAT-023 (`read_tensor_slices_parallel`).
- [x] Le prefetch est **opportuniste** : il **ne modifie pas** le résultat du routage (`top_k = 8` inchangé), seulement la **résidence**.
- [x] Levier configurable : `AIBOX_MOE_PREFETCH=<N>` (`0` = désactivé, comportement actuel).

> **⚠️ Expérience terminée — résultat NÉGATIF (voir §5)** : le prefetch par requête est un **no-op**
> (les experts du top-N du prompt sont **déjà résidents** après le prefill). Le mécanisme est
> implémenté et **laissé désactivé par défaut** (`AIBOX_MOE_PREFETCH=0`).

### 3. Politique de rétention (hot set épinglé + queue froide)
- **Constat** : précharger des experts inutiles **pollue** le LRU et peut évincer des experts chauds.
- [x] **Épingler** les experts du prefetch/hot set (non évincés par le LRU) tant qu'ils restent dans le budget.
- [x] La **queue froide** (`~2 000` experts utilisés `< 20 %` du temps) reste gérée par le **LRU RAM/disque existant** (aucun changement).
- [x] Bornage strict par le budget : `vram_cap` + `AIBOX_MOE_RESERVE_MB` ; jamais d'éviction d'un buffer critique (`hidden_dev`, KV, scratch batch).
- [x] Métrique de **gaspillage** : % d'experts préchargés mais jamais utilisés sur la requête (à surveiller, doit rester bas).

### 4. Mesure & garde-fous
- [x] **Aucune régression qualité** : sortie **identique** à `top_k = 8` sur les 3 scénarios longs (`docs/benchmark/gemma4-long-scenarios-topk8-2026-11-16.log`) — le prefetch ne touche pas au routage.
- [x] Mesurer `disk_misses` / `expert-load` **avant/après** (`AIBOX_MOE_STATS=1` + `AIBOX_GEMMA4_PROFILE=1`).
- [ ] Vérifier le budget VRAM (`nvidia-smi`, pics ≤ réserve) et l'absence de fuite (runs répétés).

---

## 📊 Objectif chiffré
| Métrique | Actuel | Cible |
|---|---:|---:|
| `disk_misses` (decode, 200 tok, froid) | `~4 032` | `≤ 1 600` (`−60 %`) |
| `expert-load` (decode) | `~37 s` | `≤ 15 s` |
| Decode (`top_k=8`, 420 tok) | `~10.6 tok/s` | `≥ 14 tok/s` |
| Appel pipeline financier (planificateur) | `21.3 s` chaud / `45.6 s` froid ✅ | `≤ 60 s` (maintenir) |
| Qualité (3 scénarios longs) | cohérente | **inchangée** (bloquant) |

> Justification des cibles : le hot set `~1 000–1 400` couvre `80–90 %` des routages d'une question ⇒ on attend **`−60 %`…`−80 %`** de `disk_misses` sur la portion préchargée ; la latence ne peut pas descendre sous le coût `expert-compute` (≈ `2.9 s`/200 tok, FEAT-023 §8).

### Comment mesurer
```powershell
$env:AIBOX_GPU_LIB = "…\app\target\release\aibox_gpu.dll"
$env:AIBOX_MOE_STATS = "1"
$env:AIBOX_GEMMA4_PROFILE = "1"
# prefetch piloté par le routeur (0 = désactivé → référence)
$env:AIBOX_MOE_PREFETCH = "1000"
# (optionnel) trace d'activation des experts
$env:AIBOX_MOE_TRACE = "1"
$env:AIBOX_MOE_TRACE_FILE = "…\trace.tsv"
# Ne PAS définir AIBOX_MOE_TOP_K / AIBOX_MOE_TOP_P (conserve top_k = 8 → qualité)
.\app\target\release\aibox.exe run C:\Users\Bui\.aibox\models\gemma-4-26b-q4-moe2 "Photosynthesis" --max-tokens 160 --gpu
```
Sortie attendue en fin de run :
```
[MoE Cache] total=… vram_hits=… (…%) ram_hits=… (…%) disk_misses=…
[MoE Prefetch] requested=… pinned=… wasted=… (…%)
[Gemma4 Profile] decode: tokens=… layers=… h2d=… d2h=…
[Gemma4 Profile]   attn=…ms (…%)  resid+glue=…ms (…%)  ffn/moe=…ms (…%)  logits=…ms (…%)
[MoE Trace] tokens=… layers=… routing_events=… distinct_experts=…/…
[MoE Trace]   experts to cover 50%->… 80%->… 90%->… 95%->… 99%->… (of … active)
```
Comparer avec `AIBOX_MOE_PREFETCH=0` (référence), même prompt, même nombre de tokens.

---

## 🧪 5. Expérience — prefetch piloté par le routeur (2026-11-18) → **NÉGATIF**

**Protocole** : Docker GPU (RTX 4050 6 Go), `gemma-4-26b-q4-moe2`, prompt
*« Explain how GPU thread blocks and warps cooperate to hide memory latency in a modern GPU. »*,
`--max-tokens 64`, `top_k = 8` verrouillé, `vram_cap = 1817`. `AIBOX_MOE_STATS=1` + `AIBOX_GEMMA4_PROFILE=1`.
Fichiers de log : `C:\Users\Bui\AppData\Local\Temp\prefetch_test\pf_*.log` et `ab_*.log`.

**Résultats (cache OS chaud — A/B alterné `0,1000,0,1000,0,1000`)** :

| Config | `disk_misses` | `vram_hits` | `ram_hits` | `expert-load` |
|---|---:|---:|---:|---:|
| `AIBOX_MOE_PREFETCH=0` (réf.) | **6874** | 14678 (68.0 %) | 48 | `1.66 – 1.85 s` |
| `AIBOX_MOE_PREFETCH=1000` | **6874** | 14672 (67.9 %) | 54 | `1.84 – 2.29 s` |
| `AIBOX_MOE_PREFETCH=1400` | **6874** | 14675 (67.9 %) | 51 | `1.77 s` |

**Détail prefetch** :
```
[MoE Prefetch] requested=1000 pinned=1000 preloaded_from_disk=0 already_resident=1000 vram_cap=1817 prefetch_ms=0.0
[MoE Prefetch] pinned=1000 loaded_from_disk=0 already_resident=1000 re_used_in_decode=701 (70.1%) waste=29.9% decode_tokens=64
[MoE Prefetch] pinned=1400 loaded_from_disk=0 already_resident=1400 re_used_in_decode=859 (61.4%) waste=38.6% decode_tokens=64
```

**Constats** :
1. **`preloaded_from_disk=0`, `already_resident=1000/1400`** : après le prefill, **100 %** des experts du top-N du prompt sont **déjà en VRAM**. Le prefill a déjà rempli la tête du LRU (la `vram_cap = 1817 > N`). Épingler la tête ne charge donc **rien de neuf** → `prefetch_ms = 0.0`. La prémisse du ticket (« le prefill jette cette information ») est vraie pour l'**histogramme**, mais la **résidence** de la tête est déjà un effet de bord gratuit du routage de prefill.
2. **`disk_misses` strictement identiques (`6874`)** quel que soit `N` : les *cold misses* du decode sont **entièrement dans la queue froide**, non prédite par l'histogramme du prompt. Réciproquement, le top-N du prompt **ne couvre qu'une minorité** de l'ensemble distinct routé pendant le decode (`re_used_in_decode` = `70 %` pour `N=1000`, `61 %` pour `N=1400`).
3. **Effet légèrement négatif** : épingler protège la tête **au prix de la rétention de la queue** → `vram_hits 14678 → 14672` (`−6`), `ram_hits 48 → 54` (`+6`). Négligeable mais confirme que l'épinglage est **neutre à nuisible**.
4. **Faux gain des 1<sup>ers</sup> runs** : le run « froid » initial affichait `expert-load ≈ 4369 ms → 1429 ms`, mais après préchauffage du cache de pages OS les deux configs **convergent** à `~1.7–2.3 s`. L'écart initial est un **effet de cache OS**, pas du prefetch.

**Conclusion** : le **prefetch par requête piloté par le routeur est un no-op** — le goulot n'est **pas** un défaut de *prefetch* mais le fait que **le jeu de travail du decode dépasse la `vram_cap`** (thrash LRU sur la queue). Épingler la tête ne peut pas réduire une queue non couverte. Les cibles (`disk_misses ≤ 1 600`) **ne sont pas atteignables** par ce mécanisme.

⇒ Le mécanisme reste **implémenté et désactivé par défaut** (`AIBOX_MOE_PREFETCH=0`, comportement inchangé). Le levier suivant n'est pas l'ordonnancement des chargements mais **la taille du jeu résident** (empreinte par expert / budget) — axes déjà explorés dans FEAT-023.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-18 | - | TODO | Antigravity | Création du ticket. Issu du verdict de FEAT-023 §10 : la sélection d'experts **par thème** est rejetée (jeux actifs `2 100–2 500`, union 8 domaines = `90 %`, recouvrement `60–85 %`) ; le levier retenu est un **hot set classé par fréquence** (top `~1 000–1 400` = `80–90 %` des routages) + **prefetch piloté par le prefill**. S'appuie sur l'instrumentation `AIBOX_MOE_TRACE` livrée dans FEAT-023. |
| 2026-11-18 | TODO | IN_PROGRESS | Antigravity | Implémentation du prefetch par requête (`AIBOX_MOE_PREFETCH=<N>`, épinglage LRU via `pop_oldest_entry_if_full_unpinned`, `pin_top_experts`/`pinned_missing`/`report_prefetch`) + expérience A/B → **§5 : résultat négatif** (`disk_misses` identiques `6874`, tête déjà résidente, `prefetch_ms = 0.0`). Mécanisme laissé **désactivé par défaut**. |
