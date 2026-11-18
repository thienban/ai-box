# FEAT-025: Dissection interne du Gemma 4 MoE (routeur, experts, tuned lens)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Docs <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-18
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-023](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-023-gemma4-vram-resident-hidden-and-dynamic-topk.md), [FEAT-024](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-024-gemma4-router-driven-expert-prefetch.md)

---

## 🎯 Objectif & Contexte
Comprendre **mécaniquement** pourquoi Gemma 4 MoE se comporte comme observé : (a) le `Top-P` dynamique dégrade la qualité (FEAT-023 §9), (b) le prefetch piloté par le routeur est un **no-op** (FEAT-024 §5), (c) une seule question active déjà `2 100–2 500` experts (FEAT-023 §10). Ce ticket **enregistre** les mesures de dissection déjà obtenues (§A) puis planifie/exécute les **3 expériences suivantes** (§B).

Le levier technique est une instrumentation **gated par variable d'environnement** (coût nul si désactivée), branchée sur le **point de passage unique** du routage (`route_experts_for_token`) et sur les frontières de couches des deux hot paths (`forward_token_with_scratch` pour le decode, `forward_prefill_batch_with_scratch` pour le prefill).

### Variables d'environnement (instrumentation)
| Variable | Effet |
|---|---|
| `AIBOX_MOE_TRACE=1` + `AIBOX_MOE_TRACE_FILE=<tsv>` | historique `(couche, expert)` → couverture |
| `AIBOX_MOE_DUMP=<tsv>` | **A1/A2** : une ligne par **événement de routage** : entropie softmax (128 experts), `top1_prob`, `margin` (poids rang `k-1` − rang `k`), liste `expert:rang:poids` |
| `AIBOX_LAYER_LOGITS=1` | **D1** : logit-lens naïf — `final_norm`+`lm_head` appliqués au hidden à chaque frontière de couche (dernier token du prompt) |
| `AIBOX_MOE_TOP_K=<n>` | override du `top_k` de routage (`1..=8`), **process-level** (OnceLock) |
| `AIBOX_MOE_TOP_P=<p>` | override du `Top-P` dynamique (`0.5..1.0`) |

> Implémentation dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs) : `record_router_dump` / `report_dump` / `report_lens`, champs de `Gemma4GpuScratch`.

---

## 📋 Critères d'Acceptation

### A. Findings déjà obtenus (à enregistrer) — ✅
- [x] **A1 — Le routeur est quasi-uniforme.** Sur `1 260` événements de routage (30 couches × 42 positions) : `mean_router_entropy = 4.4220 nats` soit **`91.1 %`** de l'entropie maximale (`ln 128 = 4.8520`). `mean_top1_prob = 0.0873` (uniforme `1/128 = 0.0078` ; `8/128 = 0.0625`).
- [x] **A2 — Le 8ᵉ expert est essentiellement du bruit.** Marge au seuil `top_k = 8` : `margin < 1e-3` dans **`56.3 %`** des cas, `margin < 1e-2` dans **`99.2 %`**. Poids retenus : rang `0 = 0.281`, rang `1 = 0.168`, moyenne rangs `2..7 = 0.092`.
- [x] **A3 — L'entropie croît avec la profondeur** : `L0 ≈ 86 %` → `L7 ≈ 95 %` → `L29 ≈ 95 %` de l'entropie max (le routage est plus tranché en bas, quasi plat en haut).
- [x] **D1 — Le logit-lens naïf ne « polit » pas avec la profondeur.** Couches intermédiaires : `rank ≈ 2.0e5`, `top1_p = 1.00000` (dégénéré, artefacts de `final_norm`). La réponse ne « clique » qu'à `L29` (`rank = 0`, `p = 0.99996`).
- [x] **Verdict croisé** : A1/A2 expliquent mécaniquement FEAT-023 §9 (Top-P rejette une masse de routage déjà plate → tronquer change peu les logits mais dégrade), FEAT-024 §5 (aucun hot set « chaud » stable à épingler) et FEAT-023 §10 (activations diffuses, pas de sous-ensemble par thème).

### B. Expériences suivantes (exécutées) — ✅

#### B1. Ablation **causale** du `top_k` avec KL
- **Constat** : la marge au 8ᵉ seuil est `< 1e-2` dans `99.2 %` des routages (A2) ⇒ **hypothèse** : retirer le 8ᵉ expert ne change **pas** la distribution de sortie.
- [x] Instrumenter un **dump des logits** par pas de decode (`AIBOX_LOGITS_DUMP=<path>`), en mode **greedy déterministe**.
- [x] Exécuter deux runs identiques (même prompt, même seed) : `AIBOX_MOE_TOP_K=8` (référence) vs `=7` (ablation).
- [x] Mesurer **par pas** (jusqu'à la première divergence de token) : `KL(p_8 ∥ p_7)`, divergence JS, accord du `argmax`, rang du token de référence dans `p_7`.
- [x] Mesurer aussi le **KL de routage** : masse retirée par le 8ᵉ expert (`w_8`) et son effet.
- [x] **Contrainte** : aucun changement du `top_k = 8` de production (mesure uniquement).

> **Protocole** : prompt `"Photosynthesis"`, `--max-tokens 64`, greedy (`temperature=0`), même seed ; dumps `logits_k8.bin` / `logits_k7.bin` (64 pas chacun) + `router_k8.tsv` / `router_k7.tsv` (`AIBOX_MOE_DUMP`, 2 220 événements chacun).

**B1.a — Divergence des distributions (par pas de decode)**

| Phase | `KL(p_8 ∥ p_7)` (nats) | `JS` | Accord `argmax` | Rang du token réf. dans `p_7` |
|---|---:|---:|---:|---:|
| Avant divergence (pas `0..9`) | mean `0.018769` / méd `0.000000` / max `0.187595` | mean `0.006152` | **100.0 %** | mean `0.000` |
| Après divergence (pas `10..63`) | jusqu'à `49.9` | → `0.693147` (`= ln 2`) | `20.3 %` (global) | méd `140.5`, max `164 573` |
| **Séquence entière** | mean `20.99` | mean `0.532519` | `20.3 %` | mean `7 029` |

- **Premier pas divergent : pas `10` (pos `21`)** — `k=8 → 1657`, `k=7 → 15898`. Les `10` premiers tokens générés sont **identiques**.
- `JS → ln 2` après la bascule ⇒ supports quasi-**disjoints** (chaque trajectoire devient déterministe sur son propre token).
- L'écart `k=7` vs `k=8` observé sur le texte (« convert solar energy into chemical energy » vs « convert a light energy field (from sunlight) into ») est donc **la conséquence d'une unique bascule de token**, pas d'une dérive continue.

**B1.b — Masse de routage retirée par le 8ᵉ expert (`w_8`, 2 220 événements × 30 couches)**

| Grandeur | Valeur |
|---|---:|
| `Σ` des 8 poids (contrôle de normalisation) | mean `1.00000` (`min 0.99998` / `max 1.00003`) |
| `w_8` (poids normalisé du rang 7) | mean **`0.06029`** / méd `0.06090` / P95 `0.09147` / max `0.10878` |
| Réf. uniforme : `1/8` = `0.12500` ; `1/128` = `0.00781` | — |
| `w_8 / w_top1` | mean `0.2440` |
| Perturbation de renormalisation `w_8/(1-w_8)` | mean `0.06462` / max `0.12206` |
| Poids moyens par rang `0..7` | `0.2954 · 0.1764 · 0.1318 · 0.1057 · 0.0881 · 0.0756 · 0.0667 · 0.0603` |

- **Contrôle A2 sur ce run** (74 positions × 30 couches) : `mean_router_entropy = 4.3078` (≈ `88.8 %` de `ln 128 = 4.8520`), `margin < 1e-3` dans `48.0 %`, `margin < 1e-2` dans `98.8 %`.
- **Cohérence `k8` vs `k7`** : ensembles top-7 identiques dans `26.8 %` seulement (`77 %` avant divergence) — car les états cachés **diffèrent dès la 1ʳᵉ couche** (la sortie MoE inclut/exclut le 8ᵉ expert).

> **Verdict B1** : l'hypothèse « retirer le 8ᵉ expert ne change pas la sortie » est **vraie à l'échelle d'un pas** (`KL ≈ 0.019 nat`, `argmax` 100 % avant bascule, `w_8 ≈ 6 %` de la masse) mais **fausse sur la trajectoire complète** (`20.3 %` de tokens communs) à cause de la **sensibilité chaotique** du greedy. Cela **justifie de conserver `top_k = 8`** (coût marginal, l'expert 8 pèse `≈ 6 %` de la masse) tout en confirmant A2 (expert 8 quasi-neutre localement, `margin < 1e-2` dans `98.8 %`).

#### B2. Similarité inter-experts (poids)
- **Constat** : aucune mesure de redondance entre les `128` experts d'une couche.
- [x] Déquantifier `q2_affine` (`w = q*scale + zero`, groupe `64`, `scale/zero` f16) des tenseurs `ffn_gate_exps` / `ffn_up_exps` / `ffn_down_exps`.
- [x] Calculer la **similarité cosinus par paires** des `128` experts (par couche, par matrice `gate`/`up`/`down` + concaténée).
- [x] Rapporter distribution (moyenne, P95, max), paires « quasi-dupliquées » (`cos > 0.99`) et effet d'une **couche**.

> **Méthode** : chaque expert est un tenseur `R × C` aplati (`gate`/`up` : `704 × 2816` ; `down` : `2816 × 704`). Cosinus « plein » sur `R·C ≈ 1.98 M` dims ; pour tenir le budget calcul on **sous-échantillonne `R/2` lignes de sortie** (dim `≈ 9.9e5`, estimateur sans biais, `SE ≈ 1e-3`, seed fixé). `8 128` paires par matrice.

| Matrice | `cos_mean` (moy. 30 couches) | Min couche | Max couche | `cos > 0.99` | `cos > 0.9` |
|---|---:|---:|---:|---:|---:|
| `gate` | **`0.0094`** | `0.0015` (L28) | `0.0238` (L4) | **0** | **0** |
| `up` | `−0.0000` | — | — | **0** | **0** |
| `down` | `−0.0000` | — | — | **0** | **0** |
| **concaténée** | **`0.0022`** | `0.0003` (L28) | `0.0057` (L4) | **0** | **0** |

- **P95 / max** : `gate` P95 `≈ 0.010–0.014`, max `0.0433` (L19) ; `up`/`down` max `< 0.006` ; concaténée max `< 0.008`.
- **Effet de couche** (concaténée) : plus corrélée en bas (`L4 = 0.0057`, `L6 = 0.0052`) et quasi-décorrélée en haut (`L27 = 0.0005`, `L28 = 0.0003`, `L29 = 0.0005`).

> **Verdict B2** : les `128` experts sont **quasi-orthogonaux** — **aucune paire quasi-dupliquée** (`cos > 0.99` : `0`), **aucune redondance exploitable**. Pas d'opportunité de fusion / déduplication ; chaque expert porte une direction distincte. Cohérent avec FEAT-023 §10 (activations diffuses, pas de sous-ensemble par thème) et avec A1/A2 (le routeur ne peut pas « économiser » en fusionnant des experts similaires).

#### B3. **Tuned lens** / sonde linéaire par couche
- **Constat** : le logit-lens naïf (D1) est artefact-prone (application directe de `final_norm` à un hidden non final).
- [x] Capturer les hidden **par couche** (f16) sur toutes les positions du prefill (`AIBOX_HIDDEN_DUMP=<path>`).
- [x] Ajuster un **traducteur affine** `h_L → h_final` par couche (ridge / régression linéaire), c.-à-d. un *tuned lens*.
- [x] Évaluer, par couche : `KL(tuned ∥ final)` vs `KL(logit-lens ∥ final)`, rang du token correct, et similarité cosinus au hidden final → courbe de **raffinement avec la profondeur**.

> **Méthode** : dump `hidden.bin` (31 blocs f16 : sortie de chaque couche `h_L` + bloc final `h_final`), `m = 276` positions. Sonde ridge duale `h_L → h_final`, évaluée en **validation croisée 5-fold hors-échantillon**, `λ` choisi par grille (proxy MSE). `final = lm_head(rms_norm(h_final))`, `logit-lens = lm_head(rms_norm(h_L))`.

| Couche | `KL_tuned` | `KL_ll` (naïf) | top-1 tuned | top-1 naïf | `cos(h_L,h_final)` | `cos(ŷ,h_final)` |
|---|---:|---:|---:|---:|---:|---:|
| L00 | `0.5354` | `14.4820` | `22.1 %` | `0.0 %` | `0.0009` | `0.8467` |
| L10 | `0.4707` | `11.9709` | `21.4 %` | `0.7 %` | `−0.0137` | `0.8611` |
| L15 | `0.4217` | `2.1749` | `26.4 %` | `2.9 %` | `0.0339` | `0.8829` |
| L20 | `0.3643` | `2.7150` | `33.7 %` | `1.1 %` | `0.2837` | `0.8979` |
| L25 | `0.3201` | `5.1466` | `34.1 %` | `0.0 %` | `0.3813` | `0.9097` |
| L28 | `0.2898` | `4.4068` | `33.3 %` | `0.4 %` | `0.1105` | `0.9207` |
| **L29** (contrôle identité) | `0.2060` | `0.0000` | `46.4 %` | **100.0 %** | **`1.0000`** | `0.9642` |

- **Cible atteinte** : `KL(tuned ∥ final)` mean `0.4362` (toutes couches), `0.3595` (couches `15-29`), `0.2876` (couches `25-29`), max `0.6604` → **`< 1 nat`** partout. `KL_ll` naïf mean `8.09` (max `14.48`), dégénéré.
- **Courbe de raffinement** : `KL_tuned` décroît `0.5354 (L0) → 0.2898 (L28)` ; `cos(h_L,h_final)` brut monte `0.0009 → 0.40` (L22) `→ 1.0` (L29) ; `cos(ŷ,h_final)` monte `0.85 → 0.92`.
- **⚠️ Caveat / plancher du contrôle L29** : à `L29`, `h_L ≡ h_final` (`cos = 1.0`, `KL_ll = 0`), mais la sonde ridge redonne `KL_tuned = 0.206` (`top-1 46.4 %`). C'est un **plancher** dû à `n = 276 ≪ d = 2816` : la solution duale projette sur un sous-espace de rang `≤ 275`, donc même la carte identité n'est pas exactement recouvrée. Les `KL_tuned` absolus sont donc des **bornes supérieures** ; la **comparaison tuned vs naïf** (÷≈`18`) est le résultat robuste.

> **Verdict B3** : la sonde linéaire (*tuned lens*) rend le logit-lens **exploitable** — elle réduit le `KL` moyen d'un facteur `≈ 18` (`8.09 → 0.44 nat`) et satisfait la cible (`< 1 nat`, `0.36 nat` pour `L ≥ 15`) avec une décroissance nette avec la profondeur. Le logit-lens naïf reste inutilisable (artefacts de `final_norm`). Le résultat est limité par la taille d'échantillon du probe (un seul prompt de `276` positions) : le `KL` mesuré est une borne supérieure, améliorable en accumulant davantage de prompts (le dump est en mode *append*).

---

## 📊 Objectif chiffré
| Métrique | Cible | Résultat |
|---|---|---:|
| `KL(p_8 ∥ p_7)` (decode, greedy, avant divergence) | documenter ; **justifier** `top_k = 8` | mean `0.0188` nat (méd `0`, max `0.188`) |
| `w_8` — masse retirée par le 8ᵉ expert | documenter | mean `0.0603` (`≈ 6 %` de la masse ; `w_8/w_top1 = 0.24`) |
| Divergence de séquence `k=7` vs `k=8` | documenter | `13/64` = `20.3 %` ; 1ʳᵉ bascule au pas `10` |
| Similarité cosinus inter-experts (moy.) | documenter (duplicats ?) | `0.0022` (concat) / `0.0094` (gate) ; **0** paire `> 0.99` |
| `KL(tuned-lens ∥ final)` (couches ≥ 15) | `< 1` nat, décroissant avec la profondeur | mean `0.3595` ✅ (décroît `0.535 → 0.290`) ; naïf `8.09` |

### Comment mesurer
```powershell
$env:AIBOX_GPU_LIB = "…\app\target\release\aibox_gpu.dll"
$env:AIBOX_GEMMA4_PROFILE = "1"; $env:AIBOX_MOE_STATS = "1"
# B1 : dump des logits de decode + top_k verrouillé/uniquement pour la mesure
$env:AIBOX_LOGITS_DUMP = "…\logits_k8.bin"; .\app\target\release\aibox.exe run <model> "<prompt>" --max-tokens 64 --gpu
# B3 : capture des hidden par couche (bloc binaire f16, append : plusieurs prompts s'accumulent)
$env:AIBOX_HIDDEN_DUMP = "…\hidden.bin"; $env:AIBOX_LAYER_LOGITS = "1"
```
Analyse hors-ligne : Python 3.13 + numpy 2.2.5 (+ `safetensors` 0.7.0 pour B2).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-18 | - | IN_PROGRESS | Antigravity | Création. Enregistre les findings A1/A2/D1 (routeur quasi-uniforme `91.1 %`, `margin<1e-2` dans `99.2 %`, logit-lens non raffinant) et planifie les 3 expériences B1 (ablation causale `top_k` + KL), B2 (similarité inter-experts), B3 (tuned lens / sonde linéaire par couche). |
| 2026-11-18 | IN_PROGRESS | DONE | Antigravity | §B exécuté (build + runs Docker GPU, analyse Python hors-ligne). **B1** : `KL(p_8∥p_7)` avant divergence `0.019` nat, accord `argmax` `100 %`, 1ʳᵉ bascule au pas `10` → divergence totale (`13/64` tokens) ; `w_8` mean `0.0603` (`≈6 %` masse, `w_8/w_top1=0.24`) ⇒ garder `top_k=8` justifié. **B2** : experts quasi-orthogonaux (`cos` concat `0.0022`, gate `0.0094`), **0** duplicat `>0.99`. **B3** : tuned lens `KL` mean `0.36` nat (`L≥15`) vs naïf `8.09` (÷≈18), décroissant avec la profondeur ; plancher `L29=0.206` documenté (`n≪d`). |
