# FEAT-014: Accélération Streaming MoE — `mmap` SafeTensors, Prefetch Asynchrone & Compression Mixte

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-16
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-013](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-013-gemma4-geglu-vram-and-lru-warmup.md)
- **Références**: [AirLLM Compression (3x)](https://github.com/lyogavin/airllm#model-compression---3x-inference-speed-up), [arXiv:2212.09720](https://arxiv.org/abs/2212.09720)

---

## 🎯 Objectif & Contexte
Les benchmarks de [FEAT-013](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-013-gemma4-geglu-vram-and-lru-warmup.md) sur `gemma-4-26b-q4` (64 tokens) montrent un taux de *hit* VRAM de **`82,9 %`** (`14 727 / 17 760`), mais **`2 244` `disk_misses`** limitent encore le débit à **`0.91–1.25 tok/s`**.

Origine des pistes (adaptées au MoE avec cache LRU VRAM à 82,9 % de *hit*) :
- **Inspiré d'[AirLLM](https://github.com/lyogavin/airllm#model-compression---3x-inference-speed-up)** : recouvrement I/O disque + calcul GPU (*async prefetch* des experts manquants pendant l'exécution des *hits* VRAM) et suppression des syscalls de lecture via `mmap` / *page cache* OS (contrairement à AirLLM qui streame 100 % d'un modèle dense à chaque token, `aibox` ne streame que les ~13 % d'experts manquants).
- **Inspiré de [Dettmers & Zettlemoyer (arXiv:2212.09720)](https://arxiv.org/abs/2212.09720)** : conservation des couches critiques en `q4_affine` par groupes de `64` (point *Pareto-optimal*) et compression mixte sur les experts froids pour dépasser `1 050` slots VRAM (`> 90 %` de *hit* VRAM).

---

## 📋 Critères d'Acceptation
- [ ] **Lecture `mmap` / zéro-syscall dans [safetensors.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/safetensors.rs)** :
  - Remplacer les `seek + read_exact` répétés de `read_tensor_slice` par un mapping mémoire en lecture seule (`mmap`) des shards SafeTensors (déléguant le caching au *Page Cache* OS sans copies intermédiaires).
- [ ] **Recouvrement I/O + Calcul GPU (*Async Prefetch*)** :
  - Dans `forward_moe_lazy_vram` ([gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs)), séparer les experts déjà en VRAM (exécutés en priorité sur GPU) des experts absents (*misses*), dont la lecture/upload est préparée en parallèle sur un *worker thread* + stream CUDA secondaire.
- [ ] **Compression mixte (Q4 critique / Q2-Q3 froid)** :
  - Conserver Attention, Shared Experts et couches sensibles en `q4_affine` (optimal selon *arXiv:2212.09720*), tout en permettant de compresser davantage les experts MoE froids ou certaines couches afin d'augmenter la capacité de `vram_experts` au-delà de `1 050` slots (> 90 % de *hit* VRAM en 6 Go).

---

## 🛠️ Plan d'Implémentation
- [x] Étape 1 : Tester `mmap` (read-only) dans `SafeTensorsFile` ([safetensors.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/safetensors.rs)) — **Rejeté après benchmark** : sur bind-mount Windows/WSL2 (8 Go RAM conteneur), `mmap` provoque du *page-cache thrashing* et des fautes de page 4 Ko (`0.18 tok/s`, `524s` prefill vs `0.91–1.25 tok/s` avec `seek + read_exact`). Code `mmap` annulé.
- [x] Étape 2 : Ordonner l'exécution des 8 experts par couche (*VRAM hits* d'abord, *misses* préchargés en parallèle via `std::thread::scope`) — **Rejeté** : sur bind-mount 9P Windows/WSL2, un *disk miss* coûte ~250 ms vs ~2 ms pour les *hits* VRAM (gain négligeable pour +60 lignes de complexité) ; `gemma4.rs` nettoyé avec `expert_ram_cache(1024)` pour éviter le swap WSL2.
- [x] Étape 3 : Étendre [convert_gguf_to_aibox.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/convert_gguf_to_aibox.py) pour supporter une configuration mixte par tenseur/couche (`q4_affine` + `q2_affine`) et valider la cohérence + le taux de *hit* VRAM sur `gemma-4-26B-A4B`.

---

## ✅ Vérification & Tests
- [ ] `cargo test --workspace --manifest-path app/Cargo.toml` : 100 % PASS (`0` warning)
- [ ] Benchmark GPU `gemma-4-26b-q4` (`Photosynthesis --max-tokens 64 --gpu`) : mesure comparative `Prefill` / `Decode tok/s` et `disk_misses` vs `1.25 tok/s` (`2 244` misses).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-16 13:13 | - | TODO | Antigravity | Création du ticket suite à l'analyse d'AirLLM (streaming/compression) et arXiv:2212.09720 (4-bit scaling laws) : `mmap`, prefetch asynchrone des *misses*, et compression mixte pour augmenter le hit rate VRAM > 90 %. |
| 2026-11-16 14:08 | IN_PROGRESS | IN_PROGRESS | Antigravity | Étape 2 (`std::thread::scope` sur les *misses*) testée et annulée : le goulot sur cold cache vient des ~250 ms/miss du bind-mount 9P Windows/WSL2 et du swap avec `expert_ram_cache(1500)`. `gemma4.rs` nettoyé sur `0e9dffe` avec `expert_ram_cache(1024)` (~3,25 Go RAM). Prochaine piste : stockage sur `ext4` WSL2 natif ou compression mixte (Étape 3). |
| 2026-11-16 14:38 | IN_PROGRESS | IN_PROGRESS | Antigravity | Étape 3 implémentée et validée : compression mixte `--quant-format q4_affine --moe-bits 2` (`gemma-4-26b-q4-moe2`, 4 shards ~7,5 Go), `matmul_q2_affine_buffers_dev` (`gpu.rs`), `VramLinearBufs` unifié Q2/Q4 + `vram_experts(1850)` + `expert_ram_cache(2048)` (`gemma4.rs`). Benchmark GPU `Photosynthesis --max-tokens 64` : **sortie cohérente**, `vram_hits=15599 (87.8%)`, `disk_misses=2161`, `Prefill 11 tok: 183167.5 ms \| Decode 64 tok: 140753.4 ms (0.45 tok/s)` sur cache froid 9P. |
| 2026-11-16 14:49 | IN_PROGRESS | IN_PROGRESS | Antigravity | Calcul dynamique du budget VRAM maximal via `aibox_gpu_vram_free_bytes()` (`cudaMemGetInfo`). Mesure sur RTX 4050 (`free=5072 MiB`, `reserved=1400 MiB`, `expert=1.77 MiB`) : `vram_cap=2071/3840` (`53,9 %` du modèle en VRAM), `vram_hits=15614 (87.9%)`, `disk_misses=2146` (*compulsory misses* à froid sur le 1er run CLI), `Decode 64 tok: 170280.7 ms (0.38 tok/s)`. |
