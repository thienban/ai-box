# FEAT-016: Victim Cache Exclusif VRAM + RAM (100 % du Modèle 26B sans Duplication)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-16
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-014](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-014-gemma4-mmap-async-prefetch-and-mixed-precision.md)

---

## 🎯 Objectif & Contexte
Jusqu'ici, `vram_experts(2071)` et `expert_ram_cache(2048)` étaient inclusifs : les mêmes ~2 048 experts chauds étaient dupliqués en VRAM (`~3,66 Go`) et en RAM (`~3,62 Go`), laissant les `1 769` autres experts sur disque.

En transformant `expert_ram_cache` en **Victim Cache exclusif** :
- Tout expert chargé en VRAM est retiré de `expert_ram_cache` (`remove`).
- Tout expert évincé de VRAM (`pop_oldest_entry_if_full`) est rapatrié en RAM (`to_expert_weights` via `GpuBuffer::download`) avant réutilisation du slot VRAM.
- Capacité combinée sur `gemma-4-26b-q4-moe2` : `2 071 (VRAM) + 2 048 (RAM) = 4 119 > 3 840` experts (**100 % du modèle 26B en VRAM + RAM sans duplication**).

---

## 📋 Critères d'Acceptation
- [x] `GpuBuffer<T>::download(&self, out: &mut [T])` générique (`u8`, `u16`, `f32`) dans [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs).
- [x] `LruMap::pop_oldest_entry_if_full()` et `LruMap::remove()` dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs).
- [x] `VramLinearBufs::to_linear()` + `VramExpertEntry::to_expert_weights()` + éviction VRAM $\rightarrow$ RAM dans `insert_or_reuse_vram_expert`.

---

## 📊 Benchmark Validé (`gemma-4-26b-q4-moe2`, `Photosynthesis --max-tokens 160 --gpu`, `AIBOX_MOE_TOP_K=6`)
- **Budget VRAM** : `vram_cap = 2 071 / 3 840` (`free=5072 MiB`, `expert=1.77 MiB`)
- **Statistiques MoE** : `total=30600`, **`vram_hits=28250 (92.3%)`**, **`ram_hits=24`**, `disk_misses=2326`
- **Vitesse** : `Prefill 11 tok: 19310.8 ms | Decode 160 tok: 41869.9 ms` (**`3.82 tok/s`**)

---

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : **33/33 PASS** (`0` warning)
- [x] Benchmark GPU 160 tokens validé (`3.82 tok/s`, `92.3%` VRAM hits, `ram_hits=24`).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-16 15:31 | - | DONE | Antigravity | Implémentation du Victim Cache exclusif VRAM + RAM (`GpuBuffer<T>::download`, `to_expert_weights`, `pop_oldest_entry_if_full`, `remove`). Windows 33/33 PASS (0 warning), benchmark GPU 160 tokens validé à `3.82 tok/s` (`92.3%` VRAM hits, `ram_hits=24`). |
