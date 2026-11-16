# FEAT-017: Attention, RoPE, KV Cache en VRAM & Fusion `q2_affine_geglu` (Gemma 4)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-16
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-016](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-016-gemma4-exclusive-victim-cache.md)

---

## 🎯 Objectif & Contexte
Après [FEAT-016](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-016-gemma4-exclusive-victim-cache.md) (`3.82 tok/s`), deux sources d'overhead GPU/CPU subsistaient :
1. **3 lancements de kernels par expert MoE** (`gate`, `up`, `geglu_inplace_f16`) relisant `x_dev` deux fois.
2. **~145–180 synchronisations CPU $\leftrightarrow$ GPU par token dans l'Attention** (`attn_q/k/v/o` + `q/k/v_norm` + `RoPE` + `KV cache` + `GQA` exécutés sur CPU).

---

## 📋 Critères d'Acceptation
- [x] **Test 1 — Fusion `gate + up + GeGLU` 2-bit** (`q2_affine_gemv_geglu_f16_kernel` dans [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu) et `matmul_q2_affine_geglu_buffers_dev` dans [gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs)).
- [x] **Test 2 — Attention, `q/k/v_norm`, Proportional `RoPE`, KV Cache (512 tok) & `GQA` windowed 100 % en VRAM** (`gemma4_norm_rope_f16_kernel`, `gemma4_kv_append_f16_kernel`, `gemma4_gqa_attention_f16_kernel`, `aibox_gpu_copy_d2d_bytes` et `try_forward_attention_vram` dans [gemma4.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine/gemma4.rs)).

---

## 📊 Benchmarks Validés (`gemma-4-26b-q4-moe2`, `Photosynthesis --max-tokens 160 --gpu`, `AIBOX_MOE_TOP_K=6`)
| Étape | `Prefill (11 tok)` | `Decode (160 tok)` | `vram_hits` | `ram_hits` | `disk_misses` |
|---|---|---|---|---|---|
| Baseline ([FEAT-016](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-016-gemma4-exclusive-victim-cache.md)) | `19.31 s` | `3.82 tok/s` (`41.87 s`) | `28 250 (92,3 %)` | `24` | `2 326` |
| Test 1 (Fusion `q2_affine_geglu`) | `17.54 s` | `3.62 tok/s` (`44.23 s`) | `28 264 (92,4 %)` | `19` | `2 317` |
| **Test 2 (+ Attention & KV Cache VRAM)** | `22.23 s` | **`4.47 tok/s`** (`35.83 s`) | `28 108 (91,9 %)` | **`92`** | `2 400` |

---

## ✅ Vérification & Tests
- [x] `cargo test --workspace --manifest-path app/Cargo.toml` : **33/33 PASS** (`0` warning)
- [x] `nvcc -O3 -arch=sm_89` + `cargo build --release` : `0` warning
- [x] Benchmark GPU 160 tokens validé à **`4.47 tok/s`** (**23,5×** vs `0.19 tok/s` initial).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-16 15:52 | - | DONE | Antigravity | Implémentation Test 1 (`q2_affine_gemv_geglu_f16_kernel`) + Test 2 (`gemma4_norm_rope_f16_kernel`, `gemma4_kv_append_f16_kernel`, `gemma4_gqa_attention_f16_kernel`, `try_forward_attention_vram`). Windows 33/33 PASS (0 warning), `nvcc` 0 warning, benchmark GPU 160 tokens validé à `4.47 tok/s` (`91.9%` VRAM hits, `ram_hits=92`). |
