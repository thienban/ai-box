# FEAT-027: Gemma 4 MoE — VRAM utilization tuning (target ~90%) via `AIBOX_MOE_RESERVE_MB`

- **Status**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priority**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Performance <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Created**: 2026-11-19
- **Assignee**: TRAE
- **Depends on**: [FEAT-026](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-026-gemma4-expert-cache-and-kernel-levers-remesure.md), [FEAT-024](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-024-gemma4-router-driven-expert-prefetch.md), [FEAT-023](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-023-gemma4-vram-resident-hidden-and-dynamic-topk.md)

---

## Objective & Context

Observation while serving `gemma-4-26b-q4-moe2` (Gemma 4 26B‑A4B MoE) with `aibox serve` on a 6 GB RTX 4050: VRAM usage sat at **~43 % (2644 / 6141 MiB)** while idle — far below capacity. Question raised: *is it normal not to use full VRAM, and can we push it to ~90 %?*

The engine reserves VRAM headroom **by design**. At first inference the expert VRAM cache is sized as:

```rust
let reserved_mb = env::var("AIBOX_MOE_RESERVE_MB")...unwrap_or(1850);      // default 1850 MiB
let vram_cap = ((free_bytes - reserved_bytes) / bytes_per_exp).clamp(256, total_experts);
```
[gemma4.rs#L2150-L2179](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs#L2150-L2179)

Because experts then fill toward `free − reserve`, once the expert LRU saturates:

```
steady-state VRAM used ≈ total − reserve
```

With `total = 6141 MiB`, the default reserve (1850 MiB) caps usage at **~70 %**. The reserve exists to cover CUDA-context fragmentation, activation staging, the fixed KV cache (`max_kv_seq = 2048`, [gemma4.rs#L1014-L1021](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs#L1014-L1021)) and other GPU consumers.

**Question of this ticket**: does lowering `AIBOX_MOE_RESERVE_MB` to ~600 MiB raise steady-state VRAM to ~90 % and improve decode throughput (fewer cold expert disk misses), **without OOM**?

### Capacity math (q2 experts)
- `bytes_per_exp = 3 × moe_intermediate_size(704) × hidden_size(2816) × 5/16 ≈ 1.77 MiB`
- `total_experts = num_layers(30) × num_experts(128) = 3840` → full cache ≈ **6.8 GiB** (> 6141 MiB VRAM, so the `clamp(256, 3840)` never binds — **the reserve is the only lever**).

| `AIBOX_MOE_RESERVE_MB` | ≈ VRAM used | % of 6141 MiB |
|---|---:|---:|
| 1850 (default) | ~4291 MiB | ~70 % |
| 921 | ~5220 MiB | ~85 % |
| **600** | **~5527 MiB** | **~90 %** |
| 300 | ~5841 MiB | ~95 % |

### Vram cap is a ceiling, not a pre-allocation
`Gemma4GpuScratch::new` only sizes the LRU (`vram_experts: LruMap::new(vram_expert_cap)`, [gemma4.rs#L1052](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs#L1052)); device buffers are allocated **per expert on first use** in `insert_or_reuse_vram_expert` ([gemma4.rs#L817-L845](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-core/src/engine/gemma4.rs#L817-L845)). So usage only rises as experts are actually routed — a short run may never reach the cap.

---

## Protocol

- Container `aibox-g4srv` (runtime image `nvidia/cuda:12.4.1-runtime-ubuntu22.04`, `--gpus all`).
- Model on the **named volume** `aibox-models-vol:/models-vol:ro` (Linux fs, fast expert reads — the FEAT-023/026 standard), **not** the Windows bind mount (`c:\Users\Bui\.aibox\models`), whose per-miss latency is much higher.
- **Fixed env**: `AIBOX_MOE_STATS=1`, `AIBOX_MOE_PREFETCH=64`, `AIBOX_GPU_LIB=/app/target/release/libaibox_gpu.so`.
- **Vary**: `AIBOX_MOE_RESERVE_MB ∈ {1850 (baseline), 600}`.
- Fixed prompt, `max_tokens=512`, `temperature=0`; per config: 1 warm-up run (init scratch + warm page/VRAM cache) then 1 timed run.
- Sample `nvidia-smi` right after the timed run; capture `[VRAM Budget]` / `[MoE Prefetch]` from `docker logs`; tok/s = `usage.completion_tokens / elapsed`.

### Env knobs referenced
| Variable | Effect | State |
|---|---|---|
| `AIBOX_MOE_RESERVE_MB=<mib>` | VRAM headroom subtracted before sizing `vram_cap` (default `1850`) | existing |
| `AIBOX_MOE_PREFETCH=<n>` | after prefill, pin & preload the top-N most-routed experts into VRAM | existing |
| `AIBOX_MOE_STATS=1` | logs `[VRAM Budget]` / `[MoE Prefetch]` / `[MoE Cache]` | existing |
| `AIBOX_MOE_RAM_CACHE=<n>` | RAM LRU cache cap (default `1024` q4 / `2048` q2) | existing |

---

## Acceptance Criteria

- [x] Measure `vram_cap`, observed VRAM (MiB, %), `disk_misses`, decode time, tok/s for `RESERVE_MB = 1850` and `= 600`.
- [x] Target: steady-state VRAM **≥ 85 %** of 6141 MiB with **no OOM**.
- [x] Confirm tok/s ≥ baseline (or document if the LRU does not saturate on a 512-token run).

### Results (measured)

Both configs run under identical protocol on `aibox-g4srv`, model on `aibox-models-vol`, `AIBOX_MOE_PREFETCH=64`, `max_tokens=512`, `temperature=0`. `[VRAM Budget]` line shows `free_bytes = 5072 MiB` at init (≈1 GiB already held by other GPU consumers, hence the `total − reserve` prediction of §Capacity math being optimistic vs the observed figure).

| `RESERVE_MB` | `vram_cap` (log) | VRAM used (MiB) | % of 6141 | `disk_misses` | decode (s) | tok/s | OOM? |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1850 (baseline) | 1817 / 3840 | 5535 | 90.2 % | 11070 | 32.7 | 15.66 | no |
| 600 | 2523 / 3840 | 5901 | 96.1 % | 1039 | 32.2 | 15.92 | no |

Supporting counters (`[MoE Cache]`, identical `total = 139440` for both = 512 tok × 30 layers × ~8.9 experts):

| `RESERVE_MB` | `vram_hits` | `ram_hits` | `disk_misses` | prefetch |
|---:|---:|---:|---:|---|
| 1850 | 128370 (92.1 %) | 0 (0.0 %) | 11070 | re_used = 100 % |
| 600 | 138401 (99.3 %) | 0 (0.0 %) | 1039 | already_resident = 64 |

Warm-up runs (16 tok): 14.5 s (1.10 tok/s) @1850, 14.6 s (1.09 tok/s) @600 — page/VRAM cache priming.

### Conclusion

- **Target met**: lowering `AIBOX_MOE_RESERVE_MB` 1850 → 600 raises steady-state VRAM from **90.2 %** to **96.1 % (5901 / 6141 MiB)**, with **no OOM** and the container still alive. `vram_cap` scales 1817 → 2523 experts.
- **Disk misses cut 10.7×** (11070 → 1039) — the extra VRAM-resident experts eliminate most cold re-reads, and prefetch turns from `re_used = 100 %` to `already_resident = 64`.
- **But throughput barely moves**: 15.66 → 15.92 tok/s (**+1.7 %**). At steady state on the fast named volume, decode is **expert-compute-bound, not disk-bound** — the removed misses (≈11070 × ~1.9 ms would be ≈21 s if strictly serial on the critical path) are largely hidden behind compute. This matches FEAT-026's finding that `expert-load` domination collapses once caching is adequate.
- **Practical takeaway**: for **latency/tok-s**, the default `1850` reserve is already near-optimal and safer (more headroom for CUDA fragmentation / other consumers). Lowering it is only worth it to **maximise cache residency** (e.g. more concurrent sequences, or hosts where disk is the real bottleneck). Note the observed VRAM (~90 %) exceeds the ticket's `total − reserve` prediction (~70 %) because `free_bytes` at init was 5072 MiB, not 6141.

---

## Changelog
| Date | From | To | Author | Note |
|---|---|---|---|---|
| 2026-11-19 | - | TODO | TRAE | Created. Investigate pushing VRAM usage toward ~90 % by lowering `AIBOX_MOE_RESERVE_MB` (default 1850) to ~600; measure VRAM + tok/s on the fast named volume. |
| 2026-11-19 | TODO | IN_PROGRESS | TRAE | Running A/B (`RESERVE_MB` 1850 vs 600) on `aibox-g4srv` with `AIBOX_MOE_STATS=1`, `AIBOX_MOE_PREFETCH=64`, model on `aibox-models-vol`. |
| 2026-11-19 | IN_PROGRESS | DONE | TRAE | A/B complete. `RESERVE_MB=1850` → `vram_cap=1817`, VRAM 5535 MiB (90.2 %), `disk_misses=11070`, 32.7 s / 15.66 tok/s. `RESERVE_MB=600` → `vram_cap=2523`, VRAM 5901 MiB (96.1 %), `disk_misses=1039`, 32.2 s / 15.92 tok/s, no OOM. Target (≥85 % VRAM) met; disk misses cut 10.7× but tok/s only +1.7 % ⇒ decode is expert-compute-bound, not disk-bound, at steady state. |
