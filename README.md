# aibox (`tiny-gpu`) — Rust + CUDA Inference Engine for 1-bit / 2-bit / 4-bit LLMs

A lightweight local LLM inference engine written in **Rust** and **CUDA**, designed to run large language models (**8B to 27B dense, hybrid SSM, and 26B MoE**) on **consumer GPUs with 6 GB VRAM** such as an NVIDIA RTX 4050 Laptop.

---

## 1. The Problem

Single-batch (`batch = 1`) LLM decoding is **memory-bandwidth bound** and strictly constrained by **VRAM capacity**:
- An **8B** model in FP16 takes ~16 GB, and a **26B–27B** model exceeds 52 GB — neither fits on a **6 GB VRAM** GPU (`6,141 MiB`, ~5.4 GB free under Windows).
- Even at 4-bit quantization, a 26B–27B model requires ~14+ GB and forces heavy PCIe/disk paging, collapsing throughput below `1 tok/s`.
- Modern architectures (**Qwen 3.5 Gated DeltaNet**, **Gemma 4 iSWA + 128-expert MoE**) also require native SSM state, proportional RoPE, sliding-window KV caching, and dynamic expert streaming.

---

## 2. The Solution

**`aibox`** combines fused CUDA kernels, zero-overhead graph execution, and selective VRAM/RAM expert caching in a single **100% local Rust binary (`aibox-cli`)** with zero Python runtime dependency:
- **Multi-architecture support**:
  - `qwen3` (standard GQA Attention — e.g., `Bonsai-8B`)
  - `qwen3.5` (hybrid **Gated DeltaNet SSM + Full Attention** — e.g., `Bonsai-27B`)
  - `gemma4` (hybrid **iSWA + Global Attention + 128-expert MoE + Shared Dense FFN** — e.g., `Gemma-4-26B-A4B`)
- **Dual backend**: **CUDA** GPU (`libaibox_gpu.so`) with **CPU AVX2** fallback.

---

## 3. Optimization Techniques & Methods Used

| Technique | Implementation Details | Tickets |
|---|---|---|
| **Native 1-bit & ternary weights** | CUDA kernels read packed weights (`bits = 1` or `2`) directly with FP16 group-128 scales (`group_size = 128`). | [FEAT-004](.tickets/FEAT-004-native-1bit-q1-0-kernel.md) |
| **Mixed-precision MoE (`Q4` + `Q2`)** | Attention, shared FFN, and embeddings stay in **`q4_affine`** (VRAM-resident) while 3,840 MoE experts use **`q2_affine`** (`1.77 MiB`/expert) so **2,003 / 3,840** experts fit in 6 GB VRAM. | [FEAT-012](.tickets/FEAT-012-gemma4-moe-2bit-affine.md), [FEAT-014](.tickets/FEAT-014-gemma4-mmap-async-prefetch-and-mixed-precision.md) |
| **Exclusive VRAM ↔ RAM Victim Cache** | Evicted VRAM experts move to host RAM (`ExpertRamLru`) without duplication, delivering **~89–92% VRAM hit rate**. | [FEAT-016](.tickets/FEAT-016-gemma4-exclusive-victim-cache.md) |
| **Q4 `lm_head` & CPU embeddings** | Quantized `lm_head` kernel reduces footprint & bandwidth; `token_embeddings` stays in CPU RAM. | [FEAT-006](.tickets/FEAT-006-quantized-lm-head.md) |
| **CUDA kernel fusion & VRAM Attention** | Fused `RMSNorm + RoPE`, `SwiGLU` / `GeGLU`, grouped MoE kernels (`grouped_q2_affine_gemv_geglu` + `down_acc`), on-GPU `argmax`, and 100% VRAM Attention/KV cache. | [FEAT-001](.tickets/FEAT-001-cuda-kernel-fusion-and-argmax.md), [FEAT-007](.tickets/FEAT-007-decode-60-toks-optimizations.md), [FEAT-017](.tickets/FEAT-017-gemma4-gpu-attention-kv-and-fused-geglu.md) |
| **CUDA Graph & Hybrid GDN (`qwen35`)** | Full-stack `cudaGraphExec_t` capture + dedicated `gdn_conv_silu_f16` / `gdn_step_f16` SSM kernels. | [FEAT-002](.tickets/FEAT-002-cuda-graph-decode-loop.md), [FEAT-008](.tickets/FEAT-008-qwen35-gated-deltanet-ssm-bonsai27b.md) |
| **Batched Prefill (`M > 1`)** | Tiled GEMM prefill for dense/SSM models and `expert_idx`-grouped batched prefill for Gemma 4 MoE (`-36.8%` prefill latency). | [FEAT-005](.tickets/FEAT-005-batched-prefill-gemm.md), [FEAT-018](.tickets/FEAT-018-gemma4-batched-prefill.md) |

---

## 4. Measured Benchmarks (RTX 4050 Laptop — 6 GB VRAM, 192 GB/s)

| Model | Architecture | Disk Size | Peak / Budget | Decode Throughput (`--gpu`) |
|---|---|---:|---|---|
| **`bonsai-8b-1bit-q4head`** | `qwen3` (36 layers, 1-bit) | `2,422.5 MB` | 100% VRAM-resident | **`76.83 tok/s`** |
| **`bonsai-27b-1bit-q4head`** | `qwen35` (48 GDN + 16 Attn, 1-bit) | `6,362.9 MB` | `5,114 MiB` VRAM | **`20.27 tok/s`** |
| **`gemma-4-26b-q4-moe2`** (`top_k=8`) | `gemma4` (30L, 128 exp/L, `Q4+Q2`) | `~7.5 GB` (4 shards) | `2,003 / 3,840` exp VRAM (`89.4%` hits) | **`10.59 tok/s`** *(420-tok run)* |
| **`gemma-4-26b-q4-moe2`** (`top_k=6`) | `gemma4` (30L, 128 exp/L, `Q4+Q2`) | `~7.5 GB` (4 shards) | `2,003 / 3,840` exp VRAM (`89.8%` hits) | **`11.04 tok/s`** *(**58×** vs `0.19 tok/s` baseline)* |

> Full Gemma 4 MoE journey & long-scenario logs: [benchmark_tickets_feat012_to_feat018.md](docs/benchmark/benchmark_tickets_feat012_to_feat018.md) and [architecture-gemma4-streaming.md](docs/architecture/architecture-gemma4-streaming.md).

### How to Save \$1,000,000 😉💸
- **Step 1**: Do **not** buy 8× H100s (\$300k+), a liquid-cooled rack, or a second mortgage on cloud GPU bills just to run a 27B model.
- **Step 2**: Squeeze 26B–27B parameters into a **6 GB laptop GPU** using 1-bit / 2-bit math, a ruthless VRAM/RAM eviction policy, and zero Python overhead.
- **Step 3**: Pocket the remaining **\$999,000** (and still have enough VRAM left to keep your browser open 😅).

---

## 5. Project Structure

- [app/crates/aibox-cli](app/crates/aibox-cli): CLI, HTTP server, tokenizer, and inference engines ([engine.rs](app/crates/aibox-cli/src/engine.rs), [gemma4.rs](app/crates/aibox-cli/src/engine/gemma4.rs)).
- [app/crates/aibox-kernels](app/crates/aibox-kernels): CPU (AVX2) kernels and CUDA FFI wrappers ([gpu.rs](app/crates/aibox-kernels/src/gpu.rs), [aibox_ternary_gpu.cu](app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu)).
- [app/crates/aibox-mistral](app/crates/aibox-mistral): Mistral compatibility crate.
- [scripts/convert_gguf_to_aibox.py](scripts/convert_gguf_to_aibox.py): GGUF to `aibox` `safetensors` + `config.json` / `quant_config.json` converter.
- [.tickets/](.tickets): Development & optimization tickets (`FEAT-001` through `FEAT-018`).
- [docs/](docs): Architecture docs, benchmarks, and research references ([arxiv_references.md](docs/arxiv_references.md)).

---

## 6. Build & Run

### Unit Tests (Windows / Linux)
```powershell
cargo test --workspace --manifest-path app/Cargo.toml
```

### CUDA Build & GPU Inference (Docker)
```powershell
docker run --rm --gpus all `
  -v "${PWD}/app:/app" `
  -v "${env:USERPROFILE}\.aibox\models:/models:ro" `
  -w /app nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c '
    nvcc -O3 -arch=sm_89 --shared -Xcompiler -fPIC crates/aibox-kernels/cuda/aibox_ternary_gpu.cu -o /app/target/release/libaibox_gpu.so &&
    cargo build --release &&
    AIBOX_GPU_LIB=/app/target/release/libaibox_gpu.so ./target/release/aibox run /models/gemma-4-26b-q4-moe2 Photosynthesis --max-tokens 160 --gpu
  '
```
