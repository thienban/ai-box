# aibox (`tiny-gpu`) — Rust + CUDA Inference Engine for 1-bit / Ternary LLMs

A lightweight local LLM inference engine written in **Rust** and **CUDA**, designed to run large language models (**8B to 27B parameters**) on **consumer GPUs with limited VRAM (6 GB)** such as an NVIDIA RTX 4050 Laptop.

---

## 1. The Problem

Single-batch (`batch = 1`) LLM decoding is **memory-bandwidth bound** and strictly constrained by **VRAM capacity**:
- An **8B** model in FP16 takes ~16 GB, and a **27B** model exceeds 54 GB — neither fits on a **6 GB VRAM** GPU (`6,141 MiB`, ~5.4 GB free under Windows).
- Even at standard 4-bit quantization, a 27B model requires ~14+ GB and forces heavy paging to system RAM over PCIe, collapsing throughput to a few tokens per second.
- Modern hybrid architectures (such as **Qwen 3.5 / Gated DeltaNet + Attention**) also require native support for recurrent state (`conv_state`, `ssm_state`) alongside standard KV caching.

---

## 2. The Solution

**`aibox`** solves these constraints with a specialized **1-bit (`binary_1bit`) / ternary (`1.58-bit`)** inference engine combining fused CUDA kernels and zero-overhead graph execution:
- **100% local execution** in a single Rust binary (`aibox-cli`), with no Python runtime dependency.
- **Multi-architecture support**:
  - `qwen3` (standard GQA Attention — e.g., `Bonsai-8B`)
  - `qwen35` (hybrid **Gated DeltaNet SSM + Full Attention** — e.g., `Bonsai-27B`)
- **Dual backend**: **CUDA** GPU (`libaibox_gpu.so`) with **CPU AVX2** fallback.

---

## 3. Optimization Techniques & Methods Used

| Technique | Implementation Details | Ticket |
|---|---|---|
| **Native 1-bit & ternary weights** | CUDA kernels read packed weights (`bits = 1` or `2`) directly with FP16 group-128 scales (`group_size = 128`), avoiding intermediate FP16 dequantization. | [FEAT-004](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-004-native-1bit-q1-0-kernel.md) |
| **Q4 `lm_head` & CPU-resident embeddings** | `lm_head` is quantized to Q4 (~636 MB for `vocab_size = 248,320`) with a dedicated kernel, while `token_embeddings` stays in CPU RAM (only the current token row is read). | [FEAT-006](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-006-quantized-lm-head.md) |
| **FP16 activations & KV Cache** | Halves GPU activation and KV cache memory footprint and bandwidth compared to FP32. | [FEAT-003](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-003-fp16-activations-and-kv-cache.md) |
| **CUDA kernel fusion** | Fused `RMSNorm + RoPE` (full or partial), fused `SwiGLU` on FFN, and on-GPU `argmax` to avoid GPU → CPU logits transfers. | [FEAT-001](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-001-cuda-kernel-fusion-and-argmax.md), [FEAT-007](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-007-decode-60-toks-optimizations.md) |
| **CUDA Graph decode loop** | Captures the full layer stack (`Attention` / `GatedDeltaNet` + `FFN` + `lm_head`) into a `cudaGraphExec_t` to eliminate per-layer CPU launch latency. | [FEAT-002](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-002-cuda-graph-decode-loop.md) |
| **Batched prefill GEMM** | Tiled matrix multiplication (`m > 1`) during prompt prefill instead of sequential GEMV. | [FEAT-005](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-005-batched-prefill-gemm.md) |
| **Hybrid Gated DeltaNet (`qwen35`)** | Dedicated `gdn_conv_silu_f16_kernel` and `gdn_step_f16_kernel` for the 48 SSM layers in `Bonsai-27B`. | [FEAT-008](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-008-qwen35-gated-deltanet-ssm-bonsai27b.md) |

---

## 4. Measured Benchmarks (RTX 4050 Laptop — 6 GB VRAM, 192 GB/s)

| Model | Architecture | Disk Size (`safetensors`) | Decode Throughput (`--gpu`) |
|---|---|---|---|
| `bonsai-8b-1bit-q4head` | `qwen3` (36 layers) | `2,422.5 MB` | **`76.83 tok/s`** |
| `bonsai-27b-1bit-q4head` | `qwen35` (48 GDN + 16 Attention) | `6,362.9 MB` *(peak `5,114 MiB` VRAM)* | **`20.27 tok/s`** |

> **Active optimization track**: [FEAT-009](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-009-optimize-gdn-step-ssm-state.md) targets eliminating the redundant `ssm_state` write in `gdn_step_f16_kernel` and evaluating `fp16` storage (~75 MB VRAM saved).

---

## 5. Project Structure

- [app/crates/aibox-cli](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli): CLI, HTTP server, tokenizer, and inference engine ([engine.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/engine.rs)).
- [app/crates/aibox-kernels](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels): CPU (AVX2) kernels and CUDA FFI wrappers ([gpu.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/src/gpu.rs), [aibox_ternary_gpu.cu](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu)).
- [app/crates/aibox-mistral](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-mistral): Mistral compatibility crate.
- [scripts/convert_gguf_to_aibox.py](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/convert_gguf_to_aibox.py): GGUF to `aibox` `safetensors` + `config.json` / `quant_config.json` converter.
- [.tickets/](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets): Development & optimization tickets (`FEAT-001` through `FEAT-009`).
- [docs/](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/docs): Architecture docs, benchmarks, and research references ([arxiv_references.md](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/docs/arxiv_references.md)).

---

## 6. Build & Run

### Unit Tests (Windows / Linux)
```powershell
cargo test --workspace --manifest-path app/Cargo.toml
```

### CUDA Build & GPU Inference (Docker)
```powershell
docker run --rm --gpus all `
  -v "${PWD}:/work" `
  -v "${env:USERPROFILE}\.aibox\models:/models:ro" `
  -w /work/app nvidia/cuda:12.4.1-devel-ubuntu22.04 bash -c '
    cargo build --release &&
    nvcc -O3 -arch=sm_89 --shared -Xcompiler -fPIC /work/app/crates/aibox-kernels/cuda/aibox_ternary_gpu.cu -o /work/app/target/release/libaibox_gpu.so &&
    LD_LIBRARY_PATH=/work/app/target/release ./target/release/aibox run /models/bonsai-8b-1bit-q4head Hello --max-tokens 32 --gpu
  '
```
