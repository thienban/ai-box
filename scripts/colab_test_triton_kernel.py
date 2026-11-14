# %% [markdown]
# # Prototype Colab (KERN-05) : Kernel Triton Fusionné (Hadamard + Ternaire 2-bit + GEMV)
#
# Ce script permet de tester et benchmarker sur **Google Colab (GPU T4 gratuit)**
# ou sur une **NVIDIA RTX / AMD ROCm** le kernel fusionné décrit dans `spec_technique_kernels_tmac_lds.md`.
#
# **Ce que fait le kernel en 1 seul passage GPU :**
# 1. Charge un groupe de 128 activations X
# 2. Applique la rotation de Hadamard (X * H) sans repasser par la VRAM
# 3. Lit et décode à la volée les poids 2-bit (`00=0, 01=+1, 11=-1`)
# 4. Multiplie par l'échelle `FP16` (g128) et accumule le résultat

# %%
# ==============================================================================
# CELLULE 1 : Imports & Vérification GPU
# ==============================================================================
import math
import torch
import triton
import triton.language as tl

assert torch.cuda.is_available(), "Un GPU CUDA (Colab T4/L4/A100 ou RTX) est requis."
print(f"GPU détecté : {torch.cuda.get_device_name(0)}")

# %%
# ==============================================================================
# CELLULE 2 : Utilitaires PyTorch (Hadamard + Packing 2-bit)
# ==============================================================================
def build_hadamard_matrix(block_size: int, device: torch.device) -> torch.Tensor:
    """Construit la matrice orthogonale de Walsh-Hadamard (K x K)."""
    h = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    while h.shape[0] < block_size:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(block_size)


def pack_ternary_2bit_rows(w_ternary: torch.Tensor) -> torch.Tensor:
    """Compacte une matrice [N, K] de {-1, 0, +1} en [N, K // 4] uint8."""
    n_rows, k_cols = w_ternary.shape
    assert k_cols % 4 == 0
    codes = torch.zeros_like(w_ternary, dtype=torch.uint8)
    codes[w_ternary == 1] = 1
    codes[w_ternary == -1] = 3
    codes = codes.reshape(n_rows, k_cols // 4, 4)
    return codes[..., 0] | (codes[..., 1] << 2) | (codes[..., 2] << 4) | (codes[..., 3] << 6)

# %%
# ==============================================================================
# CELLULE 3 : Kernel Triton Fusionné (Decode GEMV : M=1, N, K)
# ==============================================================================
@triton.jit
def hadamard_ternary_gemv_kernel(
    x_ptr,          # [K] float16/float32
    packed_w_ptr,   # [N, K // 4] uint8
    scales_ptr,     # [N, K // 128] float16
    h_ptr,          # [128, 128] float32
    y_ptr,          # [N] float32
    N,
    K,
    stride_wn,
    stride_sn,
    BLOCK_N: tl.constexpr,
):
    """Calcule Y = (X @ H) @ (W_ternary * scale)^T en un seul passage mémoire (optimisé)."""
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    offs_k128 = tl.arange(0, 128)
    offs_bytes = tl.arange(0, 32)
    shifts = tl.arange(0, 4) * 2

    # Charge la matrice de Hadamard UNE SEULE FOIS avant la boucle sur K
    h_tile = tl.load(h_ptr + offs_k128[:, None] * 128 + offs_k128[None, :])
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    num_groups = K // 128
    for g in range(0, num_groups):
        k_base = g * 128

        # 1. Rotation Hadamard (X_g @ H)
        x_g = tl.load(x_ptr + k_base + offs_k128).to(tl.float32)
        x_rot = tl.sum(x_g[:, None] * h_tile, axis=0)  # [128]

        # 2. Charge 32 octets et décode les 128 poids ternaires en une passe vectorisée
        byte_ptrs = packed_w_ptr + offs_n[:, None] * stride_wn + (g * 32 + offs_bytes[None, :])
        packed_bytes = tl.load(byte_ptrs, mask=mask_n[:, None], other=0)
        codes = (packed_bytes[:, :, None] >> shifts[None, None, :]) & 0x3
        codes = tl.reshape(codes, (BLOCK_N, 128))
        w_tile = tl.where(codes == 3, -1.0, tl.where(codes == 1, 1.0, 0.0))

        # 3. Produit scalaire & échelle FP16
        scale_g = tl.load(scales_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.sum(w_tile * x_rot[None, :], axis=1) * scale_g

    tl.store(y_ptr + offs_n, acc, mask=mask_n)


def run_triton_hadamard_ternary(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    h_mat: torch.Tensor,
) -> torch.Tensor:
    """Wrapper Python lançant le kernel Decode Triton."""
    n_rows, k_packed = packed_w.shape
    k_cols = k_packed * 4
    y = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    block_n = 64
    grid = (triton.cdiv(n_rows, block_n),)

    hadamard_ternary_gemv_kernel[grid](
        x,
        packed_w,
        scales,
        h_mat,
        y,
        n_rows,
        k_cols,
        packed_w.stride(0),
        scales.stride(0),
        BLOCK_N=block_n,
        num_warps=4,
    )
    return y


@triton.jit
def hadamard_ternary_gemm_kernel(
    x_rot_ptr,      # [M, K] float16 (déjà tourné par blocs de 128)
    packed_w_ptr,   # [N, K // 4] uint8
    scales_ptr,     # [N, K // 128] float16
    y_ptr,          # [M, N] float32
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_sn,
    stride_ym,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Kernel Prefill (M >= 16) : Décodage 2-bit + Tensor Core tl.dot (tient sous 24 Ko SRAM)."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_k128 = tl.arange(0, 128)
    offs_bytes = tl.arange(0, 32)
    shifts = tl.arange(0, 4) * 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_groups = K // 128

    for g in range(0, num_groups):
        k_base = g * 128
        x_ptrs = x_rot_ptr + offs_m[:, None] * stride_xm + (k_base + offs_k128[None, :])
        x_rot = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float16)

        # Décodage vectorisé sans boucle : [BLOCK_N, 32] -> [BLOCK_N, 128]
        byte_ptrs = packed_w_ptr + offs_n[:, None] * stride_wn + (g * 32 + offs_bytes[None, :])
        packed_bytes = tl.load(byte_ptrs, mask=mask_n[:, None], other=0)
        codes = (packed_bytes[:, :, None] >> shifts[None, None, :]) & 0x3
        codes = tl.reshape(codes, (BLOCK_N, 128))
        w_tile = tl.where(codes == 3, -1.0, tl.where(codes == 1, 1.0, 0.0)).to(tl.float16)

        scale_g = tl.load(scales_ptr + offs_n * stride_sn + g, mask=mask_n, other=0.0).to(tl.float32)
        acc += tl.dot(x_rot, tl.trans(w_tile)) * scale_g[None, :]

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def run_triton_hadamard_ternary_prefill(
    x: torch.Tensor,
    packed_w: torch.Tensor,
    scales: torch.Tensor,
    h_mat_fp16: torch.Tensor,
) -> torch.Tensor:
    """Applique la rotation Hadamard 1 fois sur X puis lance le kernel Prefill GEMM."""
    m_rows, k_cols = x.shape
    n_rows = packed_w.shape[0]
    x_rot = torch.matmul(x.reshape(m_rows, k_cols // 128, 128), h_mat_fp16).reshape(m_rows, k_cols).contiguous()

    y = torch.empty((m_rows, n_rows), device=x.device, dtype=torch.float32)
    block_m, block_n = 32, 32
    grid = (triton.cdiv(m_rows, block_m), triton.cdiv(n_rows, block_n))

    hadamard_ternary_gemm_kernel[grid](
        x_rot,
        packed_w,
        scales,
        y,
        m_rows,
        n_rows,
        k_cols,
        x_rot.stride(0),
        packed_w.stride(0),
        scales.stride(0),
        y.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    return y

# %%
# ==============================================================================
# CELLULE 4 : Validation Numérique (Golden Reference) & Benchmark
# ==============================================================================
def test_accuracy_and_benchmark() -> None:
    """Vérifie la précision contre PyTorch et mesure la latence en mode Decode."""
    device = torch.device("cuda")
    N, K = 4096, 4096
    group_size = 128

    torch.manual_seed(42)
    x = torch.randn(K, device=device, dtype=torch.float16)
    h_mat = build_hadamard_matrix(group_size, device)

    # Génération de poids ternaires aléatoires {-1, 0, +1} + échelles FP16
    w_ternary = torch.randint(-1, 2, (N, K), device=device, dtype=torch.int8)
    scales = torch.rand((N, K // group_size), device=device, dtype=torch.float16) * 0.1 + 0.01

    packed_w = pack_ternary_2bit_rows(w_ternary)

    # Référence PyTorch exacte : Y = (X @ H) @ (W_ternary * scale)^T
    x_rot_ref = (x.float().reshape(K // group_size, group_size) @ h_mat).reshape(K)
    w_dequant_ref = (
        w_ternary.float().reshape(N, K // group_size, group_size) * scales.float().unsqueeze(-1)
    ).reshape(N, K)
    y_ref = torch.matmul(w_dequant_ref, x_rot_ref)

    # Exécution Kernel Triton
    y_triton = run_triton_hadamard_ternary(x, packed_w, scales, h_mat)

    max_err = torch.max(torch.abs(y_ref - y_triton)).item()
    print(f"Erreur maximale vs PyTorch FP32 : {max_err:.6f}")
    assert max_err < 1e-2, f"Erreur trop élevée : {max_err}"
    print("Validation numérique RÉUSSIE !")

    # Benchmark Decode (M=1)
    w_fp16 = w_dequant_ref.half()
    ms_pytorch = triton.testing.do_bench(lambda: torch.matmul(w_fp16, x))
    ms_triton = triton.testing.do_bench(lambda: run_triton_hadamard_ternary(x, packed_w, scales, h_mat))
    print(f"Latence PyTorch FP16 (4096x4096) : {ms_pytorch:.4f} ms")
    print(f"Latence Triton 2-bit + Hadamard  : {ms_triton:.4f} ms")


test_accuracy_and_benchmark()

# %%
# ==============================================================================
# CELLULE 5 : Test Complet Bonsai 27B (Inspection GGUF + Prefill/Decode)
# ==============================================================================
# Optionnel dans Colab : !pip install -q gguf

def inspect_bonsai_gguf_spec(gguf_path: str | None = None) -> tuple[int, list[tuple[str, int, int]]]:
    """Lit les vraies dimensions de Bonsai 27B depuis un GGUF, ou utilise les specs 27B."""
    if gguf_path:
        from gguf import GGUFReader  # type: ignore

        reader = GGUFReader(gguf_path)
        layer_shapes: list[tuple[str, int, int]] = []
        n_layers = 64
        for tensor in reader.tensors:
            if tensor.name.startswith("blk.0.") and tensor.name.endswith(".weight"):
                if len(tensor.shape) == 2:
                    n_out, k_in = int(tensor.shape[1]), int(tensor.shape[0])
                    layer_shapes.append((tensor.name, n_out, k_in))
        print(f"GGUF lu ({gguf_path}) : {len(layer_shapes)} matrices linéaires dans blk.0")
        return n_layers, layer_shapes

    # Dimensions typiques d'un bloc Transformer 27B (type Qwen3.8-27B / Bonsai 27B)
    default_shapes = [
        ("attn_q", 5120, 5120),
        ("attn_k", 1024, 5120),
        ("attn_v", 1024, 5120),
        ("attn_o", 5120, 5120),
        ("ffn_gate", 17408, 5120),
        ("ffn_up", 17408, 5120),
        ("ffn_down", 5120, 17408),
    ]
    return 64, default_shapes


def benchmark_full_bonsai_27b(gguf_path: str | None = None, prefill_tokens: int = 128) -> None:
    """Benchmarke toutes les projections d'une couche 27B et projette sur les 64 couches."""
    device = torch.device("cuda")
    group_size = 128
    n_layers, block_shapes = inspect_bonsai_gguf_spec(gguf_path)

    h_mat_fp32 = build_hadamard_matrix(group_size, device)
    h_mat_fp16 = h_mat_fp32.half()

    total_decode_ms = 0.0
    total_prefill_ms = 0.0

    print(f"\n--- Benchmark Bonsai 27B ({n_layers} couches, Prefill={prefill_tokens} tokens) ---")
    for name, n_out, k_in in block_shapes:
        if k_in % group_size != 0:
            continue

        x_dec = torch.randn(k_in, device=device, dtype=torch.float16)
        x_pre = torch.randn((prefill_tokens, k_in), device=device, dtype=torch.float16)
        w_ternary = torch.randint(-1, 2, (n_out, k_in), device=device, dtype=torch.int8)
        scales = torch.ones((n_out, k_in // group_size), device=device, dtype=torch.float16) * 0.05
        packed_w = pack_ternary_2bit_rows(w_ternary)

        ms_dec = triton.testing.do_bench(lambda: run_triton_hadamard_ternary(x_dec, packed_w, scales, h_mat_fp32))
        ms_pre = triton.testing.do_bench(
            lambda: run_triton_hadamard_ternary_prefill(x_pre, packed_w, scales, h_mat_fp16)
        )
        total_decode_ms += ms_dec
        total_prefill_ms += ms_pre
        print(f"  {name:12s} ({n_out:5d} x {k_in:5d}) | Decode: {ms_dec:.3f} ms | Prefill({prefill_tokens}): {ms_pre:.3f} ms")

    full_decode_ms = total_decode_ms * n_layers
    full_prefill_ms = total_prefill_ms * n_layers
    print(f"\nProjection Modèle Complet ({n_layers} couches linéaires) :")
    print(f"  Decode  estimé : {full_decode_ms:.1f} ms/token -> {1000.0 / full_decode_ms:.2f} tokens/s")
    print(f"  Prefill estimé : {full_prefill_ms:.1f} ms/{prefill_tokens} tok -> {prefill_tokens * 1000.0 / full_prefill_ms:.2f} tokens/s")


# Passer un chemin .gguf réel si vous l'avez téléchargé, sinon None utilise les specs 27B :
benchmark_full_bonsai_27b(gguf_path=None, prefill_tokens=128)
