# AI Box — Backend Rust (`aibox-kernels`)

Workspace Rust implémentant les tickets **KERN-01 à KERN-03** de [spec_technique_kernels_tmac_lds.md](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/docs/dev/spec_technique_kernels_tmac_lds.md) :

- **KERN-01** (`src/tensor.rs`) : `TernaryTensor`, packing/unpacking 2-bit (`00 -> 0`, `01 -> +1`, `11 -> -1`, 4 poids/octet LSB-first), conversion IEEE-754 `f16 <-> f32`, validation d'alignement 32 octets (`group_size = 128`).
- **KERN-02** (`src/fwht.rs`, `src/scalar.rs`) : `fwht_128_inplace` (Fast Walsh-Hadamard Transform normalisée) et `ScalarBackend` implémentant `TernaryHadamardBackend`.
- **KERN-03** (`src/avx2.rs`) : `Avx2Backend` utilisant `_mm256_shuffle_epi8` (`pshufb`) + LUTs `i16` par paires d'activations, avec détection automatique runtime via `TmacCpuBackend::new()`.

## Commandes

```powershell
# Installer Rust si absent (Windows) : https://rustup.rs/
cargo test
cargo bench
```
