//! Kernels CPU T-MAC (Scalaire & AVX2) et transformation Hadamard (FWHT)
//! pour l'inférence ternaire 2-bit (`{-1, 0, +1}`) par groupes de 128 poids.

pub mod fwht;
pub mod scalar;
pub mod tensor;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;

use std::fmt;

/// Nombre de poids ternaires par groupe de quantification.
pub const GROUP_SIZE: usize = 128;

/// Nombre d'octets packés (4 poids/octet) pour un groupe de 128 poids.
pub const PACKED_BYTES_PER_GROUP: usize = GROUP_SIZE / 4; // 32 octets

/// Erreurs retournées par les loaders et backends de calcul.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    InvalidDimensions(String),
    BufferLengthMismatch(String),
    InvalidTernaryValue(i8),
    UnsupportedFeature(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {msg}"),
            Self::BufferLengthMismatch(msg) => write!(f, "Buffer length mismatch: {msg}"),
            Self::InvalidTernaryValue(val) => {
                write!(f, "Invalid ternary weight {val}: expected -1, 0, or 1")
            }
            Self::UnsupportedFeature(msg) => write!(f, "Unsupported CPU feature: {msg}"),
        }
    }
}

impl std::error::Error for BackendError {}

/// Interface unifiée ("Interface First") pour tout backend matériel (CPU/GPU).
pub trait TernaryHadamardBackend: Send + Sync {
    /// Exécute `Y = (X * H_128) @ (W_ternary * scale_fp16)^T`.
    ///
    /// - `activations`: `[m, k]` en row-major (`f32`)
    /// - `packed_weights`: `[n, k / 4]` (`u8`, 4 poids/octet LSB-first)
    /// - `scales_fp16`: `[n, k / 128]` (`u16` IEEE-754 half)
    fn matmul_hadamard_ternary(
        &self,
        activations: &[f32],
        packed_weights: &[u8],
        scales_fp16: &[u16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, BackendError>;

    /// Exécute la multiplication ternaire avec ou sans rotation Hadamard `H_128`.
    fn matmul_ternary(
        &self,
        activations: &[f32],
        packed_weights: &[u8],
        scales_fp16: &[u16],
        m: usize,
        n: usize,
        k: usize,
        use_hadamard: bool,
    ) -> Result<Vec<f32>, BackendError> {
        if use_hadamard {
            self.matmul_hadamard_ternary(activations, packed_weights, scales_fp16, m, n, k)
        } else {
            Err(BackendError::UnsupportedFeature(
                "Plain ternary matmul (hadamard=false) not implemented for this backend".to_string(),
            ))
        }
    }
}

/// Valide la géométrie d'un tenseur ternaire par groupes de 128 poids.
pub(crate) fn validate_ternary_geometry(
    rows: usize,
    cols: usize,
    packed_len: Option<usize>,
    scales_len: Option<usize>,
) -> Result<(), BackendError> {
    if rows == 0 || cols == 0 {
        return Err(BackendError::InvalidDimensions(format!(
            "rows={rows}, cols={cols} must both be non-zero"
        )));
    }
    if cols % GROUP_SIZE != 0 {
        return Err(BackendError::InvalidDimensions(format!(
            "cols={cols} must be a multiple of GROUP_SIZE ({GROUP_SIZE})"
        )));
    }
    if let Some(len) = packed_len {
        let expected_w = rows * (cols / GROUP_SIZE) * PACKED_BYTES_PER_GROUP;
        if len != expected_w {
            return Err(BackendError::BufferLengthMismatch(format!(
                "packed_weights len {len} != expected {expected_w}"
            )));
        }
    }
    if let Some(len) = scales_len {
        let expected_s = rows * (cols / GROUP_SIZE);
        if len != expected_s {
            return Err(BackendError::BufferLengthMismatch(format!(
                "scales_fp16 len {len} != expected {expected_s}"
            )));
        }
    }
    Ok(())
}

/// Valide les dimensions et tailles de tampons avant exécution d'un kernel.
pub(crate) fn validate_matmul_inputs(
    activations: &[f32],
    packed_weights: &[u8],
    scales_fp16: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), BackendError> {
    if m == 0 {
        return Err(BackendError::InvalidDimensions(
            "m must be non-zero".to_string(),
        ));
    }
    validate_ternary_geometry(n, k, Some(packed_weights.len()), Some(scales_fp16.len()))?;
    let expected_act = m * k;
    if activations.len() != expected_act {
        return Err(BackendError::BufferLengthMismatch(format!(
            "activations len {} != expected {expected_act}",
            activations.len()
        )));
    }
    Ok(())
}

/// Backend CPU principal sélectionnant automatiquement AVX2 (`pshufb`) ou Scalaire.
#[derive(Debug, Clone, Copy, Default)]
pub struct TmacCpuBackend {
    force_scalar: bool,
}

impl TmacCpuBackend {
    /// Crée un backend avec détection automatique des instructions SIMD au runtime.
    pub fn new() -> Self {
        Self { force_scalar: false }
    }

    /// Force l'utilisation du backend scalaire de référence (utile pour tests/benchmarks).
    pub fn scalar_only() -> Self {
        Self { force_scalar: true }
    }

    /// Indique si l'accélération AVX2 sera utilisée sur cette machine.
    pub fn uses_avx2(&self) -> bool {
        if self.force_scalar {
            return false;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            return is_x86_feature_detected!("avx2");
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            false
        }
    }
}

impl TernaryHadamardBackend for TmacCpuBackend {
    fn matmul_hadamard_ternary(
        &self,
        activations: &[f32],
        packed_weights: &[u8],
        scales_fp16: &[u16],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, BackendError> {
        self.matmul_ternary(activations, packed_weights, scales_fp16, m, n, k, true)
    }

    fn matmul_ternary(
        &self,
        activations: &[f32],
        packed_weights: &[u8],
        scales_fp16: &[u16],
        m: usize,
        n: usize,
        k: usize,
        use_hadamard: bool,
    ) -> Result<Vec<f32>, BackendError> {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if self.uses_avx2() {
            return avx2::Avx2Backend.matmul_ternary(
                activations,
                packed_weights,
                scales_fp16,
                m,
                n,
                k,
                use_hadamard,
            );
        }

        scalar::ScalarBackend.matmul_ternary(
            activations,
            packed_weights,
            scales_fp16,
            m,
            n,
            k,
            use_hadamard,
        )
    }
}
