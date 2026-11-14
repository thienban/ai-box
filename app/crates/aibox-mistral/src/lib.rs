//! KERN-07 : Adaptateur `AiboxTernaryLinear` reliant `aibox-kernels` à `mistral.rs` / `candle`.

use aibox_kernels::tensor::TernaryTensor;
use aibox_kernels::{BackendError, TernaryHadamardBackend, TmacCpuBackend};

/// Couche linéaire ternaire prête à être encapsulée dans `mistralrs_quant::QuantMethod`.
pub struct AiboxTernaryLinear {
    tensor: TernaryTensor,
    backend: TmacCpuBackend,
}

impl AiboxTernaryLinear {
    pub fn new(tensor: TernaryTensor) -> Self {
        Self {
            tensor,
            backend: TmacCpuBackend::new(),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.tensor.n_rows, self.tensor.k_cols)
    }

    pub fn uses_avx2(&self) -> bool {
        self.backend.uses_avx2()
    }

    /// Exécute `Y = (X * H_128) @ (W_ternary * scale)^T` sur un tenseur aplati `[m, k]`.
    pub fn forward_f32(&self, activations: &[f32], m: usize) -> Result<Vec<f32>, BackendError> {
        self.backend.matmul_hadamard_ternary(
            activations,
            &self.tensor.packed_weights,
            &self.tensor.scales_fp16,
            m,
            self.tensor.n_rows,
            self.tensor.k_cols,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows};
    use aibox_kernels::GROUP_SIZE;

    #[test]
    fn linear_adapter_executes_ternary_forward() {
        let n_rows = 32;
        let k_cols = 256;
        let weights: Vec<i8> = (0..(n_rows * k_cols))
            .map(|idx| match idx % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let packed = pack_ternary_2bit_rows(&weights, n_rows, k_cols).unwrap();
        let scales = vec![f32_to_f16(0.25); n_rows * (k_cols / GROUP_SIZE)];
        let tensor = TernaryTensor::from_raw(n_rows, k_cols, packed, scales).unwrap();

        let linear = AiboxTernaryLinear::new(tensor);
        assert_eq!(linear.shape(), (n_rows, k_cols));

        let batch_seq = 2;
        let act = vec![0.5f32; batch_seq * k_cols];
        let out = linear.forward_f32(&act, batch_seq).expect("forward failed");
        assert_eq!(out.len(), batch_seq * n_rows);
    }
}
