//! KERN-07 : Adaptateur `AiboxTernaryLinear` reliant `aibox-kernels` à `mistral.rs` / `candle`.

use aibox_kernels::tensor::TernaryTensor;
use aibox_kernels::{BackendError, TernaryHadamardBackend, TmacCpuBackend};

/// Couche linéaire ternaire prête à être encapsulée dans `mistralrs_quant::QuantMethod`.
#[derive(Debug)]
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

#[cfg(feature = "mistral")]
mod mistral_impl {
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    use aibox_kernels::tensor::{f16_to_f32, unpack_ternary_2bit_rows};
    use aibox_kernels::GROUP_SIZE;
    use candle_core::{DType, Device, Result as CandleResult, Tensor};
    use mistralrs_quant::{
        IsqPlanParams, IsqRequest, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
        QuantizedSerde,
    };

    use super::AiboxTernaryLinear;

    impl QuantizedSerde for AiboxTernaryLinear {
        fn name(&self) -> &'static str {
            "aibox_ternary_hadamard"
        }
    }

    impl QuantMethod for AiboxTernaryLinear {
        fn new(_method: QuantMethodConfig) -> CandleResult<Self> {
            candle_core::bail!("Use AiboxTernaryLinear::new(TernaryTensor) directly")
        }

        fn dequantize_w(&self) -> CandleResult<Tensor> {
            let unpacked = unpack_ternary_2bit_rows(
                &self.tensor.packed_weights,
                self.tensor.n_rows,
                self.tensor.k_cols,
            )
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?;

            let groups_per_row = self.tensor.k_cols / GROUP_SIZE;
            let mut weights_f32 = Vec::with_capacity(self.tensor.n_rows * self.tensor.k_cols);
            for (idx, &ternary_val) in unpacked.iter().enumerate() {
                let row = idx / self.tensor.k_cols;
                let group = (idx % self.tensor.k_cols) / GROUP_SIZE;
                let scale = f16_to_f32(self.tensor.scales_fp16[row * groups_per_row + group]);
                weights_f32.push((ternary_val as f32) * scale);
            }
            Tensor::from_vec(
                weights_f32,
                (self.tensor.n_rows, self.tensor.k_cols),
                &Device::Cpu,
            )
        }

        fn forward_raw(&self, a: &Tensor) -> CandleResult<Tensor> {
            let orig_dims = a.dims();
            let k = *orig_dims
                .last()
                .ok_or_else(|| candle_core::Error::Msg("Input tensor has 0 dimensions".into()))?;
            let m = a.elem_count() / k;
            let act_vec = a.flatten_all()?.to_vec1::<f32>()?;
            let out_vec = self
                .forward_f32(&act_vec, m)
                .map_err(|err| candle_core::Error::Msg(err.to_string()))?;

            let mut out_dims = orig_dims.to_vec();
            if let Some(last) = out_dims.last_mut() {
                *last = self.tensor.n_rows;
            }
            Tensor::from_vec(out_vec, out_dims, a.device())
        }

        fn quantized_act_type(&self) -> Option<DType> {
            Some(DType::F32)
        }

        fn dtype_and_device(&self) -> (DType, Device) {
            (DType::F32, Device::Cpu)
        }

        fn plan_isq(&self, _request: &IsqRequest) -> CandleResult<IsqPlanParams> {
            candle_core::bail!("Already quantized as aibox_ternary_hadamard")
        }

        fn add_delta_w(&self, _delta: &Tensor) -> CandleResult<Arc<dyn QuantMethod>> {
            candle_core::bail!("LoRA delta_w not supported on aibox_ternary_hadamard")
        }

        fn apply_isq(
            self: Arc<Self>,
            _dtype: Option<IsqType>,
            _device: Device,
            _n_quantized: &AtomicUsize,
            _imatrix_weight: Option<Vec<f32>>,
            _guard: QuantizeOntoGuard,
        ) -> CandleResult<Arc<dyn QuantMethod>> {
            Ok(self)
        }
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
