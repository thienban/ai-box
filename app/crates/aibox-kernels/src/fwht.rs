//! KERN-02 : Fast Walsh-Hadamard Transform (FWHT) par blocs de 128 sur CPU.

use crate::{BackendError, GROUP_SIZE};

/// Facteur de normalisation `1 / sqrt(128)` pour rendre `H_128` orthonormale.
const INV_SQRT_128: f32 = 0.088_388_35_f32;

/// Applique la transformation de Walsh-Hadamard normalisée en place sur un bloc de 128 `f32`.
pub fn fwht_128_inplace(block: &mut [f32; GROUP_SIZE]) {
    let mut step = 1;
    while step < GROUP_SIZE {
        let stride = step * 2;
        let mut base = 0;
        while base < GROUP_SIZE {
            for offset in 0..step {
                let left_idx = base + offset;
                let right_idx = left_idx + step;
                let left = block[left_idx];
                let right = block[right_idx];
                block[left_idx] = left + right;
                block[right_idx] = left - right;
            }
            base += stride;
        }
        step = stride;
    }

    for val in block.iter_mut() {
        *val *= INV_SQRT_128;
    }
}

/// Applique `fwht_128_inplace` sur chaque bloc de 128 éléments de chaque ligne `[m, k]`.
pub fn apply_fwht_128_rows(
    activations: &[f32],
    m: usize,
    k: usize,
) -> Result<Vec<f32>, BackendError> {
    if k == 0 || k % GROUP_SIZE != 0 {
        return Err(BackendError::InvalidDimensions(format!(
            "k={k} must be a non-zero multiple of {GROUP_SIZE}"
        )));
    }
    if activations.len() != m * k {
        return Err(BackendError::BufferLengthMismatch(format!(
            "activations len {} != expected {}",
            activations.len(),
            m * k
        )));
    }

    let mut rotated = activations.to_vec();
    for chunk in rotated.chunks_exact_mut(GROUP_SIZE) {
        let block: &mut [f32; GROUP_SIZE] = chunk
            .try_into()
            .expect("chunks_exact_mut(128) always yields 128 elements");
        fwht_128_inplace(block);
    }
    Ok(rotated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_128_is_involutive_orthogonal() {
        let mut original = [0.0f32; GROUP_SIZE];
        for (idx, slot) in original.iter_mut().enumerate() {
            *slot = ((idx as f32) * 0.137).sin();
        }
        let mut work = original;
        fwht_128_inplace(&mut work);
        fwht_128_inplace(&mut work);

        for idx in 0..GROUP_SIZE {
            assert!(
                (original[idx] - work[idx]).abs() < 1e-5,
                "mismatch at {idx}: {} vs {}",
                original[idx],
                work[idx]
            );
        }
    }
}
