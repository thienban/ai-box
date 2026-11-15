//! KERN-02 : Backend de référence scalaire (pure Rust sans SIMD).

use crate::{
    fwht::apply_fwht_128_rows,
    tensor::{decode_ternary, f16_to_f32},
    validate_matmul_inputs, validate_packed_geometry, BackendError, TernaryHadamardBackend,
    BINARY_PACKED_BYTES_PER_GROUP, GROUP_SIZE, PACKED_BYTES_PER_GROUP,
};

/// Implémentation scalaire de référence de `TernaryHadamardBackend`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScalarBackend;

impl TernaryHadamardBackend for ScalarBackend {
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
        validate_matmul_inputs(activations, packed_weights, scales_fp16, m, n, k)?;
        matmul_packed_scalar(activations, packed_weights, scales_fp16, m, n, k, use_hadamard, 2)
    }
}

/// Multiplication matricielle 1-bit (`Q1_0`, 8 poids/octet LSB-first) de référence sur CPU.
pub fn matmul_binary_1bit(
    activations: &[f32],
    packed_weights: &[u8],
    scales_fp16: &[u16],
    m: usize,
    n: usize,
    k: usize,
    use_hadamard: bool,
) -> Result<Vec<f32>, BackendError> {
    if activations.len() != m * k {
        return Err(BackendError::BufferLengthMismatch(format!(
            "activations len {} != m*k ({})",
            activations.len(),
            m * k
        )));
    }
    validate_packed_geometry(
        n,
        k,
        BINARY_PACKED_BYTES_PER_GROUP,
        Some(packed_weights.len()),
        Some(scales_fp16.len()),
    )?;
    matmul_packed_scalar(activations, packed_weights, scales_fp16, m, n, k, use_hadamard, 1)
}

fn matmul_packed_scalar(
    activations: &[f32],
    packed_weights: &[u8],
    scales_fp16: &[u16],
    m: usize,
    n: usize,
    k: usize,
    use_hadamard: bool,
    bits: u8,
) -> Result<Vec<f32>, BackendError> {
    let rotated_x;
    let x_input = if use_hadamard {
        rotated_x = apply_fwht_128_rows(activations, m, k)?;
        rotated_x.as_slice()
    } else {
        activations
    };
    let bytes_per_group = if bits == 1 { BINARY_PACKED_BYTES_PER_GROUP } else { PACKED_BYTES_PER_GROUP };
    let groups_per_row = k / GROUP_SIZE;
    let packed_row_stride = groups_per_row * bytes_per_group;
    let mut output = vec![0.0f32; m * n];

    for row_m in 0..m {
        let x_row = &x_input[row_m * k..(row_m + 1) * k];
        let out_row = &mut output[row_m * n..(row_m + 1) * n];
        for row_n in 0..n {
            let w_row = &packed_weights[row_n * packed_row_stride..(row_n + 1) * packed_row_stride];
            let s_row = &scales_fp16[row_n * groups_per_row..(row_n + 1) * groups_per_row];
            out_row[row_n] = if bits == 1 {
                dot_binary_row_scalar(x_row, w_row, s_row, groups_per_row)
            } else {
                dot_row_scalar(x_row, w_row, s_row, groups_per_row)
            };
        }
    }
    Ok(output)
}

fn dot_binary_row_scalar(x_row: &[f32], w_row: &[u8], s_row: &[u16], groups_per_row: usize) -> f32 {
    let mut total = 0.0f32;
    for g in 0..groups_per_row {
        let x_g = &x_row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
        let w_g = &w_row[g * BINARY_PACKED_BYTES_PER_GROUP..(g + 1) * BINARY_PACKED_BYTES_PER_GROUP];
        let mut acc = 0.0f32;
        for (byte_idx, &byte) in w_g.iter().enumerate() {
            let base = byte_idx * 8;
            for bit in 0..8 {
                let sign = if ((byte >> bit) & 1) == 1 { 1.0f32 } else { -1.0f32 };
                acc += sign * x_g[base + bit];
            }
        }
        total += acc * f16_to_f32(s_row[g]);
    }
    total
}

/// Calcule le produit scalaire complet d'une ligne de poids ternaires par groupes de 128.
pub(crate) fn dot_row_scalar(
    x_row: &[f32],
    w_row: &[u8],
    s_row: &[u16],
    groups_per_row: usize,
) -> f32 {
    let mut total = 0.0f32;
    for group_idx in 0..groups_per_row {
        let x_group = &x_row[group_idx * GROUP_SIZE..(group_idx + 1) * GROUP_SIZE];
        let w_group = &w_row[group_idx * PACKED_BYTES_PER_GROUP..(group_idx + 1) * PACKED_BYTES_PER_GROUP];
        let scale = f16_to_f32(s_row[group_idx]);
        total += dot_group_scalar(x_group, w_group) * scale;
    }
    total
}

/// Accumule les 128 poids d'un groupe sans multiplication (`+`, `-`, ou `0`).
#[inline]
fn dot_group_scalar(x_group: &[f32], w_group: &[u8]) -> f32 {
    let mut group_acc = 0.0f32;
    for (byte_idx, &packed_byte) in w_group.iter().enumerate() {
        let base_k = byte_idx * 4;
        for sub in 0..4 {
            let weight = decode_ternary(packed_byte >> (sub * 2));
            if weight == 1 {
                group_acc += x_group[base_k + sub];
            } else if weight == -1 {
                group_acc -= x_group[base_k + sub];
            }
        }
    }
    group_acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{f32_to_f16, pack_ternary_2bit_rows};

    fn dense_hadamard_128() -> Vec<f32> {
        let mut h = vec![1.0f32];
        let mut size = 1;
        while size < GROUP_SIZE {
            let next_size = size * 2;
            let mut next = vec![0.0f32; next_size * next_size];
            for r in 0..size {
                for c in 0..size {
                    let v = h[r * size + c];
                    next[r * next_size + c] = v;
                    next[r * next_size + (c + size)] = v;
                    next[(r + size) * next_size + c] = v;
                    next[(r + size) * next_size + (c + size)] = -v;
                }
            }
            h = next;
            size = next_size;
        }
        let inv = 1.0 / (GROUP_SIZE as f32).sqrt();
        for v in &mut h {
            *v *= inv;
        }
        h
    }

    #[test]
    fn scalar_matches_dense_hadamard_matmul() {
        let m = 2;
        let n = 4;
        let k = 256;
        let groups = k / GROUP_SIZE;

        let activations: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.11).cos()).collect();
        let weights: Vec<i8> = (0..(n * k))
            .map(|i| match (i * 7 + 3) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let scales_f32: Vec<f32> = (0..(n * groups)).map(|i| 0.25 + 0.1 * (i as f32)).collect();
        let scales_fp16: Vec<u16> = scales_f32.iter().map(|&s| f32_to_f16(s)).collect();
        let packed = pack_ternary_2bit_rows(&weights, n, k).unwrap();

        let h_128 = dense_hadamard_128();
        let mut expected = vec![0.0f32; m * n];
        for rm in 0..m {
            let mut x_rot = vec![0.0f32; k];
            for g in 0..groups {
                for r in 0..GROUP_SIZE {
                    let mut sum = 0.0f32;
                    for c in 0..GROUP_SIZE {
                        sum += activations[rm * k + g * GROUP_SIZE + c] * h_128[r * GROUP_SIZE + c];
                    }
                    x_rot[g * GROUP_SIZE + r] = sum;
                }
            }
            for rn in 0..n {
                let mut total = 0.0f32;
                for g in 0..groups {
                    let s = f16_to_f32(scales_fp16[rn * groups + g]);
                    let mut g_acc = 0.0f32;
                    for idx in 0..GROUP_SIZE {
                        let w = weights[rn * k + g * GROUP_SIZE + idx] as f32;
                        g_acc += x_rot[g * GROUP_SIZE + idx] * w;
                    }
                    total += g_acc * s;
                }
                expected[rm * n + rn] = total;
            }
        }

        let got = ScalarBackend
            .matmul_hadamard_ternary(&activations, &packed, &scales_fp16, m, n, k)
            .unwrap();

        for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
            assert!((e - g).abs() < 1e-4, "idx={i} expected={e} got={g}");
        }
    }

    #[test]
    fn scalar_binary_1bit_matches_2bit_ternary() {
        use crate::tensor::{pack_binary_1bit_rows, pack_ternary_2bit_rows};

        let m = 2usize;
        let n = 3usize;
        let k = 256usize;
        let groups = k / GROUP_SIZE;

        let activations: Vec<f32> = (0..(m * k)).map(|i| ((i % 19) as f32 - 9.0) * 0.25).collect();
        let weights: Vec<i8> = (0..(n * k)).map(|i| if (i % 3) == 0 { -1 } else { 1 }).collect();
        let scales_fp16: Vec<u16> = (0..(n * groups))
            .map(|i| f32_to_f16(0.5 + (i % 5) as f32 * 0.25))
            .collect();

        let packed_1bit = pack_binary_1bit_rows(&weights, n, k).unwrap();
        let packed_2bit = pack_ternary_2bit_rows(&weights, n, k).unwrap();

        let got_1bit_rot = matmul_binary_1bit(&activations, &packed_1bit, &scales_fp16, m, n, k, true).unwrap();
        let got_2bit_rot = ScalarBackend
            .matmul_hadamard_ternary(&activations, &packed_2bit, &scales_fp16, m, n, k)
            .unwrap();
        for (i, (&a, &b)) in got_1bit_rot.iter().zip(got_2bit_rot.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "rot idx={i} 1bit={a} 2bit={b}");
        }

        let got_1bit_plain = matmul_binary_1bit(&activations, &packed_1bit, &scales_fp16, m, n, k, false).unwrap();
        let got_2bit_plain = ScalarBackend
            .matmul_ternary(&activations, &packed_2bit, &scales_fp16, m, n, k, false)
            .unwrap();
        for (i, (&a, &b)) in got_1bit_plain.iter().zip(got_2bit_plain.iter()).enumerate() {
            assert!((a - b).abs() < 1e-4, "plain idx={i} 1bit={a} 2bit={b}");
        }
    }
}
