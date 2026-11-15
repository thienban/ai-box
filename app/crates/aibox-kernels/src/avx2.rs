//! KERN-03 : Kernel T-MAC AVX2 (`_mm256_shuffle_epi8` / `pshufb`).

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::{
    fwht::apply_fwht_128_rows,
    scalar::dot_row_scalar,
    tensor::{decode_ternary, f16_to_f32},
    validate_matmul_inputs, BackendError, TernaryHadamardBackend, GROUP_SIZE,
    PACKED_BYTES_PER_GROUP,
};

const QUANT_MAX_I16: f32 = 16_383.0;
const TILE_N: usize = 32;

/// Backend T-MAC accéléré par AVX2 (`pshufb`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Avx2Backend;

impl TernaryHadamardBackend for Avx2Backend {
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
        if !is_x86_feature_detected!("avx2") {
            return Err(BackendError::UnsupportedFeature(
                "AVX2 is not available on this CPU".to_string(),
            ));
        }
        validate_matmul_inputs(activations, packed_weights, scales_fp16, m, n, k)?;

        let rotated_x;
        let x_input = if use_hadamard {
            rotated_x = apply_fwht_128_rows(activations, m, k)?;
            rotated_x.as_slice()
        } else {
            activations
        };
        let mut output = vec![0.0f32; m * n];

        unsafe {
            matmul_avx2_dispatch(
                x_input,
                packed_weights,
                scales_fp16,
                &mut output,
                m,
                n,
                k,
            );
        }

        Ok(output)
    }
}

#[derive(Clone, Copy)]
struct PairLutBytes {
    lo: [u8; 16],
    hi: [u8; 16],
}

struct GroupLuts {
    act_scale: f32,
    pairs: Vec<PairLutBytes>,
}

fn build_group_luts(x_group: &[f32]) -> GroupLuts {
    let max_abs = x_group.iter().fold(0.0f32, |acc, &val| acc.max(val.abs()));
    if max_abs == 0.0 {
        return GroupLuts {
            act_scale: 0.0,
            pairs: vec![PairLutBytes { lo: [0; 16], hi: [0; 16] }; GROUP_SIZE / 2],
        };
    }

    let act_scale = max_abs / QUANT_MAX_I16;
    let inv_scale = 1.0 / act_scale;
    let mut pairs = Vec::with_capacity(GROUP_SIZE / 2);

    for pair in x_group.chunks_exact(2) {
        let q0 = (pair[0] * inv_scale).round() as i32;
        let q1 = (pair[1] * inv_scale).round() as i32;
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];

        for code4 in 0..16u8 {
            let w0 = decode_ternary(code4) as i32;
            let w1 = decode_ternary(code4 >> 2) as i32;
            let sum = (w0 * q0 + w1 * q1) as i16 as u16;
            lo[code4 as usize] = (sum & 0xFF) as u8;
            hi[code4 as usize] = (sum >> 8) as u8;
        }
        pairs.push(PairLutBytes { lo, hi });
    }

    GroupLuts { act_scale, pairs }
}

#[target_feature(enable = "avx2")]
unsafe fn process_row_chunk_avx2(
    x_row: &[f32],
    group_luts: &[GroupLuts],
    packed_weights: &[u8],
    scales_fp16: &[u16],
    out_chunk: &mut [f32],
    row_start: usize,
) {
    let groups_per_row = group_luts.len();
    let packed_row_stride = groups_per_row * PACKED_BYTES_PER_GROUP;
    let full_tiles_end = (out_chunk.len() / TILE_N) * TILE_N;

    for tile_offset in (0..full_tiles_end).step_by(TILE_N) {
        let global_row = row_start + tile_offset;
        accumulate_tile32_avx2(
            group_luts,
            &packed_weights[global_row * packed_row_stride..],
            &scales_fp16[global_row * groups_per_row..],
            &mut out_chunk[tile_offset..tile_offset + TILE_N],
            packed_row_stride,
        );
    }

    for tail_offset in full_tiles_end..out_chunk.len() {
        let global_row = row_start + tail_offset;
        let w_row = &packed_weights[global_row * packed_row_stride..(global_row + 1) * packed_row_stride];
        let s_row = &scales_fp16[global_row * groups_per_row..(global_row + 1) * groups_per_row];
        out_chunk[tail_offset] = dot_row_scalar(x_row, w_row, s_row, groups_per_row);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn matmul_avx2_dispatch(
    rotated_x: &[f32],
    packed_weights: &[u8],
    scales_fp16: &[u16],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    let groups_per_row = k / GROUP_SIZE;
    let num_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);

    for row_m in 0..m {
        let x_row = &rotated_x[row_m * k..(row_m + 1) * k];
        let out_row = &mut output[row_m * n..(row_m + 1) * n];
        let group_luts: Vec<GroupLuts> = (0..groups_per_row)
            .map(|group_idx| build_group_luts(&x_row[group_idx * GROUP_SIZE..(group_idx + 1) * GROUP_SIZE]))
            .collect();

        if n < TILE_N * 2 || num_threads <= 1 {
            process_row_chunk_avx2(x_row, &group_luts, packed_weights, scales_fp16, out_row, 0);
            continue;
        }

        let chunk_rows = n.div_ceil(TILE_N).div_ceil(num_threads) * TILE_N;
        let luts_ref = &group_luts;

        std::thread::scope(|scope| {
            for (chunk_idx, out_chunk) in out_row.chunks_mut(chunk_rows).enumerate() {
                let row_start = chunk_idx * chunk_rows;
                scope.spawn(move || {
                    process_row_chunk_avx2(
                        x_row,
                        luts_ref,
                        packed_weights,
                        scales_fp16,
                        out_chunk,
                        row_start,
                    );
                });
            }
        });
    }
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn broadcast_lut_128(bytes: &[u8; 16]) -> __m256i {
    let v128 = _mm_loadu_si128(bytes.as_ptr() as *const __m128i);
    _mm256_broadcastsi128_si256(v128)
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn lookup_pair_accumulate_32(
    pair: &PairLutBytes,
    nibbles: __m256i,
    accs: &mut [__m256i; 4],
) {
    let lut_lo = broadcast_lut_128(&pair.lo);
    let lut_hi = broadcast_lut_128(&pair.hi);
    let res_lo = _mm256_shuffle_epi8(lut_lo, nibbles);
    let res_hi = _mm256_shuffle_epi8(lut_hi, nibbles);

    let i16_lo = _mm256_unpacklo_epi8(res_lo, res_hi);
    let i16_hi = _mm256_unpackhi_epi8(res_lo, res_hi);

    accs[0] = _mm256_add_epi32(accs[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16_lo)));
    accs[1] = _mm256_add_epi32(accs[1], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(i16_hi)));
    accs[2] = _mm256_add_epi32(
        accs[2],
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(i16_lo)),
    );
    accs[3] = _mm256_add_epi32(
        accs[3],
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(i16_hi)),
    );
}

#[target_feature(enable = "avx2")]
unsafe fn accumulate_group_avx2(
    luts: &GroupLuts,
    w_tile_base: &[u8],
    group_byte_offset: usize,
    packed_row_stride: usize,
) -> [i32; TILE_N] {
    let nibble_mask = _mm256_set1_epi8(0x0F);
    let mut accs = [_mm256_setzero_si256(); 4];
    let mut gathered = [0u8; TILE_N];

    for byte_idx in 0..PACKED_BYTES_PER_GROUP {
        let col_offset = group_byte_offset + byte_idx;
        for row_offset in 0..TILE_N {
            gathered[row_offset] = *w_tile_base.get_unchecked(row_offset * packed_row_stride + col_offset);
        }

        let w_vec = _mm256_loadu_si256(gathered.as_ptr() as *const __m256i);
        let nib_lo = _mm256_and_si256(w_vec, nibble_mask);
        let nib_hi = _mm256_and_si256(_mm256_srli_epi16(w_vec, 4), nibble_mask);

        lookup_pair_accumulate_32(&luts.pairs[byte_idx * 2], nib_lo, &mut accs);
        lookup_pair_accumulate_32(&luts.pairs[byte_idx * 2 + 1], nib_hi, &mut accs);
    }

    let mut i32_sums = [0i32; TILE_N];
    for chunk_idx in 0..4 {
        let start = chunk_idx * 8;
        _mm256_storeu_si256(i32_sums[start..start + 8].as_mut_ptr() as *mut __m256i, accs[chunk_idx]);
    }
    i32_sums
}

#[target_feature(enable = "avx2")]
unsafe fn accumulate_tile32_avx2(
    group_luts: &[GroupLuts],
    w_tile_base: &[u8],
    s_tile_base: &[u16],
    out_tile: &mut [f32],
    packed_row_stride: usize,
) {
    let groups_per_row = group_luts.len();
    let mut f_acc = [0.0f32; TILE_N];

    for (group_idx, luts) in group_luts.iter().enumerate() {
        if luts.act_scale == 0.0 {
            continue;
        }
        let i32_sums = accumulate_group_avx2(
            luts,
            w_tile_base,
            group_idx * PACKED_BYTES_PER_GROUP,
            packed_row_stride,
        );
        for row_offset in 0..TILE_N {
            let w_scale = f16_to_f32(*s_tile_base.get_unchecked(row_offset * groups_per_row + group_idx));
            f_acc[row_offset] += (i32_sums[row_offset] as f32) * luts.act_scale * w_scale;
        }
    }

    out_tile.copy_from_slice(&f_acc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::ScalarBackend;
    use crate::tensor::{f32_to_f16, pack_ternary_2bit_rows};

    #[test]
    fn avx2_matches_scalar_within_tolerance() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let m = 2;
        let n = 69;
        let k = 512;
        let groups = k / GROUP_SIZE;

        let activations: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.073).sin()).collect();
        let weights: Vec<i8> = (0..(n * k))
            .map(|i| match (i * 11 + 5) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let scales_fp16: Vec<u16> = (0..(n * groups))
            .map(|i| f32_to_f16(0.02 + 0.005 * ((i % 17) as f32)))
            .collect();
        let packed = pack_ternary_2bit_rows(&weights, n, k).unwrap();

        let ref_out = ScalarBackend
            .matmul_hadamard_ternary(&activations, &packed, &scales_fp16, m, n, k)
            .unwrap();
        let avx_out = Avx2Backend
            .matmul_hadamard_ternary(&activations, &packed, &scales_fp16, m, n, k)
            .unwrap();

        let max_diff = ref_out
            .iter()
            .zip(avx_out.iter())
            .map(|(&r, &a)| (r - a).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max_diff={max_diff} >= 1e-3");
    }

    #[test]
    fn avx2_matches_scalar_without_hadamard_and_with_outliers() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let m = 2;
        let n = 69;
        let k = 512;
        let groups = k / GROUP_SIZE;

        let mut activations: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.073).sin()).collect();
        // Injecte des outliers réalistes (non lissés quand hadamard=false)
        activations[3] = 42.5;
        activations[k + 129] = -38.0;

        let weights: Vec<i8> = (0..(n * k))
            .map(|i| match (i * 11 + 5) % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let scales_fp16: Vec<u16> = (0..(n * groups))
            .map(|i| f32_to_f16(0.02 + 0.005 * ((i % 17) as f32)))
            .collect();
        let packed = pack_ternary_2bit_rows(&weights, n, k).unwrap();

        let ref_out = ScalarBackend
            .matmul_ternary(&activations, &packed, &scales_fp16, m, n, k, false)
            .unwrap();
        let avx_out = Avx2Backend
            .matmul_ternary(&activations, &packed, &scales_fp16, m, n, k, false)
            .unwrap();

        let max_rel_diff = ref_out
            .iter()
            .zip(avx_out.iter())
            .map(|(&r, &a)| (r - a).abs() / r.abs().max(1.0))
            .fold(0.0f32, f32::max);
        assert!(max_rel_diff < 1e-3, "max_rel_diff={max_rel_diff} >= 1e-3");
    }
}
