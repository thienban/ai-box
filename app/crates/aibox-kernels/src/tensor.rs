//! KERN-01 : Loader, validation d'alignement 32 octets et packing 2-bit ternaire.

use crate::{
    validate_group64_affine_geometry, validate_packed_geometry, validate_q2_affine_geometry,
    validate_ternary_geometry, BackendError, BINARY_PACKED_BYTES_PER_GROUP, GROUP_SIZE,
    PACKED_BYTES_PER_GROUP, Q2_GROUP_SIZE, Q2_PACKED_BYTES_PER_GROUP, Q4_GROUP_SIZE,
    Q4_PACKED_BYTES_PER_GROUP,
};

/// Encode un poids ternaire `{-1, 0, +1}` sur 2 bits (`00 -> 0`, `01 -> +1`, `11 -> -1`).
#[inline]
pub fn encode_ternary(weight: i8) -> Result<u8, BackendError> {
    match weight {
        0 => Ok(0b00),
        1 => Ok(0b01),
        -1 => Ok(0b11),
        other => Err(BackendError::InvalidTernaryValue(other)),
    }
}

/// Décode un code 2-bit (`0..=3`) vers `{-1, 0, +1}`.
#[inline]
pub const fn decode_ternary(code: u8) -> i8 {
    match code & 0b11 {
        0b01 => 1,
        0b11 => -1,
        _ => 0,
    }
}

/// Compacte une matrice `[n_rows, k_cols]` de `{-1, 0, +1}` en `[n_rows, k_cols / 4]` octets.
/// Convention identique au kernel Triton : 4 poids par octet, LSB-first.
pub fn pack_ternary_2bit_rows(
    weights: &[i8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<u8>, BackendError> {
    validate_ternary_geometry(n_rows, k_cols, None, None)?;
    let expected = n_rows * k_cols;
    if weights.len() != expected {
        return Err(BackendError::BufferLengthMismatch(format!(
            "weights len {} != expected {expected}",
            weights.len()
        )));
    }

    let mut packed = Vec::with_capacity(expected / 4);
    for chunk in weights.chunks_exact(4) {
        let c0 = encode_ternary(chunk[0])?;
        let c1 = encode_ternary(chunk[1])?;
        let c2 = encode_ternary(chunk[2])?;
        let c3 = encode_ternary(chunk[3])?;
        packed.push(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
    }
    Ok(packed)
}

/// Décompacte `[n_rows, k_cols / 4]` octets vers `[n_rows, k_cols]` en `{-1, 0, +1}`.
pub fn unpack_ternary_2bit_rows(
    packed: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<i8>, BackendError> {
    validate_ternary_geometry(n_rows, k_cols, Some(packed.len()), None)?;

    let mut unpacked = Vec::with_capacity(n_rows * k_cols);
    for &byte in packed {
        unpacked.push(decode_ternary(byte));
        unpacked.push(decode_ternary(byte >> 2));
        unpacked.push(decode_ternary(byte >> 4));
        unpacked.push(decode_ternary(byte >> 6));
    }
    Ok(unpacked)
}

/// Compacte une matrice `[n_rows, k_cols]` de `{-1, +1}` en `[n_rows, k_cols / 8]` octets (`Q1_0`, 8 poids/octet LSB-first).
pub fn pack_binary_1bit_rows(
    weights: &[i8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<u8>, BackendError> {
    validate_packed_geometry(n_rows, k_cols, BINARY_PACKED_BYTES_PER_GROUP, None, None)?;
    let expected = n_rows * k_cols;
    if weights.len() != expected {
        return Err(BackendError::BufferLengthMismatch(format!(
            "weights len {} != expected {expected}",
            weights.len()
        )));
    }

    let mut packed = Vec::with_capacity(expected / 8);
    for chunk in weights.chunks_exact(8) {
        let mut byte = 0u8;
        for (bit_idx, &w) in chunk.iter().enumerate() {
            match w {
                1 => byte |= 1 << bit_idx,
                -1 => {}
                other => return Err(BackendError::InvalidTernaryValue(other)),
            }
        }
        packed.push(byte);
    }
    Ok(packed)
}

/// Décompacte `[n_rows, k_cols / 8]` octets (`Q1_0`) vers `[n_rows, k_cols]` en `{-1, +1}`.
pub fn unpack_binary_1bit_rows(
    packed: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<i8>, BackendError> {
    validate_packed_geometry(
        n_rows,
        k_cols,
        BINARY_PACKED_BYTES_PER_GROUP,
        Some(packed.len()),
        None,
    )?;

    let mut unpacked = Vec::with_capacity(n_rows * k_cols);
    for &byte in packed {
        for bit_idx in 0..8 {
            unpacked.push(if ((byte >> bit_idx) & 1) == 1 { 1 } else { -1 });
        }
    }
    Ok(unpacked)
}

/// Représentation mémoire validée d'un tenseur quantifié (2-bit ternaire ou 1-bit binaire) par groupes de 128.
#[derive(Debug, Clone)]
pub struct TernaryTensor {
    pub n_rows: usize,
    pub k_cols: usize,
    pub bits: u8,
    pub packed_weights: Vec<u8>,
    pub scales_fp16: Vec<u16>,
}

impl TernaryTensor {
    /// Construit un `TernaryTensor` (1-bit ou 2-bit) en validant sa géométrie par groupes de 128 poids.
    pub fn from_raw_with_bits(
        n_rows: usize,
        k_cols: usize,
        bits: u8,
        packed_weights: Vec<u8>,
        scales_fp16: Vec<u16>,
    ) -> Result<Self, BackendError> {
        let bytes_per_group = match bits {
            1 => BINARY_PACKED_BYTES_PER_GROUP,
            2 => PACKED_BYTES_PER_GROUP,
            4 | 8 => (GROUP_SIZE * bits as usize) / 8,
            other => {
                return Err(BackendError::InvalidDimensions(format!(
                    "unsupported quantized bit width {other} (expected 1, 2, 4 or 8)"
                )))
            }
        };
        validate_packed_geometry(
            n_rows,
            k_cols,
            bytes_per_group,
            Some(packed_weights.len()),
            Some(scales_fp16.len()),
        )?;
        Ok(Self {
            n_rows,
            k_cols,
            bits,
            packed_weights,
            scales_fp16,
        })
    }

    /// Construit un `TernaryTensor` 2-bit en validant les dimensions et la géométrie par groupes de 32 octets.
    pub fn from_raw(
        n_rows: usize,
        k_cols: usize,
        packed_weights: Vec<u8>,
        scales_fp16: Vec<u16>,
    ) -> Result<Self, BackendError> {
        Self::from_raw_with_bits(n_rows, k_cols, 2, packed_weights, scales_fp16)
    }

    /// Construit un `TernaryTensor` 1-bit (`Q1_0`) en validant la géométrie par groupes de 16 octets.
    pub fn from_binary_raw(
        n_rows: usize,
        k_cols: usize,
        packed_weights: Vec<u8>,
        scales_fp16: Vec<u16>,
    ) -> Result<Self, BackendError> {
        Self::from_raw_with_bits(n_rows, k_cols, 1, packed_weights, scales_fp16)
    }

    pub fn is_binary(&self) -> bool {
        self.bits == 1
    }

    /// Vérifie le stride d'une ligne en octets (16, 32, 64 ou 128 octets par groupe).
    pub fn row_stride_bytes(&self) -> usize {
        (self.k_cols / GROUP_SIZE) * ((GROUP_SIZE * self.bits as usize) / 8)
    }
}

/// Tenseur quantifié 2-bit affine (`uint2` in `[0..=3]`, `w = q * scale + zero`) par groupes de 128 poids.
#[derive(Debug, Clone)]
pub struct Q2AffineTensor {
    pub n_rows: usize,
    pub k_cols: usize,
    pub packed_weights: Vec<u8>,
    pub scales_fp16: Vec<u16>,
    pub zeros_fp16: Vec<u16>,
}

impl Q2AffineTensor {
    /// Construit un `Q2AffineTensor` en validant les dimensions (multiples de `Q2_GROUP_SIZE = 64`).
    pub fn from_raw(
        n_rows: usize,
        k_cols: usize,
        packed_weights: Vec<u8>,
        scales_fp16: Vec<u16>,
        zeros_fp16: Vec<u16>,
    ) -> Result<Self, BackendError> {
        validate_q2_affine_geometry(
            n_rows,
            k_cols,
            Some(packed_weights.len()),
            Some(scales_fp16.len()),
        )?;
        let expected_zeros = n_rows * (k_cols / Q2_GROUP_SIZE);
        if zeros_fp16.len() != expected_zeros {
            return Err(BackendError::BufferLengthMismatch(format!(
                "zeros_fp16 len {} != expected {expected_zeros}",
                zeros_fp16.len()
            )));
        }
        Ok(Self {
            n_rows,
            k_cols,
            packed_weights,
            scales_fp16,
            zeros_fp16,
        })
    }

    pub fn row_stride_bytes(&self) -> usize {
        (self.k_cols / Q2_GROUP_SIZE) * Q2_PACKED_BYTES_PER_GROUP
    }
}

/// Compacte une matrice `[n_rows, k_cols]` de codes `0..=3` en `[n_rows, k_cols / 4]` octets (LSB-first).
pub fn pack_q2_affine_rows(
    codes: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<u8>, BackendError> {
    validate_q2_affine_geometry(n_rows, k_cols, None, None)?;
    let expected = n_rows * k_cols;
    if codes.len() != expected {
        return Err(BackendError::BufferLengthMismatch(format!(
            "codes len {} != expected {expected}",
            codes.len()
        )));
    }
    let mut packed = Vec::with_capacity(expected / 4);
    for chunk in codes.chunks_exact(4) {
        if chunk.iter().any(|&c| c > 3) {
            return Err(BackendError::InvalidDimensions(
                "q2_affine codes must be in 0..=3".to_string(),
            ));
        }
        packed.push(chunk[0] | (chunk[1] << 2) | (chunk[2] << 4) | (chunk[3] << 6));
    }
    Ok(packed)
}

/// Décompacte `[n_rows, k_cols / 4]` octets `q2_affine` vers `[n_rows, k_cols]` en `0..=3`.
pub fn unpack_q2_affine_rows(
    packed: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<u8>, BackendError> {
    validate_q2_affine_geometry(n_rows, k_cols, Some(packed.len()), None)?;
    let mut unpacked = Vec::with_capacity(n_rows * k_cols);
    for &byte in packed {
        unpacked.push(byte & 0b11);
        unpacked.push((byte >> 2) & 0b11);
        unpacked.push((byte >> 4) & 0b11);
        unpacked.push((byte >> 6) & 0b11);
    }
    Ok(unpacked)
}

/// Quantifie une matrice `f32` `[n_rows, k_cols]` en `Q2AffineTensor` par groupes de 64 poids.
pub fn quantize_q2_affine_f32(
    weights: &[f32],
    n_rows: usize,
    k_cols: usize,
) -> Result<Q2AffineTensor, BackendError> {
    validate_q2_affine_geometry(n_rows, k_cols, None, None)?;
    let expected = n_rows * k_cols;
    if weights.len() != expected {
        return Err(BackendError::BufferLengthMismatch(format!(
            "weights len {} != expected {expected}",
            weights.len()
        )));
    }
    let num_groups = expected / Q2_GROUP_SIZE;
    let mut codes = Vec::with_capacity(expected);
    let mut scales_fp16 = Vec::with_capacity(num_groups);
    let mut zeros_fp16 = Vec::with_capacity(num_groups);

    for group in weights.chunks_exact(Q2_GROUP_SIZE) {
        let mut min_w = f32::INFINITY;
        let mut max_w = f32::NEG_INFINITY;
        for &w in group {
            min_w = min_w.min(w);
            max_w = max_w.max(w);
        }
        let scale = (max_w - min_w) / 3.0;
        let s_half = f32_to_f16(scale);
        let z_half = f32_to_f16(min_w);
        let s = f16_to_f32(s_half);
        let z = f16_to_f32(z_half);
        scales_fp16.push(s_half);
        zeros_fp16.push(z_half);
        for &w in group {
            let q = if s > 1e-12 {
                ((w - z) / s).round().clamp(0.0, 3.0) as u8
            } else {
                0
            };
            codes.push(q);
        }
    }

    let packed_weights = pack_q2_affine_rows(&codes, n_rows, k_cols)?;
    Q2AffineTensor::from_raw(n_rows, k_cols, packed_weights, scales_fp16, zeros_fp16)
}

/// Tenseur quantifié 4-bit affine (`uint4` in `[0..=15]`, `w = q * scale + zero`) par groupes de 64 poids.
#[derive(Debug, Clone)]
pub struct Q4AffineTensor {
    pub n_rows: usize,
    pub k_cols: usize,
    pub packed_weights: Vec<u8>,
    pub scales_fp16: Vec<u16>,
    pub zeros_fp16: Vec<u16>,
}

impl Q4AffineTensor {
    /// Construit un `Q4AffineTensor` en validant les dimensions (multiples de `Q4_GROUP_SIZE = 64`).
    pub fn from_raw(
        n_rows: usize,
        k_cols: usize,
        packed_weights: Vec<u8>,
        scales_fp16: Vec<u16>,
        zeros_fp16: Vec<u16>,
    ) -> Result<Self, BackendError> {
        validate_group64_affine_geometry(
            n_rows,
            k_cols,
            Q4_PACKED_BYTES_PER_GROUP,
            Some(packed_weights.len()),
            Some(scales_fp16.len()),
        )?;
        let expected_zeros = n_rows * (k_cols / Q4_GROUP_SIZE);
        if zeros_fp16.len() != expected_zeros {
            return Err(BackendError::BufferLengthMismatch(format!(
                "zeros_fp16 len {} != expected {expected_zeros}",
                zeros_fp16.len()
            )));
        }
        Ok(Self {
            n_rows,
            k_cols,
            packed_weights,
            scales_fp16,
            zeros_fp16,
        })
    }

    pub fn row_stride_bytes(&self) -> usize {
        (self.k_cols / Q4_GROUP_SIZE) * Q4_PACKED_BYTES_PER_GROUP
    }
}

/// Compacte une matrice `[n_rows, k_cols]` de codes `0..=15` en `[n_rows, k_cols / 2]` octets (LSB-first).
pub fn pack_q4_affine_rows(
    codes: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<u8>, BackendError> {
    validate_group64_affine_geometry(n_rows, k_cols, Q4_PACKED_BYTES_PER_GROUP, None, None)?;
    let expected = n_rows * k_cols;
    if codes.len() != expected {
        return Err(BackendError::BufferLengthMismatch(format!(
            "codes len {} != expected {expected}",
            codes.len()
        )));
    }
    let mut packed = Vec::with_capacity(expected / 2);
    for chunk in codes.chunks_exact(2) {
        if chunk[0] > 15 || chunk[1] > 15 {
            return Err(BackendError::InvalidDimensions(
                "q4_affine codes must be in 0..=15".to_string(),
            ));
        }
        packed.push(chunk[0] | (chunk[1] << 4));
    }
    Ok(packed)
}

/// Décompacte `[n_rows, k_cols / 2]` octets `q4_affine` vers `[n_rows, k_cols]` en `0..=15`.
pub fn unpack_q4_affine_rows(
    packed: &[u8],
    n_rows: usize,
    k_cols: usize,
) -> Result<Vec<u8>, BackendError> {
    validate_group64_affine_geometry(
        n_rows,
        k_cols,
        Q4_PACKED_BYTES_PER_GROUP,
        Some(packed.len()),
        None,
    )?;
    let mut unpacked = Vec::with_capacity(n_rows * k_cols);
    for &byte in packed {
        unpacked.push(byte & 0x0F);
        unpacked.push((byte >> 4) & 0x0F);
    }
    Ok(unpacked)
}

/// Quantifie une matrice `f32` `[n_rows, k_cols]` en `Q4AffineTensor` par groupes de 64 poids.
pub fn quantize_q4_affine_f32(
    weights: &[f32],
    n_rows: usize,
    k_cols: usize,
) -> Result<Q4AffineTensor, BackendError> {
    validate_group64_affine_geometry(n_rows, k_cols, Q4_PACKED_BYTES_PER_GROUP, None, None)?;
    let expected = n_rows * k_cols;
    if weights.len() != expected {
        return Err(BackendError::BufferLengthMismatch(format!(
            "weights len {} != expected {expected}",
            weights.len()
        )));
    }
    let num_groups = expected / Q4_GROUP_SIZE;
    let mut codes = Vec::with_capacity(expected);
    let mut scales_fp16 = Vec::with_capacity(num_groups);
    let mut zeros_fp16 = Vec::with_capacity(num_groups);

    for group in weights.chunks_exact(Q4_GROUP_SIZE) {
        let mut min_w = f32::INFINITY;
        let mut max_w = f32::NEG_INFINITY;
        for &w in group {
            min_w = min_w.min(w);
            max_w = max_w.max(w);
        }
        let scale = (max_w - min_w) / 15.0;
        let s_half = f32_to_f16(scale);
        let z_half = f32_to_f16(min_w);
        let s = f16_to_f32(s_half);
        let z = f16_to_f32(z_half);
        scales_fp16.push(s_half);
        zeros_fp16.push(z_half);
        for &w in group {
            let q = if s > 1e-12 {
                ((w - z) / s).round().clamp(0.0, 15.0) as u8
            } else {
                0
            };
            codes.push(q);
        }
    }

    let packed_weights = pack_q4_affine_rows(&codes, n_rows, k_cols)?;
    Q4AffineTensor::from_raw(n_rows, k_cols, packed_weights, scales_fp16, zeros_fp16)
}

/// Convertit un `u16` IEEE-754 half-precision (FP16) en `f32`.
#[inline(always)]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits & 0x7C00) >> 10) as u32;
    let mant = (bits & 0x03FF) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        sign | (e << 23) | (mant << 13)
    };

    f32::from_bits(f32_bits)
}

/// Convertit un `f32` en `u16` IEEE-754 half-precision (FP16, arrondi au plus proche).
#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        return sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = (mant | 0x0080_0000) >> (1 - new_exp + 13);
        return sign | (m as u16);
    }

    let rounded_mant = (mant + 0x1000) >> 13;
    if rounded_mant >= 0x0400 {
        return sign | (((new_exp as u16) + 1) << 10);
    }
    sign | ((new_exp as u16) << 10) | (rounded_mant as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_pack_unpack_ternary() {
        let n_rows = 4;
        let k_cols = 256;
        let mut weights = Vec::with_capacity(n_rows * k_cols);
        for i in 0..(n_rows * k_cols) {
            weights.push(match i % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            });
        }

        let packed = pack_ternary_2bit_rows(&weights, n_rows, k_cols).expect("pack failed");
        assert_eq!(packed.len(), n_rows * (k_cols / 4));

        let unpacked = unpack_ternary_2bit_rows(&packed, n_rows, k_cols).expect("unpack failed");
        assert_eq!(unpacked, weights);

        let scales = vec![f32_to_f16(0.5); n_rows * (k_cols / GROUP_SIZE)];
        let tensor = TernaryTensor::from_raw(n_rows, k_cols, packed, scales).expect("tensor valid");
        assert_eq!(tensor.row_stride_bytes(), 64);
        assert_eq!(tensor.row_stride_bytes() % PACKED_BYTES_PER_GROUP, 0);
    }

    #[test]
    fn roundtrip_pack_unpack_binary_1bit() {
        let n_rows = 4;
        let k_cols = 256;
        let weights: Vec<i8> = (0..(n_rows * k_cols))
            .map(|i| if (i % 5) < 2 { 1 } else { -1 })
            .collect();

        let packed = pack_binary_1bit_rows(&weights, n_rows, k_cols).expect("pack 1-bit failed");
        assert_eq!(packed.len(), n_rows * (k_cols / 8));

        let unpacked = unpack_binary_1bit_rows(&packed, n_rows, k_cols).expect("unpack 1-bit failed");
        assert_eq!(unpacked, weights);

        let scales = vec![f32_to_f16(0.5); n_rows * (k_cols / GROUP_SIZE)];
        let tensor = TernaryTensor::from_binary_raw(n_rows, k_cols, packed, scales).expect("1-bit tensor valid");
        assert!(tensor.is_binary());
        assert_eq!(tensor.row_stride_bytes(), 32);
    }

    #[test]
    fn f16_roundtrip_accuracy() {
        for &val in &[0.0f32, 1.0, -1.0, 0.125, -0.375, 2.5, 42.0] {
            let half_bits = f32_to_f16(val);
            let back = f16_to_f32(half_bits);
            assert!((val - back).abs() < 1e-3, "val={val} back={back}");
        }
    }

    #[test]
    fn roundtrip_pack_unpack_q2_affine() {
        let n_rows = 2;
        let k_cols = 256;
        let weights: Vec<f32> = (0..(n_rows * k_cols))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        let tensor = quantize_q2_affine_f32(&weights, n_rows, k_cols).expect("q2_affine quant");
        assert_eq!(tensor.row_stride_bytes(), 64);
        let unpacked = unpack_q2_affine_rows(&tensor.packed_weights, n_rows, k_cols).expect("unpack");
        assert_eq!(unpacked.len(), n_rows * k_cols);
        assert!(unpacked.iter().all(|&q| q <= 3));
    }

    #[test]
    fn roundtrip_pack_unpack_q4_affine() {
        let n_rows = 2;
        let k_cols = 192;
        let weights: Vec<f32> = (0..(n_rows * k_cols))
            .map(|i| ((i % 13) as f32 - 6.0) * 0.25)
            .collect();
        let tensor = quantize_q4_affine_f32(&weights, n_rows, k_cols).expect("q4_affine quant");
        assert_eq!(tensor.row_stride_bytes(), 96);
        let unpacked = unpack_q4_affine_rows(&tensor.packed_weights, n_rows, k_cols).expect("unpack");
        assert_eq!(unpacked.len(), n_rows * k_cols);
        assert!(unpacked.iter().all(|&q| q <= 15));
    }
}
