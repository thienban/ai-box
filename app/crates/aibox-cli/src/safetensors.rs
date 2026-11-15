//! SERV-02 : Loader SafeTensors et mapping direct vers `TernaryTensor` (`KERN-01`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use aibox_kernels::tensor::{f16_to_f32, TernaryTensor};
use aibox_kernels::{BINARY_PACKED_BYTES_PER_GROUP, GROUP_SIZE, PACKED_BYTES_PER_GROUP};

use crate::json::{parse_json, JsonValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
}

/// Lecteur SafeTensors basé sur un fichier disque (lecture sélective par offset) ou tampon mémoire.
pub struct SafeTensorsFile {
    data_start: u64,
    tensors: BTreeMap<String, TensorMeta>,
    source: SafeTensorsSource,
}

enum SafeTensorsSource {
    File(File),
    #[cfg(test)]
    Memory(Vec<u8>),
}

impl SafeTensorsFile {
    /// Ouvre un fichier `.safetensors` et parse uniquement son en-tête JSON (démarrage instantané).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path.as_ref())
            .map_err(|e| format!("Failed to open '{}': {e}", path.as_ref().display()))?;

        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| format!("Failed to read safetensors header length: {e}"))?;
        let header_len = u64::from_le_bytes(len_buf) as usize;

        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)
            .map_err(|e| format!("Failed to read safetensors header: {e}"))?;

        let header_str = std::str::from_utf8(&header_bytes)
            .map_err(|e| format!("Invalid UTF-8 in safetensors header: {e}"))?;
        let tensors = parse_header_metadata(header_str)?;

        Ok(Self {
            data_start: 8 + header_len as u64,
            tensors,
            source: SafeTensorsSource::File(file),
        })
    }

    /// Parse un fichier `.safetensors` déjà présent en mémoire.
    #[cfg(test)]
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("SafeTensors buffer too short (< 8 bytes)".to_string());
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        if bytes.len() < header_end {
            return Err("SafeTensors buffer truncated inside header".to_string());
        }
        let header_str = std::str::from_utf8(&bytes[8..header_end])
            .map_err(|e| format!("Invalid UTF-8 in safetensors header: {e}"))?;
        let tensors = parse_header_metadata(header_str)?;

        Ok(Self {
            data_start: header_end as u64,
            tensors,
            source: SafeTensorsSource::Memory(bytes),
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorMeta> {
        &self.tensors
    }

    /// Lit les octets bruts d'un tenseur nommé.
    pub fn read_tensor_bytes(&mut self, name: &str) -> Result<Vec<u8>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor '{name}' not found in safetensors"))?;
        let (start, end) = meta.data_offsets;
        if end < start {
            return Err(format!("Invalid offsets [{start}, {end}] for '{name}'"));
        }
        let len = end - start;

        match &mut self.source {
            #[cfg(test)]
            SafeTensorsSource::Memory(buf) => {
                let abs_start = self.data_start as usize + start;
                let abs_end = self.data_start as usize + end;
                if abs_end > buf.len() {
                    return Err(format!("Tensor '{name}' exceeds buffer size"));
                }
                Ok(buf[abs_start..abs_end].to_vec())
            }
            SafeTensorsSource::File(file) => {
                file.seek(SeekFrom::Start(self.data_start + start as u64))
                    .map_err(|e| format!("Seek failed for '{name}': {e}"))?;
                let mut out = vec![0u8; len];
                file.read_exact(&mut out)
                    .map_err(|e| format!("Read failed for '{name}': {e}"))?;
                Ok(out)
            }
        }
    }

    /// Charge une couche linéaire ternaire (`U8` `[N, K/4]` + `F16` `[N, K/128]`) vers `TernaryTensor`.
    pub fn load_ternary_layer(
        &mut self,
        weight_name: &str,
        scale_name: &str,
    ) -> Result<TernaryTensor, String> {
        let w_meta = self
            .tensors
            .get(weight_name)
            .ok_or_else(|| format!("Missing weight tensor '{weight_name}'"))?
            .clone();
        let s_meta = self
            .tensors
            .get(scale_name)
            .ok_or_else(|| format!("Missing scale tensor '{scale_name}'"))?
            .clone();

        if w_meta.dtype != "U8" || w_meta.shape.len() != 2 {
            return Err(format!(
                "Weight '{weight_name}' must be 2D U8, got {} {:?}",
                w_meta.dtype, w_meta.shape
            ));
        }
        if s_meta.dtype != "F16" || s_meta.shape.len() != 2 {
            return Err(format!(
                "Scale '{scale_name}' must be 2D F16, got {} {:?}",
                s_meta.dtype, s_meta.shape
            ));
        }

        let n_rows = w_meta.shape[0];
        let n_groups = s_meta.shape[1];
        if s_meta.shape[0] != n_rows || n_groups == 0 {
            return Err(format!(
                "Shape mismatch between '{weight_name}' {:?} and '{scale_name}' {:?}",
                w_meta.shape, s_meta.shape
            ));
        }
        let k_cols = n_groups * GROUP_SIZE;
        let bits = match w_meta.shape[1] / n_groups {
            BINARY_PACKED_BYTES_PER_GROUP if w_meta.shape[1] == n_groups * BINARY_PACKED_BYTES_PER_GROUP => 1u8,
            PACKED_BYTES_PER_GROUP if w_meta.shape[1] == n_groups * PACKED_BYTES_PER_GROUP => 2u8,
            _ => {
                return Err(format!(
                    "Shape mismatch between '{weight_name}' {:?} and '{scale_name}' {:?} (expected {BINARY_PACKED_BYTES_PER_GROUP} or {PACKED_BYTES_PER_GROUP} bytes/group)",
                    w_meta.shape, s_meta.shape
                ));
            }
        };

        let packed_weights = self.read_tensor_bytes(weight_name)?;
        let scale_bytes = self.read_tensor_bytes(scale_name)?;
        if scale_bytes.len() % 2 != 0 {
            return Err(format!("Invalid F16 byte length for '{scale_name}'"));
        }
        let scales_fp16: Vec<u16> = scale_bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect();

        TernaryTensor::from_raw_with_bits(n_rows, k_cols, bits, packed_weights, scales_fp16)
            .map_err(|e| format!("Invalid {bits}-bit TernaryTensor '{weight_name}': {e}"))
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Charge un tenseur F16 ou F32 (normes, embeddings) en `Vec<f32>`.
    pub fn load_f32_vector(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let dtype = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Missing tensor '{name}'"))?
            .dtype
            .clone();
        let bytes = self.read_tensor_bytes(name)?;
        match dtype.as_str() {
            "F16" => {
                if bytes.len() % 2 != 0 {
                    return Err(format!("Invalid F16 byte length for '{name}'"));
                }
                Ok(bytes
                    .chunks_exact(2)
                    .map(|pair| f16_to_f32(u16::from_le_bytes([pair[0], pair[1]])))
                    .collect())
            }
            "F32" => {
                if bytes.len() % 4 != 0 {
                    return Err(format!("Invalid F32 byte length for '{name}'"));
                }
                Ok(bytes
                    .chunks_exact(4)
                    .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
                    .collect())
            }
            other => Err(format!("Unsupported dtype '{other}' for '{name}' (expected F16 or F32)")),
        }
    }

    /// Charge un tenseur F16 brut en `Vec<u16>` (divise par 2 l'empreinte RAM des embeddings).
    pub fn load_f16_raw(&mut self, name: &str) -> Result<Vec<u16>, String> {
        let dtype = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Missing tensor '{name}'"))?
            .dtype
            .clone();
        if dtype != "F16" {
            return Err(format!("Tensor '{name}' must be F16, got '{dtype}'"));
        }
        let bytes = self.read_tensor_bytes(name)?;
        if bytes.len() % 2 != 0 {
            return Err(format!("Invalid F16 byte length for '{name}'"));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }
}

fn parse_header_metadata(header_str: &str) -> Result<BTreeMap<String, TensorMeta>, String> {
    let root = parse_json(header_str)?;
    let obj = root
        .as_object()
        .ok_or_else(|| "SafeTensors header must be a JSON object".to_string())?;

    let mut tensors = BTreeMap::new();
    for (key, val) in obj {
        if key == "__metadata__" {
            continue;
        }
        let t_obj = val
            .as_object()
            .ok_or_else(|| format!("Tensor entry '{key}' must be an object"))?;
        let dtype = t_obj
            .get("dtype")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("Missing dtype for '{key}'"))?
            .to_string();
        let shape_arr = t_obj
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| format!("Missing shape for '{key}'"))?;
        let shape: Vec<usize> = shape_arr
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .ok_or_else(|| format!("Invalid shape element in '{key}'"))
            })
            .collect::<Result<_, _>>()?;

        let offsets_arr = t_obj
            .get("data_offsets")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| format!("Missing data_offsets for '{key}'"))?;
        if offsets_arr.len() != 2 {
            return Err(format!("data_offsets must have 2 elements in '{key}'"));
        }
        let start = offsets_arr[0]
            .as_u64()
            .ok_or_else(|| format!("Invalid start offset in '{key}'"))? as usize;
        let end = offsets_arr[1]
            .as_u64()
            .ok_or_else(|| format!("Invalid end offset in '{key}'"))? as usize;

        tensors.insert(
            key.clone(),
            TensorMeta {
                dtype,
                shape,
                data_offsets: (start, end),
            },
        );
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows};
    use aibox_kernels::{TernaryHadamardBackend, TmacCpuBackend};

    #[test]
    fn loads_safetensors_and_runs_tmac_kernel() {
        let n_rows = 32;
        let k_cols = 256;
        let groups = k_cols / GROUP_SIZE;

        let weights: Vec<i8> = (0..(n_rows * k_cols))
            .map(|i| match i % 3 {
                0 => 0,
                1 => 1,
                _ => -1,
            })
            .collect();
        let packed = pack_ternary_2bit_rows(&weights, n_rows, k_cols).unwrap();
        let scales_u16 = vec![f32_to_f16(0.25); n_rows * groups];
        let mut scales_bytes = Vec::with_capacity(scales_u16.len() * 2);
        for s in &scales_u16 {
            scales_bytes.extend_from_slice(&s.to_le_bytes());
        }

        let w_end = packed.len();
        let s_end = w_end + scales_bytes.len();
        let header = format!(
            r#"{{"blk.0.attn_q.weight":{{"dtype":"U8","shape":[{n_rows},{}],"data_offsets":[0,{w_end}]}},"blk.0.attn_q.scales":{{"dtype":"F16","shape":[{n_rows},{groups}],"data_offsets":[{w_end},{s_end}]}}}}"#,
            k_cols / 4
        );
        let header_bytes = header.as_bytes();
        let mut st_buf = Vec::new();
        st_buf.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        st_buf.extend_from_slice(header_bytes);
        st_buf.extend_from_slice(&packed);
        st_buf.extend_from_slice(&scales_bytes);

        let mut st = SafeTensorsFile::from_bytes(st_buf).expect("parse safetensors");
        let tensor = st
            .load_ternary_layer("blk.0.attn_q.weight", "blk.0.attn_q.scales")
            .expect("load ternary layer");
        assert_eq!(tensor.n_rows, n_rows);
        assert_eq!(tensor.k_cols, k_cols);

        let act = vec![0.5f32; k_cols];
        let out = TmacCpuBackend::new()
            .matmul_hadamard_ternary(
                &act,
                &tensor.packed_weights,
                &tensor.scales_fp16,
                1,
                tensor.n_rows,
                tensor.k_cols,
            )
            .expect("kernel execution");
        assert_eq!(out.len(), n_rows);
    }
}
