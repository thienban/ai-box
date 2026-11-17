//! SERV-02 : Loader SafeTensors et mapping direct vers `TernaryTensor` (`KERN-01`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use aibox_kernels::tensor::{f16_to_f32, Q2AffineTensor, Q4AffineTensor, TernaryTensor};
use aibox_kernels::{
    BINARY_PACKED_BYTES_PER_GROUP, GROUP_SIZE, PACKED_BYTES_PER_GROUP, Q2_GROUP_SIZE,
    Q2_PACKED_BYTES_PER_GROUP, Q4_GROUP_SIZE, Q4_PACKED_BYTES_PER_GROUP,
};

use crate::json::{parse_json, JsonValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (usize, usize),
    pub shard_idx: usize,
}

/// Lecteur SafeTensors basé sur un ou plusieurs fichiers disque (lecture sélective par offset) ou tampon mémoire.
pub struct SafeTensorsFile {
    tensors: BTreeMap<String, TensorMeta>,
    source: SafeTensorsSource,
}

enum SafeTensorsSource {
    Files(Vec<(u64, File)>),
    #[cfg(test)]
    Memory { data_start: u64, buf: Vec<u8> },
}

/// Nombre maximal de lectures disque simultanées pour `read_tensor_slices_parallel`.
const MAX_PARALLEL_READS: usize = 8;

/// Lecture positionnée complète sans déplacer le curseur du fichier (`&self`, thread-safe).
fn positioned_read_exact(file: &File, offset: u64, out: &mut [u8]) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(out, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < out.len() {
            let read = file.seek_read(&mut out[done..], offset + done as u64)?;
            if read == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "positioned read reached EOF",
                ));
            }
            done += read;
        }
        Ok(())
    }
}

fn read_shard_header(path: &Path) -> Result<(u64, File, BTreeMap<String, TensorMeta>), String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open '{}': {e}", path.display()))?;
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
    Ok((8 + header_len as u64, file, tensors))
}

impl SafeTensorsFile {
    /// Ouvre `model.safetensors` ou tous les shards `model-*-of-*.safetensors` d'un dossier.
    pub fn open_dir<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Self::open(single);
        }
        let mut shard_paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("Failed to read '{}': {e}", dir.display()))?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|s| s.starts_with("model-") && s.ends_with(".safetensors"))
            })
            .collect();
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(format!("No model.safetensors or shards found in '{}'", dir.display()));
        }
        let mut files = Vec::with_capacity(shard_paths.len());
        let mut all_tensors = BTreeMap::new();
        for (shard_idx, path) in shard_paths.iter().enumerate() {
            let (data_start, file, tensors) = read_shard_header(path)?;
            files.push((data_start, file));
            for (name, mut meta) in tensors {
                meta.shard_idx = shard_idx;
                all_tensors.insert(name, meta);
            }
        }
        Ok(Self {
            tensors: all_tensors,
            source: SafeTensorsSource::Files(files),
        })
    }

    /// Ouvre un fichier `.safetensors` et parse uniquement son en-tête JSON (démarrage instantané).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let (data_start, file, tensors) = read_shard_header(path.as_ref())?;
        Ok(Self {
            tensors,
            source: SafeTensorsSource::Files(vec![(data_start, file)]),
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
            tensors,
            source: SafeTensorsSource::Memory {
                data_start: header_end as u64,
                buf: bytes,
            },
        })
    }

    pub fn tensors(&self) -> &BTreeMap<String, TensorMeta> {
        &self.tensors
    }

    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// Lit une tranche d'octets `[byte_offset .. byte_offset + len]` d'un tenseur nommé (streaming disque).
    pub fn read_tensor_slice(&mut self, name: &str, byte_offset: usize, len: usize) -> Result<Vec<u8>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("Tensor '{name}' not found in safetensors"))?;
        let (start, end) = meta.data_offsets;
        if end < start || byte_offset + len > (end - start) {
            return Err(format!(
                "Slice [{byte_offset}..{}] out of bounds for '{name}' (len {})",
                byte_offset + len,
                end.saturating_sub(start)
            ));
        }
        let shard_idx = meta.shard_idx;
        let abs_offset = start + byte_offset;
        match &mut self.source {
            #[cfg(test)]
            SafeTensorsSource::Memory { data_start, buf } => {
                let s = *data_start as usize + abs_offset;
                Ok(buf[s..s + len].to_vec())
            }
            SafeTensorsSource::Files(files) => {
                let (data_start, file) = &mut files[shard_idx];
                file.seek(SeekFrom::Start(*data_start + abs_offset as u64))
                    .map_err(|e| format!("Seek failed for '{name}': {e}"))?;
                let mut out = vec![0u8; len];
                file.read_exact(&mut out)
                    .map_err(|e| format!("Read failed for '{name}': {e}"))?;
                Ok(out)
            }
        }
    }

    /// Lit plusieurs tranches `(nom, offset_relatif, longueur)` en parallèle.
    ///
    /// Utilise des lectures positionnées (curseur fichier non partagé) : chaque tranche est
    /// réellement lue en simultané. Réduit la latence I/O sur les systèmes de fichiers à
    /// forte latence par appel (bind mounts conteneurs, réseau, virtiofs).
    pub fn read_tensor_slices_parallel(
        &self,
        reqs: &[(&str, usize, usize)],
    ) -> Result<Vec<Vec<u8>>, String> {
        let mut resolved: Vec<(usize, u64, usize)> = Vec::with_capacity(reqs.len());
        for &(name, byte_offset, len) in reqs {
            let meta = self
                .tensors
                .get(name)
                .ok_or_else(|| format!("Tensor '{name}' not found in safetensors"))?;
            let (start, end) = meta.data_offsets;
            if end < start || byte_offset + len > (end - start) {
                return Err(format!(
                    "Slice [{byte_offset}..{}] out of bounds for '{name}'",
                    byte_offset + len
                ));
            }
            resolved.push((meta.shard_idx, (start + byte_offset) as u64, len));
        }
        let files = match &self.source {
            SafeTensorsSource::Files(files) => files,
            #[cfg(test)]
            SafeTensorsSource::Memory { data_start, buf } => {
                // Source en mémoire : lecture séquentielle directe (pas d'I/O à paralléliser).
                let data_start = *data_start as usize;
                let mut outs = Vec::with_capacity(resolved.len());
                for &(_, offset, len) in &resolved {
                    let s = data_start + offset as usize;
                    outs.push(buf[s..s + len].to_vec());
                }
                return Ok(outs);
            }
        };
        let n = resolved.len();
        let n_threads = MAX_PARALLEL_READS.min(n);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let resolved_ref = &resolved;
        let files_ref = files;
        let collected: Vec<Vec<(usize, Vec<u8>)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let next = &next;
                    scope.spawn(move || {
                        let mut local: Vec<(usize, Vec<u8>)> = Vec::new();
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= n {
                                break;
                            }
                            let (shard_idx, offset, len) = resolved_ref[i];
                            let mut out = vec![0u8; len];
                            // `offset` est relatif à la section de données du shard : il faut
                            // ajouter `data_start` (fin de l'en-tête), comme `read_tensor_slice`.
                            let data_start = files_ref[shard_idx].0;
                            positioned_read_exact(&files_ref[shard_idx].1, data_start + offset, &mut out)
                                .map_err(|e| format!("Read failed: {e}"))?;
                            local.push((i, out));
                        }
                        Ok::<_, String>(local)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| Err("parallel read thread panicked".to_string()))
                })
                .collect::<Result<Vec<_>, String>>()
        })?;
        let mut outs: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        for part in collected {
            for (i, data) in part {
                outs[i] = Some(data);
            }
        }
        outs.into_iter()
            .map(|o| o.ok_or_else(|| "missing parallel read result".to_string()))
            .collect()
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
        let shard_idx = meta.shard_idx;
        let len = end - start;

        match &mut self.source {
            #[cfg(test)]
            SafeTensorsSource::Memory { data_start, buf } => {
                let abs_start = *data_start as usize + start;
                let abs_end = *data_start as usize + end;
                if abs_end > buf.len() {
                    return Err(format!("Tensor '{name}' exceeds buffer size"));
                }
                Ok(buf[abs_start..abs_end].to_vec())
            }
            SafeTensorsSource::Files(files) => {
                let (data_start, file) = &mut files[shard_idx];
                file.seek(SeekFrom::Start(*data_start + start as u64))
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
        let bytes_per_group = w_meta.shape[1] / n_groups;
        let bits = match bytes_per_group {
            BINARY_PACKED_BYTES_PER_GROUP if w_meta.shape[1] == n_groups * BINARY_PACKED_BYTES_PER_GROUP => 1u8,
            PACKED_BYTES_PER_GROUP if w_meta.shape[1] == n_groups * PACKED_BYTES_PER_GROUP => 2u8,
            64 if w_meta.shape[1] == n_groups * 64 => 4u8,
            128 if w_meta.shape[1] == n_groups * 128 => 8u8,
            _ => {
                return Err(format!(
                    "Shape mismatch between '{weight_name}' {:?} and '{scale_name}' {:?} (expected 16, 32, 64 or 128 bytes/group)",
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

    /// Charge une couche linéaire 2-bit affine (`U8` `[N, K/4]` + `F16` scales/zeros `[N, K/128]`).
    pub fn load_q2_affine_layer(
        &mut self,
        weight_name: &str,
        scale_name: &str,
        zero_name: &str,
    ) -> Result<Q2AffineTensor, String> {
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
        let z_meta = self
            .tensors
            .get(zero_name)
            .ok_or_else(|| format!("Missing zero tensor '{zero_name}'"))?
            .clone();

        if w_meta.dtype != "U8" || w_meta.shape.len() != 2 {
            return Err(format!("Weight '{weight_name}' must be 2D U8"));
        }
        if s_meta.dtype != "F16" || z_meta.dtype != "F16" || s_meta.shape != z_meta.shape || s_meta.shape.len() != 2 {
            return Err(format!("Scale/zero '{scale_name}'/'{zero_name}' must be matching 2D F16"));
        }
        let n_rows = w_meta.shape[0];
        let n_groups = s_meta.shape[1];
        if s_meta.shape[0] != n_rows || n_groups == 0 || w_meta.shape[1] != n_groups * Q2_PACKED_BYTES_PER_GROUP {
            return Err(format!("Shape mismatch for q2_affine '{weight_name}'"));
        }
        let k_cols = n_groups * Q2_GROUP_SIZE;
        let packed_weights = self.read_tensor_bytes(weight_name)?;
        let scale_bytes = self.read_tensor_bytes(scale_name)?;
        let zero_bytes = self.read_tensor_bytes(zero_name)?;
        let scales_fp16 = scale_bytes
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        let zeros_fp16 = zero_bytes
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        Q2AffineTensor::from_raw(n_rows, k_cols, packed_weights, scales_fp16, zeros_fp16)
            .map_err(|e| format!("Invalid Q2AffineTensor '{weight_name}': {e}"))
    }

    /// Charge une couche linéaire 4-bit affine (`U8` `[N, K/2]` + `F16` scales/zeros `[N, K/64]`).
    pub fn load_q4_affine_layer(
        &mut self,
        weight_name: &str,
        scale_name: &str,
        zero_name: &str,
    ) -> Result<Q4AffineTensor, String> {
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
        let z_meta = self
            .tensors
            .get(zero_name)
            .ok_or_else(|| format!("Missing zero tensor '{zero_name}'"))?
            .clone();

        if w_meta.dtype != "U8" || w_meta.shape.len() != 2 {
            return Err(format!("Weight '{weight_name}' must be 2D U8"));
        }
        if s_meta.dtype != "F16" || z_meta.dtype != "F16" || s_meta.shape != z_meta.shape || s_meta.shape.len() != 2 {
            return Err(format!("Scale/zero '{scale_name}'/'{zero_name}' must be matching 2D F16"));
        }
        let n_rows = w_meta.shape[0];
        let n_groups = s_meta.shape[1];
        if s_meta.shape[0] != n_rows || n_groups == 0 || w_meta.shape[1] != n_groups * Q4_PACKED_BYTES_PER_GROUP {
            return Err(format!("Shape mismatch for q4_affine '{weight_name}'"));
        }
        let k_cols = n_groups * Q4_GROUP_SIZE;
        let packed_weights = self.read_tensor_bytes(weight_name)?;
        let scale_bytes = self.read_tensor_bytes(scale_name)?;
        let zero_bytes = self.read_tensor_bytes(zero_name)?;
        let scales_fp16 = scale_bytes
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        let zeros_fp16 = zero_bytes
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        Q4AffineTensor::from_raw(n_rows, k_cols, packed_weights, scales_fp16, zeros_fp16)
            .map_err(|e| format!("Invalid Q4AffineTensor '{weight_name}': {e}"))
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
                shard_idx: 0,
            },
        );
    }

    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows};
    use aibox_kernels::{ComputeBackend, TmacCpuBackend};

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
