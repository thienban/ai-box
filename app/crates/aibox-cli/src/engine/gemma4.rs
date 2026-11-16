//! Module isolé pour l'architecture `gemma4` (MoE Top-K, GeGLU, Sliding-Window Attention,
//! Per-Layer Embeddings et quantification 2-bit affine `q2_affine`).
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aibox_kernels::scalar::{matmul_q2_affine, matmul_q4_affine};
use aibox_kernels::tensor::{Q2AffineTensor, Q4AffineTensor};
use aibox_kernels::{GpuBackend, TernaryHadamardBackend};

use super::attention::rms_norm_heads_inplace;
use super::{add_residual, rms_norm, EmbeddingTable, GenerationStats, KvCache, LmHead, ModelConfig};
use crate::json::{parse_json, JsonValue};
use crate::safetensors::SafeTensorsFile;
use crate::sampling::{sample_token, SamplingParams, XorShift64};

/// Configuration spécifique à Gemma 4 MoE (`config.json` -> `"architecture": "gemma4"`).
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub base: ModelConfig,
    pub sliding_window: usize,
    pub global_attn_every: usize,
    pub head_dim_swa: usize,
    pub rope_dim_swa: usize,
    pub rope_theta_swa: f32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub has_shared_expert: bool,
    pub ple_dim: usize,
}

impl Default for Gemma4Config {
    fn default() -> Self {
        Self {
            base: ModelConfig::default(),
            sliding_window: 1024,
            global_attn_every: 6,
            head_dim_swa: 0,
            rope_dim_swa: 0,
            rope_theta_swa: 10_000.0,
            num_experts: 128,
            num_experts_per_tok: 8,
            moe_intermediate_size: 704,
            has_shared_expert: true,
            ple_dim: 0,
        }
    }
}

impl Gemma4Config {
    /// Détecte `"architecture": "gemma4"` dans `config.json`.
    pub fn is_gemma4_json(val: &JsonValue) -> bool {
        val.as_object()
            .and_then(|o| o.get("architecture").or_else(|| o.get("model_type")))
            .and_then(JsonValue::as_str)
            .is_some_and(|s| s.eq_ignore_ascii_case("gemma4"))
    }

    /// Construit `Gemma4Config` depuis un objet JSON `config.json`.
    pub fn from_json(val: &JsonValue) -> Result<Self, String> {
        let base = ModelConfig::from_json(val)?;
        let obj = val
            .as_object()
            .ok_or_else(|| "config.json must be a JSON object".to_string())?;
        let get_opt = |key: &str, default: usize| -> usize {
            obj.get(key)
                .and_then(JsonValue::as_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        Ok(Self {
            sliding_window: get_opt("sliding_window", 1024),
            global_attn_every: get_opt("global_attn_every", 6),
            head_dim_swa: get_opt("head_dim_swa", 0),
            rope_dim_swa: get_opt("rope_dim_swa", 0),
            rope_theta_swa: match obj.get("rope_theta_swa") {
                Some(JsonValue::Number(n)) => *n as f32,
                _ => base.rope_theta,
            },
            num_experts: get_opt("num_experts", 0),
            num_experts_per_tok: get_opt("num_experts_per_tok", 8),
            moe_intermediate_size: get_opt("moe_intermediate_size", base.intermediate_size),
            has_shared_expert: obj
                .get("has_shared_expert")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true),
            ple_dim: get_opt("ple_dim", 0),
            base,
        })
    }
}

/// Poids linéaires supportés par `gemma4` (`q2_affine` 2-bit ou `f32` dense pour petits tenseurs/routeur).
#[derive(Debug, Clone)]
pub enum Gemma4Linear {
    Q2Affine(Q2AffineTensor),
    Q4Affine(Q4AffineTensor),
    Dense {
        rows: usize,
        cols: usize,
        weights: Vec<f32>,
    },
}

impl Gemma4Linear {
    pub fn forward(&self, input: &[f32], backend: &dyn TernaryHadamardBackend) -> Result<Vec<f32>, String> {
        if let Some(gpu) = backend.as_gpu() {
            match self {
                Self::Q2Affine(t) => {
                    let mut x_dev = gpu.alloc_f16(t.k_cols).map_err(|e| e.to_string())?;
                    x_dev.upload_f32_as_f16(input).map_err(|e| e.to_string())?;
                    let mut out_dev = gpu.alloc_f16(t.n_rows).map_err(|e| e.to_string())?;
                    gpu.matmul_q2_affine_f16_dev(
                        &x_dev,
                        &t.packed_weights,
                        &t.scales_fp16,
                        &t.zeros_fp16,
                        &mut out_dev,
                        1,
                        t.n_rows,
                        t.k_cols,
                    )
                    .map_err(|e| e.to_string())?;
                    let mut out = vec![0.0f32; t.n_rows];
                    out_dev.download_as_f32(&mut out).map_err(|e| e.to_string())?;
                    return Ok(out);
                }
                Self::Q4Affine(t) => {
                    let mut x_dev = gpu.alloc_f16(t.k_cols).map_err(|e| e.to_string())?;
                    x_dev.upload_f32_as_f16(input).map_err(|e| e.to_string())?;
                    let mut out_dev = gpu.alloc_f16(t.n_rows).map_err(|e| e.to_string())?;
                    gpu.matmul_q4_affine_f16_dev(
                        &x_dev,
                        &t.packed_weights,
                        &t.scales_fp16,
                        &t.zeros_fp16,
                        &mut out_dev,
                        1,
                        t.n_rows,
                        t.k_cols,
                    )
                    .map_err(|e| e.to_string())?;
                    let mut out = vec![0.0f32; t.n_rows];
                    out_dev.download_as_f32(&mut out).map_err(|e| e.to_string())?;
                    return Ok(out);
                }
                Self::Dense { .. } => {}
            }
        }
        self.forward_cpu(input)
    }

    fn forward_q4_into_dev(
        &self,
        gpu: &GpuBackend,
        x_dev: &aibox_kernels::GpuBuffer<'_, u16>,
        out_dev: &mut aibox_kernels::GpuBuffer<'_, u16>,
    ) -> Result<usize, String> {
        let Self::Q4Affine(t) = self else {
            return Err("forward_q4_into_dev requires Q4Affine".to_string());
        };
        gpu.matmul_q4_affine_f16_dev(
            x_dev,
            &t.packed_weights,
            &t.scales_fp16,
            &t.zeros_fp16,
            out_dev,
            1,
            t.n_rows,
            t.k_cols,
        )
        .map_err(|e| e.to_string())?;
        Ok(t.n_rows)
    }

    fn forward_with_scratch(
        &self,
        input: &[f32],
        scratch: Option<&mut Gemma4GpuScratch<'_>>,
        backend: &dyn TernaryHadamardBackend,
        resident: bool,
    ) -> Result<Vec<f32>, String> {
        if let Some(sc) = scratch {
            match self {
                Self::Q2Affine(t) => {
                    sc.x_dev.upload_f32_as_f16(input).map_err(|e| e.to_string())?;
                    sc.gpu
                        .matmul_q2_affine_f16_dev(
                            &sc.x_dev,
                            &t.packed_weights,
                            &t.scales_fp16,
                            &t.zeros_fp16,
                            &mut sc.out_dev,
                            1,
                            t.n_rows,
                            t.k_cols,
                        )
                        .map_err(|e| e.to_string())?;
                    let mut out = vec![0.0f32; t.n_rows];
                    sc.out_dev.download_as_f32(&mut out).map_err(|e| e.to_string())?;
                    return Ok(out);
                }
                Self::Q4Affine(t) => {
                    sc.x_dev.upload_f32_as_f16(input).map_err(|e| e.to_string())?;
                    let s_bytes = std::mem::size_of_val(t.scales_fp16.as_slice());
                    let needed = t.packed_weights.len() + 2 * s_bytes;
                    let can_stream = !resident
                        && t.packed_weights.len() <= sc.stage_w.len()
                        && t.scales_fp16.len() <= sc.stage_s.len()
                        && needed <= sc.pinned_cap;
                    if can_stream {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                t.packed_weights.as_ptr(),
                                sc.pinned_stage,
                                t.packed_weights.len(),
                            );
                            std::ptr::copy_nonoverlapping(
                                t.scales_fp16.as_ptr() as *const u8,
                                sc.pinned_stage.add(t.packed_weights.len()),
                                s_bytes,
                            );
                            std::ptr::copy_nonoverlapping(
                                t.zeros_fp16.as_ptr() as *const u8,
                                sc.pinned_stage.add(t.packed_weights.len() + s_bytes),
                                s_bytes,
                            );
                            sc.gpu
                                .matmul_q4_affine_f16_streaming_dev(
                                    &sc.x_dev,
                                    sc.pinned_stage,
                                    t.packed_weights.len(),
                                    t.scales_fp16.len(),
                                    &mut sc.stage_w,
                                    &mut sc.stage_s,
                                    &mut sc.stage_z,
                                    &mut sc.out_dev,
                                    1,
                                    t.n_rows,
                                    t.k_cols,
                                )
                                .map_err(|e| e.to_string())?;
                        }
                    } else {
                        sc.gpu
                            .matmul_q4_affine_f16_dev(
                                &sc.x_dev,
                                &t.packed_weights,
                                &t.scales_fp16,
                                &t.zeros_fp16,
                                &mut sc.out_dev,
                                1,
                                t.n_rows,
                                t.k_cols,
                            )
                            .map_err(|e| e.to_string())?;
                    }
                    let mut out = vec![0.0f32; t.n_rows];
                    sc.out_dev.download_as_f32(&mut out).map_err(|e| e.to_string())?;
                    return Ok(out);
                }
                Self::Dense { .. } => {}
            }
        }
        self.forward(input, backend)
    }

    pub fn forward_cpu(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        match self {
            Self::Q2Affine(t) => matmul_q2_affine(
                input,
                &t.packed_weights,
                &t.scales_fp16,
                &t.zeros_fp16,
                1,
                t.n_rows,
                t.k_cols,
            )
            .map_err(|e| e.to_string()),
            Self::Q4Affine(t) => matmul_q4_affine(
                input,
                &t.packed_weights,
                &t.scales_fp16,
                &t.zeros_fp16,
                1,
                t.n_rows,
                t.k_cols,
            )
            .map_err(|e| e.to_string()),
            Self::Dense {
                rows,
                cols,
                weights,
            } => {
                if input.len() != *cols {
                    return Err(format!("Dense input len {} != cols {cols}", input.len()));
                }
                let mut out = vec![0.0f32; *rows];
                for (r, out_val) in out.iter_mut().enumerate() {
                    let row = &weights[r * cols..(r + 1) * cols];
                    *out_val = row.iter().zip(input.iter()).map(|(&w, &x)| w * x).sum();
                }
                Ok(out)
            }
        }
    }

    #[allow(dead_code)]
    pub fn forward_gpu_f16(
        &self,
        gpu: &GpuBackend,
        x_dev: &aibox_kernels::GpuBuffer<'_, u16>,
        out_dev: &mut aibox_kernels::GpuBuffer<'_, u16>,
    ) -> Result<(), String> {
        match self {
            Self::Q2Affine(t) => gpu
                .matmul_q2_affine_f16_dev(
                    x_dev,
                    &t.packed_weights,
                    &t.scales_fp16,
                    &t.zeros_fp16,
                    out_dev,
                    1,
                    t.n_rows,
                    t.k_cols,
                )
                .map_err(|e| e.to_string()),
            Self::Q4Affine(t) => gpu
                .matmul_q4_affine_f16_dev(
                    x_dev,
                    &t.packed_weights,
                    &t.scales_fp16,
                    &t.zeros_fp16,
                    out_dev,
                    1,
                    t.n_rows,
                    t.k_cols,
                )
                .map_err(|e| e.to_string()),
            Self::Dense { .. } => Err("Dense Gemma4Linear GPU path requires Q2Affine/Q4Affine".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Gemma4ExpertWeights {
    pub gate: Gemma4Linear,
    pub up: Gemma4Linear,
    pub down: Gemma4Linear,
}

#[derive(Debug, Clone)]
pub struct LazyMoeExperts {
    pub layer_idx: usize,
    pub inter_rows: usize,
    pub down_rows: usize,
    pub hidden_size: usize,
    pub is_q4: bool,
}

#[derive(Debug, Clone)]
pub enum Gemma4Experts {
    InMemory(Vec<Gemma4ExpertWeights>),
    LazyDisk(LazyMoeExperts),
}

#[derive(Debug, Clone)]
pub struct Gemma4MoeWeights {
    pub router: Vec<f32>,
    pub router_scale: Option<Vec<f32>>,
    pub expert_scales: Option<Vec<f32>>,
    pub experts: Gemma4Experts,
    pub shared_expert: Option<Gemma4ExpertWeights>,
}

fn slice_q2_affine_expert(
    stacked: &Q2AffineTensor,
    expert_idx: usize,
    rows_per_expert: usize,
) -> Result<Q2AffineTensor, String> {
    let row_bytes = stacked.row_stride_bytes();
    let groups_per_row = stacked.k_cols / aibox_kernels::Q2_GROUP_SIZE;
    let w_start = expert_idx * rows_per_expert * row_bytes;
    let w_end = w_start + rows_per_expert * row_bytes;
    let s_start = expert_idx * rows_per_expert * groups_per_row;
    let s_end = s_start + rows_per_expert * groups_per_row;
    Q2AffineTensor::from_raw(
        rows_per_expert,
        stacked.k_cols,
        stacked.packed_weights[w_start..w_end].to_vec(),
        stacked.scales_fp16[s_start..s_end].to_vec(),
        stacked.zeros_fp16[s_start..s_end].to_vec(),
    )
    .map_err(|e| e.to_string())
}

#[derive(Debug, Clone)]
pub enum Gemma4Ffn {
    Dense(Gemma4ExpertWeights),
    Moe(Gemma4MoeWeights),
}

#[derive(Debug, Clone)]
pub struct Gemma4Layer {
    pub attn_norm: Vec<f32>,
    pub attn_q: Gemma4Linear,
    pub attn_k: Gemma4Linear,
    pub attn_v: Option<Gemma4Linear>,
    pub attn_o: Gemma4Linear,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub is_global_attn: bool,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub ffn_norm: Vec<f32>,
    pub post_attention_norm: Option<Vec<f32>>,
    pub post_ffw_norm: Option<Vec<f32>>,
    pub pre_ffw_norm_2: Option<Vec<f32>>,
    pub post_ffw_norm_1: Option<Vec<f32>>,
    pub post_ffw_norm_2: Option<Vec<f32>>,
    pub layer_output_scale: Option<f32>,
    pub ffn: Gemma4Ffn,
    pub ple_weights: Option<Vec<f32>>,
}

pub struct LruMap<K, V> {
    capacity: usize,
    map: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Copy + Eq + std::hash::Hash, V> LruMap<K, V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            map: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if !self.map.contains_key(key) {
            return None;
        }
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
            self.order.push_back(*key);
        }
        self.map.get(key)
    }

    pub fn pop_oldest_if_full(&mut self) -> Option<V> {
        if self.order.len() < self.capacity {
            return None;
        }
        let oldest = self.order.pop_front()?;
        self.map.remove(&oldest)
    }

    pub fn insert(&mut self, key: K, val: V) {
        if self.map.contains_key(&key) {
            self.map.insert(key, val);
            return;
        }
        let _ = self.pop_oldest_if_full();
        self.order.push_back(key);
        self.map.insert(key, val);
    }
}

pub struct Gemma4Model {
    pub config: Gemma4Config,
    pub token_embeddings: EmbeddingTable,
    pub layers: Vec<Gemma4Layer>,
    pub final_norm: Vec<f32>,
    pub lm_head: LmHead,
    pub rope_freqs: Option<Vec<f32>>,
    pub safetensors: Option<Mutex<SafeTensorsFile>>,
    pub expert_ram_cache: Mutex<LruMap<(usize, usize), Arc<Gemma4ExpertWeights>>>,
}

fn load_affine_slice_from_st(
    st: &mut SafeTensorsFile,
    prefix: &str,
    expert_idx: usize,
    rows_per_expert: usize,
    k_cols: usize,
    is_q4: bool,
) -> Result<Gemma4Linear, String> {
    let bytes_per_group = if is_q4 {
        aibox_kernels::Q4_PACKED_BYTES_PER_GROUP
    } else {
        aibox_kernels::Q2_PACKED_BYTES_PER_GROUP
    };
    let groups_per_row = k_cols / aibox_kernels::Q2_GROUP_SIZE;
    let w_bytes = rows_per_expert * groups_per_row * bytes_per_group;
    let s_elems = rows_per_expert * groups_per_row;
    let s_bytes = s_elems * 2;
    let w_raw = st.read_tensor_slice(&format!("{prefix}.weight"), expert_idx * w_bytes, w_bytes)?;
    let s_raw = st.read_tensor_slice(&format!("{prefix}.scales"), expert_idx * s_bytes, s_bytes)?;
    let z_raw = st.read_tensor_slice(&format!("{prefix}.zeros"), expert_idx * s_bytes, s_bytes)?;
    let scales_fp16: Vec<u16> = s_raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
    let zeros_fp16: Vec<u16> = z_raw.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
    if is_q4 {
        Ok(Gemma4Linear::Q4Affine(
            Q4AffineTensor::from_raw(rows_per_expert, k_cols, w_raw, scales_fp16, zeros_fp16)
                .map_err(|e| e.to_string())?,
        ))
    } else {
        Ok(Gemma4Linear::Q2Affine(
            Q2AffineTensor::from_raw(rows_per_expert, k_cols, w_raw, scales_fp16, zeros_fp16)
                .map_err(|e| e.to_string())?,
        ))
    }
}

struct VramLinearBufs<'a> {
    w: aibox_kernels::GpuBuffer<'a, u8>,
    s: aibox_kernels::GpuBuffer<'a, u16>,
    z: aibox_kernels::GpuBuffer<'a, u16>,
    n_rows: usize,
    k_cols: usize,
    is_q4: bool,
}

impl<'a> VramLinearBufs<'a> {
    fn extract_affine_parts(lin: &Gemma4Linear) -> Result<(usize, usize, &[u8], &[u16], &[u16], bool), String> {
        match lin {
            Gemma4Linear::Q4Affine(t) => Ok((t.n_rows, t.k_cols, &t.packed_weights, &t.scales_fp16, &t.zeros_fp16, true)),
            Gemma4Linear::Q2Affine(t) => Ok((t.n_rows, t.k_cols, &t.packed_weights, &t.scales_fp16, &t.zeros_fp16, false)),
            Gemma4Linear::Dense { .. } => Err("VramExpertLru requires Q2Affine or Q4Affine".to_string()),
        }
    }

    fn from_linear(gpu: &'a GpuBackend, lin: &Gemma4Linear) -> Result<Self, String> {
        let (n_rows, k_cols, pw, sf, zf, is_q4) = Self::extract_affine_parts(lin)?;
        let mut w = gpu.alloc_u8(pw.len()).map_err(|e| e.to_string())?;
        let mut s = gpu.alloc_u16(sf.len()).map_err(|e| e.to_string())?;
        let mut z = gpu.alloc_u16(zf.len()).map_err(|e| e.to_string())?;
        gpu.upload_u8_slice(&mut w, pw).map_err(|e| e.to_string())?;
        gpu.upload_u16_slice(&mut s, sf).map_err(|e| e.to_string())?;
        gpu.upload_u16_slice(&mut z, zf).map_err(|e| e.to_string())?;
        Ok(Self {
            w,
            s,
            z,
            n_rows,
            k_cols,
            is_q4,
        })
    }

    fn overwrite_linear(&mut self, gpu: &'a GpuBackend, lin: &Gemma4Linear) -> Result<(), String> {
        let (n_rows, k_cols, pw, sf, zf, is_q4) = Self::extract_affine_parts(lin)?;
        gpu.upload_u8_slice(&mut self.w, pw).map_err(|e| e.to_string())?;
        gpu.upload_u16_slice(&mut self.s, sf).map_err(|e| e.to_string())?;
        gpu.upload_u16_slice(&mut self.z, zf).map_err(|e| e.to_string())?;
        self.n_rows = n_rows;
        self.k_cols = k_cols;
        self.is_q4 = is_q4;
        Ok(())
    }

    fn forward_into_dev(
        &self,
        gpu: &GpuBackend,
        x_dev: &aibox_kernels::GpuBuffer<'_, u16>,
        out_dev: &mut aibox_kernels::GpuBuffer<'_, u16>,
    ) -> Result<(), String> {
        if self.is_q4 {
            gpu.matmul_q4_affine_buffers_dev(x_dev, &self.w, &self.s, &self.z, out_dev, 1, self.n_rows, self.k_cols)
        } else {
            gpu.matmul_q2_affine_buffers_dev(x_dev, &self.w, &self.s, &self.z, out_dev, 1, self.n_rows, self.k_cols)
        }
        .map_err(|e| e.to_string())
    }
}

struct VramExpertEntry<'a> {
    gate: VramLinearBufs<'a>,
    up: VramLinearBufs<'a>,
    down: VramLinearBufs<'a>,
}

fn insert_or_reuse_vram_expert<'a>(
    cache: &mut LruMap<(usize, usize), VramExpertEntry<'a>>,
    gpu: &'a GpuBackend,
    key: (usize, usize),
    exp: &Gemma4ExpertWeights,
) -> Result<&'a VramExpertEntry<'a>, String> {
    let slot = if let Some(mut reused) = cache.pop_oldest_if_full() {
        reused.gate.overwrite_linear(gpu, &exp.gate)?;
        reused.up.overwrite_linear(gpu, &exp.up)?;
        reused.down.overwrite_linear(gpu, &exp.down)?;
        reused
    } else {
        VramExpertEntry {
            gate: VramLinearBufs::from_linear(gpu, &exp.gate)?,
            up: VramLinearBufs::from_linear(gpu, &exp.up)?,
            down: VramLinearBufs::from_linear(gpu, &exp.down)?,
        }
    };
    cache.insert(key, slot);
    // Safety: caller borrows `cache` mutably and only reads the returned entry before mutating `cache` again.
    let ptr: *const VramExpertEntry<'a> = &cache.map[&key];
    Ok(unsafe { &*ptr })
}

struct Gemma4GpuScratch<'a> {
    gpu: &'a GpuBackend,
    x_dev: aibox_kernels::GpuBuffer<'a, u16>,
    out_dev: aibox_kernels::GpuBuffer<'a, u16>,
    gate_dev: aibox_kernels::GpuBuffer<'a, u16>,
    up_dev: aibox_kernels::GpuBuffer<'a, u16>,
    moe_acc_dev: aibox_kernels::GpuBuffer<'a, f32>,
    logits_dev: aibox_kernels::GpuBuffer<'a, f32>,
    stage_w: aibox_kernels::GpuBuffer<'a, u8>,
    stage_s: aibox_kernels::GpuBuffer<'a, u16>,
    stage_z: aibox_kernels::GpuBuffer<'a, u16>,
    pinned_stage: *mut u8,
    pinned_cap: usize,
    vram_experts: LruMap<(usize, usize), VramExpertEntry<'a>>,
    vram_hits: usize,
    ram_hits: usize,
    disk_misses: usize,
}

impl Drop for Gemma4GpuScratch<'_> {
    fn drop(&mut self) {
        if !self.pinned_stage.is_null() {
            unsafe { self.gpu.free_pinned_bytes(self.pinned_stage) };
        }
    }
}

impl<'a> Gemma4GpuScratch<'a> {
    fn new(gpu: &'a GpuBackend, cfg: &Gemma4Config, vram_expert_cap: usize) -> Result<Self, String> {
        let max_dim = cfg
            .base
            .intermediate_size
            .max(cfg.moe_intermediate_size)
            .max(cfg.base.hidden_size * 4)
            .max(8192);
        let stage_w_bytes = 16 * 1024 * 1024;
        let stage_s_elems = 512 * 1024;
        let pinned_cap = stage_w_bytes + 2 * stage_s_elems * std::mem::size_of::<u16>();
        let pinned_stage = gpu.alloc_pinned_bytes(pinned_cap).map_err(|e| e.to_string())?;
        Ok(Self {
            gpu,
            x_dev: gpu.alloc_f16(max_dim).map_err(|e| e.to_string())?,
            out_dev: gpu.alloc_f16(max_dim).map_err(|e| e.to_string())?,
            gate_dev: gpu.alloc_f16(max_dim).map_err(|e| e.to_string())?,
            up_dev: gpu.alloc_f16(max_dim).map_err(|e| e.to_string())?,
            moe_acc_dev: gpu.alloc_f32(cfg.base.hidden_size).map_err(|e| e.to_string())?,
            logits_dev: gpu.alloc_f32(cfg.base.vocab_size).map_err(|e| e.to_string())?,
            stage_w: gpu.alloc_u8(stage_w_bytes).map_err(|e| e.to_string())?,
            stage_s: gpu.alloc_u16(stage_s_elems).map_err(|e| e.to_string())?,
            stage_z: gpu.alloc_u16(stage_s_elems).map_err(|e| e.to_string())?,
            pinned_stage,
            pinned_cap,
            vram_experts: LruMap::new(vram_expert_cap),
            vram_hits: 0,
            ram_hits: 0,
            disk_misses: 0,
        })
    }
}

fn apply_rope_with_freqs_inplace(
    vec: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    theta: f32,
    rope_freqs: Option<&[f32]>,
) {
    let half_rope = rope_dim / 2;
    let pos_f = pos as f32;
    for head in 0..num_heads {
        let base = head * head_dim;
        for i in 0..half_rope {
            let freq = match rope_freqs.and_then(|rf| rf.get(i).copied()) {
                Some(f) if f.is_infinite() || f.abs() > 1e20 => 0.0,
                Some(f) => 1.0 / (theta.powf((2 * i) as f32 / (rope_dim as f32)) * f),
                None => 1.0 / theta.powf((2 * i) as f32 / (rope_dim as f32)),
            };
            let angle = pos_f * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = vec[base + i];
            let x1 = vec[base + i + half_rope];
            vec[base + i] = x0 * cos_a - x1 * sin_a;
            vec[base + i + half_rope] = x0 * sin_a + x1 * cos_a;
        }
    }
}

#[inline]
pub fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.79788456;
    const COEFF: f32 = 0.044715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Routeur MoE Top-K avec softmax normalisé sur les `top_k` meilleurs experts.
pub fn route_top_k(
    hidden: &[f32],
    router: &[f32],
    num_experts: usize,
    top_k: usize,
) -> Result<Vec<(usize, f32)>, String> {
    let hidden_size = hidden.len();
    if router.len() != num_experts * hidden_size {
        return Err(format!(
            "router len {} != num_experts*hidden_size ({})",
            router.len(),
            num_experts * hidden_size
        ));
    }
    let k = top_k.min(num_experts).max(1);
    let mut scores: Vec<(usize, f32)> = (0..num_experts)
        .map(|e| {
            let row = &router[e * hidden_size..(e + 1) * hidden_size];
            let dot: f32 = row.iter().zip(hidden.iter()).map(|(&w, &x)| w * x).sum();
            (e, dot)
        })
        .collect();

    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    scores.truncate(k);

    let max_logit = scores[0].1;
    let mut sum_exp = 0.0f32;
    for (_, s) in &mut scores {
        *s = (*s - max_logit).exp();
        sum_exp += *s;
    }
    if sum_exp > 0.0 {
        for (_, s) in &mut scores {
            *s /= sum_exp;
        }
    }
    Ok(scores)
}

fn attention_gqa_windowed(
    query: &[f32],
    keys_cache: &[f32],
    values_cache: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
) -> Vec<f32> {
    let kv_stride = num_kv_heads * head_dim;
    let total_seq = keys_cache.len() / kv_stride;
    let start_pos = match window_size {
        Some(w) if total_seq > w => total_seq - w,
        _ => 0,
    };
    let group_size = num_q_heads / num_kv_heads;
    // Gemma 4 utilise attention_scale = 1.0 (la normalisation QK gère déjà l'échelle).
    let scale = 1.0f32;
    let mut out = vec![0.0f32; num_q_heads * head_dim];

    for qh in 0..num_q_heads {
        let kvh = qh / group_size;
        let q_vec = &query[qh * head_dim..(qh + 1) * head_dim];
        let mut scores = Vec::with_capacity(total_seq - start_pos);
        let mut max_score = f32::NEG_INFINITY;
        for t in start_pos..total_seq {
            let k_vec = &keys_cache[t * kv_stride + kvh * head_dim..t * kv_stride + (kvh + 1) * head_dim];
            let dot: f32 = q_vec.iter().zip(k_vec.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
            max_score = max_score.max(dot);
            scores.push(dot);
        }
        let mut sum_exp = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
        let out_head = &mut out[qh * head_dim..(qh + 1) * head_dim];
        for (idx, &prob) in scores.iter().enumerate() {
            let t = start_pos + idx;
            let v_vec = &values_cache[t * kv_stride + kvh * head_dim..t * kv_stride + (kvh + 1) * head_dim];
            let w = prob * inv_sum;
            for d in 0..head_dim {
                out_head[d] += w * v_vec[d];
            }
        }
    }
    out
}

fn try_forward_resident_geglu_vram(
    normed: &[f32],
    expert: &Gemma4ExpertWeights,
    sc: &mut Gemma4GpuScratch<'_>,
) -> Result<Option<Vec<f32>>, String> {
    if !matches!(
        (&expert.gate, &expert.up, &expert.down),
        (Gemma4Linear::Q4Affine(_), Gemma4Linear::Q4Affine(_), Gemma4Linear::Q4Affine(_))
    ) {
        return Ok(None);
    }
    sc.x_dev.upload_f32_as_f16(normed).map_err(|e| e.to_string())?;
    let gate_rows = expert.gate.forward_q4_into_dev(sc.gpu, &sc.x_dev, &mut sc.gate_dev)?;
    let _ = expert.up.forward_q4_into_dev(sc.gpu, &sc.x_dev, &mut sc.up_dev)?;
    sc.gpu
        .geglu_inplace_f16(&mut sc.gate_dev, &sc.up_dev, gate_rows)
        .map_err(|e| e.to_string())?;
    let down_rows = expert.down.forward_q4_into_dev(sc.gpu, &sc.gate_dev, &mut sc.out_dev)?;
    let mut out = vec![0.0f32; down_rows];
    sc.out_dev.download_as_f32(&mut out).map_err(|e| e.to_string())?;
    Ok(Some(out))
}

fn forward_geglu_expert(
    normed: &[f32],
    expert: &Gemma4ExpertWeights,
    mut scratch: Option<&mut Gemma4GpuScratch<'_>>,
    backend: &dyn TernaryHadamardBackend,
    resident: bool,
) -> Result<Vec<f32>, String> {
    if resident {
        if let Some(sc) = scratch.as_deref_mut() {
            if let Some(out) = try_forward_resident_geglu_vram(normed, expert, sc)? {
                return Ok(out);
            }
        }
    }
    let gate = expert.gate.forward_with_scratch(normed, scratch.as_deref_mut(), backend, resident)?;
    let up = expert.up.forward_with_scratch(normed, scratch.as_deref_mut(), backend, resident)?;
    let hidden: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| gelu_tanh(g) * u)
        .collect();
    expert.down.forward_with_scratch(&hidden, scratch, backend, resident)
}

fn forward_vram_geglu_expert_into_acc(
    entry: &VramExpertEntry<'_>,
    sc: &mut Gemma4GpuScratch<'_>,
    weight: f32,
    reset: bool,
) -> Result<(), String> {
    entry.gate.forward_into_dev(sc.gpu, &sc.x_dev, &mut sc.gate_dev)?;
    entry.up.forward_into_dev(sc.gpu, &sc.x_dev, &mut sc.up_dev)?;
    sc.gpu
        .geglu_inplace_f16(&mut sc.gate_dev, &sc.up_dev, entry.gate.n_rows)
        .map_err(|e| e.to_string())?;
    entry.down.forward_into_dev(sc.gpu, &sc.gate_dev, &mut sc.out_dev)?;
    sc.gpu
        .weighted_acc_f16_to_f32(&mut sc.moe_acc_dev, &sc.out_dev, weight, entry.down.n_rows, reset)
        .map_err(|e| e.to_string())
}

fn slice_q4_affine_expert(
    stacked: &Q4AffineTensor,
    expert_idx: usize,
    rows_per_expert: usize,
) -> Result<Q4AffineTensor, String> {
    let row_bytes = stacked.row_stride_bytes();
    let groups_per_row = stacked.k_cols / aibox_kernels::Q4_GROUP_SIZE;
    let w_start = expert_idx * rows_per_expert * row_bytes;
    let w_end = w_start + rows_per_expert * row_bytes;
    let s_start = expert_idx * rows_per_expert * groups_per_row;
    let s_end = s_start + rows_per_expert * groups_per_row;
    Q4AffineTensor::from_raw(
        rows_per_expert,
        stacked.k_cols,
        stacked.packed_weights[w_start..w_end].to_vec(),
        stacked.scales_fp16[s_start..s_end].to_vec(),
        stacked.zeros_fp16[s_start..s_end].to_vec(),
    )
    .map_err(|e| e.to_string())
}

impl Gemma4Model {
    /// Charge un modèle Gemma 4 (`gemma-4-12B` dense ou `gemma-4-26B-A4B` MoE) en `q2_affine` ou `q4_affine`.
    pub fn load_from_dir(model_dir: &Path) -> Result<Self, String> {
        let cfg_raw = fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let config = Gemma4Config::from_json(&parse_json(&cfg_raw)?)?;
        let quant_json = fs::read_to_string(model_dir.join("quant_config.json"))
            .ok()
            .and_then(|s| parse_json(&s).ok());
        let is_q4 = quant_json
            .as_ref()
            .and_then(|v| {
                v.as_object()
                    .and_then(|o| o.get("format"))
                    .and_then(JsonValue::as_str)
                    .map(|f| f == "q4_affine")
            })
            .unwrap_or(false);
        let is_moe_q4 = quant_json
            .as_ref()
            .and_then(|v| {
                v.as_object()
                    .and_then(|o| o.get("moe_format").or_else(|| o.get("format")))
                    .and_then(JsonValue::as_str)
                    .map(|f| f == "q4_affine")
            })
            .unwrap_or(is_q4);
        let mut st = SafeTensorsFile::open_dir(model_dir)?;

        let load_q2 = |st: &mut SafeTensorsFile, prefix: &str| -> Result<Gemma4Linear, String> {
            let (w, s, z) = (
                format!("{prefix}.weight"),
                format!("{prefix}.scales"),
                format!("{prefix}.zeros"),
            );
            if is_q4 {
                Ok(Gemma4Linear::Q4Affine(st.load_q4_affine_layer(&w, &s, &z)?))
            } else {
                Ok(Gemma4Linear::Q2Affine(st.load_q2_affine_layer(&w, &s, &z)?))
            }
        };

        let mut layers = Vec::with_capacity(config.base.num_hidden_layers);
        for idx in 0..config.base.num_hidden_layers {
            let p = format!("blk.{idx}");
            let is_global_attn = config.global_attn_every > 0
                && ((idx + 1) % config.global_attn_every == 0);
            let head_dim = if !is_global_attn && config.head_dim_swa > 0 {
                config.head_dim_swa
            } else {
                config.base.head_dim()
            };
            let rope_dim = if !is_global_attn && config.rope_dim_swa > 0 {
                config.rope_dim_swa.min(head_dim)
            } else {
                config.base.rope_dim().min(head_dim)
            };
            let rope_theta = if is_global_attn {
                config.base.rope_theta
            } else {
                config.rope_theta_swa
            };

            let attn_norm = st.load_f32_vector(&format!("{p}.attn_norm.weight"))?;
            let ffn_norm = st.load_f32_vector(&format!("{p}.ffn_norm.weight"))?;
            let attn_q = load_q2(&mut st, &format!("{p}.attn_q"))?;
            let attn_k = load_q2(&mut st, &format!("{p}.attn_k"))?;
            let k_rows = match &attn_k {
                Gemma4Linear::Q2Affine(t) => t.n_rows,
                Gemma4Linear::Q4Affine(t) => t.n_rows,
                Gemma4Linear::Dense { rows, .. } => *rows,
            };
            let num_kv_heads = (k_rows / head_dim).max(1);
            let attn_v = if st.has_tensor(&format!("{p}.attn_v.weight")) {
                Some(load_q2(&mut st, &format!("{p}.attn_v"))?)
            } else {
                None
            };
            let attn_o = load_q2(&mut st, &format!("{p}.attn_o"))?;
            let attn_q_norm = if st.has_tensor(&format!("{p}.attn_q_norm.weight")) {
                Some(st.load_f32_vector(&format!("{p}.attn_q_norm.weight"))?)
            } else {
                None
            };
            let attn_k_norm = if st.has_tensor(&format!("{p}.attn_k_norm.weight")) {
                Some(st.load_f32_vector(&format!("{p}.attn_k_norm.weight"))?)
            } else {
                None
            };

            let ffn = if config.num_experts > 0 && st.has_tensor(&format!("{p}.ffn_gate_inp.weight")) {
                let router = st.load_f32_vector(&format!("{p}.ffn_gate_inp.weight"))?;
                let router_scale = if st.has_tensor(&format!("{p}.ffn_gate_inp.scale")) {
                    Some(st.load_f32_vector(&format!("{p}.ffn_gate_inp.scale"))?)
                } else {
                    None
                };
                let expert_scales = if st.has_tensor(&format!("{p}.ffn_down_exps.scale")) {
                    Some(st.load_f32_vector(&format!("{p}.ffn_down_exps.scale"))?)
                } else {
                    None
                };
                let experts = if st.has_tensor(&format!("{p}.ffn_gate_exps.weight")) {
                    let gate_meta = st
                        .tensor_meta(&format!("{p}.ffn_gate_exps.weight"))
                        .ok_or_else(|| format!("Missing {p}.ffn_gate_exps.weight"))?;
                    let inter_rows = gate_meta.shape[0] / config.num_experts;
                    Gemma4Experts::LazyDisk(LazyMoeExperts {
                        layer_idx: idx,
                        inter_rows,
                        down_rows: config.base.hidden_size,
                        hidden_size: config.base.hidden_size,
                        is_q4: is_moe_q4,
                    })
                } else {
                    let mut list = Vec::with_capacity(config.num_experts);
                    for e in 0..config.num_experts {
                        list.push(Gemma4ExpertWeights {
                            gate: load_q2(&mut st, &format!("{p}.ffn_gate.{e}"))?,
                            up: load_q2(&mut st, &format!("{p}.ffn_up.{e}"))?,
                            down: load_q2(&mut st, &format!("{p}.ffn_down.{e}"))?,
                        });
                    }
                    Gemma4Experts::InMemory(list)
                };
                let shared_expert = if st.has_tensor(&format!("{p}.ffn_gate_shexp.weight")) {
                    Some(Gemma4ExpertWeights {
                        gate: load_q2(&mut st, &format!("{p}.ffn_gate_shexp"))?,
                        up: load_q2(&mut st, &format!("{p}.ffn_up_shexp"))?,
                        down: load_q2(&mut st, &format!("{p}.ffn_down_shexp"))?,
                    })
                } else if st.has_tensor(&format!("{p}.ffn_gate.weight")) {
                    Some(Gemma4ExpertWeights {
                        gate: load_q2(&mut st, &format!("{p}.ffn_gate"))?,
                        up: load_q2(&mut st, &format!("{p}.ffn_up"))?,
                        down: load_q2(&mut st, &format!("{p}.ffn_down"))?,
                    })
                } else {
                    None
                };
                Gemma4Ffn::Moe(Gemma4MoeWeights {
                    router,
                    router_scale,
                    expert_scales,
                    experts,
                    shared_expert,
                })
            } else {
                Gemma4Ffn::Dense(Gemma4ExpertWeights {
                    gate: load_q2(&mut st, &format!("{p}.ffn_gate"))?,
                    up: load_q2(&mut st, &format!("{p}.ffn_up"))?,
                    down: load_q2(&mut st, &format!("{p}.ffn_down"))?,
                })
            };

            let post_attention_norm = if st.has_tensor(&format!("{p}.post_attention_norm.weight")) {
                Some(st.load_f32_vector(&format!("{p}.post_attention_norm.weight"))?)
            } else {
                None
            };
            let post_ffw_norm = if st.has_tensor(&format!("{p}.post_ffw_norm.weight")) {
                Some(st.load_f32_vector(&format!("{p}.post_ffw_norm.weight"))?)
            } else {
                None
            };
            let pre_ffw_norm_2 = if st.has_tensor(&format!("{p}.pre_ffw_norm_2.weight")) {
                Some(st.load_f32_vector(&format!("{p}.pre_ffw_norm_2.weight"))?)
            } else {
                None
            };
            let post_ffw_norm_1 = if st.has_tensor(&format!("{p}.post_ffw_norm_1.weight")) {
                Some(st.load_f32_vector(&format!("{p}.post_ffw_norm_1.weight"))?)
            } else {
                None
            };
            let post_ffw_norm_2 = if st.has_tensor(&format!("{p}.post_ffw_norm_2.weight")) {
                Some(st.load_f32_vector(&format!("{p}.post_ffw_norm_2.weight"))?)
            } else {
                None
            };
            let layer_output_scale = if st.has_tensor(&format!("{p}.layer_output_scale.weight")) {
                st.load_f32_vector(&format!("{p}.layer_output_scale.weight"))?.first().copied()
            } else {
                None
            };

            layers.push(Gemma4Layer {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_o,
                attn_q_norm,
                attn_k_norm,
                is_global_attn,
                head_dim,
                num_kv_heads,
                rope_dim,
                rope_theta,
                ffn_norm,
                post_attention_norm,
                post_ffw_norm,
                pre_ffw_norm_2,
                post_ffw_norm_1,
                post_ffw_norm_2,
                layer_output_scale,
                ffn,
                ple_weights: None,
            });
        }

        let token_embeddings = if st.has_tensor("token_embd.scales") {
            EmbeddingTable::Quantized(st.load_ternary_layer("token_embd.weight", "token_embd.scales")?)
        } else {
            EmbeddingTable::F16(st.load_f16_raw("token_embd.weight")?)
        };
        let final_norm = st.load_f32_vector("output_norm.weight")?;
        let lm_head = if st.has_tensor("output.weight") {
            LmHead::Dense(EmbeddingTable::F16(st.load_f16_raw("output.weight")?))
        } else {
            LmHead::TiedEmbedding
        };

        let rope_freqs = if st.has_tensor("rope_freqs.weight") {
            Some(st.load_f32_vector("rope_freqs.weight")?)
        } else {
            None
        };

        Ok(Self {
            config,
            token_embeddings,
            layers,
            final_norm,
            lm_head,
            rope_freqs,
            safetensors: Some(Mutex::new(st)),
            expert_ram_cache: Mutex::new(LruMap::new(if is_moe_q4 { 1024 } else { 2048 })),
        })
    }

    pub fn generate_tokens(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        params: SamplingParams,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<GenerationStats, String> {
        if prompt_tokens.is_empty() {
            return Err("prompt_tokens must not be empty".to_string());
        }
        let mut kv_cache = KvCache::new(self.layers.len());
        let is_moe_q4 = self.layers.iter().any(|l| match &l.ffn {
            Gemma4Ffn::Moe(m) => matches!(&m.experts, Gemma4Experts::LazyDisk(lazy) if lazy.is_q4),
            Gemma4Ffn::Dense(_) => false,
        });
        let total_experts = self.layers.len() * self.config.num_experts.max(1);
        let mut scratch = match backend.as_gpu() {
            Some(gpu) => {
                let free_bytes = gpu.vram_free_bytes();
                let reserved_bytes = 1400usize * 1024 * 1024;
                let weights_per_exp = 3 * self.config.moe_intermediate_size * self.config.base.hidden_size;
                let bytes_per_exp = if is_moe_q4 {
                    (weights_per_exp * 9) / 16
                } else {
                    (weights_per_exp * 5) / 16
                }
                .max(1);
                let vram_cap = if free_bytes > reserved_bytes {
                    ((free_bytes - reserved_bytes) / bytes_per_exp).clamp(256, total_experts)
                } else if is_moe_q4 {
                    1050
                } else {
                    1850
                };
                if std::env::var("AIBOX_MOE_STATS").is_ok() {
                    eprintln!(
                        "[VRAM Budget] free={:.0} MiB reserved=1400 MiB expert={:.2} MiB -> vram_cap={vram_cap}/{total_experts}",
                        free_bytes as f64 / (1024.0 * 1024.0),
                        bytes_per_exp as f64 / (1024.0 * 1024.0)
                    );
                }
                Some(Gemma4GpuScratch::new(gpu, &self.config, vram_cap)?)
            }
            None => None,
        };

        let prefill_start = Instant::now();
        let mut logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            logits = self.forward_token_with_scratch(
                tok,
                pos,
                &mut kv_cache,
                scratch.as_mut(),
                backend,
            )?;
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = Instant::now();
        let mut rng = XorShift64::new(params.seed);
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut cur_pos = prompt_tokens.len();

        for step in 0..max_new_tokens {
            let next_tok = sample_token(&logits, params, &mut rng)?;
            generated.push(next_tok);
            if Some(next_tok) == eos_token_id || step + 1 == max_new_tokens {
                break;
            }
            logits = self.forward_token_with_scratch(
                next_tok,
                cur_pos,
                &mut kv_cache,
                scratch.as_mut(),
                backend,
            )?;
            cur_pos += 1;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(sc) = scratch.as_ref() {
            if std::env::var("AIBOX_MOE_STATS").is_ok() {
                let total = sc.vram_hits + sc.ram_hits + sc.disk_misses;
                let vram_pct = if total > 0 { 100.0 * sc.vram_hits as f64 / total as f64 } else { 0.0 };
                let ram_pct = if total > 0 { 100.0 * sc.ram_hits as f64 / total as f64 } else { 0.0 };
                eprintln!(
                    "[MoE Cache] total={total} vram_hits={} ({vram_pct:.1}%) ram_hits={} ({ram_pct:.1}%) disk_misses={}",
                    sc.vram_hits, sc.ram_hits, sc.disk_misses
                );
            }
        }
        Ok(GenerationStats {
            tokens: generated,
            prefill_ms,
            decode_ms,
        })
    }

    pub fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        self.forward_token_with_scratch(token_id, pos, kv_cache, None, backend)
    }

    fn load_or_cache_ram_expert(
        &self,
        lazy: &LazyMoeExperts,
        expert_idx: usize,
    ) -> Result<(Arc<Gemma4ExpertWeights>, bool), String> {
        let key = (lazy.layer_idx, expert_idx);
        {
            let mut guard = self
                .expert_ram_cache
                .lock()
                .map_err(|_| "expert_ram_cache poisoned".to_string())?;
            if let Some(exp) = guard.get(&key) {
                return Ok((Arc::clone(exp), true));
            }
        }
        let st_mutex = self
            .safetensors
            .as_ref()
            .ok_or_else(|| "Missing safetensors for LazyDisk expert".to_string())?;
        let mut st = st_mutex
            .lock()
            .map_err(|_| "safetensors mutex poisoned".to_string())?;
        let p = format!("blk.{}", lazy.layer_idx);
        let exp = Arc::new(Gemma4ExpertWeights {
            gate: load_affine_slice_from_st(
                &mut st,
                &format!("{p}.ffn_gate_exps"),
                expert_idx,
                lazy.inter_rows,
                lazy.hidden_size,
                lazy.is_q4,
            )?,
            up: load_affine_slice_from_st(
                &mut st,
                &format!("{p}.ffn_up_exps"),
                expert_idx,
                lazy.inter_rows,
                lazy.hidden_size,
                lazy.is_q4,
            )?,
            down: load_affine_slice_from_st(
                &mut st,
                &format!("{p}.ffn_down_exps"),
                expert_idx,
                lazy.down_rows,
                lazy.inter_rows,
                lazy.is_q4,
            )?,
        });
        let mut guard = self
            .expert_ram_cache
            .lock()
            .map_err(|_| "expert_ram_cache poisoned".to_string())?;
        guard.insert(key, Arc::clone(&exp));
        Ok((exp, false))
    }

    fn forward_moe_lazy_vram(
        &self,
        lazy: &LazyMoeExperts,
        selected: &[(usize, f32)],
        expert_scales: Option<&[f32]>,
        moe_in: &[f32],
        sc: &mut Gemma4GpuScratch<'_>,
    ) -> Result<Vec<f32>, String> {
        sc.x_dev.upload_f32_as_f16(moe_in).map_err(|e| e.to_string())?;
        for (idx, &(expert_idx, weight)) in selected.iter().enumerate() {
            let key = (lazy.layer_idx, expert_idx);
            let exp_scale = expert_scales.and_then(|es| es.get(expert_idx).copied()).unwrap_or(1.0);
            let w = weight * exp_scale;
            if let Some(entry) = sc.vram_experts.get(&key) {
                let ptr: *const VramExpertEntry<'_> = entry;
                sc.vram_hits += 1;
                forward_vram_geglu_expert_into_acc(unsafe { &*ptr }, sc, w, idx == 0)?;
                continue;
            }
            let (expert, from_ram) = self.load_or_cache_ram_expert(lazy, expert_idx)?;
            if from_ram {
                sc.ram_hits += 1;
            } else {
                sc.disk_misses += 1;
            }
            let entry = insert_or_reuse_vram_expert(&mut sc.vram_experts, sc.gpu, key, &expert)?;
            forward_vram_geglu_expert_into_acc(entry, sc, w, idx == 0)?;
        }
        let mut moe_out = vec![0.0f32; lazy.down_rows];
        sc.moe_acc_dev.download(&mut moe_out).map_err(|e| e.to_string())?;
        Ok(moe_out)
    }

    fn forward_token_with_scratch(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        mut scratch: Option<&mut Gemma4GpuScratch<'_>>,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config.base;
        if token_id >= cfg.vocab_size {
            return Err(format!("token_id {token_id} >= vocab_size {}", cfg.vocab_size));
        }
        let hidden_size = cfg.hidden_size;
        let mut hidden = self.token_embeddings.read_row(token_id, hidden_size);
        let emb_scale = (hidden_size as f32).sqrt();
        for val in &mut hidden {
            *val *= emb_scale;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed_attn = rms_norm(&hidden, &layer.attn_norm, cfg.rms_norm_eps);
            let mut q = layer.attn_q.forward_with_scratch(&normed_attn, scratch.as_deref_mut(), backend, true)?;
            let mut k = layer.attn_k.forward_with_scratch(&normed_attn, scratch.as_deref_mut(), backend, true)?;
            let mut v = match &layer.attn_v {
                Some(v_proj) => v_proj.forward_with_scratch(&normed_attn, scratch.as_deref_mut(), backend, true)?,
                None => k.clone(),
            };

            if let Some(ref q_norm) = layer.attn_q_norm {
                rms_norm_heads_inplace(&mut q, cfg.num_attention_heads, layer.head_dim, q_norm, cfg.rms_norm_eps);
            }
            if let Some(ref k_norm) = layer.attn_k_norm {
                rms_norm_heads_inplace(&mut k, layer.num_kv_heads, layer.head_dim, k_norm, cfg.rms_norm_eps);
            }
            // Gemma 4 applique une RMSNorm unitaire (sans poids) sur les têtes V.
            for h in 0..layer.num_kv_heads {
                let slice = &mut v[h * layer.head_dim..(h + 1) * layer.head_dim];
                let mean_sq: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / (layer.head_dim as f32);
                let inv = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
                for val in slice {
                    *val *= inv;
                }
            }

            let global_freqs = if layer.is_global_attn {
                self.rope_freqs.as_deref()
            } else {
                None
            };
            apply_rope_with_freqs_inplace(
                &mut q,
                cfg.num_attention_heads,
                layer.head_dim,
                layer.rope_dim,
                pos,
                layer.rope_theta,
                global_freqs,
            );
            apply_rope_with_freqs_inplace(
                &mut k,
                layer.num_kv_heads,
                layer.head_dim,
                layer.rope_dim,
                pos,
                layer.rope_theta,
                global_freqs,
            );

            kv_cache.keys[layer_idx].extend_from_slice(&k);
            kv_cache.values[layer_idx].extend_from_slice(&v);

            let window = if layer.is_global_attn {
                None
            } else {
                Some(self.config.sliding_window)
            };
            let ctx = attention_gqa_windowed(
                &q,
                &kv_cache.keys[layer_idx],
                &kv_cache.values[layer_idx],
                cfg.num_attention_heads,
                layer.num_kv_heads,
                layer.head_dim,
                window,
            );
            let attn_out = layer.attn_o.forward_with_scratch(&ctx, scratch.as_deref_mut(), backend, true)?;
            let attn_out = if let Some(ref norm_w) = layer.post_attention_norm {
                rms_norm(&attn_out, norm_w, cfg.rms_norm_eps)
            } else {
                attn_out
            };
            add_residual(&mut hidden, &attn_out);

            let normed_ffn = rms_norm(&hidden, &layer.ffn_norm, cfg.rms_norm_eps);
            let mut ffn_out = match &layer.ffn {
                Gemma4Ffn::Dense(dense_expert) => {
                    forward_geglu_expert(&normed_ffn, dense_expert, scratch.as_deref_mut(), backend, true)?
                }
                Gemma4Ffn::Moe(moe) => {
                    let mean_sq: f32 =
                        hidden.iter().map(|&x| x * x).sum::<f32>() / (hidden.len() as f32);
                    let inv_rms = 1.0 / (mean_sq + cfg.rms_norm_eps).sqrt();
                    let scalar_root_size = 1.0 / (hidden_size as f32).sqrt();
                    let router_in: Vec<f32> = if let Some(ref rs) = moe.router_scale {
                        hidden
                            .iter()
                            .zip(rs.iter())
                            .map(|(&x, &s)| (x * inv_rms) * s * scalar_root_size)
                            .collect()
                    } else {
                        hidden.iter().map(|&x| x * inv_rms * scalar_root_size).collect()
                    };
                    let top_k = {
                        static OVERRIDE_K: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
                        let ov = *OVERRIDE_K.get_or_init(|| {
                            std::env::var("AIBOX_MOE_TOP_K").ok().and_then(|s| s.parse::<usize>().ok())
                        });
                        ov.unwrap_or(self.config.num_experts_per_tok)
                            .clamp(1, self.config.num_experts_per_tok)
                    };
                    let selected = route_top_k(
                        &router_in,
                        &moe.router,
                        self.config.num_experts,
                        top_k,
                    )?;
                    let moe_in = if let Some(ref norm_w) = layer.pre_ffw_norm_2 {
                        rms_norm(&hidden, norm_w, cfg.rms_norm_eps)
                    } else {
                        normed_ffn.clone()
                    };
                    let moe_out = match (&moe.experts, scratch.as_deref_mut()) {
                        (Gemma4Experts::LazyDisk(lazy), Some(sc)) => self.forward_moe_lazy_vram(
                            lazy,
                            &selected,
                            moe.expert_scales.as_deref(),
                            &moe_in,
                            sc,
                        )?,
                        _ => {
                            let mut acc_out = vec![0.0f32; hidden_size];
                            for (expert_idx, weight) in selected {
                                let exp_vec = match &moe.experts {
                                    Gemma4Experts::InMemory(list) => forward_geglu_expert(
                                        &moe_in,
                                        &list[expert_idx],
                                        scratch.as_deref_mut(),
                                        backend,
                                        false,
                                    )?,
                                    Gemma4Experts::LazyDisk(lazy) => {
                                        let (expert, _) = self.load_or_cache_ram_expert(lazy, expert_idx)?;
                                        forward_geglu_expert(
                                            &moe_in,
                                            &expert,
                                            scratch.as_deref_mut(),
                                            backend,
                                            false,
                                        )?
                                    }
                                };
                                let exp_scale = moe
                                    .expert_scales
                                    .as_ref()
                                    .and_then(|es| es.get(expert_idx).copied())
                                    .unwrap_or(1.0);
                                let w = weight * exp_scale;
                                for (acc, &val) in acc_out.iter_mut().zip(exp_vec.iter()) {
                                    *acc += w * val;
                                }
                            }
                            acc_out
                        }
                    };
                    let mut out = if let Some(ref norm_w) = layer.post_ffw_norm_2 {
                        rms_norm(&moe_out, norm_w, cfg.rms_norm_eps)
                    } else {
                        moe_out
                    };
                    if let Some(ref shared) = moe.shared_expert {
                        let shared_vec =
                            forward_geglu_expert(&normed_ffn, shared, scratch.as_deref_mut(), backend, true)?;
                        let shared_vec = if let Some(ref norm_w) = layer.post_ffw_norm_1 {
                            rms_norm(&shared_vec, norm_w, cfg.rms_norm_eps)
                        } else {
                            shared_vec
                        };
                        add_residual(&mut out, &shared_vec);
                    }
                    out
                }
            };
            if let Some(ref ple) = layer.ple_weights {
                for (acc, &p) in ffn_out.iter_mut().zip(ple.iter()) {
                    *acc += p;
                }
            }
            let ffn_out = if let Some(ref norm_w) = layer.post_ffw_norm {
                rms_norm(&ffn_out, norm_w, cfg.rms_norm_eps)
            } else {
                ffn_out
            };
            add_residual(&mut hidden, &ffn_out);
            if let Some(scale) = layer.layer_output_scale {
                for val in &mut hidden {
                    *val *= scale;
                }
            }
        }

        let final_hidden = rms_norm(&hidden, &self.final_norm, cfg.rms_norm_eps);
        let table = match &self.lm_head {
            LmHead::Dense(t) => t,
            LmHead::TiedEmbedding => &self.token_embeddings,
            LmHead::Ternary(_) => return Err("Gemma4Model expects Dense or TiedEmbedding LmHead".to_string()),
        };
        if let Some(sc) = scratch {
            sc.x_dev.upload_f32_as_f16(&final_hidden).map_err(|e| e.to_string())?;
            match table {
                EmbeddingTable::Quantized(tensor) => {
                    sc.gpu
                        .lm_head_quant_act_f16(
                            &sc.x_dev,
                            &tensor.packed_weights,
                            &tensor.scales_fp16,
                            &mut sc.logits_dev,
                            cfg.vocab_size,
                            hidden_size,
                            tensor.bits,
                        )
                        .map_err(|e| e.to_string())?;
                    let mut logits = vec![0.0f32; cfg.vocab_size];
                    sc.logits_dev.download(&mut logits).map_err(|e| e.to_string())?;
                    return Ok(logits);
                }
                EmbeddingTable::F16(data) => {
                    sc.gpu
                        .lm_head_f16_act_f16(&sc.x_dev, data, &mut sc.logits_dev, cfg.vocab_size, hidden_size)
                        .map_err(|e| e.to_string())?;
                    let mut logits = vec![0.0f32; cfg.vocab_size];
                    sc.logits_dev.download(&mut logits).map_err(|e| e.to_string())?;
                    return Ok(logits);
                }
                EmbeddingTable::F32(_) => {}
            }
        }
        Ok(table.dot_all_rows(&final_hidden, cfg.vocab_size, hidden_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aibox_kernels::scalar::ScalarBackend;
    use aibox_kernels::tensor::quantize_q2_affine_f32;

    fn make_q2_linear(rows: usize, cols: usize, seed: f32) -> Gemma4Linear {
        let weights: Vec<f32> = (0..(rows * cols))
            .map(|i| (((i % 11) as f32) - 5.0) * 0.02 * seed)
            .collect();
        Gemma4Linear::Q2Affine(quantize_q2_affine_f32(&weights, rows, cols).unwrap())
    }

    #[test]
    fn gemma4_router_top_k_selects_highest_and_normalizes() {
        let hidden = vec![1.0, 0.5, -0.5, 0.25];
        let router = vec![
            0.1, 0.0, 0.0, 0.0, // expert 0: 0.1
            1.0, 1.0, 0.0, 0.0, // expert 1: 1.5
            -1.0, 0.0, 0.0, 0.0, // expert 2: -1.0
            0.5, 0.5, 0.0, 0.0, // expert 3: 0.75
        ];
        let selected = route_top_k(&hidden, &router, 4, 2).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].0, 1);
        assert_eq!(selected[1].0, 3);
        let sum_w: f32 = selected.iter().map(|&(_, w)| w).sum();
        assert!((sum_w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gemma4_geglu_and_sliding_window_forward_work_end_to_end() {
        let hidden_size = 128;
        let intermediate = 128;
        let vocab_size = 16;
        let base = ModelConfig {
            hidden_size,
            intermediate_size: intermediate,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            rope_dim: 32,
            vocab_size,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            ..ModelConfig::default()
        };
        let cfg = Gemma4Config {
            base,
            sliding_window: 2,
            global_attn_every: 2,
            head_dim_swa: 32,
            rope_dim_swa: 32,
            rope_theta_swa: 10_000.0,
            num_experts: 4,
            num_experts_per_tok: 2,
            moe_intermediate_size: intermediate,
            has_shared_expert: true,
            ple_dim: hidden_size,
        };
        let make_expert = |s: f32| Gemma4ExpertWeights {
            gate: make_q2_linear(intermediate, hidden_size, s),
            up: make_q2_linear(intermediate, hidden_size, s + 0.5),
            down: make_q2_linear(hidden_size, intermediate, s + 1.0),
        };
        let layers = (0..2)
            .map(|idx| Gemma4Layer {
                attn_norm: vec![1.0; hidden_size],
                attn_q: make_q2_linear(hidden_size, hidden_size, 1.0),
                attn_k: make_q2_linear(64, hidden_size, 1.1),
                attn_v: Some(make_q2_linear(64, hidden_size, 1.2)),
                attn_o: make_q2_linear(hidden_size, hidden_size, 0.9),
                attn_q_norm: Some(vec![1.0; 32]),
                attn_k_norm: Some(vec![1.0; 32]),
                is_global_attn: idx == 1,
                head_dim: 32,
                num_kv_heads: 2,
                rope_dim: 32,
                rope_theta: 10_000.0,
                ffn_norm: vec![1.0; hidden_size],
                post_attention_norm: None,
                post_ffw_norm: None,
                pre_ffw_norm_2: None,
                post_ffw_norm_1: None,
                post_ffw_norm_2: None,
                layer_output_scale: None,
                ffn: if idx == 0 {
                    Gemma4Ffn::Moe(Gemma4MoeWeights {
                        router: (0..(4 * hidden_size)).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect(),
                        router_scale: None,
                        expert_scales: None,
                        experts: Gemma4Experts::InMemory(
                            (0..4).map(|e| make_expert(1.0 + e as f32 * 0.1)).collect(),
                        ),
                        shared_expert: Some(make_expert(0.8)),
                    })
                } else {
                    Gemma4Ffn::Dense(make_expert(0.9))
                },
                ple_weights: Some(vec![0.01; hidden_size]),
            })
            .collect();

        let emb = EmbeddingTable::F32(vec![0.1; vocab_size * hidden_size]);
        let lm_head = LmHead::Dense(EmbeddingTable::F32(vec![0.05; vocab_size * hidden_size]));
        let model = Gemma4Model {
            config: cfg,
            token_embeddings: emb,
            layers,
            final_norm: vec![1.0; hidden_size],
            lm_head,
            rope_freqs: None,
            safetensors: None,
            expert_ram_cache: Mutex::new(LruMap::new(16)),
        };
        let mut kv = KvCache::new(2);
        let backend = ScalarBackend;

        for pos in 0..3 {
            let logits = model.forward_token(pos + 1, pos, &mut kv, &backend).unwrap();
            assert_eq!(logits.len(), vocab_size);
            assert!(logits.iter().all(|v| v.is_finite()));
        }
        let mut lru = LruMap::new(2);
        lru.insert((0usize, 0usize), 10);
        lru.insert((0, 1), 20);
        assert_eq!(lru.get(&(0, 0)), Some(&10));
        lru.insert((0, 2), 30);
        assert_eq!(lru.get(&(0, 1)), None);
    }

    #[test]
    fn gemma4_config_detects_architecture_and_dense_linear_works() {
        let json = crate::json::parse_json(
            r#"{"architecture":"gemma4","hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":32,"sliding_window":1024,"num_experts":128,"num_experts_per_tok":8}"#,
        )
        .unwrap();
        assert!(Gemma4Config::is_gemma4_json(&json));
        let cfg = Gemma4Config::from_json(&json).unwrap();
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.global_attn_every, 6);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 256);
        assert!(cfg.has_shared_expert);
        assert_eq!(cfg.ple_dim, 0);

        let dense = Gemma4Linear::Dense {
            rows: 2,
            cols: 3,
            weights: vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0],
        };
        let out = dense.forward_cpu(&[1.0, 1.0, 1.0]).unwrap();
        assert_eq!(out, vec![6.0, 0.0]);
    }
}
