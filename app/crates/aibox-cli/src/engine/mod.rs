//! SERV-03 : Boucle d'inférence autorégressive Transformer (GQA + RoPE + SwiGLU + KV Cache)
//! utilisant `TernaryHadamardBackend`.

mod attention;
mod gdn;
pub mod gemma4;

use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use aibox_kernels::tensor::{f16_to_f32, f32_to_f16, pack_ternary_2bit_rows, TernaryTensor};
use aibox_kernels::{GpuBackend, GpuBuffer, TernaryHadamardBackend, GROUP_SIZE};

use crate::json::{parse_json, JsonValue};
use crate::safetensors::SafeTensorsFile;
use crate::sampling::{sample_token, SamplingParams, XorShift64};

#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub ssm_conv_kernel: usize,
    pub ssm_group_count: usize,
    pub ssm_inner_size: usize,
    pub ssm_state_size: usize,
    pub ssm_time_step_rank: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub use_hadamard: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        if self.head_dim > 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    pub fn rope_dim(&self) -> usize {
        let hd = self.head_dim();
        if self.rope_dim > 0 && self.rope_dim <= hd {
            self.rope_dim
        } else {
            hd
        }
    }

    pub fn from_json(val: &JsonValue) -> Result<Self, String> {
        let obj = val
            .as_object()
            .ok_or_else(|| "config.json must be a JSON object".to_string())?;
        let get_usize = |key: &str| -> Result<usize, String> {
            obj.get(key)
                .and_then(JsonValue::as_u64)
                .map(|num| num as usize)
                .ok_or_else(|| format!("Missing or invalid '{key}' in config.json"))
        };
        let get_opt_usize = |key: &str, default: usize| -> usize {
            obj.get(key)
                .and_then(JsonValue::as_u64)
                .map(|num| num as usize)
                .unwrap_or(default)
        };
        let hidden_size = get_usize("hidden_size")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = get_opt_usize("num_key_value_heads", num_attention_heads);
        let head_dim = get_opt_usize("head_dim", hidden_size / num_attention_heads);

        Ok(Self {
            hidden_size,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rope_dim: get_opt_usize("rope_dim", 0),
            ssm_conv_kernel: get_opt_usize("ssm_conv_kernel", 0),
            ssm_group_count: get_opt_usize("ssm_group_count", 0),
            ssm_inner_size: get_opt_usize("ssm_inner_size", 0),
            ssm_state_size: get_opt_usize("ssm_state_size", 0),
            ssm_time_step_rank: get_opt_usize("ssm_time_step_rank", 0),
            vocab_size: get_usize("vocab_size")?,
            rms_norm_eps: match obj.get("rms_norm_eps") {
                Some(JsonValue::Number(num)) => *num as f32,
                _ => 1e-6,
            },
            rope_theta: match obj.get("rope_theta") {
                Some(JsonValue::Number(num)) => *num as f32,
                _ => 10_000.0,
            },
            use_hadamard: true,
        })
    }
}

fn f16_lut() -> &'static [f32] {
    static LUT: OnceLock<Vec<f32>> = OnceLock::new();
    LUT.get_or_init(|| (0..=u16::MAX).map(f16_to_f32).collect())
}

pub enum EmbeddingTable {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Quantized(TernaryTensor),
}

impl EmbeddingTable {
    fn read_row(&self, token_id: usize, hidden_size: usize) -> Vec<f32> {
        let start = token_id * hidden_size;
        match self {
            Self::F32(data) => data[start..start + hidden_size].to_vec(),
            Self::F16(data) => {
                let lut = f16_lut();
                data[start..start + hidden_size]
                    .iter()
                    .map(|&raw| lut[raw as usize])
                    .collect()
            }
            Self::Quantized(tensor) => {
                let lut = f16_lut();
                let groups_per_row = hidden_size / GROUP_SIZE;
                let row_bytes = tensor.row_stride_bytes();
                let w_row = &tensor.packed_weights[token_id * row_bytes..(token_id + 1) * row_bytes];
                let s_row = &tensor.scales_fp16[token_id * groups_per_row..(token_id + 1) * groups_per_row];
                let mut out = Vec::with_capacity(hidden_size);
                for (g, &s_bits) in s_row.iter().enumerate() {
                    let scale = lut[s_bits as usize];
                    if tensor.bits == 4 {
                        let w_g = &w_row[g * 64..(g + 1) * 64];
                        for &byte in w_g {
                            out.push(((byte & 0x0F) as i8 - 8) as f32 * scale);
                            out.push((((byte >> 4) & 0x0F) as i8 - 8) as f32 * scale);
                        }
                    } else {
                        let w_g = &w_row[g * GROUP_SIZE..(g + 1) * GROUP_SIZE];
                        for &byte in w_g {
                            out.push((byte as i8) as f32 * scale);
                        }
                    }
                }
                out
            }
        }
    }

    fn dot_all_rows(&self, hidden: &[f32], vocab_size: usize, hidden_size: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; vocab_size];
        let num_threads = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(1);
        let chunk_rows = if vocab_size >= 1024 && num_threads > 1 {
            vocab_size.div_ceil(num_threads)
        } else {
            vocab_size
        };

        match self {
            Self::F32(data) => {
                std::thread::scope(|scope| {
                    for (chunk_idx, out_chunk) in logits.chunks_mut(chunk_rows).enumerate() {
                        let start_row = chunk_idx * chunk_rows;
                        let data_slice = &data[start_row * hidden_size..];
                        scope.spawn(move || {
                            for (local_row, slot) in out_chunk.iter_mut().enumerate() {
                                let row_slice = &data_slice[local_row * hidden_size..(local_row + 1) * hidden_size];
                                *slot = row_slice.iter().zip(hidden.iter()).map(|(&w, &x)| w * x).sum();
                            }
                        });
                    }
                });
            }
            Self::F16(data) => {
                let lut = f16_lut();
                std::thread::scope(|scope| {
                    for (chunk_idx, out_chunk) in logits.chunks_mut(chunk_rows).enumerate() {
                        let start_row = chunk_idx * chunk_rows;
                        let data_slice = &data[start_row * hidden_size..];
                        scope.spawn(move || {
                            for (local_row, slot) in out_chunk.iter_mut().enumerate() {
                                let row_slice = &data_slice[local_row * hidden_size..(local_row + 1) * hidden_size];
                                *slot = row_slice
                                    .iter()
                                    .zip(hidden.iter())
                                    .map(|(&w_u16, &x)| lut[w_u16 as usize] * x)
                                    .sum();
                            }
                        });
                    }
                });
            }
            Self::Quantized(tensor) => {
                return dot_quantized_lm_head_cpu(hidden, tensor);
            }
        }
        logits
    }
}

pub enum LmHead {
    Ternary(TernaryTensor),
    Dense(EmbeddingTable),
    TiedEmbedding,
}

fn load_dense_table(st: &mut SafeTensorsFile, name: &str) -> Result<EmbeddingTable, String> {
    if st.tensors().get(name).map(|m| m.dtype.as_str()) == Some("F16") {
        Ok(EmbeddingTable::F16(st.load_f16_raw(name)?))
    } else {
        Ok(EmbeddingTable::F32(st.load_f32_vector(name)?))
    }
}

pub struct AttentionBlockWeights {
    pub attn_q: TernaryTensor,
    pub attn_k: TernaryTensor,
    pub attn_v: TernaryTensor,
    pub attn_o: TernaryTensor,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
    pub has_q_gate: bool,
}

pub enum LinearWeight {
    Ternary(TernaryTensor),
    DenseF16 {
        data: Vec<u16>,
        n_rows: usize,
        k_cols: usize,
    },
}

pub struct GatedDeltaNetWeights {
    pub attn_qkv: TernaryTensor,
    pub attn_gate: TernaryTensor,
    pub ssm_conv1d: Vec<f32>,
    pub ssm_alpha: LinearWeight,
    pub ssm_beta: LinearWeight,
    pub ssm_a: Vec<f32>,
    pub ssm_dt_bias: Vec<f32>,
    pub ssm_norm: Vec<f32>,
    pub ssm_out: TernaryTensor,
}

pub enum LayerBlock {
    Attention(AttentionBlockWeights),
    GatedDeltaNet(GatedDeltaNetWeights),
}

pub struct TernaryLayerWeights {
    pub attn_norm: Vec<f32>,
    pub block: LayerBlock,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: TernaryTensor,
    pub ffn_up: TernaryTensor,
    pub ffn_down: TernaryTensor,
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub token_embeddings: EmbeddingTable,
    pub layers: Vec<TernaryLayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: LmHead,
}

pub struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    conv_states: Vec<Vec<f32>>,
    ssm_states: Vec<Vec<f32>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
            conv_states: vec![Vec::new(); num_layers],
            ssm_states: vec![Vec::new(); num_layers],
        }
    }
}

pub struct GenerationStats {
    pub tokens: Vec<usize>,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

/// Construit un tenseur ternaire synthétique déterministe (tests et benchmark).
pub fn build_synthetic_tensor(rows: usize, cols: usize, seed: usize) -> Result<TernaryTensor, String> {
    let weights: Vec<i8> = (0..(rows * cols))
        .map(|idx| match (idx * 7 + seed) % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect();
    let packed = pack_ternary_2bit_rows(&weights, rows, cols).map_err(|err| err.to_string())?;
    let scales = vec![f32_to_f16(0.05); rows * (cols / GROUP_SIZE)];
    TernaryTensor::from_raw(rows, cols, packed, scales).map_err(|err| err.to_string())
}

pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = input.iter().map(|&val| val * val).sum::<f32>() / (input.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    input
        .iter()
        .zip(weight.iter())
        .map(|(&val, &norm_weight)| val * inv_rms * norm_weight)
        .collect()
}

fn rms_norm_batch(input: &[f32], weight: &[f32], m: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * dim);
    for t in 0..m {
        out.extend_from_slice(&rms_norm(&input[t * dim..(t + 1) * dim], weight, eps));
    }
    out
}

fn project_linear_batch(
    backend: &dyn TernaryHadamardBackend,
    input: &[f32],
    weight: &LinearWeight,
    m: usize,
    use_hadamard: bool,
) -> Result<Vec<f32>, String> {
    match weight {
        LinearWeight::Ternary(t) => project_ternary_batch(backend, input, t, m, use_hadamard),
        LinearWeight::DenseF16 { data, n_rows, k_cols } => {
            let lut = f16_lut();
            let mut out = vec![0.0f32; m * n_rows];
            for t in 0..m {
                let x = &input[t * k_cols..(t + 1) * k_cols];
                for r in 0..*n_rows {
                    let row = &data[r * k_cols..(r + 1) * k_cols];
                    out[t * n_rows + r] = row
                        .iter()
                        .zip(x.iter())
                        .map(|(&w_u16, &xv)| lut[w_u16 as usize] * xv)
                        .sum();
                }
            }
            Ok(out)
        }
    }
}

fn load_linear_weight(
    st: &mut SafeTensorsFile,
    weight_name: &str,
    scales_name: &str,
    n_rows: usize,
    k_cols: usize,
) -> Result<LinearWeight, String> {
    if st.has_tensor(scales_name) {
        return Ok(LinearWeight::Ternary(st.load_ternary_layer(weight_name, scales_name)?));
    }
    let data = match load_dense_table(st, weight_name)? {
        EmbeddingTable::F16(v) => v,
        EmbeddingTable::F32(v) => v.into_iter().map(f32_to_f16).collect(),
        EmbeddingTable::Quantized(_) => {
            return Err(format!("Expected dense F16/F32 tensor for '{weight_name}'"));
        }
    };
    Ok(LinearWeight::DenseF16 { data, n_rows, k_cols })
}

fn project_ternary(
    backend: &dyn TernaryHadamardBackend,
    input: &[f32],
    tensor: &TernaryTensor,
    use_hadamard: bool,
) -> Result<Vec<f32>, String> {
    project_ternary_batch(backend, input, tensor, 1, use_hadamard)
}

fn project_ternary_batch(
    backend: &dyn TernaryHadamardBackend,
    input: &[f32],
    tensor: &TernaryTensor,
    m: usize,
    use_hadamard: bool,
) -> Result<Vec<f32>, String> {
    if tensor.is_binary() {
        return aibox_kernels::scalar::matmul_binary_1bit(
            input,
            &tensor.packed_weights,
            &tensor.scales_fp16,
            m,
            tensor.n_rows,
            tensor.k_cols,
            use_hadamard,
        )
        .map_err(|err| err.to_string());
    }
    backend
        .matmul_ternary(
            input,
            &tensor.packed_weights,
            &tensor.scales_fp16,
            m,
            tensor.n_rows,
            tensor.k_cols,
            use_hadamard,
        )
        .map_err(|err| err.to_string())
}

fn add_residual(target: &mut [f32], delta: &[f32]) {
    for (slot, &val) in target.iter_mut().zip(delta.iter()) {
        *slot += val;
    }
}

#[inline]
fn silu(val: f32) -> f32 {
    val / (1.0 + (-val).exp())
}

fn dot_quantized_group(x_g: &[f32], w_g: &[u8], bits: u8) -> f32 {
    if bits == 8 {
        return x_g.iter().zip(w_g.iter()).map(|(&x, &b)| x * (b as i8) as f32).sum();
    }
    w_g.iter()
        .enumerate()
        .map(|(i, &b)| {
            let q0 = ((b & 0x0F) as i32 - 8) as f32;
            let q1 = (((b >> 4) & 0x0F) as i32 - 8) as f32;
            x_g[i * 2] * q0 + x_g[i * 2 + 1] * q1
        })
        .sum()
}

fn dot_quantized_lm_head_cpu(hidden: &[f32], tensor: &TernaryTensor) -> Vec<f32> {
    let lut = f16_lut();
    let n_groups = tensor.k_cols / aibox_kernels::GROUP_SIZE;
    let bytes_per_group = (aibox_kernels::GROUP_SIZE * tensor.bits as usize) / 8;
    let row_stride = n_groups * bytes_per_group;
    (0..tensor.n_rows)
        .map(|row| {
            let w_row = &tensor.packed_weights[row * row_stride..(row + 1) * row_stride];
            let s_row = &tensor.scales_fp16[row * n_groups..(row + 1) * n_groups];
            (0..n_groups)
                .map(|g| {
                    let x_g = &hidden[g * aibox_kernels::GROUP_SIZE..(g + 1) * aibox_kernels::GROUP_SIZE];
                    let w_g = &w_row[g * bytes_per_group..(g + 1) * bytes_per_group];
                    dot_quantized_group(x_g, w_g, tensor.bits) * lut[s_row[g] as usize]
                })
                .sum()
        })
        .collect()
}

impl TransformerModel {
    /// Charge un modèle réel AI Box (config.json + quant_config.json + model.safetensors).
    pub fn load_from_dir(model_dir: &Path) -> Result<Self, String> {
        let cfg_raw = fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        let mut config = ModelConfig::from_json(&parse_json(&cfg_raw)?)?;

        let qcfg_raw = fs::read_to_string(model_dir.join("quant_config.json"))
            .map_err(|e| format!("Failed to read quant_config.json: {e}"))?;
        let qcfg = parse_json(&qcfg_raw)?;
        if let Some(hadamard) = qcfg.as_object().and_then(|o| o.get("hadamard")).and_then(JsonValue::as_bool) {
            config.use_hadamard = hadamard;
        }

        let mut st = SafeTensorsFile::open(model_dir.join("model.safetensors"))?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let p = format!("blk.{idx}");
            let attn_norm = st.load_f32_vector(&format!("{p}.attn_norm.weight"))?;
            let ffn_norm = if st.has_tensor(&format!("{p}.post_attention_norm.weight")) {
                st.load_f32_vector(&format!("{p}.post_attention_norm.weight"))?
            } else {
                st.load_f32_vector(&format!("{p}.ffn_norm.weight"))?
            };

            let block = if st.has_tensor(&format!("{p}.attn_qkv.weight")) {
                LayerBlock::GatedDeltaNet(GatedDeltaNetWeights {
                    attn_qkv: st.load_ternary_layer(&format!("{p}.attn_qkv.weight"), &format!("{p}.attn_qkv.scales"))?,
                    attn_gate: st.load_ternary_layer(&format!("{p}.attn_gate.weight"), &format!("{p}.attn_gate.scales"))?,
                    ssm_conv1d: st.load_f32_vector(&format!("{p}.ssm_conv1d.weight"))?,
                    ssm_alpha: load_linear_weight(
                        &mut st,
                        &format!("{p}.ssm_alpha.weight"),
                        &format!("{p}.ssm_alpha.scales"),
                        config.ssm_time_step_rank,
                        config.hidden_size,
                    )?,
                    ssm_beta: load_linear_weight(
                        &mut st,
                        &format!("{p}.ssm_beta.weight"),
                        &format!("{p}.ssm_beta.scales"),
                        config.ssm_time_step_rank,
                        config.hidden_size,
                    )?,
                    ssm_a: st.load_f32_vector(&format!("{p}.ssm_a"))?,
                    ssm_dt_bias: st.load_f32_vector(&format!("{p}.ssm_dt.bias"))?,
                    ssm_norm: st.load_f32_vector(&format!("{p}.ssm_norm.weight"))?,
                    ssm_out: st.load_ternary_layer(&format!("{p}.ssm_out.weight"), &format!("{p}.ssm_out.scales"))?,
                })
            } else {
                let attn_q = st.load_ternary_layer(&format!("{p}.attn_q.weight"), &format!("{p}.attn_q.scales"))?;
                let expected_q_rows = config.num_attention_heads * config.head_dim();
                let has_q_gate = attn_q.n_rows == expected_q_rows * 2;
                if !has_q_gate && config.head_dim == config.hidden_size / config.num_attention_heads {
                    config.head_dim = attn_q.n_rows / config.num_attention_heads;
                }
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
                LayerBlock::Attention(AttentionBlockWeights {
                    attn_q,
                    attn_k: st.load_ternary_layer(&format!("{p}.attn_k.weight"), &format!("{p}.attn_k.scales"))?,
                    attn_v: st.load_ternary_layer(&format!("{p}.attn_v.weight"), &format!("{p}.attn_v.scales"))?,
                    attn_o: st.load_ternary_layer(&format!("{p}.attn_o.weight"), &format!("{p}.attn_o.scales"))?,
                    attn_q_norm,
                    attn_k_norm,
                    has_q_gate,
                })
            };

            layers.push(TernaryLayerWeights {
                attn_norm,
                block,
                ffn_norm,
                ffn_gate: st.load_ternary_layer(&format!("{p}.ffn_gate.weight"), &format!("{p}.ffn_gate.scales"))?,
                ffn_up: st.load_ternary_layer(&format!("{p}.ffn_up.weight"), &format!("{p}.ffn_up.scales"))?,
                ffn_down: st.load_ternary_layer(&format!("{p}.ffn_down.weight"), &format!("{p}.ffn_down.scales"))?,
            });
        }

        let token_embeddings = load_dense_table(&mut st, "token_embd.weight")?;
        let final_norm = st.load_f32_vector("output_norm.weight")?;
        let lm_head = if st.has_tensor("output.weight") && st.has_tensor("output.scales") {
            LmHead::Ternary(st.load_ternary_layer("output.weight", "output.scales")?)
        } else if st.has_tensor("output.weight") {
            LmHead::Dense(load_dense_table(&mut st, "output.weight")?)
        } else {
            LmHead::TiedEmbedding
        };

        Ok(Self {
            config,
            token_embeddings,
            layers,
            final_norm,
            lm_head,
        })
    }

    fn forward_attention_block(
        &self,
        hidden: &mut [f32],
        layer: &TernaryLayerWeights,
        layer_idx: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<(), String> {
        match &layer.block {
            LayerBlock::Attention(attn) => self.forward_full_attention_block(
                hidden,
                &layer.attn_norm,
                attn,
                layer_idx,
                pos,
                kv_cache,
                backend,
            ),
            LayerBlock::GatedDeltaNet(gdn) => {
                self.forward_gdn_block(hidden, &layer.attn_norm, gdn, layer_idx, kv_cache, backend)
            }
        }
    }

    fn forward_ffn_block(
        &self,
        hidden: &mut [f32],
        layer: &TernaryLayerWeights,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<(), String> {
        let use_hadamard = self.config.use_hadamard;
        let normed = rms_norm(hidden, &layer.ffn_norm, self.config.rms_norm_eps);
        let gate = project_ternary(backend, &normed, &layer.ffn_gate, use_hadamard)?;
        let up = project_ternary(backend, &normed, &layer.ffn_up, use_hadamard)?;
        let swiglu: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&gate_val, &up_val)| silu(gate_val) * up_val)
            .collect();
        let ffn_out = project_ternary(backend, &swiglu, &layer.ffn_down, use_hadamard)?;
        add_residual(hidden, &ffn_out);
        Ok(())
    }

    pub fn forward_token(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        self.forward_token_internal(token_id, pos, kv_cache, backend, true)
    }

    fn forward_token_internal(
        &self,
        token_id: usize,
        pos: usize,
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
        compute_logits: bool,
    ) -> Result<Vec<f32>, String> {
        let hidden_size = self.config.hidden_size;
        if token_id >= self.config.vocab_size {
            return Err(format!("token_id {token_id} >= vocab_size {}", self.config.vocab_size));
        }

        let mut hidden = self.token_embeddings.read_row(token_id, hidden_size);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_attention_block(&mut hidden, layer, layer_idx, pos, kv_cache, backend)?;
            self.forward_ffn_block(&mut hidden, layer, backend)?;
        }

        if !compute_logits {
            return Ok(Vec::new());
        }

        let final_hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps);
        self.compute_logits_cpu(&final_hidden, backend)
    }

    fn compute_logits_cpu(
        &self,
        final_hidden: &[f32],
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        let hidden_size = self.config.hidden_size;
        match &self.lm_head {
            LmHead::Ternary(tensor) if tensor.bits == 4 || tensor.bits == 8 => {
                Ok(dot_quantized_lm_head_cpu(final_hidden, tensor))
            }
            LmHead::Ternary(tensor) => {
                project_ternary(backend, final_hidden, tensor, self.config.use_hadamard)
            }
            LmHead::Dense(table) => Ok(table.dot_all_rows(
                final_hidden,
                self.config.vocab_size,
                hidden_size,
            )),
            LmHead::TiedEmbedding => Ok(self.token_embeddings.dot_all_rows(
                final_hidden,
                self.config.vocab_size,
                hidden_size,
            )),
        }
    }

    fn prefill_prompt(
        &self,
        prompt_tokens: &[usize],
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        let m = prompt_tokens.len();
        if m == 1 {
            return self.forward_token_internal(prompt_tokens[0], 0, kv_cache, backend, true);
        }

        let hidden_size = self.config.hidden_size;
        let use_hadamard = self.config.use_hadamard;
        let eps = self.config.rms_norm_eps;

        let mut hidden = Vec::with_capacity(m * hidden_size);
        for &tok in prompt_tokens {
            if tok >= self.config.vocab_size {
                return Err(format!("token_id {tok} >= vocab_size {}", self.config.vocab_size));
            }
            hidden.extend_from_slice(&self.token_embeddings.read_row(tok, hidden_size));
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm_batch(&hidden, &layer.attn_norm, m, hidden_size, eps);
            match &layer.block {
                LayerBlock::Attention(attn) => {
                    self.prefill_full_attention_cpu(&normed, attn, layer_idx, m, &mut hidden, kv_cache, backend)?;
                }
                LayerBlock::GatedDeltaNet(gdn) => {
                    let ssm_proj = self.run_gdn_sequence_cpu(&normed, gdn, layer_idx, m, kv_cache, backend)?;
                    add_residual(&mut hidden, &ssm_proj);
                }
            }

            let normed_ffn = rms_norm_batch(&hidden, &layer.ffn_norm, m, hidden_size, eps);
            let gate = project_ternary_batch(backend, &normed_ffn, &layer.ffn_gate, m, use_hadamard)?;
            let up = project_ternary_batch(backend, &normed_ffn, &layer.ffn_up, m, use_hadamard)?;
            let swiglu: Vec<f32> = gate
                .iter()
                .zip(up.iter())
                .map(|(&g, &u)| silu(g) * u)
                .collect();
            let ffn_out = project_ternary_batch(backend, &swiglu, &layer.ffn_down, m, use_hadamard)?;
            add_residual(&mut hidden, &ffn_out);
        }

        let last_hidden = &hidden[(m - 1) * hidden_size..m * hidden_size];
        let final_hidden = rms_norm(last_hidden, &self.final_norm, eps);
        self.compute_logits_cpu(&final_hidden, backend)
    }

    pub fn generate_tokens(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        params: SamplingParams,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<GenerationStats, String> {
        self.generate_tokens_streaming(prompt_tokens, max_new_tokens, eos_token_id, params, backend, |_| {
            Ok(())
        })
    }

    pub fn generate_tokens_streaming<F>(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        params: SamplingParams,
        backend: &dyn TernaryHadamardBackend,
        mut on_token: F,
    ) -> Result<GenerationStats, String>
    where
        F: FnMut(usize) -> Result<(), String>,
    {
        if prompt_tokens.is_empty() {
            return Err("prompt_tokens cannot be empty".to_string());
        }
        if let Some(gpu) = backend.as_gpu() {
            return self.generate_tokens_streaming_gpu(
                gpu,
                prompt_tokens,
                max_new_tokens,
                eos_token_id,
                params,
                on_token,
            );
        }

        let mut kv_cache = KvCache::new(self.layers.len());
        let mut rng = XorShift64::new(params.seed);

        let prefill_start = Instant::now();
        let last_logits = self.prefill_prompt(prompt_tokens, &mut kv_cache, backend)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next_token = sample_token(&last_logits, params, &mut rng)?;

        for step in 0..max_new_tokens {
            if Some(next_token) == eos_token_id {
                break;
            }
            generated.push(next_token);
            on_token(next_token)?;
            if step + 1 == max_new_tokens {
                break;
            }
            let pos = prompt_tokens.len() + step;
            let logits = self.forward_token(next_token, pos, &mut kv_cache, backend)?;
            next_token = sample_token(&logits, params, &mut rng)?;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationStats {
            tokens: generated,
            prefill_ms,
            decode_ms,
        })
    }

    fn generate_tokens_streaming_gpu<F>(
        &self,
        gpu: &GpuBackend,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
        params: SamplingParams,
        mut on_token: F,
    ) -> Result<GenerationStats, String>
    where
        F: FnMut(usize) -> Result<(), String>,
    {
        let max_seq = prompt_tokens.len() + max_new_tokens;
        let mut session = GpuSession::new(gpu, &self.config, &self.layers, max_seq, prompt_tokens.len())?;
        let mut rng = XorShift64::new(params.seed);

        let prefill_start = Instant::now();
        self.forward_tokens_gpu(gpu, &mut session, prompt_tokens, 0, true)?;
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next_token = self.sample_next_token_gpu(gpu, &session, params, &mut rng)?;
        let h = self.config.hidden_size;
        let mut graph_captured = false;

        for step in 0..max_new_tokens {
            if Some(next_token) == eos_token_id {
                break;
            }
            generated.push(next_token);
            on_token(next_token)?;
            if step + 1 == max_new_tokens {
                break;
            }
            let pos = prompt_tokens.len() + step;
            if next_token >= self.config.vocab_size {
                return Err(format!("token_id {next_token} >= vocab_size {}", self.config.vocab_size));
            }
            if !graph_captured {
                self.capture_decode_graph_gpu(gpu, &mut session, pos)?;
                graph_captured = true;
            }
            let emb_row = self.token_embeddings.read_row(next_token, h);
            gpu.graph_launch(&mut session.hidden, &emb_row, pos).map_err(|e| e.to_string())?;
            next_token = self.sample_next_token_gpu(gpu, &session, params, &mut rng)?;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationStats {
            tokens: generated,
            prefill_ms,
            decode_ms,
        })
    }

    fn capture_decode_graph_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        pos: usize,
    ) -> Result<(), String> {
        gpu.graph_begin_capture().map_err(|e| e.to_string())?;
        let body_res = (|| {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                self.forward_attention_gpu(gpu, s, layer, layer_idx, pos, 1)?;
                self.forward_ffn_gpu(gpu, s, layer, 1)?;
            }
            self.compute_logits_gpu(gpu, s)
        })();
        let end_res = gpu.graph_end_capture().map_err(|e| e.to_string());
        body_res?;
        end_res
    }

    fn forward_tokens_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        tokens: &[usize],
        start_pos: usize,
        compute_logits: bool,
    ) -> Result<(), String> {
        let m = tokens.len();
        let h = self.config.hidden_size;
        let mut emb_batch = Vec::with_capacity(m * h);
        for &tok in tokens {
            if tok >= self.config.vocab_size {
                return Err(format!("token_id {tok} >= vocab_size {}", self.config.vocab_size));
            }
            emb_batch.extend_from_slice(&self.token_embeddings.read_row(tok, h));
        }
        s.hidden.upload_f32_as_f16(&emb_batch).map_err(|e| e.to_string())?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_attention_gpu(gpu, s, layer, layer_idx, start_pos, m)?;
            self.forward_ffn_gpu(gpu, s, layer, m)?;
        }
        if compute_logits {
            s.hidden.copy_row_to_front(m - 1, h).map_err(|e| e.to_string())?;
            self.compute_logits_gpu(gpu, s)?;
        }
        Ok(())
    }

    fn sample_next_token_gpu(
        &self,
        gpu: &GpuBackend,
        s: &GpuSession<'_>,
        params: SamplingParams,
        rng: &mut XorShift64,
    ) -> Result<usize, String> {
        if params.temperature <= 0.0 {
            return gpu.argmax_f32(&s.logits, self.config.vocab_size).map_err(|e| e.to_string());
        }
        let margin = (16.0 * params.temperature).max(12.0);
        if let Some(mut candidates) = gpu
            .sample_candidates_f32(&s.logits, self.config.vocab_size, margin, 1024)
            .map_err(|e| e.to_string())?
        {
            let enough_for_top_k = params.top_k == 0 || candidates.len() >= params.top_k;
            if enough_for_top_k && !candidates.is_empty() {
                return crate::sampling::sample_from_candidates(&mut candidates, params, rng);
            }
        }
        let mut host_logits = vec![0.0f32; self.config.vocab_size];
        s.logits.download(&mut host_logits).map_err(|e| e.to_string())?;
        sample_token(&host_logits, params, rng)
    }

    fn forward_attention_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        layer: &TernaryLayerWeights,
        layer_idx: usize,
        start_pos: usize,
        m: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;
        let use_hadamard = cfg.use_hadamard;

        if use_hadamard {
            gpu.rms_norm_fwht128_f16(&s.hidden, &layer.attn_norm, &mut s.normed, m, h, eps).map_err(|e| e.to_string())?;
        } else {
            gpu.rms_norm_f16(&s.hidden, &layer.attn_norm, &mut s.normed, m, h, eps).map_err(|e| e.to_string())?;
        }

        match &layer.block {
            LayerBlock::Attention(attn) => {
                self.forward_full_attention_gpu(gpu, s, attn, layer_idx, start_pos, m)
            }
            LayerBlock::GatedDeltaNet(gdn) => self.forward_gdn_gpu(gpu, s, gdn, layer_idx, m),
        }
    }

    fn forward_ffn_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        layer: &TernaryLayerWeights,
        m: usize,
    ) -> Result<(), String> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let use_hadamard = self.config.use_hadamard;

        if use_hadamard {
            gpu.rms_norm_fwht128_f16(&s.hidden, &layer.ffn_norm, &mut s.normed, m, h, self.config.rms_norm_eps)
                .map_err(|e| e.to_string())?;
        } else {
            gpu.rms_norm_f16(&s.hidden, &layer.ffn_norm, &mut s.normed, m, h, self.config.rms_norm_eps)
                .map_err(|e| e.to_string())?;
        }
        if m == 1 && !use_hadamard && h % 512 == 0 && layer.ffn_gate.bits == layer.ffn_up.bits {
            gpu.matmul_ternary_swiglu_f16_dev(
                &s.normed,
                &layer.ffn_gate.packed_weights,
                &layer.ffn_gate.scales_fp16,
                &layer.ffn_up.packed_weights,
                &layer.ffn_up.scales_fp16,
                &mut s.gate,
                inter,
                h,
                layer.ffn_gate.bits,
            )
            .map_err(|e| e.to_string())?;
        } else {
            gpu.matmul_ternary_f16_dev(&s.normed, &layer.ffn_gate.packed_weights, &layer.ffn_gate.scales_fp16, &mut s.gate, m, inter, h, layer.ffn_gate.bits).map_err(|e| e.to_string())?;
            gpu.matmul_ternary_f16_dev(&s.normed, &layer.ffn_up.packed_weights, &layer.ffn_up.scales_fp16, &mut s.up, m, inter, h, layer.ffn_up.bits).map_err(|e| e.to_string())?;
            if use_hadamard {
                gpu.swiglu_fwht128_inplace_f16(&mut s.gate, &s.up, m * inter).map_err(|e| e.to_string())?;
            } else {
                gpu.swiglu_inplace_f16(&mut s.gate, &s.up, m * inter).map_err(|e| e.to_string())?;
            }
        }
        gpu.matmul_ternary_f16_add_dev(&s.gate, &layer.ffn_down.packed_weights, &layer.ffn_down.scales_fp16, &mut s.hidden, m, h, inter, layer.ffn_down.bits)
            .map_err(|e| e.to_string())
    }

    fn compute_logits_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        match &self.lm_head {
            LmHead::Ternary(tensor) if tensor.bits == 4 || tensor.bits == 8 => {
                gpu.rms_norm_f16(&s.hidden, &self.final_norm, &mut s.normed, 1, h, cfg.rms_norm_eps)
                    .map_err(|e| e.to_string())?;
                gpu.lm_head_quant_act_f16(
                    &s.normed,
                    &tensor.packed_weights,
                    &tensor.scales_fp16,
                    &mut s.logits,
                    cfg.vocab_size,
                    h,
                    tensor.bits,
                )
                .map_err(|e| e.to_string())
            }
            LmHead::Ternary(tensor) => {
                if cfg.use_hadamard {
                    gpu.rms_norm_fwht128_f16(&s.hidden, &self.final_norm, &mut s.normed, 1, h, cfg.rms_norm_eps)
                        .map_err(|e| e.to_string())?;
                } else {
                    gpu.rms_norm_f16(&s.hidden, &self.final_norm, &mut s.normed, 1, h, cfg.rms_norm_eps)
                        .map_err(|e| e.to_string())?;
                }
                gpu.matmul_ternary_f16_to_f32_dev(&s.normed, &tensor.packed_weights, &tensor.scales_fp16, &mut s.logits, 1, cfg.vocab_size, h, tensor.bits).map_err(|e| e.to_string())
            }
            LmHead::Dense(_) | LmHead::TiedEmbedding => {
                gpu.rms_norm_f16(&s.hidden, &self.final_norm, &mut s.normed, 1, h, cfg.rms_norm_eps)
                    .map_err(|e| e.to_string())?;
                let table = match &self.lm_head {
                    LmHead::Dense(t) => t,
                    _ => &self.token_embeddings,
                };
                match table {
                    EmbeddingTable::F16(data) => gpu.lm_head_f16_act_f16(&s.normed, data, &mut s.logits, cfg.vocab_size, h).map_err(|e| e.to_string()),
                    EmbeddingTable::F32(data) => gpu.lm_head_f32_act_f16(&s.normed, data, &mut s.logits, cfg.vocab_size, h).map_err(|e| e.to_string()),
                    EmbeddingTable::Quantized(tensor) => gpu.lm_head_quant_act_f16(&s.normed, &tensor.packed_weights, &tensor.scales_fp16, &mut s.logits, cfg.vocab_size, h, tensor.bits).map_err(|e| e.to_string()),
                }
            }
        }
    }
}

struct GpuSession<'a> {
    hidden: GpuBuffer<'a, u16>,
    normed: GpuBuffer<'a, u16>,
    q: GpuBuffer<'a, u16>,
    k: GpuBuffer<'a, u16>,
    v: GpuBuffer<'a, u16>,
    attn_ctx: GpuBuffer<'a, u16>,
    gate: GpuBuffer<'a, u16>,
    up: GpuBuffer<'a, u16>,
    logits: GpuBuffer<'a, f32>,
    k_cache: GpuBuffer<'a, u16>,
    v_cache: GpuBuffer<'a, u16>,
    layer_kv_stride: usize,
    kv_layer_indices: Vec<usize>,
    conv_state: GpuBuffer<'a, f32>,
    ssm_state: GpuBuffer<'a, u16>,
    conv_state_stride: usize,
    ssm_state_stride: usize,
    gdn_layer_indices: Vec<usize>,
}

impl<'a> GpuSession<'a> {
    fn new(
        gpu: &'a GpuBackend,
        cfg: &ModelConfig,
        layers: &[TernaryLayerWeights],
        max_seq: usize,
        max_batch: usize,
    ) -> Result<Self, String> {
        let m = max_batch.max(1);
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;

        let mut kv_layer_indices = vec![0usize; layers.len()];
        let mut gdn_layer_indices = vec![0usize; layers.len()];
        let mut num_attn_layers = 0usize;
        let mut num_gdn_layers = 0usize;
        for (idx, layer) in layers.iter().enumerate() {
            match layer.block {
                LayerBlock::Attention(_) => {
                    kv_layer_indices[idx] = num_attn_layers;
                    num_attn_layers += 1;
                }
                LayerBlock::GatedDeltaNet(_) => {
                    gdn_layer_indices[idx] = num_gdn_layers;
                    num_gdn_layers += 1;
                }
            }
        }

        let layer_kv_stride = max_seq.max(1) * kv_dim;
        let total_kv = num_attn_layers.max(1) * layer_kv_stride;

        let conv_channels = 2 * cfg.ssm_group_count * cfg.ssm_state_size + cfg.ssm_inner_size;
        let conv_hist = cfg.ssm_conv_kernel.saturating_sub(1);
        let conv_state_stride = conv_channels * conv_hist;
        let ssm_state_stride = cfg.ssm_time_step_rank * cfg.ssm_state_size * cfg.ssm_state_size;
        let total_conv = (num_gdn_layers * conv_state_stride).max(1);
        let total_ssm = (num_gdn_layers * ssm_state_stride).max(1);

        let mut conv_state = gpu.alloc_f32(total_conv).map_err(|e| e.to_string())?;
        let mut ssm_state = gpu.alloc_f16(total_ssm).map_err(|e| e.to_string())?;
        if num_gdn_layers > 0 {
            conv_state.upload_f32(&vec![0.0f32; total_conv]).map_err(|e| e.to_string())?;
            ssm_state.upload_f32_as_f16(&vec![0.0f32; total_ssm]).map_err(|e| e.to_string())?;
        }

        Ok(Self {
            hidden: gpu.alloc_f16(m * h).map_err(|e| e.to_string())?,
            normed: gpu.alloc_f16(m * h).map_err(|e| e.to_string())?,
            q: gpu.alloc_f16(m * q_dim.max(conv_channels)).map_err(|e| e.to_string())?,
            k: gpu.alloc_f16(m * kv_dim.max(cfg.ssm_time_step_rank)).map_err(|e| e.to_string())?,
            v: gpu.alloc_f16(m * kv_dim.max(cfg.ssm_time_step_rank)).map_err(|e| e.to_string())?,
            attn_ctx: gpu.alloc_f16(m * q_dim.max(cfg.ssm_inner_size)).map_err(|e| e.to_string())?,
            gate: gpu.alloc_f16(m * inter.max(q_dim * 2).max(conv_channels)).map_err(|e| e.to_string())?,
            up: gpu.alloc_f16(m * inter.max(q_dim).max(cfg.ssm_inner_size)).map_err(|e| e.to_string())?,
            logits: gpu.alloc_f32(cfg.vocab_size).map_err(|e| e.to_string())?,
            k_cache: gpu.alloc_f16(total_kv).map_err(|e| e.to_string())?,
            v_cache: gpu.alloc_f16(total_kv).map_err(|e| e.to_string())?,
            layer_kv_stride,
            kv_layer_indices,
            conv_state,
            ssm_state,
            conv_state_stride,
            ssm_state_stride,
            gdn_layer_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aibox_kernels::TmacCpuBackend;

    #[test]
    fn forward_and_generate_work_end_to_end() {
        let config = ModelConfig {
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            use_hadamard: true,
            ..ModelConfig::default()
        };

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| TernaryLayerWeights {
                attn_norm: vec![1.0; config.hidden_size],
                block: LayerBlock::Attention(AttentionBlockWeights {
                    attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1).unwrap(),
                    attn_k: build_synthetic_tensor(64, config.hidden_size, layer_idx + 2).unwrap(),
                    attn_v: build_synthetic_tensor(64, config.hidden_size, layer_idx + 3).unwrap(),
                    attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4).unwrap(),
                    attn_q_norm: None,
                    attn_k_norm: None,
                    has_q_gate: false,
                }),
                ffn_norm: vec![1.0; config.hidden_size],
                ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 5).unwrap(),
                ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 6).unwrap(),
                ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, layer_idx + 7).unwrap(),
            })
            .collect();

        let model = TransformerModel {
            token_embeddings: EmbeddingTable::F32(
                (0..(config.vocab_size * config.hidden_size))
                    .map(|idx| ((idx as f32) * 0.01).sin())
                    .collect(),
            ),
            layers,
            final_norm: vec![1.0; config.hidden_size],
            lm_head: LmHead::Ternary(build_synthetic_tensor(config.vocab_size, config.hidden_size, 99).unwrap()),
            config,
        };

        let backend = TmacCpuBackend::new();
        let stats = model
            .generate_tokens(&[1, 5, 9], 4, None, SamplingParams::default(), &backend)
            .expect("generation failed");

        assert_eq!(stats.tokens.len(), 4);
        assert!(stats.tokens.iter().all(|&tok| tok < 256));
    }

    #[test]
    fn batched_prefill_matches_sequential_prefill() {
        let config = ModelConfig {
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            use_hadamard: false,
            ..ModelConfig::default()
        };
        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| TernaryLayerWeights {
                attn_norm: vec![1.0; config.hidden_size],
                block: LayerBlock::Attention(AttentionBlockWeights {
                    attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1).unwrap(),
                    attn_k: build_synthetic_tensor(64, config.hidden_size, layer_idx + 2).unwrap(),
                    attn_v: build_synthetic_tensor(64, config.hidden_size, layer_idx + 3).unwrap(),
                    attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4).unwrap(),
                    attn_q_norm: Some(vec![1.0; 32]),
                    attn_k_norm: Some(vec![1.0; 32]),
                    has_q_gate: false,
                }),
                ffn_norm: vec![1.0; config.hidden_size],
                ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 5).unwrap(),
                ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 6).unwrap(),
                ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, layer_idx + 7).unwrap(),
            })
            .collect();
        let model = TransformerModel {
            token_embeddings: EmbeddingTable::F32(
                (0..(config.vocab_size * config.hidden_size))
                    .map(|idx| ((idx as f32) * 0.01).sin())
                    .collect(),
            ),
            layers,
            final_norm: vec![1.0; config.hidden_size],
            lm_head: LmHead::TiedEmbedding,
            config,
        };

        let backend = TmacCpuBackend::new();
        let prompt = [3usize, 11, 27, 42];
        let mut kv_batch = KvCache::new(model.layers.len());
        let logits_batch = model.prefill_prompt(&prompt, &mut kv_batch, &backend).unwrap();

        let mut kv_seq = KvCache::new(model.layers.len());
        let mut logits_seq = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits_seq = model
                .forward_token_internal(tok, pos, &mut kv_seq, &backend, pos + 1 == prompt.len())
                .unwrap();
        }

        let max_diff = logits_batch
            .iter()
            .zip(logits_seq.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max_diff={max_diff} >= 1e-4");
    }

    #[test]
    fn dot_quantized_lm_head_cpu_matches_q8_and_q4_reference() {
        let rows = 2;
        let cols = 128;
        let act: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.03) - 1.5).collect();
        let scale_f16 = aibox_kernels::tensor::f32_to_f16(0.25);
        let scale = aibox_kernels::tensor::f16_to_f32(scale_f16);

        let q8_packed: Vec<u8> = (0..(rows * cols))
            .map(|i| (((i % 251) as i32 - 125) as i8) as u8)
            .collect();
        let q8_tensor = TernaryTensor::from_raw_with_bits(
            rows,
            cols,
            8,
            q8_packed.clone(),
            vec![scale_f16; rows],
        )
        .unwrap();
        let q8_out = dot_quantized_lm_head_cpu(&act, &q8_tensor);
        for r in 0..rows {
            let expected: f32 = (0..cols)
                .map(|c| (q8_packed[r * cols + c] as i8 as f32) * scale * act[c])
                .sum();
            assert!((q8_out[r] - expected).abs() < 1e-3, "q8 row {r}: {} vs {expected}", q8_out[r]);
        }

        let q4_packed: Vec<u8> = (0..(rows * 64)).map(|i| ((i * 37) & 0xFF) as u8).collect();
        let q4_tensor = TernaryTensor::from_raw_with_bits(
            rows,
            cols,
            4,
            q4_packed.clone(),
            vec![scale_f16; rows],
        )
        .unwrap();
        let q4_out = dot_quantized_lm_head_cpu(&act, &q4_tensor);
        for r in 0..rows {
            let mut expected = 0.0f32;
            for b in 0..64 {
                let byte = q4_packed[r * 64 + b];
                let q0 = ((byte & 0x0F) as i32 - 8) as f32;
                let q1 = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
                expected += (q0 * act[2 * b] + q1 * act[2 * b + 1]) * scale;
            }
            assert!((q4_out[r] - expected).abs() < 1e-3, "q4 row {r}: {} vs {expected}", q4_out[r]);
        }
    }

    #[test]
    fn qwen35_hybrid_batched_prefill_matches_sequential() {
        let config = ModelConfig {
            hidden_size: 128,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            rope_dim: 16,
            ssm_conv_kernel: 4,
            ssm_group_count: 2,
            ssm_inner_size: 128,
            ssm_state_size: 32,
            ssm_time_step_rank: 4,
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            use_hadamard: false,
        };
        let conv_channels = 2 * config.ssm_group_count * config.ssm_state_size + config.ssm_inner_size;
        let gdn_layer = TernaryLayerWeights {
            attn_norm: vec![1.0; config.hidden_size],
            block: LayerBlock::GatedDeltaNet(GatedDeltaNetWeights {
                attn_qkv: build_synthetic_tensor(conv_channels, config.hidden_size, 11).unwrap(),
                attn_gate: build_synthetic_tensor(config.ssm_inner_size, config.hidden_size, 12).unwrap(),
                ssm_conv1d: vec![0.25; conv_channels * config.ssm_conv_kernel],
                ssm_alpha: LinearWeight::Ternary(build_synthetic_tensor(config.ssm_time_step_rank, config.hidden_size, 13).unwrap()),
                ssm_beta: LinearWeight::Ternary(build_synthetic_tensor(config.ssm_time_step_rank, config.hidden_size, 14).unwrap()),
                ssm_a: vec![-0.3; config.ssm_time_step_rank],
                ssm_dt_bias: vec![0.1; config.ssm_time_step_rank],
                ssm_norm: vec![1.0; config.ssm_state_size],
                ssm_out: build_synthetic_tensor(config.hidden_size, config.ssm_inner_size, 15).unwrap(),
            }),
            ffn_norm: vec![1.0; config.hidden_size],
            ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, 16).unwrap(),
            ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, 17).unwrap(),
            ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, 18).unwrap(),
        };
        let attn_layer = TernaryLayerWeights {
            attn_norm: vec![1.0; config.hidden_size],
            block: LayerBlock::Attention(AttentionBlockWeights {
                attn_q: build_synthetic_tensor(config.hidden_size * 2, config.hidden_size, 21).unwrap(),
                attn_k: build_synthetic_tensor(64, config.hidden_size, 22).unwrap(),
                attn_v: build_synthetic_tensor(64, config.hidden_size, 23).unwrap(),
                attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, 24).unwrap(),
                attn_q_norm: Some(vec![1.0; 32]),
                attn_k_norm: Some(vec![1.0; 32]),
                has_q_gate: true,
            }),
            ffn_norm: vec![1.0; config.hidden_size],
            ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, 25).unwrap(),
            ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, 26).unwrap(),
            ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, 27).unwrap(),
        };
        let model = TransformerModel {
            token_embeddings: EmbeddingTable::F32(
                (0..(config.vocab_size * config.hidden_size))
                    .map(|idx| ((idx as f32) * 0.01).sin())
                    .collect(),
            ),
            layers: vec![gdn_layer, attn_layer],
            final_norm: vec![1.0; config.hidden_size],
            lm_head: LmHead::TiedEmbedding,
            config,
        };

        let backend = TmacCpuBackend::new();
        let prompt = [2usize, 7, 19, 31];
        let mut kv_batch = KvCache::new(model.layers.len());
        let logits_batch = model.prefill_prompt(&prompt, &mut kv_batch, &backend).unwrap();

        let mut kv_seq = KvCache::new(model.layers.len());
        let mut logits_seq = Vec::new();
        for (pos, &tok) in prompt.iter().enumerate() {
            logits_seq = model
                .forward_token_internal(tok, pos, &mut kv_seq, &backend, pos + 1 == prompt.len())
                .unwrap();
        }

        let max_diff = logits_batch
            .iter()
            .zip(logits_seq.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max_diff={max_diff} >= 1e-4");
    }
}
