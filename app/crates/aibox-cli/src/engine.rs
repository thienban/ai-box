//! SERV-03 : Boucle d'inférence autorégressive Transformer (GQA + RoPE + SwiGLU + KV Cache)
//! utilisant `TernaryHadamardBackend`.

use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use aibox_kernels::tensor::{f16_to_f32, f32_to_f16, pack_ternary_2bit_rows, TernaryTensor};
use aibox_kernels::{GpuBackend, GpuBuffer, TernaryHadamardBackend, GROUP_SIZE};

use crate::json::{parse_json, JsonValue};
use crate::safetensors::SafeTensorsFile;
use crate::sampling::{sample_token, softmax_inplace, SamplingParams, XorShift64};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
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
        let hidden_size = get_usize("hidden_size")?;
        let num_attention_heads = get_usize("num_attention_heads")?;
        let num_key_value_heads = obj
            .get("num_key_value_heads")
            .and_then(JsonValue::as_u64)
            .map(|num| num as usize)
            .unwrap_or(num_attention_heads);
        let head_dim = obj
            .get("head_dim")
            .and_then(JsonValue::as_u64)
            .map(|num| num as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        Ok(Self {
            hidden_size,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
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
        }
        logits
    }
}

pub enum LmHead {
    Ternary(TernaryTensor),
    TiedEmbedding,
}

pub struct TernaryLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: TernaryTensor,
    pub attn_k: TernaryTensor,
    pub attn_v: TernaryTensor,
    pub attn_o: TernaryTensor,
    pub attn_q_norm: Option<Vec<f32>>,
    pub attn_k_norm: Option<Vec<f32>>,
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
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![Vec::new(); num_layers],
            values: vec![Vec::new(); num_layers],
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

pub fn apply_rope_inplace(
    vec: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    pos: usize,
    theta: f32,
) {
    let half = head_dim / 2;
    for head in 0..num_heads {
        let base = head * head_dim;
        for pair_idx in 0..half {
            let freq = 1.0 / theta.powf((2 * pair_idx) as f32 / (head_dim as f32));
            let (sin_a, cos_a) = ((pos as f32) * freq).sin_cos();
            let first = vec[base + pair_idx];
            let second = vec[base + pair_idx + half];
            vec[base + pair_idx] = first * cos_a - second * sin_a;
            vec[base + pair_idx + half] = first * sin_a + second * cos_a;
        }
    }
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

fn rms_norm_heads_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, weight: &[f32], eps: f32) {
    for head in 0..num_heads {
        let slice = &mut vec[head * head_dim..(head + 1) * head_dim];
        let mean_sq = slice.iter().map(|&val| val * val).sum::<f32>() / (head_dim as f32);
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for (val, &w) in slice.iter_mut().zip(weight.iter()) {
            *val = *val * inv_rms * w;
        }
    }
}

fn compute_head_attention(
    q_slice: &[f32],
    keys_cache: &[f32],
    values_cache: &[f32],
    kv_offset: usize,
    kv_dim: usize,
    out_slice: &mut [f32],
) {
    let head_dim = q_slice.len();
    let seq_len = keys_cache.len() / kv_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len];

    for step in 0..seq_len {
        let k_slice = &keys_cache[step * kv_dim + kv_offset..][..head_dim];
        let dot: f32 = q_slice
            .iter()
            .zip(k_slice.iter())
            .map(|(&q_val, &k_val)| q_val * k_val)
            .sum();
        scores[step] = dot * scale;
    }

    softmax_inplace(&mut scores);

    for (step, &prob) in scores.iter().enumerate() {
        let v_slice = &values_cache[step * kv_dim + kv_offset..][..head_dim];
        for dim_idx in 0..head_dim {
            out_slice[dim_idx] += prob * v_slice[dim_idx];
        }
    }
}

fn attention_gqa(
    query: &[f32],
    keys_cache: &[f32],
    values_cache: &[f32],
    config: &ModelConfig,
) -> Vec<f32> {
    let head_dim = config.head_dim();
    let kv_dim = config.num_key_value_heads * head_dim;
    let heads_per_kv = config.num_attention_heads / config.num_key_value_heads;
    let q_dim = config.num_attention_heads * head_dim;
    let mut output = vec![0.0f32; q_dim];

    for q_head in 0..config.num_attention_heads {
        let kv_offset = (q_head / heads_per_kv) * head_dim;
        let q_slice = &query[q_head * head_dim..(q_head + 1) * head_dim];
        let out_slice = &mut output[q_head * head_dim..(q_head + 1) * head_dim];
        compute_head_attention(q_slice, keys_cache, values_cache, kv_offset, kv_dim, out_slice);
    }

    output
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
            let attn_q = st.load_ternary_layer(&format!("{p}.attn_q.weight"), &format!("{p}.attn_q.scales"))?;
            if idx == 0 {
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
            layers.push(TernaryLayerWeights {
                attn_norm: st.load_f32_vector(&format!("{p}.attn_norm.weight"))?,
                attn_q,
                attn_k: st.load_ternary_layer(&format!("{p}.attn_k.weight"), &format!("{p}.attn_k.scales"))?,
                attn_v: st.load_ternary_layer(&format!("{p}.attn_v.weight"), &format!("{p}.attn_v.scales"))?,
                attn_o: st.load_ternary_layer(&format!("{p}.attn_o.weight"), &format!("{p}.attn_o.scales"))?,
                attn_q_norm,
                attn_k_norm,
                ffn_norm: st.load_f32_vector(&format!("{p}.ffn_norm.weight"))?,
                ffn_gate: st.load_ternary_layer(&format!("{p}.ffn_gate.weight"), &format!("{p}.ffn_gate.scales"))?,
                ffn_up: st.load_ternary_layer(&format!("{p}.ffn_up.weight"), &format!("{p}.ffn_up.scales"))?,
                ffn_down: st.load_ternary_layer(&format!("{p}.ffn_down.weight"), &format!("{p}.ffn_down.scales"))?,
            });
        }

        let token_embeddings = if st.tensors().get("token_embd.weight").map(|m| m.dtype.as_str()) == Some("F16") {
            EmbeddingTable::F16(st.load_f16_raw("token_embd.weight")?)
        } else {
            EmbeddingTable::F32(st.load_f32_vector("token_embd.weight")?)
        };
        let final_norm = st.load_f32_vector("output_norm.weight")?;
        let lm_head = if st.has_tensor("output.weight") && st.has_tensor("output.scales") {
            LmHead::Ternary(st.load_ternary_layer("output.weight", "output.scales")?)
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
        let head_dim = self.config.head_dim();
        let use_hadamard = self.config.use_hadamard;
        let normed = rms_norm(hidden, &layer.attn_norm, self.config.rms_norm_eps);
        let mut query = project_ternary(backend, &normed, &layer.attn_q, use_hadamard)?;
        let mut key = project_ternary(backend, &normed, &layer.attn_k, use_hadamard)?;
        let value = project_ternary(backend, &normed, &layer.attn_v, use_hadamard)?;

        if let Some(ref q_norm) = layer.attn_q_norm {
            rms_norm_heads_inplace(
                &mut query,
                self.config.num_attention_heads,
                head_dim,
                q_norm,
                self.config.rms_norm_eps,
            );
        }
        if let Some(ref k_norm) = layer.attn_k_norm {
            rms_norm_heads_inplace(
                &mut key,
                self.config.num_key_value_heads,
                head_dim,
                k_norm,
                self.config.rms_norm_eps,
            );
        }

        apply_rope_inplace(&mut query, self.config.num_attention_heads, head_dim, pos, self.config.rope_theta);
        apply_rope_inplace(&mut key, self.config.num_key_value_heads, head_dim, pos, self.config.rope_theta);

        kv_cache.keys[layer_idx].extend_from_slice(&key);
        kv_cache.values[layer_idx].extend_from_slice(&value);

        let attn_ctx = attention_gqa(&query, &kv_cache.keys[layer_idx], &kv_cache.values[layer_idx], &self.config);
        let attn_proj = project_ternary(backend, &attn_ctx, &layer.attn_o, use_hadamard)?;
        add_residual(hidden, &attn_proj);
        Ok(())
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
        match &self.lm_head {
            LmHead::Ternary(tensor) => {
                project_ternary(backend, &final_hidden, tensor, self.config.use_hadamard)
            }
            LmHead::TiedEmbedding => Ok(self.token_embeddings.dot_all_rows(
                &final_hidden,
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
        let head_dim = self.config.head_dim();
        let q_dim = self.config.num_attention_heads * head_dim;
        let kv_dim = self.config.num_key_value_heads * head_dim;
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
            let mut queries = project_ternary_batch(backend, &normed, &layer.attn_q, m, use_hadamard)?;
            let mut keys = project_ternary_batch(backend, &normed, &layer.attn_k, m, use_hadamard)?;
            let values = project_ternary_batch(backend, &normed, &layer.attn_v, m, use_hadamard)?;

            for t in 0..m {
                let q_slice = &mut queries[t * q_dim..(t + 1) * q_dim];
                let k_slice = &mut keys[t * kv_dim..(t + 1) * kv_dim];
                if let Some(ref q_norm) = layer.attn_q_norm {
                    rms_norm_heads_inplace(q_slice, self.config.num_attention_heads, head_dim, q_norm, eps);
                }
                if let Some(ref k_norm) = layer.attn_k_norm {
                    rms_norm_heads_inplace(k_slice, self.config.num_key_value_heads, head_dim, k_norm, eps);
                }
                apply_rope_inplace(q_slice, self.config.num_attention_heads, head_dim, t, self.config.rope_theta);
                apply_rope_inplace(k_slice, self.config.num_key_value_heads, head_dim, t, self.config.rope_theta);
            }

            kv_cache.keys[layer_idx].extend_from_slice(&keys);
            kv_cache.values[layer_idx].extend_from_slice(&values);

            let mut attn_ctx_batch = Vec::with_capacity(m * q_dim);
            for t in 0..m {
                let q_slice = &queries[t * q_dim..(t + 1) * q_dim];
                let end_kv = (t + 1) * kv_dim;
                let ctx = attention_gqa(
                    q_slice,
                    &kv_cache.keys[layer_idx][..end_kv],
                    &kv_cache.values[layer_idx][..end_kv],
                    &self.config,
                );
                attn_ctx_batch.extend_from_slice(&ctx);
            }

            let attn_proj = project_ternary_batch(backend, &attn_ctx_batch, &layer.attn_o, m, use_hadamard)?;
            add_residual(&mut hidden, &attn_proj);

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
        match &self.lm_head {
            LmHead::Ternary(tensor) => project_ternary(backend, &final_hidden, tensor, use_hadamard),
            LmHead::TiedEmbedding => Ok(self.token_embeddings.dot_all_rows(
                &final_hidden,
                self.config.vocab_size,
                hidden_size,
            )),
        }
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
        let mut session = GpuSession::new(gpu, &self.config, self.layers.len(), max_seq)?;
        let mut rng = XorShift64::new(params.seed);

        let prefill_start = Instant::now();
        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            let need_logits = pos + 1 == prompt_tokens.len();
            last_logits = self.forward_token_gpu(gpu, &mut session, tok, pos, need_logits)?;
        }
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
            let logits = self.forward_token_gpu(gpu, &mut session, next_token, pos, true)?;
            next_token = sample_token(&logits, params, &mut rng)?;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationStats {
            tokens: generated,
            prefill_ms,
            decode_ms,
        })
    }

    fn forward_token_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        token_id: usize,
        pos: usize,
        compute_logits: bool,
    ) -> Result<Vec<f32>, String> {
        let h = self.config.hidden_size;
        if token_id >= self.config.vocab_size {
            return Err(format!("token_id {token_id} >= vocab_size {}", self.config.vocab_size));
        }
        let emb_row = self.token_embeddings.read_row(token_id, h);
        s.hidden.upload(&emb_row).map_err(|e| e.to_string())?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_attention_gpu(gpu, s, layer, layer_idx, pos)?;
            self.forward_ffn_gpu(gpu, s, layer)?;
        }

        if !compute_logits {
            return Ok(Vec::new());
        }
        self.forward_lm_head_gpu(gpu, s)
    }

    fn forward_attention_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        layer: &TernaryLayerWeights,
        layer_idx: usize,
        pos: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let eps = cfg.rms_norm_eps;
        let use_hadamard = cfg.use_hadamard;

        gpu.rms_norm(&s.hidden, &layer.attn_norm, &mut s.normed, 1, h, eps).map_err(|e| e.to_string())?;
        if use_hadamard {
            gpu.fwht128_inplace(&mut s.normed, 1, h).map_err(|e| e.to_string())?;
        }
        gpu.matmul_ternary_dev(&s.normed, &layer.attn_q.packed_weights, &layer.attn_q.scales_fp16, &mut s.q, 1, q_dim, h).map_err(|e| e.to_string())?;
        gpu.matmul_ternary_dev(&s.normed, &layer.attn_k.packed_weights, &layer.attn_k.scales_fp16, &mut s.k, 1, kv_dim, h).map_err(|e| e.to_string())?;
        gpu.matmul_ternary_dev(&s.normed, &layer.attn_v.packed_weights, &layer.attn_v.scales_fp16, &mut s.v, 1, kv_dim, h).map_err(|e| e.to_string())?;

        gpu.rms_norm_heads_rope(&mut s.q, layer.attn_q_norm.as_deref(), cfg.num_attention_heads, head_dim, pos, cfg.rope_theta, eps).map_err(|e| e.to_string())?;
        gpu.rms_norm_heads_rope(&mut s.k, layer.attn_k_norm.as_deref(), cfg.num_key_value_heads, head_dim, pos, cfg.rope_theta, eps).map_err(|e| e.to_string())?;

        gpu.kv_append_gqa_layer(
            &s.q,
            &s.k,
            &s.v,
            &mut s.k_cache,
            &mut s.v_cache,
            layer_idx * s.layer_kv_stride,
            &mut s.attn_ctx,
            pos,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            head_dim,
        ).map_err(|e| e.to_string())?;

        if use_hadamard {
            gpu.fwht128_inplace(&mut s.attn_ctx, 1, q_dim).map_err(|e| e.to_string())?;
        }
        gpu.matmul_ternary_dev(&s.attn_ctx, &layer.attn_o.packed_weights, &layer.attn_o.scales_fp16, &mut s.attn_proj, 1, h, q_dim).map_err(|e| e.to_string())?;
        gpu.add_inplace(&mut s.hidden, &s.attn_proj, h).map_err(|e| e.to_string())
    }

    fn forward_ffn_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        layer: &TernaryLayerWeights,
    ) -> Result<(), String> {
        let h = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        let use_hadamard = self.config.use_hadamard;

        gpu.rms_norm(&s.hidden, &layer.ffn_norm, &mut s.normed, 1, h, self.config.rms_norm_eps).map_err(|e| e.to_string())?;
        if use_hadamard {
            gpu.fwht128_inplace(&mut s.normed, 1, h).map_err(|e| e.to_string())?;
        }
        gpu.matmul_ternary_dev(&s.normed, &layer.ffn_gate.packed_weights, &layer.ffn_gate.scales_fp16, &mut s.gate, 1, inter, h).map_err(|e| e.to_string())?;
        gpu.matmul_ternary_dev(&s.normed, &layer.ffn_up.packed_weights, &layer.ffn_up.scales_fp16, &mut s.up, 1, inter, h).map_err(|e| e.to_string())?;
        gpu.swiglu_inplace(&mut s.gate, &s.up, inter).map_err(|e| e.to_string())?;
        if use_hadamard {
            gpu.fwht128_inplace(&mut s.gate, 1, inter).map_err(|e| e.to_string())?;
        }
        gpu.matmul_ternary_dev(&s.gate, &layer.ffn_down.packed_weights, &layer.ffn_down.scales_fp16, &mut s.ffn_out, 1, h, inter).map_err(|e| e.to_string())?;
        gpu.add_inplace(&mut s.hidden, &s.ffn_out, h).map_err(|e| e.to_string())
    }

    fn forward_lm_head_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        gpu.rms_norm(&s.hidden, &self.final_norm, &mut s.normed, 1, h, cfg.rms_norm_eps).map_err(|e| e.to_string())?;
        match &self.lm_head {
            LmHead::Ternary(tensor) => {
                if cfg.use_hadamard {
                    gpu.fwht128_inplace(&mut s.normed, 1, h).map_err(|e| e.to_string())?;
                }
                gpu.matmul_ternary_dev(&s.normed, &tensor.packed_weights, &tensor.scales_fp16, &mut s.logits, 1, cfg.vocab_size, h).map_err(|e| e.to_string())?;
            }
            LmHead::TiedEmbedding => match &self.token_embeddings {
                EmbeddingTable::F16(data) => gpu.lm_head_f16(&s.normed, data, &mut s.logits, cfg.vocab_size, h).map_err(|e| e.to_string())?,
                EmbeddingTable::F32(data) => gpu.lm_head_f32(&s.normed, data, &mut s.logits, cfg.vocab_size, h).map_err(|e| e.to_string())?,
            },
        }
        let mut host_logits = vec![0.0f32; cfg.vocab_size];
        s.logits.download(&mut host_logits).map_err(|e| e.to_string())?;
        Ok(host_logits)
    }
}

struct GpuSession<'a> {
    hidden: GpuBuffer<'a>,
    normed: GpuBuffer<'a>,
    q: GpuBuffer<'a>,
    k: GpuBuffer<'a>,
    v: GpuBuffer<'a>,
    attn_ctx: GpuBuffer<'a>,
    attn_proj: GpuBuffer<'a>,
    gate: GpuBuffer<'a>,
    up: GpuBuffer<'a>,
    ffn_out: GpuBuffer<'a>,
    logits: GpuBuffer<'a>,
    k_cache: GpuBuffer<'a>,
    v_cache: GpuBuffer<'a>,
    layer_kv_stride: usize,
}

impl<'a> GpuSession<'a> {
    fn new(gpu: &'a GpuBackend, cfg: &ModelConfig, num_layers: usize, max_seq: usize) -> Result<Self, String> {
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;
        let layer_kv_stride = max_seq.max(1) * kv_dim;
        let total_kv = num_layers.max(1) * layer_kv_stride;

        Ok(Self {
            hidden: gpu.alloc_f32(h).map_err(|e| e.to_string())?,
            normed: gpu.alloc_f32(h).map_err(|e| e.to_string())?,
            q: gpu.alloc_f32(q_dim).map_err(|e| e.to_string())?,
            k: gpu.alloc_f32(kv_dim).map_err(|e| e.to_string())?,
            v: gpu.alloc_f32(kv_dim).map_err(|e| e.to_string())?,
            attn_ctx: gpu.alloc_f32(q_dim).map_err(|e| e.to_string())?,
            attn_proj: gpu.alloc_f32(h).map_err(|e| e.to_string())?,
            gate: gpu.alloc_f32(inter).map_err(|e| e.to_string())?,
            up: gpu.alloc_f32(inter).map_err(|e| e.to_string())?,
            ffn_out: gpu.alloc_f32(h).map_err(|e| e.to_string())?,
            logits: gpu.alloc_f32(cfg.vocab_size).map_err(|e| e.to_string())?,
            k_cache: gpu.alloc_f32(total_kv).map_err(|e| e.to_string())?,
            v_cache: gpu.alloc_f32(total_kv).map_err(|e| e.to_string())?,
            layer_kv_stride,
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
        };

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| TernaryLayerWeights {
                attn_norm: vec![1.0; config.hidden_size],
                attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1).unwrap(),
                attn_k: build_synthetic_tensor(64, config.hidden_size, layer_idx + 2).unwrap(),
                attn_v: build_synthetic_tensor(64, config.hidden_size, layer_idx + 3).unwrap(),
                attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4).unwrap(),
                attn_q_norm: None,
                attn_k_norm: None,
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
        };
        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| TernaryLayerWeights {
                attn_norm: vec![1.0; config.hidden_size],
                attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1).unwrap(),
                attn_k: build_synthetic_tensor(64, config.hidden_size, layer_idx + 2).unwrap(),
                attn_v: build_synthetic_tensor(64, config.hidden_size, layer_idx + 3).unwrap(),
                attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4).unwrap(),
                attn_q_norm: Some(vec![1.0; 32]),
                attn_k_norm: Some(vec![1.0; 32]),
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
}
