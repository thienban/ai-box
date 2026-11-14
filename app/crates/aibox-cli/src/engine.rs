//! SERV-03 : Boucle d'inférence autorégressive Transformer (GQA + RoPE + SwiGLU + KV Cache)
//! utilisant `TernaryHadamardBackend`.

use std::time::Instant;

use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows, TernaryTensor};
use aibox_kernels::{TernaryHadamardBackend, GROUP_SIZE};

use crate::json::JsonValue;
use crate::sampling::{sample_token, softmax_inplace, SamplingParams, XorShift64};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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

        Ok(Self {
            hidden_size,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            vocab_size: get_usize("vocab_size")?,
            rms_norm_eps: match obj.get("rms_norm_eps") {
                Some(JsonValue::Number(num)) => *num as f32,
                _ => 1e-6,
            },
            rope_theta: match obj.get("rope_theta") {
                Some(JsonValue::Number(num)) => *num as f32,
                _ => 10_000.0,
            },
        })
    }
}

pub struct TernaryLayerWeights {
    pub attn_norm: Vec<f32>,
    pub attn_q: TernaryTensor,
    pub attn_k: TernaryTensor,
    pub attn_v: TernaryTensor,
    pub attn_o: TernaryTensor,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate: TernaryTensor,
    pub ffn_up: TernaryTensor,
    pub ffn_down: TernaryTensor,
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub token_embeddings: Vec<f32>,
    pub layers: Vec<TernaryLayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: TernaryTensor,
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
) -> Result<Vec<f32>, String> {
    backend
        .matmul_hadamard_ternary(
            input,
            &tensor.packed_weights,
            &tensor.scales_fp16,
            1,
            tensor.n_rows,
            tensor.k_cols,
        )
        .map_err(|err| err.to_string())
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
    let mut output = vec![0.0f32; config.hidden_size];

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
        let normed = rms_norm(hidden, &layer.attn_norm, self.config.rms_norm_eps);
        let mut query = project_ternary(backend, &normed, &layer.attn_q)?;
        let mut key = project_ternary(backend, &normed, &layer.attn_k)?;
        let value = project_ternary(backend, &normed, &layer.attn_v)?;

        apply_rope_inplace(&mut query, self.config.num_attention_heads, head_dim, pos, self.config.rope_theta);
        apply_rope_inplace(&mut key, self.config.num_key_value_heads, head_dim, pos, self.config.rope_theta);

        kv_cache.keys[layer_idx].extend_from_slice(&key);
        kv_cache.values[layer_idx].extend_from_slice(&value);

        let attn_ctx = attention_gqa(&query, &kv_cache.keys[layer_idx], &kv_cache.values[layer_idx], &self.config);
        let attn_proj = project_ternary(backend, &attn_ctx, &layer.attn_o)?;
        add_residual(hidden, &attn_proj);
        Ok(())
    }

    fn forward_ffn_block(
        &self,
        hidden: &mut [f32],
        layer: &TernaryLayerWeights,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<(), String> {
        let normed = rms_norm(hidden, &layer.ffn_norm, self.config.rms_norm_eps);
        let gate = project_ternary(backend, &normed, &layer.ffn_gate)?;
        let up = project_ternary(backend, &normed, &layer.ffn_up)?;
        let swiglu: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&gate_val, &up_val)| silu(gate_val) * up_val)
            .collect();
        let ffn_out = project_ternary(backend, &swiglu, &layer.ffn_down)?;
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
        let hidden_size = self.config.hidden_size;
        if token_id >= self.config.vocab_size {
            return Err(format!("token_id {token_id} >= vocab_size {}", self.config.vocab_size));
        }

        let start = token_id * hidden_size;
        let mut hidden = self.token_embeddings[start..start + hidden_size].to_vec();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.forward_attention_block(&mut hidden, layer, layer_idx, pos, kv_cache, backend)?;
            self.forward_ffn_block(&mut hidden, layer, backend)?;
        }

        let final_hidden = rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps);
        project_ternary(backend, &final_hidden, &self.lm_head)
    }

    fn prefill_prompt(
        &self,
        prompt_tokens: &[usize],
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        let mut last_logits = Vec::new();
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            last_logits = self.forward_token(tok, pos, kv_cache, backend)?;
        }
        Ok(last_logits)
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
            return Err("prompt_tokens cannot be empty".to_string());
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
            vocab_size: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        };

        let layers = (0..config.num_hidden_layers)
            .map(|layer_idx| TernaryLayerWeights {
                attn_norm: vec![1.0; config.hidden_size],
                attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1).unwrap(),
                attn_k: build_synthetic_tensor(64, config.hidden_size, layer_idx + 2).unwrap(),
                attn_v: build_synthetic_tensor(64, config.hidden_size, layer_idx + 3).unwrap(),
                attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4).unwrap(),
                ffn_norm: vec![1.0; config.hidden_size],
                ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 5).unwrap(),
                ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 6).unwrap(),
                ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, layer_idx + 7).unwrap(),
            })
            .collect();

        let model = TransformerModel {
            token_embeddings: (0..(config.vocab_size * config.hidden_size))
                .map(|idx| ((idx as f32) * 0.01).sin())
                .collect(),
            layers,
            final_norm: vec![1.0; config.hidden_size],
            lm_head: build_synthetic_tensor(config.vocab_size, config.hidden_size, 99).unwrap(),
            config,
        };

        let backend = TmacCpuBackend::new();
        let stats = model
            .generate_tokens(&[1, 5, 9], 4, None, SamplingParams::default(), &backend)
            .expect("generation failed");

        assert_eq!(stats.tokens.len(), 4);
        assert!(stats.tokens.iter().all(|&tok| tok < 256));
    }
}
