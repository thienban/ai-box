use aibox_kernels::{GpuBackend, TernaryHadamardBackend};

use super::attention::sigmoid;
use super::{
    add_residual, project_linear_batch, project_ternary_batch, rms_norm, silu,
    GatedDeltaNetWeights, GpuSession, KvCache, LinearWeight, TransformerModel,
};

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn l2_norm_heads_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, eps: f32) {
    for head in 0..num_heads {
        let slice = &mut vec[head * head_dim..(head + 1) * head_dim];
        let sum_sq: f32 = slice.iter().map(|&v| v * v).sum();
        let inv_norm = 1.0 / (sum_sq + eps).sqrt();
        for v in slice.iter_mut() {
            *v *= inv_norm;
        }
    }
}

fn apply_causal_conv1d_silu(
    qkv: &[f32],
    conv_weight: &[f32],
    conv_state: &mut [f32],
    conv_channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let hist = kernel.saturating_sub(1);
    let mut out = vec![0.0f32; conv_channels];
    for ch in 0..conv_channels {
        let w = &conv_weight[ch * kernel..(ch + 1) * kernel];
        let st = &mut conv_state[ch * hist..(ch + 1) * hist];
        let mut acc = 0.0f32;
        for k in 0..hist {
            acc += st[k] * w[k];
        }
        acc += qkv[ch] * w[hist];
        for k in 0..hist.saturating_sub(1) {
            st[k] = st[k + 1];
        }
        if hist > 0 {
            st[hist - 1] = qkv[ch];
        }
        out[ch] = silu(acc);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn step_gdn_recurrence(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    alpha: &[f32],
    beta_raw: &[f32],
    ssm_a: &[f32],
    ssm_dt_bias: &[f32],
    ssm_state: &mut [f32],
    num_k_heads: usize,
    num_v_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_group = num_v_heads / num_k_heads;
    let q_scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; num_v_heads * head_dim];

    for hv in 0..num_v_heads {
        let hk = hv / heads_per_group;
        let q_h = &q[hk * head_dim..(hk + 1) * head_dim];
        let k_h = &k[hk * head_dim..(hk + 1) * head_dim];
        let v_h = &v[hv * head_dim..(hv + 1) * head_dim];
        let state_h = &mut ssm_state[hv * head_dim * head_dim..(hv + 1) * head_dim * head_dim];

        let decay = (ssm_a[hv] * softplus(alpha[hv] + ssm_dt_bias[hv])).exp();
        let beta = sigmoid(beta_raw[hv]);

        for val in state_h.iter_mut() {
            *val *= decay;
        }
        for j in 0..head_dim {
            let mut kv_mem = 0.0f32;
            for i in 0..head_dim {
                kv_mem += state_h[i * head_dim + j] * k_h[i];
            }
            let delta = (v_h[j] - kv_mem) * beta;
            let mut o_j = 0.0f32;
            for i in 0..head_dim {
                let updated = state_h[i * head_dim + j] + k_h[i] * delta;
                state_h[i * head_dim + j] = updated;
                o_j += updated * q_h[i] * q_scale;
            }
            out[hv * head_dim + j] = o_j;
        }
    }
    out
}

fn apply_gated_rms_norm(
    out: &[f32],
    z: &[f32],
    norm_weight: &[f32],
    num_v_heads: usize,
    head_dim: usize,
    eps: f32,
) -> Vec<f32> {
    let mut result = vec![0.0f32; num_v_heads * head_dim];
    for hv in 0..num_v_heads {
        let base = hv * head_dim;
        let o_h = &out[base..base + head_dim];
        let z_h = &z[base..base + head_dim];
        let mean_sq = o_h.iter().map(|&val| val * val).sum::<f32>() / (head_dim as f32);
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        for j in 0..head_dim {
            result[base + j] = o_h[j] * inv_rms * norm_weight[j] * silu(z_h[j]);
        }
    }
    result
}

impl TransformerModel {
    pub(super) fn forward_gdn_block(
        &self,
        hidden: &mut [f32],
        attn_norm: &[f32],
        gdn: &GatedDeltaNetWeights,
        layer_idx: usize,
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<(), String> {
        let normed = rms_norm(hidden, attn_norm, self.config.rms_norm_eps);
        let ssm_proj = self.run_gdn_sequence_cpu(&normed, gdn, layer_idx, 1, kv_cache, backend)?;
        add_residual(hidden, &ssm_proj);
        Ok(())
    }

    pub(super) fn run_gdn_sequence_cpu(
        &self,
        normed: &[f32],
        gdn: &GatedDeltaNetWeights,
        layer_idx: usize,
        m: usize,
        kv_cache: &mut KvCache,
        backend: &dyn TernaryHadamardBackend,
    ) -> Result<Vec<f32>, String> {
        let cfg = &self.config;
        let head_dim = cfg.ssm_state_size;
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let kernel = cfg.ssm_conv_kernel;
        let conv_channels = 2 * num_k_heads * head_dim + num_v_heads * head_dim;
        let d_inner = num_v_heads * head_dim;
        let use_hadamard = cfg.use_hadamard;
        let q_len = num_k_heads * head_dim;
        let k_len = num_k_heads * head_dim;

        let qkv_batch = project_ternary_batch(backend, normed, &gdn.attn_qkv, m, use_hadamard)?;
        let z_batch = project_ternary_batch(backend, normed, &gdn.attn_gate, m, use_hadamard)?;
        let alpha_batch = project_linear_batch(backend, normed, &gdn.ssm_alpha, m, use_hadamard)?;
        let beta_batch = project_linear_batch(backend, normed, &gdn.ssm_beta, m, use_hadamard)?;

        let conv_state = &mut kv_cache.conv_states[layer_idx];
        let conv_hist = kernel.saturating_sub(1);
        if conv_state.is_empty() {
            conv_state.resize(conv_channels * conv_hist, 0.0);
        }
        let ssm_state = &mut kv_cache.ssm_states[layer_idx];
        if ssm_state.is_empty() {
            ssm_state.resize(num_v_heads * head_dim * head_dim, 0.0);
        }

        let mut gated_batch = Vec::with_capacity(m * d_inner);
        for t in 0..m {
            let mut conv_out = apply_causal_conv1d_silu(
                &qkv_batch[t * conv_channels..(t + 1) * conv_channels],
                &gdn.ssm_conv1d,
                conv_state,
                conv_channels,
                kernel,
            );
            let (q_slice, rest) = conv_out.split_at_mut(q_len);
            let (k_slice, v_slice) = rest.split_at_mut(k_len);
            l2_norm_heads_inplace(q_slice, num_k_heads, head_dim, cfg.rms_norm_eps);
            l2_norm_heads_inplace(k_slice, num_k_heads, head_dim, cfg.rms_norm_eps);

            let out_t = step_gdn_recurrence(
                q_slice,
                k_slice,
                v_slice,
                &alpha_batch[t * num_v_heads..(t + 1) * num_v_heads],
                &beta_batch[t * num_v_heads..(t + 1) * num_v_heads],
                &gdn.ssm_a,
                &gdn.ssm_dt_bias,
                ssm_state,
                num_k_heads,
                num_v_heads,
                head_dim,
            );
            let gated_t = apply_gated_rms_norm(
                &out_t,
                &z_batch[t * d_inner..(t + 1) * d_inner],
                &gdn.ssm_norm,
                num_v_heads,
                head_dim,
                cfg.rms_norm_eps,
            );
            gated_batch.extend_from_slice(&gated_t);
        }
        project_ternary_batch(backend, &gated_batch, &gdn.ssm_out, m, use_hadamard)
    }

    pub(super) fn forward_gdn_gpu<'a>(
        &self,
        gpu: &'a GpuBackend,
        s: &mut GpuSession<'a>,
        gdn: &GatedDeltaNetWeights,
        layer_idx: usize,
        m: usize,
    ) -> Result<(), String> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let ssm_head_dim = cfg.ssm_state_size;
        let num_k_heads = cfg.ssm_group_count;
        let num_v_heads = cfg.ssm_time_step_rank;
        let kernel = cfg.ssm_conv_kernel;
        let conv_channels = 2 * num_k_heads * ssm_head_dim + num_v_heads * ssm_head_dim;
        let d_inner = num_v_heads * ssm_head_dim;
        let gdn_idx = s.gdn_layer_indices[layer_idx];

        gpu.matmul_ternary_f16_dev(&s.normed, &gdn.attn_qkv.packed_weights, &gdn.attn_qkv.scales_fp16, &mut s.gate, m, conv_channels, h, gdn.attn_qkv.bits).map_err(|e| e.to_string())?;
        gpu.matmul_ternary_f16_dev(&s.normed, &gdn.attn_gate.packed_weights, &gdn.attn_gate.scales_fp16, &mut s.up, m, d_inner, h, gdn.attn_gate.bits).map_err(|e| e.to_string())?;
        match &gdn.ssm_alpha {
            LinearWeight::Ternary(t) => gpu.matmul_ternary_f16_dev(&s.normed, &t.packed_weights, &t.scales_fp16, &mut s.k, m, num_v_heads, h, t.bits).map_err(|e| e.to_string())?,
            LinearWeight::DenseF16 { data, n_rows, k_cols } => gpu.matmul_dense_f16_dev(&s.normed, data, &mut s.k, m, *n_rows, *k_cols).map_err(|e| e.to_string())?,
        }
        match &gdn.ssm_beta {
            LinearWeight::Ternary(t) => gpu.matmul_ternary_f16_dev(&s.normed, &t.packed_weights, &t.scales_fp16, &mut s.v, m, num_v_heads, h, t.bits).map_err(|e| e.to_string())?,
            LinearWeight::DenseF16 { data, n_rows, k_cols } => gpu.matmul_dense_f16_dev(&s.normed, data, &mut s.v, m, *n_rows, *k_cols).map_err(|e| e.to_string())?,
        }

        for t in 0..m {
            gpu.gdn_conv_silu_f16(
                &s.gate,
                t * conv_channels,
                &gdn.ssm_conv1d,
                &mut s.conv_state,
                gdn_idx * s.conv_state_stride,
                &mut s.q,
                conv_channels,
                kernel,
            ).map_err(|e| e.to_string())?;
            gpu.gdn_step_f16(
                &s.q,
                &s.k,
                t * num_v_heads,
                &s.v,
                t * num_v_heads,
                &gdn.ssm_a,
                &gdn.ssm_dt_bias,
                &gdn.ssm_norm,
                &s.up,
                t * d_inner,
                &mut s.ssm_state,
                gdn_idx * s.ssm_state_stride,
                &mut s.attn_ctx,
                t * d_inner,
                num_k_heads,
                num_v_heads,
                ssm_head_dim,
                cfg.rms_norm_eps,
            ).map_err(|e| e.to_string())?;
        }

        if cfg.use_hadamard {
            gpu.fwht128_inplace_f16(&mut s.attn_ctx, m, d_inner).map_err(|e| e.to_string())?;
        }
        gpu.matmul_ternary_f16_add_dev(&s.attn_ctx, &gdn.ssm_out.packed_weights, &gdn.ssm_out.scales_fp16, &mut s.hidden, m, h, d_inner, gdn.ssm_out.bits)
            .map_err(|e| e.to_string())
    }
}
