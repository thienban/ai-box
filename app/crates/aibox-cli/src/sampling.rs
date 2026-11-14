//! SERV-03 : Fonctions de sampling (greedy, temperature, top-k, top-p) sans dépendance externe.

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            seed: 42,
        }
    }
}

/// Générateur pseudo-aléatoire léger (XorShift64) pour le sampling reproductible.
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        let mut state = self.state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        self.state = state;
        ((state >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

pub fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for val in values.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for val in values.iter_mut() {
            *val *= inv_sum;
        }
    }
}

pub fn sample_token(logits: &[f32], params: SamplingParams, rng: &mut XorShift64) -> Result<usize, String> {
    if logits.is_empty() {
        return Err("Cannot sample from empty logits".to_string());
    }
    if params.temperature <= 0.0 {
        return Ok(argmax(logits));
    }

    let inv_temp = 1.0 / params.temperature;
    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &val)| (idx, val * inv_temp))
        .collect();

    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if params.top_k > 0 && candidates.len() > params.top_k {
        candidates.truncate(params.top_k);
    }

    let mut probs: Vec<f32> = candidates.iter().map(|&(_, logit)| logit).collect();
    softmax_inplace(&mut probs);

    let cutoff = apply_top_p(&probs, params.top_p);
    let total_mass: f32 = probs[..cutoff].iter().sum();
    let target = rng.next_f32() * total_mass;

    let mut cumulative = 0.0f32;
    for idx in 0..cutoff {
        cumulative += probs[idx];
        if cumulative >= target {
            return Ok(candidates[idx].0);
        }
    }
    Ok(candidates[0].0)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn apply_top_p(sorted_probs: &[f32], top_p: f32) -> usize {
    if top_p <= 0.0 || top_p >= 1.0 {
        return sorted_probs.len();
    }
    let mut cumulative = 0.0f32;
    for (idx, &prob) in sorted_probs.iter().enumerate() {
        cumulative += prob;
        if cumulative >= top_p {
            return idx + 1;
        }
    }
    sorted_probs.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_sampling_picks_highest_logit() {
        let logits = [-1.0f32, 0.2, 3.5, 1.1];
        let mut rng = XorShift64::new(1);
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample_token(&logits, params, &mut rng).unwrap(), 2);
    }

    #[test]
    fn nucleus_sampling_respects_top_k() {
        let logits = [0.0f32, 10.0, 9.5, -5.0];
        let mut rng = XorShift64::new(7);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 2,
            top_p: 0.95,
            seed: 7,
        };
        let tok = sample_token(&logits, params, &mut rng).unwrap();
        assert!(tok == 1 || tok == 2, "unexpected token {tok}");
    }
}
