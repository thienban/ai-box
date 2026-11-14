//! Point d'entrée CLI du binaire `aibox` (SERV-01 / SERV-02).

mod engine;
mod json;
mod models;
mod safetensors;
mod sampling;

use std::env;
use std::process::ExitCode;

use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows};
use aibox_kernels::{TernaryHadamardBackend, TmacCpuBackend, GROUP_SIZE};
use engine::{build_synthetic_tensor, ModelConfig, TernaryLayerWeights, TransformerModel};
use sampling::SamplingParams;

fn print_usage() {
    println!(
        "aibox — Moteur d'inférence local ternaire (SafeTensors + T-MAC)\n\
         \n\
         COMMANDES:\n\
         \x20 aibox pull <org/repo-hf>    Télécharge un modèle depuis Hugging Face (~/.aibox/models)\n\
         \x20 aibox list                  Liste les modèles en cache local\n\
         \x20 aibox inspect <model|dir>   Vérifie config.json, quant_config.json et SafeTensors\n\
         \x20 aibox selftest              Vérifie le backend CPU (AVX2/Scalar) bout en bout\n\
         \x20 aibox bench-forward         Benchmark SERV-03 (Prefill/Decode Transformer synthétique)"
    );
}

fn run_selftest() -> Result<(), String> {
    let backend = TmacCpuBackend::new();
    println!(
        "Backend CPU actif : {}",
        if backend.uses_avx2() {
            "AVX2 (pshufb)"
        } else {
            "Scalar (fallback)"
        }
    );

    let m = 1;
    let n = 64;
    let k = 256;
    let act = vec![0.25f32; m * k];
    let weights: Vec<i8> = (0..(n * k))
        .map(|i| match i % 3 {
            0 => 0,
            1 => 1,
            _ => -1,
        })
        .collect();
    let packed = pack_ternary_2bit_rows(&weights, n, k).map_err(|e| e.to_string())?;
    let scales = vec![f32_to_f16(0.5); n * (k / GROUP_SIZE)];

    let out = backend
        .matmul_hadamard_ternary(&act, &packed, &scales, m, n, k)
        .map_err(|e| e.to_string())?;
    println!("Selftest OK : {} sorties calculées (y[0] = {:.4})", out.len(), out[0]);
    Ok(())
}

fn run_bench_forward() -> Result<(), String> {
    let config = ModelConfig {
        hidden_size: 1024,
        intermediate_size: 2816,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        vocab_size: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
    };
    let kv_dim = config.num_key_value_heads * config.head_dim();
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        layers.push(TernaryLayerWeights {
            attn_norm: vec![1.0; config.hidden_size],
            attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1)?,
            attn_k: build_synthetic_tensor(kv_dim, config.hidden_size, layer_idx + 2)?,
            attn_v: build_synthetic_tensor(kv_dim, config.hidden_size, layer_idx + 3)?,
            attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4)?,
            ffn_norm: vec![1.0; config.hidden_size],
            ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 5)?,
            ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 6)?,
            ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, layer_idx + 7)?,
        });
    }
    let model = TransformerModel {
        token_embeddings: vec![0.02; config.vocab_size * config.hidden_size],
        layers,
        final_norm: vec![1.0; config.hidden_size],
        lm_head: build_synthetic_tensor(config.vocab_size, config.hidden_size, 99)?,
        config,
    };

    let backend = TmacCpuBackend::new();
    let prompt = [1, 4, 8, 15, 16, 23, 42, 64];
    let max_new = 16;
    let stats = model.generate_tokens(&prompt, max_new, None, SamplingParams::default(), &backend)?;

    let prefill_tok_s = (prompt.len() as f64) / (stats.prefill_ms / 1000.0);
    let decode_tok_s = (stats.tokens.len() as f64) / (stats.decode_ms / 1000.0);
    println!(
        "SERV-03 Forward (4 couches, hidden=1024, AVX2={}) :",
        backend.uses_avx2()
    );
    println!("  Prefill ({} tok) : {:.2} ms ({:.2} tok/s)", prompt.len(), stats.prefill_ms, prefill_tok_s);
    println!("  Decode  ({} tok) : {:.2} ms ({:.2} tok/s)", stats.tokens.len(), stats.decode_ms, decode_tok_s);
    Ok(())
}

fn dispatch(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        None | Some("-h") | Some("--help") | Some("help") => {
            print_usage();
            Ok(())
        }
        Some("pull") => {
            let repo = args
                .get(1)
                .ok_or_else(|| "Usage: aibox pull <org/repo-hf>".to_string())?;
            let path = models::pull_huggingface_model(repo)?;
            println!("Modèle prêt dans {}", path.display());
            Ok(())
        }
        Some("list") => {
            let list = models::list_cached_models()?;
            if list.is_empty() {
                println!("Aucun modèle dans le cache ~/.aibox/models.");
            } else {
                for (name, complete) in list {
                    let status = if complete { "prêt" } else { "incomplet" };
                    println!("- {name} [{status}]");
                }
            }
            Ok(())
        }
        Some("inspect") => {
            let target = args
                .get(1)
                .ok_or_else(|| "Usage: aibox inspect <org/repo-hf | path>".to_string())?;
            models::inspect_model(target)
        }
        Some("selftest") => run_selftest(),
        Some("bench-forward") => run_bench_forward(),
        Some(other) => Err(format!("Commande inconnue '{other}'. Lancez `aibox --help`.")),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if let Err(err) = dispatch(&args) {
        eprintln!("Erreur : {err}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
