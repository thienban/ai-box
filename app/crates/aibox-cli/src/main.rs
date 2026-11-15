//! Point d'entrée CLI du binaire `aibox` (SERV-01 / SERV-02).

mod engine;
mod json;
mod models;
mod safetensors;
mod sampling;
mod server;
mod tokenizer;

use std::env;
use std::process::ExitCode;

use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows};
use aibox_kernels::{select_backend, TernaryHadamardBackend, TmacCpuBackend, GROUP_SIZE};
use engine::{
    build_synthetic_tensor, EmbeddingTable, LmHead, ModelConfig, TernaryLayerWeights,
    TransformerModel,
};
use sampling::SamplingParams;

fn print_usage() {
    println!(
        "aibox — Moteur d'inférence local ternaire (SafeTensors + T-MAC)\n\
         \n\
         COMMANDES:\n\
         \x20 aibox pull <org/repo-hf>    Télécharge un modèle depuis Hugging Face (~/.aibox/models)\n\
         \x20 aibox list                  Liste les modèles en cache local\n\
         \x20 aibox inspect <model|dir>   Vérifie config.json, quant_config.json et SafeTensors\n\
         \x20 aibox run <model|dir> <txt> [--max-tokens N] [--gpu] Inférence réelle SafeTensors\n\
         \x20 aibox selftest              Vérifie le backend CPU (AVX2/Scalar) bout en bout\n\
         \x20 aibox bench-forward         Benchmark SERV-03 (Prefill/Decode Transformer synthétique)\n\
         \x20 aibox serve [--model dir] [--gpu] Démarre le serveur HTTP OpenAI + Ollama (SERV-04)"
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
        head_dim: 128,
        vocab_size: 1024,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        use_hadamard: true,
        ..ModelConfig::default()
    };
    let kv_dim = config.num_key_value_heads * config.head_dim();
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        layers.push(TernaryLayerWeights {
            attn_norm: vec![1.0; config.hidden_size],
            block: engine::LayerBlock::Attention(engine::AttentionBlockWeights {
                attn_q: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 1)?,
                attn_k: build_synthetic_tensor(kv_dim, config.hidden_size, layer_idx + 2)?,
                attn_v: build_synthetic_tensor(kv_dim, config.hidden_size, layer_idx + 3)?,
                attn_o: build_synthetic_tensor(config.hidden_size, config.hidden_size, layer_idx + 4)?,
                attn_q_norm: None,
                attn_k_norm: None,
                has_q_gate: false,
            }),
            ffn_norm: vec![1.0; config.hidden_size],
            ffn_gate: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 5)?,
            ffn_up: build_synthetic_tensor(config.intermediate_size, config.hidden_size, layer_idx + 6)?,
            ffn_down: build_synthetic_tensor(config.hidden_size, config.intermediate_size, layer_idx + 7)?,
        });
    }
    let model = TransformerModel {
        token_embeddings: EmbeddingTable::F32(vec![0.02; config.vocab_size * config.hidden_size]),
        layers,
        final_norm: vec![1.0; config.hidden_size],
        lm_head: LmHead::Ternary(build_synthetic_tensor(config.vocab_size, config.hidden_size, 99)?),
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

fn resolve_model_path(target: &str) -> Result<std::path::PathBuf, String> {
    let direct = std::path::PathBuf::from(target);
    if direct.is_dir() {
        return Ok(direct);
    }
    let cached = models::resolve_models_dir()?.join(models::sanitize_repo_id(target)?);
    if cached.is_dir() {
        return Ok(cached);
    }
    Err(format!("Modèle introuvable : '{target}'"))
}

fn run_inference_cli(args: &[String]) -> Result<(), String> {
    let target = args
        .first()
        .ok_or_else(|| "Usage: aibox run <model|dir> <prompt> [--max-tokens N]".to_string())?;
    let prompt = args
        .get(1)
        .ok_or_else(|| "Usage: aibox run <model|dir> <prompt> [--max-tokens N]".to_string())?;
    let mut max_tokens = 32usize;
    let mut use_gpu = false;
    let mut temperature = 0.0f32;
    let mut idx = 2usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--max-tokens" => {
                idx += 1;
                let val = args
                    .get(idx)
                    .ok_or_else(|| "Valeur manquante pour --max-tokens".to_string())?;
                max_tokens = val
                    .parse::<usize>()
                    .map_err(|_| format!("max-tokens invalide: '{val}'"))?;
            }
            "--temp" => {
                idx += 1;
                let val = args
                    .get(idx)
                    .ok_or_else(|| "Valeur manquante pour --temp".to_string())?;
                temperature = val
                    .parse::<f32>()
                    .map_err(|_| format!("temp invalide: '{val}'"))?;
            }
            "--gpu" => {
                use_gpu = true;
            }
            other => return Err(format!("Option inconnue pour run : '{other}'")),
        }
        idx += 1;
    }

    let model_dir = resolve_model_path(target)?;
    println!("Chargement du modèle depuis {}...", model_dir.display());
    let model = TransformerModel::load_from_dir(&model_dir)?;
    let tok = tokenizer::SimpleTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
    let prompt_ids = tok.encode_chat(prompt);

    let backend = select_backend(use_gpu);
    let params = SamplingParams {
        temperature,
        top_k: if temperature > 0.0 { 20 } else { 1 },
        top_p: if temperature > 0.0 { 0.95 } else { 1.0 },
        seed: 42,
    };
    let stats = model.generate_tokens(&prompt_ids, max_tokens, tok.eos_token_id(), params, &*backend)?;
    let text = tok.decode(&stats.tokens);
    let decode_tok_s = (stats.tokens.len() as f64) / (stats.decode_ms / 1000.0).max(1e-6);

    println!("\n{text}");
    println!(
        "\n[Prefill {} tok: {:.1} ms | Decode {} tok: {:.1} ms ({:.2} tok/s)]",
        prompt_ids.len(),
        stats.prefill_ms,
        stats.tokens.len(),
        stats.decode_ms,
        decode_tok_s
    );
    Ok(())
}

fn run_serve(args: &[String]) -> Result<(), String> {
    let mut config = server::ServerConfig::default();
    let mut idx = 0usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--host" => {
                idx += 1;
                let host = args.get(idx).ok_or_else(|| "Valeur manquante pour --host".to_string())?;
                config.host = host.clone();
            }
            "--port" => {
                idx += 1;
                let port_str = args.get(idx).ok_or_else(|| "Valeur manquante pour --port".to_string())?;
                config.port = port_str
                    .parse::<u16>()
                    .map_err(|_| format!("Port invalide '{port_str}'"))?;
            }
            "--model" => {
                idx += 1;
                let model = args.get(idx).ok_or_else(|| "Valeur manquante pour --model".to_string())?;
                config.model_dir = Some(resolve_model_path(model)?);
            }
            "--gpu" => {
                config.use_gpu = true;
            }
            other => return Err(format!("Option inconnue pour serve : '{other}'")),
        }
        idx += 1;
    }
    server::run_server(config)
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
        Some("run") => run_inference_cli(&args[1..]),
        Some("selftest") => run_selftest(),
        Some("bench-forward") => run_bench_forward(),
        Some("serve") => run_serve(&args[1..]),
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
