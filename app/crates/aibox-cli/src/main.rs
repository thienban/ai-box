//! Point d'entrée CLI du binaire `aibox` (SERV-01 / SERV-02).

mod json;
mod models;
mod safetensors;

use std::env;
use std::process::ExitCode;

use aibox_kernels::tensor::{f32_to_f16, pack_ternary_2bit_rows};
use aibox_kernels::{TernaryHadamardBackend, TmacCpuBackend, GROUP_SIZE};

fn print_usage() {
    println!(
        "aibox — Moteur d'inférence local ternaire (SafeTensors + T-MAC)\n\
         \n\
         COMMANDES:\n\
         \x20 aibox pull <org/repo-hf>    Télécharge un modèle depuis Hugging Face (~/.aibox/models)\n\
         \x20 aibox list                  Liste les modèles en cache local\n\
         \x20 aibox inspect <model|dir>   Vérifie quant_config.json et l'en-tête SafeTensors\n\
         \x20 aibox selftest              Vérifie le backend CPU (AVX2/Scalar) bout en bout"
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
