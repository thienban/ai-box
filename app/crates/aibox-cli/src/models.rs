//! SERV-01 : Gestion du cache `~/.aibox/models/` et téléchargement Hugging Face (`pull`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::engine::ModelConfig;
use crate::json::parse_json;
use crate::safetensors::SafeTensorsFile;

pub const REQUIRED_MODEL_FILES: [&str; 4] = [
    "model.safetensors",
    "quant_config.json",
    "config.json",
    "tokenizer.json",
];

/// Retourne le répertoire racine de cache (`$AIBOX_HOME/models` ou `~/.aibox/models`).
pub fn resolve_models_dir() -> Result<PathBuf, String> {
    if let Ok(custom) = std::env::var("AIBOX_HOME") {
        return Ok(PathBuf::from(custom).join("models"));
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map_err(|_| "Cannot determine home directory (USERPROFILE/HOME unset)".to_string())?;
    Ok(PathBuf::from(home).join(".aibox").join("models"))
}

/// Convertit `org/repo` en nom de dossier sûr (`org--repo`).
pub fn sanitize_repo_id(repo_id: &str) -> Result<String, String> {
    let trimmed = repo_id.trim().trim_matches('/');
    if !trimmed.contains('/') || trimmed.contains("..") {
        return Err(format!(
            "Invalid Hugging Face repo id '{repo_id}': expected '<org>/<repo>'"
        ));
    }
    Ok(trimmed.replace('/', "--"))
}

/// Liste les modèles présents dans le cache local `~/.aibox/models/`.
pub fn list_cached_models() -> Result<Vec<(String, bool)>, String> {
    let dir = resolve_models_dir()?;
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| format!("Failed to read {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("Directory entry error: {e}"))?;
        if entry.path().is_dir() {
            let name = entry.file_name().to_string_lossy().replace("--", "/");
            let complete = verify_model_files(&entry.path()).is_ok();
            entries.push((name, complete));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// Vérifie la présence des fichiers requis (mono-fichier ou multi-shards) dans un dossier modèle.
pub fn verify_model_files(model_dir: &Path) -> Result<(), String> {
    for file in ["quant_config.json", "config.json", "tokenizer.json"] {
        let path = model_dir.join(file);
        if !path.is_file() {
            return Err(format!("Missing required file '{}'", path.display()));
        }
    }
    if model_dir.join("model.safetensors").is_file() {
        return Ok(());
    }
    let has_shard = fs::read_dir(model_dir)
        .map_err(|e| format!("Failed to read '{}': {e}", model_dir.display()))?
        .filter_map(Result::ok)
        .any(|entry| {
            let name = entry.file_name();
            let s = name.to_string_lossy();
            s.starts_with("model-") && s.ends_with(".safetensors") && entry.path().is_file()
        });
    if !has_shard {
        return Err(format!(
            "Missing 'model.safetensors' or 'model-*.safetensors' in '{}'",
            model_dir.display()
        ));
    }
    Ok(())
}

/// Télécharge un dépôt Hugging Face via `curl` (supporte `HF_TOKEN` pour dépôts privés).
pub fn pull_huggingface_model(repo_id: &str) -> Result<PathBuf, String> {
    let safe_name = sanitize_repo_id(repo_id)?;
    let model_dir = resolve_models_dir()?.join(safe_name);
    fs::create_dir_all(&model_dir)
        .map_err(|e| format!("Failed to create '{}': {e}", model_dir.display()))?;

    let token = std::env::var("HF_TOKEN").ok();
    for filename in REQUIRED_MODEL_FILES {
        let url = format!("https://huggingface.co/{repo_id}/resolve/main/{filename}");
        let dest = model_dir.join(filename);
        println!("Téléchargement {filename} -> {}", dest.display());

        let mut cmd = Command::new("curl");
        cmd.arg("-fL")
            .arg("--retry")
            .arg("3")
            .arg("-o")
            .arg(&dest);
        if let Some(ref tok) = token {
            cmd.arg("-H").arg(format!("Authorization: Bearer {tok}"));
        }
        cmd.arg(&url);

        let status = cmd
            .status()
            .map_err(|e| format!("Failed to launch curl for '{url}': {e}"))?;
        if !status.success() {
            return Err(format!(
                "Download failed for '{url}' (exit code: {:?}). Check repo name or HF_TOKEN.",
                status.code()
            ));
        }
    }

    verify_model_files(&model_dir)?;
    Ok(model_dir)
}

/// Inspecte `quant_config.json` et l'en-tête `model.safetensors` d'un modèle en cache ou local.
pub fn inspect_model(target: &str) -> Result<(), String> {
    let as_path = PathBuf::from(target);
    let model_dir = if as_path.is_dir() {
        as_path
    } else {
        resolve_models_dir()?.join(sanitize_repo_id(target)?)
    };

    verify_model_files(&model_dir)?;

    let quant_raw = fs::read_to_string(model_dir.join("quant_config.json"))
        .map_err(|e| format!("Cannot read quant_config.json: {e}"))?;
    let quant_json = parse_json(&quant_raw)?;
    let config_raw = fs::read_to_string(model_dir.join("config.json"))
        .map_err(|e| format!("Cannot read config.json: {e}"))?;
    let model_cfg = ModelConfig::from_json(&parse_json(&config_raw)?)?;
    println!("Modèle        : {}", model_dir.display());
    println!(
        "Architecture  : {} couches, hidden={}, heads={} (kv={}), vocab={}",
        model_cfg.num_hidden_layers,
        model_cfg.hidden_size,
        model_cfg.num_attention_heads,
        model_cfg.num_key_value_heads,
        model_cfg.vocab_size
    );
    if let Some(obj) = quant_json.as_object() {
        let summary: Vec<String> = obj.iter().map(|(k, v)| format!("{k}={v:?}")).collect();
        println!("Quant config  : {}", summary.join(", "));
    }

    let mut st = SafeTensorsFile::open(model_dir.join("model.safetensors"))?;
    let total_bytes: usize = st
        .tensors()
        .values()
        .map(|m| m.data_offsets.1.saturating_sub(m.data_offsets.0))
        .sum();
    println!(
        "SafeTensors   : {} tenseurs ({:.2} Mo)",
        st.tensors().len(),
        (total_bytes as f64) / (1024.0 * 1024.0)
    );

    let first_pair = st
        .tensors()
        .keys()
        .filter(|name| name.ends_with(".weight"))
        .find_map(|w_name| {
            let s_name = w_name.strip_suffix(".weight").map(|p| format!("{p}.scales"))?;
            st.tensors().contains_key(&s_name).then(|| (w_name.clone(), s_name))
        });
    if let Some((w_name, s_name)) = first_pair {
        let layer = st.load_ternary_layer(&w_name, &s_name)?;
        println!(
            "Couche test   : {w_name} [{} x {}] (stride={} octets)",
            layer.n_rows,
            layer.k_cols,
            layer.row_stride_bytes()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_repo_ids() {
        assert_eq!(sanitize_repo_id("org/bonsai-27b").unwrap(), "org--bonsai-27b");
        assert!(sanitize_repo_id("invalid-repo").is_err());
        assert!(sanitize_repo_id("../escape/repo").is_err());
    }

    #[test]
    fn verifies_single_and_sharded_model_layouts() {
        let base = std::env::temp_dir().join(format!("aibox_models_test_{}", std::process::id()));
        let single = base.join("single");
        let sharded = base.join("sharded");
        fs::create_dir_all(&single).unwrap();
        fs::create_dir_all(&sharded).unwrap();
        for dir in [&single, &sharded] {
            for f in ["quant_config.json", "config.json", "tokenizer.json"] {
                fs::write(dir.join(f), "{}").unwrap();
            }
        }
        assert!(verify_model_files(&single).is_err());
        fs::write(single.join("model.safetensors"), "x").unwrap();
        assert!(verify_model_files(&single).is_ok());

        fs::write(sharded.join("model-00001-of-00002.safetensors"), "x").unwrap();
        assert!(verify_model_files(&sharded).is_ok());
        let _ = fs::remove_dir_all(&base);
    }
}
