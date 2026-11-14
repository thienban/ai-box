# `aibox-mistral` (`KERN-07`)

Adaptateur reliant [aibox-kernels](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-kernels/Cargo.toml) (`TmacCpuBackend` + `TernaryTensor`) au moteur **`mistral.rs`** (voir [spec_kern07_mistral_rs.md](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/docs/dev/spec_kern07_mistral_rs.md)).

## Activation complète avec `mistral.rs` (en ligne)
1. Décommentez `candle-core` et `mistralrs-quant` dans `Cargo.toml`.
2. Implémentez `mistralrs_quant::QuantMethod` pour `AiboxTernaryLinear` en délégant `forward` vers `self.forward_f32(...)`.
3. Testez :
   ```powershell
   cargo test -p aibox-mistral
   ```
