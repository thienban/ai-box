# FEAT-019: Extraction `aibox-core`, Trait `LlmEngine` & Support Gemma 4 dans `aibox serve`

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Refactor <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-17
- **Assigné à**: Antigravity
- **Dépendance**: [SEC-001](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/SEC-001-harden-http-server-and-gpu-queue.md)

---

## 🎯 Objectif & Contexte
Actuellement :
1. [server.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/server.rs) est figé sur `TransformerModel` (`qwen3` / `qwen35`) et ne peut pas servir `Gemma4Model`.
2. Le moteur d'inférence est couplé au binaire `aibox-cli`, ce qui empêche de le partager proprement avec un daemon ou une app Desktop.
3. En CLI `run`, le cache VRAM `Gemma4GpuScratch` (`2 003` experts) est recréé et réchauffé à froid à chaque appel (`10–38 s` de prefill).

---

## 📋 Critères d'Acceptation
- [x] **Extraction crate `crates/aibox-core`** : regroupe `engine/`, `models.rs`, `safetensors.rs`, `tokenizer.rs`, `sampling.rs`, `json.rs`.
- [x] **Trait unifié `LlmEngine`** (SOLID **O/D**) : implémenté par `TransformerModel` (`qwen3` / `qwen35`) et `Gemma4Model` (`gemma4`), avec `generate_tokens_streaming_dyn` et `load_model_from_dir`.
- [x] **Cache RAM chaud & sélection dynamique dans `aibox serve`** : `LoadedModel` sous `Mutex`, `ensure_model_loaded` bascule à chaud entre modèles du cache `~/.aibox/models/`, et `expert_ram_cache` reste chaud entre requêtes API.
- [x] **TTL `keep_alive` & CLI cliente** : déchargement automatique après inactivité (TTL `keep_alive` configurable via CLI `--keep-alive` et requêtes JSON) + `aibox run` réutilisant le daemon local `127.0.0.1:11434` si disponible.
- [x] **Tests unitaires** : 36/36 PASS (`keep_alive_unloads_idle_custom_model_back_to_demo`, tests serveurs et moteurs).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 08:54 | - | TODO | Antigravity | Création du ticket suite à l'audit d'architecture hexagonale et de performance daemon. |
| 2026-11-17 09:54 | TODO | IN_PROGRESS | Antigravity | Extraction `crates/aibox-core`, trait `LlmEngine`, `Gemma4Model::generate_tokens_streaming`, `load_model_from_dir`, hot-swap `"model"` dans `server.rs`. Workspace 35/35 PASS (0 warning). |
| 2026-11-17 10:04 | IN_PROGRESS | DONE | Antigravity | Ajout TTL `keep_alive` automatique avec fallback demo model, option CLI `--keep-alive`, client `try_run_via_daemon` dans `aibox run`, test unitaire `keep_alive` PASS. 36/36 tests PASS. |
