# FEAT-019: Extraction `aibox-core`, Trait `LlmEngine` & Support Gemma 4 dans `aibox serve`

- **Statut**: TODO <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
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
- [ ] **Extraction crate `crates/aibox-core`** : regroupe `engine/`, `models.rs`, `safetensors.rs`, `tokenizer.rs`, `sampling.rs`, `json.rs`.
- [ ] **Trait unifié `LlmEngine`** (SOLID **O/D**) : implémenté par `TransformerModel` (`qwen3` / `qwen35`) et `Gemma4Model` (`gemma4`), avec génération streaming token par token et annulation.
- [ ] **Cache VRAM persistant en daemon (`aibox serve`)** : conserver `Gemma4GpuScratch` chaud entre requêtes API successives pour éviter la re-chauffe à froid (`disk_misses`), avec paramètre `keep_alive` (déchargement automatique après inactivité, façon Ollama).
- [ ] **Sélection dynamique par `"model"`** dans `/v1/chat/completions`, `/api/chat` et `/api/generate` + détection automatique mono/multi-shards dans `/v1/models` et `/api/tags`.
- [ ] **CLI `aibox run` cliente** : si `aibox serve` tourne sur `127.0.0.1:11434`, réutiliser le daemon chaud au lieu de recharger le modèle.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 08:54 | - | TODO | Antigravity | Création du ticket suite à l'audit d'architecture hexagonale et de performance daemon. |
