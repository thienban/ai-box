# SEC-001: Durcissement Sécurité & Résilience du Serveur HTTP (`server.rs`)

- **Statut**: DONE <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: HIGH <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Security <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance | Security -->
- **Date de création**: 2026-11-17
- **Assigné à**: Antigravity
- **Dépendance**: -

---

## 🎯 Objectif & Contexte
L'audit de [server.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/server.rs) et [models.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/app/crates/aibox-cli/src/models.rs) a mis en évidence plusieurs vecteurs de déni de service (DoS) et d'exposition locale avant toute distribution en daemon/desktop :
1. **`Content-Length` non borné** (`vec![0u8; content_length]` ligne 222) $\rightarrow$ OOM instantané sur requête malformée.
2. **`thread::spawn` illimité** (ligne 105) et **absence de timeouts** TCP $\rightarrow$ Slowloris et risque d'OOM VRAM 6 Go si plusieurs requêtes sollicitent le GPU en parallèle.
3. **CORS ouvert (`*`)** (lignes 510, 581, 673) $\rightarrow$ appels cross-origin possibles depuis n'importe quel onglet navigateur vers `127.0.0.1:11434`.
4. **Absence de vérification d'intégrité** sur `pull_huggingface_model` et non-support des modèles multi-shards dans `verify_model_files`.

---

## 📋 Critères d'Acceptation
- [x] **Plafond HTTP `MAX_BODY_BYTES`** (`4 MiB`) vérifié avant allocation, retournant `413 Payload Too Large` en cas de dépassement.
- [x] **Timeouts TCP** (`set_read_timeout` / `set_write_timeout` à `30s`) + **connexions concurrentes bornées** (`MAX_CONCURRENT_CONNECTIONS = 16`, `503 Service Unavailable`).
- [x] **File GPU sérialisée (`inference_lock: Mutex<()>`)** : 1 seule inférence GPU active à la fois + arrêt immédiat sur déconnexion client SSE/NDJSON.
- [x] **CORS restreint** aux origines locales (`localhost`, `127.0.0.1`, `tauri://localhost`) + support optionnel d'un token Bearer `AIBOX_API_KEY` (`401 Unauthorized`).
- [x] **Validation `models.rs`** : support des modèles multi-shards (`model-*.safetensors`) dans `verify_model_files`.
- [x] **Tests unitaires** : `rejects_oversized_content_length_and_enforces_cors_and_auth` et `verifies_single_and_sharded_model_layouts` PASS.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 08:54 | - | TODO | Antigravity | Création du ticket suite à l'audit sécurité de `server.rs` et `models.rs`. |
| 2026-11-17 09:18 | TODO | DONE | Antigravity | Implémentation du durcissement HTTP/CORS/Bearer/GPU lock dans `server.rs` et support multi-shards dans `models.rs`. 35/35 tests PASS, 0 warning clippy sur `server.rs`/`models.rs`. |
