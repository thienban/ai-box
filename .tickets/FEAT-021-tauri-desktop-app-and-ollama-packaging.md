# FEAT-021: Packaging Desktop Tauri v2 & Distribution façon Ollama

- **Statut**: TODO <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
- **Priorité**: MEDIUM <!-- Options: LOW | MEDIUM | HIGH | CRITICAL -->
- **Type**: Feature <!-- Options: Feature | Bug | Refactor | Docs | Chore | Performance -->
- **Date de création**: 2026-11-17
- **Assigné à**: Antigravity
- **Dépendance**: [FEAT-019](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-019-aibox-core-unified-engine-and-gemma4-serve.md), [FEAT-020](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/.tickets/FEAT-020-windows-native-cuda-dll.md)

---

## 🎯 Objectif & Contexte
Offrir une expérience utilisateur clé en main comparable à **Ollama Desktop** : installation en 1 clic (`.exe`/`.msi` Windows, `.deb`/`.AppImage` Linux), icône dans la barre des tâches (*System Tray*), gestion automatique du daemon `aibox serve`, et interface légère de chat / gestion des modèles / monitoring VRAM.

---

## 📋 Critères d'Acceptation
- [ ] **Workspace `desktop/` (Tauri v2)** :
  - Embarque `aibox` + `aibox_gpu.dll` (ou `.so`) en **sidecar** Tauri.
  - **System Tray** : démarrage/arrêt du daemon `aibox serve`, statut GPU/VRAM, ouverture du chat, lancement au démarrage OS optionnel.
- [ ] **UI Desktop légère** :
  - Liste des modèles locaux (`~/.aibox/models/`) + téléchargement (`pull`) avec barre de progression.
  - Interface de chat streaming connectée à `http://127.0.0.1:11434/v1/chat/completions`.
  - Jauge VRAM / débit `tok/s` en temps réel.
- [ ] **Installeurs & Distribution** :
  - Windows : installeur NSIS/MSI (`cargo tauri build`) ajoutant `aibox.exe` au `PATH` utilisateur.
  - Linux : `.AppImage` / `.deb` + unité `systemd --user` optionnelle.

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 08:54 | - | TODO | Antigravity | Création du ticket de packaging Desktop Tauri v2 & distribution multi-plateforme. |
