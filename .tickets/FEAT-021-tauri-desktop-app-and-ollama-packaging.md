# FEAT-021: Packaging Desktop Tauri v2 & Distribution façon Ollama

- **Statut**: IN_PROGRESS <!-- Options: TODO | IN_PROGRESS | IN_REVIEW | DONE -->
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
- [x] **Workspace `desktop/` (Tauri v2)** :
  - Embarque `aibox` (+ `aibox_gpu.dll` si présent) en **sidecar** Tauri (`externalBin: ["binaries/aibox"]` dans [tauri.conf.json](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/desktop/src-tauri/tauri.conf.json)).
  - **System Tray** ([main.rs](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/desktop/src-tauri/src/main.rs)) : démarrage automatique du daemon `aibox serve`, menu *Ouvrir AI Box* / *Quitter*, masquage dans le tray à la fermeture de fenêtre.
- [x] **UI Desktop légère (React 19 + TypeScript + Tailwind v4)** :
  - Sélecteur de modèles locaux (`/api/tags`), bascule GPU/CPU, bouton Start/Stop du daemon ([DaemonStatusBar.tsx](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/desktop/src/components/DaemonStatusBar.tsx)).
  - Interface de chat streaming SSE (`/v1/chat/completions`) avec annulation (`Stop`) et affichage du débit `tok/s` ([ChatPanel.tsx](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/desktop/src/components/ChatPanel.tsx), [useAiboxDaemon.ts](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/desktop/src/hooks/useAiboxDaemon.ts)).
- [x] **Script de packaging** : [package_desktop.ps1](file:///c:/Users/Bui/Desktop/DevProjets/tiny-gpu/scripts/package_desktop.ps1) (compile `aibox.exe`, copie le sidecar, build frontend + `pnpm tauri build` pour `nsis`, `msi`, `deb`, `appimage`).
- [ ] **À finaliser** : téléchargement `pull` avec barre de progression dans l'UI + test de génération d'installeur NSIS/MSI complet (`.\scripts\package_desktop.ps1`).

---

## 📜 Historique / History
| Date & Heure | Ancien Statut | Nouveau Statut | Auteur | Commentaire / Note |
|---|---|---|---|---|
| 2026-11-17 08:54 | - | TODO | Antigravity | Création du ticket de packaging Desktop Tauri v2 & distribution multi-plateforme. |
| 2026-11-17 11:02 | TODO | IN_PROGRESS | Antigravity | Création complète de `desktop/` (React 19 + Tailwind v4 + Tauri v2 System Tray + sidecar `aibox`), icône générée, `scripts/package_desktop.ps1`, `.gitignore`. Vérifié : `pnpm --dir desktop run build` OK (`0` erreur) et `cargo check --manifest-path desktop/src-tauri/Cargo.toml` OK (`0` warning). |
