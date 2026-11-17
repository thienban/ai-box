#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Mutex;

use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::process::CommandChild;
use tauri_plugin_shell::ShellExt;

struct DaemonState {
    child: Mutex<Option<CommandChild>>,
}

fn spawn_sidecar(app: &AppHandle, state: &DaemonState, use_gpu: bool) -> Result<(), String> {
    let mut guard = state.child.lock().map_err(|_| "Daemon lock poisoned".to_string())?;
    if guard.is_some() {
        return Ok(());
    }
    let mut args = vec!["serve", "--host", "127.0.0.1", "--port", "11434"];
    if use_gpu {
        args.push("--gpu");
    }
    let sidecar = app
        .shell()
        .sidecar("binaries/aibox")
        .map_err(|e| e.to_string())?
        .args(args);
    let (_rx, child) = sidecar.spawn().map_err(|e| e.to_string())?;
    *guard = Some(child);
    Ok(())
}

fn kill_sidecar(state: &DaemonState) -> Result<(), String> {
    let mut guard = state.child.lock().map_err(|_| "Daemon lock poisoned".to_string())?;
    if let Some(child) = guard.take() {
        child.kill().map_err(|e| e.to_string())?;
    }
    Ok(())
}

#[tauri::command]
fn start_daemon_gpu(app: AppHandle, state: State<'_, DaemonState>) -> Result<(), String> {
    spawn_sidecar(&app, &state, true)
}

#[tauri::command]
fn start_daemon_cpu(app: AppHandle, state: State<'_, DaemonState>) -> Result<(), String> {
    spawn_sidecar(&app, &state, false)
}

#[tauri::command]
fn stop_daemon(state: State<'_, DaemonState>) -> Result<(), String> {
    kill_sidecar(&state)
}

fn show_main_window(app: &AppHandle) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.show();
        let _ = win.set_focus();
    }
}

fn is_daemon_listening() -> bool {
    let Ok(addr) = "127.0.0.1:11434".parse() else {
        return false;
    };
    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(100)).is_ok()
}

fn setup_tray(app: &tauri::App) -> tauri::Result<()> {
    let show_item = MenuItem::with_id(app, "show", "Ouvrir AI Box", true, None::<&str>)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quitter", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show_item, &quit_item])?;

    TrayIconBuilder::new()
        .menu(&menu)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => show_main_window(app),
            "quit" => {
                let state = app.state::<DaemonState>();
                let _ = kill_sidecar(&state);
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        })
        .build(app)?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(DaemonState {
            child: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            start_daemon_gpu,
            start_daemon_cpu,
            stop_daemon
        ])
        .setup(|app| {
            setup_tray(app)?;
            if !is_daemon_listening() {
                let state = app.state::<DaemonState>();
                let _ = spawn_sidecar(app.handle(), &state, true);
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .run(tauri::generate_context!())
        .expect("Erreur au lancement d'AI Box Desktop");
}
