use tauri_plugin_shell::ShellExt;
use tauri_plugin_shell::process::CommandEvent;
use tauri::Manager;
use std::sync::Mutex;

#[allow(dead_code)]
struct SidecarChild(Mutex<Option<tauri_plugin_shell::process::CommandChild>>);

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let sidecar_command = app.shell().sidecar("api").unwrap()
                .env("ELATO_DB_PATH", "/Users/akashdeepdeb/Desktop/neutts-openai-api/elato.db"); // Point to dev DB for now

            let (mut rx, child) = sidecar_command
                .spawn()
                .expect("Failed to spawn sidecar");

            // Keep the child alive for the duration of the app.
            // If dropped at end of setup, the sidecar may terminate immediately.
            app.manage(SidecarChild(Mutex::new(Some(child))));

            tauri::async_runtime::spawn(async move {
                // read events such as stdout
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            println!("[SIDECAR:stdout] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Stderr(line) => {
                            eprintln!("[SIDECAR:stderr] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Error(err) => {
                            eprintln!("[SIDECAR:error] {}", err);
                        }
                        CommandEvent::Terminated(payload) => {
                            eprintln!("[SIDECAR:terminated] {:?}", payload);
                        }
                        _ => {}
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
