use anyhow::{Context, Result};
use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub(crate) enum RuntimeCommand {
    /// Show locally served runtime status on a running mesh-llm instance.
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Load a local-only model into a running mesh-llm instance.
    Load {
        /// Model name/path/url to load
        name: String,
        /// API port of the running mesh-llm instance (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Unload a local runtime-loaded model from a running mesh-llm instance.
    #[command(alias = "drop")]
    Unload {
        /// Model name to unload
        name: String,
        /// API port of the running mesh-llm instance (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
}

pub(crate) async fn run_drop(model_name: &str, port: u16) -> Result<()> {
    run_control_request("/mesh/drop", model_name, port, "Dropped").await
}

pub(crate) async fn run_load(model_name: &str, port: u16) -> Result<()> {
    run_control_request("/mesh/load", model_name, port, "Loaded").await
}

pub(crate) async fn run_models(port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime");
    let body = client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let models = body["models"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid runtime status payload"))?;

    println!("⚙️  Runtime");
    println!();

    let primary_model = body["primary_model"].as_str().unwrap_or("none");
    println!("🧠 Primary: {primary_model}");

    if models.is_empty() {
        println!("📦 Models served locally: 0");
        println!();
        println!("No local models are currently being served.");
        return Ok(());
    }

    println!("📦 Models served locally: {}", models.len());
    println!();

    println!(
        "{:<42} {:<8} {:<8} {:<10} {:<6} {:<8}",
        "Model", "Role", "Backend", "State", "Port", "Source"
    );
    for model in models {
        let name = model["name"].as_str().unwrap_or("unknown");
        let kind = display_runtime_role(model["kind"].as_str().unwrap_or("unknown"));
        let backend = display_backend_label(model["backend"].as_str().unwrap_or("unknown"));
        let status = display_runtime_state(model["status"].as_str().unwrap_or("unknown"));
        let port = model["port"]
            .as_u64()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let source = if model["startup_managed"].as_bool().unwrap_or(false) {
            "Startup"
        } else {
            "Runtime"
        };
        println!(
            "{:<42} {:<8} {:<8} {:<10} {:<6} {:<8}",
            name, kind, backend, status, port, source
        );
    }

    Ok(())
}

pub(crate) async fn run_ps(port: u16) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    let url = format!("http://127.0.0.1:{port}/api/runtime/processes");
    let body = client
        .get(&url)
        .send()
        .await
        .with_context(|| {
            format!("Can't connect to mesh-llm console on port {port}. Is it running?")
        })?
        .error_for_status()?
        .json::<serde_json::Value>()
        .await?;

    let processes = body["processes"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Invalid runtime process payload"))?;

    println!("⚙️  Processes");
    println!();

    if processes.is_empty() {
        println!("📦 Local processes: 0");
        println!();
        println!("No local inference processes are currently running.");
        return Ok(());
    }

    println!("📦 Local processes: {}", processes.len());
    println!();

    println!(
        "{:<42} {:<8} {:<8} {:<8} {:<6} {:<8}",
        "Model", "Role", "Backend", "Pid", "Port", "Source"
    );
    for process in processes {
        let name = process["name"].as_str().unwrap_or("unknown");
        let kind = display_runtime_role(process["kind"].as_str().unwrap_or("unknown"));
        let backend = display_backend_label(process["backend"].as_str().unwrap_or("unknown"));
        let port = process["port"]
            .as_u64()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let pid = process["pid"]
            .as_u64()
            .map(|p| p.to_string())
            .unwrap_or_else(|| "-".into());
        let source = if process["startup_managed"].as_bool().unwrap_or(false) {
            "Startup"
        } else {
            "Runtime"
        };
        println!(
            "{:<42} {:<8} {:<8} {:<8} {:<6} {:<8}",
            name, kind, backend, pid, port, source
        );
    }

    Ok(())
}

async fn run_control_request(path: &str, model_name: &str, port: u16, verb: &str) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let body = serde_json::json!({ "model": model_name }).to_string();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = tokio::net::TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .with_context(|| format!("Can't connect to mesh-llm on port {port}. Is it running?"))?;
    stream.write_all(request.as_bytes()).await?;

    let mut response = vec![0u8; 4096];
    let n = stream.read(&mut response).await?;
    let resp = String::from_utf8_lossy(&response[..n]);

    if resp.contains("200 OK") || resp.contains("201 Created") {
        let action = if verb == "Loaded" {
            "Loaded"
        } else {
            "Unloaded"
        };
        eprintln!("✅ {action} runtime model");
        eprintln!();
        eprintln!("Model: {model_name}");
        eprintln!("Scope: Local node");
    } else {
        let action = if verb == "Loaded" { "load" } else { "unload" };
        eprintln!("❌ Failed to {action} runtime model");
        eprintln!();
        eprintln!("Model: {model_name}");
        eprintln!("Reason: {}", resp.lines().last().unwrap_or("unknown error"));
    }

    Ok(())
}

fn display_runtime_role(value: &str) -> &'static str {
    match value {
        "primary" => "Primary",
        "runtime" => "Runtime",
        _ => "Unknown",
    }
}

fn display_runtime_state(value: &str) -> &'static str {
    match value {
        "ready" => "Ready",
        "starting" => "Starting",
        "stopped" => "Stopped",
        _ => "Unknown",
    }
}

fn display_backend_label(value: &str) -> &'static str {
    match value {
        "llama" => "Llama",
        "mlx" => "MLX",
        "vllm" => "vLLM",
        _ => "Unknown",
    }
}
