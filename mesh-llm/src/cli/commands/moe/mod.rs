mod analyze;
mod formatters;
mod formatters_console;
mod formatters_json;
mod hf_jobs;
mod model_card;
mod publish;
mod upload;

use anyhow::{bail, Context, Result};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::cli::moe::{MoeAnalyzeCommand, MoeCommand};
use crate::cli::terminal_progress::start_spinner;
use crate::cli::Cli;
use crate::models;
use crate::system::moe_planner::{self, MoePlanArgs};

use formatters::moe_plan_formatter;

const MICRO_PROMPTS: &[&str] = &[
    "Write a concise explanation of how a rainbow forms.",
    "Summarize the causes and effects of inflation in a paragraph.",
    "Explain why distributed systems are hard to debug.",
    "Give three practical tips for writing reliable shell scripts.",
    "Describe the water cycle for a middle school student.",
    "Compare TCP and QUIC in two short paragraphs.",
    "Explain the difference between RAM and disk storage.",
    "Write a short answer on why model evaluation matters.",
];

const SHARE_UPLOAD_BATCH_MAX_FILES: usize = 8;
const SHARE_UPLOAD_BATCH_MAX_BYTES: u64 = 1_500_000_000;
const SHARE_UPLOAD_STALL_TIMEOUT: Duration = Duration::from_secs(180);
const SHARE_UPLOAD_POLL_INTERVAL: Duration = Duration::from_secs(5);
const SHARE_UPLOAD_MAX_RETRIES: usize = 3;
const SHARE_REPO_READY_TIMEOUT: Duration = Duration::from_secs(120);
const SHARE_REPO_READY_POLL_INTERVAL: Duration = Duration::from_millis(750);
const FULL_ANALYZE_CONTEXT_SIZE: u32 = 4096;
const FULL_ANALYZE_GPU_LAYERS: u32 = 0;
const FULL_ANALYZE_TOKEN_COUNT: u32 = 32;

struct TempRootGuard(PathBuf);

impl Drop for TempRootGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(crate) async fn dispatch_moe_command(command: &MoeCommand, cli: &Cli) -> Result<()> {
    match command {
        MoeCommand::Plan {
            model,
            json,
            max_vram,
            nodes,
            catalog_repo,
        } => {
            run_plan(
                model,
                *json,
                max_vram.or(cli.max_vram),
                *nodes,
                catalog_repo,
            )
            .await
        }
        MoeCommand::Analyze { command } => match command {
            MoeAnalyzeCommand::Full {
                model,
                context_size,
                n_gpu_layers,
            } => analyze::run_analyze_full(model, *context_size, *n_gpu_layers).await,
            MoeAnalyzeCommand::Micro {
                model,
                prompt_count,
                token_count,
                context_size,
                n_gpu_layers,
            } => {
                analyze::run_analyze_micro(
                    model,
                    *prompt_count,
                    *token_count,
                    *context_size,
                    *n_gpu_layers,
                )
                .await
            }
        },
        MoeCommand::Publish {
            model,
            catalog_repo,
            namespace,
            hf_job,
        } => publish::run_publish(model, catalog_repo, namespace.as_deref(), hf_job).await,
    }
}

async fn run_plan(
    model: &str,
    json_output: bool,
    max_vram: Option<f64>,
    nodes: Option<usize>,
    catalog_repo: &str,
) -> Result<()> {
    if !json_output {
        eprintln!("📍 Resolving MoE model: {model}");
        eprintln!("📦 Checking local MoE package cache...");
        eprintln!("☁️ Checking {catalog_repo} for published MoE packages...");
    }
    let report = moe_planner::plan_moe(MoePlanArgs {
        model: model.to_string(),
        max_vram_gb: max_vram,
        nodes,
        catalog_repo: catalog_repo.to_string(),
        progress: !json_output,
    })
    .await?;
    moe_plan_formatter(json_output).render(&report)
}

fn resolve_analyze_binary() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("Failed to determine own binary path")?;
    let bin_dir = exe
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Current executable has no parent directory"))?;
    let candidates = [
        bin_dir.join("llama-moe-analyze"),
        bin_dir.join("../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../llama.cpp/build/bin/llama-moe-analyze"),
        bin_dir.join("../../../llama.cpp/build/bin/llama-moe-analyze"),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate.canonicalize().unwrap_or(candidate));
        }
    }
    bail!(
        "llama-moe-analyze not found next to {} or nearby llama.cpp/build/bin directories",
        bin_dir.display()
    )
}

fn log_path_for(model_path: &Path, analyzer_id: &str) -> PathBuf {
    let stem = model_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    models::mesh_llm_cache_dir()
        .join("moe")
        .join("logs")
        .join(format!("{stem}.{analyzer_id}.log"))
}

fn run_analyzer_command(command: &[String], log_path: &Path, label: &str) -> Result<()> {
    let mut spinner = start_spinner(&format!("Running {label}"));
    let output = Command::new(&command[0])
        .args(&command[1..])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| format!("Run {}", command[0]))?;
    spinner.finish();
    fs::write(
        log_path,
        format!(
            "$ {}\n\n[stdout]\n{}\n[stderr]\n{}",
            shell_join(command),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )?;
    if !output.status.success() {
        bail!(
            "MoE analysis failed. Log: {}. Cause: llama-moe-analyze exited with {}",
            log_path.display(),
            output.status
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct AnalyzeRow {
    expert_id: u32,
    gate_mass: f64,
    selection_count: u64,
}

fn read_analyze_rows(path: &Path) -> Result<Vec<AnalyzeRow>> {
    let content = fs::read_to_string(path).with_context(|| format!("Read {}", path.display()))?;
    let mut rows = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 4 {
            continue;
        }
        rows.push(AnalyzeRow {
            expert_id: parts[0].parse()?,
            gate_mass: parts[1].parse()?,
            selection_count: parts[3].parse()?,
        });
    }
    Ok(rows)
}

fn shell_join(command: &[String]) -> String {
    command
        .iter()
        .map(|part| {
            if part.contains([' ', '\n', '\t', '"', '\'']) {
                format!("{:?}", part)
            } else {
                part.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
