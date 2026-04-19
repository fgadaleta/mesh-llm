use anyhow::{bail, Context, Result};
use hf_hub::{RepoInfo, RepoInfoParams, RepoType};
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::path::Path;
use std::time::Duration;

use crate::cli::moe::{HfJobArgs, HfJobReleaseTarget};
use crate::inference::launch::BinaryFlavor;
use crate::models;
use crate::system::release_target::{CanonicalArch, CanonicalOs, ReleaseTarget};

const DEFAULT_HF_ENDPOINT: &str = "https://huggingface.co";
const CPU_JOB_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm";
const CUDA_JOB_IMAGE: &str = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel";
const ROCM_JOB_IMAGE: &str = "rocm/pytorch:rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0";
const VULKAN_JOB_IMAGE: &str = "ghcr.io/astral-sh/uv:python3.12-bookworm";
const HF_JOB_WRAPPER_TEMPLATE: &str = include_str!("hf_job_wrapper.sh");
const DEFAULT_HF_JOB_FLAVOR: &str = "cpu-xl";
const DEFAULT_HF_JOB_TIMEOUT: &str = "1h";

pub(crate) async fn submit_publish_job(
    model_spec: &str,
    catalog_repo: &str,
    package_namespace: Option<&str>,
    options: &HfJobArgs,
) -> Result<()> {
    let identity = remote_identity(model_spec).await?;
    let spec = JobSubmissionSpec {
        model_ref: identity.distribution_ref(),
        catalog_repo: catalog_repo.to_string(),
        package_namespace: package_namespace.map(ToOwned::to_owned),
        hf_confirm: options.hf_confirm,
        flavor: options.hf_job_flavor.clone(),
        timeout: options.hf_job_timeout.clone(),
        job_namespace: options.hf_job_namespace.clone(),
        release_repo: options.hf_job_release_repo.clone(),
        release_tag: options.hf_job_release_tag.clone(),
        release_target: options.hf_job_release_target,
        source_repo: identity.repo_id.clone(),
        source_revision: identity.revision.clone(),
        source_file: identity.file.clone(),
        distribution_id: crate::system::moe_planner::normalize_distribution_id(
            &identity.local_file_name,
        ),
    };
    submit_job(spec).await
}

#[derive(Clone)]
struct JobSubmissionSpec {
    model_ref: String,
    catalog_repo: String,
    package_namespace: Option<String>,
    hf_confirm: bool,
    flavor: String,
    timeout: String,
    job_namespace: Option<String>,
    release_repo: String,
    release_tag: String,
    release_target: HfJobReleaseTarget,
    source_repo: String,
    source_revision: String,
    source_file: String,
    distribution_id: String,
}

#[derive(Deserialize)]
struct WhoAmIResponse {
    name: String,
}

#[derive(Deserialize)]
struct JobCreateResponse {
    id: String,
    owner: JobOwner,
    status: JobStatus,
}

#[derive(Deserialize)]
struct JobOwner {
    name: String,
}

#[derive(Deserialize)]
struct JobStatus {
    stage: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HardwareFlavor {
    name: String,
    #[serde(default)]
    pretty_name: Option<String>,
    #[serde(default, rename = "unitCostUSD")]
    unit_cost_usd: Option<f64>,
    #[serde(default, rename = "unitCostMicroUSD")]
    unit_cost_micro_usd: Option<u64>,
    #[serde(default)]
    unit_label: Option<String>,
}

impl HardwareFlavor {
    fn pretty_name(&self) -> &str {
        self.pretty_name.as_deref().unwrap_or(&self.name)
    }

    fn unit_label(&self) -> &str {
        self.unit_label.as_deref().unwrap_or("minute")
    }

    fn resolved_unit_cost_usd(&self) -> Result<f64> {
        if let Some(unit_cost_usd) = self.unit_cost_usd {
            return Ok(unit_cost_usd);
        }
        if let Some(unit_cost_micro_usd) = self.unit_cost_micro_usd {
            return Ok(unit_cost_micro_usd as f64 / 1_000_000.0);
        }
        bail!(
            "Hugging Face hardware flavor {} is missing both unitCostUSD and unitCostMicroUSD",
            self.name
        );
    }
}

#[derive(Clone, Debug)]
struct PricingEstimate {
    flavor: HardwareFlavor,
    unit_cost_usd: f64,
    max_cost_usd: f64,
}

#[derive(Clone, Debug)]
struct JobExecutionPlan {
    namespace: String,
    release_url: String,
    flavor: String,
    timeout_seconds: u64,
    minimum_timeout_seconds: u64,
    release_target: HfJobReleaseTarget,
    pricing: PricingEstimate,
    auto_selected_hardware: bool,
    timeout_bumped_to_minimum: bool,
    model_size_bytes: u64,
}

async fn submit_job(spec: JobSubmissionSpec) -> Result<()> {
    let token = models::hf_token_override().ok_or_else(|| {
        anyhow::anyhow!(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before using --hf-job."
        )
    })?;
    let endpoint = hf_endpoint();
    let namespace = match spec.job_namespace.clone() {
        Some(namespace) => namespace,
        None => resolve_namespace(&endpoint, &token).await?,
    };
    let plan = build_job_execution_plan(&endpoint, &spec, &namespace).await?;
    let script = remote_bash_script(&spec);
    let payload = json!({
        "dockerImage": job_image(plan.release_target),
        "command": ["bash", "-lc", script],
        "arguments": [],
        "environment": {
            "MESH_LLM_RELEASE_URL": plan.release_url,
            "MODEL_REF": spec.model_ref,
            "SOURCE_REPO": spec.source_repo,
            "SOURCE_REVISION": spec.source_revision,
            "SOURCE_FILE": spec.source_file,
            "CATALOG_REPO": spec.catalog_repo,
            "HF_JOB_FLAVOR": plan.pricing.flavor.name,
            "HF_JOB_FLAVOR_PRETTY": plan.pricing.flavor.pretty_name(),
            "HF_JOB_UNIT_COST_USD": format!("{:.6}", plan.pricing.unit_cost_usd),
            "HF_JOB_UNIT_LABEL": plan.pricing.flavor.unit_label(),
            "HF_JOB_TIMEOUT_SECONDS": plan.timeout_seconds.to_string(),
            "HF_JOB_MAX_COST_USD": format!("{:.2}", plan.pricing.max_cost_usd),
        },
        "secrets": {
            "HF_TOKEN": token,
        },
        "flavor": plan.flavor,
        "timeoutSeconds": plan.timeout_seconds,
        "labels": {
            "app": "mesh-llm",
            "workflow": "moe-publish",
            "source_repo": sanitize_label(&spec.source_repo),
            "source_revision": sanitize_label(&spec.source_revision),
            "distribution_id": sanitize_label(&spec.distribution_id),
            "catalog_repo": sanitize_label(&spec.catalog_repo),
        }
    });

    print_job_plan(&spec, &plan, false);
    if !spec.hf_confirm {
        println!("🧪 Dry run only");
        println!("   rerun with `--hf-confirm` to submit this Hugging Face Job");
        return Ok(());
    }

    println!("☁️ Hugging Face Job submission");
    println!("📦 Model: {}", spec.model_ref);
    println!("📤 Workflow: moe publish");
    println!("✅ Confirmation received; submitting remote job");

    let url = format!(
        "{}/api/jobs/{}",
        endpoint.trim_end_matches('/'),
        plan.namespace
    );
    let response = reqwest::Client::new()
        .post(&url)
        .bearer_auth(&token)
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("POST {}", url))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("HF Job submission failed: {}: {}", status, body.trim());
    }
    let job: JobCreateResponse = response
        .json()
        .await
        .context("Decode Hugging Face Jobs response")?;
    println!("✅ Submitted Hugging Face Job");
    println!("🆔 Job: {}", job.id);
    println!("📡 Status: {}", job.status.stage);
    println!(
        "🔗 URL: {}/jobs/{}/{}",
        endpoint.trim_end_matches('/'),
        job.owner.name,
        job.id
    );
    Ok(())
}

async fn build_job_execution_plan(
    endpoint: &str,
    spec: &JobSubmissionSpec,
    namespace: &str,
) -> Result<JobExecutionPlan> {
    let model_size_bytes =
        resolve_model_size_bytes(&spec.source_repo, &spec.source_revision, &spec.source_file)
            .await
            .unwrap_or(0);
    let auto_selected_hardware = spec.flavor == DEFAULT_HF_JOB_FLAVOR
        && spec.timeout == DEFAULT_HF_JOB_TIMEOUT
        && spec.release_target == HfJobReleaseTarget::Cpu;
    let (release_target, flavor, minimum_timeout_seconds) = if auto_selected_hardware {
        recommend_hf_job_resources(model_size_bytes)
    } else {
        (
            spec.release_target,
            spec.flavor.clone(),
            recommended_min_timeout_seconds(model_size_bytes, spec.release_target),
        )
    };
    let requested_timeout_seconds = parse_timeout_seconds(&spec.timeout)?;
    let timeout_seconds = requested_timeout_seconds.max(minimum_timeout_seconds);
    let timeout_bumped_to_minimum = timeout_seconds != requested_timeout_seconds;
    let pricing = fetch_pricing_estimate(endpoint, &flavor, timeout_seconds).await?;
    let release_url = release_download_url(
        &spec.release_repo,
        &spec.release_tag,
        &release_asset_name(&spec.release_tag, release_target),
    );
    Ok(JobExecutionPlan {
        namespace: namespace.to_string(),
        release_url,
        flavor,
        timeout_seconds,
        minimum_timeout_seconds,
        release_target,
        pricing,
        auto_selected_hardware,
        timeout_bumped_to_minimum,
        model_size_bytes,
    })
}

async fn resolve_model_size_bytes(
    source_repo: &str,
    source_revision: &str,
    source_file: &str,
) -> Result<u64> {
    let local_path = crate::models::local::huggingface_snapshot_path(
        source_repo,
        RepoType::Model,
        source_revision,
    )
    .join(source_file);
    if let Ok(metadata) = std::fs::metadata(&local_path) {
        return Ok(metadata.len());
    }

    let source_repo = source_repo.to_string();
    let source_revision = source_revision.to_string();
    let source_file = source_file.to_string();
    crate::models::run_hf_blocking(move || {
        let api = crate::models::build_hf_api(false)?;
        let (owner, name) = source_repo
            .split_once('/')
            .unwrap_or(("", source_repo.as_str()));
        let info = api
            .model(owner, name)
            .info(
                &RepoInfoParams::builder()
                    .revision(source_revision.clone())
                    .build(),
            )
            .with_context(|| format!("Fetch Hugging Face repo {source_repo}@{source_revision}"))?;
        let RepoInfo::Model(info) = info else {
            bail!("Expected model repo info for {}", source_repo);
        };
        let size = info
            .siblings
            .unwrap_or_default()
            .into_iter()
            .find(|sibling| sibling.rfilename == source_file)
            .and_then(|sibling| sibling.size)
            .ok_or_else(|| anyhow::anyhow!("No size metadata found for {}", source_file))?;
        Ok(size)
    })
}

fn recommend_hf_job_resources(model_size_bytes: u64) -> (HfJobReleaseTarget, String, u64) {
    let gib = model_size_bytes as f64 / 1024_f64.powi(3);
    if gib <= 8.0 {
        (HfJobReleaseTarget::Cpu, "cpu-xl".to_string(), 2 * 60 * 60)
    } else if gib <= 16.0 {
        (
            HfJobReleaseTarget::Cuda,
            "t4-medium".to_string(),
            4 * 60 * 60,
        )
    } else if gib <= 28.0 {
        (HfJobReleaseTarget::Cuda, "l40sx1".to_string(), 6 * 60 * 60)
    } else if gib <= 48.0 {
        (HfJobReleaseTarget::Cuda, "h200".to_string(), 8 * 60 * 60)
    } else {
        (HfJobReleaseTarget::Cuda, "h200x2".to_string(), 12 * 60 * 60)
    }
}

fn recommended_min_timeout_seconds(
    model_size_bytes: u64,
    release_target: HfJobReleaseTarget,
) -> u64 {
    let gib = model_size_bytes as f64 / 1024_f64.powi(3);
    match release_target {
        HfJobReleaseTarget::Cpu => {
            if gib <= 8.0 {
                2 * 60 * 60
            } else {
                6 * 60 * 60
            }
        }
        HfJobReleaseTarget::Cuda | HfJobReleaseTarget::Rocm | HfJobReleaseTarget::Vulkan => {
            if gib <= 16.0 {
                4 * 60 * 60
            } else if gib <= 28.0 {
                6 * 60 * 60
            } else if gib <= 48.0 {
                8 * 60 * 60
            } else {
                12 * 60 * 60
            }
        }
    }
}

fn print_job_plan(spec: &JobSubmissionSpec, plan: &JobExecutionPlan, confirmed: bool) {
    println!(
        "☁️ Hugging Face Job {}",
        if confirmed { "submission" } else { "dry run" }
    );
    println!("📦 Model: {}", spec.model_ref);
    println!("📤 Workflow: moe publish");
    println!("🗂️ Catalog: {}", spec.catalog_repo);
    if let Some(package_namespace) = &spec.package_namespace {
        println!("📦 Package namespace: {}", package_namespace);
    }
    if plan.model_size_bytes > 0 {
        println!("📏 Model size: {:.1}GB", plan.model_size_bytes as f64 / 1e9);
    }
    println!(
        "🖥️ Hardware: {} ({}, target={:?}){}",
        plan.pricing.flavor.pretty_name(),
        plan.flavor,
        plan.release_target,
        if plan.auto_selected_hardware {
            " [auto]"
        } else {
            ""
        }
    );
    println!(
        "⏱️ Minimum timeout: {}",
        format_timeout(plan.minimum_timeout_seconds)
    );
    println!("⏱️ Requested timeout: {}", spec.timeout);
    println!(
        "⏱️ Effective timeout: {}{}",
        format_timeout(plan.timeout_seconds),
        if plan.timeout_bumped_to_minimum {
            " (bumped to minimum)"
        } else {
            ""
        }
    );
    println!("📥 Release: {} {}", spec.release_repo, spec.release_tag);
    println!(
        "💵 Pricing: {} @ ${:.6}/{}",
        plan.pricing.flavor.pretty_name(),
        plan.pricing.unit_cost_usd,
        plan.pricing.flavor.unit_label()
    );
    println!("🧮 Max cost: ${:.2} USD", plan.pricing.max_cost_usd);
}

fn format_timeout(timeout_seconds: u64) -> String {
    let duration = Duration::from_secs(timeout_seconds);
    let hours = duration.as_secs() / 3600;
    let minutes = (duration.as_secs() % 3600) / 60;
    let seconds = duration.as_secs() % 60;
    if hours > 0 && minutes > 0 {
        format!("{hours}h {minutes}m")
    } else if hours > 0 {
        format!("{hours}h")
    } else if minutes > 0 && seconds > 0 {
        format!("{minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m")
    } else {
        format!("{seconds}s")
    }
}

async fn fetch_pricing_estimate(
    endpoint: &str,
    flavor: &str,
    timeout_seconds: u64,
) -> Result<PricingEstimate> {
    let url = format!("{}/api/jobs/hardware", endpoint.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .get(&url)
        .send()
        .await
        .with_context(|| format!("GET {}", url))?;
    if response.status() != StatusCode::OK {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!(
            "Failed to resolve Hugging Face Jobs pricing: {}: {}",
            status,
            body.trim()
        );
    }
    let hardware: Vec<HardwareFlavor> = response
        .json()
        .await
        .context("Decode Hugging Face hardware pricing response")?;
    let flavor = hardware
        .into_iter()
        .find(|candidate| candidate.name == flavor)
        .ok_or_else(|| anyhow::anyhow!("Unknown Hugging Face Jobs flavor: {flavor}"))?;
    let unit_cost_usd = flavor.resolved_unit_cost_usd()?;
    let max_cost_usd = estimate_cost_usd(unit_cost_usd, flavor.unit_label(), timeout_seconds)?;
    Ok(PricingEstimate {
        flavor,
        unit_cost_usd,
        max_cost_usd,
    })
}

async fn remote_identity(model_spec: &str) -> Result<models::local::HuggingFaceModelIdentity> {
    let path = Path::new(model_spec);
    if path.exists() {
        return models::huggingface_identity_for_path(path).ok_or_else(|| {
            anyhow::anyhow!(
                "--hf-job requires a Hugging Face-backed local model path or a Hugging Face model ref."
            )
        });
    }
    models::resolve_huggingface_model_identity(model_spec)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "--hf-job requires a Hugging Face-backed model ref, selector, URL, or cached HF local path."
            )
        })
}

async fn resolve_namespace(endpoint: &str, token: &str) -> Result<String> {
    let url = format!("{}/api/whoami-v2", endpoint.trim_end_matches('/'));
    let response = reqwest::Client::new()
        .get(&url)
        .bearer_auth(token)
        .send()
        .await
        .with_context(|| format!("GET {}", url))?;
    if response.status() != StatusCode::OK {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!(
            "Failed to resolve Hugging Face namespace: {}: {}",
            status,
            body.trim()
        );
    }
    let whoami: WhoAmIResponse = response.json().await.context("Decode whoami response")?;
    Ok(whoami.name)
}

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_HF_ENDPOINT.to_string())
}

fn remote_bash_script(spec: &JobSubmissionSpec) -> String {
    let mut publish_command =
        "./mesh-llm moe publish \"$MODEL_REF\" --catalog-repo \"$CATALOG_REPO\"".to_string();
    if let Some(namespace) = &spec.package_namespace {
        publish_command.push_str(" --namespace ");
        publish_command.push_str(&shell_single_quote(namespace));
    }
    HF_JOB_WRAPPER_TEMPLATE.replace("__PUBLISH_COMMAND__", &shell_single_quote(&publish_command))
}

fn parse_timeout_seconds(input: &str) -> Result<u64> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("HF job timeout must not be empty");
    }
    let split_at = trimmed
        .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .unwrap_or(trimmed.len());
    let (number, suffix) = trimmed.split_at(split_at);
    let value: f64 = number
        .parse()
        .with_context(|| format!("Parse timeout value from {}", input))?;
    let factor = match suffix {
        "" | "s" => 1.0,
        "m" => 60.0,
        "h" => 3600.0,
        "d" => 86400.0,
        _ => bail!("Unsupported timeout unit in {}. Use s, m, h, or d.", input),
    };
    Ok((value * factor).round() as u64)
}

fn estimate_cost_usd(unit_cost_usd: f64, unit_label: &str, timeout_seconds: u64) -> Result<f64> {
    let timeout_seconds = timeout_seconds as f64;
    let max_cost = match unit_label {
        "second" => unit_cost_usd * timeout_seconds,
        "minute" => unit_cost_usd * (timeout_seconds / 60.0),
        "hour" => unit_cost_usd * (timeout_seconds / 3600.0),
        "day" => unit_cost_usd * (timeout_seconds / 86_400.0),
        other => bail!("Unsupported Hugging Face pricing unit: {other}"),
    };
    Ok(max_cost)
}

fn hf_job_binary_flavor(release_target: HfJobReleaseTarget) -> BinaryFlavor {
    match release_target {
        HfJobReleaseTarget::Cpu => BinaryFlavor::Cpu,
        HfJobReleaseTarget::Cuda => BinaryFlavor::Cuda,
        HfJobReleaseTarget::Rocm => BinaryFlavor::Rocm,
        HfJobReleaseTarget::Vulkan => BinaryFlavor::Vulkan,
    }
}

fn linux_x86_64_release_target(flavor: BinaryFlavor) -> ReleaseTarget {
    ReleaseTarget::new(CanonicalOs::Linux, CanonicalArch::X86_64, flavor)
}

fn release_asset_name(release_tag: &str, release_target: HfJobReleaseTarget) -> String {
    let target = linux_x86_64_release_target(hf_job_binary_flavor(release_target));
    if release_tag == "latest" {
        target
            .stable_asset_name()
            .expect("linux x86_64 HF job targets must have a stable release asset")
    } else {
        target
            .versioned_asset_name(release_tag)
            .expect("linux x86_64 HF job targets must have a versioned release asset")
    }
}

#[cfg(test)]
pub(crate) fn release_target_versioned_linux_asset_name(
    release_tag: &str,
    flavor: BinaryFlavor,
) -> Option<String> {
    linux_x86_64_release_target(flavor).versioned_asset_name(release_tag)
}

fn release_download_url(release_repo: &str, release_tag: &str, asset_name: &str) -> String {
    if release_tag == "latest" {
        format!("https://github.com/{release_repo}/releases/latest/download/{asset_name}")
    } else {
        format!("https://github.com/{release_repo}/releases/download/{release_tag}/{asset_name}")
    }
}

fn job_image(release_target: HfJobReleaseTarget) -> &'static str {
    match release_target {
        HfJobReleaseTarget::Cpu => CPU_JOB_IMAGE,
        HfJobReleaseTarget::Cuda => CUDA_JOB_IMAGE,
        HfJobReleaseTarget::Rocm => ROCM_JOB_IMAGE,
        HfJobReleaseTarget::Vulkan => VULKAN_JOB_IMAGE,
    }
}

fn sanitize_label(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '=' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

fn shell_single_quote(value: &str) -> String {
    value.replace('\'', "'\"'\"'")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_timeout_accepts_supported_units() {
        assert_eq!(parse_timeout_seconds("90").unwrap(), 90);
        assert_eq!(parse_timeout_seconds("1.5h").unwrap(), 5400);
        assert_eq!(parse_timeout_seconds("30m").unwrap(), 1800);
    }

    #[test]
    fn estimate_cost_supports_minute_pricing() {
        let cost = estimate_cost_usd(2.0, "minute", 1800).unwrap();
        assert!((cost - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn estimate_cost_supports_hour_pricing() {
        let cost = estimate_cost_usd(12.0, "hour", 1800).unwrap();
        assert!((cost - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sanitize_label_replaces_unsupported_characters() {
        assert_eq!(sanitize_label("meshllm/catalog"), "meshllm-catalog");
        assert_eq!(sanitize_label("  weird value  "), "weird-value");
    }

    #[test]
    fn shell_single_quote_escapes_embedded_quotes() {
        assert_eq!(shell_single_quote("a'b"), "a'\"'\"'b");
    }

    #[test]
    fn job_images_match_runtime_targets() {
        assert_eq!(job_image(HfJobReleaseTarget::Cpu), CPU_JOB_IMAGE);
        assert_eq!(job_image(HfJobReleaseTarget::Cuda), CUDA_JOB_IMAGE);
        assert_eq!(job_image(HfJobReleaseTarget::Rocm), ROCM_JOB_IMAGE);
        assert_eq!(job_image(HfJobReleaseTarget::Vulkan), VULKAN_JOB_IMAGE);
    }

    #[test]
    fn release_assets_match_expected_linux_names() {
        let versioned = release_target_versioned_linux_asset_name("v0.1.2", BinaryFlavor::Cpu)
            .expect("linux x86_64 cpu release asset");
        assert_eq!(
            release_asset_name("v0.1.2", HfJobReleaseTarget::Cpu),
            versioned
        );
    }

    #[test]
    fn release_target_hf_jobs_parity() {
        for target in [
            HfJobReleaseTarget::Cpu,
            HfJobReleaseTarget::Cuda,
            HfJobReleaseTarget::Rocm,
            HfJobReleaseTarget::Vulkan,
        ] {
            let _ = release_asset_name("latest", target);
        }
    }

    #[test]
    fn remote_publish_script_includes_namespace_when_requested() {
        let spec = JobSubmissionSpec {
            model_ref: "mesh/model:Q4".to_string(),
            catalog_repo: "meshllm/catalog".to_string(),
            package_namespace: Some("meshllm".to_string()),
            hf_confirm: true,
            flavor: "cpu-xl".to_string(),
            timeout: "1h".to_string(),
            job_namespace: None,
            release_repo: "Mesh-LLM/mesh-llm".to_string(),
            release_tag: "latest".to_string(),
            release_target: HfJobReleaseTarget::Cpu,
            source_repo: "unsloth/demo".to_string(),
            source_revision: "deadbeef".to_string(),
            source_file: "model.gguf".to_string(),
            distribution_id: "Demo-Q4".to_string(),
        };
        let script = remote_bash_script(&spec);
        assert!(script.contains("__PUBLISH_COMMAND__") == false);
        assert!(script.contains("moe publish"));
        assert!(script.contains("--namespace"));
    }

    #[test]
    fn recommend_hf_job_resources_uses_gpu_for_mid_sized_models() {
        let (target, flavor, timeout) = recommend_hf_job_resources(22 * 1024 * 1024 * 1024);
        assert_eq!(target, HfJobReleaseTarget::Cuda);
        assert_eq!(flavor, "l40sx1");
        assert_eq!(timeout, 6 * 60 * 60);
    }

    #[test]
    fn recommended_min_timeout_scales_with_model_size() {
        assert_eq!(
            recommended_min_timeout_seconds(6 * 1024 * 1024 * 1024, HfJobReleaseTarget::Cpu),
            2 * 60 * 60
        );
        assert_eq!(
            recommended_min_timeout_seconds(22 * 1024 * 1024 * 1024, HfJobReleaseTarget::Cuda),
            6 * 60 * 60
        );
    }
}
