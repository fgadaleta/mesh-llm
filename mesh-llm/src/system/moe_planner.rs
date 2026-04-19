mod catalog_resolution;
mod fit;
mod package;

use anyhow::{bail, Context, Result};
use hf_hub::{RepoDownloadFileParams, RepoInfoParams};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::cli::terminal_progress::start_spinner;
use crate::inference::{election, moe};
use crate::models::{
    build_hf_api, catalog, gguf::GgufTensorByteProfile, huggingface_identity_for_path,
    resolve_model_spec, resolve_model_spec_with_progress,
};

#[cfg(test)]
use self::catalog_resolution::{
    analyzer_id_from_analysis, join_repo_relative, select_preferred_ranking,
};
pub(crate) use self::catalog_resolution::{
    catalog_entry_path_for_source_repo, fetch_remote_package_expert_components,
    resolve_runtime_ranking, sort_catalog_package_pointers,
};
use self::catalog_resolution::{fetch_remote_package_ranking, resolve_best_ranking};
use self::fit::{
    aggregate_tensor_byte_profile, build_mass_checkpoints, expert_bytes_json,
    predict_plan_fit_for_nodes,
};
pub(crate) use self::fit::{estimate_startup_fit_from_analysis, fetch_remote_startup_fit};
pub(crate) use self::package::{
    analysis_log_path, build_meshllm_descriptor, build_submit_bundle,
    canonical_model_ref_from_source, default_package_repo_name_for_model,
    infer_variant_key_from_gguf_file, normalize_variant_selector, read_analysis_json, sha256_file,
    variant_key_for_model, write_analysis_json, write_package_analysis_json,
    write_package_benchmark_json,
};

const DEFAULT_DATASET_REVISION: &str = "main";
pub(crate) const DEFAULT_MOE_CATALOG_DATASET: &str = "meshllm/catalog";
const ANALYSIS_JSON_FILENAME: &str = "analysis.json";
const DEFAULT_MASS_CHECKPOINTS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
const MODEL_LOAD_HEADROOM_NUMERATOR: u64 = 11;
const MODEL_LOAD_HEADROOM_DENOMINATOR: u64 = 10;

#[derive(Clone, Debug)]
pub(crate) struct MoePlanArgs {
    pub model: String,
    pub max_vram_gb: Option<f64>,
    pub nodes: Option<usize>,
    pub catalog_repo: String,
    pub progress: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct MoePlanReport {
    pub model: MoeModelContext,
    pub ranking: ResolvedRanking,
    pub target_vram_bytes: u64,
    pub recommended_nodes: usize,
    pub max_supported_nodes: usize,
    pub feasible: bool,
    pub assumptions: Vec<String>,
    pub assignments: Vec<moe::NodeAssignment>,
    pub shared_mass_pct: Option<f64>,
    pub max_node_mass_pct: Option<f64>,
    pub min_node_mass_pct: Option<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeModelContext {
    pub input: String,
    pub path: PathBuf,
    pub display_name: String,
    pub source_repo: Option<String>,
    pub source_revision: Option<String>,
    pub distribution_id: String,
    pub expert_count: u32,
    pub used_expert_count: u32,
    pub min_experts_per_node: u32,
    pub total_model_bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedRanking {
    pub path: PathBuf,
    pub metadata_path: Option<PathBuf>,
    pub analysis_path: Option<PathBuf>,
    pub analyzer_id: String,
    pub source: RankingSource,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub(crate) struct MoeSubmitBundle {
    pub model_ref: String,
    pub variant: String,
    pub variant_root: String,
    pub ranking_repo_path: String,
    pub analysis_repo_path: String,
    pub manifest_repo_path: String,
    pub log_repo_path: Option<String>,
    pub commit_message: String,
    pub commit_description: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct MeshllmPackageJson {
    pub schema_version: u32,
    pub source: MeshllmPackageSource,
    pub variants: BTreeMap<String, MeshllmPackageVariant>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct MeshllmPackageSource {
    pub repo: String,
    pub revision: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct MeshllmPackageVariant {
    pub distribution_id: String,
    pub manifest: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct MoePackageManifest {
    pub schema_version: u32,
    pub format: String,
    pub ranking_sha256: String,
    pub n_expert: u32,
    pub n_expert_used: u32,
    pub min_experts_per_node: u32,
    pub trunk: moe::ExpertComponentFile,
    pub experts: Vec<moe::ExpertComponentFile>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResolvedExpertComponents {
    pub prefix: String,
    pub trunk_path: PathBuf,
    pub expert_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisJson {
    pub schema_version: u32,
    pub ranking: MoeAnalysisRanking,
    pub model: MoeAnalysisModel,
    pub memory: MoeAnalysisMemory,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisRanking {
    pub sha256: String,
    pub rows: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisModel {
    pub expert_count: u32,
    pub expert_used_count: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct MoeAnalysisMemory {
    pub full_model_bytes: u64,
    pub base_resident_bytes: u64,
    pub shard_file_overhead_bytes: u64,
    pub expert_tensor_bytes_total: u64,
    pub expert_bytes: MoeAnalysisExpertBytes,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum MoeAnalysisExpertBytes {
    Uniform { bytes_per_expert: u64 },
    DensePerExpert { values: Vec<u64> },
}

impl MoeAnalysisExpertBytes {
    fn bytes_for(&self, expert_id: u32) -> Result<u64> {
        match self {
            Self::Uniform { bytes_per_expert } => Ok(*bytes_per_expert),
            Self::DensePerExpert { values } => values
                .get(expert_id as usize)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("expert id {} missing from analysis", expert_id)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MoeStartupFitEstimate {
    pub analyzer_id: String,
    pub ranking_source: &'static str,
    pub target_vram_bytes: u64,
    pub required_experts_per_node: u32,
    pub full_model_bytes: u64,
    pub full_model_launch_bytes: u64,
    pub recommended_nodes: Option<usize>,
    pub predicted_max_shard_bytes: Option<u64>,
    pub predicted_max_shard_launch_bytes: Option<u64>,
}

impl MoeStartupFitEstimate {
    pub(crate) fn full_model_fits(&self) -> bool {
        self.full_model_launch_bytes <= self.target_vram_bytes
    }

    pub(crate) fn shard_plan_fits(&self) -> bool {
        self.predicted_max_shard_launch_bytes
            .is_some_and(|bytes| bytes <= self.target_vram_bytes)
    }

    pub(crate) fn any_fit_exists(&self) -> bool {
        self.full_model_fits() || self.shard_plan_fits()
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RankingSource {
    LocalCache,
    HuggingFacePackage,
}

impl RankingSource {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::LocalCache => "local cache",
            Self::HuggingFacePackage => "Hugging Face package",
        }
    }
}

#[derive(Debug)]
struct AnalyzeMassProfile {
    ranking: Vec<u32>,
    masses: Vec<(u32, f64)>,
    total_mass: f64,
}

#[derive(Clone, Debug, Default)]
struct AggregatedTensorByteProfile {
    full_model_bytes: u64,
    base_resident_bytes: u64,
    expert_tensor_bytes: u64,
    file_overhead_bytes: u64,
}

pub(crate) async fn plan_moe(args: MoePlanArgs) -> Result<MoePlanReport> {
    if let Some(nodes) = args.nodes {
        if nodes == 0 {
            bail!("--nodes must be at least 1");
        }
    }
    let mut model = resolve_model_context_with_progress(&args.model, args.progress).await?;
    let ranking = resolve_best_ranking(&model, &args).await?;
    let target_vram_bytes = resolve_target_vram_bytes(args.max_vram_gb);
    if target_vram_bytes == 0 {
        bail!(
            "No VRAM target available. Pass --max-vram <GB> or run on a machine with detectable GPU memory."
        );
    }

    let ranking_vec = moe::load_cached_ranking(&ranking.path)
        .ok_or_else(|| anyhow::anyhow!("cached ranking not found: {}", ranking.path.display()))
        .with_context(|| format!("Load ranking {}", ranking.path.display()))?;
    let heuristic_recommended_nodes = ((model.total_model_bytes as f64) / target_vram_bytes as f64)
        .ceil()
        .max(1.0) as usize;
    let analysis = ranking
        .analysis_path
        .as_deref()
        .map(read_analysis_json)
        .transpose()?;

    let (recommended_nodes, max_supported_nodes, feasible, mut assumptions) = if let Some(
        ref analysis,
    ) = analysis
    {
        let fit = estimate_startup_fit_from_analysis(
            &ranking.path,
            analysis,
            target_vram_bytes,
            &ranking.analyzer_id,
            ranking.source.label(),
        )?;
        let required_experts_per_node = fit.required_experts_per_node;
        model.min_experts_per_node = required_experts_per_node;
        let max_supported_nodes =
            (model.expert_count / required_experts_per_node.max(1)).max(1) as usize;
        let recommended_nodes = args
            .nodes
            .unwrap_or_else(|| fit.recommended_nodes.unwrap_or(max_supported_nodes));
        let feasible = if recommended_nodes == 1 {
            fit.full_model_fits()
        } else if recommended_nodes > max_supported_nodes {
            false
        } else {
            let (_, _, predicted_max_shard_launch_bytes) = predict_plan_fit_for_nodes(
                &ranking_vec,
                analysis,
                recommended_nodes,
                required_experts_per_node,
            )?;
            predicted_max_shard_launch_bytes <= target_vram_bytes
        };

        let mut assumptions = vec![
            format!(
                "Full-model launch estimate from analysis.json: {:.1}GB against {:.1}GB target",
                fit.full_model_launch_bytes as f64 / 1e9,
                target_vram_bytes as f64 / 1e9
            ),
            format!(
                "Shared core uses the current 50% planning heuristic: {} / {} experts per node",
                required_experts_per_node, model.expert_count
            ),
        ];
        if recommended_nodes > 1 {
            let (_, predicted_max_shard_bytes, predicted_max_shard_launch_bytes) =
                predict_plan_fit_for_nodes(
                    &ranking_vec,
                    analysis,
                    recommended_nodes,
                    required_experts_per_node,
                )?;
            assumptions.push(format!(
                "Estimated shard launch for {} nodes from analysis.json: {:.1}GB max per node ({:.1}GB raw shard bytes)",
                recommended_nodes,
                predicted_max_shard_launch_bytes as f64 / 1e9,
                predicted_max_shard_bytes as f64 / 1e9
            ));
        }
        if args.nodes.is_none() && !fit.any_fit_exists() {
            assumptions.push(
                "No node count up to the current max useful node count fits this VRAM target under the conservative shared-core heuristic."
                    .to_string(),
            );
        } else if args.nodes.is_some() && !feasible {
            assumptions.push(format!(
                "Requested node count {} does not fit this VRAM target under the current shared-core heuristic.",
                recommended_nodes
            ));
        }

        (
            recommended_nodes,
            max_supported_nodes,
            feasible,
            assumptions,
        )
    } else {
        let recommended_nodes = args.nodes.unwrap_or(heuristic_recommended_nodes);
        let max_supported_nodes =
            (model.expert_count / model.min_experts_per_node.max(1)).max(1) as usize;
        let feasible = recommended_nodes <= max_supported_nodes;
        let assumptions = vec![
            format!(
                "Minimum nodes estimated from total model bytes / target VRAM: {:.1}GB / {:.1}GB",
                model.total_model_bytes as f64 / 1e9,
                target_vram_bytes as f64 / 1e9
            ),
            format!(
                "Minimum experts per node falls back to local planner state: {}",
                model.min_experts_per_node
            ),
            "No analysis.json was available, so shard byte fit uses the legacy heuristic."
                .to_string(),
        ];
        (
            recommended_nodes,
            max_supported_nodes,
            feasible,
            assumptions,
        )
    };

    let assignments = moe::compute_assignments_with_overlap(
        &ranking_vec,
        recommended_nodes,
        model.min_experts_per_node,
        1,
    );
    let profile = load_analyze_mass_profile(&ranking.path).ok();
    let (shared_mass_pct, max_node_mass_pct, min_node_mass_pct) = if let Some(ref profile) = profile
    {
        let shared = assignments
            .first()
            .map(|assignment| assignment.n_shared)
            .unwrap_or_default();
        let shared_mass_pct = mass_pct_for_experts(
            profile,
            &profile.ranking[..shared.min(profile.ranking.len())],
        );
        let node_mass: Vec<f64> = assignments
            .iter()
            .map(|assignment| mass_pct_for_experts(profile, &assignment.experts))
            .collect();
        (
            Some(shared_mass_pct),
            node_mass.iter().copied().reduce(f64::max),
            node_mass.iter().copied().reduce(f64::min),
        )
    } else {
        (None, None, None)
    };
    if profile.is_none() {
        assumptions.push(
            "Ranking file does not include gate-mass columns, so shared/node mass percentages are unavailable."
                .to_string(),
        );
    }

    Ok(MoePlanReport {
        model,
        ranking,
        target_vram_bytes,
        recommended_nodes,
        max_supported_nodes,
        feasible,
        assumptions,
        assignments,
        shared_mass_pct,
        max_node_mass_pct,
        min_node_mass_pct,
    })
}

pub(crate) async fn resolve_model_context(model_spec: &str) -> Result<MoeModelContext> {
    resolve_model_context_with_progress(model_spec, true).await
}

pub(crate) async fn resolve_model_context_with_progress(
    model_spec: &str,
    progress: bool,
) -> Result<MoeModelContext> {
    let path = if progress {
        resolve_model_spec(Path::new(model_spec)).await?
    } else {
        resolve_model_spec_with_progress(Path::new(model_spec), false).await?
    };
    build_model_context(model_spec, path)
}

pub(crate) fn resolve_model_context_for_path(path: &Path) -> Result<MoeModelContext> {
    build_model_context(&path.to_string_lossy(), path.to_path_buf())
}

fn build_model_context(model_spec: &str, path: PathBuf) -> Result<MoeModelContext> {
    let info = moe::detect_moe(&path).with_context(|| {
        format!(
            "Model is not auto-detected as MoE from the GGUF header: {}",
            path.display()
        )
    })?;
    let display_name = model_display_name(&path);
    let identity = huggingface_identity_for_path(&path);
    let source_repo = identity.as_ref().map(|identity| identity.repo_id.clone());
    let source_revision = identity.as_ref().map(|identity| identity.revision.clone());
    let distribution_id = identity
        .as_ref()
        .map(|identity| normalize_distribution_id(&identity.local_file_name))
        .unwrap_or_else(|| normalize_distribution_id(&display_name));
    let min_experts_per_node = bundled_min_experts(&display_name)
        .unwrap_or_else(|| ((info.expert_count as f64) * 0.5).ceil() as u32);
    Ok(MoeModelContext {
        input: model_spec.to_string(),
        total_model_bytes: election::total_model_bytes(&path),
        path,
        display_name,
        source_repo,
        source_revision,
        distribution_id,
        expert_count: info.expert_count,
        used_expert_count: info.expert_used_count,
        min_experts_per_node,
    })
}

pub(crate) fn resolve_target_vram_bytes(max_vram_gb: Option<f64>) -> u64 {
    crate::mesh::detect_vram_bytes_capped(max_vram_gb)
}

pub(crate) fn normalize_distribution_id(name: &str) -> String {
    let stem = Path::new(name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(name)
        .trim_end_matches(".gguf");
    if let Some((prefix, suffix)) = stem.rsplit_once("-of-") {
        let has_numeric_suffix = suffix.len() == 5 && suffix.chars().all(|ch| ch.is_ascii_digit());
        let has_numeric_prefix = prefix.len() > 6
            && prefix[prefix.len() - 6..].starts_with('-')
            && prefix[prefix.len() - 5..]
                .chars()
                .all(|ch| ch.is_ascii_digit());
        if has_numeric_suffix && has_numeric_prefix {
            return prefix[..prefix.len() - 6].to_string();
        }
    }
    stem.to_string()
}

#[cfg(test)]
pub(crate) fn local_submit_ranking(model: &MoeModelContext) -> Result<ResolvedRanking> {
    let path = moe::package_cache_ranking_path(&model.path);
    if !path.exists() {
        bail!(
            "No local ranking artifact found for {}. Run `mesh-llm moe analyze full {}` or `mesh-llm moe analyze micro {}` first.",
            model.display_name,
            model.input,
            model.input
        );
    };
    Ok(ResolvedRanking {
        analysis_path: sibling_analysis_path(&path),
        path,
        metadata_path: None,
        analyzer_id: analyzer_id_from_analysis(&moe::package_cache_analysis_path(&model.path))
            .unwrap_or_else(|_| "full-v1".to_string()),
        source: RankingSource::LocalCache,
        reason: "local materialized package ranking".to_string(),
    })
}

pub(crate) fn validate_ranking(model: &MoeModelContext, ranking: &ResolvedRanking) -> Result<()> {
    let loaded = moe::load_cached_ranking(&ranking.path)
        .ok_or_else(|| anyhow::anyhow!("Could not parse ranking {}", ranking.path.display()))?;
    let artifact = moe::SharedRankingArtifact {
        kind: ranking_kind_for_analyzer(&ranking.analyzer_id),
        origin: moe::SharedRankingOrigin::LegacyCache,
        ranking: loaded,
        micro_prompt_count: None,
        micro_tokens: None,
        micro_layer_scope: None,
    };
    moe::validate_shared_ranking_artifact(&model.path, &artifact)?;
    load_analyze_mass_profile(&ranking.path).with_context(|| {
        format!(
            "Ranking {} must include gate-mass columns",
            ranking.path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
fn infer_analyzer_from_ranking_path(path: &Path) -> Option<&'static str> {
    let text = path.to_string_lossy().to_ascii_lowercase();
    if text.contains("/full-v1/") || text.contains("\\full-v1\\") {
        return Some("full-v1");
    }
    if text.contains("/micro-v1/") || text.contains("\\micro-v1\\") {
        return Some("micro-v1");
    }

    let file_name = path.file_name()?.to_string_lossy().to_ascii_lowercase();
    if file_name.contains("micro-v1") {
        return Some("micro-v1");
    }
    if file_name.contains("full-v1") {
        return Some("full-v1");
    }
    if file_name.starts_with("local-")
        && file_name.contains(".micro-p")
        && file_name.ends_with(".csv")
    {
        return Some("micro-v1");
    }
    if file_name.starts_with("local-") && file_name.ends_with(".csv") {
        return Some("full-v1");
    }

    None
}

fn ranking_kind_for_analyzer(analyzer_id: &str) -> moe::SharedRankingKind {
    if analyzer_id.starts_with("micro") {
        moe::SharedRankingKind::MicroAnalyze
    } else {
        moe::SharedRankingKind::Analyze
    }
}

#[cfg(test)]
fn sibling_analysis_path(path: &Path) -> Option<PathBuf> {
    let parent = path.parent()?;
    let analysis = parent.join(ANALYSIS_JSON_FILENAME);
    analysis.exists().then_some(analysis)
}

fn model_display_name(model_path: &Path) -> String {
    if let Some(value) = model_path.file_stem().and_then(|value| value.to_str()) {
        value.to_string()
    } else {
        model_path.to_string_lossy().to_string()
    }
}

fn bundled_min_experts(model_name: &str) -> Option<u32> {
    let q = model_name.to_lowercase();
    catalog::MODEL_CATALOG
        .iter()
        .find(|model| model.name.to_lowercase() == q || model.file.to_lowercase().contains(&q))
        .and_then(|model| model.moe.as_ref().map(|cfg| cfg.min_experts_per_node))
}

fn load_analyze_mass_profile(path: &Path) -> Result<AnalyzeMassProfile> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Read ranking {}", path.display()))?;
    let mut ranking = Vec::new();
    let mut masses = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("expert") {
            continue;
        }
        let parts = trimmed.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let expert_id: u32 = parts[0]
            .parse()
            .with_context(|| format!("Parse expert id from {}", path.display()))?;
        let gate_mass: f64 = parts[1]
            .parse()
            .with_context(|| format!("Parse gate mass from {}", path.display()))?;
        ranking.push(expert_id);
        masses.push((expert_id, gate_mass));
    }
    if masses.is_empty() {
        bail!("ranking does not include gate-mass rows");
    }
    let total_mass = masses.iter().map(|(_, mass)| *mass).sum::<f64>();
    Ok(AnalyzeMassProfile {
        ranking,
        masses,
        total_mass,
    })
}

fn mass_pct_for_experts(profile: &AnalyzeMassProfile, experts: &[u32]) -> f64 {
    if profile.total_mass <= f64::EPSILON {
        return 0.0;
    }
    let mut total = 0.0;
    for expert in experts {
        if let Some((_, mass)) = profile
            .masses
            .iter()
            .find(|(candidate, _)| candidate == expert)
        {
            total += *mass;
        }
    }
    (total / profile.total_mass) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp_case_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mesh-llm-moe-planner-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn fake_ranking(path: &str, analyzer_id: &str, source: RankingSource) -> ResolvedRanking {
        ResolvedRanking {
            path: PathBuf::from(path),
            metadata_path: None,
            analysis_path: None,
            analyzer_id: analyzer_id.to_string(),
            source,
            reason: "test fixture".to_string(),
        }
    }

    #[test]
    fn normalize_distribution_id_strips_split_suffix() {
        assert_eq!(
            normalize_distribution_id("GLM-5.1-UD-IQ2_M-00001-of-00006.gguf"),
            "GLM-5.1-UD-IQ2_M"
        );
    }

    #[test]
    fn normalize_distribution_id_keeps_unsplit_name() {
        assert_eq!(
            normalize_distribution_id("gemma-4-26B-A4B-it-UD-Q4_K_S.gguf"),
            "gemma-4-26B-A4B-it-UD-Q4_K_S"
        );
    }

    #[test]
    fn infer_analyzer_from_ranking_path_supports_micro_and_full() {
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/a/micro-v1/ranking.csv")),
            Some("micro-v1")
        );
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/a/full-v1/ranking.csv")),
            Some("full-v1")
        );
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/local-demo.micro-p8-t128-all.csv")),
            Some("micro-v1")
        );
        assert_eq!(
            infer_analyzer_from_ranking_path(Path::new("/tmp/local-demo.csv")),
            Some("full-v1")
        );
    }

    #[test]
    fn local_submit_ranking_reads_materialized_package_artifacts() {
        let dir = temp_case_dir("submit-package");
        let previous_xdg = std::env::var_os("XDG_CACHE_HOME");
        std::env::set_var("XDG_CACHE_HOME", &dir);

        let model = MoeModelContext {
            input: "demo".to_string(),
            path: PathBuf::from(
                "/tmp/hf/models--unsloth--demo/snapshots/abcdef123456/demo-Q4_K_M.gguf",
            ),
            display_name: "demo.gguf".to_string(),
            source_repo: Some("unsloth/demo".to_string()),
            source_revision: Some("abcdef123456".to_string()),
            distribution_id: "demo".to_string(),
            expert_count: 8,
            used_expert_count: 2,
            min_experts_per_node: 4,
            total_model_bytes: 1024,
        };

        let ranking_path = moe::package_cache_ranking_path(&model.path);
        let analysis_path = moe::package_cache_analysis_path(&model.path);
        fs::create_dir_all(ranking_path.parent().unwrap()).unwrap();
        fs::write(&ranking_path, "0\n1\n").unwrap();
        fs::write(
            &analysis_path,
            serde_json::json!({ "analyzer_id": "micro-v1" }).to_string() + "\n",
        )
        .unwrap();

        let resolved = local_submit_ranking(&model).unwrap();
        assert_eq!(resolved.analyzer_id, "micro-v1");
        assert!(resolved.metadata_path.is_none());
        assert_eq!(
            resolved.analysis_path.as_deref(),
            Some(analysis_path.as_path())
        );
        if let Some(previous) = previous_xdg {
            std::env::set_var("XDG_CACHE_HOME", previous);
        } else {
            std::env::remove_var("XDG_CACHE_HOME");
        }
        let _ = fs::remove_dir_all(&dir);
    }

    fn write_test_ranking(path: &Path, rows: &[(u32, f64)]) {
        let mut content = String::from("expert_id,total_mass,mass_fraction,selection_count\n");
        for (expert_id, mass) in rows {
            content.push_str(&format!("{expert_id},{mass},0.0,1\n"));
        }
        fs::write(path, content).unwrap();
    }

    fn align_offset(offset: u64, alignment: u32) -> u64 {
        let alignment = u64::from(alignment.max(1));
        ((offset + alignment - 1) / alignment) * alignment
    }

    fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_tensor_info(bytes: &mut Vec<u8>, name: &str, offset: u64) {
        push_gguf_string(bytes, name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&16u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }

    fn write_test_gguf(path: &Path, expert_count: u32, expert_used_count: u32) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2i64.to_le_bytes());
        bytes.extend_from_slice(&3i64.to_le_bytes());

        push_gguf_string(&mut bytes, "general.alignment");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&32u32.to_le_bytes());

        push_gguf_string(&mut bytes, "llama.expert_count");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&expert_count.to_le_bytes());

        push_gguf_string(&mut bytes, "llama.expert_used_count");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&expert_used_count.to_le_bytes());

        push_tensor_info(&mut bytes, "blk.0.ffn_up_exps.weight", 0);
        push_tensor_info(&mut bytes, "blk.0.attn_q.weight", 64);

        let data_start = align_offset(bytes.len() as u64, 32) as usize;
        bytes.resize(data_start, 0);
        bytes.resize(data_start + 96, 0);

        let mut file = fs::File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
    }

    fn test_analysis(full_model_bytes: u64, base_resident_bytes: u64) -> MoeAnalysisJson {
        MoeAnalysisJson {
            schema_version: 1,
            ranking: MoeAnalysisRanking {
                sha256: "sha256:test".to_string(),
                rows: 4,
            },
            model: MoeAnalysisModel {
                expert_count: 4,
                expert_used_count: 2,
            },
            memory: MoeAnalysisMemory {
                full_model_bytes,
                base_resident_bytes,
                shard_file_overhead_bytes: 100_000_000,
                expert_tensor_bytes_total: 4_000_000_000,
                expert_bytes: MoeAnalysisExpertBytes::Uniform {
                    bytes_per_expert: 1_000_000_000,
                },
            },
        }
    }

    #[test]
    fn estimate_startup_fit_reports_full_model_fit() {
        let dir = temp_case_dir("startup-fit-full");
        let ranking_path = dir.join("ranking.csv");
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);

        let estimate = estimate_startup_fit_from_analysis(
            &ranking_path,
            &test_analysis(8_000_000_000, 2_000_000_000),
            9_000_000_000,
            "full-v1",
            "Hugging Face dataset",
        )
        .unwrap();

        assert!(estimate.full_model_fits());
        assert_eq!(estimate.recommended_nodes, Some(1));
        assert_eq!(estimate.predicted_max_shard_bytes, None);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn estimate_startup_fit_reports_split_fit_when_full_model_is_too_large() {
        let dir = temp_case_dir("startup-fit-split");
        let ranking_path = dir.join("ranking.csv");
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);

        let estimate = estimate_startup_fit_from_analysis(
            &ranking_path,
            &test_analysis(10_000_000_000, 500_000_000),
            4_000_000_000,
            "full-v1",
            "Hugging Face dataset",
        )
        .unwrap();

        assert!(!estimate.full_model_fits());
        assert!(estimate.shard_plan_fits());
        assert_eq!(estimate.recommended_nodes, Some(2));
        assert_eq!(estimate.predicted_max_shard_bytes, Some(3_600_000_000));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn estimate_startup_fit_reports_no_viable_fit() {
        let dir = temp_case_dir("startup-fit-none");
        let ranking_path = dir.join("ranking.csv");
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);

        let estimate = estimate_startup_fit_from_analysis(
            &ranking_path,
            &test_analysis(10_000_000_000, 500_000_000),
            3_800_000_000,
            "full-v1",
            "Hugging Face dataset",
        )
        .unwrap();

        assert!(!estimate.any_fit_exists());
        assert_eq!(estimate.recommended_nodes, None);
        assert_eq!(
            estimate.predicted_max_shard_launch_bytes,
            Some(3_960_000_000)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn select_preferred_ranking_prefers_full_over_micro() {
        let local = fake_ranking("/tmp/local.csv", "micro-v1", RankingSource::LocalCache);
        let remote = fake_ranking(
            "/tmp/remote.csv",
            "full-v1",
            RankingSource::HuggingFacePackage,
        );

        let selected = select_preferred_ranking(Some(local), Some(remote)).unwrap();
        assert_eq!(selected.analyzer_id, "full-v1");
        assert!(matches!(selected.source, RankingSource::HuggingFacePackage));
    }

    #[test]
    fn select_preferred_ranking_prefers_local_when_methods_match() {
        let local = fake_ranking("/tmp/local.csv", "micro-v1", RankingSource::LocalCache);
        let remote = fake_ranking(
            "/tmp/remote.csv",
            "micro-v1",
            RankingSource::HuggingFacePackage,
        );

        let selected = select_preferred_ranking(Some(local), Some(remote)).unwrap();
        assert!(matches!(selected.source, RankingSource::LocalCache));
    }

    #[test]
    fn build_submit_bundle_uses_canonical_dataset_layout() {
        let dir = temp_case_dir("submit-bundle");
        let analyzer_dir = dir.join("full-v1");
        let ranking_path = analyzer_dir.join("ranking.csv");
        let analysis_path = analyzer_dir.join("analysis.json");
        let log_path = analyzer_dir.join("run.log");
        fs::create_dir_all(&analyzer_dir).unwrap();
        write_test_ranking(&ranking_path, &[(0, 4.0), (1, 3.0), (2, 2.0), (3, 1.0)]);
        fs::write(&analysis_path, "{}\n").unwrap();
        fs::write(&log_path, "ok\n").unwrap();

        let model_file = dir.join("gemma-4-26B-A4B-it-UD-Q4_K_S.gguf");
        write_test_gguf(&model_file, 64, 4);
        let model = MoeModelContext {
            input: "unsloth/gemma".to_string(),
            path: model_file,
            display_name: "gemma-4-26B-A4B-it-UD-Q4_K_S.gguf".to_string(),
            source_repo: Some("unsloth/gemma-4-26B-A4B-it-GGUF".to_string()),
            source_revision: Some("9c718328e1620e7280a93e1a809e805e0f3e4839".to_string()),
            distribution_id: "gemma-4-26B-A4B-it-UD-Q4_K_S".to_string(),
            expert_count: 64,
            used_expert_count: 4,
            min_experts_per_node: 32,
            total_model_bytes: 123,
        };
        let ranking = ResolvedRanking {
            path: ranking_path,
            metadata_path: None,
            analysis_path: Some(analysis_path),
            analyzer_id: "full-v1".to_string(),
            source: RankingSource::LocalCache,
            reason: "test".to_string(),
        };

        let bundle = build_submit_bundle(&model, &ranking, Some(&log_path)).unwrap();
        assert_eq!(bundle.model_ref, "unsloth/gemma-4-26B-A4B-it-GGUF:Q4_K_S");
        assert_eq!(bundle.variant, "Q4_K_S");
        assert_eq!(bundle.variant_root, "variants/Q4_K_S");
        assert_eq!(bundle.ranking_repo_path, "variants/Q4_K_S/ranking.csv");
        assert_eq!(bundle.analysis_repo_path, "variants/Q4_K_S/analysis.json");
        assert_eq!(bundle.manifest_repo_path, "variants/Q4_K_S/manifest.json");
        assert_eq!(
            bundle.log_repo_path.as_deref(),
            Some("variants/Q4_K_S/run.log")
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn default_package_repo_name_uses_source_repo_name() {
        let model = MoeModelContext {
            input: "demo".to_string(),
            path: PathBuf::from("/tmp/model.gguf"),
            display_name: "demo".to_string(),
            source_repo: Some("unsloth/Qwen3.6-35B-A3B-GGUF".to_string()),
            source_revision: Some("deadbeef".to_string()),
            distribution_id: "Qwen3.6-35B-A3B-UD-Q4_K_XL".to_string(),
            expert_count: 8,
            used_expert_count: 2,
            min_experts_per_node: 4,
            total_model_bytes: 1,
        };
        assert_eq!(
            default_package_repo_name_for_model(&model).unwrap(),
            "qwen3.6-35b-a3b-gguf-moe"
        );
    }

    #[test]
    fn sort_catalog_package_pointers_prefers_canonical_before_community() {
        let mut packages = vec![
            crate::models::catalog::CatalogPackagePointer {
                package_repo: "alice/demo".to_string(),
                package_revision: "bbbb".to_string(),
                publisher: "alice".to_string(),
                trust: "community".to_string(),
            },
            crate::models::catalog::CatalogPackagePointer {
                package_repo: "meshllm/demo".to_string(),
                package_revision: "aaaa".to_string(),
                publisher: "meshllm".to_string(),
                trust: "canonical".to_string(),
            },
        ];
        sort_catalog_package_pointers(&mut packages);
        assert_eq!(packages[0].trust, "canonical");
        assert_eq!(packages[0].package_repo, "meshllm/demo");
    }

    #[test]
    fn sort_catalog_package_pointers_is_stable_for_multiple_community_entries() {
        let mut packages = vec![
            crate::models::catalog::CatalogPackagePointer {
                package_repo: "zoe/demo".to_string(),
                package_revision: "2222".to_string(),
                publisher: "zoe".to_string(),
                trust: "community".to_string(),
            },
            crate::models::catalog::CatalogPackagePointer {
                package_repo: "alice/demo".to_string(),
                package_revision: "3333".to_string(),
                publisher: "alice".to_string(),
                trust: "community".to_string(),
            },
            crate::models::catalog::CatalogPackagePointer {
                package_repo: "alice/demo".to_string(),
                package_revision: "1111".to_string(),
                publisher: "alice".to_string(),
                trust: "community".to_string(),
            },
        ];
        sort_catalog_package_pointers(&mut packages);
        assert_eq!(packages[0].publisher, "alice");
        assert_eq!(packages[0].package_revision, "1111");
        assert_eq!(packages[1].publisher, "alice");
        assert_eq!(packages[1].package_revision, "3333");
        assert_eq!(packages[2].publisher, "zoe");
    }

    #[test]
    fn join_repo_relative_resolves_paths_from_manifest_parent() {
        assert_eq!(
            join_repo_relative("variants/Q4_K_XL/manifest.json", "trunk.gguf"),
            "variants/Q4_K_XL/trunk.gguf"
        );
        assert_eq!(
            join_repo_relative("variants/Q4_K_XL/manifest.json", "experts/expert-000.gguf"),
            "variants/Q4_K_XL/experts/expert-000.gguf"
        );
    }
}
