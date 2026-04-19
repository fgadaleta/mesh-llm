use anyhow::{bail, Context, Result};
use hf_hub::{
    AddSource, CommitInfo, CommitOperation, CreateRepoParams, FileProgress, FileStatus, HFClient,
    HFError, Progress, ProgressEvent, ProgressHandler, RepoCreateCommitParams, RepoInfo,
    RepoInfoParams, RepoListRefsParams, RepoType, UploadEvent, UploadPhase,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::cli::terminal_progress::{clear_stderr_line, start_spinner, SpinnerHandle};
use crate::inference::moe;
use crate::models;
use crate::system::moe_planner;

use super::{
    TempRootGuard, SHARE_REPO_READY_POLL_INTERVAL, SHARE_REPO_READY_TIMEOUT,
    SHARE_UPLOAD_BATCH_MAX_BYTES, SHARE_UPLOAD_BATCH_MAX_FILES, SHARE_UPLOAD_MAX_RETRIES,
    SHARE_UPLOAD_POLL_INTERVAL, SHARE_UPLOAD_STALL_TIMEOUT,
};

#[derive(Clone, Debug)]
pub(super) struct SharePublishTarget {
    pub(super) package_repo: String,
    pub(super) publisher: String,
    pub(super) trust: &'static str,
}

struct ShareUploadFileState {
    bytes_completed: u64,
    total_bytes: u64,
}

#[derive(Clone, Debug)]
pub(super) struct StagedUploadFile {
    pub(super) repo_path: String,
    pub(super) local_path: PathBuf,
    pub(super) size_bytes: u64,
}

#[derive(Clone, Debug)]
pub(super) struct ShareUploadBatch {
    pub(super) files: Vec<StagedUploadFile>,
    pub(super) total_bytes: u64,
}

struct ShareUploadProgressState {
    spinner: Option<SpinnerHandle>,
    phase: Option<UploadPhase>,
    overall_total_files: usize,
    overall_total_bytes: u64,
    prior_completed_files: usize,
    prior_completed_bytes: u64,
    current_batch_index: usize,
    total_batches: usize,
    current_batch_total_files: usize,
    current_batch_total_bytes: u64,
    batch_bytes_completed: u64,
    bytes_per_sec: Option<f64>,
    batch_transfer_bytes_completed: u64,
    batch_transfer_bytes: u64,
    transfer_bytes_per_sec: Option<f64>,
    completed_files: BTreeSet<String>,
    active_files: BTreeMap<String, ShareUploadFileState>,
    last_draw: Option<std::time::Instant>,
    last_progress_change: std::time::Instant,
    last_progress_snapshot: (u64, u64, usize),
}

pub(super) struct ShareUploadProgress {
    state: Mutex<ShareUploadProgressState>,
}

impl ShareUploadProgress {
    pub(super) fn new(overall_total_files: usize, overall_total_bytes: u64) -> Self {
        Self {
            state: Mutex::new(ShareUploadProgressState {
                spinner: None,
                phase: None,
                overall_total_files,
                overall_total_bytes,
                prior_completed_files: 0,
                prior_completed_bytes: 0,
                current_batch_index: 0,
                total_batches: 0,
                current_batch_total_files: 0,
                current_batch_total_bytes: 0,
                batch_bytes_completed: 0,
                bytes_per_sec: None,
                batch_transfer_bytes_completed: 0,
                batch_transfer_bytes: 0,
                transfer_bytes_per_sec: None,
                completed_files: BTreeSet::new(),
                active_files: BTreeMap::new(),
                last_draw: None,
                last_progress_change: std::time::Instant::now(),
                last_progress_snapshot: (0, 0, 0),
            }),
        }
    }

    pub(super) fn begin_batch(
        &self,
        batch_index: usize,
        total_batches: usize,
        prior_completed_files: usize,
        prior_completed_bytes: u64,
        batch: &ShareUploadBatch,
    ) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if let Some(mut spinner) = state.spinner.take() {
            spinner.finish();
        }
        state.phase = None;
        state.prior_completed_files = prior_completed_files;
        state.prior_completed_bytes = prior_completed_bytes;
        state.current_batch_index = batch_index;
        state.total_batches = total_batches;
        state.current_batch_total_files = batch.files.len();
        state.current_batch_total_bytes = batch.total_bytes;
        state.batch_bytes_completed = 0;
        state.bytes_per_sec = None;
        state.batch_transfer_bytes_completed = 0;
        state.batch_transfer_bytes = 0;
        state.transfer_bytes_per_sec = None;
        state.completed_files.clear();
        state.active_files.clear();
        state.last_draw = None;
        state.last_progress_change = std::time::Instant::now();
        state.last_progress_snapshot = (0, 0, 0);
    }

    fn transition_phase(state: &mut ShareUploadProgressState, phase: &UploadPhase) {
        if state.phase.as_ref() == Some(phase) {
            return;
        }
        if let Some(mut spinner) = state.spinner.take() {
            spinner.finish();
        }
        state.phase = Some(phase.clone());
        match phase {
            UploadPhase::Preparing => {
                state.spinner = Some(start_spinner(&format!(
                    "Preparing upload batch {}/{}",
                    state.current_batch_index, state.total_batches
                )));
            }
            UploadPhase::CheckingUploadMode => {
                state.spinner = Some(start_spinner(&format!(
                    "Checking upload mode for batch {}/{}",
                    state.current_batch_index, state.total_batches
                )));
            }
            UploadPhase::Uploading => {
                let _ = clear_stderr_line();
                eprintln!(
                    "⬆️ Uploading batch {}/{}...",
                    state.current_batch_index, state.total_batches
                );
            }
            UploadPhase::Committing => {
                let done = (state.prior_completed_files
                    + state
                        .current_batch_total_files
                        .min(state.completed_files.len()))
                .min(state.overall_total_files);
                state.spinner = Some(start_spinner(&format!(
                    "Creating contribution PR ({done}/{})",
                    state.overall_total_files
                )));
            }
        }
    }

    fn apply_file_progress(state: &mut ShareUploadProgressState, file: &FileProgress) {
        match file.status {
            FileStatus::Started | FileStatus::InProgress => {
                state.active_files.insert(
                    file.filename.clone(),
                    ShareUploadFileState {
                        bytes_completed: file.bytes_completed,
                        total_bytes: file.total_bytes,
                    },
                );
            }
            FileStatus::Complete => {
                state.completed_files.insert(file.filename.clone());
                state.active_files.remove(&file.filename);
            }
        }
    }

    fn draw(state: &mut ShareUploadProgressState, force: bool) {
        let now = std::time::Instant::now();
        if !force
            && state.last_draw.is_some_and(|last| {
                now.duration_since(last) < std::time::Duration::from_millis(700)
            })
        {
            return;
        }
        state.last_draw = Some(now);

        let batch_completed_files = if state.phase == Some(UploadPhase::Committing) {
            state.current_batch_total_files
        } else {
            state
                .completed_files
                .len()
                .min(state.current_batch_total_files)
        };
        let done =
            (state.prior_completed_files + batch_completed_files).min(state.overall_total_files);
        let percent = if state.overall_total_files == 0 {
            0.0
        } else {
            (done as f64 / state.overall_total_files as f64) * 100.0
        };
        let total_processed_bytes = state.prior_completed_bytes + state.batch_bytes_completed;
        let processing = if state.overall_total_bytes > 0 {
            format!(
                " processed {}/{}",
                format_share_bytes(total_processed_bytes),
                format_share_bytes(state.overall_total_bytes)
            )
        } else {
            String::new()
        };
        let transfer = if state.batch_transfer_bytes > 0 {
            format!(
                ", uploading {}/{}",
                format_share_bytes(state.batch_transfer_bytes_completed),
                format_share_bytes(state.batch_transfer_bytes)
            )
        } else {
            String::new()
        };
        let speed = state
            .transfer_bytes_per_sec
            .or(state.bytes_per_sec)
            .filter(|bytes_per_sec| *bytes_per_sec > 0.0)
            .map(|bytes_per_sec| format!(" at {}/s", format_share_bytes(bytes_per_sec as u64)))
            .unwrap_or_default();

        let _ = clear_stderr_line();
        eprintln!(
            "⬆️ Uploading batch {}/{} {:>5.1}% [{}/{} files]{}{}{}",
            state.current_batch_index,
            state.total_batches,
            percent,
            done,
            state.overall_total_files,
            processing,
            transfer,
            speed
        );

        let mut active_files: Vec<(&String, &ShareUploadFileState)> =
            state.active_files.iter().collect();
        active_files.sort_by(|(left_name, left_file), (right_name, right_file)| {
            right_file
                .bytes_completed
                .cmp(&left_file.bytes_completed)
                .then_with(|| right_file.total_bytes.cmp(&left_file.total_bytes))
                .then_with(|| left_name.cmp(right_name))
        });

        let active_count = active_files.len();
        for (name, file) in active_files.into_iter().take(8) {
            let file_percent = if file.total_bytes == 0 {
                0.0
            } else {
                (file.bytes_completed as f64 / file.total_bytes as f64) * 100.0
            };
            eprintln!(
                "   {} {:>5.1}% ({}/{})",
                display_upload_filename(name),
                file_percent,
                format_share_bytes(file.bytes_completed),
                format_share_bytes(file.total_bytes)
            );
        }
        if active_count > 8 {
            eprintln!("   … {} more tracked file(s)", active_count - 8);
        }
        let _ = std::io::stderr().flush();
    }

    fn note_progress_change(state: &mut ShareUploadProgressState) {
        let snapshot = (
            state.batch_bytes_completed,
            state.batch_transfer_bytes_completed,
            state.completed_files.len(),
        );
        if snapshot != state.last_progress_snapshot {
            state.last_progress_snapshot = snapshot;
            state.last_progress_change = std::time::Instant::now();
        }
    }

    pub(super) fn stall_message(&self, timeout: Duration) -> Option<String> {
        let Ok(state) = self.state.lock() else {
            return None;
        };
        if state.phase != Some(UploadPhase::Uploading) {
            return None;
        }
        let stalled_for = std::time::Instant::now().duration_since(state.last_progress_change);
        if stalled_for < timeout {
            return None;
        }
        let active = state
            .active_files
            .keys()
            .take(3)
            .map(|path| display_upload_filename(path))
            .collect::<Vec<_>>();
        let active_suffix = if active.is_empty() {
            String::new()
        } else {
            format!(" Active files: {}.", active.join(", "))
        };
        Some(format!(
            "upload batch {}/{} stalled for {} with no byte progress.{}",
            state.current_batch_index,
            state.total_batches,
            format_duration(stalled_for),
            active_suffix
        ))
    }
}

impl ProgressHandler for ShareUploadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Upload(event) = event else {
            return;
        };
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        match event {
            UploadEvent::Start { .. } => {}
            UploadEvent::Progress {
                phase,
                bytes_completed,
                total_bytes,
                bytes_per_sec,
                transfer_bytes_completed,
                transfer_bytes,
                transfer_bytes_per_sec,
                files,
            } => {
                Self::transition_phase(&mut state, phase);
                state.batch_bytes_completed = *bytes_completed;
                if *total_bytes > 0 {
                    state.current_batch_total_bytes = *total_bytes;
                }
                state.bytes_per_sec = *bytes_per_sec;
                state.batch_transfer_bytes_completed = *transfer_bytes_completed;
                if *transfer_bytes > 0 {
                    state.batch_transfer_bytes = *transfer_bytes;
                }
                state.transfer_bytes_per_sec = *transfer_bytes_per_sec;
                for file in files {
                    Self::apply_file_progress(&mut state, file);
                }
                Self::note_progress_change(&mut state);
                if *phase == UploadPhase::Uploading {
                    Self::draw(&mut state, false);
                }
            }
            UploadEvent::FileComplete { files, phase } => {
                Self::transition_phase(&mut state, phase);
                for name in files {
                    state.completed_files.insert(name.clone());
                    state.active_files.remove(name);
                }
                Self::note_progress_change(&mut state);
                if *phase == UploadPhase::Uploading {
                    Self::draw(&mut state, true);
                }
            }
            UploadEvent::Complete => {
                if let Some(mut spinner) = state.spinner.take() {
                    spinner.finish();
                }
                if state.current_batch_total_files > 0 {
                    let remaining: Vec<String> = state.active_files.keys().cloned().collect();
                    state.completed_files.extend(remaining);
                    state.active_files.clear();
                    state.batch_bytes_completed = state.current_batch_total_bytes;
                    state.batch_transfer_bytes_completed = state.batch_transfer_bytes;
                    Self::note_progress_change(&mut state);
                    Self::draw(&mut state, true);
                }
            }
        }
    }
}

impl Drop for ShareUploadProgress {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if let Some(mut spinner) = state.spinner.take() {
                spinner.finish();
            }
        }
    }
}

fn display_upload_filename(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string())
}

fn format_share_bytes(bytes: u64) -> String {
    const KB: f64 = 1_000.0;
    const MB: f64 = 1_000_000.0;
    const GB: f64 = 1_000_000_000.0;

    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / GB)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / MB)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / KB)
    } else {
        format!("{bytes}B")
    }
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

#[derive(Debug)]
pub(super) struct ShareBranchState {
    pub(super) head_commit: String,
    pub(super) uploaded_paths: BTreeSet<String>,
}

pub(super) async fn dataset_branch_head(
    dataset: &hf_hub::HFRepository,
    branch: &str,
) -> Result<Option<String>> {
    let refs = dataset
        .list_refs(&RepoListRefsParams::builder().build())
        .await?;
    Ok(refs
        .branches
        .into_iter()
        .find(|entry| entry.name == branch)
        .map(|entry| entry.target_commit))
}

pub(super) async fn load_share_branch_state(
    dataset: &hf_hub::HFRepository,
    branch: &str,
    staged_files: &[StagedUploadFile],
) -> Result<Option<ShareBranchState>> {
    let Some(head_commit) = dataset_branch_head(dataset, branch).await? else {
        return Ok(None);
    };
    let mut spinner = start_spinner(&format!("Checking existing files on {branch}"));
    let mut uploaded_paths = BTreeSet::new();
    for (index, file) in staged_files.iter().enumerate() {
        spinner.set_message(format!(
            "Checking branch files {}/{}",
            index + 1,
            staged_files.len()
        ));
        if dataset
            .file_exists(
                &hf_hub::RepoFileExistsParams::builder()
                    .filename(file.repo_path.clone())
                    .revision(branch.to_string())
                    .build(),
            )
            .await?
        {
            uploaded_paths.insert(file.repo_path.clone());
        }
    }
    spinner.finish();
    Ok(Some(ShareBranchState {
        head_commit,
        uploaded_paths,
    }))
}

pub(super) async fn upload_share_batch_with_retry(
    dataset: &hf_hub::HFRepository,
    branch: Option<&str>,
    parent_commit: Option<String>,
    batch: &ShareUploadBatch,
    progress: &Arc<ShareUploadProgress>,
    commit_message: String,
    commit_description: String,
    create_pr: bool,
) -> Result<CommitInfo> {
    let batch_paths = batch
        .files
        .iter()
        .map(|file| file.repo_path.clone())
        .collect::<BTreeSet<_>>();
    let operations = batch
        .files
        .iter()
        .map(|file| CommitOperation::Add {
            path_in_repo: file.repo_path.clone(),
            source: AddSource::File(file.local_path.clone()),
        })
        .collect::<Vec<_>>();
    let mut last_error = None;
    let mut current_parent_commit = parent_commit;

    for attempt in 1..=SHARE_UPLOAD_MAX_RETRIES {
        let progress_handler: Progress = Some(progress.clone());
        let params = if let Some(branch) = branch {
            let builder = RepoCreateCommitParams::builder()
                .operations(operations.clone())
                .commit_message(commit_message.clone())
                .commit_description(commit_description.clone())
                .revision(branch.to_string())
                .create_pr(create_pr)
                .progress(progress_handler);
            if let Some(parent_commit) = current_parent_commit.clone() {
                builder.parent_commit(parent_commit).build()
            } else {
                builder.build()
            }
        } else {
            let builder = RepoCreateCommitParams::builder()
                .operations(operations.clone())
                .commit_message(commit_message.clone())
                .commit_description(commit_description.clone())
                .create_pr(create_pr)
                .progress(progress_handler);
            if let Some(parent_commit) = current_parent_commit.clone() {
                builder.parent_commit(parent_commit).build()
            } else {
                builder.build()
            }
        };
        match create_commit_with_stall_monitor(dataset, &params, progress).await {
            Ok(commit) => return Ok(commit),
            Err(err) => {
                let repo_not_ready = err
                    .downcast_ref::<HFError>()
                    .is_some_and(|hf| matches!(hf, HFError::RepoNotFound { .. }));
                last_error = Some(err);
                if let Some(branch) = branch {
                    match load_share_branch_state(dataset, branch, &batch.files).await {
                        Ok(Some(state)) => {
                            current_parent_commit = Some(state.head_commit.clone());
                            if state.uploaded_paths == batch_paths {
                                return Ok(CommitInfo {
                                    commit_url: None,
                                    commit_message: Some(commit_message.clone()),
                                    commit_description: Some(commit_description.clone()),
                                    commit_oid: Some(state.head_commit),
                                    pr_url: None,
                                    pr_num: None,
                                });
                            }
                        }
                        Ok(None) => {}
                        Err(state_err)
                            if repo_not_ready
                                && state_err.downcast_ref::<HFError>().is_some_and(|hf| {
                                    matches!(hf, HFError::RepoNotFound { .. })
                                }) => {}
                        Err(state_err) => return Err(state_err),
                    }
                }
                if let Some(branch) = branch {
                    if let Ok(Some(head_commit)) = dataset_branch_head(dataset, branch).await {
                        current_parent_commit = Some(head_commit);
                    }
                }
                if repo_not_ready {
                    tokio::time::sleep(SHARE_REPO_READY_POLL_INTERVAL).await;
                }
                if attempt < SHARE_UPLOAD_MAX_RETRIES {
                    eprintln!(
                        "↻ Retrying upload batch to {} (attempt {}/{})",
                        branch.unwrap_or("default branch"),
                        attempt + 1,
                        SHARE_UPLOAD_MAX_RETRIES
                    );
                    if repo_not_ready {
                        eprintln!("   repository is still propagating on the Hub");
                    }
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("upload batch failed")))
}

async fn create_commit_with_stall_monitor(
    dataset: &hf_hub::HFRepository,
    params: &RepoCreateCommitParams,
    progress: &ShareUploadProgress,
) -> Result<CommitInfo> {
    let upload = dataset.create_commit(params);
    tokio::pin!(upload);
    loop {
        tokio::select! {
            result = &mut upload => return result.map_err(Into::into),
            _ = tokio::time::sleep(SHARE_UPLOAD_POLL_INTERVAL) => {
                if let Some(message) = progress.stall_message(SHARE_UPLOAD_STALL_TIMEOUT) {
                    bail!("{message}");
                }
            }
        }
    }
}

pub(super) fn build_upload_batches(files: &[StagedUploadFile]) -> Vec<ShareUploadBatch> {
    let mut batches = Vec::new();
    let mut current_files = Vec::new();
    let mut current_bytes = 0u64;

    for file in files.iter().cloned() {
        let would_overflow = !current_files.is_empty()
            && (current_files.len() >= SHARE_UPLOAD_BATCH_MAX_FILES
                || current_bytes.saturating_add(file.size_bytes) > SHARE_UPLOAD_BATCH_MAX_BYTES);
        if would_overflow {
            batches.push(ShareUploadBatch {
                files: current_files,
                total_bytes: current_bytes,
            });
            current_files = Vec::new();
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(file.size_bytes);
        current_files.push(file);
        if current_files.len() >= SHARE_UPLOAD_BATCH_MAX_FILES
            || current_bytes >= SHARE_UPLOAD_BATCH_MAX_BYTES
        {
            batches.push(ShareUploadBatch {
                files: current_files,
                total_bytes: current_bytes,
            });
            current_files = Vec::new();
            current_bytes = 0;
        }
    }

    if !current_files.is_empty() {
        batches.push(ShareUploadBatch {
            files: current_files,
            total_bytes: current_bytes,
        });
    }
    batches
}

pub(super) fn batch_commit_message(base: &str, batch_index: usize, total_batches: usize) -> String {
    if total_batches <= 1 {
        base.to_string()
    } else {
        format!("{base} (batch {batch_index}/{total_batches})")
    }
}

pub(super) fn batch_commit_description(
    base: &str,
    batch_index: usize,
    total_batches: usize,
) -> String {
    if total_batches <= 1 {
        base.to_string()
    } else {
        format!("{base}\n\nUpload batch {batch_index}/{total_batches}.")
    }
}

pub(super) fn share_branch_name(dataset_prefix: &str) -> String {
    let digest = hex::encode(Sha256::digest(dataset_prefix.as_bytes()));
    let hint = dataset_prefix
        .split('/')
        .rev()
        .take(3)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(sanitize_branch_component)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("-");
    format!("mesh-llm-moe-{}-{}", hint, &digest[..12])
}

pub(super) fn sanitize_branch_component(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

pub(super) fn staged_upload_file(
    temp_root: &Path,
    relative_path: &str,
) -> Result<StagedUploadFile> {
    let local_path = temp_root.join(relative_path);
    let size_bytes = fs::metadata(&local_path)
        .with_context(|| format!("Read metadata for staged {}", relative_path))?
        .len();
    Ok(StagedUploadFile {
        repo_path: relative_path.to_string(),
        local_path,
        size_bytes,
    })
}

pub(super) fn variant_component_paths(prefix: &str, expert_count: u32) -> Vec<String> {
    let mut paths = vec![
        format!("{prefix}/manifest.json"),
        format!("{prefix}/trunk.gguf"),
    ];
    for expert_id in 0..expert_count {
        paths.push(format!(
            "{prefix}/experts/{}",
            moe::expert_component_filename(expert_id, expert_count)
        ));
    }
    paths
}

pub(super) fn stage_variant_components(
    temp_root: &Path,
    model: &moe_planner::MoeModelContext,
    ranking: &moe_planner::ResolvedRanking,
    bundle: &moe_planner::MoeSubmitBundle,
    bin_dir: &Path,
) -> Result<Vec<String>> {
    let cached_manifest_path = moe::package_cache_manifest_path(&model.path);
    if cached_manifest_path.exists() {
        let cached_manifest_text = fs::read_to_string(&cached_manifest_path)
            .with_context(|| format!("Read {}", cached_manifest_path.display()))?;
        let cached_manifest: moe_planner::MoePackageManifest =
            serde_json::from_str(&cached_manifest_text)
                .with_context(|| format!("Parse {}", cached_manifest_path.display()))?;
        if cached_manifest.ranking_sha256 == moe_planner::sha256_file(&ranking.path)?
            && cached_manifest.n_expert == model.expert_count
            && cached_manifest.n_expert_used == model.used_expert_count
            && moe::component_trunk_path(&model.path).exists()
            && (0..model.expert_count).all(|expert_id| {
                moe::component_expert_path(&model.path, expert_id, model.expert_count).exists()
            })
        {
            println!(
                "   reusing local package cache from {}",
                moe::package_cache_variant_dir(&model.path).display()
            );
            let manifest_repo_path = temp_root.join(&bundle.manifest_repo_path);
            if let Some(parent) = manifest_repo_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(&cached_manifest_path, &manifest_repo_path)?;

            let trunk_repo_path = temp_root.join(format!("{}/trunk.gguf", bundle.variant_root));
            if let Some(parent) = trunk_repo_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(moe::component_trunk_path(&model.path), &trunk_repo_path)?;

            for expert_id in 0..model.expert_count {
                let filename = moe::expert_component_filename(expert_id, model.expert_count);
                let repo_path =
                    temp_root.join(format!("{}/experts/{filename}", bundle.variant_root));
                if let Some(parent) = repo_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::copy(
                    moe::component_expert_path(&model.path, expert_id, model.expert_count),
                    &repo_path,
                )?;
            }

            return Ok(variant_component_paths(
                &bundle.variant_root,
                model.expert_count,
            ));
        }
    }

    println!("   extracting trunk");
    let trunk_path = moe::component_trunk_path(&model.path);
    moe::run_extract_trunk(bin_dir, &model.path, &trunk_path)?;

    let mut spinner = start_spinner("Extracting expert components");
    let mut expert_files = Vec::with_capacity(model.expert_count as usize);
    for expert_id in 0..model.expert_count {
        spinner.set_message(format!(
            "Extracting expert {}/{}",
            expert_id + 1,
            model.expert_count
        ));
        let filename = moe::expert_component_filename(expert_id, model.expert_count);
        let output_path = moe::component_expert_path(&model.path, expert_id, model.expert_count);
        moe::run_extract_expert(bin_dir, &model.path, expert_id, &output_path)?;
        expert_files.push(moe::ExpertComponentFile {
            path: format!("experts/{filename}"),
            sha256: moe_planner::sha256_file(&output_path)?,
            expert_id: Some(expert_id),
        });
    }
    spinner.finish();

    let manifest = moe_planner::MoePackageManifest {
        schema_version: 1,
        format: "meshllm-moe-components".to_string(),
        ranking_sha256: moe_planner::sha256_file(&ranking.path)?,
        n_expert: model.expert_count,
        n_expert_used: model.used_expert_count,
        min_experts_per_node: model.min_experts_per_node,
        trunk: moe::ExpertComponentFile {
            path: "trunk.gguf".to_string(),
            sha256: moe_planner::sha256_file(&trunk_path)?,
            expert_id: None,
        },
        experts: expert_files,
    };
    if let Some(parent) = cached_manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        &cached_manifest_path,
        serde_json::to_string_pretty(&manifest)? + "\n",
    )
    .with_context(|| format!("Write {}", cached_manifest_path.display()))?;

    let manifest_repo_path = temp_root.join(&bundle.manifest_repo_path);
    if let Some(parent) = manifest_repo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&cached_manifest_path, &manifest_repo_path)?;

    let trunk_repo_path = temp_root.join(format!("{}/trunk.gguf", bundle.variant_root));
    if let Some(parent) = trunk_repo_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::copy(&trunk_path, &trunk_repo_path)?;

    for expert_id in 0..model.expert_count {
        let filename = moe::expert_component_filename(expert_id, model.expert_count);
        let repo_path = temp_root.join(format!("{}/experts/{filename}", bundle.variant_root));
        if let Some(parent) = repo_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(
            moe::component_expert_path(&model.path, expert_id, model.expert_count),
            &repo_path,
        )?;
    }
    Ok(variant_component_paths(
        &bundle.variant_root,
        model.expert_count,
    ))
}

pub(super) async fn resolve_publish_target(
    api: &HFClient,
    model: &moe_planner::MoeModelContext,
    namespace: Option<&str>,
) -> Result<SharePublishTarget> {
    let whoami = api.whoami().await.context("Fetch Hugging Face identity")?;
    let publisher = whoami.username;
    let repo_name = moe_planner::default_package_repo_name_for_model(model)?;

    if let Some(namespace) = namespace {
        let package_repo = format!("{namespace}/{repo_name}");
        ensure_repo_exists(api, &package_repo, RepoType::Model)
            .await
            .with_context(|| {
                format!(
                    "Access or create model repo {}. Check that your Hugging Face token can create or write model repos in namespace {}.",
                    package_repo, namespace
                )
            })?;
        return Ok(SharePublishTarget {
            package_repo,
            publisher,
            trust: if namespace == "meshllm" {
                "canonical"
            } else {
                "community"
            },
        });
    }

    let canonical_repo = format!("meshllm/{repo_name}");
    match ensure_repo_exists(api, &canonical_repo, RepoType::Model).await {
        Ok(()) => Ok(SharePublishTarget {
            package_repo: canonical_repo,
            publisher,
            trust: "canonical",
        }),
        Err(err)
            if err
                .downcast_ref::<HFError>()
                .is_some_and(|hf| matches!(hf, HFError::Forbidden | HFError::AuthRequired)) =>
        {
            let package_repo = format!("{}/{}", publisher, repo_name);
            eprintln!(
                "↪ No permission to publish in meshllm. Falling back to community package repo {}",
                package_repo
            );
            ensure_repo_exists(api, &package_repo, RepoType::Model)
                .await
                .with_context(|| {
                    format!(
                        "Access or create fallback model repo {}. Check that your Hugging Face token can create or write model repos in namespace {}.",
                        package_repo, publisher
                    )
                })?;
            Ok(SharePublishTarget {
                package_repo,
                publisher,
                trust: "community",
            })
        }
        Err(err) => Err(err.into()),
    }
}

async fn ensure_repo_exists(api: &HFClient, repo_id: &str, repo_type: RepoType) -> Result<()> {
    if repo_exists(api, repo_id, repo_type).await? {
        return Ok(());
    }

    match api
        .create_repo(
            &CreateRepoParams::builder()
                .repo_id(repo_id.to_string())
                .repo_type(repo_type)
                .exist_ok(true)
                .build(),
        )
        .await
    {
        Ok(_) => Ok(()),
        Err(err @ (HFError::Forbidden | HFError::AuthRequired)) => {
            if repo_exists(api, repo_id, repo_type).await? {
                Ok(())
            } else {
                Err(anyhow::Error::from(err))
            }
        }
        Err(err) => Err(err.into()),
    }
}

pub(super) async fn ensure_repo_ready(
    api: &HFClient,
    repo_id: &str,
    repo_type: RepoType,
) -> Result<()> {
    ensure_repo_exists(api, repo_id, repo_type).await?;
    let started = std::time::Instant::now();
    loop {
        match repo_visible_on_hub(repo_id, repo_type).await {
            Ok(true) => return Ok(()),
            Ok(false) => {
                if started.elapsed() >= SHARE_REPO_READY_TIMEOUT {
                    bail!(
                        "Repository {} still is not visible after {:.0}s",
                        repo_id,
                        SHARE_REPO_READY_TIMEOUT.as_secs_f64()
                    );
                }
            }
            Err(err) => return Err(err),
        }
        tokio::time::sleep(SHARE_REPO_READY_POLL_INTERVAL).await;
    }
}

async fn repo_visible_on_hub(repo_id: &str, repo_type: RepoType) -> Result<bool> {
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string());
    let base = endpoint.trim_end_matches('/');
    let repo_url = match repo_type {
        RepoType::Model => format!("{base}/api/models/{repo_id}"),
        RepoType::Dataset => format!("{base}/api/datasets/{repo_id}"),
        RepoType::Space => format!("{base}/api/spaces/{repo_id}"),
        RepoType::Kernel => {
            return Err(anyhow::anyhow!(
                "Kernel repositories are not supported for MoE package publication"
            ));
        }
    };

    let client = reqwest::Client::new();
    let mut request = client.get(&repo_url);
    if let Some(token) = models::hf_token_override() {
        request = request.bearer_auth(token);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("Check Hub visibility for {repo_id}"))?;

    let status = response.status();
    match status.as_u16() {
        200 => Ok(true),
        401 | 403 => Err(anyhow::anyhow!(
            "Not authorized to inspect {} on the Hub ({})",
            repo_id,
            status
        )),
        404 => Ok(false),
        _ => {
            let detail = response.text().await.unwrap_or_default();
            Err(anyhow::anyhow!(
                "Unexpected Hub response while checking {} visibility: {} {}",
                repo_id,
                status,
                detail
            ))
        }
    }
}

async fn repo_exists(api: &HFClient, repo_id: &str, repo_type: RepoType) -> Result<bool> {
    let (owner, name) = parse_repo_id(repo_id)?;
    let params = RepoInfoParams::builder()
        .revision("main".to_string())
        .build();
    let result = match repo_type {
        RepoType::Model => api.model(owner, name).info(&params).await,
        RepoType::Dataset => api.dataset(owner, name).info(&params).await,
        RepoType::Space => api.space(owner, name).info(&params).await,
        RepoType::Kernel => {
            return Err(anyhow::anyhow!(
                "Kernel repositories are not supported for MoE package publication"
            ));
        }
    };
    match result {
        Ok(_) => Ok(true),
        Err(HFError::RepoNotFound { .. }) => Ok(false),
        Err(HFError::RevisionNotFound { .. }) => Ok(true),
        Err(err) => Err(err.into()),
    }
}

pub(super) fn repo_info_siblings_and_sha(
    info: &RepoInfo,
) -> Result<(Vec<hf_hub::RepoSibling>, Option<String>)> {
    match info {
        RepoInfo::Model(info) => Ok((info.siblings.clone().unwrap_or_default(), info.sha.clone())),
        RepoInfo::Dataset(info) => {
            Ok((info.siblings.clone().unwrap_or_default(), info.sha.clone()))
        }
        RepoInfo::Space(info) => Ok((info.siblings.clone().unwrap_or_default(), info.sha.clone())),
    }
}

pub(super) async fn download_repo_text(
    repo: &hf_hub::HFRepository,
    path: &str,
    revision: &str,
) -> Result<String> {
    let downloaded = repo
        .download_file(
            &hf_hub::RepoDownloadFileParams::builder()
                .filename(path.to_string())
                .revision(revision.to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Download {}", path))?;
    fs::read_to_string(&downloaded).with_context(|| format!("Read {}", downloaded.display()))
}

pub(super) fn parse_repo_id(repo_id: &str) -> Result<(&str, &str)> {
    repo_id
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("Repository id must look like `owner/name`, got {repo_id}"))
}

pub(super) fn make_temp_root(prefix: &str) -> Result<PathBuf> {
    let temp_root = std::env::temp_dir().join(format!(
        "{}-{}-{}",
        prefix,
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    fs::create_dir_all(&temp_root)?;
    Ok(temp_root)
}

pub(super) fn stage_share_text(temp_root: &Path, relative_path: &str, content: &str) -> Result<()> {
    let target = temp_root.join(relative_path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(target, content).with_context(|| format!("Write staged {}", relative_path))?;
    Ok(())
}

pub(super) fn stage_share_file(temp_root: &Path, relative_path: &str, source: &Path) -> Result<()> {
    let target = temp_root.join(relative_path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)?;
    }
    if target.exists() {
        fs::remove_file(&target).with_context(|| format!("Remove staged {}", target.display()))?;
    }
    match fs::hard_link(source, &target) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, &target)
                .with_context(|| format!("Copy {} to {}", source.display(), target.display()))?;
            Ok(())
        }
    }
}

pub(super) async fn contribute_catalog_entry(
    api: &HFClient,
    catalog_repo: &str,
    repo_path: &str,
    variant: &str,
    source: crate::models::catalog::CatalogSource,
    package_pointer: crate::models::catalog::CatalogPackagePointer,
) -> Result<CommitInfo> {
    let (owner, name) = parse_repo_id(catalog_repo)?;
    let dataset = api.dataset(owner, name);
    let mut spinner = start_spinner(&format!("Opening catalog PR in {}", catalog_repo));
    let source_repo = source.repo.clone();
    let info = dataset
        .info(
            &RepoInfoParams::builder()
                .revision("main".to_string())
                .build(),
        )
        .await
        .with_context(|| format!("Fetch main branch info for {}", catalog_repo))?;
    let (siblings, main_head) = repo_info_siblings_and_sha(&info)?;
    let mut entry = if siblings
        .iter()
        .any(|sibling| sibling.rfilename == repo_path)
    {
        let existing = download_repo_text(&dataset, repo_path, "main")
            .await
            .with_context(|| format!("Download existing catalog entry {}", repo_path))?;
        let parsed: crate::models::catalog::CatalogRepoEntry = serde_json::from_str(&existing)
            .with_context(|| format!("Parse existing catalog entry {}", repo_path))?;
        anyhow::ensure!(
            parsed.source_repo == source.repo,
            "Catalog entry {} belongs to {}, expected {}",
            repo_path,
            parsed.source_repo,
            source.repo
        );
        parsed
    } else {
        crate::models::catalog::CatalogRepoEntry {
            schema_version: 1,
            source_repo: source.repo.clone(),
            variants: BTreeMap::new(),
        }
    };
    let variant_entry = entry
        .variants
        .entry(variant.to_string())
        .or_insert_with(|| crate::models::catalog::CatalogVariantEntry {
            source: source.clone(),
            curated: None,
            packages: Vec::new(),
        });
    variant_entry.source = source;
    variant_entry
        .packages
        .retain(|existing| existing.package_repo != package_pointer.package_repo);
    variant_entry.packages.push(package_pointer.clone());
    moe_planner::sort_catalog_package_pointers(&mut variant_entry.packages);
    let temp_root = TempRootGuard(make_temp_root("mesh-llm-catalog")?);
    stage_share_text(
        &temp_root.0,
        repo_path,
        &(serde_json::to_string_pretty(&entry)? + "\n"),
    )?;
    let builder = RepoCreateCommitParams::builder()
        .operations(vec![CommitOperation::Add {
            path_in_repo: repo_path.to_string(),
            source: AddSource::File(temp_root.0.join(repo_path)),
        }])
        .commit_message(format!(
            "Register {} package for {}:{}",
            package_pointer.trust, source_repo, variant
        ))
        .commit_description(format!(
            "Register `{}` as the {} package for `{}` variant `{}`.",
            package_pointer.package_repo, package_pointer.trust, source_repo, variant
        ))
        .revision("main".to_string())
        .create_pr(true);
    let params = if let Some(main_head) = main_head {
        builder.parent_commit(main_head).build()
    } else {
        builder.build()
    };
    let result = dataset
        .create_commit(&params)
        .await
        .map_err(anyhow::Error::from);
    spinner.finish();
    result
}
