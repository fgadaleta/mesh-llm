use super::*;

type AnalysisDetails = (Option<&'static str>, Option<usize>, u32, u32, bool);

pub(crate) fn build_submit_bundle(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    log_path: Option<&Path>,
) -> Result<MoeSubmitBundle> {
    let variant = variant_key_for_model(model);
    let variant_root = format!("variants/{variant}");
    let ranking_rel = format!("{variant_root}/ranking.csv");
    let analysis_rel = format!("{variant_root}/{ANALYSIS_JSON_FILENAME}");
    let manifest_rel = format!("{variant_root}/manifest.json");
    let log_rel = format!("{variant_root}/run.log");

    let log_path = log_path.filter(|path| path.exists()).map(Path::to_path_buf);
    let model_ref = canonical_model_ref_for_model(model)?;
    Ok(MoeSubmitBundle {
        model_ref,
        variant,
        variant_root,
        ranking_repo_path: ranking_rel,
        analysis_repo_path: analysis_rel,
        manifest_repo_path: manifest_rel,
        log_repo_path: log_path.as_ref().map(|_| log_rel),
        commit_message: format!(
            "Publish {} {} package",
            model.distribution_id, ranking.analyzer_id,
        ),
        commit_description: format!(
            "Publish {} Mesh package artifacts for {} ({})",
            ranking.analyzer_id, model.display_name, model.input
        ),
    })
}

pub(crate) fn build_meshllm_descriptor(
    existing: Option<&str>,
    model: &MoeModelContext,
    manifest_path: &str,
) -> Result<MeshllmPackageJson> {
    let source_repo = model.source_repo.clone().ok_or_else(|| {
        anyhow::anyhow!("Package publishing requires a Hugging Face-backed model")
    })?;
    let source_revision = model.source_revision.clone().ok_or_else(|| {
        anyhow::anyhow!("Package publishing requires a resolved Hugging Face source revision")
    })?;
    let variant = variant_key_for_model(model);
    let mut meshllm = if let Some(existing) = existing {
        let parsed: MeshllmPackageJson =
            serde_json::from_str(existing).context("Parse existing meshllm.json")?;
        if parsed.source.repo != source_repo {
            bail!(
                "Existing meshllm.json source repo {} does not match {}",
                parsed.source.repo,
                source_repo
            );
        }
        parsed
    } else {
        MeshllmPackageJson {
            schema_version: 1,
            source: MeshllmPackageSource {
                repo: source_repo.clone(),
                revision: source_revision.clone(),
            },
            variants: BTreeMap::new(),
        }
    };
    meshllm.source.revision = source_revision;
    meshllm.variants.insert(
        variant,
        MeshllmPackageVariant {
            distribution_id: model.distribution_id.clone(),
            manifest: manifest_path.to_string(),
        },
    );
    Ok(meshllm)
}

fn build_package_analysis_json(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    model_files: &[(String, PathBuf)],
) -> Result<Value> {
    let (prompt_set, prompt_count, token_count, context_size, all_layers) =
        infer_analysis_details(ranking, model)?;
    let mass_profile = load_analyze_mass_profile(&ranking.path)?;
    let tensor_profile = aggregate_tensor_byte_profile(model_files)?;

    Ok(json!({
        "schema_version": 1,
        "analyzer_id": ranking.analyzer_id,
        "created_at": iso8601_now(),
        "tool": {
            "name": "llama-moe-analyze",
            "version": "mesh-llm-fork"
        },
        "parameters": {
            "prompt_set": prompt_set,
            "prompt_count": prompt_count,
            "token_count": token_count,
            "context_size": context_size,
            "all_layers": all_layers
        },
        "ranking": {
            "sha256": sha256_file(&ranking.path)?,
            "rows": mass_profile.ranking.len(),
            "mass_checkpoints": build_mass_checkpoints(&mass_profile),
        },
        "summary": {
            "n_expert": model.expert_count,
            "n_expert_used": model.used_expert_count,
            "min_experts_per_node": model.min_experts_per_node
        },
        "memory": {
            "full_model_bytes": tensor_profile.full_model_bytes,
            "base_resident_bytes": tensor_profile.base_resident_bytes,
            "shard_file_overhead_bytes": tensor_profile.file_overhead_bytes,
            "expert_tensor_bytes_total": tensor_profile.expert_tensor_bytes,
            "expert_bytes": expert_bytes_json(tensor_profile.expert_tensor_bytes, model.expert_count)?,
        },
        "planner": {
            "recommended_overlap": 1,
        },
        "artifacts": {
            "ranking": "ranking.csv",
            "manifest": "manifest.json"
        }
    }))
}

fn build_analysis_json(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    model_files: &[(String, PathBuf)],
) -> Result<Value> {
    let mass_profile = load_analyze_mass_profile(&ranking.path)?;
    let tensor_profile = aggregate_tensor_byte_profile(model_files)?;
    Ok(json!({
        "schema_version": 1,
        "ranking": {
            "sha256": sha256_file(&ranking.path)?,
            "rows": mass_profile.ranking.len(),
            "mass_checkpoints": build_mass_checkpoints(&mass_profile),
        },
        "model": {
            "expert_count": model.expert_count,
            "expert_used_count": model.used_expert_count,
        },
        "memory": {
            "full_model_bytes": tensor_profile.full_model_bytes,
            "base_resident_bytes": tensor_profile.base_resident_bytes,
            "shard_file_overhead_bytes": tensor_profile.file_overhead_bytes,
            "expert_tensor_bytes_total": tensor_profile.expert_tensor_bytes,
            "expert_bytes": expert_bytes_json(tensor_profile.expert_tensor_bytes, model.expert_count)?,
        }
    }))
}

pub(crate) fn default_package_repo_name_for_model(model: &MoeModelContext) -> Result<String> {
    let Some(source_repo) = model.source_repo.as_ref() else {
        bail!("A Hugging Face-backed model is required to derive the package repo name.");
    };
    let repo_name = source_repo
        .split_once('/')
        .map(|(_, repo)| repo)
        .unwrap_or(source_repo);
    Ok(format!("{}-moe", sanitize_repo_name_component(repo_name)))
}

pub(crate) fn canonical_model_ref_for_model(model: &MoeModelContext) -> Result<String> {
    let Some(source_repo) = model.source_repo.as_ref() else {
        bail!("A Hugging Face-backed model is required to derive the canonical model ref.");
    };
    Ok(canonical_model_ref_from_source(
        source_repo,
        &model.distribution_id,
        Some(
            crate::models::huggingface_identity_for_path(&model.path)
                .map(|identity| identity.file)
                .or_else(|| {
                    model
                        .path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .map(ToOwned::to_owned)
                })
                .unwrap_or_else(|| model.display_name.clone()),
        ),
    ))
}

pub(crate) fn canonical_model_ref_from_source(
    source_repo: &str,
    distribution_id: &str,
    file_hint: Option<String>,
) -> String {
    let variant = file_hint
        .as_deref()
        .and_then(infer_variant_key_from_gguf_file)
        .unwrap_or_else(|| normalize_variant_selector(distribution_id));
    format!("{source_repo}:{variant}")
}

pub(crate) fn variant_key_for_model(model: &MoeModelContext) -> String {
    let candidate = crate::models::huggingface_identity_for_path(&model.path)
        .map(|identity| identity.file)
        .or_else(|| {
            model
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| model.display_name.clone());
    infer_variant_key_from_gguf_file(&candidate).unwrap_or_else(|| model.distribution_id.clone())
}

pub(crate) fn sanitize_repo_name_component(input: &str) -> String {
    let sanitized = input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    let trimmed = sanitized.trim_matches('-');
    if trimmed.is_empty() {
        "mesh-llm-moe".to_string()
    } else {
        trimmed.to_string()
    }
}

pub(crate) fn infer_variant_key_from_gguf_file(file: &str) -> Option<String> {
    if !file.ends_with(".gguf") {
        return None;
    }
    if let Some((prefix, _)) = file.split_once('/') {
        if is_quant_like_selector(prefix) {
            return Some(normalize_variant_selector(prefix));
        }
    }
    let basename = Path::new(file).file_name()?.to_str()?;
    let mut stem = basename.strip_suffix(".gguf")?;
    if let Some((base, shard)) = stem.rsplit_once("-00001-of-") {
        if shard.len() == 5 && shard.chars().all(|ch| ch.is_ascii_digit()) {
            stem = base;
        }
    }
    for marker in [
        "-UD-", ".UD-", "-IQ", ".IQ", "-Q", ".Q", "-BF16", ".BF16", "-F16", ".F16", "-F32", ".F32",
    ] {
        if let Some(pos) = stem.rfind(marker) {
            return Some(normalize_variant_selector(&stem[pos + 1..]));
        }
    }
    None
}

pub(crate) fn normalize_variant_selector(selector: &str) -> String {
    selector.strip_prefix("UD-").unwrap_or(selector).to_string()
}

fn is_quant_like_selector(value: &str) -> bool {
    let upper = value.to_ascii_uppercase();
    upper.starts_with("UD-")
        || upper.starts_with("Q")
        || upper.starts_with("IQ")
        || upper == "BF16"
        || upper == "F16"
        || upper == "F32"
}

pub(crate) fn read_analysis_json(path: &Path) -> Result<MoeAnalysisJson> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Read analysis {}", path.display()))?;
    let parsed: MoeAnalysisJson = serde_json::from_str(&content)
        .with_context(|| format!("Parse analysis {}", path.display()))?;
    if parsed.schema_version != 1 {
        bail!(
            "Unsupported analysis schema {} in {}",
            parsed.schema_version,
            path.display()
        );
    }
    Ok(parsed)
}

pub(crate) fn write_analysis_json(
    model: &MoeModelContext,
    ranking_path: &Path,
    analyzer_id: &str,
) -> Result<PathBuf> {
    let ranking = ResolvedRanking {
        path: ranking_path.to_path_buf(),
        metadata_path: None,
        analysis_path: None,
        analyzer_id: analyzer_id.to_string(),
        source: RankingSource::LocalCache,
        reason: "local analysis artifact".to_string(),
    };
    let model_files = discover_distribution_files(model)?;
    let content =
        serde_json::to_string_pretty(&build_analysis_json(model, &ranking, &model_files)?)? + "\n";
    let analysis_path = ranking_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(ANALYSIS_JSON_FILENAME);
    if let Some(parent) = analysis_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(&analysis_path, content)
        .with_context(|| format!("Write {}", analysis_path.display()))?;
    Ok(analysis_path)
}

pub(crate) fn write_package_analysis_json(
    model: &MoeModelContext,
    ranking: &ResolvedRanking,
    output_path: &Path,
) -> Result<()> {
    let model_files = discover_distribution_files(model)?;
    let mut content_value = build_package_analysis_json(model, ranking, &model_files)?;
    let current_ranking_sha = content_value
        .get("ranking")
        .and_then(|value| value.get("sha256"))
        .and_then(|value| value.as_str())
        .map(ToOwned::to_owned);
    if let Ok(existing) = fs::read_to_string(output_path) {
        if let Ok(existing_value) = serde_json::from_str::<Value>(&existing) {
            let existing_ranking_sha = existing_value
                .get("ranking")
                .and_then(|value| value.get("sha256"))
                .and_then(|value| value.as_str());
            if current_ranking_sha.as_deref() == existing_ranking_sha {
                if let Some(benchmark) = existing_value.get("benchmark") {
                    content_value["benchmark"] = benchmark.clone();
                }
            }
        }
    }
    let content = serde_json::to_string_pretty(&content_value)? + "\n";
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(output_path, content).with_context(|| format!("Write {}", output_path.display()))
}

pub(crate) fn write_package_benchmark_json(output_path: &Path, benchmark: &Value) -> Result<()> {
    let mut content_value = if output_path.exists() {
        let existing = fs::read_to_string(output_path)
            .with_context(|| format!("Read {}", output_path.display()))?;
        serde_json::from_str::<Value>(&existing)
            .with_context(|| format!("Parse {}", output_path.display()))?
    } else {
        json!({
            "schema_version": 1,
        })
    };
    content_value["benchmark"] = benchmark.clone();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("Create {}", parent.display()))?;
    }
    fs::write(
        output_path,
        serde_json::to_string_pretty(&content_value)? + "\n",
    )
    .with_context(|| format!("Write {}", output_path.display()))
}

fn infer_analysis_details(
    ranking: &ResolvedRanking,
    model: &MoeModelContext,
) -> Result<AnalysisDetails> {
    let context_size_default = 4096;
    let log_path = analysis_log_path(model, &ranking.analyzer_id);
    let log_text = log_path
        .as_ref()
        .filter(|path| path.exists())
        .map(fs::read_to_string)
        .transpose()
        .with_context(|| "Read local MoE analysis log".to_string())?;

    let token_count = extract_first_arg_value(log_text.as_deref(), "-n")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or_else(|| {
            if ranking.analyzer_id.starts_with("micro") {
                128
            } else {
                32
            }
        });
    let context_size = extract_first_arg_value(log_text.as_deref(), "-c")
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(context_size_default);
    let all_layers = log_text
        .as_deref()
        .map(|text| text.contains("--all-layers"))
        .unwrap_or(true);

    if ranking.analyzer_id.starts_with("micro") {
        let prompt_count = log_text
            .as_deref()
            .map(|text| text.matches("[prompt ").count())
            .filter(|count| *count > 0)
            .unwrap_or(8);
        Ok((
            Some("meshllm-micro-v1"),
            Some(prompt_count),
            token_count,
            context_size,
            all_layers,
        ))
    } else {
        Ok((None, None, token_count, context_size, all_layers))
    }
}

fn extract_first_arg_value<'a>(text: Option<&'a str>, flag: &str) -> Option<&'a str> {
    let text = text?;
    let parts = text.split_whitespace().collect::<Vec<_>>();
    parts
        .windows(2)
        .find_map(|window| (window[0] == flag).then_some(window[1]))
}

pub(crate) fn analysis_log_path(model: &MoeModelContext, analyzer_id: &str) -> Option<PathBuf> {
    let stem = model
        .path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    Some(
        crate::models::mesh_llm_cache_dir()
            .join("moe")
            .join("logs")
            .join(format!("{stem}.{analyzer_id}.log")),
    )
}

fn discover_distribution_files(model: &MoeModelContext) -> Result<Vec<(String, PathBuf)>> {
    let Some(identity) = huggingface_identity_for_path(&model.path) else {
        return Ok(vec![(
            model
                .path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("model.gguf")
                .to_string(),
            model.path.clone(),
        )]);
    };

    let snapshot_root = snapshot_root_for_relative_file(&model.path, &identity.file)
        .unwrap_or_else(|| {
            model
                .path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let mut files = Vec::new();
    collect_distribution_files(
        &snapshot_root,
        &snapshot_root,
        &model.distribution_id,
        &mut files,
    )?;
    if files.is_empty() {
        files.push((identity.file.clone(), model.path.clone()));
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn snapshot_root_for_relative_file(path: &Path, relative_file: &str) -> Option<PathBuf> {
    let mut root = path.to_path_buf();
    for _ in Path::new(relative_file).components() {
        root = root.parent()?.to_path_buf();
    }
    Some(root)
}

fn collect_distribution_files(
    snapshot_root: &Path,
    current: &Path,
    distribution_id: &str,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<()> {
    for entry in fs::read_dir(current).with_context(|| format!("Read {}", current.display()))? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_distribution_files(snapshot_root, &path, distribution_id, files)?;
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("gguf") {
            continue;
        }
        let relative = path
            .strip_prefix(snapshot_root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if normalize_distribution_id(&relative) == distribution_id {
            files.push((relative, path));
        }
    }
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    let mut file = fs::File::open(path).with_context(|| format!("Open {}", path.display()))?;
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let read = std::io::Read::read(&mut file, &mut buf)
            .with_context(|| format!("Hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buf[..read]);
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn iso8601_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    chrono::DateTime::<chrono::Utc>::from_timestamp(now as i64, 0)
        .unwrap_or_else(chrono::Utc::now)
        .to_rfc3339()
}
