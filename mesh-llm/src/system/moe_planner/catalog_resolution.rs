use super::*;

pub(crate) fn resolve_runtime_ranking(
    model_path: &Path,
    catalog_repo_name: &str,
) -> Result<Option<ResolvedRanking>> {
    let local_legacy = resolve_local_runtime_ranking(model_path);

    let Some(identity) = huggingface_identity_for_path(model_path) else {
        return Ok(local_legacy);
    };

    if local_legacy
        .as_ref()
        .is_some_and(|ranking| ranking_method_priority(ranking) >= 2)
    {
        return Ok(local_legacy);
    }

    let model_ref = canonical_model_ref_from_source(
        &identity.repo_id,
        &normalize_distribution_id(&identity.local_file_name),
        Some(identity.local_file_name.clone()),
    );
    let remote = match fetch_remote_package_ranking(catalog_repo_name, &model_ref, false) {
        Ok(remote) => remote,
        Err(error) => return local_legacy.ok_or(error).map(Some),
    };
    Ok(select_preferred_ranking(local_legacy, remote))
}

fn resolve_local_runtime_ranking(model_path: &Path) -> Option<ResolvedRanking> {
    let path = moe::package_cache_ranking_path(model_path);
    if !path.exists() {
        return None;
    }
    let analysis_path = moe::package_cache_analysis_path(model_path);
    let analyzer_id =
        analyzer_id_from_analysis(&analysis_path).unwrap_or_else(|_| "full-v1".to_string());
    Some(ResolvedRanking {
        path,
        metadata_path: None,
        analysis_path: analysis_path.exists().then_some(analysis_path),
        analyzer_id,
        source: RankingSource::LocalCache,
        reason: "local materialized package ranking".to_string(),
    })
}

pub(crate) async fn resolve_best_ranking(
    model: &MoeModelContext,
    args: &MoePlanArgs,
) -> Result<ResolvedRanking> {
    let local_legacy = resolve_local_runtime_ranking(&model.path);

    let Some(source_repo) = model.source_repo.as_ref() else {
        return local_legacy.ok_or_else(|| {
            anyhow::anyhow!(
                "No published ranking lookup is possible for non-HF model {} and no local ranking cache exists.",
                model.display_name
            )
        });
    };
    if local_legacy
        .as_ref()
        .is_some_and(|ranking| ranking_method_priority(ranking) >= 2)
    {
        return Ok(local_legacy.expect("checked is_some above"));
    }

    let model_ref = canonical_model_ref_from_source(
        source_repo,
        &model.distribution_id,
        Some(model.distribution_id.clone()),
    );
    let local_fallback = local_legacy.clone();
    let mut remote_spinner = args.progress.then(|| {
        start_spinner(&format!(
            "Checking published MoE package for {}",
            model.display_name
        ))
    });
    if let Some(spinner) = remote_spinner.as_mut() {
        spinner.set_message(format!(
            "Fetching published MoE package ranking and analysis for {}",
            model.display_name
        ));
    }
    let remote_lookup = fetch_remote_package_ranking(&args.catalog_repo, &model_ref, args.progress);
    if let Some(spinner) = remote_spinner.as_mut() {
        spinner.finish();
    }

    let remote = match remote_lookup {
        Ok(remote) => remote,
        Err(error) => {
            return local_fallback.ok_or(error);
        }
    };
    select_preferred_ranking(local_legacy, remote).ok_or_else(|| {
        anyhow::anyhow!(
            "No published MoE package ranking found in {} for {}:{} and no local cache exists.",
            args.catalog_repo,
            source_repo,
            model.distribution_id
        )
    })
}

fn ranking_method_priority(ranking: &ResolvedRanking) -> u8 {
    if ranking.analyzer_id.starts_with("full") {
        2
    } else if ranking.analyzer_id.starts_with("micro") {
        1
    } else {
        0
    }
}

pub(crate) fn select_preferred_ranking(
    local: Option<ResolvedRanking>,
    remote: Option<ResolvedRanking>,
) -> Option<ResolvedRanking> {
    match (local, remote) {
        (Some(local), Some(remote)) => {
            if ranking_method_priority(&local) >= ranking_method_priority(&remote) {
                Some(local)
            } else {
                Some(remote)
            }
        }
        (Some(local), None) => Some(local),
        (None, Some(remote)) => Some(remote),
        (None, None) => None,
    }
}

pub(crate) fn catalog_entry_path_for_source_repo(source_repo: &str) -> Result<String> {
    let (source_owner, source_name) = source_repo
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("Invalid source repo {}", source_repo))?;
    anyhow::ensure!(
        !source_owner.is_empty()
            && !source_name.is_empty()
            && source_owner != "."
            && source_owner != ".."
            && source_name != "."
            && source_name != ".."
            && !source_name.contains('/'),
        "Invalid source repo {}",
        source_repo
    );
    Ok(format!("entries/{source_owner}/{source_name}.json"))
}

fn package_variant_for_model_ref(model_ref: &str) -> Result<&str> {
    model_ref
        .rsplit_once(':')
        .map(|(_, variant)| variant)
        .ok_or_else(|| anyhow::anyhow!("Invalid Mesh model ref {model_ref}"))
}

pub(crate) fn join_repo_relative(parent_file: &str, relative_path: &str) -> String {
    let relative = relative_path.trim_start_matches('/');
    let Some(parent) = Path::new(parent_file).parent() else {
        return relative.to_string();
    };
    let joined = parent.join(relative);
    joined.to_string_lossy().replace('\\', "/")
}

pub(crate) fn analyzer_id_from_analysis(analysis_path: &Path) -> Result<String> {
    let analysis_content = fs::read_to_string(analysis_path)
        .with_context(|| format!("Read {}", analysis_path.display()))?;
    let parsed: Value = serde_json::from_str(&analysis_content)
        .with_context(|| format!("Parse {}", analysis_path.display()))?;
    Ok(parsed
        .get("analyzer_id")
        .and_then(Value::as_str)
        .unwrap_or("full-v1")
        .to_string())
}

fn resolve_catalog_package_pointer(
    api: &hf_hub::HFClientSync,
    catalog_repo_name: &str,
    model_ref: &str,
) -> Result<Option<crate::models::catalog::CatalogPackagePointer>> {
    let (owner, name) = catalog_repo_name
        .split_once('/')
        .unwrap_or(("", catalog_repo_name));
    let catalog_repo = api.dataset(owner, name);
    let info = catalog_repo.info(
        &RepoInfoParams::builder()
            .revision(DEFAULT_DATASET_REVISION.to_string())
            .build(),
    )?;
    let hf_hub::RepoInfo::Dataset(info) = info else {
        bail!("Expected dataset repo info for {}", catalog_repo_name);
    };

    let dataset_revision = info.sha.as_deref().unwrap_or(DEFAULT_DATASET_REVISION);
    let (source_repo, variant) = model_ref
        .rsplit_once(':')
        .ok_or_else(|| anyhow::anyhow!("Invalid Mesh model ref {model_ref}"))?;
    let entry_path = catalog_entry_path_for_source_repo(source_repo)?;
    if !info
        .siblings
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .any(|entry| entry.rfilename == entry_path)
    {
        return Ok(None);
    }

    let downloaded = catalog_repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename(entry_path.clone())
                .revision(dataset_revision.to_string())
                .build(),
        )
        .with_context(|| format!("Download {}", entry_path))?;
    let content = fs::read_to_string(&downloaded)
        .with_context(|| format!("Read {}", downloaded.display()))?;
    let entry: crate::models::catalog::CatalogRepoEntry = serde_json::from_str(&content)
        .with_context(|| format!("Parse {}", downloaded.display()))?;
    let mut packages = entry
        .variants
        .get(variant)
        .map(|variant_entry| variant_entry.packages.clone())
        .unwrap_or_default();
    sort_catalog_package_pointers(&mut packages);
    Ok(packages.into_iter().next())
}

pub(crate) fn sort_catalog_package_pointers(
    packages: &mut [crate::models::catalog::CatalogPackagePointer],
) {
    packages.sort_by(compare_catalog_package_pointers);
}

fn compare_catalog_package_pointers(
    left: &crate::models::catalog::CatalogPackagePointer,
    right: &crate::models::catalog::CatalogPackagePointer,
) -> std::cmp::Ordering {
    catalog_package_trust_rank(&right.trust)
        .cmp(&catalog_package_trust_rank(&left.trust))
        .then_with(|| left.publisher.cmp(&right.publisher))
        .then_with(|| left.package_repo.cmp(&right.package_repo))
        .then_with(|| left.package_revision.cmp(&right.package_revision))
}

fn catalog_package_trust_rank(trust: &str) -> u8 {
    match trust {
        "canonical" => 2,
        "community" => 1,
        _ => 0,
    }
}

fn resolve_package_variant(
    api: &hf_hub::HFClientSync,
    package_repo: &str,
    package_revision: &str,
    model_ref: &str,
) -> Result<Option<(Vec<hf_hub::RepoSibling>, String, String)>> {
    let (owner, name) = package_repo.split_once('/').unwrap_or(("", package_repo));
    let repo = api.model(owner, name);
    let info = repo.info(
        &RepoInfoParams::builder()
            .revision(package_revision.to_string())
            .build(),
    )?;
    let hf_hub::RepoInfo::Model(info) = info else {
        bail!("Expected model repo info for {}", package_repo);
    };
    let resolved_revision = info.sha.unwrap_or_else(|| package_revision.to_string());
    let siblings = info.siblings.unwrap_or_default();
    if !siblings
        .iter()
        .any(|entry| entry.rfilename == "meshllm.json")
    {
        return Ok(None);
    }

    let meshllm_path = repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("meshllm.json".to_string())
                .revision(resolved_revision.clone())
                .build(),
        )
        .with_context(|| format!("Download meshllm.json from {}", package_repo))?;
    let meshllm_content = fs::read_to_string(&meshllm_path)
        .with_context(|| format!("Read {}", meshllm_path.display()))?;
    let meshllm: MeshllmPackageJson = serde_json::from_str(&meshllm_content)
        .with_context(|| format!("Parse {}", meshllm_path.display()))?;
    let variant = package_variant_for_model_ref(model_ref)?;
    let Some(variant_entry) = meshllm.variants.get(variant) else {
        return Ok(None);
    };
    Ok(Some((
        siblings,
        resolved_revision,
        variant_entry.manifest.clone(),
    )))
}

pub(crate) fn fetch_remote_package_ranking(
    catalog_repo_name: &str,
    model_ref: &str,
    progress: bool,
) -> Result<Option<ResolvedRanking>> {
    let catalog_repo_name = catalog_repo_name.to_string();
    let model_ref = model_ref.to_string();
    crate::models::run_hf_blocking(move || {
        let api =
            build_hf_api(progress).context("Build Hugging Face client for MoE package lookup")?;
        let Some(entry) = resolve_catalog_package_pointer(&api, &catalog_repo_name, &model_ref)?
        else {
            return Ok(None);
        };
        let Some((siblings, package_revision, manifest_path)) = resolve_package_variant(
            &api,
            &entry.package_repo,
            &entry.package_revision,
            &model_ref,
        )?
        else {
            return Ok(None);
        };

        let ranking_rel = join_repo_relative(&manifest_path, "ranking.csv");
        let analysis_rel = join_repo_relative(&manifest_path, ANALYSIS_JSON_FILENAME);
        if !(siblings.iter().any(|entry| entry.rfilename == ranking_rel)
            && siblings.iter().any(|entry| entry.rfilename == analysis_rel))
        {
            return Ok(None);
        }

        let (owner, name) = entry
            .package_repo
            .split_once('/')
            .unwrap_or(("", entry.package_repo.as_str()));
        let repo = api.model(owner, name);
        let ranking_path = repo
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(ranking_rel.clone())
                    .revision(package_revision.clone())
                    .build(),
            )
            .with_context(|| format!("Download {}", ranking_rel))?;
        let analysis_path = repo
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(analysis_rel.clone())
                    .revision(package_revision.clone())
                    .build(),
            )
            .with_context(|| format!("Download {}", analysis_rel))?;
        let analyzer_id = analyzer_id_from_analysis(&analysis_path)?;
        Ok(Some(ResolvedRanking {
            path: ranking_path,
            metadata_path: None,
            analysis_path: Some(analysis_path),
            analyzer_id,
            source: RankingSource::HuggingFacePackage,
            reason: format!("published package ranking in {}", entry.package_repo),
        }))
    })
}

pub(crate) fn fetch_remote_package_expert_components(
    catalog_repo_name: &str,
    model_ref: &str,
    ranking_sha256: &str,
    expected_experts: &[u32],
    progress: bool,
) -> Result<Option<ResolvedExpertComponents>> {
    let catalog_repo_name = catalog_repo_name.to_string();
    let model_ref = model_ref.to_string();
    let ranking_sha256 = ranking_sha256.to_string();
    let expected_experts = expected_experts.to_vec();
    crate::models::run_hf_blocking(move || {
        let api = build_hf_api(progress)
            .context("Build Hugging Face client for MoE expert component lookup")?;
        let Some(entry) = resolve_catalog_package_pointer(&api, &catalog_repo_name, &model_ref)?
        else {
            return Ok(None);
        };
        let Some((siblings, package_revision, manifest_rel)) = resolve_package_variant(
            &api,
            &entry.package_repo,
            &entry.package_revision,
            &model_ref,
        )?
        else {
            return Ok(None);
        };
        if !siblings
            .iter()
            .any(|sibling| sibling.rfilename == manifest_rel)
        {
            return Ok(None);
        }

        let (owner, name) = entry
            .package_repo
            .split_once('/')
            .unwrap_or(("", entry.package_repo.as_str()));
        let repo = api.model(owner, name);
        let manifest_path = repo
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(manifest_rel.clone())
                    .revision(package_revision.clone())
                    .build(),
            )
            .with_context(|| format!("Download {}", manifest_rel))?;
        let manifest_content = fs::read_to_string(&manifest_path)
            .with_context(|| format!("Read {}", manifest_path.display()))?;
        let manifest: MoePackageManifest = serde_json::from_str(&manifest_content)
            .with_context(|| format!("Parse {}", manifest_path.display()))?;
        if manifest.ranking_sha256 != ranking_sha256 {
            return Ok(None);
        }
        let prefix = Path::new(&manifest_rel)
            .parent()
            .map(|parent| parent.to_string_lossy().replace('\\', "/"))
            .unwrap_or_default();

        let trunk_rel = join_repo_relative(&manifest_rel, &manifest.trunk.path);
        let trunk_path = repo
            .download_file(
                &RepoDownloadFileParams::builder()
                    .filename(trunk_rel.clone())
                    .revision(package_revision.clone())
                    .build(),
            )
            .with_context(|| format!("Download {}", trunk_rel))?;
        if sha256_file(&trunk_path)? != manifest.trunk.sha256 {
            bail!("Hash mismatch for {}", trunk_rel);
        }

        let mut expert_paths = Vec::with_capacity(expected_experts.len());
        for expert_id in expected_experts {
            let Some(expert) = manifest
                .experts
                .iter()
                .find(|expert| expert.expert_id == Some(expert_id))
            else {
                bail!(
                    "Published manifest {} is missing expert {}",
                    manifest_rel,
                    expert_id
                );
            };
            let expert_rel = join_repo_relative(&manifest_rel, &expert.path);
            if !siblings
                .iter()
                .any(|sibling| sibling.rfilename == expert_rel)
            {
                bail!(
                    "Published package {} is missing required expert file {}",
                    entry.package_repo,
                    expert_rel
                );
            }
            let expert_path = repo
                .download_file(
                    &RepoDownloadFileParams::builder()
                        .filename(expert_rel.clone())
                        .revision(package_revision.clone())
                        .build(),
                )
                .with_context(|| format!("Download {}", expert_rel))?;
            if sha256_file(&expert_path)? != expert.sha256 {
                bail!("Hash mismatch for {}", expert_rel);
            }
            expert_paths.push(expert_path);
        }

        Ok(Some(ResolvedExpertComponents {
            prefix,
            trunk_path,
            expert_paths,
        }))
    })
}
