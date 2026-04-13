mod formatters;
mod formatters_console;
mod formatters_json;

use crate::cli::models::{ModelsCommand, RecommendedCommand};
use crate::models::{
    catalog, download_exact_ref, find_catalog_model_exact, installed_model_capabilities,
    resolve_huggingface_model_identity, scan_installed_models, search_catalog_models,
    search_huggingface, show_exact_model, show_model_variants_with_progress, SearchArtifactFilter,
    SearchProgress, ShowVariantsProgress,
};
use anyhow::{anyhow, Context, Result};
use base64::Engine;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::json;
use std::io::{IsTerminal, Write};
use std::path::Path;
use std::time::Instant;

use formatters::{
    catalog_model_is_mlx, model_kind_code, models_formatter, search_formatter, InstalledRow,
};

pub async fn run_model_search(
    query: &[String],
    prefer_gguf: bool,
    prefer_mlx: bool,
    catalog_only: bool,
    limit: usize,
    json_output: bool,
) -> Result<()> {
    let formatter = search_formatter(json_output);
    let query = query.join(" ");
    let filter = if prefer_mlx {
        SearchArtifactFilter::Mlx
    } else if prefer_gguf {
        SearchArtifactFilter::Gguf
    } else {
        SearchArtifactFilter::Gguf
    };

    if catalog_only {
        let results: Vec<_> = search_catalog_models(&query)
            .into_iter()
            .filter(|model| match filter {
                SearchArtifactFilter::Gguf => !catalog_model_is_mlx(model),
                SearchArtifactFilter::Mlx => catalog_model_is_mlx(model),
            })
            .collect();
        if results.is_empty() {
            return formatter.render_catalog_empty(&query, filter);
        }
        return formatter.render_catalog_results(&query, filter, &results, limit);
    }

    if !formatter.is_json() {
        eprintln!(
            "🔎 Searching Hugging Face {} repos for '{}'...",
            formatters::filter_label(filter),
            query
        );
    }
    let mut announced_repo_scan = false;
    let results = search_huggingface(&query, limit, filter, |progress| match progress {
        SearchProgress::SearchingHub => {}
        SearchProgress::InspectingRepos { completed, total } => {
            if formatter.is_json() {
                return;
            }
            if total == 0 {
                return;
            }
            if !announced_repo_scan {
                announced_repo_scan = true;
                eprintln!("   Inspecting {total} candidate repos...");
            }
            if completed == 0 {
                return;
            }
            eprint!("\r   Inspected {completed}/{total} candidate repos...");
            let _ = std::io::stderr().flush();
            if completed == total {
                eprintln!();
            }
        }
    })
    .await?;
    if results.is_empty() {
        return formatter.render_hf_empty(&query, filter);
    }
    formatter.render_hf_results(&query, filter, &results)
}

pub fn run_model_recommended(json_output: bool) -> Result<()> {
    if !json_output {
        eprintln!(
            "🔎 Fetching recommended models from Hugging Face dataset {}...",
            catalog::DEFAULT_RECOMMENDED_MODELS_DATASET_REPO
        );
        catalog::preload_catalog_dataset_with_progress(
            catalog::DEFAULT_RECOMMENDED_MODELS_DATASET_REPO,
            |progress| match progress {
                catalog::CatalogLoadProgress::ListingEntries => {}
                catalog::CatalogLoadProgress::LoadingEntry { completed, total } => {
                    if total == 0 {
                        return;
                    }
                    eprint!("\r   Loaded {completed}/{total} recommended entries...");
                    let _ = std::io::stderr().flush();
                    if completed == total {
                        eprintln!();
                    }
                }
            },
        )?;
    }
    let formatter = models_formatter(json_output);
    let models: Vec<_> = catalog::MODEL_CATALOG.iter().collect();
    formatter.render_recommended(&models)
}

async fn run_model_recommended_share(
    model: &str,
    description: &str,
    name: Option<&str>,
    draft: Option<&str>,
    dataset_repo: &str,
    json_output: bool,
) -> Result<()> {
    let details = show_exact_model(model).await?;
    let identity = resolve_huggingface_model_identity(model)
        .await?
        .ok_or_else(|| anyhow!("Recommended models must resolve to a Hugging Face-backed model"))?;

    let catalog_model = build_recommended_catalog_model(
        model,
        description,
        name,
        draft,
        &identity,
        &details,
        fetch_recommended_repo_manifest(&identity).await?,
    );
    let metadata_path = catalog::dataset_metadata_path_for_model_id(model);
    let metadata_body = catalog::serialize_recommended_model_metadata(&catalog_model)?;
    let mut index_entries = catalog::load_catalog_index(dataset_repo).unwrap_or_default();
    if index_entries.iter().any(|entry| entry.id == model) {
        if json_output {
            formatters::print_json(json!({
                "status": "already_published",
                "dataset_repo": dataset_repo,
                "path": metadata_path,
                "id": model,
            }))?;
        } else {
            println!("✅ Already published");
            println!("   repo: {dataset_repo}");
            println!("   id: {model}");
        }
        return Ok(());
    }
    index_entries.push(catalog::build_catalog_index_entry(&catalog_model));
    let index_body = catalog::serialize_recommended_catalog_index(&index_entries)?;

    let api = crate::models::build_hf_api(false).context("Build Hugging Face client")?;
    let dataset = api.repo(hf_hub::Repo::with_revision(
        dataset_repo.to_string(),
        hf_hub::RepoType::Dataset,
        "main".to_string(),
    ));
    let info = dataset
        .info()
        .with_context(|| format!("Fetch dataset info for {}", dataset_repo))?;
    if info
        .siblings
        .iter()
        .any(|entry| entry.rfilename == metadata_path)
    {
        if json_output {
            formatters::print_json(json!({
                "status": "already_published",
                "dataset_repo": dataset_repo,
                "path": metadata_path,
                "id": model,
            }))?;
        } else {
            println!("✅ Already published");
            println!("   repo: {dataset_repo}");
            println!("   path: {metadata_path}");
        }
        return Ok(());
    }

    let token = crate::models::hf_token_override().ok_or_else(|| {
        anyhow!(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before running `mesh-llm models recommended share`."
        )
    })?;
    let endpoint = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".to_string());
    let commit_url = format!(
        "{}/api/datasets/{}/commit/main",
        endpoint.trim_end_matches('/'),
        dataset_repo
    );
    let commit_message = format!("Add recommended model {}", catalog_model.id);
    let commit_description = format!(
        "Publish recommended model entry for {} ({})",
        catalog_model.name, catalog_model.id
    );
    let body = vec![
        ndjson_header(&commit_message, &commit_description),
        ndjson_file_op(
            catalog::recommended_models_index_path(),
            index_body.as_bytes(),
        ),
        ndjson_file_op(&metadata_path, metadata_body.as_bytes()),
    ]
    .into_iter()
    .map(|value| serde_json::to_string(&value))
    .collect::<std::result::Result<Vec<_>, _>>()?
    .join("\n")
        + "\n";

    if !json_output {
        println!("📤 Recommended model share");
        println!("📦 {}", catalog_model.name);
        println!("   id: {}", catalog_model.id);
        println!("   source: {}", details.exact_ref);
        println!("☁️ Dataset contribution");
        println!("   repo: {dataset_repo}");
        println!("   path: {metadata_path}");
        println!("⬆️ Opening contribution PR...");
    }

    let response = reqwest::Client::new()
        .post(&commit_url)
        .bearer_auth(token)
        .query(&[("create_pr", "1")])
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send()
        .await
        .with_context(|| format!("POST {}", commit_url))?;
    if response.status() != StatusCode::OK {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(anyhow!(
            "Dataset contribution failed: {}: {}",
            status,
            body.trim()
        ));
    }
    let commit: HfCommitResponse = response
        .json()
        .await
        .context("Decode Hugging Face response")?;
    if json_output {
        formatters::print_json(json!({
            "status": "opened_pr",
            "dataset_repo": dataset_repo,
            "path": metadata_path,
            "id": model,
            "commit_oid": commit.commit_oid,
            "commit_url": commit.commit_url,
            "pull_request_url": commit.pull_request_url,
        }))?;
    } else {
        println!("✅ Opened recommended model contribution");
        println!("   commit: {}", commit.commit_oid);
        println!("   url: {}", commit.commit_url);
        if let Some(pr_url) = commit.pull_request_url.as_deref() {
            println!("   pr: {}", pr_url);
        }
    }
    Ok(())
}

fn build_recommended_catalog_model(
    input_id: &str,
    description: &str,
    name: Option<&str>,
    draft: Option<&str>,
    identity: &crate::models::local::HuggingFaceModelIdentity,
    details: &crate::models::ModelDetails,
    manifest: RecommendedRepoManifest,
) -> catalog::CatalogModel {
    let file = identity
        .file
        .rsplit('/')
        .next()
        .unwrap_or(&identity.file)
        .to_string();
    catalog::CatalogModel {
        id: input_id.to_string(),
        name: name
            .map(str::to_string)
            .unwrap_or_else(|| recommended_model_name_from_file(&file)),
        file: file.clone(),
        url: format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            identity.repo_id, identity.revision, identity.file
        ),
        primary_size_bytes: manifest.primary_size_bytes,
        size: details
            .size_label
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
        description: description.to_string(),
        draft: draft.map(str::to_string),
        moe: details.moe.clone(),
        extra_files: split_gguf_extra_files(identity, &manifest),
        mmproj: manifest.mmproj,
    }
}

#[derive(Clone, Debug, Default)]
struct RecommendedRepoManifest {
    primary_size_bytes: Option<u64>,
    mmproj: Option<catalog::CatalogAsset>,
    split_sizes: std::collections::HashMap<String, Option<u64>>,
}

async fn fetch_recommended_repo_manifest(
    identity: &crate::models::local::HuggingFaceModelIdentity,
) -> Result<RecommendedRepoManifest> {
    let api = crate::models::build_hf_tokio_api(false)?;
    let info = api
        .repo(hf_hub::Repo::with_revision(
            identity.repo_id.clone(),
            hf_hub::RepoType::Model,
            identity.revision.clone(),
        ))
        .info()
        .await
        .with_context(|| {
            format!(
                "Fetch Hugging Face repo {}@{}",
                identity.repo_id, identity.revision
            )
        })?;
    let primary_size_bytes = info
        .siblings
        .iter()
        .find(|entry| entry.rfilename == identity.file)
        .and_then(|entry| entry.size);
    let mmproj = info.siblings.iter().find_map(|entry| {
        let basename = entry
            .rfilename
            .rsplit('/')
            .next()
            .unwrap_or(&entry.rfilename);
        let lower = basename.to_ascii_lowercase();
        if lower.starts_with("mmproj") && lower.ends_with(".gguf") {
            Some(catalog::CatalogAsset {
                file: basename.to_string(),
                url: format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    identity.repo_id, identity.revision, entry.rfilename
                ),
                size_bytes: entry.size,
            })
        } else {
            None
        }
    });
    let split_sizes = info
        .siblings
        .iter()
        .map(|entry| (entry.rfilename.clone(), entry.size))
        .collect();
    Ok(RecommendedRepoManifest {
        primary_size_bytes,
        mmproj,
        split_sizes,
    })
}

fn recommended_model_name_from_file(file: &str) -> String {
    let basename = Path::new(file)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(file);
    let mut stem = basename
        .trim_end_matches(".gguf")
        .trim_end_matches(".safetensors")
        .trim_end_matches(".json")
        .to_string();
    let split_re = regex_lite::Regex::new(r"-00001-of-\d{5}$").unwrap();
    stem = split_re.replace(&stem, "").into_owned();
    stem
}

fn split_gguf_extra_files(
    identity: &crate::models::local::HuggingFaceModelIdentity,
    manifest: &RecommendedRepoManifest,
) -> Vec<catalog::CatalogAsset> {
    let re = regex_lite::Regex::new(r"-00001-of-(\d{5})\.gguf$").unwrap();
    let Some(caps) = re.captures(&identity.file) else {
        return Vec::new();
    };
    let Ok(count) = caps[1].parse::<u32>() else {
        return Vec::new();
    };
    (2..=count)
        .map(|index| {
            let file = identity
                .file
                .replace("-00001-of-", &format!("-{index:05}-of-"));
            catalog::CatalogAsset {
                file: file.rsplit('/').next().unwrap_or(&file).to_string(),
                url: format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    identity.repo_id, identity.revision, file
                ),
                size_bytes: manifest.split_sizes.get(&file).copied().flatten(),
            }
        })
        .collect()
}

pub fn run_model_installed(json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
    let rows: Vec<InstalledRow> = scan_installed_models()
        .into_iter()
        .map(|name| {
            let path = crate::models::find_model_path(&name);
            let size = std::fs::metadata(&path).map(|meta| meta.len()).ok();
            let catalog_model = find_catalog_model_exact(&name);
            let capabilities = installed_model_capabilities(&name);
            InstalledRow {
                name,
                path,
                size,
                catalog_model,
                capabilities,
            }
        })
        .collect();
    formatter.render_installed(&rows)
}

pub async fn run_model_show(model_ref: &str, json_output: bool) -> Result<()> {
    let formatter = models_formatter(json_output);
    let interactive = !json_output && std::io::stdout().is_terminal();
    let detail_started = Instant::now();
    if interactive {
        eprintln!("🔎 Resolving model details from Hugging Face...");
    }
    let details = show_exact_model(model_ref).await?;
    if interactive {
        eprintln!(
            "✅ Resolved model details ({:.1}s)",
            detail_started.elapsed().as_secs_f32()
        );
    }
    let is_gguf = model_kind_code(details.kind) == "gguf";
    let variants = if is_gguf {
        let variants_started = Instant::now();
        if interactive {
            eprintln!("🔎 Fetching GGUF variants from Hugging Face...");
        }
        let variants = show_model_variants_with_progress(&details.exact_ref, |progress| {
            if !interactive {
                return;
            }
            match progress {
                ShowVariantsProgress::Inspecting { completed, total } => {
                    if total == 0 {
                        return;
                    }
                    eprint!("\r   Inspecting variant sizes {completed}/{total}...");
                    let _ = std::io::stderr().flush();
                    if completed == total {
                        eprintln!();
                    }
                }
            }
        })
        .await?;
        if let Some(variants) = &variants {
            if interactive {
                eprintln!(
                    "✅ Fetched {} GGUF variants ({:.1}s)",
                    variants.len(),
                    variants_started.elapsed().as_secs_f32()
                );
            }
        } else if interactive {
            eprintln!(
                "✅ No GGUF variants for this ref ({:.1}s)",
                variants_started.elapsed().as_secs_f32()
            );
        }
        variants
    } else {
        None
    };
    formatter.render_show(&details, variants.as_deref())
}

pub async fn run_model_download(
    model_ref: &str,
    include_draft: bool,
    json_output: bool,
) -> Result<()> {
    let formatter = models_formatter(json_output);
    let details = show_exact_model(model_ref).await.ok();
    let download_ref = details
        .as_ref()
        .map(|d| d.exact_ref.as_str())
        .unwrap_or(model_ref);
    let path = download_exact_ref(download_ref).await?;
    if !include_draft {
        return formatter.render_download(model_ref, &path, details.as_ref(), false, None);
    }

    let mut draft_out: Option<(String, std::path::PathBuf)> = None;
    if let Some(details_ref) = details.as_ref() {
        if let Some(draft_name) = details_ref.draft.as_deref() {
            let draft_model = find_catalog_model_exact(draft_name)
                .ok_or_else(|| anyhow!("Draft model '{}' not found in catalog", draft_name))?;
            let draft_path = catalog::download_model(draft_model).await?;
            draft_out = Some((draft_name.to_string(), draft_path));
        } else if !json_output {
            eprintln!(
                "⚠ No draft model available for {}",
                details_ref.display_name
            );
        }
    }
    formatter.render_download(
        model_ref,
        &path,
        details.as_ref(),
        true,
        draft_out.as_ref().map(|(n, p)| (n.as_str(), p.as_path())),
    )
}

pub async fn dispatch_models_command(command: &ModelsCommand) -> Result<()> {
    match command {
        ModelsCommand::Recommended { command, json } => match command {
            Some(RecommendedCommand::Share {
                model,
                description,
                name,
                draft,
                dataset_repo,
                json: share_json,
            }) => {
                run_model_recommended_share(
                    model,
                    description,
                    name.as_deref(),
                    draft.as_deref(),
                    dataset_repo,
                    *share_json,
                )
                .await?
            }
            None => run_model_recommended(*json)?,
        },
        ModelsCommand::List { json } => run_model_recommended(*json)?,
        ModelsCommand::Installed { json } => run_model_installed(*json)?,
        ModelsCommand::Search {
            query,
            gguf,
            mlx,
            catalog,
            limit,
            json,
        } => run_model_search(query, *gguf, *mlx, *catalog, *limit, *json).await?,
        ModelsCommand::Show { model, json } => run_model_show(model, *json).await?,
        ModelsCommand::Download { model, draft, json } => {
            run_model_download(model, *draft, *json).await?
        }
        ModelsCommand::Updates {
            repo,
            all,
            check,
            json,
        } => {
            crate::models::run_update(repo.as_deref(), *all, *check)?;
            if *json {
                let formatter = models_formatter(*json);
                formatter.render_updates_status(repo.as_deref(), *all, *check)?;
            }
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct HfCommitResponse {
    #[serde(rename = "commitOid")]
    commit_oid: String,
    #[serde(rename = "commitUrl")]
    commit_url: String,
    #[serde(rename = "pullRequestUrl")]
    pull_request_url: Option<String>,
}

fn ndjson_header(summary: &str, description: &str) -> serde_json::Value {
    json!({
        "key": "header",
        "value": {
            "summary": summary,
            "description": description,
        }
    })
}

fn ndjson_file_op(path_in_repo: &str, content: &[u8]) -> serde_json::Value {
    json!({
        "key": "file",
        "value": {
            "content": base64::engine::general_purpose::STANDARD.encode(content),
            "path": path_in_repo,
            "encoding": "base64",
        }
    })
}
