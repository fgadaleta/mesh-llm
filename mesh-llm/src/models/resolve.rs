use super::ModelCapabilities;
use super::{capabilities, catalog, find_model_path, format_size_bytes, hf_token_override};
use anyhow::{anyhow, bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub repo_id: String,
    pub file: String,
    pub exact_ref: String,
    pub size_label: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub catalog: Option<&'static catalog::CatalogModel>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Debug)]
pub struct ModelDetails {
    pub display_name: String,
    pub exact_ref: String,
    pub source: &'static str,
    pub download_url: String,
    pub size_label: Option<String>,
    pub description: Option<String>,
    pub draft: Option<String>,
    pub capabilities: ModelCapabilities,
    pub moe: Option<catalog::MoeConfig>,
}

#[derive(Clone, Debug)]
enum ExactModelRef {
    Catalog(&'static catalog::CatalogModel),
    HuggingFace {
        repo: String,
        revision: Option<String>,
        file: String,
    },
    Url {
        url: String,
        filename: String,
    },
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRepoSummary {
    id: String,
    downloads: Option<u64>,
    likes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRepoDetail {
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "modelId")]
    model_id: Option<String>,
    #[serde(default)]
    siblings: Vec<HuggingFaceSibling>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSibling {
    rfilename: String,
}

fn merge_capabilities(left: ModelCapabilities, right: ModelCapabilities) -> ModelCapabilities {
    ModelCapabilities {
        vision: left.vision.max(right.vision),
        reasoning: left.reasoning.max(right.reasoning),
    }
}

pub fn find_catalog_model_exact(query: &str) -> Option<&'static catalog::CatalogModel> {
    let q = query.to_lowercase();
    catalog::MODEL_CATALOG.iter().find(|model| {
        model.name.to_lowercase() == q
            || model.file.to_lowercase() == q
            || model.file.trim_end_matches(".gguf").to_lowercase() == q
    })
}

pub fn search_catalog_models(query: &str) -> Vec<&'static catalog::CatalogModel> {
    let q = query.to_lowercase();
    let mut results: Vec<_> = catalog::MODEL_CATALOG
        .iter()
        .filter(|model| {
            model.name.to_lowercase().contains(&q)
                || model.file.to_lowercase().contains(&q)
                || model.description.to_lowercase().contains(&q)
        })
        .collect();
    results.sort_by(|left, right| left.name.cmp(&right.name));
    results
}

pub async fn download_exact_ref(input: &str) -> Result<PathBuf> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Catalog(model) => catalog::download_model(model).await,
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => catalog::download_hf_repo_file(&repo, revision.as_deref(), &file).await,
        ExactModelRef::Url { url, filename } => {
            let dest = catalog::models_dir().join(&filename);
            if existing_download(&dest).await {
                return Ok(dest);
            }
            eprintln!("📥 Downloading {}...", dest.display());
            catalog::download_url(&url, &dest).await?;
            Ok(dest)
        }
    }
}

pub async fn show_exact_model(input: &str) -> Result<ModelDetails> {
    match parse_exact_model_ref(input)? {
        ExactModelRef::Catalog(model) => Ok(ModelDetails {
            display_name: model.name.to_string(),
            exact_ref: model.name.to_string(),
            source: "catalog",
            download_url: match (
                model.source_repo(),
                model.source_revision(),
                model.source_file(),
            ) {
                (Some(repo), revision, Some(file)) => huggingface_resolve_url(repo, revision, file),
                _ => model.url.to_string(),
            },
            size_label: Some(model.size.to_string()),
            description: Some(model.description.to_string()),
            draft: model.draft.clone(),
            capabilities: capabilities::infer_catalog_capabilities(model),
            moe: model.moe.clone(),
        }),
        ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        } => {
            let exact_ref = format_huggingface_exact_ref(&repo, revision.as_deref(), &file);
            let catalog = matching_catalog_model_for_huggingface(&repo, revision.as_deref(), &file);
            let download_url = huggingface_resolve_url(&repo, revision.as_deref(), &file);
            let size_label = match catalog {
                Some(model) => Some(model.size.to_string()),
                None => remote_size_label(&download_url).await,
            };
            let capabilities = match catalog {
                Some(model) => {
                    let base = capabilities::infer_catalog_capabilities(model);
                    let remote = capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &file,
                        None,
                    )
                    .await;
                    merge_capabilities(base, remote)
                }
                None => {
                    capabilities::infer_remote_hf_capabilities(
                        &repo,
                        revision.as_deref(),
                        &file,
                        None,
                    )
                    .await
                }
            };
            Ok(ModelDetails {
                display_name: Path::new(&file)
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or(&file)
                    .to_string(),
                exact_ref,
                source: "huggingface",
                download_url,
                size_label,
                description: catalog.map(|model| model.description.to_string()),
                draft: catalog.and_then(|model| model.draft.clone()),
                capabilities,
                moe: catalog.and_then(|model| model.moe.clone()),
            })
        }
        ExactModelRef::Url { url, filename } => {
            let catalog = matching_catalog_model_for_url(&url);
            let size_label = match catalog {
                Some(model) => Some(model.size.to_string()),
                None => remote_size_label(&url).await,
            };
            Ok(ModelDetails {
                display_name: filename,
                exact_ref: url.clone(),
                source: "url",
                download_url: url,
                size_label,
                description: catalog.map(|model| model.description.to_string()),
                draft: catalog.and_then(|model| model.draft.clone()),
                capabilities: catalog
                    .map(capabilities::infer_catalog_capabilities)
                    .unwrap_or_default(),
                moe: catalog.and_then(|model| model.moe.clone()),
            })
        }
    }
}

// Keep search custom for now. `hf-hub` handles cache and file transport well,
// but it does not expose a Hub search surface in this crate version.
pub async fn search_huggingface(query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let repo_limit = limit.clamp(1, 100);
    let client = http_client()?;
    let mut request = client.get("https://huggingface.co/api/models").query(&[
        ("search", query),
        ("filter", "gguf"),
        ("limit", &repo_limit.to_string()),
    ]);
    if let Some(token) = hf_token_override() {
        request = request.bearer_auth(token);
    }
    let repos: Vec<HuggingFaceRepoSummary> = request
        .send()
        .await
        .context("Search Hugging Face")?
        .error_for_status()
        .context("Hugging Face search failed")?
        .json()
        .await
        .context("Parse Hugging Face search response")?;

    let mut hits = Vec::new();
    for repo in repos {
        let mut detail_request =
            client.get(format!("https://huggingface.co/api/models/{}", repo.id));
        if let Some(token) = hf_token_override() {
            detail_request = detail_request.bearer_auth(token);
        }
        let detail: HuggingFaceRepoDetail = detail_request
            .send()
            .await
            .with_context(|| format!("Fetch Hugging Face repo {}", repo.id))?
            .error_for_status()
            .with_context(|| format!("Hugging Face repo {} returned an error", repo.id))?
            .json()
            .await
            .with_context(|| format!("Parse Hugging Face repo {}", repo.id))?;

        let repo_id = detail.id.or(detail.model_id).unwrap_or(repo.id.clone());
        let sibling_names: Vec<String> = detail
            .siblings
            .iter()
            .map(|sibling| sibling.rfilename.clone())
            .collect();
        let mut files: Vec<String> = detail
            .siblings
            .into_iter()
            .map(|sibling| sibling.rfilename)
            .filter(|file| file.ends_with(".gguf"))
            .collect();
        if files.is_empty() {
            continue;
        }
        files.sort_by(|left, right| {
            file_preference_score(left)
                .cmp(&file_preference_score(right))
                .then_with(|| left.cmp(right))
        });
        if let Some(file) = files.into_iter().next() {
            let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &file);
            let download_url = huggingface_resolve_url(&repo_id, None, &file);
            let size_label = match catalog {
                Some(model) => Some(model.size.to_string()),
                None => remote_size_label(&download_url).await,
            };
            let remote_caps = capabilities::infer_remote_hf_capabilities(
                &repo_id,
                None,
                &file,
                Some(&sibling_names),
            )
            .await;
            let capabilities = match catalog {
                Some(model) => {
                    let base = capabilities::infer_catalog_capabilities(model);
                    merge_capabilities(base, remote_caps)
                }
                None => remote_caps,
            };
            hits.push(SearchHit {
                repo_id: repo_id.clone(),
                file: file.clone(),
                exact_ref: format!("{repo_id}/{file}"),
                size_label,
                downloads: repo.downloads,
                likes: repo.likes,
                catalog,
                capabilities,
            });
            if hits.len() >= limit {
                return Ok(hits);
            }
        }
    }
    Ok(hits)
}

pub fn installed_model_capabilities(model_name: &str) -> ModelCapabilities {
    let path = find_model_path(model_name);
    let catalog = find_catalog_model_exact(model_name);
    capabilities::infer_local_model_capabilities(model_name, &path, catalog)
}

pub fn installed_model_display_name(model_name: &str) -> String {
    find_catalog_model_exact(model_name)
        .map(|model| model.name.clone())
        .unwrap_or_else(|| model_name.to_string())
}

pub(super) fn catalog_hf_match(
    path: &Path,
) -> Option<(
    &'static catalog::CatalogModel,
    String,
    Option<String>,
    String,
)> {
    let model = catalog_match(path)?;
    let file_name = path.file_name()?.to_str()?;
    if model.file == file_name {
        if let (Some(repo), revision, Some(file)) = (
            model.source_repo(),
            model.source_revision(),
            model.source_file(),
        ) {
            return Some((
                model,
                repo.to_string(),
                revision.map(str::to_string),
                file.to_string(),
            ));
        }
        return None;
    }

    let source_url = if let Some(asset) = model
        .extra_files
        .iter()
        .find(|asset| asset.file == file_name)
    {
        asset.url.as_str()
    } else if let Some(asset) = model.mmproj.as_ref() {
        if asset.file == file_name {
            asset.url.as_str()
        } else {
            return None;
        }
    } else {
        return None;
    };

    let (repo, revision, file) = parse_hf_resolve_url(source_url)?;
    Some((model, repo, revision, file))
}

pub(super) fn catalog_hf_asset_ref(
    model: &'static catalog::CatalogModel,
    file_name: &str,
) -> Option<(String, Option<String>, String)> {
    if model.file == file_name {
        return Some((
            model.source_repo()?.to_string(),
            model.source_revision().map(str::to_string),
            model.source_file()?.to_string(),
        ));
    }

    let source_url = if let Some(asset) = model
        .extra_files
        .iter()
        .find(|asset| asset.file == file_name)
    {
        asset.url.as_str()
    } else if let Some(asset) = model.mmproj.as_ref() {
        if asset.file == file_name {
            asset.url.as_str()
        } else {
            return None;
        }
    } else {
        return None;
    };

    parse_hf_resolve_url(source_url)
}

pub(super) fn catalog_match(path: &Path) -> Option<&'static catalog::CatalogModel> {
    let file_name = path.file_name()?.to_str()?;
    catalog::MODEL_CATALOG.iter().find(|model| {
        model.file == file_name
            || model
                .extra_files
                .iter()
                .any(|asset| asset.file == file_name)
            || model
                .mmproj
                .as_ref()
                .map(|asset| asset.file == file_name)
                .unwrap_or(false)
    })
}

fn matching_catalog_model_for_huggingface(
    repo: &str,
    revision: Option<&str>,
    file: &str,
) -> Option<&'static catalog::CatalogModel> {
    let repo = repo.to_lowercase();
    let revision = revision.map(|value| value.to_lowercase());
    let file = file.to_lowercase();

    catalog::MODEL_CATALOG
        .iter()
        .find(|model| {
            std::iter::once(model.file.as_str())
                .chain(model.extra_files.iter().map(|asset| asset.file.as_str()))
                .chain(model.mmproj.iter().map(|asset| asset.file.as_str()))
                .any(|asset_name| {
                    let Some((asset_repo, asset_revision, asset_file)) =
                        catalog_hf_asset_ref(model, asset_name)
                    else {
                        return false;
                    };
                    if asset_repo.to_lowercase() != repo || asset_file.to_lowercase() != file {
                        return false;
                    }
                    match &revision {
                        Some(revision) => {
                            asset_revision.map(|value| value.to_lowercase())
                                == Some(revision.clone())
                        }
                        None => true,
                    }
                })
        })
        .or_else(|| {
            if revision.is_some() {
                None
            } else {
                matching_catalog_model_by_basename(file.as_str())
            }
        })
}

fn matching_catalog_model_for_url(url: &str) -> Option<&'static catalog::CatalogModel> {
    catalog::MODEL_CATALOG
        .iter()
        .find(|model| model.url.eq_ignore_ascii_case(url))
        .or_else(|| matching_catalog_model_by_basename(url))
}

fn matching_catalog_model_by_basename(repo_file: &str) -> Option<&'static catalog::CatalogModel> {
    let basename = Path::new(repo_file)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(repo_file)
        .to_lowercase();
    catalog::MODEL_CATALOG.iter().find(|model| {
        model.file.to_lowercase() == basename
            || model.file.trim_end_matches(".gguf").to_lowercase()
                == basename.trim_end_matches(".gguf")
    })
}

pub(super) fn parse_hf_resolve_url(url: &str) -> Option<(String, Option<String>, String)> {
    let tail = url
        .strip_prefix("https://huggingface.co/")
        .or_else(|| url.strip_prefix("http://huggingface.co/"))?;
    let parts: Vec<&str> = tail.split('/').collect();
    if parts.len() < 5 || parts.get(2) != Some(&"resolve") {
        return None;
    }
    Some((
        format!("{}/{}", parts[0], parts[1]),
        parts.get(3).map(|value| value.to_string()),
        parts[4..].join("/"),
    ))
}

pub(super) fn parse_huggingface_ref(input: &str) -> Option<(String, Option<String>, String)> {
    if let Some(parsed) = parse_hf_resolve_url(input) {
        return Some(parsed);
    }
    if !input.ends_with(".gguf") {
        return None;
    }

    let parts: Vec<&str> = input.splitn(3, '/').collect();
    if parts.len() != 3 {
        return None;
    }
    let (repo_tail, revision) = match parts[1].split_once('@') {
        Some((repo, revision)) => (repo, Some(revision.to_string())),
        None => (parts[1], None),
    };
    Some((
        format!("{}/{}", parts[0], repo_tail),
        revision,
        parts[2].to_string(),
    ))
}

fn parse_exact_model_ref(input: &str) -> Result<ExactModelRef> {
    if let Some(model) = find_catalog_model_exact(input) {
        return Ok(ExactModelRef::Catalog(model));
    }
    if let Some((repo, revision, file)) = parse_huggingface_ref(input) {
        return Ok(ExactModelRef::HuggingFace {
            repo,
            revision,
            file,
        });
    }
    if input.starts_with("http://") || input.starts_with("https://") {
        return Ok(ExactModelRef::Url {
            url: input.to_string(),
            filename: remote_filename(input)?,
        });
    }
    bail!(
        "Expected an exact model ref. Use a catalog id, a Hugging Face ref like org/repo/file.gguf, or a direct URL."
    )
}

pub(super) fn huggingface_resolve_url(repo: &str, revision: Option<&str>, file: &str) -> String {
    let revision = revision.unwrap_or("main");
    format!("https://huggingface.co/{repo}/resolve/{revision}/{file}")
}

fn format_huggingface_exact_ref(repo: &str, revision: Option<&str>, file: &str) -> String {
    match revision {
        Some(revision) => format!("{repo}@{revision}/{file}"),
        None => format!("{repo}/{file}"),
    }
}

fn remote_filename(input: &str) -> Result<String> {
    input
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .ok_or_else(|| anyhow!("Cannot extract filename from URL: {input}"))
}

async fn existing_download(path: &Path) -> bool {
    tokio::fs::metadata(path)
        .await
        .map(|meta| meta.len() > 1_000_000)
        .unwrap_or(false)
}

fn file_preference_score(file: &str) -> usize {
    if file.contains("-00001-of-") {
        return 0;
    }
    const PREFERRED: &[&str] = &[
        "Q4_K_M", "Q4_K_S", "Q4_1", "Q5_K_M", "Q5_K_S", "Q8_0", "BF16",
    ];
    PREFERRED
        .iter()
        .position(|needle| file.contains(needle))
        .unwrap_or(PREFERRED.len() + 1)
}

pub(super) fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .connect_timeout(std::time::Duration::from_secs(30))
        .user_agent(format!("mesh-llm/{}", crate::VERSION))
        .build()
        .context("Build HTTP client")
}

async fn remote_size_label(url: &str) -> Option<String> {
    let client = http_client().ok()?;
    let mut request = client.head(url);
    if url.contains("huggingface.co/") {
        if let Some(token) = hf_token_override() {
            request = request.bearer_auth(token);
        }
    }
    let response = request.send().await.ok()?.error_for_status().ok()?;
    let size = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .parse::<u64>()
        .ok()?;
    Some(format_size_bytes(size))
}
