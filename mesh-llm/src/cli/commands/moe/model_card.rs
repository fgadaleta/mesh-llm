use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;

use hf_hub::{HFClient, HFError, RepoDownloadFileParams};

use crate::system::moe_planner;

use super::upload::parse_repo_id;

#[derive(Clone, Debug, Default)]
pub(super) struct ModelCardMetadata {
    pub(super) license: Option<String>,
    pub(super) language: Vec<String>,
    pub(super) pipeline_tag: Option<String>,
    pub(super) tags: Vec<String>,
}

pub(super) async fn load_source_model_card_metadata(
    api: &HFClient,
    source_repo: &str,
    source_revision: &str,
) -> Result<ModelCardMetadata> {
    let (owner, name) = parse_repo_id(source_repo)?;
    let repo = api.model(owner, name);
    let readme_path = match repo
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename("README.md".to_string())
                .revision(source_revision.to_string())
                .build(),
        )
        .await
    {
        Ok(path) => path,
        Err(HFError::EntryNotFound { .. }) => return Ok(ModelCardMetadata::default()),
        Err(err) => return Err(err.into()),
    };
    let readme = fs::read_to_string(&readme_path)
        .with_context(|| format!("Read {}", readme_path.display()))?;
    Ok(parse_model_card_metadata(&readme))
}

fn parse_model_card_metadata(readme: &str) -> ModelCardMetadata {
    let Some(front_matter) = readme
        .strip_prefix("---\n")
        .and_then(|rest| rest.split_once("\n---\n").map(|(yaml, _)| yaml))
        .or_else(|| {
            readme
                .strip_prefix("---\r\n")
                .and_then(|rest| rest.split_once("\r\n---\r\n").map(|(yaml, _)| yaml))
        })
    else {
        return ModelCardMetadata::default();
    };

    let value: serde_yaml::Value = match serde_yaml::from_str(front_matter) {
        Ok(value) => value,
        Err(_) => return ModelCardMetadata::default(),
    };
    let Some(map) = value.as_mapping() else {
        return ModelCardMetadata::default();
    };

    ModelCardMetadata {
        license: yaml_string(map, "license"),
        language: yaml_strings(map, "language"),
        pipeline_tag: yaml_string(map, "pipeline_tag"),
        tags: yaml_strings(map, "tags"),
    }
}

fn yaml_key(key: &str) -> serde_yaml::Value {
    serde_yaml::Value::String(key.to_string())
}

fn yaml_string(map: &serde_yaml::Mapping, key: &str) -> Option<String> {
    map.get(yaml_key(key))
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn yaml_strings(map: &serde_yaml::Mapping, key: &str) -> Vec<String> {
    let Some(value) = map.get(yaml_key(key)) else {
        return Vec::new();
    };
    match value {
        serde_yaml::Value::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                Vec::new()
            } else {
                vec![value.to_string()]
            }
        }
        serde_yaml::Value::Sequence(values) => values
            .iter()
            .filter_map(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn build_package_readme(
    meshllm: &moe_planner::MeshllmPackageJson,
    package_repo: &str,
    metadata: &ModelCardMetadata,
) -> String {
    let mut out = String::new();
    let mut tags = BTreeSet::from([
        "mesh-llm".to_string(),
        "moe".to_string(),
        "gguf".to_string(),
        "distributed-inference".to_string(),
        "topology-independent".to_string(),
    ]);
    tags.extend(metadata.tags.iter().cloned());
    let pipeline_tag = metadata
        .pipeline_tag
        .as_deref()
        .unwrap_or("text-generation");
    let _ = writeln!(&mut out, "---");
    if let Some(license) = metadata.license.as_deref() {
        let _ = writeln!(&mut out, "license: {license}");
    }
    if !metadata.language.is_empty() {
        let _ = writeln!(&mut out, "language:");
        for language in &metadata.language {
            let _ = writeln!(&mut out, "- {language}");
        }
    }
    let _ = writeln!(&mut out, "base_model:");
    let _ = writeln!(&mut out, "- {}", meshllm.source.repo);
    let _ = writeln!(&mut out, "pipeline_tag: {pipeline_tag}");
    let _ = writeln!(&mut out, "library_name: mesh-llm");
    let _ = writeln!(&mut out, "tags:");
    for tag in tags {
        let _ = writeln!(&mut out, "- {tag}");
    }
    let _ = writeln!(&mut out, "---");
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "# Mesh-LLM MoE Package");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "This repository stores Mesh-LLM topology-independent MoE package artifacts derived from `{}`.",
        meshllm.source.repo
    );
    let _ = writeln!(
        &mut out,
        "It is published as `{}` and is meant to be consumed by `mesh-llm serve`.",
        package_repo
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "## Source");
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "- repo: `{}`", meshllm.source.repo);
    let _ = writeln!(&mut out, "- revision: `{}`", meshllm.source.revision);
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "## What This Repository Contains");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "- `meshllm.json` describes the upstream source repo and all published variants in this package repository."
    );
    let _ = writeln!(
        &mut out,
        "- `variants/<variant>/manifest.json` is the runtime entrypoint used to materialize MoE shards."
    );
    let _ = writeln!(
        &mut out,
        "- `variants/<variant>/ranking.csv` and `variants/<variant>/analysis.json` contain the analyzer output for that variant."
    );
    let _ = writeln!(
        &mut out,
        "- `variants/<variant>/trunk.gguf` plus `variants/<variant>/experts/` hold the topology-independent component artifacts."
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "## Available Variants");
    let _ = writeln!(&mut out);
    for (variant, entry) in &meshllm.variants {
        let model_ref = format!("{}:{}", meshllm.source.repo, variant);
        let _ = writeln!(&mut out, "### `{}`", variant);
        let _ = writeln!(&mut out);
        let _ = writeln!(&mut out, "- Mesh model ref: `{}`", model_ref);
        let _ = writeln!(&mut out, "- Distribution id: `{}`", entry.distribution_id);
        let _ = writeln!(&mut out, "- Manifest: `{}`", entry.manifest);
        let _ = writeln!(&mut out, "- Serve with:");
        let _ = writeln!(&mut out);
        let _ = writeln!(&mut out, "```bash");
        let _ = writeln!(&mut out, "mesh-llm serve '{}'", model_ref);
        let _ = writeln!(&mut out, "```");
        let _ = writeln!(&mut out);
    }
    let _ = writeln!(&mut out, "## Notes");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "- This is a derived Mesh-LLM package repository, not the original upstream model repository."
    );
    let _ = writeln!(
        &mut out,
        "- `mesh-llm` will prefer published package artifacts from this repository when a matching catalog entry exists."
    );
    out
}
