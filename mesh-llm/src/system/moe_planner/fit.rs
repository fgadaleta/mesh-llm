use super::*;

pub(crate) fn estimate_startup_fit_from_analysis(
    ranking_path: &Path,
    analysis: &MoeAnalysisJson,
    target_vram_bytes: u64,
    analyzer_id: &str,
    ranking_source: &'static str,
) -> Result<MoeStartupFitEstimate> {
    let ranking = moe::load_cached_ranking(ranking_path)
        .ok_or_else(|| anyhow::anyhow!("Could not parse ranking {}", ranking_path.display()))?;
    if analysis.ranking.rows != ranking.len() {
        bail!(
            "Ranking row count mismatch for {}: analysis says {}, CSV has {}",
            ranking_path.display(),
            analysis.ranking.rows,
            ranking.len()
        );
    }

    let required_experts_per_node = default_required_experts_per_node(analysis.model.expert_count);
    let full_model_launch_bytes = apply_model_load_headroom(analysis.memory.full_model_bytes);
    let mut estimate = MoeStartupFitEstimate {
        analyzer_id: analyzer_id.to_string(),
        ranking_source,
        target_vram_bytes,
        required_experts_per_node,
        full_model_bytes: analysis.memory.full_model_bytes,
        full_model_launch_bytes,
        recommended_nodes: None,
        predicted_max_shard_bytes: None,
        predicted_max_shard_launch_bytes: None,
    };

    if estimate.full_model_fits() {
        estimate.recommended_nodes = Some(1);
        return Ok(estimate);
    }

    let max_supported_nodes =
        (analysis.model.expert_count / required_experts_per_node.max(1)).max(1) as usize;
    let mut best_failed_shard: Option<(u64, u64)> = None;
    for nodes in 2..=max_supported_nodes {
        let assignments =
            moe::compute_assignments_with_overlap(&ranking, nodes, required_experts_per_node, 1);
        let max_shard_bytes = assignments
            .iter()
            .map(|assignment| predict_shard_bytes(&analysis.memory, &assignment.experts))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or_default();
        let max_shard_launch_bytes = apply_model_load_headroom(max_shard_bytes);
        if max_shard_launch_bytes <= target_vram_bytes {
            estimate.recommended_nodes = Some(nodes);
            estimate.predicted_max_shard_bytes = Some(max_shard_bytes);
            estimate.predicted_max_shard_launch_bytes = Some(max_shard_launch_bytes);
            return Ok(estimate);
        }
        best_failed_shard = Some((max_shard_bytes, max_shard_launch_bytes));
    }

    if let Some((max_shard_bytes, max_shard_launch_bytes)) = best_failed_shard {
        estimate.predicted_max_shard_bytes = Some(max_shard_bytes);
        estimate.predicted_max_shard_launch_bytes = Some(max_shard_launch_bytes);
    } else if max_supported_nodes == 1 {
        let max_shard_bytes = predict_shard_bytes(&analysis.memory, &ranking)?;
        estimate.predicted_max_shard_bytes = Some(max_shard_bytes);
        estimate.predicted_max_shard_launch_bytes =
            Some(apply_model_load_headroom(max_shard_bytes));
    }

    Ok(estimate)
}

pub(crate) fn fetch_remote_startup_fit(
    catalog_repo_name: &str,
    source_repo: &str,
    distribution_id: &str,
    target_vram_bytes: u64,
    progress: bool,
) -> Result<Option<MoeStartupFitEstimate>> {
    let model_ref = canonical_model_ref_from_source(source_repo, distribution_id, None);
    let Some(ranking) = fetch_remote_package_ranking(catalog_repo_name, &model_ref, progress)?
    else {
        return Ok(None);
    };
    let Some(analysis_path) = ranking.analysis_path.as_ref() else {
        return Ok(None);
    };
    let analysis = read_analysis_json(analysis_path)?;
    let estimate = estimate_startup_fit_from_analysis(
        &ranking.path,
        &analysis,
        target_vram_bytes,
        &ranking.analyzer_id,
        ranking.source.label(),
    )?;
    Ok(Some(estimate))
}

pub(crate) fn build_mass_checkpoints(profile: &AnalyzeMassProfile) -> Vec<Value> {
    let mut checkpoints = DEFAULT_MASS_CHECKPOINTS
        .into_iter()
        .filter(|top_n| *top_n <= profile.masses.len())
        .collect::<Vec<_>>();
    if !checkpoints.contains(&profile.masses.len()) {
        checkpoints.push(profile.masses.len());
    }
    checkpoints
        .into_iter()
        .map(|top_n| {
            let captured_mass = profile
                .masses
                .iter()
                .take(top_n)
                .map(|(_, mass)| *mass)
                .sum::<f64>();
            json!({
                "top_n": top_n,
                "mass_fraction": if profile.total_mass > f64::EPSILON {
                    captured_mass / profile.total_mass
                } else {
                    0.0
                }
            })
        })
        .collect()
}

pub(crate) fn aggregate_tensor_byte_profile(
    model_files: &[(String, PathBuf)],
) -> Result<AggregatedTensorByteProfile> {
    let mut aggregated = AggregatedTensorByteProfile::default();
    for (_, path) in model_files {
        let profile: GgufTensorByteProfile =
            crate::models::gguf::scan_gguf_tensor_byte_profile(path).ok_or_else(|| {
                anyhow::anyhow!("Could not derive GGUF tensor bytes for {}", path.display())
            })?;
        aggregated.full_model_bytes = aggregated
            .full_model_bytes
            .saturating_add(profile.full_model_bytes);
        aggregated.base_resident_bytes = aggregated
            .base_resident_bytes
            .saturating_add(profile.base_resident_bytes);
        aggregated.expert_tensor_bytes = aggregated
            .expert_tensor_bytes
            .saturating_add(profile.expert_tensor_bytes);
        aggregated.file_overhead_bytes = aggregated
            .file_overhead_bytes
            .saturating_add(profile.file_overhead_bytes);
    }
    Ok(aggregated)
}

pub(crate) fn expert_bytes_json(expert_tensor_bytes: u64, expert_count: u32) -> Result<Value> {
    if expert_count == 0 {
        bail!("cannot derive expert bytes for expert_count=0");
    }
    let expert_count_u64 = u64::from(expert_count);
    if expert_tensor_bytes % expert_count_u64 == 0 {
        return Ok(json!({
            "kind": "uniform",
            "bytes_per_expert": expert_tensor_bytes / expert_count_u64,
        }));
    }

    let base = expert_tensor_bytes / expert_count_u64;
    let remainder = (expert_tensor_bytes % expert_count_u64) as usize;
    let mut values = vec![base; expert_count as usize];
    for value in values.iter_mut().take(remainder) {
        *value += 1;
    }
    Ok(json!({
        "kind": "dense_per_expert",
        "values": values,
    }))
}

fn default_required_experts_per_node(expert_count: u32) -> u32 {
    ((expert_count as f64) * 0.5).ceil() as u32
}

fn apply_model_load_headroom(bytes: u64) -> u64 {
    bytes
        .saturating_mul(MODEL_LOAD_HEADROOM_NUMERATOR)
        .div_ceil(MODEL_LOAD_HEADROOM_DENOMINATOR)
}

pub(crate) fn predict_plan_fit_for_nodes(
    ranking: &[u32],
    analysis: &MoeAnalysisJson,
    nodes: usize,
    required_experts_per_node: u32,
) -> Result<(Vec<moe::NodeAssignment>, u64, u64)> {
    let assignments =
        moe::compute_assignments_with_overlap(ranking, nodes, required_experts_per_node, 1);
    let max_shard_bytes = assignments
        .iter()
        .map(|assignment| predict_shard_bytes(&analysis.memory, &assignment.experts))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or_default();
    Ok((
        assignments,
        max_shard_bytes,
        apply_model_load_headroom(max_shard_bytes),
    ))
}

fn predict_shard_bytes(memory: &MoeAnalysisMemory, experts: &[u32]) -> Result<u64> {
    let mut total = memory
        .base_resident_bytes
        .saturating_add(memory.shard_file_overhead_bytes);
    for expert_id in experts {
        total = total
            .checked_add(memory.expert_bytes.bytes_for(*expert_id)?)
            .ok_or_else(|| anyhow::anyhow!("predicted shard bytes overflow"))?;
    }
    Ok(total)
}
