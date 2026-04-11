use clap::Subcommand;
use std::path::PathBuf;

#[derive(Subcommand, Debug)]
pub(crate) enum MoeCommand {
    /// Plan an MoE split using cached or published expert rankings.
    Plan {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        model: String,
        /// Override the ranking CSV path instead of resolving from cache or Hugging Face.
        #[arg(long)]
        ranking_file: Option<PathBuf>,
        /// Emit JSON output.
        #[arg(long)]
        json: bool,
        /// Cap VRAM used for planning (GB). Matches the existing global naming.
        #[arg(long)]
        max_vram: Option<f64>,
        /// Optional node count override. When omitted, mesh-llm recommends a minimum node count.
        #[arg(long)]
        nodes: Option<usize>,
        /// Published dataset repo used for MoE ranking lookup.
        #[arg(long, default_value = "meshllm/moe-rankings")]
        dataset_repo: String,
    },
    /// Run the canonical full MoE analysis and cache it locally.
    #[command(name = "analyze-full")]
    AnalyzeFull {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        model: String,
        /// Override context size passed to llama-moe-analyze.
        #[arg(long, default_value = "4096")]
        context_size: u32,
    },
    /// Run the canonical micro MoE analysis and cache it locally.
    #[command(name = "analyze-micro")]
    AnalyzeMicro {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        model: String,
        /// Number of canonical prompts to use.
        #[arg(long, default_value = "8")]
        prompt_count: usize,
        /// Token budget per prompt.
        #[arg(long, default_value = "128")]
        token_count: u32,
        /// Override context size passed to llama-moe-analyze.
        #[arg(long, default_value = "4096")]
        context_size: u32,
    },
    /// Open a contribution PR for a local ranking artifact on the canonical Hugging Face dataset.
    Submit {
        /// Model spec: local path, catalog name, HF exact ref, or HF URL.
        model: String,
        /// Override the ranking CSV path instead of resolving a local cached artifact.
        /// The path must include `micro-v1` or `full-v1` so mesh-llm can infer the analyzer id.
        #[arg(long)]
        ranking_file: Option<PathBuf>,
        /// Published dataset repo used for duplicate checks and PR target reporting.
        #[arg(long, default_value = "meshllm/moe-rankings")]
        dataset_repo: String,
    },
}
