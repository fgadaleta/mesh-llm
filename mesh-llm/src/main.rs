// No-op Rust touch to force CI cache validation on PR #284.
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    mesh_llm::run().await
}
