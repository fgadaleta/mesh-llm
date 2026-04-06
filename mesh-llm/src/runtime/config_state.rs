use anyhow::Result;
use std::path::{Path, PathBuf};

use crate::plugin::{load_config, validate_config, MeshConfig};
use crate::protocol::convert::{canonical_config_hash, mesh_config_to_proto};

#[derive(Debug)]
pub(crate) enum ApplyResult {
    Applied {
        revision: u64,
        hash: [u8; 32],
        saved_to_disk: bool,
    },
    RevisionConflict {
        current_revision: u64,
    },
    ValidationError(String),
    PersistError(String),
}

pub(crate) struct ConfigState {
    revision: u64,
    config_hash: [u8; 32],
    config: MeshConfig,
    config_path: PathBuf,
}

fn revision_sidecar_path(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .unwrap_or(Path::new("."))
        .join("config-revision")
}

fn read_revision(sidecar: &Path) -> u64 {
    std::fs::read_to_string(sidecar)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

fn atomic_write(target: &Path, contents: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp_name = target.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);
    std::fs::write(&tmp, contents)?;
    std::fs::rename(&tmp, target)?;
    Ok(())
}

impl Default for ConfigState {
    fn default() -> Self {
        let config = crate::plugin::MeshConfig::default();
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        Self {
            revision: 0,
            config_hash,
            config,
            config_path: std::path::PathBuf::from("config.toml"),
        }
    }
}

impl ConfigState {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let config = load_config(Some(path)).unwrap_or_default();
        let revision = read_revision(&revision_sidecar_path(path));
        let proto = mesh_config_to_proto(&config);
        let config_hash = canonical_config_hash(&proto);
        Ok(Self {
            revision,
            config_hash,
            config,
            config_path: path.to_path_buf(),
        })
    }

    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn config_hash(&self) -> &[u8; 32] {
        &self.config_hash
    }

    pub(crate) fn config(&self) -> &MeshConfig {
        &self.config
    }

    pub(crate) fn apply(&mut self, new_config: MeshConfig, expected_revision: u64) -> ApplyResult {
        if let Err(e) = validate_config(&new_config) {
            return ApplyResult::ValidationError(e.to_string());
        }

        if expected_revision != self.revision {
            return ApplyResult::RevisionConflict {
                current_revision: self.revision,
            };
        }

        let proto = mesh_config_to_proto(&new_config);
        let new_hash = canonical_config_hash(&proto);

        let toml_str = match toml::to_string(&new_config) {
            Ok(s) => s,
            Err(e) => return ApplyResult::PersistError(format!("toml serialization failed: {e}")),
        };

        if let Err(e) = atomic_write(&self.config_path, toml_str.as_bytes()) {
            return ApplyResult::PersistError(format!("failed to write config: {e}"));
        }

        let new_revision = self.revision + 1;
        let sidecar = revision_sidecar_path(&self.config_path);
        if let Err(e) = atomic_write(&sidecar, new_revision.to_string().as_bytes()) {
            return ApplyResult::PersistError(format!("failed to write revision sidecar: {e}"));
        }

        self.config = new_config;
        self.config_hash = new_hash;
        self.revision = new_revision;

        ApplyResult::Applied {
            revision: self.revision,
            hash: self.config_hash,
            saved_to_disk: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::{GpuAssignment, GpuConfig, MeshConfig};

    fn test_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mesh-llm-config-state-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    fn minimal_valid_config() -> MeshConfig {
        MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![],
            plugins: vec![],
        }
    }

    #[test]
    fn config_sync_state_load() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        std::fs::write(
            &config_path,
            "version = 1\n\n[gpu]\nassignment = \"auto\"\n",
        )
        .expect("write config");

        let state = ConfigState::load(&config_path).expect("load");
        assert_eq!(state.revision(), 0);
        assert_eq!(state.config().version, Some(1));
        assert_eq!(state.config().gpu.assignment, GpuAssignment::Auto);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_apply_success() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        let mut state = ConfigState::load(&config_path).expect("load");
        assert_eq!(state.revision(), 0);

        let result = state.apply(minimal_valid_config(), 0);
        match result {
            ApplyResult::Applied {
                revision,
                hash: _,
                saved_to_disk,
            } => {
                assert_eq!(revision, 1);
                assert!(saved_to_disk);
            }
            other => panic!("expected Applied, got {other:?}"),
        }

        assert!(config_path.exists(), "config file not written");

        let sidecar = revision_sidecar_path(&config_path);
        let sidecar_contents = std::fs::read_to_string(&sidecar).expect("read sidecar");
        assert_eq!(sidecar_contents.trim(), "1");

        assert_eq!(state.revision(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_conflict() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");

        let mut state = ConfigState::load(&config_path).expect("load");

        let result = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(result, ApplyResult::Applied { revision: 1, .. }),
            "first apply failed: {result:?}"
        );

        let result2 = state.apply(minimal_valid_config(), 0);
        match result2 {
            ApplyResult::RevisionConflict { current_revision } => {
                assert_eq!(current_revision, 1);
            }
            other => panic!("expected RevisionConflict, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_concurrent_applies() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();

        let r1 = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(r1, ApplyResult::Applied { revision: 1, .. }),
            "first apply must succeed: {r1:?}"
        );

        let r2 = state.apply(minimal_valid_config(), 0);
        assert!(
            matches!(
                r2,
                ApplyResult::RevisionConflict {
                    current_revision: 1
                }
            ),
            "second apply with stale revision must conflict: {r2:?}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_revision_monotonic() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();

        assert_eq!(state.revision(), 0);
        state.apply(minimal_valid_config(), 0);
        assert_eq!(state.revision(), 1);
        state.apply(minimal_valid_config(), 1);
        assert_eq!(state.revision(), 2);
        state.apply(minimal_valid_config(), 2);
        assert_eq!(state.revision(), 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_sync_state_hash_changes_on_different_config() {
        let dir = test_dir();
        let config_path = dir.join("config.toml");
        let mut state = ConfigState::load(&config_path).unwrap();
        let initial_hash = *state.config_hash();

        let config_with_model = MeshConfig {
            version: Some(1),
            gpu: GpuConfig {
                assignment: GpuAssignment::Auto,
            },
            models: vec![crate::plugin::ModelConfigEntry {
                model: "test.gguf".to_string(),
                mmproj: None,
                ctx_size: None,
            }],
            plugins: vec![],
        };
        state.apply(config_with_model, 0);
        let new_hash = *state.config_hash();
        assert_ne!(
            initial_hash, new_hash,
            "hash must change when config changes"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
