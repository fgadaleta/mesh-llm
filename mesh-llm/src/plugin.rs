mod config;
mod manager;
mod runtime;
mod support;
mod transport;

use anyhow::{bail, Context, Result};
pub use mesh_llm_plugin::proto;
use mesh_llm_plugin::{MeshVisibility, STARTUP_DISABLED_ERROR_CODE};
use rmcp::model::{
    CallToolResult as McpCallToolResult, InitializeRequestParams, ListToolsResult,
    PaginatedRequestParams, ServerInfo,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::{mpsc, oneshot, Mutex};

pub use self::config::{
    config_path, load_config, resolve_plugins, ExternalPluginSpec, PluginHostMode, ResolvedPlugins,
};
pub(crate) use self::support::parse_optional_json;
#[cfg(test)]
use self::transport::make_instance_id;
#[cfg(all(test, unix))]
use self::transport::unix_socket_path;

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
pub(crate) const PROTOCOL_VERSION: u32 = mesh_llm_plugin::PROTOCOL_VERSION;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;
const HEALTH_CHECK_INTERVAL_SECS: u64 = 15;

#[derive(Clone, Debug)]
pub enum PluginMeshEvent {
    Channel {
        plugin_id: String,
        message: proto::ChannelMessage,
    },
    BulkTransfer {
        plugin_id: String,
        message: proto::BulkTransferMessage,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

#[derive(Clone, Debug)]
pub struct ToolCallResult {
    pub content_json: String,
    pub is_error: bool,
}

#[derive(Clone, Debug)]
pub struct RpcResult {
    pub result_json: String,
}

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub trait PluginRpcBridge: Send + Sync {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>>;

    fn handle_notification(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<()>;
}

#[derive(Clone, Debug, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<PluginManagerInner>,
}

struct PluginManagerInner {
    plugins: BTreeMap<String, ExternalPlugin>,
    inactive: BTreeMap<String, PluginSummary>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
}

struct ExternalPlugin {
    spec: ExternalPluginSpec,
    instance_id: String,
    host_mode: PluginHostMode,
    summary: Arc<Mutex<PluginSummary>>,
    server_info: Arc<Mutex<Option<ServerInfo>>>,
    runtime: Arc<Mutex<Option<PluginRuntime>>>,
    mesh_tx: mpsc::Sender<PluginMeshEvent>,
    rpc_bridge: Arc<Mutex<Option<Arc<dyn PluginRpcBridge>>>>,
    restart_lock: Arc<Mutex<()>>,
    next_request_id: AtomicU64,
    next_generation: AtomicU64,
}

struct PluginRuntime {
    generation: u64,
    _child: Child,
    outbound_tx: mpsc::Sender<proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => crate::plugins::blackboard::run_plugin(name).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

fn proto_mesh_visibility(mesh_visibility: MeshVisibility) -> i32 {
    match mesh_visibility {
        MeshVisibility::Private => proto::MeshVisibility::Private as i32,
        MeshVisibility::Public => proto::MeshVisibility::Public as i32,
    }
}

#[cfg(test)]
mod tests {
    use super::config::{MeshConfig, PluginConfigEntry};
    use super::*;

    fn private_host_mode() -> PluginHostMode {
        PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        }
    }

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default(), private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_can_be_disabled() {
        let config = MeshConfig {
            self_update: None,
            plugins: vec![PluginConfigEntry {
                name: BLACKBOARD_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert!(resolved.externals.is_empty());
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn blackboard_is_resolved_on_public_meshes() {
        let resolved = resolve_plugins(
            &MeshConfig::default(),
            PluginHostMode {
                mesh_visibility: MeshVisibility::Public,
            },
        )
        .unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            self_update: None,
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
            }],
        };
        let resolved = resolve_plugins(&config, private_host_mode()).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[1].name, "demo");
        assert!(resolved.inactive.is_empty());
    }

    #[test]
    fn self_update_defaults_to_enabled() {
        assert!(MeshConfig::default().self_update_enabled());
        let config: MeshConfig = toml::from_str("self_update = false").unwrap();
        assert!(!config.self_update_enabled());
    }

    #[test]
    fn instance_ids_include_pid_and_random_suffix() {
        let instance_id = make_instance_id();
        let prefix = format!("p{}-", std::process::id());
        assert!(instance_id.starts_with(&prefix));
        assert_eq!(instance_id.len(), prefix.len() + 8);
        assert!(instance_id[prefix.len()..]
            .chars()
            .all(|ch| ch.is_ascii_hexdigit()));
    }

    #[cfg(unix)]
    #[test]
    fn unix_socket_path_is_namespaced_by_instance_id() {
        let path = unix_socket_path("p1234-deadbeef", "Pipes").unwrap();
        assert_eq!(
            path.file_name().and_then(|value| value.to_str()),
            Some("p1234-deadbeef-Pipes.sock")
        );
    }

    #[cfg(windows)]
    #[test]
    fn windows_pipe_name_is_namespaced_by_instance_id() {
        assert_eq!(
            windows_pipe_name("p1234-deadbeef", "Pipes"),
            r"\\.\pipe\mesh-llm-p1234-deadbeef-Pipes"
        );
    }
}
