//! Prefix affinity and sticky routing helpers for inference target selection.

use crate::inference::election;
use iroh::EndpointId;
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const AFFINITY_TTL: Duration = Duration::from_secs(20 * 60);
const AFFINITY_MAX_ENTRIES: usize = 4096;

/// How long a remembered auto-routed model stays valid for a given session
/// key. Matches the prefix affinity TTL so sticky chats and sticky routing
/// expire in lockstep.
const AUTO_MODEL_TTL: Duration = AFFINITY_TTL;
/// Upper bound on the auto-model cache. Each entry is small (session hash +
/// model name + timestamp) so this is generous.
const AUTO_MODEL_MAX_ENTRIES: usize = 1024;

#[derive(Clone, Debug, Default, Serialize)]
pub struct AffinityStatsSnapshot {
    pub prefix_enabled: bool,
    pub sticky_enabled: bool,
    pub prefix_entries: usize,
    pub prefix_lookups: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub prefix_stale: u64,
    pub prefix_routes: u64,
    pub sticky_routes: u64,
    pub session_routes: u64,
    pub learned: u64,
    pub evicted: u64,
}

fn prefix_only_enabled() -> bool {
    std::env::var("MESH_LLM_PREFIX_ONLY")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[derive(Clone, Copy, Debug)]
struct AffinityConfig {
    prefix_enabled: bool,
    sticky_enabled: bool,
}

impl AffinityConfig {
    fn from_env() -> Self {
        Self {
            prefix_enabled: std::env::var_os("MESH_LLM_DISABLE_PREFIX_AFFINITY").is_none(),
            sticky_enabled: std::env::var_os("MESH_LLM_DISABLE_STICKY_ROUTING").is_none(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct AffinityKey {
    model: String,
    prefix_hash: u64,
}

#[derive(Clone, Debug)]
struct AffinityEntry {
    target: election::InferenceTarget,
    last_used: Instant,
}

#[derive(Clone, Debug)]
struct AutoModelEntry {
    model: String,
    last_used: Instant,
}

#[derive(Default)]
struct AffinityState {
    entries: HashMap<AffinityKey, AffinityEntry>,
    lru: VecDeque<AffinityKey>,
    stats: AffinityStatsSnapshot,
    auto_models: HashMap<u64, AutoModelEntry>,
    auto_lru: VecDeque<u64>,
}

#[derive(Clone)]
pub struct AffinityRouter {
    inner: Arc<Mutex<AffinityState>>,
    config: Arc<AffinityConfig>,
}

impl AffinityRouter {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            config: Arc::new(AffinityConfig::from_env()),
        }
    }

    #[cfg(test)]
    fn with_config(prefix_enabled: bool, sticky_enabled: bool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AffinityState::default())),
            config: Arc::new(AffinityConfig {
                prefix_enabled,
                sticky_enabled,
            }),
        }
    }

    pub fn stats_snapshot(&self) -> AffinityStatsSnapshot {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        let mut stats = state.stats.clone();
        stats.prefix_entries = state.entries.len();
        stats.prefix_enabled = self.config.prefix_enabled;
        stats.sticky_enabled = self.config.sticky_enabled;
        stats
    }

    pub fn sticky_enabled(&self) -> bool {
        self.config.sticky_enabled
    }

    pub fn record_sticky_route(&self) {
        let mut state = self.inner.lock().unwrap();
        state.stats.sticky_routes += 1;
    }

    pub fn record_session_route(&self) {
        let mut state = self.inner.lock().unwrap();
        state.stats.session_routes += 1;
    }

    pub fn lookup_target(
        &self,
        model: &str,
        prefix_hash: u64,
        candidates: &[election::InferenceTarget],
    ) -> Option<election::InferenceTarget> {
        if !self.config.prefix_enabled {
            return None;
        }
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        state.stats.prefix_lookups += 1;
        let entry = match state.entries.get(&key).cloned() {
            Some(entry) => entry,
            None => {
                state.stats.prefix_misses += 1;
                return None;
            }
        };
        if !candidates.contains(&entry.target) {
            state.remove_key(&key);
            state.stats.prefix_stale += 1;
            state.stats.prefix_misses += 1;
            return None;
        }
        state.touch_key(&key);
        if let Some(existing) = state.entries.get_mut(&key) {
            existing.last_used = Instant::now();
        }
        state.stats.prefix_hits += 1;
        state.stats.prefix_routes += 1;
        Some(entry.target)
    }

    pub fn learn_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        if !self.config.prefix_enabled
            || matches!(
                target,
                election::InferenceTarget::None
                    | election::InferenceTarget::MoeLocal(_)
                    | election::InferenceTarget::MoeRemote(_)
            )
        {
            return;
        }

        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune_expired();
        state.entries.insert(
            key.clone(),
            AffinityEntry {
                target: target.clone(),
                last_used: now,
            },
        );
        state.touch_key(&key);
        state.stats.learned += 1;
        while state.entries.len() > AFFINITY_MAX_ENTRIES {
            if let Some(oldest) = state.lru.pop_front() {
                if state.entries.remove(&oldest).is_some() {
                    state.stats.evicted += 1;
                }
            } else {
                break;
            }
        }
    }

    pub fn forget_target(&self, model: &str, prefix_hash: u64, target: &election::InferenceTarget) {
        if !self.config.prefix_enabled {
            return;
        }
        let key = AffinityKey {
            model: model.to_string(),
            prefix_hash,
        };
        let mut state = self.inner.lock().unwrap();
        if state
            .entries
            .get(&key)
            .map(|entry| &entry.target == target)
            .unwrap_or(false)
        {
            state.remove_key(&key);
            state.stats.prefix_stale += 1;
        }
    }

    /// Look up a previously-classified model name for an auto-routed session.
    ///
    /// Auto routing classifies each request and picks a model. Without
    /// memory, a follow-up turn whose classification shifts (e.g. "hi" then
    /// "write code") would get routed to a different model on a different
    /// peer with a cold KV cache. Remembering the first pick keeps the
    /// whole chat on one model, so prefix affinity actually has a chance to
    /// keep it on one peer too.
    pub fn lookup_auto_model(&self, session_key: u64) -> Option<String> {
        if !self.config.sticky_enabled {
            return None;
        }
        let mut state = self.inner.lock().unwrap();
        state.prune_auto_expired();
        let entry = state.auto_models.get(&session_key).cloned()?;
        state.touch_auto_key(session_key);
        if let Some(existing) = state.auto_models.get_mut(&session_key) {
            existing.last_used = Instant::now();
        }
        Some(entry.model)
    }

    pub fn remember_auto_model(&self, session_key: u64, model: &str) {
        if !self.config.sticky_enabled {
            return;
        }
        let mut state = self.inner.lock().unwrap();
        state.prune_auto_expired();
        state.auto_models.insert(
            session_key,
            AutoModelEntry {
                model: model.to_string(),
                last_used: Instant::now(),
            },
        );
        state.touch_auto_key(session_key);
        while state.auto_models.len() > AUTO_MODEL_MAX_ENTRIES {
            if let Some(oldest) = state.auto_lru.pop_front() {
                state.auto_models.remove(&oldest);
            } else {
                break;
            }
        }
    }

    pub fn forget_auto_model(&self, session_key: u64) {
        let mut state = self.inner.lock().unwrap();
        state.remove_auto_key(session_key);
    }

    /// Like `lookup_auto_model` but returns `None` if the cached model is
    /// currently in TTFB backoff. This unsticks sessions from drowning
    /// models so the router reclassifies and picks a responsive one.
    pub fn lookup_auto_model_if_responsive(
        &self,
        session_key: u64,
        tracker: &ModelLatencyTracker,
    ) -> Option<String> {
        let model = self.lookup_auto_model(session_key)?;
        if tracker.is_slow(&model, TTFB_SLOW_THRESHOLD) {
            tracing::debug!(
                "auto: cached model {model} is in TTFB backoff, forgetting session {session_key:016x}"
            );
            self.forget_auto_model(session_key);
            None
        } else {
            Some(model)
        }
    }
}

/// Compute the session-level key used to cache an auto-routed model choice.
///
/// Prefers an explicit session hint from the request body (e.g. an
/// OpenAI-style `user` field), then falls back to the same
/// prefix/first-user-message hash sticky routing already uses. That way
/// turn 2+ of a chat reliably maps to the same key.
pub fn auto_model_session_key(parsed_body: Option<&Value>) -> Option<u64> {
    routing_keys(parsed_body).sticky_hash
}

impl Default for AffinityRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl AffinityState {
    fn prune_expired(&mut self) {
        let now = Instant::now();

        while let Some(key) = self.lru.front() {
            let front_key = key.clone();

            match self.entries.get(&front_key) {
                Some(entry) => {
                    if now.duration_since(entry.last_used) > AFFINITY_TTL {
                        // Oldest entry is expired: evict it.
                        self.lru.pop_front();
                        if self.entries.remove(&front_key).is_some() {
                            self.stats.prefix_stale += 1;
                        }
                        // Continue to check next-oldest entry.
                    } else {
                        // Oldest entry is not expired; newer ones cannot be expired yet.
                        break;
                    }
                }
                None => {
                    // Key is in LRU but missing from entries; drop it from LRU and continue.
                    self.lru.pop_front();
                }
            }
        }
    }

    fn touch_key(&mut self, key: &AffinityKey) {
        if let Some(pos) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }

    fn remove_key(&mut self, key: &AffinityKey) {
        self.entries.remove(key);
        if let Some(pos) = self.lru.iter().position(|existing| existing == key) {
            self.lru.remove(pos);
        }
    }

    fn prune_auto_expired(&mut self) {
        let now = Instant::now();
        while let Some(key) = self.auto_lru.front().copied() {
            match self.auto_models.get(&key) {
                Some(entry) => {
                    if now.duration_since(entry.last_used) > AUTO_MODEL_TTL {
                        self.auto_lru.pop_front();
                        self.auto_models.remove(&key);
                    } else {
                        break;
                    }
                }
                None => {
                    self.auto_lru.pop_front();
                }
            }
        }
    }

    fn touch_auto_key(&mut self, key: u64) {
        if let Some(pos) = self.auto_lru.iter().position(|existing| *existing == key) {
            self.auto_lru.remove(pos);
        }
        self.auto_lru.push_back(key);
    }

    fn remove_auto_key(&mut self, key: u64) {
        self.auto_models.remove(&key);
        if let Some(pos) = self.auto_lru.iter().position(|existing| *existing == key) {
            self.auto_lru.remove(pos);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RoutingKeys {
    session_hash: Option<u64>,
    prefix_hash: Option<u64>,
    sticky_hash: Option<u64>,
}

pub struct TargetSelection {
    pub target: election::InferenceTarget,
    pub learn_prefix_hash: Option<u64>,
    pub cached_target: Option<election::InferenceTarget>,
}

pub struct PreparedTargets {
    pub ordered: Vec<election::InferenceTarget>,
    pub learn_prefix_hash: Option<u64>,
    pub cached_target: Option<election::InferenceTarget>,
}

pub(crate) fn extract_session_hint_from_body(body: &Value) -> Option<String> {
    top_level_string(body, "user").or_else(|| top_level_string(body, "session_id"))
}

fn top_level_string(body: &Value, key: &str) -> Option<String> {
    body.get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn message_text(msg: &Value) -> Option<String> {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return Some(s.to_string());
    }
    if let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) {
        let mut out = String::new();
        for block in blocks {
            if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                out.push_str(text);
                out.push('\n');
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
    }
    None
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |acc, &b| {
        (acc ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn hash_combine(a: u64, b: u64) -> u64 {
    a.wrapping_mul(31).wrapping_add(b)
}

fn hash_tagged_text(mut acc: u64, tag: &str, text: &str) -> u64 {
    acc = hash_combine(acc, hash_bytes(tag.as_bytes()));
    hash_combine(acc, hash_bytes(text.as_bytes()))
}

fn hash_json_value(mut acc: u64, value: &Value) -> u64 {
    match value {
        Value::Null => hash_combine(acc, hash_bytes(b"null")),
        Value::Bool(boolean) => {
            acc = hash_combine(acc, hash_bytes(b"bool"));
            hash_combine(acc, hash_bytes(boolean.to_string().as_bytes()))
        }
        Value::Number(number) => {
            acc = hash_combine(acc, hash_bytes(b"number"));
            hash_combine(acc, hash_bytes(number.to_string().as_bytes()))
        }
        Value::String(text) => {
            acc = hash_combine(acc, hash_bytes(b"string"));
            hash_combine(acc, hash_bytes(text.as_bytes()))
        }
        Value::Array(items) => {
            acc = hash_combine(acc, hash_bytes(b"array"));
            acc = hash_combine(acc, items.len() as u64);
            for item in items {
                acc = hash_json_value(acc, item);
            }
            acc
        }
        Value::Object(map) => {
            acc = hash_combine(acc, hash_bytes(b"object"));
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort_unstable();
            for key in keys {
                acc = hash_combine(acc, hash_bytes(key.as_bytes()));
                acc = hash_json_value(acc, &map[key]);
            }
            acc
        }
    }
}

fn hash_tagged_json(mut acc: u64, tag: &str, value: &Value) -> u64 {
    acc = hash_combine(acc, hash_bytes(tag.as_bytes()));
    hash_json_value(acc, value)
}

fn scaffold_prefix_hash_from_body(body: &Value) -> Option<u64> {
    let mut hash = 0u64;
    let mut found = false;

    for key in [
        "tools",
        "functions",
        "response_format",
        "tool_choice",
        "parallel_tool_calls",
    ] {
        if let Some(value) = body.get(key) {
            hash = hash_tagged_json(hash, key, value);
            found = true;
        }
    }

    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "system" | "developer" => {
                    if let Some(text) = message_text(msg) {
                        hash = hash_tagged_text(hash, role, &text);
                        found = true;
                    }
                }
                "user" => break,
                _ => {}
            }
        }
    }

    // Fall back to the first user message when there is no system/developer
    // prompt and no tools — plenty of real chats look like this, and without
    // a fallback the prefix cache is never populated, so turn-2+ has no way
    // to stick to the same peer and reuse its llama-server KV cache.
    if !found {
        if let Some(user_hash) = first_user_hash_from_body(body) {
            hash = hash_combine(hash, user_hash);
            found = true;
        }
    }

    found.then_some(hash)
}

fn first_user_hash_from_body(body: &Value) -> Option<u64> {
    if let Some(messages) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in messages {
            if msg.get("role").and_then(|r| r.as_str()) == Some("user") {
                return message_text(msg).map(|text| hash_tagged_text(0, "user", &text));
            }
        }
    }
    body.get("prompt")
        .and_then(|value| value.as_str())
        .map(|prompt| hash_tagged_text(0, "prompt", prompt))
}

fn routing_keys(parsed_body: Option<&Value>) -> RoutingKeys {
    let Some(body) = parsed_body else {
        return RoutingKeys::default();
    };

    let session_hash = extract_session_hint_from_body(body).map(|hint| hash_bytes(hint.as_bytes()));
    let prefix_hash = scaffold_prefix_hash_from_body(body);
    let sticky_hash = session_hash.or_else(|| {
        let mut hash = 0u64;
        let mut found = false;
        if let Some(prefix_hash) = prefix_hash {
            hash = hash_combine(hash, prefix_hash);
            found = true;
        }
        if let Some(user_hash) = first_user_hash_from_body(body) {
            hash = hash_combine(hash, user_hash);
            found = true;
        }
        found.then_some(hash)
    });

    RoutingKeys {
        session_hash,
        prefix_hash,
        sticky_hash,
    }
}

fn rotate_targets_by_hash(targets: &mut [election::InferenceTarget], key: u64) {
    if !targets.is_empty() {
        let idx = key as usize % targets.len();
        targets.rotate_left(idx);
    }
}

fn move_target_first(
    targets: &mut [election::InferenceTarget],
    target: &election::InferenceTarget,
) -> bool {
    if let Some(pos) = targets.iter().position(|candidate| candidate == target) {
        targets[..=pos].rotate_right(1);
        true
    } else {
        false
    }
}

/// Select an inference target for a model request from a caller-supplied candidate
/// list instead of pulling it from `targets`. This avoids cloning the entire
/// `ModelTargets` when the caller has already reordered the candidates (e.g. by
/// context capacity).
pub fn select_model_target_from_candidates(
    targets: &election::ModelTargets,
    candidates: &[election::InferenceTarget],
    model: &str,
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> TargetSelection {
    let routing = routing_keys(parsed_body);

    if let Some(session_hash) = routing.session_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_session_route();
        return TargetSelection {
            target: election::ModelTargets::pick_sticky_from(candidates, session_hash),
            learn_prefix_hash: None,
            cached_target: None,
        };
    }

    if let Some(prefix_hash) = routing.prefix_hash {
        if let Some(target) = affinity.lookup_target(model, prefix_hash, candidates) {
            return TargetSelection {
                target: target.clone(),
                learn_prefix_hash: Some(prefix_hash),
                cached_target: Some(target),
            };
        }

        if prefix_only_enabled() {
            return TargetSelection {
                target: election::ModelTargets::pick_sticky_from(candidates, prefix_hash),
                learn_prefix_hash: Some(prefix_hash),
                cached_target: None,
            };
        }

        if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
            affinity.record_sticky_route();
            return TargetSelection {
                target: election::ModelTargets::pick_sticky_from(candidates, sticky_hash),
                learn_prefix_hash: Some(prefix_hash),
                cached_target: None,
            };
        }

        return TargetSelection {
            target: targets.pick_from(candidates),
            learn_prefix_hash: Some(prefix_hash),
            cached_target: None,
        };
    }

    if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_sticky_route();
        return TargetSelection {
            target: election::ModelTargets::pick_sticky_from(candidates, sticky_hash),
            learn_prefix_hash: None,
            cached_target: None,
        };
    }

    TargetSelection {
        target: targets.pick_from(candidates),
        learn_prefix_hash: None,
        cached_target: None,
    }
}

pub fn prepare_remote_targets_for_request(
    model: &str,
    hosts: &[EndpointId],
    parsed_body: Option<&Value>,
    affinity: &AffinityRouter,
) -> PreparedTargets {
    let routing = routing_keys(parsed_body);
    let mut ordered: Vec<election::InferenceTarget> = hosts
        .iter()
        .copied()
        .map(election::InferenceTarget::Remote)
        .collect();
    let mut cached_target = None;
    let mut learn_prefix_hash = None;

    if let Some(session_hash) = routing.session_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_session_route();
        rotate_targets_by_hash(&mut ordered, session_hash);
        return PreparedTargets {
            ordered,
            learn_prefix_hash: None,
            cached_target: None,
        };
    }

    if let Some(prefix_hash) = routing.prefix_hash {
        learn_prefix_hash = Some(prefix_hash);
        if let Some(target) = affinity.lookup_target(model, prefix_hash, &ordered) {
            move_target_first(&mut ordered, &target);
            cached_target = Some(target);
        } else if prefix_only_enabled() {
            rotate_targets_by_hash(&mut ordered, prefix_hash);
        } else if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled())
        {
            affinity.record_sticky_route();
            rotate_targets_by_hash(&mut ordered, sticky_hash);
        }
    } else if let Some(sticky_hash) = routing.sticky_hash.filter(|_| affinity.sticky_enabled()) {
        affinity.record_sticky_route();
        rotate_targets_by_hash(&mut ordered, sticky_hash);
    }

    PreparedTargets {
        ordered,
        learn_prefix_hash,
        cached_target,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn make_id(seed: u8) -> EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn parse_body(body: &str) -> Value {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn test_extract_session_hint_from_body_user_preferred() {
        let body = parse_body(r#"{"user":"bob","session_id":"sess-1"}"#);
        assert_eq!(
            extract_session_hint_from_body(&body),
            Some("bob".to_string())
        );
    }

    #[test]
    fn test_routing_keys_prefix_shared_across_first_user_changes() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_ne!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn test_routing_keys_prefix_ignores_object_key_order() {
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run","description":"Run a command","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string"}},"required":["path","mode"]}}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"function":{"parameters":{"required":["path","mode"],"properties":{"mode":{"type":"string"},"path":{"type":"string"}},"type":"object"},"description":"Run a command","name":"run"},"type":"function"}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"fix bug B"}]}"#,
        );

        let keys_a = routing_keys(Some(&req_a));
        let keys_b = routing_keys(Some(&req_b));

        assert_eq!(keys_a.prefix_hash, keys_b.prefix_hash);
        assert_ne!(keys_a.sticky_hash, keys_b.sticky_hash);
    }

    #[test]
    fn test_select_model_target_uses_cached_prefix_target() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            "qwen".to_string(),
            vec![
                election::InferenceTarget::Remote(id_a),
                election::InferenceTarget::Remote(id_b),
            ],
        );

        let affinity = AffinityRouter::with_config(true, true);
        let req_a = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );
        let req_b = parse_body(
            r#"{"tools":[{"type":"function","function":{"name":"run"}}],"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task B"}]}"#,
        );

        let candidates = targets.candidates("qwen");
        let first = select_model_target_from_candidates(
            &targets,
            &candidates,
            "qwen",
            Some(&req_a),
            &affinity,
        );
        let prefix_hash = first.learn_prefix_hash.unwrap();
        affinity.learn_target("qwen", prefix_hash, &first.target);

        let second = select_model_target_from_candidates(
            &targets,
            &candidates,
            "qwen",
            Some(&req_b),
            &affinity,
        );
        assert_eq!(Some(second.target.clone()), second.cached_target);
        assert_eq!(first.target, second.target);
    }

    #[test]
    fn test_prepare_remote_targets_prefers_cached_host() {
        let id_a = make_id(1);
        let id_b = make_id(2);
        let hosts = vec![id_a, id_b];
        let affinity = AffinityRouter::with_config(true, true);
        let req = parse_body(
            r#"{"messages":[{"role":"system","content":"You are an agent."},{"role":"user","content":"task A"}]}"#,
        );

        let prefix_hash = routing_keys(Some(&req)).prefix_hash.unwrap();
        affinity.learn_target(
            "qwen",
            prefix_hash,
            &election::InferenceTarget::Remote(id_b),
        );

        let prepared = prepare_remote_targets_for_request("qwen", &hosts, Some(&req), &affinity);
        assert_eq!(
            prepared.ordered.first(),
            Some(&election::InferenceTarget::Remote(id_b))
        );
        assert_eq!(
            prepared.cached_target,
            Some(election::InferenceTarget::Remote(id_b))
        );
    }

    #[test]
    fn scaffold_prefix_hash_falls_back_to_first_user_message() {
        // No system/developer prompt, no tools — the old behavior returned
        // None here and the prefix cache never learned anything. Now it
        // hashes the first user message so chats without system prompts
        // can still stick to the same peer on turn 2+.
        let req = parse_body(
            r#"{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi"}]}"#,
        );
        let hash = scaffold_prefix_hash_from_body(&req);
        assert!(
            hash.is_some(),
            "expected a prefix hash for a chat with only a user message"
        );
    }

    #[test]
    fn scaffold_prefix_hash_stable_across_chat_turns() {
        // Same first user message, growing conversation — the prefix hash
        // must be identical so both turns map to the same affinity key.
        let turn_1 = parse_body(r#"{"messages":[{"role":"user","content":"tell me a joke"}]}"#);
        let turn_2 = parse_body(
            r#"{"messages":[{"role":"user","content":"tell me a joke"},{"role":"assistant","content":"why did ..."},{"role":"user","content":"another one"}]}"#,
        );
        assert_eq!(
            scaffold_prefix_hash_from_body(&turn_1),
            scaffold_prefix_hash_from_body(&turn_2),
        );
    }

    #[test]
    fn scaffold_prefix_hash_differs_between_sessions() {
        let a = parse_body(r#"{"messages":[{"role":"user","content":"topic a"}]}"#);
        let b = parse_body(r#"{"messages":[{"role":"user","content":"topic b"}]}"#);
        assert_ne!(
            scaffold_prefix_hash_from_body(&a),
            scaffold_prefix_hash_from_body(&b),
        );
    }

    #[test]
    fn auto_model_cache_round_trip() {
        let affinity = AffinityRouter::new();
        let key = 0xabcdef123456u64;
        assert_eq!(affinity.lookup_auto_model(key), None);
        affinity.remember_auto_model(key, "Qwen3.5-9B-Q4_K_M");
        assert_eq!(
            affinity.lookup_auto_model(key),
            Some("Qwen3.5-9B-Q4_K_M".to_string())
        );
    }

    #[test]
    fn auto_model_cache_forget() {
        let affinity = AffinityRouter::new();
        let key = 42u64;
        affinity.remember_auto_model(key, "some-model");
        affinity.forget_auto_model(key);
        assert_eq!(affinity.lookup_auto_model(key), None);
    }

    #[test]
    fn auto_model_cache_evicts_oldest_over_capacity() {
        let affinity = AffinityRouter::new();
        for i in 0..(AUTO_MODEL_MAX_ENTRIES as u64 + 10) {
            affinity.remember_auto_model(i, "model-x");
        }
        // The very first inserts should have been evicted.
        assert_eq!(affinity.lookup_auto_model(0), None);
        assert_eq!(affinity.lookup_auto_model(1), None);
        // Recent inserts survive.
        let recent = AUTO_MODEL_MAX_ENTRIES as u64 + 5;
        assert_eq!(
            affinity.lookup_auto_model(recent),
            Some("model-x".to_string())
        );
    }

    #[test]
    fn auto_model_session_key_matches_sticky_hash() {
        let body = parse_body(
            r#"{"messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}]}"#,
        );
        let key = auto_model_session_key(Some(&body)).expect("expected a session key");
        let sticky = routing_keys(Some(&body)).sticky_hash.unwrap();
        assert_eq!(key, sticky);
    }

    #[test]
    fn auto_model_cache_disabled_when_sticky_disabled() {
        let affinity = AffinityRouter::with_config(true, false);
        affinity.remember_auto_model(1, "model");
        assert_eq!(affinity.lookup_auto_model(1), None);
    }

    #[test]
    fn auto_model_cache_survives_forget_target_calls() {
        // forget_target operates on prefix affinity, not the auto-model
        // memo. A transient per-host prefix miss shouldn't flush the
        // session's model choice.
        let affinity = AffinityRouter::new();
        affinity.remember_auto_model(7, "chat-model");
        let target = election::InferenceTarget::Remote(make_id(5));
        affinity.forget_target("chat-model", 0xdead_beef, &target);
        assert_eq!(
            affinity.lookup_auto_model(7),
            Some("chat-model".to_string())
        );
    }
}

// ── TTFB latency tracking ───────────────────────────────────────────

/// How many recent TTFB samples to keep per model.
const TTFB_RING_SIZE: usize = 8;

/// Default threshold: a model whose recent median TTFB exceeds this is
/// considered "slow" and will be deprioritized by auto routing.
pub const TTFB_SLOW_THRESHOLD: Duration = Duration::from_secs(15);

/// First-auto timeout for small prompts with no TTFB history. A healthy
/// model answers a short prompt in 1-2s, so 15s is generous while still
/// catching a drowning host quickly.
pub const FIRST_AUTO_TIMEOUT: Duration = Duration::from_secs(15);

/// Floor for proportional timeouts when we have TTFB history.
pub const TTFB_TIMEOUT_FLOOR: Duration = Duration::from_secs(10);

/// Estimated prompt token threshold below which we use the tight
/// first-auto timeout. Above this, the model may legitimately need more
/// prefill time.
pub const SMALL_PROMPT_TOKENS: u32 = 1024;

/// TTFB multiplier for computing proportional timeouts from history.
pub const TTFB_TIMEOUT_MULTIPLIER: u32 = 3;

/// Per-model ring buffer of recent TTFB (time-to-first-byte) samples.
///
/// Used by auto routing to detect slow/overloaded models and route
/// around them. Each node tracks its own experience — nothing is
/// gossiped.
#[derive(Debug)]
struct ModelTtfbRing {
    samples: VecDeque<Duration>,
}

impl ModelTtfbRing {
    fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(TTFB_RING_SIZE),
        }
    }

    fn push(&mut self, ttfb: Duration) {
        if self.samples.len() >= TTFB_RING_SIZE {
            self.samples.pop_front();
        }
        self.samples.push_back(ttfb);
    }

    /// Median of recorded samples. Returns None if empty.
    fn median(&self) -> Option<Duration> {
        if self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<Duration> = self.samples.iter().copied().collect();
        sorted.sort();
        Some(sorted[sorted.len() / 2])
    }

    fn is_slow(&self, threshold: Duration) -> bool {
        self.median().map_or(false, |m| m >= threshold)
    }
}

/// Tracks per-model TTFB for backoff-aware routing.
///
/// Thread-safe (behind `Arc<Mutex<…>>`). Typically owned by the same
/// scope that owns the `AffinityRouter` and passed to the proxy and
/// router layers.
#[derive(Clone, Debug)]
pub struct ModelLatencyTracker {
    inner: Arc<Mutex<HashMap<String, ModelTtfbRing>>>,
}

impl ModelLatencyTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Record a TTFB sample for `model`.
    pub fn record(&self, model: &str, ttfb: Duration) {
        let mut map = self.inner.lock().unwrap();
        map.entry(model.to_string())
            .or_insert_with(ModelTtfbRing::new)
            .push(ttfb);
    }

    /// True if the model's recent median TTFB exceeds `threshold`.
    pub fn is_slow(&self, model: &str, threshold: Duration) -> bool {
        let map = self.inner.lock().unwrap();
        map.get(model).map_or(false, |ring| ring.is_slow(threshold))
    }

    /// Return the recent median TTFB for a model, if any samples exist.
    pub fn median_ttfb(&self, model: &str) -> Option<Duration> {
        let map = self.inner.lock().unwrap();
        map.get(model).and_then(|ring| ring.median())
    }

    /// Compute an appropriate first-byte timeout for a request.
    ///
    /// - No history + small prompt → tight `FIRST_AUTO_TIMEOUT` (15s)
    /// - No history + large prompt → the caller's `default_timeout`
    /// - Has history → `TTFB_TIMEOUT_MULTIPLIER × median`, floored at
    ///   `TTFB_TIMEOUT_FLOOR`, capped at `default_timeout`
    pub fn first_byte_timeout(
        &self,
        model: &str,
        estimated_prompt_tokens: Option<u32>,
        default_timeout: Duration,
    ) -> Duration {
        match self.median_ttfb(model) {
            Some(median) => {
                let proportional = median * TTFB_TIMEOUT_MULTIPLIER;
                proportional.max(TTFB_TIMEOUT_FLOOR).min(default_timeout)
            }
            None => {
                // No history: tight timeout for small prompts, default for large ones.
                let small = estimated_prompt_tokens.map_or(true, |t| t < SMALL_PROMPT_TOKENS);
                if small {
                    FIRST_AUTO_TIMEOUT
                } else {
                    default_timeout
                }
            }
        }
    }
}

/// List of model names currently in TTFB backoff. Used by the router to
/// filter candidates without passing the full tracker through the
/// classification API.
pub fn slow_models(tracker: &ModelLatencyTracker, models: &[&str]) -> Vec<String> {
    models
        .iter()
        .filter(|name| tracker.is_slow(name, TTFB_SLOW_THRESHOLD))
        .map(|name| name.to_string())
        .collect()
}

// ── TTFB tracker tests ──────────────────────────────────────────────

#[cfg(test)]
mod ttfb_tests {
    use super::*;

    #[test]
    fn records_and_reports_median() {
        let tracker = ModelLatencyTracker::new();
        tracker.record("model-a", Duration::from_secs(10));
        tracker.record("model-a", Duration::from_secs(50));
        tracker.record("model-a", Duration::from_secs(30));
        // sorted: [10, 30, 50] → median = 30
        assert_eq!(
            tracker.median_ttfb("model-a"),
            Some(Duration::from_secs(30))
        );
    }

    #[test]
    fn is_slow_above_threshold() {
        let tracker = ModelLatencyTracker::new();
        tracker.record("slow", Duration::from_secs(45));
        tracker.record("slow", Duration::from_secs(50));
        tracker.record("fast", Duration::from_secs(2));
        tracker.record("fast", Duration::from_secs(3));

        assert!(tracker.is_slow("slow", TTFB_SLOW_THRESHOLD));
        assert!(!tracker.is_slow("fast", TTFB_SLOW_THRESHOLD));
    }

    #[test]
    fn backoff_decays_with_fast_samples() {
        let tracker = ModelLatencyTracker::new();
        // Start slow
        tracker.record("model", Duration::from_secs(45));
        tracker.record("model", Duration::from_secs(50));
        assert!(tracker.is_slow("model", TTFB_SLOW_THRESHOLD));

        // Fast samples push out the slow ones over time
        for _ in 0..7 {
            tracker.record("model", Duration::from_secs(2));
        }
        assert!(!tracker.is_slow("model", TTFB_SLOW_THRESHOLD));
    }

    #[test]
    fn unknown_model_not_slow() {
        let tracker = ModelLatencyTracker::new();
        assert!(!tracker.is_slow("never-seen", TTFB_SLOW_THRESHOLD));
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let tracker = ModelLatencyTracker::new();
        // Fill the ring with slow samples
        for _ in 0..TTFB_RING_SIZE {
            tracker.record("model", Duration::from_secs(60));
        }
        assert!(tracker.is_slow("model", TTFB_SLOW_THRESHOLD));

        // Now push TTFB_RING_SIZE fast samples to fully evict the slow ones
        for _ in 0..TTFB_RING_SIZE {
            tracker.record("model", Duration::from_secs(1));
        }
        assert!(!tracker.is_slow("model", TTFB_SLOW_THRESHOLD));
    }

    #[test]
    fn first_byte_timeout_no_history_small_prompt() {
        let tracker = ModelLatencyTracker::new();
        let default = Duration::from_secs(300);
        let timeout = tracker.first_byte_timeout("model", Some(100), default);
        assert_eq!(timeout, FIRST_AUTO_TIMEOUT);
    }

    #[test]
    fn first_byte_timeout_no_history_large_prompt() {
        let tracker = ModelLatencyTracker::new();
        let default = Duration::from_secs(300);
        let timeout = tracker.first_byte_timeout("model", Some(5000), default);
        assert_eq!(timeout, default);
    }

    #[test]
    fn first_byte_timeout_with_history_proportional() {
        let tracker = ModelLatencyTracker::new();
        tracker.record("model", Duration::from_secs(5));
        tracker.record("model", Duration::from_secs(5));
        let default = Duration::from_secs(300);
        let timeout = tracker.first_byte_timeout("model", Some(100), default);
        // 3 × 5s = 15s, above floor of 10s, below cap of 300s
        assert_eq!(timeout, Duration::from_secs(15));
    }

    #[test]
    fn first_byte_timeout_respects_floor() {
        let tracker = ModelLatencyTracker::new();
        tracker.record("model", Duration::from_secs(1));
        let default = Duration::from_secs(300);
        let timeout = tracker.first_byte_timeout("model", Some(100), default);
        // 3 × 1s = 3s, but floor is 10s
        assert_eq!(timeout, TTFB_TIMEOUT_FLOOR);
    }

    #[test]
    fn first_byte_timeout_capped_at_default() {
        let tracker = ModelLatencyTracker::new();
        tracker.record("model", Duration::from_secs(200));
        let default = Duration::from_secs(300);
        let timeout = tracker.first_byte_timeout("model", Some(100), default);
        // 3 × 200s = 600s, capped at 300s
        assert_eq!(timeout, default);
    }

    #[test]
    fn slow_models_filters_correctly() {
        let tracker = ModelLatencyTracker::new();
        tracker.record("MiniMax", Duration::from_secs(45));
        tracker.record("GLM", Duration::from_secs(2));
        tracker.record("Qwen", Duration::from_secs(3));

        let names = vec!["MiniMax", "GLM", "Qwen"];
        let slow = slow_models(&tracker, &names);
        assert_eq!(slow, vec!["MiniMax".to_string()]);
    }

    #[test]
    fn auto_model_lookup_respects_backoff() {
        let affinity = AffinityRouter::new();
        let tracker = ModelLatencyTracker::new();
        affinity.remember_auto_model(0xDEAD, "slow-model");
        tracker.record("slow-model", Duration::from_secs(45));

        // lookup_auto_model_if_responsive should return None for slow model
        let result = affinity.lookup_auto_model_if_responsive(0xDEAD, &tracker);
        assert!(result.is_none());

        // But a fast model should still be returned
        affinity.remember_auto_model(0xBEEF, "fast-model");
        tracker.record("fast-model", Duration::from_secs(2));
        let result = affinity.lookup_auto_model_if_responsive(0xBEEF, &tracker);
        assert_eq!(result, Some("fast-model".to_string()));
    }
}
