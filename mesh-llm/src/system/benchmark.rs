use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::system::hardware::{GpuFacts, HardwareSurvey};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkOutput {
    pub device: String,
    pub buffer_mb: u32,
    pub runs: u32,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    pub noise_pct: f64,
    pub runtime_s: f64,
    pub rated_gbps: Option<f64>,
    pub rated_estimated: Option<bool>,
    pub efficiency_pct: Option<f64>,
    pub bus_width_bits: Option<u32>,
    pub mem_clock_mhz: Option<u64>,
    pub gcn_arch: Option<String>,
    pub hbm: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkFingerprint {
    pub gpus: Vec<GpuFacts>,
    pub is_soc: bool,
    pub timestamp_secs: u64,
}

/// Returns true if the current hardware differs from the fingerprint's recorded hardware.
/// Compares GPU names, VRAM sizes (by index), and the is_soc flag.
pub fn hardware_changed(fingerprint: &BenchmarkFingerprint, hw: &HardwareSurvey) -> bool {
    if fingerprint.is_soc != hw.is_soc {
        return true;
    }

    let hw_gpus = &hw.gpus;

    if fingerprint.gpus.len() != hw_gpus.len() {
        return true;
    }

    for (i, cached) in fingerprint.gpus.iter().enumerate() {
        if cached.name != hw_gpus[i].name || cached.vram_bytes != hw_gpus[i].vram_bytes {
            return true;
        }
    }
    false
}

/// Returns `~/.mesh-llm/benchmark-fingerprint.json`.
/// Falls back to the platform temp directory if home dir is unavailable.
pub fn fingerprint_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join(".mesh-llm")
        .join("benchmark-fingerprint.json")
}

fn benchmark_binary_name_for(os: &str, base: &str) -> String {
    if os == "windows" {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn detect_benchmark_binary_for(os: &str, hw: &HardwareSurvey, bin_dir: &Path) -> Option<PathBuf> {
    if hw.gpus.is_empty() {
        tracing::debug!("no GPUs detected — skipping benchmark");
        return None;
    }

    let gpu_upper = hw
        .gpus
        .first()
        .map(|g| g.name.to_uppercase())
        .unwrap_or_default();

    let candidate_name = if os == "macos" && hw.is_soc {
        benchmark_binary_name_for(os, "membench-fingerprint")
    } else if os == "linux" || os == "windows" {
        if gpu_upper.contains("NVIDIA") {
            benchmark_binary_name_for(os, "membench-fingerprint-cuda")
        } else if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            benchmark_binary_name_for(os, "membench-fingerprint-hip")
        } else if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!("Intel Arc benchmark is unvalidated — results may be inaccurate");
            benchmark_binary_name_for(os, "membench-fingerprint-intel")
        } else if os == "linux" && hw.is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA — attempting");
            benchmark_binary_name_for(os, "membench-fingerprint-cuda")
        } else {
            let gpu_name = hw.gpus.first().map(|g| g.name.as_str()).unwrap_or("");
            tracing::warn!(
                "could not identify benchmark binary for this GPU platform: {:?}",
                gpu_name
            );
            return None;
        }
    } else {
        let gpu_name = hw.gpus.first().map(|g| g.name.as_str()).unwrap_or("");
        tracing::warn!(
            "could not identify benchmark binary for this GPU platform: {:?}",
            gpu_name
        );
        return None;
    };

    let candidate = bin_dir.join(&candidate_name);
    if candidate.exists() {
        return Some(candidate);
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let fallback = exe_dir.join(&candidate_name);
            if fallback.exists() {
                return Some(fallback);
            }
        }
    }

    tracing::warn!(
        "{candidate_name} not found in {:?} or adjacent to mesh-llm executable",
        bin_dir
    );
    None
}

/// Load a `BenchmarkFingerprint` from disk.  Returns `None` on any error.
pub fn load_fingerprint(path: &Path) -> Option<BenchmarkFingerprint> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Atomically write a `BenchmarkFingerprint` to disk.
/// Uses a `.json.tmp` staging file + rename for crash safety.
/// Logs a warning on failure — never panics.
pub fn save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) {
    let tmp = path.with_extension("json.tmp");

    if let Err(e) = std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new("."))) {
        tracing::warn!("benchmark: failed to create cache dir: {e}");
        return;
    }

    let json = match serde_json::to_string_pretty(fp) {
        Ok(j) => j,
        Err(e) => {
            tracing::warn!("benchmark: failed to serialize fingerprint: {e}");
            return;
        }
    };

    if let Err(e) = std::fs::write(&tmp, &json) {
        tracing::warn!("benchmark: failed to write tmp fingerprint: {e}");
        return;
    }

    if let Err(e) = std::fs::rename(&tmp, path) {
        tracing::warn!("benchmark: failed to rename fingerprint into place: {e}");
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Determine which benchmark binary to use for the current hardware platform.
///
/// Returns `None` (soft failure) if:
/// - No GPUs are present
/// - The binary is not found on disk
/// - The platform/GPU combination is unrecognised
///
/// Never panics or hard-fails with `ensure!`.
pub fn detect_benchmark_binary(hw: &HardwareSurvey, bin_dir: &Path) -> Option<PathBuf> {
    detect_benchmark_binary_for(std::env::consts::OS, hw, bin_dir)
}

/// Parse raw stdout bytes from a benchmark run into a vec of per-device outputs.
///
/// Expects a JSON array of [`BenchmarkOutput`].  Returns `None` on any parse
/// failure or if the device list is empty.
pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    match serde_json::from_slice::<Vec<BenchmarkOutput>>(stdout) {
        Ok(outputs) if !outputs.is_empty() => Some(outputs),
        Ok(_) => {
            tracing::debug!("benchmark returned empty device list");
            None
        }
        Err(err) => {
            if let Ok(val) = serde_json::from_slice::<serde_json::Value>(stdout) {
                if let Some(msg) = val.get("error").and_then(|v| v.as_str()) {
                    tracing::warn!("benchmark reported error: {msg}");
                    return None;
                }
            }
            tracing::warn!("failed to parse benchmark output: {err}");
            None
        }
    }
}

/// Run the benchmark binary synchronously and return per-device outputs.
///
/// Spawns the binary as a subprocess and polls for completion up to `timeout`.
/// If the process exceeds the timeout, it is killed to avoid zombie processes.
///
/// Designed to be called inside `tokio::task::spawn_blocking` — never `async`.
pub fn run_benchmark(binary: &Path, timeout: Duration) -> Option<Vec<BenchmarkOutput>> {
    use std::io::Read;

    let mut child = match std::process::Command::new(binary)
        .arg("--json")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to spawn {binary:?}: {e}");
            return None;
        }
    };

    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    tracing::warn!("benchmark timed out after {timeout:?}, killing subprocess");
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                tracing::error!("error waiting for benchmark: {e}");
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };

    if !status.success() {
        tracing::warn!("benchmark exited with {:?}", status);
        return None;
    }

    let mut stdout_bytes = Vec::new();
    if let Some(mut pipe) = child.stdout.take() {
        let _ = pipe.read_to_end(&mut stdout_bytes);
    }
    parse_benchmark_output(&stdout_bytes)
}

/// Load a cached fingerprint if hardware is unchanged, otherwise run the
/// benchmark binary and persist the result.
///
/// Not `async` — intended for use inside `tokio::task::spawn_blocking`.
pub fn run_or_load(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Option<Vec<GpuFacts>> {
    let path = fingerprint_path();

    if let Some(ref cached) = load_fingerprint(&path) {
        if !hardware_changed(cached, hw) {
            tracing::info!(
                "Using cached bandwidth fingerprint: {} GPUs",
                cached.gpus.len()
            );
            return Some(cached.gpus.clone());
        }
    }

    tracing::info!("Hardware changed or no cache — running memory bandwidth benchmark");

    let binary = detect_benchmark_binary(hw, bin_dir)?;
    let outputs = run_benchmark(&binary, timeout)?;

    let count = if hw.gpus.is_empty() {
        outputs.len()
    } else {
        outputs.len().min(hw.gpus.len())
    };

    let gpus: Vec<GpuFacts> = (0..count)
        .map(|i| GpuFacts {
            index: i,
            name: hw.gpus.get(i).map(|g| g.name.clone()).unwrap_or_default(),
            vram_bytes: hw.gpus.get(i).map(|g| g.vram_bytes).unwrap_or(0),
            bandwidth_gbps: Some(outputs[i].p90_gbps),
        })
        .collect();

    let fingerprint = BenchmarkFingerprint {
        gpus: gpus.clone(),
        is_soc: hw.is_soc,
        timestamp_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    save_fingerprint(&path, &fingerprint);
    Some(gpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_survey(gpus: Vec<GpuFacts>, is_soc: bool) -> HardwareSurvey {
        HardwareSurvey {
            gpus,
            is_soc,
            ..Default::default()
        }
    }

    fn make_fingerprint(gpus: Vec<GpuFacts>, is_soc: bool) -> BenchmarkFingerprint {
        BenchmarkFingerprint {
            gpus,
            is_soc,
            timestamp_secs: 0,
        }
    }

    fn gpu(index: usize, name: &str, vram_bytes: u64) -> GpuFacts {
        GpuFacts {
            index,
            name: name.to_string(),
            vram_bytes,
            bandwidth_gbps: None,
        }
    }

    fn gpu_with_bw(index: usize, name: &str, vram_bytes: u64, bw: f64) -> GpuFacts {
        GpuFacts {
            index,
            name: name.to_string(),
            vram_bytes,
            bandwidth_gbps: Some(bw),
        }
    }

    #[test]
    fn test_hardware_changed_same() {
        let hw = make_survey(vec![gpu(0, "A100", 80_000_000_000)], false);
        let fp = make_fingerprint(vec![gpu_with_bw(0, "A100", 80_000_000_000, 1948.7)], false);
        assert!(!hardware_changed(&fp, &hw));
    }

    #[test]
    fn test_hardware_changed_vram() {
        let hw = make_survey(vec![gpu(0, "A100", 40_000_000_000)], false);
        let fp = make_fingerprint(vec![gpu_with_bw(0, "A100", 80_000_000_000, 1948.7)], false);
        assert!(hardware_changed(&fp, &hw));
    }

    #[test]
    fn test_hardware_changed_gpu_count() {
        let hw = make_survey(
            vec![
                gpu(0, "A100", 80_000_000_000),
                gpu(1, "A100", 80_000_000_000),
            ],
            false,
        );
        let fp = make_fingerprint(vec![gpu_with_bw(0, "A100", 80_000_000_000, 1948.7)], false);
        assert!(hardware_changed(&fp, &hw));
    }

    #[test]
    fn test_hardware_changed_soc_flag() {
        let hw = make_survey(vec![], false);
        let fp = make_fingerprint(vec![], true);
        assert!(hardware_changed(&fp, &hw));
    }

    #[test]
    fn test_benchmark_output_deserialize_cuda_single() {
        let json_str = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].p90_gbps, 1948.7);
    }

    #[test]
    fn test_benchmark_output_deserialize_multi_gpu() {
        let json_str = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs: Vec<BenchmarkOutput> = serde_json::from_str(json_str).expect("should parse");
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_benchmark_output_deserialize_error_json() {
        let json_str = r#"{"error":"No CUDA-capable device found"}"#;
        let result = serde_json::from_str::<Vec<BenchmarkOutput>>(json_str);
        assert!(result.is_err(), "expected Err, got Ok");
    }

    #[test]
    fn test_parse_benchmark_output_single_gpu() {
        let json = r#"[{"device":"NVIDIA A100-SXM4-80GB","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215}]"#;
        let result = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].p90_gbps, 1948.7);
    }

    #[test]
    fn test_parse_benchmark_output_multi_gpu_sum() {
        let json = r#"[{"device":"NVIDIA A100","buffer_mb":512,"runs":20,"p50_gbps":1935.2,"p90_gbps":1948.7,"noise_pct":0.4,"runtime_s":1.23,"rated_gbps":2000,"rated_estimated":false,"efficiency_pct":96.8,"bus_width_bits":5120,"mem_clock_mhz":1215},{"device":"NVIDIA A6000","buffer_mb":512,"runs":20,"p50_gbps":768.0,"p90_gbps":780.1,"noise_pct":0.6,"runtime_s":1.15,"rated_gbps":768,"rated_estimated":false,"efficiency_pct":100.0,"bus_width_bits":384,"mem_clock_mhz":2000}]"#;
        let outputs = parse_benchmark_output(json.as_bytes()).expect("should return Some");
        assert_eq!(outputs.len(), 2);
        let sum: f64 = outputs.iter().map(|o| o.p90_gbps).sum();
        assert!(
            (sum - 2728.8_f64).abs() < 0.01,
            "expected ~2728.8, got {sum}"
        );
    }

    #[test]
    fn test_parse_benchmark_output_error_json() {
        let json = r#"{"error": "No CUDA devices found"}"#;
        let result = parse_benchmark_output(json.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_benchmark_output_empty_array() {
        let result = parse_benchmark_output(b"[]");
        assert!(result.is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_gpu_count_zero() {
        let hw = HardwareSurvey::default();
        let result = detect_benchmark_binary(&hw, Path::new("/tmp"));
        assert!(result.is_none());
    }

    #[test]
    fn test_benchmark_binary_name_for_windows() {
        assert_eq!(
            benchmark_binary_name_for("windows", "membench-fingerprint-cuda"),
            "membench-fingerprint-cuda.exe"
        );
        assert_eq!(
            benchmark_binary_name_for("linux", "membench-fingerprint-cuda"),
            "membench-fingerprint-cuda"
        );
    }

    #[test]
    fn test_detect_benchmark_binary_windows_cuda_missing_is_soft_failure() {
        let hw = make_survey(vec![gpu(0, "NVIDIA RTX 4090", 24_000_000_000)], false);
        assert!(detect_benchmark_binary_for("windows", &hw, Path::new("C:\\bench")).is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_windows_hip_missing_is_soft_failure() {
        let hw = make_survey(
            vec![gpu(0, "AMD Radeon RX 7900 XTX", 24_000_000_000)],
            false,
        );
        assert!(detect_benchmark_binary_for("windows", &hw, Path::new("C:\\bench")).is_none());
    }

    #[test]
    fn test_detect_benchmark_binary_windows_intel_missing_is_soft_failure() {
        let hw = make_survey(vec![gpu(0, "Intel Arc A770", 16_000_000_000)], false);
        assert!(detect_benchmark_binary_for("windows", &hw, Path::new("C:\\bench")).is_none());
    }

    #[test]
    fn test_hardware_changed_gpu_name() {
        let hw = make_survey(vec![gpu(0, "NVIDIA A6000", 80_000_000_000)], false);
        let fp = make_fingerprint(
            vec![gpu_with_bw(0, "NVIDIA A100", 80_000_000_000, 1948.7)],
            false,
        );
        assert!(
            hardware_changed(&fp, &hw),
            "name change should trigger hardware_changed"
        );
    }

    #[test]
    fn test_fingerprint_cache_roundtrip() {
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-roundtrip.json");
        let fp = make_fingerprint(
            vec![gpu_with_bw(0, "NVIDIA A100", 80_000_000_000, 1948.7)],
            false,
        );
        save_fingerprint(&path, &fp);
        let loaded = load_fingerprint(&path).expect("fingerprint should round-trip");
        let _ = std::fs::remove_file(&path);

        let hw = make_survey(vec![gpu(0, "NVIDIA A100", 80_000_000_000)], false);
        assert!(
            !hardware_changed(&loaded, &hw),
            "same hardware should not trigger hardware_changed after round-trip"
        );
    }

    #[test]
    fn test_old_cache_format_fails_parse() {
        let old_json = r#"{
            "hardware_key": {
                "gpu_count": 1,
                "gpu_vram": [80000000000],
                "gpu_name": "NVIDIA A100",
                "is_soc": false
            },
            "mem_bandwidth_gbps": 1948.7,
            "p50_gbps": 1935.2,
            "timestamp_secs": 1700000000
        }"#;
        let path = std::env::temp_dir().join("mesh-llm-test-fingerprint-old-format.json");
        std::fs::write(&path, old_json).expect("write should succeed");
        let result = load_fingerprint(&path);
        let _ = std::fs::remove_file(&path);
        assert!(
            result.is_none(),
            "old cache format should fail to parse and return None"
        );
    }

    #[test]
    fn test_fingerprint_path_filename() {
        let path = fingerprint_path();
        assert!(
            path.ends_with("benchmark-fingerprint.json"),
            "fingerprint_path() should use 'benchmark-fingerprint.json', got {:?}",
            path.file_name()
        );
        let parent = path.parent().expect("path should have parent");
        assert!(
            parent.ends_with(".mesh-llm"),
            "fingerprint should be under .mesh-llm directory, got {:?}",
            parent
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_run_benchmark_kills_on_timeout() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join("mesh-llm-test-bm-timeout");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create test dir");

        let script = dir.join("hang.sh");
        std::fs::write(&script, "#!/bin/sh\nsleep 999\n").expect("write script");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
            .expect("set permissions");

        let start = std::time::Instant::now();
        let result = run_benchmark(&script, Duration::from_secs(1));
        let elapsed = start.elapsed();

        let _ = std::fs::remove_dir_all(&dir);

        assert!(
            result.is_none(),
            "hanging benchmark should return None on timeout"
        );
        assert!(
            elapsed < Duration::from_secs(5),
            "should kill subprocess promptly, took {:?}",
            elapsed
        );
    }
}
