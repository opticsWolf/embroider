//! Device requests and session tuning policy.
//!
//! Policy is data, chosen by the caller — never a hardcoded default.
//! `SessionPolicy::text_embed()` carries the measured tuning for
//! text-embedding sessions (benchmark table in the README);
//! `ort_defaults()` leaves ONNX at its own defaults (what bobine's vision
//! sessions use today — adopted with zero behavior change).

use crate::error::Result;
use crate::probe::cuda_available;
use anyhow::anyhow;
use ort::session::builder::GraphOptimizationLevel;

/// Inference device request. Accelerators are opportunistic:
/// requested-but-missing warns once and degrades to CPU (never fatal).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DeviceReq {
    Auto,
    Cpu,
    Cuda,
}

impl DeviceReq {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" | "gpu" => Ok(Self::Cuda),
            other => Err(anyhow!("device must be 'auto', 'cpu' or 'cuda', got '{other}'")),
        }
    }
}

/// Explicit per-workload session tuning. `None` fields leave the ORT
/// default in place.
#[derive(Clone, Copy, Debug)]
pub struct SessionPolicy {
    pub opt_level: Option<GraphOptimizationLevel>,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
}

impl SessionPolicy {
    /// Measured policy for text-embedding sessions: Level3, intra =
    /// physical-cores/2, inter = 1 (fastest of the tested configs; do not
    /// mix tuning levels inside one vector index — see the README table).
    pub fn text_embed() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        Self {
            opt_level: Some(ort::session::builder::GraphOptimizationLevel::Level3),
            intra_threads: Some(std::cmp::max(1, threads / 2)), // physical cores over logical
            inter_threads: Some(1),
        }
    }

    /// ORT defaults — for consumers that never tuned their sessions.
    pub fn ort_defaults() -> Self {
        Self { opt_level: None, intra_threads: None, inter_threads: None }
    }
}

/// Resolve the internal provider list for a device request. Accelerators
/// stay opportunistic: requested-but-missing warns once and degrades to CPU.
pub(crate) fn resolve_provider_names(device: DeviceReq) -> (Vec<String>, bool) {
    let cuda = cuda_available();
    let used_cuda = cuda && !matches!(device, DeviceReq::Cpu);
    let names = match device {
        DeviceReq::Cpu => vec![],
        DeviceReq::Auto => {
            if cuda {
                vec!["cuda".to_string()]
            } else {
                vec![]
            }
        }
        DeviceReq::Cuda if cuda => vec!["cuda".to_string()],
        DeviceReq::Cuda => {
            eprintln!(
                "embroider: CUDA requested but no CUDA execution provider in the loaded \
                 ONNX Runtime — falling back to CPU. Install onnxruntime-gpu for acceleration."
            );
            vec![]
        }
    };
    (names, used_cuda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parse_accepts_aliases_case_insensitively() {
        assert_eq!(DeviceReq::parse("auto").unwrap(), DeviceReq::Auto);
        assert_eq!(DeviceReq::parse("CPU").unwrap(), DeviceReq::Cpu);
        assert_eq!(DeviceReq::parse("cuda").unwrap(), DeviceReq::Cuda);
        assert_eq!(DeviceReq::parse("GPU").unwrap(), DeviceReq::Cuda);
    }

    #[test]
    fn device_parse_rejects_unknown_with_message() {
        let err = DeviceReq::parse("tpu").unwrap_err().to_string();
        assert!(err.contains("device must be"), "{err}");
        assert!(err.contains("'tpu'"), "{err}");
    }
}
