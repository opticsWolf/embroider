//! Device requests and session tuning policy.
//!
//! Policy is data, chosen by the caller — never a hardcoded default.
//! `SessionPolicy::text_embed()` carries the measured tuning for
//! text-embedding sessions (benchmark table in the README);
//! `ort_defaults()` leaves ONNX at its own defaults (what bobine's vision
//! sessions use today — adopted with zero behavior change).

use crate::error::{oe, Result};
use crate::probe::cuda_available;
use anyhow::anyhow;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

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

    /// Apply this policy to a session builder. `None` fields leave the
    /// ORT default in place — `ort_defaults().apply(b)` is a no-op that
    /// exists to make the *choice* of untuned sessions explicit and
    /// swappable per slot.
    pub fn apply(&self, mut builder: SessionBuilder) -> Result<SessionBuilder> {
        if let Some(level) = self.opt_level {
            builder = oe(builder.with_optimization_level(level))?;
        }
        if let Some(n) = self.intra_threads {
            builder = oe(builder.with_intra_threads(n))?;
        }
        if let Some(n) = self.inter_threads {
            builder = oe(builder.with_inter_threads(n))?;
        }
        Ok(builder)
    }
}

/// Weight precision request. `Auto` follows the *resolved* device
/// (CUDA → FP16, CPU → FP32) so a flat default can never land the
/// FP16-weights-on-CPU combination, which runs >40x slower than FP32-CPU
/// (emulated half-precision kernels — measured, not theorised). Explicit
/// `Fp16` on a CPU-resolved session warns at open but is honoured: slow
/// is not corrupt, and refusing would break heterogeneous fleets that
/// share one config.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Precision {
    Auto,
    Fp32,
    Fp16,
    /// Dynamic-int8 weights (explicit opt-in only). `Auto` never resolves
    /// here: int8 trades speed for size on CPU (~half the tok/s of FP32)
    /// and is a deployment choice, not a device-following default. Only
    /// models with a MEASURED int8 artifact ship one (nano probe: rank
    /// kept @0.99980, zero top-5 flips); unlisted pairs fall back to fp32.
    Int8,
}

impl Precision {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fp32" | "float32" => Ok(Self::Fp32),
            "fp16" | "float16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8),
            other => Err(anyhow!(
                "precision must be 'auto', 'fp32', 'fp16' or 'int8', got '{other}'"
            )),
        }
    }

    /// Resolve against the device the session actually landed on.
    /// Must be called with the `used_cuda` from `resolve_provider_names`
    /// — never with the request — so CUDA-requested-but-missing
    /// degrades to FP32 weights instead of stranding FP16 on CPU.
    pub fn resolve(self, used_cuda: bool) -> Self {
        match self {
            Self::Auto => {
                if used_cuda {
                    Self::Fp16
                } else {
                    Self::Fp32
                }
            }
            explicit => explicit,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
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

    #[test]
    fn precision_parse_accepts_aliases_case_insensitively() {
        assert_eq!(Precision::parse("auto").unwrap(), Precision::Auto);
        assert_eq!(Precision::parse("FP32").unwrap(), Precision::Fp32);
        assert_eq!(Precision::parse("float16").unwrap(), Precision::Fp16);
    }

    #[test]
    fn precision_parse_rejects_unknown_with_message() {
        let err = Precision::parse("int4").unwrap_err().to_string();
        assert!(err.contains("precision must be"), "{err}");
        assert!(err.contains("'int4'"), "{err}");
        // int8 graduated from rejected to explicit opt-in (nano probe).
        assert_eq!(Precision::parse("INT8").unwrap(), Precision::Int8);
        assert_eq!(Precision::Int8.resolve(true), Precision::Int8);
        assert_eq!(Precision::Int8.resolve(false), Precision::Int8);
    }

    #[test]
    fn precision_auto_follows_resolved_device() {
        assert_eq!(Precision::Auto.resolve(true), Precision::Fp16);
        assert_eq!(Precision::Auto.resolve(false), Precision::Fp32);
        // Explicit survives resolution untouched — even the slow combo.
        assert_eq!(Precision::Fp16.resolve(false), Precision::Fp16);
        assert_eq!(Precision::Fp32.resolve(true), Precision::Fp32);
    }
}
