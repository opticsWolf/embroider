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
/// default in place. `parallel` selects the parallel executor
/// (`SetSessionExecutionMode(ORT_PARALLEL)`) instead of the default
/// sequential one — per-model, not global: only the workload whose
/// graph needs it pays the scheduling change.
#[derive(Clone, Copy, Debug)]
pub struct SessionPolicy {
    pub opt_level: Option<GraphOptimizationLevel>,
    pub intra_threads: Option<usize>,
    pub inter_threads: Option<usize>,
    pub parallel: bool,
}

impl SessionPolicy {
    /// Measured policy for rerank (pair-scoring) sessions. Starts from the
    /// text-embedding tuning (Level3, intra = physical-cores/2, inter = 1);
    /// re-measure on short-pair workloads before retuning — the named
    /// constructor exists so rerank tuning can diverge without touching
    /// embedding sessions.
    pub fn rerank_pair() -> Self {
        Self::text_embed()
    }

    /// Measured policy for decide (typed-decision) sessions. Same start
    /// as rerank — short-sequence forward passes on the same runtime;
    /// re-measure on decision workloads before retuning, so decide tuning
    /// can diverge without touching embedding or rerank sessions.
    pub fn decide_pair() -> Self {
        Self::text_embed()
    }

    /// Measured policy for decider-2b (slot-layout decision) sessions:
    /// Level3 + the parallel executor, DEFAULT thread pools. The
    /// sequential planner aliases a shape buffer on this graph (ORT
    /// #4829 class — a chunk-scan Reshape reads a zeroed dim and the
    /// run fails), and the C-level `enable_mem_reuse=false` that fixes
    /// it is not reachable from the C API. Parallel execution is the
    /// sanctioned equivalent — BUT only with default pools: an explicit
    /// `inter_op_num_threads=1` re-triggers the corruption under the
    /// parallel executor (probed: inter=1 fails, inter=default matches
    /// on identical rows), so this policy sets no thread counts. Both
    /// devices are e2e-pinned; do not retune without re-running parity.
    pub fn decider2b() -> Self {
        Self {
            opt_level: Some(ort::session::builder::GraphOptimizationLevel::Level3),
            intra_threads: None,
            inter_threads: None,
            parallel: true,
        }
    }

    /// Measured policy for Laya (marker-layout decision) sessions. Same
    /// start as decide — the rows are short either way; re-measure on
    /// marker workloads (5-input collation, wider batches) before
    /// retuning, so Laya tuning can diverge without touching GLiClass.
    pub fn laya() -> Self {
        Self::text_embed()
    }

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
            parallel: false,
        }
    }

    /// ORT defaults — for consumers that never tuned their sessions.
    pub fn ort_defaults() -> Self {
        Self { opt_level: None, intra_threads: None, inter_threads: None, parallel: false }
    }

    /// Apply this policy to a session builder. `None` fields leave the
    /// ORT default in place — `ort_defaults().apply(b)` is a no-op that
    /// exists to make the *choice* of untuned sessions explicit and
    /// swappable per slot.
    pub fn apply(&self, mut builder: SessionBuilder) -> Result<SessionBuilder> {
        if let Some(level) = self.opt_level {
            builder = oe(builder.with_optimization_level(level))?;
        }
        if self.parallel {
            builder = oe(builder.with_parallel_execution(true))?;
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
    /// Weight-only int8 (MatMulNBits, explicit opt-in only). `Auto`
    /// never resolves here either: q8 trades download/RAM for a small,
    /// measured drift (decider-2b worst 0.0020, v4-edge 0.090, mmarco
    /// nDCG within 0.016) and is a deployment choice. Only models with a
    /// MEASURED q8 artifact ship one (0 flips on the reference set;
    /// GLiClass-small failed and stays fp32-only); unlisted pairs fall
    /// back to fp32. Unlike `Fp16`, q8 runs on any device (activations
    /// stay fp32, no emulated kernels).
    Q8,
}

impl Precision {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "fp32" | "float32" => Ok(Self::Fp32),
            "fp16" | "float16" => Ok(Self::Fp16),
            "int8" => Ok(Self::Int8),
            "q8" => Ok(Self::Q8),
            other => Err(anyhow!(
                "precision must be 'auto', 'fp32', 'fp16', 'int8' or 'q8', got '{other}'"
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
            Self::Q8 => "q8",
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
        // q8 likewise: explicit opt-in, device-independent (activations
        // stay fp32), Auto never yields it.
        assert_eq!(Precision::parse("Q8").unwrap(), Precision::Q8);
        assert_eq!(Precision::Q8.resolve(true), Precision::Q8);
        assert_eq!(Precision::Q8.resolve(false), Precision::Q8);
        assert_eq!(Precision::Q8.as_str(), "q8");
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
