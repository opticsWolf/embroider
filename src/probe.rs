//! CUDA availability probing against the loaded ONNX Runtime library.

use ort::ep::{ExecutionProvider, CUDA};

/// Probe the loaded ORT library for a usable CUDA execution provider.
/// Uses the EP availability check — deliberately NOT a session-builder
/// registration probe: in ort 2.0.0-rc.13 `with_execution_providers` returns
/// `Ok` even against a CPU-only dylib, which would report CUDA on every CPU
/// box. `apply_providers` still degrades gracefully if registration fails
/// despite this probe. Probed once per process.
///
/// Panic-free by contract: with load-dynamic and no resolvable dylib, the
/// first ort API call panics (`.expect`ed init). A probe reports "no CUDA"
/// instead — consumers resolve the dylib themselves before real sessions;
/// the probe must never be the thing that kills the process.
pub fn cuda_available() -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::OnceLock;
    static PROBE: OnceLock<bool> = OnceLock::new();
    *PROBE.get_or_init(|| {
        catch_unwind(AssertUnwindSafe(|| {
            CUDA::default().is_available().unwrap_or(false)
        }))
        .unwrap_or(false)
    })
}
