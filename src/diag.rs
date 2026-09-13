//! Runtime observation shared by consumers' logs and diagnostics.
//!
//! ort does not expose the library it resolved internally, so `dylib_path`
//! reports the `ORT_DYLIB_PATH` environment value (what ort will use when
//! set) — `None` means the loader used its own search order.

use crate::probe::cuda_available;

/// A snapshot of the ONNX Runtime situation in this process.
#[derive(Clone, Debug)]
pub struct OrtReport {
    /// `ORT_DYLIB_PATH` value if set (None = ort's own search order).
    pub dylib_path: Option<String>,
    /// Whether the loaded library exposes a usable CUDA execution provider.
    pub cuda_usable: bool,
}

/// Observe the current ONNX Runtime situation. Reading the env var is pure;
/// the CUDA probe may force library initialization (same as any ort call).
pub fn report() -> OrtReport {
    OrtReport {
        dylib_path: std::env::var("ORT_DYLIB_PATH").ok(),
        cuda_usable: cuda_available(),
    }
}
