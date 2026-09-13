//! ONNX Runtime execution-provider plumbing.
//!
//! Provider mapping and clone-and-fallback application — the pieces bobine's
//! `engine.rs` and okf-embed's `lib.rs` used to carry as near-identical
//! copies. Pure functions here are unit-testable without a model, an ORT
//! dylib, or Python.

use crate::error::{oe, Result};
use ort::ep::{
    ExecutionProviderDispatch, CoreML, DirectML, OpenVINO, ROCm, CUDA,
};

/// Provider mapping outcome: accelerators become a dispatch, CPU stays
/// implicit (the ORT default), unknown names warn and are skipped.
pub enum ProviderMapping {
    Cpu,
    Accelerator(ExecutionProviderDispatch),
    Unknown(String),
}

/// Map a friendly provider name to its ORT dispatch (the shared provider
/// matrix: cuda/rocm/directml/openvino/coreml + implicit cpu).
pub fn map_provider(name: &str) -> ProviderMapping {
    match name.to_ascii_lowercase().as_str() {
        "cudaexecutionprovider" | "cuda" => {
            ProviderMapping::Accelerator(CUDA::default().build())
        }
        "rocmexecutionprovider" | "rocm" => {
            ProviderMapping::Accelerator(ROCm::default().build())
        }
        "directmlexecutionprovider" | "directml" => {
            ProviderMapping::Accelerator(DirectML::default().build())
        }
        "openvinoexecutionprovider" | "openvino" => {
            ProviderMapping::Accelerator(OpenVINO::default().build())
        }
        "coremlexecutionprovider" | "coreml" => {
            ProviderMapping::Accelerator(CoreML::default().build())
        }
        "cpuexecutionprovider" | "cpu" | "" => ProviderMapping::Cpu,
        other => {
            eprintln!("embroider: unknown ORT provider '{other}', skipping");
            ProviderMapping::Unknown(other.to_string())
        }
    }
}

/// Apply provider names to a session builder (clone-and-fallback).
/// CPU/unknown names contribute nothing; when no accelerator survives, the
/// builder is returned untouched. Registration failure against the loaded
/// library degrades to CPU instead of failing model initialization.
pub fn apply_providers(
    builder: ort::session::builder::SessionBuilder,
    providers: &[String],
) -> Result<ort::session::builder::SessionBuilder> {
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
    for p in providers {
        if let ProviderMapping::Accelerator(d) = map_provider(p) {
            eps.push(d);
        }
    }
    if eps.is_empty() {
        return Ok(builder);
    }
    // The clone keeps the pristine builder for fallback: only the attempt
    // carries accelerator options.
    let attempt = builder.clone();
    match oe(attempt.with_execution_providers(&eps)) {
        Ok(configured) => Ok(configured),
        Err(e) => {
            eprintln!(
                "embroider: accelerator providers unavailable in this ONNX Runtime \
                 library ({e:#}); using CPU"
            );
            Ok(builder)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_mapping_known_accelerators() {
        for name in [
            "cuda",
            "CUDAExecutionProvider",
            "rocm",
            "ROCmExecutionProvider",
            "directml",
            "DirectMLExecutionProvider",
            "openvino",
            "OpenVINOExecutionProvider",
            "coreml",
            "CoreMLExecutionProvider",
        ] {
            assert!(
                matches!(map_provider(name), ProviderMapping::Accelerator(_)),
                "{name}"
            );
        }
    }

    #[test]
    fn provider_mapping_cpu_stays_implicit() {
        for name in ["cpu", "CPU", "CPUExecutionProvider", ""] {
            assert!(matches!(map_provider(name), ProviderMapping::Cpu), "{name}");
        }
    }

    #[test]
    fn provider_mapping_unknown_is_skipped() {
        match map_provider("tpu") {
            ProviderMapping::Unknown(s) => assert_eq!(s, "tpu"),
            ProviderMapping::Cpu | ProviderMapping::Accelerator(_) => {
                panic!("tpu must map to Unknown")
            }
        }
    }
}
