//! ONNX Runtime execution-provider plumbing.
//!
//! Provider mapping and clone-and-fallback application — the pieces bobine's
//! `engine.rs` and okf-embed's `lib.rs` used to carry as near-identical
//! copies. Pure functions here are unit-testable without a model, an ORT
//! dylib, or Python.

use ort::ep::{
    ExecutionProviderDispatch, CoreML, DirectML, OpenVINO, ROCm, CPU, CUDA,
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
///
/// Legacy entry point: keeps the historical behaviour exactly (CPU arena
/// ON, the ORT default). New callers that want arena control use
/// [`apply_providers_with_arena`]. Unchanged so downstream crates that
/// depend on this function (bobine) never observe a behaviour shift.
///
/// Infallible by design: every path yields a usable builder, so callers
/// chain straight into `commit_from_file` without an error arm.
pub fn apply_providers(
    builder: ort::session::builder::SessionBuilder,
    providers: &[String],
) -> ort::session::builder::SessionBuilder {
    apply_providers_with_arena(builder, providers, true)
}

/// Provider application with explicit CPU-arena control. `cpu_arena=false`
/// registers the CPU execution provider explicitly with its arena
/// allocator disabled — measured 8x lower peak RSS (15.3→1.9 GB on the
/// FP32 text model) for ~1.4x encode time. With `cpu_arena=true` this is
/// exactly `apply_providers` (implicit CPU, arena on).
///
/// Same infallibility contract: registration failure degrades to the
/// incoming builder instead of failing model initialization.
pub fn apply_providers_with_arena(
    builder: ort::session::builder::SessionBuilder,
    providers: &[String],
    cpu_arena: bool,
) -> ort::session::builder::SessionBuilder {
    if cpu_arena {
        // Historical path, untouched: accelerators-or-nothing.
        let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
        for p in providers {
            if let ProviderMapping::Accelerator(d) = map_provider(p) {
                eps.push(d);
            }
        }
        if eps.is_empty() {
            return builder;
        }
        return try_providers(builder, &eps, true);
    }
    // Arena-off path: the CPU EP must be registered explicitly — the
    // only way to reach `DisableCpuMemArena`. It rides last so
    // accelerators keep priority and CPU stays the fallback.
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();
    for p in providers {
        if let ProviderMapping::Accelerator(d) = map_provider(p) {
            eps.push(d);
        }
    }
    eps.push(CPU::default().with_arena_allocator(false).build());
    try_providers(builder, &eps, false)
}

/// Clone-and-fallback registration attempt shared by both entry points.
/// On the arena-off path a failed accelerator set retries CPU-only
/// before giving up, so CUDA-requested-but-missing still honours the
/// arena flag instead of silently landing on the arena-enabled fallback.
fn try_providers(
    builder: ort::session::builder::SessionBuilder,
    eps: &[ExecutionProviderDispatch],
    cpu_arena: bool,
) -> ort::session::builder::SessionBuilder {
    // The clone keeps the pristine builder for fallback: only the attempt
    // carries accelerator options.
    let attempt = builder.clone();
    match attempt.with_execution_providers(eps) {
        Ok(configured) => configured,
        Err(e) => {
            if !cpu_arena && eps.len() > 1 {
                let cpu_only =
                    vec![CPU::default().with_arena_allocator(false).build()];
                let retry = builder.clone();
                match retry.with_execution_providers(&cpu_only) {
                    Ok(configured) => {
                        eprintln!(
                            "embroider: accelerator providers unavailable in this ONNX Runtime \
                             library ({e:#}); using CPU without arena"
                        );
                        return configured;
                    }
                    Err(_) => {}
                }
            }
            eprintln!(
                "embroider: accelerator providers unavailable in this ONNX Runtime \
                 library ({e:#}); using CPU"
            );
            builder
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
