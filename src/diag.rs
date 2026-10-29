//! Runtime observation shared by consumers' logs and diagnostics.
//!
//! ort does not expose the library it resolved internally, so `dylib_path`
//! reports the `ORT_DYLIB_PATH` environment value (what ort will use when
//! set) — `None` means the loader used its own search order.
//!
//! [`runtime_report`] goes one step further: it inventories the ONNX
//! Runtime installs visible from this process (the loaded dylib, its
//! sibling provider DLLs, `onnxruntime*` dists in the active venv, and
//! the CUDA/cuDNN runtime DLLs those dists ship) as JSON-serializable
//! data, so a harness can save the exact machine layout next to its
//! numbers (`save_runtime_report`). No new dependencies: everything is
//! directory scans plus the existing CUDA probe.

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

/// Filenames (case-insensitive) that count as ORT provider libraries when
/// found next to the loaded dylib or inside an `onnxruntime*` dist.
const PROVIDER_DLLS: &[&str] = &[
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_tensorrt.dll",
];

/// A site-packages install of ONNX Runtime (CPU or GPU build).
fn scan_ort_dist(dir: &std::path::Path) -> Option<serde_json::Value> {
    let capi = dir.join("capi");
    if !capi.is_dir() {
        return None;
    }
    let dlls: Vec<String> = std::fs::read_dir(&capi)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            let l = n.to_lowercase();
            (l.ends_with(".dll") || l.ends_with(".so")) && l.contains("onnxruntime")
        })
        .collect();
    if dlls.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "package": dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
        "capi": capi.to_string_lossy(),
        "dlls": dlls,
    }))
}

/// CUDA/cuDNN runtime DLLs shipped inside an `nvidia*` namespace dir
/// (pip's consolidated `nvidia/cu13`, `nvidia/cudnn`, ... layouts),
/// grouped by sub-library, two levels deep, capped so a fat env stays
/// cheap. The filenames are what ORT's CUDA EP dlopens at session
/// build time — this list is the "which CUDA files" answer.
fn scan_nvidia_dist(
    dir: &std::path::Path,
) -> Vec<(String, Vec<String>)> {
    // sublib -> dlls, e.g. "nvidia/cu13" -> [".../cublas64_13.dll", ...]
    let mut groups: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    let mut stack: Vec<(std::path::PathBuf, u8)> = vec![(dir.to_path_buf(), 0)];
    while let Some((d, depth)) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                if depth < 3 {
                    stack.push((p, depth + 1));
                }
            } else if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                let l = name.to_lowercase();
                if !(l.ends_with(".dll") || l.ends_with(".so")) {
                    continue;
                }
                if !(l.contains("cudnn")
                    || l.contains("cublas")
                    || l.contains("cudart")
                    || l.contains("cufft")
                    || l.contains("curand")
                    || l.contains("nvrtc")
                    || l.contains("nvjitlink")
                    || l.contains("cussl")
                    || l.contains("nvshmem"))
                {
                    continue;
                }
                let rel = p.strip_prefix(dir).unwrap_or(&p);
                let top = rel
                    .iter()
                    .next()
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let ns = dir
                    .file_name()
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_default();
                groups
                    .entry(if top.is_empty() { ns.clone() } else { format!("{ns}/{top}") })
                    .or_default()
                    .push(p.to_string_lossy().into_owned());
            }
        }
    }
    groups
        .into_iter()
        .map(|(mut dlls)| {
            dlls.1.sort();
            dlls.1.truncate(64);
            serde_json::json!({ "package": dlls.0, "dlls": dlls.1 })
        })
        .map(|v| {
            (
                v["package"].as_str().unwrap_or("").to_string(),
                v["dlls"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
            )
        })
        .collect()
}

/// Candidate site-packages dirs for the active interpreter, if visible.
fn site_packages_dirs() -> Vec<std::path::PathBuf> {
    let venv = match std::env::var("VIRTUAL_ENV") {
        Ok(v) => std::path::PathBuf::from(v),
        Err(_) => return Vec::new(),
    };
    let mut out = vec![venv.join("Lib").join("site-packages")];
    out.push(venv.join("lib").join("site-packages"));
    if let Ok(rd) = std::fs::read_dir(venv.join("lib")) {
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir()
                && p.file_name().and_then(|s| s.to_str()).is_some_and(|n| n.starts_with("python"))
            {
                out.push(p.join("site-packages"));
            }
        }
    }
    out.into_iter().filter(|p| p.is_dir()).collect()
}

/// Inventory the ONNX Runtime installs visible from this process as
/// JSON-serializable data: the loaded dylib (`ORT_DYLIB_PATH`), its
/// sibling provider DLLs, every `onnxruntime*` dist in the active venv
/// (CPU and GPU builds side by side), the CUDA/cuDNN runtime DLLs the
/// `nvidia*` dists ship, and the CUDA-usability probe. Pure directory
/// scans plus the existing probe — no sessions are built.
pub fn runtime_report() -> serde_json::Value {
    let dylib = std::env::var("ORT_DYLIB_PATH").ok();
    let (dylib_exists, siblings) = match &dylib {
        Some(p) => {
            let path = std::path::Path::new(p);
            let sibs = path
                .parent()
                .map(|dir| {
                    std::fs::read_dir(dir)
                        .map(|rd| {
                            rd.filter_map(|e| e.ok())
                                .map(|e| e.file_name().to_string_lossy().into_owned())
                                .filter(|n| {
                                    PROVIDER_DLLS.contains(&n.to_lowercase().as_str())
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .unwrap_or_default();
            (path.is_file(), sibs)
        }
        None => (false, Vec::new()),
    };
    let mut ort_installs = Vec::new();
    let mut cuda_runtimes = Vec::new();
    let mut seen: std::collections::HashSet<String> = Default::default();
    for sp in site_packages_dirs() {
        let Ok(rd) = std::fs::read_dir(&sp) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            if name.starts_with("onnxruntime") && !name.ends_with(".dist-info") {
                if let Some(v) = scan_ort_dist(&p) {
                    // One venv can appear under two spellings
                    // (`Lib/site-packages` vs `lib/site-packages`); dedupe
                    // by canonical dir so each install lists once.
                    let key = std::fs::canonicalize(&p)
                        .map(|c| c.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| p.to_string_lossy().into_owned());
                    if seen.insert(key) {
                        ort_installs.push(v);
                    }
                }
            } else if name.starts_with("nvidia") && !name.ends_with(".dist-info") {
                for (package, dlls) in scan_nvidia_dist(&p) {
                    if dlls.is_empty() {
                        continue;
                    }
                    // Same spelling-duplicate venv caveat as above.
                    let key = package.clone();
                    if seen.insert(key) {
                        cuda_runtimes.push(serde_json::json!({ "package": package, "dlls": dlls }));
                    }
                }
            }
        }
    }
    serde_json::json!({
        "ort_dylib": dylib,
        "ort_dylib_exists": dylib_exists,
        "sibling_provider_dlls": siblings,
        "cuda_usable": cuda_available(),
        "ort_installs": ort_installs,
        "cuda_runtime_dlls": cuda_runtimes,
        "virtual_env": std::env::var("VIRTUAL_ENV").ok(),
    })
}

/// Serialize [`runtime_report`] as pretty JSON and write it to `path`
/// (creating parent dirs). Returns the path written. A harness saves
/// this next to its numbers so a later reader knows exactly which
/// dylib, providers and CUDA runtimes produced them.
pub fn save_runtime_report(path: &std::path::Path) -> anyhow::Result<String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let text = serde_json::to_string_pretty(&runtime_report())?;
    std::fs::write(path, format!("{text}\n"))?;
    Ok(path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_report_has_stable_shape() {
        let rep = runtime_report();
        for key in [
            "ort_dylib",
            "ort_dylib_exists",
            "sibling_provider_dlls",
            "cuda_usable",
            "ort_installs",
            "cuda_runtime_dlls",
            "virtual_env",
        ] {
            assert!(rep.get(key).is_some(), "missing key {key}");
        }
        assert!(rep["cuda_usable"].is_boolean());
        assert!(rep["ort_installs"].is_array());
        assert!(rep["cuda_runtime_dlls"].is_array());
    }

    #[test]
    fn save_runtime_report_roundtrips() {
        let path = std::env::temp_dir().join("embroider-runtime-report-test.json");
        let back = save_runtime_report(&path).expect("save");
        assert_eq!(back, path.to_string_lossy());
        let text = std::fs::read_to_string(&path).expect("read back");
        let v: serde_json::Value = serde_json::from_str(&text).expect("parses");
        assert!(v.get("cuda_usable").is_some());
        std::fs::remove_file(&path).ok();
    }
}
