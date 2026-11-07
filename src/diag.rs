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

/// Set the paths the ONNX Runtime loader needs, programmatically instead
/// of ambiently. `dylib` (when given) must exist — it becomes the new
/// `ORT_DYLIB_PATH`. `dll_dirs` (existing dirs only, silently skipping the
/// rest) are prepended to `PATH`, which is what makes the CUDA/cuDNN DLLs
/// from `load_dirs` resolve out of site-packages via plain `LoadLibrary`.
///
/// Timing contract: `ORT_DYLIB_PATH` is read when ort first initializes in
/// this process, so set `dylib` before the first `open()` — afterwards it
/// is stored but inert for the running process (still reported by
/// [`runtime_report`]). `PATH` edits apply to every later `LoadLibrary`,
/// so `dll_dirs` work at any time. Returns the post-call
/// [`runtime_report`] so the caller sees what took effect.
pub fn configure_runtime(
    dylib: Option<&str>,
    dll_dirs: &[String],
) -> anyhow::Result<serde_json::Value> {
    if let Some(d) = dylib {
        let p = std::path::Path::new(d);
        if !p.is_file() {
            anyhow::bail!("ORT dylib not found: {d}");
        }
        // SAFETY: process-start-style configuration before ort init; the
        // caller upholds the timing contract above.
        unsafe {
            std::env::set_var("ORT_DYLIB_PATH", d);
        }
    }
    if !dll_dirs.is_empty() {
        let mut seen: std::collections::HashSet<String> = Default::default();
        let mut prepend: Vec<String> = Vec::new();
        for dir in dll_dirs {
            if !std::path::Path::new(dir).is_dir() {
                continue;
            }
            if seen.insert(dir.clone()) {
                prepend.push(dir.clone());
            }
        }
        if !prepend.is_empty() {
            let sep = if cfg!(windows) { ";" } else { ":" };
            let old = std::env::var("PATH").unwrap_or_default();
            let mut parts: Vec<&str> = prepend.iter().map(String::as_str).collect();
            if !old.is_empty() {
                parts.push(&old);
            }
            // SAFETY: same configuration window as above.
            unsafe {
                std::env::set_var("PATH", parts.join(sep));
            }
        }
    }
    Ok(runtime_report())
}

/// The interpreter prefix the binding recorded at import time
/// ([`note_venv_prefix`], called from the pymodule init): `sys.prefix`
/// — the running interpreter's own root, which IS the venv root when a
/// venv is in play, and never requires activation the way
/// `VIRTUAL_ENV` does.
static VENV_PREFIX: std::sync::OnceLock<Option<String>>
    = std::sync::OnceLock::new();

/// Register the prefix (the pymodule init; the first caller wins so a
/// notebook re-import keeps the first interpreter's root).
#[doc(hidden)]
pub fn note_venv_prefix(prefix: Option<String>) {
    let _ = VENV_PREFIX.set(prefix);
}

/// The recorded prefix, if the pymodule ran and captured one.
#[doc(hidden)]
pub fn venv_prefix() -> Option<String> {
    VENV_PREFIX.get().and_then(|o| o.clone())
}

/// Where each reported DLL lives — the loader dirs of the CUDA lane:
/// the dylib's own dir plus the parent of every `cudnn`/`cublas`/… DLL
/// found in the venv's `nvidia*` pip dists and in sibling
/// `onnxruntime*` `capi` dirs. This is exactly [`runtime_report`]'s
/// `load_dirs` computation, factored out for the auto-prepend.
fn cuda_load_dirs(venv: Option<&str>) -> Vec<String> {
    let mut out: std::collections::BTreeSet<String> = Default::default();
    if let Some(p) = std::env::var("ORT_DYLIB_PATH").ok() {
        if let Some(parent) = std::path::Path::new(&p).parent() {
            out.insert(parent.to_string_lossy().into_owned());
        }
    }
    for sp in site_packages_dirs_for(venv.map(std::path::PathBuf::from)) {
        let Ok(rd) = std::fs::read_dir(&sp) else { continue };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let name = p.file_name().and_then(|s| s.to_str())
                .unwrap_or("").to_lowercase();
            if name.starts_with("nvidia") && !name.ends_with(".dist-info") {
                for (_package, dlls) in scan_nvidia_dist(&p) {
                    for dll in dlls {
                        if let Some(d) = std::path::Path::new(&dll).parent() {
                            out.insert(d.to_string_lossy().into_owned());
                        }
                    }
                }
            } else if name.starts_with("onnxruntime")
                    && !name.ends_with(".dist-info") {
                // provider DLLs live beside the runtime in `capi` —
                // a dist found here may be the loaded one already, but
                // a second install (CPU/GPU side by side) must not be
                // invisible to the search path.
                let capi = p.join("capi");
                if capi.is_dir() {
                    out.insert(capi.to_string_lossy().into_owned());
                }
            }
        }
    }
    out.into_iter().collect()
}

/// Non-pip layouts the CUDA EP's DLLs can live in: the cuDNN
/// developer-kit install (`<ProgramFiles>\\NVIDIA\\CUDNN\\v<ver>\\bin\\
/// <cuda>\\x64` — cudnn 9's Windows layout keeps `cudnn64_9.dll` there
/// and nothing else puts it on PATH), `%CUDA_PATH%\\bin[\\x64]`, and a
/// `CUDNN_PATH` env hint. *program_files* lets tests point at a fake
/// root instead of the real one.
fn sdk_cuda_dll_dirs(program_files: Option<String>) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(cp) = std::env::var("CUDA_PATH") {
        out.push(std::path::PathBuf::from(&cp).join("bin"));
        out.push(std::path::PathBuf::from(&cp).join("bin").join("x64"));
    }
    if let Ok(c) = std::env::var("CUDNN_PATH") {
        out.push(std::path::PathBuf::from(&c).join("bin"));
        out.push(std::path::PathBuf::from(c));
    }
    let pf = match program_files {
        Some(p) => std::path::PathBuf::from(p),
        None => match std::env::var("ProgramFiles") {
            Ok(pf) => std::path::PathBuf::from(pf),
            Err(_) => return out,
        },
    };
    let cudnn_root = pf.join("NVIDIA").join("CUDNN");
    if let Ok(rd) = std::fs::read_dir(&cudnn_root) {
        for v in rd.filter_map(|e| e.ok()) {
            let bin = v.path().join("bin");
            let Ok(subs) = std::fs::read_dir(&bin) else { continue };
            for sub in subs.filter_map(|e| e.ok()) {
                out.push(sub.path().join("x64"));
                out.push(sub.path());
            }
        }
    }
    out
}

fn path_has_entry(path: &str, dir: &str) -> bool {
    let norm = |s: &str| s.trim_end_matches(['/','\\']).to_lowercase();
    path.split(if cfg!(windows) { ";" } else { ":" })
        .any(|e| norm(e) == norm(dir))
}

/// The best ONNX Runtime dylib the venv ships: the `capi` dir of every
/// `onnxruntime*` dist (GPU builds first), ranked by the version the
/// sibling dist-info dir carries. ort's load-dynamic search walks PATH
/// and resolves whatever stray `onnxruntime.dll` a sibling tool dropped
/// there (observed: a 1.17.1 in a model-runner's bin) — passing the
/// venv's own dylib through `ORT_DYLIB_PATH` makes the loaded version a
/// choice instead of a lottery.
fn best_ort_dylib(venv: Option<&str>) -> Option<std::path::PathBuf> {
    let mut found: Vec<(String, std::path::PathBuf)> = Vec::new();
    // Windows filesystems are case-insensitive, so the same venv can
    // surface as both `Lib/…` and `lib/…` — dedupe by the canonical dir
    // (the same spelling caveat runtime_report handles for its report).
    let mut site_packages: Vec<std::path::PathBuf> = Vec::new();
    let mut seen_sp: std::collections::HashSet<String> = Default::default();
    for sp in site_packages_dirs_for(venv.map(std::path::PathBuf::from)) {
        let key = std::fs::canonicalize(&sp)
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|_| sp.to_string_lossy().into_owned());
        if seen_sp.insert(key) {
            site_packages.push(sp);
        }
    }
    for sp in site_packages {
        let Ok(rd) = std::fs::read_dir(&sp) else { continue };
        for e in rd.filter_map(|e| e.ok()) {
            let dist = e.path();
            if !dist.is_dir() {
                continue;
            }
            let name = dist.file_name().and_then(|s| s.to_str())
                .unwrap_or("").to_lowercase();
            if !name.starts_with("onnxruntime") || name.ends_with(".dist-info") {
                continue;
            }
            let dll = dist.join("capi").join("onnxruntime.dll");
            if !dll.is_file() {
                continue;
            }
            // The sibling dist-info carries the version the static
            // checker verifies anyway (BadVersion) — parse it for rank
            // (`onnxruntime_gpu-1.31.1.dist-info` → "1.31.1").
            let ver = std::fs::read_dir(&sp).ok()
                .and_then(|de| de.filter_map(|e| e.ok())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .find(|n| {
                        let n = n.to_lowercase();
                        n.starts_with(&name) && n.ends_with(".dist-info")
                    }))
                .and_then(|n| {
                    n.to_lowercase()
                        .rsplit_once('-')
                        .map(|(_, tail)| {
                            tail.trim_end_matches(".dist-info").to_string()
                        })
                })
                .unwrap_or_else(|| "0".to_string());
            // GPU builds outrank same-version CPU builds.
            let rank = format!("{ver}{}", if name.contains("gpu") {"1"} else {"0"});
            found.push((rank, dll));
        }
    }
    found.sort();
    found.last().map(|(_r, p)| p.clone())
}

/// Put the CUDA lane's DLL dirs on the search path so a fresh process
/// with no ambient PATH knowledge — the ``okf`` CLI subprocess, a
/// notebook, a service — can build a CUDA session against the venv's
/// pip `nvidia*` dists or a cuDNN developer-kit install. The dirs are
/// found with [`cuda_load_dirs`] + [`sdk_cuda_dll_dirs`]; dirs already
/// on PATH are left alone and the rest are *prepended* via
/// [`configure_runtime`], which re-sets PATH in one edit.
///
/// Runs before the first CUDA session build (the hook sits in the
/// provider resolution, before any EP is registered), is idempotent,
/// and never fails: missing candidates are skipped, and a loader that
/// had nothing to add answers `prepended: []`.
pub fn prepare_cuda_dirs(venv: Option<&str>) -> serde_json::Value {
    let mut candidates = cuda_load_dirs(venv);
    for d in sdk_cuda_dll_dirs(None) {
        candidates.push(d.to_string_lossy().into_owned());
    }
    let path = std::env::var("PATH").unwrap_or_default();
    let candidates = candidates.into_iter().filter(
        |d| std::path::Path::new(d).is_dir()
            && !path_has_entry(&path, d)).collect::<Vec<_>>();
    // Deliver the venv's own ONNX Runtime too — ort's load-dynamic
    // search otherwise resolves whatever onnxruntime.dll a sibling tool
    // dropped on PATH. An explicitly pinned ORT_DYLIB_PATH is respected:
    // the env is only read at first init, so consumers who set it (the
    // documented contract) keep their choice.
    let dylib = std::env::var("ORT_DYLIB_PATH").err()
        .and_then(|_| best_ort_dylib(venv))
        .map(|p| p.to_string_lossy().into_owned());
    if candidates.is_empty() && dylib.is_none() {
        return serde_json::json!({ "prepended": [], "candidates": 0,
                                   "dylib": serde_json::Value::Null });
    }
    if configure_runtime(dylib.as_deref(), &candidates).is_err() {
        // configure_runtime only bails for a missing dylib — this
        // arm is defensive and must never make the session path worse.
        return serde_json::json!({ "prepended": [], "candidates": 0,
                                   "dylib": serde_json::Value::Null });
    }
    serde_json::json!({
        "prepended": candidates, "candidates": candidates.len(),
        "dylib": dylib,
    })
}

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

/// Candidate site-packages dirs for `venv` (an explicit interpreter
/// prefix — the binding fills in `sys.prefix` when `VIRTUAL_ENV` is
/// unset, so a non-activated interpreter still scans its own venv).
fn site_packages_dirs_for(venv: Option<std::path::PathBuf>) -> Vec<std::path::PathBuf> {
    let venv = match venv {
        Some(v) => v,
        None => return Vec::new(),
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
    runtime_report_for(None)
}

/// [`runtime_report`] scoped to an explicit venv prefix (the binding
/// passes `VIRTUAL_ENV`, falling back to the interpreter's own
/// `sys.prefix`, so activation is not required to see the installs).
pub fn runtime_report_for(venv: Option<&str>) -> serde_json::Value {
    let venv = venv.map(std::path::PathBuf::from);
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
    for sp in site_packages_dirs_for(venv.clone()) {
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
    // Every directory a loader must see: the dylib's own dir plus the
    // parent of each reported DLL. A harness prepends exactly this list
    // to the DLL search path (`os.add_dll_directory` / PATH) before the
    // first session opens — that is what makes `cudnn64_9.dll` resolve
    // out of site-packages without copying files around.
    let mut load_dirs: std::collections::BTreeSet<String> = Default::default();
    if let Some(p) = &dylib {
        if let Some(parent) = std::path::Path::new(p).parent() {
            load_dirs.insert(parent.to_string_lossy().into_owned());
        }
    }
    for group in &cuda_runtimes {
        for dll in group["dlls"].as_array().cloned().unwrap_or_default() {
            if let Some(d) = dll.as_str().and_then(|d| std::path::Path::new(d).parent()) {
                load_dirs.insert(d.to_string_lossy().into_owned());
            }
        }
    }
    // The CUDA EP also resolves cuBLAS/cuDNN through PATH entries, so the
    // nvidia bin dirs matter even when the dylib lives elsewhere.
    serde_json::json!({
        "ort_dylib": dylib,
        "ort_dylib_exists": dylib_exists,
        "sibling_provider_dlls": siblings,
        "cuda_usable": cuda_available(),
        "ort_installs": ort_installs,
        "cuda_runtime_dlls": cuda_runtimes,
        "load_dirs": load_dirs.into_iter().collect::<Vec<_>>(),
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

    /// The env-touching tests here read-modify-write the process PATH;
    /// cargo runs test threads in parallel, so they serialize on this
    /// (each saves its `before` and restores, which would otherwise
    /// clobber a sibling mid-flight).
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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
            "load_dirs",
            "virtual_env",
        ] {
            assert!(rep.get(key).is_some(), "missing key {key}");
        }
        assert!(rep["cuda_usable"].is_boolean());
        assert!(rep["ort_installs"].is_array());
        assert!(rep["cuda_runtime_dlls"].is_array());
    }

    #[test]
    fn configure_runtime_rejects_missing_dylib() {
        let err = configure_runtime(Some("D:/no/such/onnxruntime.dll"), &[])
            .err()
            .expect("must fail")
            .to_string();
        assert!(err.contains("not found"), "{err}");
    }

    #[test]
    fn prepare_cuda_dirs_prepends_nvidia_pip_bin_dirs() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let root = std::env::temp_dir().join("embroider-nvidia-pip-test");
        let bin = root.join("Lib").join("site-packages").join("nvidia")
            .join("cudnn").join("bin");
        std::fs::create_dir_all(&bin).ok();
        std::fs::write(bin.join("cudnn64_9.dll"), b"x").ok();
        let before = std::env::var("PATH").unwrap_or_default();
        let rep = prepare_cuda_dirs(Some(root.to_str().unwrap()));
        let after = std::env::var("PATH").unwrap_or_default();
        assert!(after.contains("cudnn"), "cudnn dir should land on PATH {after}");
        assert!(rep["prepended"].as_array().unwrap().len() >= 1);
        // idempotent: a second call prepends nothing (already on PATH)
        let rep2 = prepare_cuda_dirs(Some(root.to_str().unwrap()));
        assert_eq!(rep2["prepended"].as_array().map(Vec::len), Some(0));
        // Restore so test order never leaks PATH edits.
        unsafe {
            std::env::set_var("PATH", before);
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn prepare_cuda_dirs_delivers_the_venv_ort_dylib() {
        let root = std::env::temp_dir().join("embroider-ort-dist-test");
        let capi = root.join("Lib").join("site-packages").join("onnxruntime_gpu")
            .join("capi");
        std::fs::create_dir_all(&capi).ok();
        let dll = capi.join("onnxruntime.dll");
        std::fs::write(&dll, b"x").ok();
        let info = root.join("Lib").join("site-packages")
            .join("onnxruntime_gpu-1.31.1.dist-info");
        std::fs::create_dir_all(&info).ok();
        std::fs::write(info.join("METADATA"), b"x").ok();
        let before = std::env::var("ORT_DYLIB_PATH").ok();
        if before.is_some() {
            // the contract: an explicitly pinned path is respected, so
            // this test needs the env unset to exercise the delivery
            unsafe { std::env::remove_var("ORT_DYLIB_PATH"); }
        }
        let before_path = std::env::var("PATH").unwrap_or_default();
        let rep = prepare_cuda_dirs(Some(root.to_str().unwrap()));
        let got = std::env::var("ORT_DYLIB_PATH").ok();
        // Compare case-insensitively: Windows is case-insensitive and
        // the scan may hand back either spelling of the same dir.
        let got_norm = got.as_ref().map(|p| p.replace("\\\\", "/").to_lowercase());
        let want_norm = dll.to_str().unwrap().replace("\\\\", "/").to_lowercase();
        assert_eq!(got_norm.as_deref(), Some(want_norm.as_str()),
                   "delivered dylib should be pinned: {got:?}");
        assert_eq!(rep["dylib"].as_str().map(|p| p.replace("\\\\", "/").to_lowercase()),
                   Some(want_norm.clone()));
        unsafe { std::env::set_var("PATH", before_path); }
        // restore the caller-precedence env (or its absence)
        match before {
            Some(v) => unsafe { std::env::set_var("ORT_DYLIB_PATH", v); },
            None => unsafe { std::env::remove_var("ORT_DYLIB_PATH"); },
        }
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn prepare_cuda_dirs_adds_the_cudnn_sdk_root() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let pf = std::env::temp_dir().join("embroider-sdk-fake");
        let laid = pf.join("NVIDIA").join("CUDNN").join("v9.5")
            .join("bin").join("13.1").join("x64");
        std::fs::create_dir_all(&laid).ok();
        std::fs::write(laid.join("cudnn64_9.dll"), b"x").ok();
        let before = std::env::var("PATH").unwrap_or_default();
        let dirs = sdk_cuda_dll_dirs(Some(pf.to_str().unwrap().to_string()));
        assert!(dirs.iter().any(|d| d == &laid), "sdk layout found: {dirs:?}");
        // and through the public entry the dir reaches PATH
        let rep = configure_runtime(None, &dirs.into_iter()
            .map(|d| d.to_string_lossy().into_owned())
            .collect::<Vec<_>>()).expect("ok");
        let after = std::env::var("PATH").unwrap_or_default();
        assert!(after.to_lowercase().contains("embroider-sdk-fake"));
        let _ = rep;
        unsafe {
            std::env::set_var("PATH", before);
        }
        std::fs::remove_dir_all(&pf).ok();
    }

    #[test]
    fn configure_runtime_prepends_dll_dirs() {
        let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let dir = std::env::temp_dir().join("embroider-dll-dir-test");
        std::fs::create_dir_all(&dir).ok();
        let before = std::env::var("PATH").unwrap_or_default();
        configure_runtime(None, &[dir.to_string_lossy().into_owned()]).expect("ok");
        let after = std::env::var("PATH").unwrap_or_default();
        assert!(after.starts_with(dir.to_str().unwrap()), "{after}");
        // Restore so test order never leaks PATH edits.
        unsafe {
            std::env::set_var("PATH", before);
        }
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
