//! Embroider — Jina v5 text embeddings through ONNX Runtime.
//!
//! Rust core with opt-in PyO3 bindings (the bobine pattern). The crate
//! layering: [`providers`]/[`probe`]/[`policy`]/[`acquire`]/[`error`]/[`diag`]
//! are the shared ONNX plumbing (session building, provider fallback, CUDA
//! probing, HF acquisition); [`jina`] is the frozen text-embedding contract.
//!
//! Provenance: a clean move out of OKFgraph's `rust/okf-embed` (itself an
//! exact port of `EmbeddingEngine._encode`, pinned by that repo's parity
//! harness at ≤1e-5). Session/threading choices follow EmbedAnything's
//! `ort_jina.rs`; device fallback semantics mirror bobine: CUDA is
//! opportunistic, never fatal.
//!
//! ## Module layout
//!
//! - [`error`] — anyhow-based error plumbing (`oe` stringification)
//! - [`providers`] — provider-name matrix + clone-and-fallback application
//! - [`probe`] — corrected CUDA availability check (OnceLock-cached)
//! - [`policy`] — `DeviceReq` + explicit `SessionPolicy` (text vs defaults)
//! - [`acquire`] — validated HF hub acquisition (blocking, cached)
//! - [`diag`] — `OrtReport` observation for logs and diagnostics
//! - [`jina`] — `JinaV5` + `TokenizerHandle` (the frozen contract)
//! - [`vision`] — `JinaV5Vision` (the dynamic-grid image contract)

pub mod acquire;
pub mod diag;
pub mod error;
pub mod jina;
pub mod policy;
pub mod vision;
pub mod probe;
pub mod providers;

pub use acquire::{
    artifact_for, builtin_models, cache_info, cache_info_files, fetch_tokenizer_file,
    lookup_model, parse_owner_name, repo_for_precision, Artifact, CacheReport, ModelSpec,
    FP16_TEXT_MODEL, FP32_TEXT_MODEL, NANO_TEXT_MODEL, TEXT_NANO, TEXT_SMALL,
};
pub use diag::{report, OrtReport};
pub use jina::{JinaV5, TokenizerHandle, MAX_LENGTH, MODEL_MAX_TOKENS, NATIVE_DIM};
pub use vision::{
    vision_target_size, JinaV5Vision, VISION_MAX_PIXELS, VISION_MIN_PIXELS,
};
pub use acquire::{VISION_NANO, VISION_NANO_MODEL, VISION_NANO_REPO};
pub use policy::{DeviceReq, Precision, SessionPolicy};
pub use probe::cuda_available;
pub use providers::{
    apply_providers, apply_providers_with_arena, map_provider, ProviderMapping,
};

// ---------------------------------------------------------------------------
// PyO3 surface
// ---------------------------------------------------------------------------

#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use std::path::PathBuf;

#[cfg(feature = "extension-module")]
#[pyclass(name = "JinaV5")]
struct PyJinaV5 {
    inner: JinaV5,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyJinaV5 {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None, truncate_dim=512, device="auto", max_length=None, precision=None, cpu_arena=false))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
        truncate_dim: usize,
        device: &str,
        max_length: Option<usize>,
        precision: Option<&str>,
        cpu_arena: bool,
    ) -> PyResult<Self> {
        let precision = match precision {
            None => Precision::Auto,
            Some(s) => Precision::parse(s)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        };
        let inner = JinaV5::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
            truncate_dim,
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            max_length,
            precision,
            cpu_arena,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (onnx_path, tokenizer_path, truncate_dim=512, device="auto", max_length=None, cpu_arena=false))]
    fn open_files(
        onnx_path: &str,
        tokenizer_path: &str,
        truncate_dim: usize,
        device: &str,
        max_length: Option<usize>,
        cpu_arena: bool,
    ) -> PyResult<Self> {
        let inner = JinaV5::open_files(
            std::path::Path::new(onnx_path),
            std::path::Path::new(tokenizer_path),
            truncate_dim,
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            max_length,
            cpu_arena,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (text, task="Document"))]
    fn encode(&self, py: Python<'_>, text: &str, task: &str) -> PyResult<Vec<f32>> {
        py.detach(|| {
            self.inner
                .encode_one(text, task)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    #[pyo3(signature = (texts, task="Document"))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        task: &str,
    ) -> PyResult<Vec<Vec<f32>>> {
        py.detach(|| {
            self.inner
                .encode_many(&texts, task)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    #[getter]
    fn used_cuda(&self) -> bool {
        self.inner.used_cuda()
    }

    #[getter]
    fn max_length(&self) -> usize {
        self.inner.max_len()
    }

    #[getter]
    fn precision(&self) -> &str {
        self.inner.precision().as_str()
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner
            .count_tokens(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    }
}

#[cfg(feature = "extension-module")]
#[pyclass(name = "JinaTokenizer")]
struct PyJinaTokenizer {
    inner: TokenizerHandle,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyJinaTokenizer {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
    ) -> PyResult<Self> {
        let inner = TokenizerHandle::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (tokenizer_path))]
    fn open_files(tokenizer_path: &str) -> PyResult<Self> {
        let inner = TokenizerHandle::open_files(std::path::Path::new(tokenizer_path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner
            .count_tokens(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    }
}

#[cfg(feature = "extension-module")]
#[pyclass(name = "JinaV5Vision")]
struct PyJinaV5Vision {
    inner: JinaV5Vision,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyJinaV5Vision {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None, truncate_dim=512, device="auto", precision=None, gpu_mem_limit=None))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
        truncate_dim: usize,
        device: &str,
        precision: Option<&str>,
        gpu_mem_limit: Option<u64>,
    ) -> PyResult<Self> {
        let precision = match precision {
            None => Precision::Auto,
            Some(s) => Precision::parse(s)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
        };
        let inner = JinaV5Vision::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
            truncate_dim,
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            precision,
            gpu_mem_limit,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (onnx_path, tokenizer_path, truncate_dim=512, device="auto", gpu_mem_limit=None))]
    fn open_files(
        onnx_path: &str,
        tokenizer_path: &str,
        truncate_dim: usize,
        device: &str,
        gpu_mem_limit: Option<u64>,
    ) -> PyResult<Self> {
        let inner = JinaV5Vision::open_files(
            std::path::Path::new(onnx_path),
            std::path::Path::new(tokenizer_path),
            truncate_dim,
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            gpu_mem_limit,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    /// Embed one already-resized RGB image: `rgb` is raw row-major
    /// `[h, w, 3]` uint8 bytes (e.g. `bytes`) where `(h, w)` is its own
    /// `vision_target_size` — Pillow bicubic resize on the caller side.
    fn encode_image(&self, py: Python<'_>, rgb: Vec<u8>, h: usize, w: usize) -> PyResult<Vec<f32>> {
        py.detach(|| {
            self.inner
                .encode_image(&rgb, h, w)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn model_id(&self) -> &str {
        self.inner.model_id()
    }

    #[getter]
    fn used_cuda(&self) -> bool {
        self.inner.used_cuda()
    }

    #[getter]
    fn precision(&self) -> &str {
        self.inner.precision().as_str()
    }
}

/// Resize target for an `(h, w)` image under the vision resolution
/// contract — the Pillow side resizes here before calling `encode_image`.
#[cfg(feature = "extension-module")]
#[pyfunction(name = "vision_target_size")]
fn vision_target_size_py(h: u32, w: u32) -> PyResult<(u32, u32)> {
    vision_target_size(h, w)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e:#}")))
}

/// Registry listing for UIs and config validation: one dict per builtin
/// model (id, native_dim, max_tokens, ladder, precisions with artifacts,
/// text_partner for vision models).
#[cfg(feature = "extension-module")]
#[pyfunction]
fn available_models() -> Vec<std::collections::HashMap<String, String>> {
    builtin_models()
        .iter()
        .map(|m| {
            let mut d = std::collections::HashMap::new();
            d.insert("id".to_string(), m.id.to_string());
            d.insert("native_dim".to_string(), m.native_dim.to_string());
            d.insert("max_tokens".to_string(), m.max_tokens.to_string());
            d.insert(
                "ladder".to_string(),
                m.ladder.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","),
            );
            let mut prec = vec!["fp32"];
            if m.fp16.is_some() {
                prec.push("fp16");
            }
            if m.int8.is_some() {
                prec.push("int8");
            }
            d.insert("precisions".to_string(), prec.join(","));
            d.insert(
                "text_partner".to_string(),
                m.text_partner.unwrap_or("").to_string(),
            );
            d
        })
        .collect()
}

/// Shared dict rendering for [`acquire::CacheReport`]: identical keys for
/// `cache_info` and `cache_info_files` (`model_id, repo, precision,
/// cache_dir, files, cached, snapshot_path, disk_usage_bytes`);
/// `precision` is None for the generic lookup, which has no tier.
#[cfg(feature = "extension-module")]
fn report_to_dict<'py>(
    py: Python<'py>,
    rep: &acquire::CacheReport,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
    out.set_item("model_id", &rep.model_id)?;
    out.set_item("repo", &rep.repo)?;
    out.set_item("precision", rep.precision.map(|p| p.as_str()))?;
    out.set_item("cache_dir", rep.cache_dir.display().to_string())?;
    let files = pyo3::types::PyDict::new(py);
    for (name, path) in &rep.files {
        files.set_item(name, path.as_ref().map(|p| p.display().to_string()))?;
    }
    out.set_item("files", files)?;
    out.set_item("cached", rep.cached)?;
    out.set_item(
        "snapshot_path",
        rep.snapshot_path.as_ref().map(|p| p.display().to_string()),
    )?;
    out.set_item("disk_usage_bytes", rep.disk_usage_bytes)?;
    Ok(out)
}

/// Offline cache inspection (see [`acquire::cache_info`]): answers "is
/// this model already cached, and where?" without a session, a download,
/// or a device probe. `precision=None` reads fp32 (side-effect free by
/// contract); 'auto' and bare legacy ids raise ValueError before any I/O.
#[cfg(feature = "extension-module")]
#[pyfunction(name = "cache_info")]
#[pyo3(signature = (model_id, revision=None, cache_dir=None, precision=None))]
fn cache_info_py<'py>(
    py: Python<'py>,
    model_id: &str,
    revision: Option<String>,
    cache_dir: Option<String>,
    precision: Option<&str>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    // Argument validation fires before any I/O, mirroring open()'s checks
    // as ValueError (the rest is offline file reading and cannot fail).
    let precision = match precision {
        // Documented, side-effect-free default: no device probe here.
        None => Precision::Fp32,
        Some(s) => Precision::parse(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
    };
    if matches!(precision, Precision::Auto) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "cache_info never probes the device, so precision='auto' cannot be resolved; \
             pass an explicit precision ('fp32', 'fp16' or 'int8')",
        ));
    }
    // Registered ids skip owner/name parsing (the vision id is
    // deliberately short); legacy ids ARE the repo and must parse.
    if lookup_model(model_id).is_none() {
        parse_owner_name(model_id)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    }
    let rep = acquire::cache_info(
        model_id,
        revision.as_deref(),
        cache_dir.map(PathBuf::from),
        precision,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;

    report_to_dict(py, &rep)
}

/// Offline cache inspection for an arbitrary (repo, files) pair (see
/// [`acquire::cache_info_files`]): same mechanics and report contract as
/// `cache_info`, for non-registry layouts (bobine's converter models).
/// `files` takes bare names (required) or `(name, required)` pairs; a bad
/// repo, an empty list, or a malformed entry raises ValueError before I/O.
#[cfg(feature = "extension-module")]
#[pyfunction(name = "cache_info_files")]
#[pyo3(signature = (repo, files, revision=None, cache_dir=None))]
fn cache_info_files_py<'py>(
    py: Python<'py>,
    repo: &str,
    files: Bound<'py, pyo3::types::PyAny>,
    revision: Option<String>,
    cache_dir: Option<String>,
) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
    let mut parsed: Vec<(String, bool)> = Vec::new();
    for item in files.try_iter().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "files must be a list of filenames or (filename, required) pairs",
        )
    })? {
        let item = item?;
        if let Ok(name) = item.extract::<String>() {
            parsed.push((name, true));
        } else if let Ok((name, required)) = item.extract::<(String, bool)>() {
            parsed.push((name, required));
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "files entries must be a filename or a (filename, required) pair",
            ));
        }
    }
    if parsed.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "files must not be empty",
        ));
    }
    // The repo is the id here: validate before any I/O, like cache_info.
    parse_owner_name(repo)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let refs: Vec<(&str, bool)> = parsed.iter().map(|(n, r)| (n.as_str(), *r)).collect();
    let rep = acquire::cache_info_files(
        repo,
        &refs,
        revision.as_deref(),
        cache_dir.map(PathBuf::from),
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;

    report_to_dict(py, &rep)
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn embroider(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJinaV5>()?;
    m.add_class::<PyJinaTokenizer>()?;
    m.add_class::<PyJinaV5Vision>()?;
    m.add_function(pyo3::wrap_pyfunction!(available_models, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cache_info_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cache_info_files_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vision_target_size_py, m)?)?;
    m.add("VISION_NANO_MODEL", VISION_NANO_MODEL)?;
    m.add("NATIVE_DIM", NATIVE_DIM)?;
    m.add("MAX_LENGTH", MAX_LENGTH)?;
    m.add("MODEL_MAX_TOKENS", MODEL_MAX_TOKENS)?;
    m.add("FP32_TEXT_MODEL", FP32_TEXT_MODEL)?;
    m.add("NANO_TEXT_MODEL", NANO_TEXT_MODEL)?;
    Ok(())
}
