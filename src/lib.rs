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

pub mod acquire;
pub mod diag;
pub mod error;
pub mod jina;
pub mod policy;
pub mod probe;
pub mod providers;

pub use acquire::{
    fetch_tokenizer_file, parse_owner_name, repo_for_precision,
    FP16_TEXT_MODEL, FP32_TEXT_MODEL,
};
pub use diag::{report, OrtReport};
pub use jina::{JinaV5, TokenizerHandle, MAX_LENGTH, MODEL_MAX_TOKENS, NATIVE_DIM};
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
#[pymodule]
fn embroider(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJinaV5>()?;
    m.add_class::<PyJinaTokenizer>()?;
    m.add("NATIVE_DIM", NATIVE_DIM)?;
    m.add("MAX_LENGTH", MAX_LENGTH)?;
    m.add("MODEL_MAX_TOKENS", MODEL_MAX_TOKENS)?;
    Ok(())
}
