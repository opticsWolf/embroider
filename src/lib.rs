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
pub mod decide;
pub mod laya;
pub mod diag;
pub mod error;
pub mod jina;
pub mod policy;
pub mod rerank;
pub mod vision;
pub mod probe;
pub mod providers;

pub use acquire::{
    artifact_for, artifact_for_task, builtin_models, builtin_tasks, cache_info, cache_info_files,
    fetch_repo_file, fetch_tokenizer_file, lookup_model, lookup_task, parse_owner_name,
    repo_for_precision,
    Artifact, CacheReport, ModelSpec, TaskKind, TaskSpec, FP16_TEXT_MODEL, FP32_TEXT_MODEL,
    GLICLASS_SMALL, GLICLASS_SMALL_MODEL, GLICLASS_V4_EDGE, GLICLASS_V4_EDGE_MODEL,
    LAYA_BASE, LAYA_BASE_MODEL, LAYA_MULTI, LAYA_MULTI_MODEL, LAYA_TD, LAYA_TD_MODEL,
    NANO_TEXT_MODEL, RERANK_MMARCO, RERANK_MMARCO_MODEL,
    RERANK_MINILM, RERANK_MINILM_MODEL, TEXT_NANO, TEXT_SMALL,
};
pub use diag::{configure_runtime, report, runtime_report, runtime_report_for, save_runtime_report, OrtReport};
pub use jina::{JinaV5, TokenizerHandle, MAX_LENGTH, MODEL_MAX_TOKENS, NATIVE_DIM};
pub use rerank::{
    Reranker, RERANK_ABSOLUTE_MAX, RERANK_DEFAULT_MAX,
};
pub use decide::{
    Decider, Decision, Question, DECIDE_ABSOLUTE_MAX, DECIDE_DEFAULT_MAX,
};
pub use laya::{LayaDecider, LayaDecision, LayaLongDecision, LayaOption, LayaQuestion, LayaTemperatures, BinningMap};
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
use pyo3::types::{PyList, PyTuple};
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
#[pyclass(name = "Reranker")]
struct PyReranker {
    inner: Reranker,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyReranker {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None, device="auto", max_length=None, precision=None, cpu_arena=false))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
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
        let inner = Reranker::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
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
    #[pyo3(signature = (onnx_path, tokenizer_path, device="auto", max_length=None, cpu_arena=false))]
    fn open_files(
        onnx_path: &str,
        tokenizer_path: &str,
        device: &str,
        max_length: Option<usize>,
        cpu_arena: bool,
    ) -> PyResult<Self> {
        let inner = Reranker::open_files(
            std::path::Path::new(onnx_path),
            std::path::Path::new(tokenizer_path),
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            max_length,
            cpu_arena,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    fn score(&self, py: Python<'_>, query: &str, doc: &str) -> PyResult<f32> {
        py.detach(|| {
            self.inner
                .score_one(query, doc)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    #[pyo3(signature = (query, doc, window=None, stride=None))]
    fn score_long(
        &self,
        py: Python<'_>,
        query: &str,
        doc: &str,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> PyResult<f32> {
        py.detach(|| {
            self.inner
                .score_long(query, doc, window, stride)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner
            .count_tokens(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    }

    #[pyo3(signature = (pairs))]
    fn score_batch(&self, py: Python<'_>, pairs: Vec<(String, String)>) -> PyResult<Vec<f32>> {
        let refs: Vec<(&str, &str)> =
            pairs.iter().map(|(q, d)| (q.as_str(), d.as_str())).collect();
        py.detach(|| {
            self.inner
                .score_many(&refs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
        })
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
}

#[cfg(feature = "extension-module")]
#[pyclass(name = "Decider")]
struct PyDecider {
    inner: Decider,
}

/// Parse one (kind, instructions, options) triple into a [`Question`].
/// `noul` parses fine and fails at decide time — the loud-unsupported
/// rule, not a silent schema gap.
#[cfg(feature = "extension-module")]
fn parse_question(kind: &str, instructions: String, options: Vec<String>) -> PyResult<Question> {
    match kind {
        "choice" => Ok(Question::Choice { instructions, options }),
        "score" => Ok(Question::Score { instructions, options }),
        "noul" => Ok(Question::Noul { instructions, options }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "question kind must be 'choice', 'score' or 'noul', got '{other}'"
        ))),
    }
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyDecider {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None, device="auto", max_length=None, precision=None, cpu_arena=false))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
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
        let inner = Decider::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
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
    #[pyo3(signature = (onnx_path, tokenizer_path, device="auto", max_length=None, prompt_first=false, cpu_arena=false))]
    fn open_files(
        onnx_path: &str,
        tokenizer_path: &str,
        device: &str,
        max_length: Option<usize>,
        prompt_first: bool,
        cpu_arena: bool,
    ) -> PyResult<Self> {
        let inner = Decider::open_files(
            std::path::Path::new(onnx_path),
            std::path::Path::new(tokenizer_path),
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            max_length,
            prompt_first,
            cpu_arena,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    /// Answer questions over one state: `questions` is a list of
    /// `(kind, instructions, options)` triples (`kind` one of 'choice',
    /// 'score', 'noul'). Returns `[(label, [(option, prob), ...]), ...]`
    /// in question order — one padded session run for the whole batch.
    #[pyo3(signature = (state, questions))]
    fn decide(
        &self,
        py: Python<'_>,
        state: &str,
        questions: Vec<(String, String, Vec<String>)>,
    ) -> PyResult<Vec<(String, Vec<(String, f32)>)>> {
        let mut qs = Vec::with_capacity(questions.len());
        for (kind, instructions, options) in questions {
            qs.push(parse_question(&kind, instructions, options)?);
        }
        py.detach(|| {
            let ds = self
                .inner
                .decide(state, &qs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
            Ok(ds.into_iter().map(|d| (d.label, d.probabilities)).collect())
        })
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner
            .count_tokens(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
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
}

/// Parse one Laya question from its Python tuple shape:
/// - `("choice", instructions, options[, option_order])` where each
///   option is a `label` string or a `(label, description|None)` pair.
/// - `("score", instructions, [levels...])`.
/// - `("noul", instructions, statement)`.
#[cfg(feature = "extension-module")]
fn parse_laya_question(obj: &Bound<'_, PyAny>) -> PyResult<LayaQuestion> {
    let tup = obj
        .cast::<PyTuple>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("a Laya question is a tuple (see LayaDecider.decide)"))?;
    let kind: String = tup
        .get_item(0)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("a Laya question needs (kind, instructions, ...)"))?
        .extract()?;
    let instructions: String = tup
        .get_item(1)
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("a Laya question needs (kind, instructions, ...)"))?
        .extract()?;
    match kind.as_str() {
        "choice" => {
            let opts_obj = tup
                .get_item(2)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("choice needs an options list"))?;
            let opts = opts_obj
                .cast::<PyList>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("choice needs an options list"))?;
            let mut options = Vec::with_capacity(opts.len());
            for o in opts.iter() {
                if let Ok(label) = o.extract::<String>() {
                    options.push(LayaOption { label, description: None });
                } else if let Ok(pair) = o.cast::<PyTuple>() {
                    let label: String = pair
                        .get_item(0)
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("an option pair is (label, description|None)"))?
                        .extract()?;
                    let desc: Option<String> = pair
                        .get_item(1)
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("an option pair is (label, description|None)"))?
                        .extract()?;
                    options.push(LayaOption { label, description: desc });
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "a choice option is a label string or a (label, description|None) pair",
                    ));
                }
            }
            let option_order: Option<Vec<usize>> =
                tup.get_item(3).ok().and_then(|v| v.extract().ok());
            Ok(LayaQuestion::Choice { instructions, options, option_order })
        }
        "score" => {
            let levels: Vec<String> = tup
                .get_item(2)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("score needs a levels list"))?
                .extract()?;
            Ok(LayaQuestion::Score { instructions, levels })
        }
        "noul" => {
            let statement: String = tup
                .get_item(2)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("noul needs a statement string"))?
                .extract()?;
            Ok(LayaQuestion::Noul { instructions, statement })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Laya question kind must be 'choice', 'score' or 'noul', got '{other}'"
        ))),
    }
}

#[cfg(feature = "extension-module")]
#[pyclass(name = "LayaDecider")]
struct PyLayaDecider {
    inner: LayaDecider,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl PyLayaDecider {
    #[staticmethod]
    #[pyo3(signature = (model_id, revision=None, cache_dir=None, device="auto", max_length=None, precision=None, cpu_arena=false))]
    fn open(
        model_id: &str,
        revision: Option<String>,
        cache_dir: Option<String>,
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
        let inner = LayaDecider::open(
            model_id,
            revision.as_deref(),
            cache_dir.map(PathBuf::from),
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
    #[pyo3(signature = (onnx_path, tokenizer_path, config_path, device="auto", max_length=None, cpu_arena=false))]
    fn open_files(
        onnx_path: &str,
        tokenizer_path: &str,
        config_path: &str,
        device: &str,
        max_length: Option<usize>,
        cpu_arena: bool,
    ) -> PyResult<Self> {
        let inner = LayaDecider::open_files(
            std::path::Path::new(onnx_path),
            std::path::Path::new(tokenizer_path),
            std::path::Path::new(config_path),
            DeviceReq::parse(device)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            max_length,
            cpu_arena,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
        Ok(Self { inner })
    }

    /// Answer questions over one state: `questions` is a list of Laya
    /// question tuples — `("choice", ins, options[, order])`,
    /// `("score", ins, levels)`, `("noul", ins, statement)` — where a
    /// choice option is a label string or a `(label, description|None)`
    /// pair. Returns `[(label, [(option, prob), ...], value|None,
    /// confidence, answer_confidence, act_probability), ...]` in question
    /// order — one padded session run for the whole batch.
    #[pyo3(signature = (state, questions))]
    fn decide(
        &self,
        py: Python<'_>,
        state: &str,
        questions: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<(String, Vec<(String, f32)>, Option<f32>, f32, f32, f32)>> {
        let mut qs = Vec::with_capacity(questions.len());
        for q in &questions {
            qs.push(parse_laya_question(q)?);
        }
        py.detach(|| {
            let ds = self
                .inner
                .decide(state, &qs)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
            Ok(ds
                .into_iter()
                .map(|d| {
                    (
                        d.label,
                        d.probabilities,
                        d.value,
                        d.confidence,
                        d.answer_confidence,
                        d.act_probability,
                    )
                })
                .collect())
        })
    }

    fn count_tokens(&self, text: &str) -> PyResult<usize> {
        self.inner
            .count_tokens(text)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    }

    /// Install a fitted binning overlay from a JSON file (bare map or
    /// laya calibration payload). Remaps `answer_confidence` on every
    /// later `decide`; buckets the map does not cover pass through.
    fn load_binning_map(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .load_binning_map(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
    }

    /// Answer questions over a long state: `window`-token windows
    /// stepping by `stride` (both `None` = the checkpoint-sized default),
    /// one shared session run for all windows, per-type aggregation
    /// (noul: strongest P(true); choice/score: most-confident window).
    /// Returns `[(label, probs, value|None, confidence,
    /// answer_confidence, act, window_start, window_end, windows), ...]`
    /// in question order.
    #[pyo3(signature = (state, questions, window=None, stride=None))]
    fn predict_long(
        &self,
        py: Python<'_>,
        state: &str,
        questions: Vec<Bound<'_, PyAny>>,
        window: Option<usize>,
        stride: Option<usize>,
    ) -> PyResult<Vec<(String, Vec<(String, f32)>, Option<f32>, f32, f32, f32, usize, usize, usize)>> {
        let mut qs = Vec::with_capacity(questions.len());
        for q in &questions {
            qs.push(parse_laya_question(q)?);
        }
        py.detach(|| {
            let ds = self
                .inner
                .predict_long(state, &qs, window, stride)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
            Ok(ds
                .into_iter()
                .map(|l| {
                    (
                        l.decision.label,
                        l.decision.probabilities,
                        l.decision.value,
                        l.decision.confidence,
                        l.decision.answer_confidence,
                        l.decision.act_probability,
                        l.window_start,
                        l.window_end,
                        l.windows,
                    )
                })
                .collect())
        })
    }

    /// Checkpoint temperatures after the [0.5, 5.0] clamp: the triple,
    /// the per-bucket map, and the load-time rejection list.
    fn temperature_report(&self) -> (Vec<f64>, Vec<(String, f64)>, Vec<String>) {        (
            self.inner.temperatures().to_vec(),
            self.inner.temperature_by_options(),
            self.inner.temperature_rejections().to_vec(),
        )
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
/// model (id, kind, native_dim, max_tokens, ladder, precisions with
/// artifacts, text_partner for vision models). Task entries (rerank,
/// decide) carry their kind with the vector-space keys empty — same
/// schema, no per-kind parsing on the caller side.
#[cfg(feature = "extension-module")]
#[pyfunction]
fn available_models() -> Vec<std::collections::HashMap<String, String>> {
    let mut out: Vec<std::collections::HashMap<String, String>> = builtin_models()
        .iter()
        .map(|m| {
            let mut d = std::collections::HashMap::new();
            d.insert("id".to_string(), m.id.to_string());
            d.insert("kind".to_string(), "embed".to_string());
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
        .collect();
    out.extend(builtin_tasks().iter().map(|t| {
        let mut d = std::collections::HashMap::new();
        d.insert("id".to_string(), t.id.to_string());
        d.insert("kind".to_string(), t.kind.as_str().to_string());
        d.insert("native_dim".to_string(), String::new());
        d.insert("max_tokens".to_string(), t.max_tokens.to_string());
        d.insert("ladder".to_string(), String::new());
        let mut prec = vec!["fp32"];
        if t.fp16.is_some() {
            prec.push("fp16");
        }
        if t.int8.is_some() {
            prec.push("int8");
        }
        d.insert("precisions".to_string(), prec.join(","));
        d.insert("text_partner".to_string(), String::new());
        d
    }));
    out
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

/// Convert a [`serde_json::Value`] tree into Python objects (None /
/// bool / int / float / str / list / dict) for [`runtime_info`].
#[cfg(feature = "extension-module")]
fn value_to_py<'py>(
    py: Python<'py>,
    v: &serde_json::Value,
) -> PyResult<pyo3::Bound<'py, pyo3::PyAny>> {
    use pyo3::IntoPyObject;
    use pyo3::types::{PyDict, PyList};
    match v {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(b) => Ok(pyo3::types::PyBool::new(py, *b).to_owned().into_any()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any())
            } else if let Some(u) = n.as_u64() {
                Ok(u.into_pyobject(py)?.into_any())
            } else {
                Ok(n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any())
            }
        }
        serde_json::Value::String(st) => Ok(st.into_pyobject(py)?.into_any()),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, item) in map {
                dict.set_item(k, value_to_py(py, item)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

/// Inventory the ONNX Runtime installs visible from this process — the
/// loaded dylib (`ORT_DYLIB_PATH`), its sibling provider DLLs, every
/// `onnxruntime*` dist in the active venv, the CUDA/cuDNN runtime DLLs
/// the `nvidia*` dists ship, and the CUDA-usability probe. Returns a
/// plain dict; save it with `save_runtime_info` next to bench numbers.
#[cfg(feature = "extension-module")]
#[pyfunction(name = "runtime_info")]
fn runtime_info_py(py: Python<'_>) -> PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
    // The interpreter knows its own prefix even when the venv was never
    // activated (no VIRTUAL_ENV) — pass it so the install scan works.
    let venv: Option<String> = std::env::var("VIRTUAL_ENV").ok().or_else(|| {
        py.import("sys")
            .ok()?
            .getattr("prefix")
            .ok()?
            .extract()
            .ok()
    });
    value_to_py(py, &diag::runtime_report_for(venv.as_deref()))
}

/// Set the paths the ONNX Runtime loader needs: `dylib_path` becomes
/// `ORT_DYLIB_PATH` (must exist; effective when set before the first
/// `open()` in this process), `dll_dirs` are prepended to `PATH`
/// (effective immediately — pass `runtime_info()["load_dirs"]` to
/// reproduce a saved layout). Returns the post-call `runtime_info`.
#[cfg(feature = "extension-module")]
#[pyfunction(name = "configure_runtime")]
#[pyo3(signature = (dylib_path=None, dll_dirs=None))]
fn configure_runtime_py(
    py: Python<'_>,
    dylib_path: Option<String>,
    dll_dirs: Option<Vec<String>>,
) -> PyResult<pyo3::Bound<'_, pyo3::PyAny>> {
    let rep = diag::configure_runtime(dylib_path.as_deref(), &dll_dirs.unwrap_or_default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
    value_to_py(py, &rep)
}

/// Write [`diag::runtime_report`] as pretty JSON to `path` (creating
/// parent dirs) and return the path written.
#[cfg(feature = "extension-module")]
#[pyfunction(name = "save_runtime_info")]
fn save_runtime_info_py(path: &str) -> PyResult<String> {    diag::save_runtime_report(std::path::Path::new(path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

#[cfg(feature = "extension-module")]
#[pymodule]
fn embroider(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyJinaV5>()?;
    m.add_class::<PyJinaTokenizer>()?;
    m.add_class::<PyReranker>()?;
    m.add_class::<PyDecider>()?;
    m.add_class::<PyLayaDecider>()?;
    m.add_class::<PyJinaV5Vision>()?;
    m.add_function(pyo3::wrap_pyfunction!(available_models, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cache_info_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(cache_info_files_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(vision_target_size_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(runtime_info_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(configure_runtime_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(save_runtime_info_py, m)?)?;
    m.add("VISION_NANO_MODEL", VISION_NANO_MODEL)?;
    m.add("NATIVE_DIM", NATIVE_DIM)?;
    m.add("MAX_LENGTH", MAX_LENGTH)?;
    m.add("MODEL_MAX_TOKENS", MODEL_MAX_TOKENS)?;
    m.add("FP32_TEXT_MODEL", FP32_TEXT_MODEL)?;
    m.add("NANO_TEXT_MODEL", NANO_TEXT_MODEL)?;
    Ok(())
}
