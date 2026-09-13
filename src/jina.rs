//! The Jina v5 text-embedding contract.
//!
//! Exact port of `okfgraph.components.embedding.EmbeddingEngine._encode`:
//! task prefix → tokenize (8192) → ONNX forward → **last-token pooling** → L2
//! → Matryoshka truncate → re-normalise. Any deviation here silently moves the
//! unified text/omni vector space, so the parity harness in the consumers pins
//! this at ≤1e-5.
//!
//! Session building applies an explicit [`SessionPolicy`] (the measured
//! `text_embed()` tuning); device fallback semantics mirror bobine and the
//! Python router: CUDA is opportunistic, never fatal.

use crate::acquire::{fetch_tokenizer_file, hf_client, parse_owner_name};
use crate::error::{oe, Result};
use crate::policy::{resolve_provider_names, DeviceReq, SessionPolicy};
use crate::providers::apply_providers;
use anyhow::anyhow;
use ndarray::{Array1, Array2};
use ort::session::Session;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Hard truncation limit, mirroring `tokenizer(..., max_length=8192)`.
pub const MAX_LENGTH: usize = 8192;
/// Native width of jina-embeddings-v5 outputs.
pub const NATIVE_DIM: usize = 1024;
/// Official Matryoshka levels (warning-only outside these, error outside 32..=1024).
pub(crate) const ALLOWED_DIMS: &[usize] = &[32, 64, 128, 256, 512, 768, 1024];

/// Task prefix (``Query:`` / ``Document:``) — idempotent: already-prefixed
/// text passes through untouched (exact port of the ``_encode`` guard).
pub(crate) fn task_prefixed<'a>(text: &'a str, task: &str) -> Cow<'a, str> {
    if text.starts_with("Query:") || text.starts_with("Document:") {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(format!("{task}: {text}"))
    }
}

/// L2-normalise at full width, truncate to ``dim``, re-normalise the
/// truncated head (exact port of the ``_encode`` post-processing / Matryoshka
/// protocol: normalise → truncate → re-normalise).
pub(crate) fn l2_truncate(pooled: &Array1<f32>, dim: usize) -> Vec<f32> {
    let norm = pooled.mapv(|x| x * x).sum().sqrt();
    let normed = if norm > 0.0 { pooled.mapv(|x| x / norm) } else { pooled.clone() };
    let mut v: Vec<f32> = normed.iter().take(dim).copied().collect();
    v.resize(dim, 0.0);
    let n2 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n2 > 0.0 {
        for x in &mut v {
            *x /= n2;
        }
    }
    v
}

/// Validate the shared Matryoshka dim range before any I/O.
pub(crate) fn check_truncate_dim(truncate_dim: usize) -> Result<()> {
    if truncate_dim == 0 || truncate_dim > NATIVE_DIM {
        return Err(anyhow!(
            "truncate_dim must be within 1..={NATIVE_DIM}, got {truncate_dim}"
        ));
    }
    if truncate_dim < 32 {
        return Err(anyhow!("truncate_dim must be >= 32, got {truncate_dim}"));
    }
    if !ALLOWED_DIMS.contains(&truncate_dim) {
        eprintln!(
            "embroider: truncate_dim={truncate_dim} is not an official Matryoshka level \
             {ALLOWED_DIMS:?}; retrieval quality may be suboptimal."
        );
    }
    Ok(())
}

/// Load a tokenizer with the Jina 8192-token truncation policy.
pub(crate) fn load_tokenizer(tok_path: &std::path::Path) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = tokenizers::Tokenizer::from_file(tok_path)
        .map_err(|e| anyhow!("tokenizer.json failed to load: {e}"))?;
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_LENGTH,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("truncation setup failed: {e}"))?;
    Ok(tokenizer)
}

/// Token count without special tokens — replaces
/// `len(tokenizer.encode(t, add_special_tokens=False))` for the
/// context-window guard, so the transformers dependency can go.
pub(crate) fn count_tokens_in(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<usize> {
    Ok(tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("tokenization failed: {e}"))?
        .len())
}

/// Tokenizer-only handle: exact token counts without the ONNX session.
/// Lets budgeted reads and the context-window guard stay cold while the
/// multi-hundred-MB session open waits for the first real encode.
pub struct TokenizerHandle {
    tokenizer: tokenizers::Tokenizer,
}

impl TokenizerHandle {
    pub fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let tok_path = fetch_tokenizer_file(model_id, revision, cache_dir)?;
        Ok(Self { tokenizer: load_tokenizer(&tok_path)? })
    }

    pub fn open_files(tokenizer_path: &Path) -> Result<Self> {
        if !tokenizer_path.is_file() {
            return Err(anyhow!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            ));
        }
        Ok(Self { tokenizer: load_tokenizer(tokenizer_path)? })
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        count_tokens_in(&self.tokenizer, text)
    }
}

/// An opened session plus its discovered contract: the expensive,
/// device-bound half of `JinaV5::open`, shared by the HF and explicit-path
/// constructors so both validate the same export contract.
pub(crate) struct LoadedSession {
    session: Session,
    used_cuda: bool,
    feed_token_type_ids: bool,
    output_name: String,
}

/// Build + contract-check a session from an ONNX file already on disk.
/// No network access: the caller owns acquisition (HF fetch or explicit path).
pub(crate) fn build_session(
    onnx_path: &Path,
    device: DeviceReq,
    policy: &SessionPolicy,
) -> Result<LoadedSession> {
    let (provider_names, used_cuda) = resolve_provider_names(device);

    let mut builder = oe(ort::session::Session::builder())?;
    if let Some(level) = policy.opt_level {
        builder = oe(builder.with_optimization_level(level))?;
    }
    if let Some(n) = policy.intra_threads {
        builder = oe(builder.with_intra_threads(n))?;
    }
    if let Some(n) = policy.inter_threads {
        builder = oe(builder.with_inter_threads(n))?;
    }
    builder = apply_providers(builder, &provider_names)?;
    let session = oe(builder.commit_from_file(onnx_path))
        .map_err(|e| anyhow!("loading {}: {e:#}", onnx_path.display()))?;

    // Contract discovery (robust to export variants): Jina v5's optimum
    // export declares only input_ids + attention_mask; token_type_ids is fed
    // solely when the graph asks for it (v2-style).
    let has_input = |want: &str| session.inputs().iter().any(|o| o.name() == want);
    for need in ["input_ids", "attention_mask"] {
        if !has_input(need) {
            let have: Vec<&str> =
                session.inputs().iter().map(|o| o.name()).collect();
            return Err(anyhow!(
                "ONNX export lacks required input '{need}' (has {have:?})"
            ));
        }
    }
    let feed_token_type_ids = has_input("token_type_ids");
    let output_name = if session.outputs().iter().any(|o| o.name() == "last_hidden_state") {
        "last_hidden_state".to_string()
    } else {
        session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| anyhow!("ONNX export declares no outputs"))?
    };
    Ok(LoadedSession { session, used_cuda, feed_token_type_ids, output_name })
}

pub struct JinaV5 {
    session: RwLock<Session>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    model_id: String,
    used_cuda: bool,
    feed_token_type_ids: bool,
    output_name: String,
}

impl JinaV5 {
    pub fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
        truncate_dim: usize,
        device: DeviceReq,
    ) -> Result<Self> {
        check_truncate_dim(truncate_dim)?;

        // ---- model acquisition (optimum `subfolder="onnx"` layout) ----
        let (owner, name) = parse_owner_name(model_id)?;
        let client = hf_client(cache_dir.clone())?;
        let repo = client.model(owner, name);
        let rev = revision.map(str::to_string);
        let onnx_path = repo
            .download_file()
            .filename("onnx/model.onnx".to_string())
            .maybe_revision(rev.clone())
            .send()
            .map_err(|e| anyhow!("onnx/model.onnx missing for '{model_id}': {e}"))?;
        // External-data sidecar must sit next to model.onnx; the hub cache
        // layout preserves that, so a plain fetch into the same dir suffices.
        if repo
            .download_file()
            .filename("onnx/model.onnx_data".to_string())
            .maybe_revision(rev)
            .send()
            .is_err()
        {
            eprintln!("embroider: no onnx/model.onnx_data sidecar; assuming inline weights");
        }
        // ---- tokenizer (no padding here; single-doc encodes need none) ----
        // Shared with TokenizerHandle so both paths load identical truncation.
        let tok_path = fetch_tokenizer_file(model_id, revision, cache_dir)?;
        let tokenizer = load_tokenizer(&tok_path)?;

        let loaded = build_session(&onnx_path, device, &SessionPolicy::text_embed())?;

        Ok(Self {
            session: RwLock::new(loaded.session),
            tokenizer,
            dim: truncate_dim,
            model_id: model_id.to_string(),
            used_cuda: loaded.used_cuda,
            feed_token_type_ids: loaded.feed_token_type_ids,
            output_name: loaded.output_name,
        })
    }

    /// Load from explicit local files — no network access. An external-data
    /// sidecar must sit next to `onnx_path` (ORT resolves it relative to the
    /// model file, same as the HF cache layout). Missing files fail before
    /// any tokenizer or session work.
    pub fn open_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        truncate_dim: usize,
        device: DeviceReq,
    ) -> Result<Self> {
        check_truncate_dim(truncate_dim)?;
        if !onnx_path.is_file() {
            return Err(anyhow!("onnx model not found: {}", onnx_path.display()));
        }
        if !tokenizer_path.is_file() {
            return Err(anyhow!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            ));
        }
        let tokenizer = load_tokenizer(tokenizer_path)?;
        let loaded = build_session(onnx_path, device, &SessionPolicy::text_embed())?;
        Ok(Self {
            session: RwLock::new(loaded.session),
            tokenizer,
            dim: truncate_dim,
            model_id: onnx_path.display().to_string(),
            used_cuda: loaded.used_cuda,
            feed_token_type_ids: loaded.feed_token_type_ids,
            output_name: loaded.output_name,
        })
    }

    /// Exact port of `EmbeddingEngine._encode` (prefix → last-token → L2 →
    /// truncate → re-normalise).
    pub fn encode_one(&self, text: &str, task: &str) -> Result<Vec<f32>> {
        let text = task_prefixed(text, task);
        let text: &str = &text;

        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenization failed: {e}"))?;
        let ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> =
            enc.get_attention_mask().iter().map(|&x| x as i64).collect();
        let t = ids.len();
        if t == 0 {
            return Err(anyhow!("tokenizer returned zero tokens"));
        }
        let mask_sum: i64 = mask.iter().sum();
        let ids = Array2::from_shape_vec((1, t), ids)
            .map_err(|e| anyhow!("input shape: {e}"))?;
        let mask_arr =
            Array2::from_shape_vec((1, t), mask).map_err(|e| anyhow!("mask shape: {e}"))?;

        let hidden = {
            let mut guard =
                self.session.write().map_err(|_| anyhow!("session lock poisoned"))?;
            let ids_t = oe(ort::value::TensorRef::from_array_view(&ids))?;
            let mask_t = oe(ort::value::TensorRef::from_array_view(&mask_arr))?;
            let outputs = if self.feed_token_type_ids {
                let zeros = Array2::<i64>::zeros((1, t));
                let zeros_t = oe(ort::value::TensorRef::from_array_view(&zeros))?;
                oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "token_type_ids" => zeros_t
                }))?
            } else {
                oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t
                }))?
            };
            oe(outputs[self.output_name.as_str()].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| anyhow!("expected [B,T,H] output: {e}"))?
        };

        // Last-token pooling: index of final real token, clamped ≥ 0.
        let last_idx = (mask_sum as usize).saturating_sub(1);
        let pooled = hidden.slice(ndarray::s![0, last_idx, ..]).to_owned();
        Ok(l2_truncate(&pooled, self.dim))
    }

    /// Sequential by design: padded batching wastes attention compute on
    /// variable-length documents (matches `_encode_batch`).
    pub fn encode_many(&self, texts: &[String], task: &str) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.encode_one(t, task)).collect()
    }

    /// Configured output dimension (post-Matryoshka).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Model id — or the ONNX path when opened via `open_files`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the session actually runs on CUDA.
    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    /// Token count without special tokens — replaces
    /// `len(tokenizer.encode(t, add_special_tokens=False))` for the
    /// context-window guard, so the transformers dependency can go.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        count_tokens_in(&self.tokenizer, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::borrow::Cow;

    fn norm2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    // ---- task prefixing ------------------------------------------------------

    #[test]
    fn task_prefix_added_once() {
        assert_eq!(task_prefixed("hello", "Document"), "Document: hello");
        assert_eq!(task_prefixed("hello", "Query"), "Query: hello");
    }

    #[test]
    fn task_prefix_is_idempotent() {
        assert_eq!(task_prefixed("Query: hello", "Document"), "Query: hello");
        assert_eq!(task_prefixed("Document: x", "Query"), "Document: x");
    }

    #[test]
    fn task_prefix_borrows_when_untouched() {
        assert!(matches!(task_prefixed("Query: hi", "Document"), Cow::Borrowed(_)));
        assert!(matches!(task_prefixed("hi", "Document"), Cow::Owned(_)));
    }

    // ---- L2 → truncate → re-normalise ---------------------------------------

    #[test]
    fn l2_truncate_unit_input_stays_unit() {
        let v = Array1::from_vec(vec![0.6, 0.8]); // already unit length
        let out = l2_truncate(&v, 2);
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);
        assert!((norm2(&out) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_truncate_normalizes_then_truncates_then_renormalizes() {
        // [3,0,4,0] → unit [0.6,0,0.8,0] → head [0.6,0] → re-norm [1,0].
        // This ordering is the Jina v5 Matryoshka protocol — truncate-then-
        // normalize would give a different (wrong) vector space.
        let v = Array1::from_vec(vec![3.0, 0.0, 4.0, 0.0]);
        let out = l2_truncate(&v, 2);
        assert!((out[0] - 1.0).abs() < 1e-6, "{out:?}");
        assert!(out[1].abs() < 1e-6);
        assert!((norm2(&out) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_truncate_zero_vector_stays_zero() {
        let v = Array1::zeros(4);
        let out = l2_truncate(&v, 2);
        assert!(out.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn l2_truncate_pads_when_dim_exceeds_width() {
        let v = Array1::from_vec(vec![1.0]);
        let out = l2_truncate(&v, 2);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn l2_truncate_preserves_sign() {
        let v = array![0.3, -0.4]; // norm 0.5 → unit [0.6, -0.8]
        let out = l2_truncate(&v, 2);
        assert!((out[0] - 0.6).abs() < 1e-6, "{out:?}");
        assert!((out[1] + 0.8).abs() < 1e-6, "{out:?}");
    }

    // ---- contract constants --------------------------------------------------

    #[test]
    fn matryoshka_levels_sorted_and_bounded() {
        assert!(ALLOWED_DIMS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*ALLOWED_DIMS.last().unwrap(), NATIVE_DIM);
        assert!(ALLOWED_DIMS.contains(&512)); // default dim
        assert_eq!(MAX_LENGTH, 8192);
    }

    #[test]
    fn open_files_rejects_bad_dims_before_io() {
        let e = JinaV5::open_files(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
            0,
            DeviceReq::Cpu,
        ).err().expect("open_files should fail").to_string();
        assert!(e.contains("1..=1024"), "{e}");
    }

    #[test]
    fn open_files_reports_missing_model_before_network() {
        let e = JinaV5::open_files(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
            512,
            DeviceReq::Cpu,
        ).err().expect("open_files should fail").to_string();
        assert!(e.contains("onnx model not found"), "{e}");
    }

    #[test]
    fn tokenizer_open_files_reports_missing_file() {
        let e = TokenizerHandle::open_files(Path::new("/nonexistent/tokenizer.json"))
            .err().expect("open_files should fail").to_string();
        assert!(e.contains("tokenizer file not found"), "{e}");
    }

    // ---- open() validation fires before any network access -------------------

    #[test]
    fn open_rejects_out_of_range_dims_before_io() {
        for bad in [0usize, 2048] {
            let e = JinaV5::open(
                "jinaai/jina-embeddings-v5-text-small-retrieval",
                None, None, bad, DeviceReq::Cpu,
            ).err().expect("open should fail").to_string();
            assert!(e.contains("1..=1024"), "dim {bad}: {e}");
        }
        let e = JinaV5::open(
            "jinaai/jina-embeddings-v5-text-small-retrieval",
            None, None, 16, DeviceReq::Cpu,
        ).err().expect("open should fail").to_string();
        assert!(e.contains(">= 32"), "{e}");
    }

    #[test]
    fn open_rejects_unqualified_model_id_before_io() {
        let e = JinaV5::open("no-slash", None, None, 512, DeviceReq::Cpu)
            .err().expect("open should fail")
            .to_string();
        assert!(e.contains("model_id must be 'owner/name'"), "{e}");
    }
}
