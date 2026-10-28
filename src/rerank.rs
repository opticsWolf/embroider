//! The cross-encoder rerank contract.
//!
//! Pair encoding → single logit. Verified artifacts (plan
//! `docs/rerank-typed-decision-plan.md` §A0, ORT 1.29-safe opset 14,
//! torch↔ONNX parity 2.9e-6 MiniLM-L-6 / 7.6e-6 mMARCO on the A0 set):
//!
//! - `cross-encoder/ms-marco-MiniLM-L-6-v2` (English, 22.7M, WordPiece,
//!   3-input graph incl. token_type_ids)
//! - `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, 118M,
//!   SentencePiece-Unigram, 2-input graph — no token_type_ids)
//!
//! Session building reuses the shared [`crate::jina::build_session`]
//! contract discovery (feed token_type_ids only when the graph declares
//! it; read the `logits` output). Pair tokenization is the reranker half:
//! truncate-doc-side-first (`OnlySecond`) over `[query, doc]` pairs with
//! per-batch longest padding.

use crate::acquire::{
    artifact_for_task, fetch_tokenizer_file, hf_client, lookup_task, parse_owner_name,
};
use crate::error::{oe, Result};
use crate::jina::{build_session, load_tokenizer};
use crate::policy::{resolve_provider_names, DeviceReq, Precision, SessionPolicy};
use anyhow::anyhow;
use ndarray::Array2;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams, TruncationStrategy};

/// Default pair cap: the cross-encoder standard. Both spike models are
/// 512-position; `open()` defaults to the registered ceiling, explicit
/// files default here.
pub const RERANK_DEFAULT_MAX: usize = 512;
/// Absolute cap for explicit files (largest ceiling on the roadmap —
/// Qwen3-seq-cls at 32K). Explicit files are caller-asserted; without a
/// registry entry there is no per-model ceiling to validate against.
pub const RERANK_ABSOLUTE_MAX: usize = 32768;

/// Validate an optional pair cap against one task model's positional
/// ceiling before any I/O: None selects the model's full context.
/// Returns the effective limit.
pub(crate) fn check_pair_limit_for(
    max_length: Option<usize>,
    ceiling: usize,
) -> Result<usize> {
    match max_length {
        None => Ok(ceiling),
        Some(n) if (1..=ceiling).contains(&n) => Ok(n),
        Some(n) => Err(anyhow!(
            "max_length must be within 1..={ceiling} (model position ceiling), got {n}"
        )),
    }
}

/// Load a pair tokenizer: truncate-doc-side-first (`OnlySecond` — the
/// query is never cut to fit the document) with per-batch longest padding
/// (`score_many` pads each batch to its own longest pair, never to the
/// cap). Pairs that still overflow after doc truncation (query alone
/// exceeds the cap) fail loudly at score time, not here.
pub(crate) fn load_pair_tokenizer(
    tok_path: &Path,
    max_length: usize,
) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = load_tokenizer(tok_path, None)?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            strategy: TruncationStrategy::OnlySecond,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("pair truncation setup failed: {e}"))?;
    // Infallible (no validation to fail) — unlike with_truncation above.
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    Ok(tokenizer)
}

pub struct Reranker {
    session: RwLock<Session>,
    tokenizer: tokenizers::Tokenizer,
    max_len: usize,
    model_id: String,
    used_cuda: bool,
    precision: Precision,
    feed_token_type_ids: bool,
    output_name: String,
}

impl Reranker {
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
        device: DeviceReq,
        max_length: Option<usize>,
        precision: Precision,
        cpu_arena: bool,
    ) -> Result<Self> {
        // Per-model contract: registered ids validate against their own
        // ceiling; unknown ids fall back to the cross-encoder standard.
        let ceiling =
            lookup_task(model_id).map(|s| s.max_tokens).unwrap_or(RERANK_DEFAULT_MAX);
        let max_length = check_pair_limit_for(max_length, ceiling)?;

        // Same device→precision ordering as JinaV5: precision follows the
        // device the session will actually land on.
        let (_, used_cuda) = resolve_provider_names(device);
        let precision = precision.resolve(used_cuda);
        if precision == Precision::Fp16 && !used_cuda {
            eprintln!(
                "embroider: FP16 weights on CPU — this runs >40x slower than FP32-CPU \
                 (emulated half-precision kernels). Explicit request honoured; pass \
                 precision='fp32' (or 'auto') for CPU sessions."
            );
        }
        // Precision selects the acquisition artifact from the task registry.
        // Unknown ids take the legacy path (id = fp32 repo, default file).
        let artifact = artifact_for_task(model_id, precision);

        // ---- model acquisition (per-artifact file layout) ----
        // No sidecar probe: both spike artifacts are A0-verified
        // single-file exports (ORT loads them standalone); a future
        // external-data export fails loudly at session load instead.
        let (owner, name) = parse_owner_name(artifact.repo)?;
        let client = hf_client(cache_dir.clone())?;
        let repo = client.model(owner, name);
        let rev = revision.map(str::to_string);
        let onnx_path = repo
            .download_file()
            .filename(artifact.file.to_string())
            .maybe_revision(rev)
            .send()
            .map_err(|e| anyhow!("{} missing for '{}': {e}", artifact.file, artifact.repo))?;
        // ---- pair tokenizer (truncation + batch padding configured) ----
        let tok_path = fetch_tokenizer_file(artifact.repo, revision, cache_dir)?;
        let tokenizer = load_pair_tokenizer(&tok_path, max_length)?;

        let loaded =
            build_session(&onnx_path, device, &SessionPolicy::rerank_pair(), cpu_arena)?;

        Ok(Self {
            session: RwLock::new(loaded.session),
            tokenizer,
            max_len: max_length,
            model_id: model_id.to_string(),
            used_cuda: loaded.used_cuda,
            precision,
            feed_token_type_ids: loaded.feed_token_type_ids,
            output_name: loaded.output_name,
        })
    }

    /// Load from explicit local files — no network access. An external-data
    /// sidecar must sit next to `onnx_path` (ORT resolves it relative to the
    /// model file). Missing files fail before any tokenizer or session work;
    /// precision reads `fp32` (the files are what they are).
    pub fn open_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        device: DeviceReq,
        max_length: Option<usize>,
        cpu_arena: bool,
    ) -> Result<Self> {
        let max_length = match max_length {
            None => RERANK_DEFAULT_MAX,
            Some(n) if (1..=RERANK_ABSOLUTE_MAX).contains(&n) => n,
            Some(n) => {
                return Err(anyhow!(
                    "max_length must be within 1..={RERANK_ABSOLUTE_MAX} for explicit files, got {n}"
                ))
            }
        };
        if !onnx_path.is_file() {
            return Err(anyhow!("onnx model not found: {}", onnx_path.display()));
        }
        if !tokenizer_path.is_file() {
            return Err(anyhow!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            ));
        }
        let tokenizer = load_pair_tokenizer(tokenizer_path, max_length)?;
        let loaded =
            build_session(onnx_path, device, &SessionPolicy::rerank_pair(), cpu_arena)?;
        Ok(Self {
            session: RwLock::new(loaded.session),
            tokenizer,
            max_len: max_length,
            model_id: onnx_path.display().to_string(),
            used_cuda: loaded.used_cuda,
            precision: Precision::Fp32,
            feed_token_type_ids: loaded.feed_token_type_ids,
            output_name: loaded.output_name,
        })
    }

    /// Score one (query, document) pair: relevance logit, higher = more
    /// relevant. Thin wrapper over [`Reranker::score_many`].
    pub fn score_one(&self, query: &str, doc: &str) -> Result<f32> {
        Ok(self.score_many(&[(query, doc)])?[0])
    }

    /// Score pairs in one padded batch (per-batch longest padding — one
    /// session run per call). Order-preserving: output[i] scores pairs[i].
    pub fn score_many(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let inputs: Vec<(&str, &str)> =
            pairs.iter().map(|(q, d)| (*q, *d)).collect();
        let enc = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow!("pair tokenization failed: {e}"))?;
        // OnlySecond truncation caps the doc side; leftover overflow means
        // the query alone exceeds the cap — loud error, not silent garbage
        // (over-long ids would read past the positional embeddings).
        for (i, e) in enc.iter().enumerate() {
            if e.len() > self.max_len {
                return Err(anyhow!(
                    "pair {i} encodes to {} tokens, over the max_length={} cap \
                     (the query alone exceeds it — shorten the query or raise max_length)",
                    e.len(),
                    self.max_len
                ));
            }
            if e.len() == 0 {
                return Err(anyhow!("pair {i}: tokenizer returned zero tokens"));
            }
        }
        let n = enc.len();
        let t = enc.iter().map(|e| e.len()).max().unwrap_or(0);
        let mut ids = Vec::with_capacity(n * t);
        let mut mask = Vec::with_capacity(n * t);
        let mut type_ids = Vec::with_capacity(n * t);
        for e in &enc {
            ids.extend(e.get_ids().iter().map(|&x| x as i64));
            mask.extend(e.get_attention_mask().iter().map(|&x| x as i64));
            type_ids.extend(e.get_type_ids().iter().map(|&x| x as i64));
        }
        let ids =
            Array2::from_shape_vec((n, t), ids).map_err(|e| anyhow!("input shape: {e}"))?;
        let mask_arr =
            Array2::from_shape_vec((n, t), mask).map_err(|e| anyhow!("mask shape: {e}"))?;

        let logits = {
            let mut guard =
                self.session.write().map_err(|_| anyhow!("session lock poisoned"))?;
            let ids_t = oe(ort::value::TensorRef::from_array_view(&ids))?;
            let mask_t = oe(ort::value::TensorRef::from_array_view(&mask_arr))?;
            let outputs = if self.feed_token_type_ids {
                let tids = Array2::from_shape_vec((n, t), type_ids)
                    .map_err(|e| anyhow!("type-ids shape: {e}"))?;
                let tids_t = oe(ort::value::TensorRef::from_array_view(&tids))?;
                oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t,
                    "token_type_ids" => tids_t
                }))?
            } else {
                oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "attention_mask" => mask_t
                }))?
            };
            oe(outputs[self.output_name.as_str()].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow!("expected [B,1] logits: {e}"))?
        };
        if logits.shape() != [n, 1] {
            return Err(anyhow!(
                "logits shape {:?} does not match {n} scored pairs",
                logits.shape()
            ));
        }
        Ok(logits.column(0).to_vec())
    }

    /// Effective pair cap (truncation ceiling for scores).
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Model id — or the ONNX path when opened via `open_files`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the session actually runs on CUDA.
    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    /// Weight precision the session was opened with (`auto` already
    /// resolved against the landed device). Explicit files report `fp32`.
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_limit_defaults_to_ceiling() {
        use crate::acquire::RERANK_MINILM_MODEL;
        let spec = lookup_task(RERANK_MINILM_MODEL).unwrap();
        assert_eq!(check_pair_limit_for(None, spec.max_tokens).unwrap(), 512);
        assert_eq!(check_pair_limit_for(Some(128), spec.max_tokens).unwrap(), 128);
        assert!(check_pair_limit_for(Some(0), spec.max_tokens).is_err());
        assert!(check_pair_limit_for(Some(513), spec.max_tokens).is_err());
    }

    #[test]
    fn open_files_rejects_missing_model_before_anything() {
        let e = Reranker::open_files(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
            DeviceReq::Cpu,
            None,
            false,
        )
        .err()
        .expect("open_files should fail")
        .to_string();
        assert!(e.contains("onnx model not found"), "{e}");
    }

    #[test]
    fn open_files_rejects_bad_pair_limit_before_io() {
        for bad in [Some(0usize), Some(RERANK_ABSOLUTE_MAX + 1)] {
            let e = Reranker::open_files(
                Path::new("/nonexistent/model.onnx"),
                Path::new("/nonexistent/tokenizer.json"),
                DeviceReq::Cpu,
                bad,
                false,
            )
            .err()
            .expect("open_files should fail")
            .to_string();
            assert!(e.contains("1..=32768"), "{bad:?}: {e}");
        }
    }

    #[test]
    fn open_rejects_out_of_range_limit_before_io() {
        use crate::acquire::RERANK_MMARCO_MODEL;
        let e = Reranker::open(
            RERANK_MMARCO_MODEL,
            None,
            None,
            DeviceReq::Cpu,
            Some(513),
            Precision::Auto,
            false,
        )
        .err()
        .expect("open should fail")
        .to_string();
        assert!(e.contains("1..=512"), "{e}");
    }

    #[test]
    fn open_rejects_unqualified_model_id_before_io() {
        let e = Reranker::open(
            "no-slash",
            None,
            None,
            DeviceReq::Cpu,
            None,
            Precision::Auto,
            false,
        )
        .err()
        .expect("open should fail")
        .to_string();
        assert!(e.contains("model_id must be 'owner/name'"), "{e}");
    }
}
