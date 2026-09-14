//! HuggingFace acquisition (blocking, cached).
//!
//! The `hf-hub` pattern both consumers used to reimplement: validated
//! `owner/name` parsing, client construction with an optional cache-dir
//! override, and the cheap tokenizer-only fetch that lets token counting
//! work without opening an ONNX session (lazy encoder lifecycle).

use crate::error::Result;
use crate::policy::Precision;
use anyhow::anyhow;
use std::path::PathBuf;

/// Default FP32 weights (Jina's official optimum export).
pub const FP32_TEXT_MODEL: &str = "jinaai/jina-embeddings-v5-text-small-retrieval";
/// FP16 mirror of the same export (same graph + contract, weights cast
/// FP32→FP16, IO kept FP32). Published alongside the quant-spike results;
/// CUDA-only by design (FP16 on CPU runs >40x slower than FP32-CPU).
pub const FP16_TEXT_MODEL: &str =
    "opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16";

/// Select the acquisition repo for a precision request. The mapping
/// applies ONLY to the default FP32 id: an explicit `model_id` (custom
/// repo, omni tower, local mirror) always wins untouched, so non-text
/// models can never be redirected at FP16 weights. `Auto` must already
/// be resolved via `Precision::resolve` before calling.
pub fn repo_for_precision(model_id: &str, precision: Precision) -> &str {
    if model_id == FP32_TEXT_MODEL && precision == Precision::Fp16 {
        FP16_TEXT_MODEL
    } else {
        model_id
    }
}

/// Split an `owner/name` model id — validated before any network access.
pub fn parse_owner_name(model_id: &str) -> Result<(String, String)> {
    model_id
        .split_once('/')
        .map(|(owner, name)| (owner.to_string(), name.to_string()))
        .ok_or_else(|| anyhow!("model_id must be 'owner/name', got '{model_id}'"))
}

/// Build an HF client with an optional cache-directory override.
pub(crate) fn hf_client(cache_dir: Option<PathBuf>) -> Result<hf_hub::HFClientSync> {
    if let Some(dir) = cache_dir {
        Ok(hf_hub::HFClientBuilder::new()
            .cache_dir(dir)
            .build()
            .map(hf_hub::HFClientSync::from_inner)
            .map_err(|e| anyhow!("{e}"))?
            .map_err(|e| anyhow!("{e}"))?)
    } else {
        Ok(hf_hub::HFClientSync::new().map_err(|e| anyhow!("{e}"))?)
    }
}

/// Fetch only `tokenizer.json` — the cheap acquisition path that lets token
/// counting work without opening the ONNX session.
pub fn fetch_tokenizer_file(
    model_id: &str,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    let (owner, name) = parse_owner_name(model_id)?;
    let client = hf_client(cache_dir)?;
    let repo = client.model(owner, name);
    repo.download_file()
        .filename("tokenizer.json".to_string())
        .maybe_revision(revision.map(str::to_string))
        .send()
        .map_err(|e| anyhow!("tokenizer.json missing for '{model_id}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_owner_name_splits_once() {
        let (owner, name) = parse_owner_name("jinaai/some-model").unwrap();
        assert_eq!(owner, "jinaai");
        assert_eq!(name, "some-model");
    }

    #[test]
    fn parse_owner_name_rejects_bare_id() {
        let err = parse_owner_name("no-slash").unwrap_err().to_string();
        assert!(err.contains("model_id must be 'owner/name'"), "{err}");
    }

    #[test]
    fn repo_selection_maps_default_id_to_fp16() {
        assert_eq!(
            repo_for_precision(FP32_TEXT_MODEL, Precision::Fp16),
            FP16_TEXT_MODEL
        );
        assert_eq!(
            repo_for_precision(FP32_TEXT_MODEL, Precision::Fp32),
            FP32_TEXT_MODEL
        );
    }

    #[test]
    fn repo_selection_never_redirects_explicit_ids() {
        // Omni tower, custom repos, mirrors: precision is moot, the id wins.
        for id in [
            "jinaai/jina-embeddings-v5-omni-small-retrieval",
            "someone/custom-embed",
        ] {
            assert_eq!(repo_for_precision(id, Precision::Fp16), id);
            assert_eq!(repo_for_precision(id, Precision::Auto), id);
        }
    }
}
