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
/// Nano tier (EuroBERT, 239M, 768-dim, 8K ctx). Jina ships official ONNX
/// in-repo (fp32 + fp16 + dynamic-int8) — no mirror needed. Measured
/// (nano probe): fp32/fp16 faithful @0.99994, int8 keeps rank @0.99980
/// with zero top-5 flips; q4/q4f16 killed @0.957 with 5/5 pos-1 flips.
pub const NANO_TEXT_MODEL: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";

/// One downloadable artifact: the repo holding it plus the model file
/// inside that repo. The external-data sidecar is derived (same stem +
/// `.onnx_data`); unknown layouts warn-and-continue at fetch time.
#[derive(Clone, Copy, Debug)]
pub struct Artifact<'a> {
    pub repo: &'a str,
    pub file: &'a str,
}

impl Artifact<'_> {
    /// `onnx/model_fp16.onnx` → `onnx/model_fp16.onnx_data`. Returns
    /// `None` for non-`.onnx` files (inline weights assumed, no sidecar).
    pub fn sidecar(self) -> Option<String> {
        self.file
            .strip_suffix(".onnx")
            .map(|stem| format!("{stem}.onnx_data"))
    }
}

/// Static model contract: everything `open()` needs beyond the id.
/// Ladders are the official Matryoshka levels (small tops at 1024,
/// nano at 768); `max_tokens` is the positional ceiling (Qwen3 32768,
/// EuroBERT 8192). Precision maps list only MEASURED artifacts —
/// unlisted (precision, model) pairs fall back to the fp32 artifact
/// rather than fetching unvalidated weights.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    pub id: &'static str,
    pub native_dim: usize,
    pub max_tokens: usize,
    pub ladder: &'static [usize],
    pub fp32: Artifact<'static>,
    pub fp16: Option<Artifact<'static>>,
    pub int8: Option<Artifact<'static>>,
}

pub const TEXT_SMALL: ModelSpec = ModelSpec {
    id: FP32_TEXT_MODEL,
    native_dim: 1024,
    max_tokens: 32768,
    ladder: &[32, 64, 128, 256, 512, 768, 1024],
    fp32: Artifact { repo: FP32_TEXT_MODEL, file: "onnx/model.onnx" },
    fp16: Some(Artifact { repo: FP16_TEXT_MODEL, file: "onnx/model.onnx" }),
    int8: None, // killed in the quant spike @0.93 drift — never shipped
};

pub const TEXT_NANO: ModelSpec = ModelSpec {
    id: NANO_TEXT_MODEL,
    native_dim: 768,
    max_tokens: 8192,
    ladder: &[32, 64, 128, 256, 512, 768],
    fp32: Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model.onnx" },
    fp16: Some(Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model_fp16.onnx" }),
    int8: Some(Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model_quantized.onnx" }),
};

/// Builtin registry. The default id stays `TEXT_SMALL.id` — adding models
/// never moves existing graphs (different weights = different vector space).
pub fn builtin_models() -> &'static [ModelSpec] {
    &[TEXT_SMALL, TEXT_NANO]
}

/// Look up a canonical id in the registry. Unknown ids (custom repos,
/// omni tower, mirrors) return `None` and take the legacy path: the id
/// itself is the fp32 repo with the default `onnx/model.onnx` layout.
pub fn lookup_model(model_id: &str) -> Option<ModelSpec> {
    builtin_models().iter().find(|m| m.id == model_id).copied()
}

/// Select the acquisition artifact for a (model, precision) request.
/// `Auto` must already be resolved via `Precision::resolve` before
/// calling (auto never yields `Int8`). Unlisted pairs fall back to the
/// spec fp32 artifact; unknown ids fall back to the legacy layout.
pub fn artifact_for<'a>(model_id: &'a str, precision: Precision) -> Artifact<'a> {
    match lookup_model(model_id) {
        Some(spec) => match precision {
            Precision::Fp16 => spec.fp16.unwrap_or(spec.fp32),
            Precision::Int8 => spec.int8.unwrap_or(spec.fp32),
            _ => spec.fp32,
        },
        None => Artifact { repo: model_id, file: "onnx/model.onnx" },
    }
}

/// Legacy repo-only selection (frozen contract: only the default id maps
/// to the FP16 mirror; everything else is the identity). Prefer
/// `artifact_for`, which additionally resolves the in-repo file layout
/// (official multi-file repos) and the int8 tier.
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

    #[test]
    fn artifact_sidecar_derives_from_stem() {
        let a = Artifact { repo: "x/y", file: "onnx/model_fp16.onnx" };
        assert_eq!(a.sidecar().as_deref(), Some("onnx/model_fp16.onnx_data"));
        let b = Artifact { repo: "x/y", file: "model.bin" };
        assert_eq!(b.sidecar(), None);
    }

    #[test]
    fn registry_resolves_nano_artifacts() {
        let spec = lookup_model(NANO_TEXT_MODEL).expect("nano registered");
        assert_eq!(spec.native_dim, 768);
        assert_eq!(spec.max_tokens, 8192);
        assert_eq!(spec.ladder, &[32, 64, 128, 256, 512, 768]);
        let fp16 = artifact_for(NANO_TEXT_MODEL, Precision::Fp16);
        assert_eq!((fp16.repo, fp16.file), (NANO_TEXT_MODEL, "onnx/model_fp16.onnx"));
        let int8 = artifact_for(NANO_TEXT_MODEL, Precision::Int8);
        assert_eq!((int8.repo, int8.file), (NANO_TEXT_MODEL, "onnx/model_quantized.onnx"));
        // Unlisted pair (small has no int8) falls back to fp32, never to junk.
        let small_int8 = artifact_for(FP32_TEXT_MODEL, Precision::Int8);
        assert_eq!((small_int8.repo, small_int8.file), (FP32_TEXT_MODEL, "onnx/model.onnx"));
    }

    #[test]
    fn registry_unknown_id_takes_legacy_layout() {
        assert!(lookup_model("someone/custom-embed").is_none());
        let a = artifact_for("someone/custom-embed", Precision::Fp16);
        assert_eq!((a.repo, a.file), ("someone/custom-embed", "onnx/model.onnx"));
    }
}
