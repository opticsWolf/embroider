//! HuggingFace acquisition (blocking, cached).
//!
//! The `hf-hub` pattern both consumers used to reimplement: validated
//! `owner/name` parsing, client construction with an optional cache-dir
//! override, and the cheap tokenizer-only fetch that lets token counting
//! work without opening an ONNX session (lazy encoder lifecycle).

use crate::error::Result;
use crate::policy::Precision;
use anyhow::anyhow;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
/// Vision repo: the dynamic-grid omni-nano export (Phase 6). One graph
/// (vision tower + merger + EuroBERT text tower) with host-computed grid
/// inputs; sidecars use the `model.onnx.data` layout, hence the explicit
/// sidecar override below. Weights CC BY-NC 4.0 (Jina), same as the text
/// models — see the model card for attribution and the changes statement.
pub const VISION_NANO_REPO: &str =
    "opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx";
/// Canonical vision model id. Short (not `owner/name`): the id names the
/// contract, the spec's artifacts name where the files live.
pub const VISION_NANO_MODEL: &str = "jina-v5-omni-nano-retrieval-vision";

/// One downloadable artifact: the repo holding it plus the model file
/// inside that repo. The external-data sidecar is derived (same stem +
/// `.onnx_data`); unknown layouts warn-and-continue at fetch time.
#[derive(Clone, Copy, Debug)]
pub struct Artifact<'a> {
    pub repo: &'a str,
    pub file: &'a str,
    /// Explicit sidecar file inside the repo. `None` derives the legacy
    /// optimum layout (`onnx/model_fp16.onnx` → `onnx/model_fp16.onnx_data`);
    /// `Some` names it verbatim (the vision export's `model.onnx.data`).
    pub sidecar: Option<&'a str>,
}

impl Artifact<'_> {
    /// Sidecar filename, or `None` for non-`.onnx` files (inline weights
    /// assumed, no sidecar).
    pub fn sidecar(self) -> Option<String> {
        if let Some(explicit) = self.sidecar {
            return Some(explicit.to_string());
        }
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
    /// Text model sharing this model's vector space, if any. Vision
    /// specs set it (image vectors are only comparable to their text
    /// partner's); text specs leave it `None`.
    pub text_partner: Option<&'static str>,
}

pub const TEXT_SMALL: ModelSpec = ModelSpec {
    id: FP32_TEXT_MODEL,
    native_dim: 1024,
    max_tokens: 32768,
    ladder: &[32, 64, 128, 256, 512, 768, 1024],
    fp32: Artifact { repo: FP32_TEXT_MODEL, file: "onnx/model.onnx", sidecar: None },
    fp16: Some(Artifact { repo: FP16_TEXT_MODEL, file: "onnx/model.onnx", sidecar: None }),
    int8: None, // killed in the quant spike @0.93 drift — never shipped
    text_partner: None,
};

pub const TEXT_NANO: ModelSpec = ModelSpec {
    id: NANO_TEXT_MODEL,
    native_dim: 768,
    max_tokens: 8192,
    ladder: &[32, 64, 128, 256, 512, 768],
    fp32: Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model.onnx", sidecar: None },
    fp16: Some(Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model_fp16.onnx", sidecar: None }),
    int8: Some(Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model_quantized.onnx", sidecar: None }),
    text_partner: None,
};

/// Vision contract: dynamic-grid omni-nano (Phase 6). Same 768-dim
/// Matryoshka ladder as text-nano, its `text_partner` — image vectors
/// only compare against text-nano vectors. Prompt fits ≤1295 tokens;
/// `max_tokens` carries the shared EuroBERT positional ceiling (8192).
/// fp16 recipe: ORT converter, keep_io_types, Pow/ReduceMean/Sqrt/
/// Reciprocal/Cos/Sin kept fp32, duplicate Casts deduped. No int8/q4
/// (community q4f16 killed @0.954 drift). Pinned artifact commit + per-file
/// sha256 live in the repo's `manifest.json`.
pub const VISION_NANO: ModelSpec = ModelSpec {
    id: VISION_NANO_MODEL,
    native_dim: 768,
    max_tokens: 8192,
    ladder: &[32, 64, 128, 256, 512, 768],
    fp32: Artifact {
        repo: VISION_NANO_REPO,
        file: "fp32/model.onnx",
        sidecar: Some("fp32/model.onnx.data"),
    },
    fp16: Some(Artifact {
        repo: VISION_NANO_REPO,
        file: "fp16/model.onnx",
        sidecar: Some("fp16/model.onnx.data"),
    }),
    int8: None,
    text_partner: Some(NANO_TEXT_MODEL),
};

/// Builtin registry. The default id stays `TEXT_SMALL.id` — adding models
/// never moves existing graphs (different weights = different vector space).
pub fn builtin_models() -> &'static [ModelSpec] {
    &[TEXT_SMALL, TEXT_NANO, VISION_NANO]
}

/// Look up a canonical id in the registry. Unknown ids (custom repos,
/// omni-small, mirrors) return `None` and take the legacy path: the id
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
        None => Artifact { repo: model_id, file: "onnx/model.onnx", sidecar: None },
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

/// Offline answer to "is this model already cached, and where?" — built
/// by [`cache_info`] without a session, a download, or a device probe.
#[derive(Clone, Debug)]
pub struct CacheReport {
    /// Model id exactly as passed in.
    pub model_id: String,
    /// Artifact repo actually used — may differ from `model_id` (the
    /// default id at fp16 resolves to the FP16 mirror).
    pub repo: String,
    /// Resolved precision tier for registry lookups (`cache_info` sets
    /// `Some`; `auto` is refused). `None` for the generic
    /// [`cache_info_files`] lookup, which has no precision tier.
    pub precision: Option<Precision>,
    /// Effective hub cache directory (override or env-resolved default).
    pub cache_dir: PathBuf,
    /// One entry per probed file — the artifact model file, its external-
    /// data sidecar when the layout has one, and `tokenizer.json`.
    /// `None` = not present in this cache.
    pub files: BTreeMap<String, Option<PathBuf>>,
    /// True when `open()` would succeed offline: model file + tokenizer
    /// both present (a missing sidecar only warns at open, so it keeps
    /// `cached` true).
    pub cached: bool,
    /// Snapshot directory holding the present files
    /// (`<cache>/models--{owner}--{name}/snapshots/<sha>`), when known.
    pub snapshot_path: Option<PathBuf>,
    /// Sum of on-disk sizes of the present files (snapshot pointers are
    /// followed to their blobs).
    pub disk_usage_bytes: u64,
}

/// Offline cache inspection for an arbitrary (repo, files) pair.
///
/// The generic core underneath [`cache_info`]: looks each named file up
/// in the hub cache — offline only, by construction (every lookup runs
/// with `local_files_only`, so it can never touch the network, never
/// build a session, never probe the device). `files` pairs each
/// repo-relative filename with whether it is required: every entry
/// appears in `report.files`, and `cached` is true when all *required*
/// files are present. `repo` must be `owner/name` and at least one file
/// is required — both are validated before any I/O (an empty lookup
/// would report `cached` vacuously true).
///
/// This is the surface non-registry layouts use (bobine's converter
/// models): same offline mechanics, same [`CacheReport`] contract —
/// `model_id` echoes `repo`, and `precision` is `None` (no tier here).
/// Any failure reads as "absent": this report is diagnostic, and
/// `open()` re-fails loudly where it matters.
pub fn cache_info_files(
    repo: &str,
    files: &[(&str, bool)],
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
) -> Result<CacheReport> {
    if files.is_empty() {
        return Err(anyhow!(
            "cache_info_files needs at least one file to look up; pass \
             [(filename, required)] pairs"
        ));
    }
    let (owner, name) = parse_owner_name(repo)?;
    let repo_handle = hf_client(cache_dir.clone())?.model(owner, name);
    let rev = revision.map(str::to_string);
    // The crate joins the repo-relative filename (may contain '/') onto the
    // snapshot dir; re-walking the components re-separates them natively
    // (Windows: `...\onnx/model.onnx` → `...\onnx\model.onnx`) without
    // resolving symlinks — canonicalize would land on the blob, not the
    // snapshot pointer.
    fn cleaned(p: &Path) -> PathBuf {
        p.components().collect::<PathBuf>()
    }
    // The crate's own offline resolution (refs → snapshots/<sha>/<file>,
    // `.no_exist` aware). Any failure reads as "absent".
    let lookup = |filename: &str| {
        repo_handle
            .download_file()
            .filename(filename.to_string())
            .maybe_revision(rev.clone())
            .local_files_only(true)
            .send()
            .ok()
            .map(|p| cleaned(&p))
    };
    let mut file_map: BTreeMap<String, Option<PathBuf>> = BTreeMap::new();
    let mut cached = true;
    for (filename, required) in files {
        let path = lookup(filename);
        if *required && path.is_none() {
            cached = false;
        }
        file_map.insert(filename.to_string(), path);
    }
    let present: Vec<&PathBuf> = file_map.values().flatten().collect();
    // The <sha> dir a found file lives in: first ancestor whose parent is
    // the snapshots/ folder. None when nothing (or no snapshot-rooted
    // file) was found.
    let snapshot_path = present
        .iter()
        .find_map(|p| {
            p.ancestors()
                .find(|a| a.parent().is_some_and(|par| par.file_name().is_some_and(|n| n == "snapshots")))
        })
        .map(|p| p.to_path_buf());
    let disk_usage_bytes = present
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    Ok(CacheReport {
        model_id: repo.to_string(),
        repo: repo.to_string(),
        precision: None,
        cache_dir: cache_dir.unwrap_or_else(hf_hub::resolve_cache_dir),
        files: file_map,
        cached,
        snapshot_path,
        disk_usage_bytes,
    })
}

/// Offline cache inspection for one (model, precision) artifact.
///
/// Resolves the artifact exactly like `JinaV5::open` / `JinaV5Vision::open`
/// (`lookup_model` → `artifact_for`), then delegates the lookup to the
/// generic [`cache_info_files`] core (model file + `tokenizer.json`
/// required, sidecar optional — matching `open()`'s warn-and-continue).
/// The report carries the id as passed plus its resolved tier.
///
/// `precision` must be explicit: `auto` needs the landed device, and a
/// guessed tier would report the wrong repo (fp32 official vs fp16
/// mirror). Bad legacy ids fail in `parse_owner_name` before any I/O;
/// registered ids skip that check (the vision id is deliberately short).
pub fn cache_info(
    model_id: &str,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
    precision: Precision,
) -> Result<CacheReport> {
    if precision == Precision::Auto {
        return Err(anyhow!(
            "cache_info never probes the device, so precision='auto' cannot be resolved; \
             pass an explicit precision ('fp32', 'fp16' or 'int8')"
        ));
    }
    let artifact = artifact_for(model_id, precision);
    if lookup_model(model_id).is_none() {
        parse_owner_name(artifact.repo)?;
    }

    let sidecar = artifact.sidecar();
    let mut files: Vec<(&str, bool)> = Vec::with_capacity(3);
    files.push((artifact.file, true));
    if let Some(sc) = sidecar.as_deref() {
        files.push((sc, false));
    }
    files.push(("tokenizer.json", true));
    let mut rep = cache_info_files(artifact.repo, &files, revision, cache_dir)?;
    // The generic core echoes the repo; the registry surface reports the
    // id as passed plus its resolved tier.
    rep.model_id = model_id.to_string();
    rep.precision = Some(precision);
    Ok(rep)
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
        let a = Artifact { repo: "x/y", file: "onnx/model_fp16.onnx", sidecar: None };
        assert_eq!(a.sidecar().as_deref(), Some("onnx/model_fp16.onnx_data"));
        let b = Artifact { repo: "x/y", file: "model.bin", sidecar: None };
        assert_eq!(b.sidecar(), None);
    }

    #[test]
    fn artifact_sidecar_explicit_override_wins() {
        let a = Artifact {
            repo: VISION_NANO_REPO,
            file: "fp32/model.onnx",
            sidecar: Some("fp32/model.onnx.data"),
        };
        assert_eq!(a.sidecar().as_deref(), Some("fp32/model.onnx.data"));
    }

    #[test]
    fn registry_vision_contract() {
        let spec = lookup_model(VISION_NANO_MODEL).expect("vision registered");
        assert_eq!(spec.native_dim, 768);
        assert_eq!(spec.ladder, &[32, 64, 128, 256, 512, 768]);
        assert_eq!(spec.text_partner, Some(NANO_TEXT_MODEL));
        assert!(TEXT_SMALL.text_partner.is_none());
        assert!(TEXT_NANO.text_partner.is_none());
        let fp32 = artifact_for(VISION_NANO_MODEL, Precision::Fp32);
        assert_eq!((fp32.repo, fp32.file), (VISION_NANO_REPO, "fp32/model.onnx"));
        assert_eq!(fp32.sidecar().as_deref(), Some("fp32/model.onnx.data"));
        let fp16 = artifact_for(VISION_NANO_MODEL, Precision::Fp16);
        assert_eq!((fp16.repo, fp16.file), (VISION_NANO_REPO, "fp16/model.onnx"));
        // Unlisted pair (vision has no int8) falls back to fp32.
        let int8 = artifact_for(VISION_NANO_MODEL, Precision::Int8);
        assert_eq!((int8.repo, int8.file), (VISION_NANO_REPO, "fp32/model.onnx"));
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

    // ---- cache_info: offline artifact resolution -----------------------------

    #[test]
    fn cache_info_resolves_registered_artifacts_offline() {
        // Point every lookup at an empty temp cache: resolution alone is
        // under test, nothing is present, and no real cache is touched.
        let empty = std::env::temp_dir().join("embroider-cache-info-empty");
        let _ = std::fs::remove_dir_all(&empty);

        // Default id + fp16 → the mirror repo; precision echoed resolved.
        let rep = cache_info(FP32_TEXT_MODEL, None, Some(empty.clone()), Precision::Fp16)
            .expect("offline resolution must not fail");
        assert_eq!(rep.repo, FP16_TEXT_MODEL);
        assert_eq!(rep.precision, Some(Precision::Fp16));
        assert_eq!(rep.cache_dir, empty);
        assert!(!rep.cached);
        assert_eq!(rep.files.get("onnx/model.onnx"), Some(&None));
        assert!(rep.snapshot_path.is_none());
        assert_eq!(rep.disk_usage_bytes, 0);

        // fp32 (what precision=None documents) stays on the official repo.
        let rep = cache_info(FP32_TEXT_MODEL, None, Some(empty.clone()), Precision::Fp32).unwrap();
        assert_eq!(rep.repo, FP32_TEXT_MODEL);

        // Multi-file repos list their sidecar (vision layout is explicit).
        let rep = cache_info(VISION_NANO_MODEL, None, Some(empty), Precision::Fp32).unwrap();
        assert_eq!(rep.repo, VISION_NANO_REPO);
        assert!(rep.files.contains_key("fp32/model.onnx"));
        assert!(rep.files.contains_key("fp32/model.onnx.data"));
        assert!(rep.files.contains_key("tokenizer.json"));
    }

    #[test]
    fn cache_info_rejects_auto_and_bad_ids_before_io() {
        let err = cache_info(
            FP32_TEXT_MODEL,
            None,
            Some(std::env::temp_dir()),
            Precision::Auto,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("precision='auto'"), "{err}");

        // Bare legacy id: parse_owner_name fires before any I/O.
        let err = cache_info("no-slash", None, None, Precision::Fp32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model_id must be 'owner/name'"), "{err}");
    }

    #[test]
    fn cache_info_fake_cache_hit_partial_and_full_miss() {
        // No network is possible (local_files_only lookups); the unroutable
        // endpoint would fail any accidental request loudly anyway.
        const ENDPOINT: &str = "HF_ENDPOINT";
        let saved = std::env::var(ENDPOINT).ok();
        std::env::set_var(ENDPOINT, "http://127.0.0.1:1");

        let root =
            std::env::temp_dir().join(format!("embroider-cache-fake-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo_folder = "models--jinaai--jina-embeddings-v5-text-nano-retrieval";
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let snap = root.join(repo_folder).join("snapshots").join(sha);
        std::fs::create_dir_all(snap.join("onnx")).unwrap();
        std::fs::create_dir_all(root.join(repo_folder).join("refs")).unwrap();
        std::fs::write(root.join(repo_folder).join("refs").join("main"), format!("{sha}\n"))
            .unwrap();

        // ---- hit: model + tokenizer cached, sidecar absent (like open()) ----
        let model_bytes = vec![0u8; 512];
        let tok_bytes = vec![0u8; 64];
        std::fs::write(snap.join("onnx/model.onnx"), &model_bytes).unwrap();
        std::fs::write(snap.join("tokenizer.json"), &tok_bytes).unwrap();

        let rep = cache_info(NANO_TEXT_MODEL, None, Some(root.clone()), Precision::Fp32).unwrap();
        assert!(rep.cached, "{rep:?}");
        assert_eq!(
            rep.files.get("onnx/model.onnx").unwrap().as_deref(),
            Some(snap.join("onnx/model.onnx").as_path())
        );
        // Missing sidecar is OK — open() only warns (inline weights).
        assert_eq!(rep.files.get("onnx/model.onnx_data"), Some(&None));
        assert!(rep.files.get("tokenizer.json").unwrap().is_some());
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, (model_bytes.len() + tok_bytes.len()) as u64);

        // ---- partial miss: tokenizer gone → open() would fail offline ----
        std::fs::remove_file(snap.join("tokenizer.json")).unwrap();
        let rep = cache_info(NANO_TEXT_MODEL, None, Some(root.clone()), Precision::Fp32).unwrap();
        assert!(!rep.cached);
        assert!(rep.files.get("onnx/model.onnx").unwrap().is_some());
        assert!(rep.files.get("tokenizer.json").unwrap().is_none());
        // Snapshot root still derivable from the file that IS present.
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, model_bytes.len() as u64);

        // ---- pinned revision: a commit hash resolves without refs/ ----
        std::fs::write(snap.join("tokenizer.json"), &tok_bytes).unwrap();
        let rep = cache_info(NANO_TEXT_MODEL, Some(sha), Some(root.clone()), Precision::Fp32)
            .unwrap();
        assert!(rep.cached);

        // ---- full miss: fresh cache dir, raises nothing, everything None ----
        let fresh = root.join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let rep = cache_info(NANO_TEXT_MODEL, None, Some(fresh), Precision::Fp32).unwrap();
        assert!(!rep.cached);
        assert!(rep.files.values().all(|p| p.is_none()));
        assert!(rep.snapshot_path.is_none());
        assert_eq!(rep.disk_usage_bytes, 0);

        match saved {
            Some(v) => std::env::set_var(ENDPOINT, v),
            None => std::env::remove_var(ENDPOINT),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- cache_info_files: the generic (repo, files) surface -------------

    /// Seed a fake hub-cache repo: refs/main + snapshots/<sha>/<files>.
    fn seed_fake_repo(root: &std::path::Path, folder: &str, blobs: &[(&str, &[u8])]) -> PathBuf {
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        let repo_dir = root.join(folder);
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), format!("{sha}\n")).unwrap();
        let snap = repo_dir.join("snapshots").join(sha);
        for (name, data) in blobs {
            let p = snap.join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, data).unwrap();
        }
        snap
    }

    #[test]
    fn cache_info_files_hit_partial_and_full_miss() {
        const ENDPOINT: &str = "HF_ENDPOINT";
        let saved = std::env::var(ENDPOINT).ok();
        std::env::set_var(ENDPOINT, "http://127.0.0.1:1");

        let root =
            std::env::temp_dir().join(format!("embroider-cache-files-fake-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let det = vec![0u8; 256];
        let rec = vec![0u8; 128];
        let snap = seed_fake_repo(
            &root,
            "models--org--converter",
            &[("det.onnx", &det), ("nested/rec.onnx", &rec)],
        );

        // ---- hit: required present, optional missing → still cached ----
        let rep = cache_info_files(
            "org/converter",
            &[("det.onnx", true), ("nested/rec.onnx", true), ("extra.onnx", false)],
            None,
            Some(root.clone()),
        )
        .unwrap();
        assert_eq!(rep.model_id, "org/converter");
        assert_eq!(rep.repo, "org/converter");
        assert_eq!(rep.precision, None); // generic lookup has no tier
        assert!(rep.cached, "{rep:?}");
        assert!(rep.files.get("det.onnx").unwrap().is_some());
        assert_eq!(rep.files.get("extra.onnx"), Some(&None));
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, (det.len() + rec.len()) as u64);

        // ---- partial: a required file gone → not cached, snapshot kept ----
        std::fs::remove_file(snap.join("nested/rec.onnx")).unwrap();
        let rep = cache_info_files(
            "org/converter",
            &[("det.onnx", true), ("nested/rec.onnx", true)],
            None,
            Some(root.clone()),
        )
        .unwrap();
        assert!(!rep.cached);
        assert!(rep.files.get("det.onnx").unwrap().is_some());
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, det.len() as u64);

        // ---- full miss: fresh dir raises nothing, everything None ----
        let fresh = root.join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let rep = cache_info_files(
            "org/converter",
            &[("det.onnx", true)],
            None,
            Some(fresh),
        )
        .unwrap();
        assert!(!rep.cached);
        assert!(rep.files.values().all(|p| p.is_none()));
        assert!(rep.snapshot_path.is_none());
        assert_eq!(rep.disk_usage_bytes, 0);

        match saved {
            Some(v) => std::env::set_var(ENDPOINT, v),
            None => std::env::remove_var(ENDPOINT),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_info_files_pinned_revision() {
        let root =
            std::env::temp_dir().join(format!("embroider-cache-files-pin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        seed_fake_repo(&root, "models--org--pinned", &[("w.onnx", &[1u8; 16])]);
        // Commit hash resolves without consulting refs/.
        let rep = cache_info_files("org/pinned", &[("w.onnx", true)], Some(sha), Some(root.clone()))
            .unwrap();
        assert!(rep.cached);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_info_files_rejects_empty_and_bad_repo_before_io() {
        let err = cache_info_files("org/converter", &[], None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one file"), "{err}");

        let err = cache_info_files("no-slash", &[("w.onnx", true)], None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model_id must be 'owner/name'"), "{err}");
    }

    #[test]
    fn cache_info_delegates_to_the_generic_core() {
        // Structural pin: the registry surface must stay exactly the
        // generic core plus id/tier patching — filenames, cached-ness,
        // snapshot and usage cannot drift apart.
        let root =
            std::env::temp_dir().join(format!("embroider-cache-deleg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let snap = seed_fake_repo(
            &root,
            "models--jinaai--jina-embeddings-v5-text-nano-retrieval",
            &[("onnx/model.onnx", &[9u8; 64]), ("tokenizer.json", &[7u8; 8])],
        );
        let via_registry =
            cache_info(NANO_TEXT_MODEL, None, Some(root.clone()), Precision::Fp32).unwrap();
        let via_generic = cache_info_files(
            NANO_TEXT_MODEL,
            &[
                ("onnx/model.onnx", true),
                ("onnx/model.onnx_data", false),
                ("tokenizer.json", true),
            ],
            None,
            Some(root.clone()),
        )
        .unwrap();
        assert_eq!(via_registry.files, via_generic.files);
        assert_eq!(via_registry.cached, via_generic.cached);
        assert_eq!(via_registry.snapshot_path, via_generic.snapshot_path);
        assert_eq!(via_registry.disk_usage_bytes, via_generic.disk_usage_bytes);
        // ...plus the registry-only patching:
        assert_eq!(via_registry.model_id, NANO_TEXT_MODEL);
        assert_eq!(via_registry.precision, Some(Precision::Fp32));
        assert_eq!(via_generic.model_id, NANO_TEXT_MODEL); // repo echoed
        assert_eq!(via_generic.precision, None);
        assert_eq!(via_registry.snapshot_path.as_deref(), Some(snap.as_path()));
        let _ = std::fs::remove_dir_all(&root);
    }
}
