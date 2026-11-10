//! The EmbeddingGemma 2 vision contract: image embeddings in the text space.
//!
//! Two graphs, one vector space. The standalone SigLIP-style vision tower
//! (`onnx/vision_encoder.onnx`: `pixel_values` + `pixel_position_ids` in,
//! `image_features` `[K,512]` out) encodes the patches; the text tower
//! (the same `EMBEDGEMMA2_MODEL` graph the text slot serves) fuses those
//! features at K `<|image|>` token positions and returns the native
//! `sentence_embedding` head. Image vectors are comparable against
//! `google/embeddinggemma-2` text vectors cross-modally — that is the
//! whole point of the slot, mirroring what `JinaV5Vision` does for nano.
//!
//! Split at the resize (deliberate, same as the Jina slot): the caller
//! Pillow-resizes to [`gemma_vision_target_size`]; embroider takes the
//! resized bytes and does rescale, patchify, position ids, padding, both
//! session runs, then L2 → truncate → re-normalise. Porting the resampler
//! was not worth the parity risk. Arrays that are not their own resize
//! target are rejected — silently accepting them would compute vectors
//! for a different resolution contract.
//!
//! Reference: `transformers` `Gemma4ImageProcessor` (aspect-ratio-preserving
//! resize to multiples of 48, rescale /255, no normalize, patch 16,
//! pool 3, default budget 280 soft tokens) + the ai.google.dev multimodal
//! guide. The default budget is fixed at 280 (70/140/560/1120 are valid
//! per the processor but need their own parity gate before shipping).

use crate::acquire::{
    artifact_for, fetch_tokenizer_file, hf_client, parse_owner_name, resolve_model,
    warn_if_unmeasured_model, EMBEDGEMMA2_MODEL, EMBEDGEMMA2_VISION_MODEL,
};
use crate::error::{oe, Result};
use crate::jina::{build_session, check_truncate_dim_for, l2_truncate};
use crate::policy::{resolve_provider_names, DeviceReq, Precision, SessionPolicy};
use crate::providers::apply_providers_with_arena;
use anyhow::anyhow;
use ndarray::Array2;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Patch edge: 16×16×3 flattened = 768 floats per patch.
pub const GEMMA_PATCH: usize = 16;
/// Spatial pooling kernel: 3×3 patches fuse to one soft token.
pub const GEMMA_POOL: usize = 3;
/// Resize granularity: dimensions round to multiples of 48.
pub const GEMMA_SIDE_MULT: u32 = 48;
/// Default vision budget: 280 soft tokens (upstream default).
pub const GEMMA_SOFT_BUDGET: usize = 280;
/// Padded patch capacity at the default budget: 280 × 9 = 2520.
pub const GEMMA_MAX_PATCHES: usize = GEMMA_SOFT_BUDGET * GEMMA_POOL * GEMMA_POOL;
/// Native width of the fused output (same as the text tower).
pub const GEMMA_VISION_NATIVE_DIM: usize = 768;
/// Token ids framing the image run: `[bos, boi, image × K, eoi, eos]`.
/// `image` resolves from the tokenizer at open (loud check); the rest are
/// config-level ids from the pinned export (no tokenizer entry), pinned
/// by the e2e input-id test.
pub const GEMMA_BOS_ID: i64 = 2;
pub const GEMMA_BOI_ID: i64 = 255999;
pub const GEMMA_IMAGE_ID: i64 = 258880;
pub const GEMMA_EOI_ID: i64 = 258882;
pub const GEMMA_EOS_ID: i64 = 1;

/// Resize target for an `(h, w)` image at `max_soft_tokens` budget: exact
/// port of `get_aspect_ratio_preserving_size` (patch 16, pool 3) — the
/// largest dimensions that fit `budget × 9` patches with both sides
/// multiples of 48. No aspect cap (the processor has none).
pub fn gemma_vision_target_size(h: u32, w: u32, max_soft_tokens: usize) -> Result<(u32, u32)> {
    if h == 0 || w == 0 {
        return Err(anyhow!("image dimensions must be nonzero, got {h}x{w}"));
    }
    if max_soft_tokens == 0 {
        return Err(anyhow!("soft-token budget must be nonzero"));
    }
    let max_patches = max_soft_tokens * GEMMA_POOL * GEMMA_POOL;
    let target_px = max_patches * GEMMA_PATCH * GEMMA_PATCH;
    let factor = (target_px as f64 / (h as u64 * w as u64) as f64).sqrt();
    let sm = GEMMA_SIDE_MULT;
    let th = (factor * h as f64 / sm as f64).floor() as u32 * sm;
    let tw = (factor * w as f64 / sm as f64).floor() as u32 * sm;
    if th == 0 || tw == 0 {
        return Err(anyhow!(
            "budget {max_soft_tokens} collapses {h}x{w} to zero — image too small for the budget"
        ));
    }
    if th as u64 * tw as u64 > target_px as u64 {
        return Err(anyhow!(
            "resizing [{h}x{w}] to [{th}x{tw}] exceeds {max_patches} patches"
        ));
    }
    Ok((th, tw))
}

/// Default-budget target size (what the caller resizes to).
pub fn gemma_default_target_size(h: u32, w: u32) -> Result<(u32, u32)> {
    gemma_vision_target_size(h, w, GEMMA_SOFT_BUDGET)
}

/// Reject arrays that are not their own default-budget resize target.
pub(crate) fn check_gemma_target_size(h: usize, w: usize) -> Result<()> {
    let h32: u32 = h.try_into().map_err(|_| anyhow!("height {h} out of range"))?;
    let w32: u32 = w.try_into().map_err(|_| anyhow!("width {w} out of range"))?;
    let (rh, rw) = gemma_default_target_size(h32, w32)?;
    if (rh as usize, rw as usize) != (h, w) {
        return Err(anyhow!(
            "image is {h}x{w}, not its Gemma resize target {rh}x{rw} — \
             resize caller-side (Pillow bicubic) before encode_image"
        ));
    }
    Ok(())
}

/// Patchify row-major `[h, w, 3]` uint8 (already at target size):
/// rescale /255, flatten each 16×16 patch channel-last
/// (`(kh, kw, c)` — the `convert_image_to_patches` order), pad with zeros
/// to `GEMMA_MAX_PATCHES`. Returns `(patches, positions, real_count)` with
/// positions as `[col, row]` pairs (meshgrid `xy` order).
pub(crate) fn gemma_patches(rgb: &[u8], h: usize, w: usize) -> Result<(Vec<f32>, Vec<[i64; 2]>, usize)> {
    if rgb.len() != h * w * 3 {
        return Err(anyhow!(
            "expected {} RGB bytes for {h}x{w}, got {}",
            h * w * 3,
            rgb.len()
        ));
    }
    let gh = h / GEMMA_PATCH;
    let gw = w / GEMMA_PATCH;
    let real = gh * gw;
    if real > GEMMA_MAX_PATCHES {
        return Err(anyhow!(
            "{h}x{w} patchifies to {real} patches, over the {GEMMA_MAX_PATCHES} budget — resize first"
        ));
    }
    let mut patches = vec![0.0f32; GEMMA_MAX_PATCHES * GEMMA_PATCH * GEMMA_PATCH * 3];
    // Padding positions are [-1, -1] (the processor's pad value — a real
    // [0, 0] would attend as the top-left patch and shift the tower).
    let mut positions = vec![[-1i64; 2]; GEMMA_MAX_PATCHES];
    for phy in 0..gh {
        for pxw in 0..gw {
            let p = phy * gw + pxw;
            positions[p] = [pxw as i64, phy as i64];
            for kh in 0..GEMMA_PATCH {
                for kw in 0..GEMMA_PATCH {
                    let y = phy * GEMMA_PATCH + kh;
                    let x = pxw * GEMMA_PATCH + kw;
                    for c in 0..3 {
                        patches[p * 768 + kh * 48 + kw * 3 + c] =
                            rgb[(y * w + x) * 3 + c] as f32 / 255.0;
                    }
                }
            }
        }
    }
    Ok((patches, positions, real))
}

/// Build + contract-check the vision tower session: `pixel_values` +
/// `pixel_position_ids` in, `image_features` out. Vision-slot policy
/// (ORT defaults, never the text tuning) with the CUDA arena strategy
/// from the Jina slot.
fn build_gemma_vision_session(
    onnx_path: &Path,
    device: DeviceReq,
    gpu_mem_limit: Option<u64>,
) -> Result<(RwLock<Session>, bool)> {
    let (provider_names, used_cuda) = resolve_provider_names(device);
    let mut builder = oe(ort::session::Session::builder())?;
    builder = SessionPolicy::ort_defaults().apply(builder)?;
    if used_cuda {
        builder = oe(builder
            .with_config_entry("arena_extend_strategy", "kSameAsRequested"))?;
        if let Some(limit) = gpu_mem_limit {
            builder =
                oe(builder.with_config_entry("gpu_mem_limit", limit.to_string()))?;
        }
    }
    builder = apply_providers_with_arena(builder, &provider_names, used_cuda);
    let session = oe(builder.commit_from_file(onnx_path))
        .map_err(|e| anyhow!("loading {}: {e:#}", onnx_path.display()))?;
    for need in ["pixel_values", "pixel_position_ids"] {
        if !session.inputs().iter().any(|o| o.name() == need) {
            let have: Vec<&str> =
                session.inputs().iter().map(|o| o.name()).collect();
            return Err(anyhow!(
                "Gemma vision export lacks required input '{need}' (has {have:?})"
            ));
        }
    }
    if !session.outputs().iter().any(|o| o.name() == "image_features") {
        let have: Vec<&str> =
            session.outputs().iter().map(|o| o.name()).collect();
        return Err(anyhow!(
            "Gemma vision export lacks the 'image_features' output (has {have:?})"
        ));
    }
    Ok((RwLock::new(session), used_cuda))
}

/// Text-tower contract for the fusion pass: the 5-input multimodal graph
/// with the native `sentence_embedding` head (same graph the text slot
/// serves at `EMBEDGEMMA2_MODEL`).
fn check_gemma_text_session(session: &Session) -> Result<()> {
    for need in [
        "input_ids",
        "attention_mask",
        "image_features",
        "video_features",
        "audio_features",
    ] {
        if !session.inputs().iter().any(|o| o.name() == need) {
            let have: Vec<&str> =
                session.inputs().iter().map(|o| o.name()).collect();
            return Err(anyhow!(
                "text tower lacks required input '{need}' (has {have:?}) — not the Gemma multimodal graph"
            ));
        }
    }
    if !session.outputs().iter().any(|o| o.name() == "sentence_embedding") {
        let have: Vec<&str> =
            session.outputs().iter().map(|o| o.name()).collect();
        return Err(anyhow!(
            "text tower lacks the 'sentence_embedding' output (has {have:?})"
        ));
    }
    Ok(())
}

pub struct GemmaVision {
    vision: RwLock<Session>,
    text: RwLock<Session>,
    dim: usize,
    model_id: String,
    used_cuda: bool,
    precision: Precision,
}

impl GemmaVision {
    /// Open the vision slot: vision tower at `(model_id, precision)` plus
    /// the text tower through `text_partner` at the same tier. Registry
    /// ids only — the two-session layout has no legacy single-file form,
    /// so unknown ids fail loudly instead of resolving half a slot.
    /// `truncate_dim` follows the text MRL ladder (default 512 in Python).
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
        truncate_dim: usize,
        device: DeviceReq,
        precision: Precision,
        gpu_mem_limit: Option<u64>,
    ) -> Result<Self> {
        let resolved = resolve_model(model_id);
        let spec = resolved.as_ref().ok_or_else(|| {
            anyhow!(
                "unknown Gemma vision id '{model_id}': open() needs the registry entry \
                 (see EMBEDGEMMA2_VISION); explicit files go through open_files()"
            )
        })?;
        check_truncate_dim_for(truncate_dim, spec.native_dim, &spec.ladder)?;
        let partner_id = spec.text_partner.clone().ok_or_else(|| {
            anyhow!(
                "registry entry '{model_id}' has no text_partner — cannot fuse without the text tower"
            )
        })?;

        // Resolve the device BEFORE acquisition: precision follows the
        // landed device. Explicit fp16-on-CPU is an error here — the tower
        // activations are fp16-hostile upstream, and a hung batch is worse
        // than a refusal (same fail-fast rule as the Jina vision slot).
        let (_, used_cuda) = resolve_provider_names(device);
        let precision = match (precision, used_cuda) {
            (Precision::Fp16, false) => {
                return Err(anyhow!(
                    "explicit fp16 on CPU for the Gemma vision slot: tower activations are \
                     fp16-hostile (upstream note) — pass precision='fp32' (or 'auto') for CPU"
                ));
            }
            (Precision::Auto, true) => Precision::Fp16,
            (Precision::Auto, false) => Precision::Fp32,
            (p, _) => p,
        };
        warn_if_unmeasured_model(spec, model_id, precision);

        // ---- acquisition: vision tower at the tier, text tower via partner ----
        let v_artifact = artifact_for(model_id, precision);
        let t_artifact = artifact_for(&partner_id, precision);
        let precision = crate::acquire::model_artifact_precision(model_id, precision);
        let v_rev = revision.map(str::to_string).or(v_artifact.revision.clone());
        let t_rev = revision.map(str::to_string).or(t_artifact.revision.clone());

        let (v_owner, v_name) = parse_owner_name(&v_artifact.repo)?;
        let (t_owner, t_name) = parse_owner_name(&t_artifact.repo)?;
        let client = hf_client(cache_dir.clone())?;
        let v_repo = client.model(&v_owner, &v_name);
        let t_repo = client.model(&t_owner, &t_name);
        let v_path = v_repo
            .download_file()
            .filename(v_artifact.file.clone())
            .maybe_revision(v_rev.clone())
            .send()
            .map_err(|e| anyhow!("{} missing for '{}': {e}", v_artifact.file, v_artifact.repo))?;
        if let Some(sidecar) = v_artifact.sidecar() {
            v_repo
                .download_file()
                .filename(sidecar.clone())
                .maybe_revision(v_rev.clone())
                .send()
                .map_err(|e| anyhow!("vision sidecar {sidecar} missing: {e}"))?;
        }
        let t_path = t_repo
            .download_file()
            .filename(t_artifact.file.clone())
            .maybe_revision(t_rev.clone())
            .send()
            .map_err(|e| anyhow!("{} missing for '{}': {e}", t_artifact.file, t_artifact.repo))?;
        if let Some(sidecar) = t_artifact.sidecar() {
            t_repo
                .download_file()
                .filename(sidecar.clone())
                .maybe_revision(t_rev.clone())
                .send()
                .map_err(|e| anyhow!("text-tower sidecar {sidecar} missing: {e}"))?;
        }
        // Tokenizer: presence check + `<|image|>` id pin (the one id with a
        // tokenizer entry; the rest are config-level, pinned by the e2e).
        let tok_path = fetch_tokenizer_file(&v_artifact.repo, v_rev.as_deref(), cache_dir)?;
        let tok = crate::jina::load_tokenizer(&tok_path, None)?;
        match tok.token_to_id("<|image|>") {
            Some(258880) => {}
            other => {
                return Err(anyhow!(
                    "tokenizer <|image|> id is {other:?}, expected 258880 — wrong tokenizer for the Gemma slot"
                ));
            }
        }

        Self::from_files(&v_path, &t_path, truncate_dim, device, gpu_mem_limit, model_id, precision)
    }

    /// Load from explicit local files — no network access. Sidecars must
    /// sit next to their ONNX files (ORT resolves them relative to the
    /// model path). Precision selection does not apply (the files are what
    /// they are); the reported precision reads `fp32`, like the Jina slot.
    pub fn open_files(
        vision_onnx_path: &Path,
        text_onnx_path: &Path,
        tokenizer_path: &Path,
        truncate_dim: usize,
        device: DeviceReq,
        gpu_mem_limit: Option<u64>,
    ) -> Result<Self> {
        let ladder = resolve_model(EMBEDGEMMA2_VISION_MODEL)
            .map(|s| s.ladder)
            .unwrap_or(vec![128, 256, 512, 768]);
        check_truncate_dim_for(truncate_dim, GEMMA_VISION_NATIVE_DIM, &ladder)?;
        for (label, p) in [("vision", vision_onnx_path), ("text tower", text_onnx_path)] {
            if !p.is_file() {
                return Err(anyhow!("{label} onnx model not found: {}", p.display()));
            }
        }
        if !tokenizer_path.is_file() {
            return Err(anyhow!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            ));
        }
        let tok = crate::jina::load_tokenizer(tokenizer_path, None)?;
        match tok.token_to_id("<|image|>") {
            Some(258880) => {}
            other => {
                return Err(anyhow!(
                    "tokenizer <|image|> id is {other:?}, expected 258880 — wrong tokenizer for the Gemma slot"
                ));
            }
        }
        Self::from_files(
            vision_onnx_path,
            text_onnx_path,
            truncate_dim,
            device,
            gpu_mem_limit,
            &vision_onnx_path.display().to_string(),
            Precision::Fp32,
        )
    }

    fn from_files(
        vision_onnx_path: &Path,
        text_onnx_path: &Path,
        truncate_dim: usize,
        device: DeviceReq,
        gpu_mem_limit: Option<u64>,
        model_id: &str,
        precision: Precision,
    ) -> Result<Self> {
        let (vision, used_cuda) =
            build_gemma_vision_session(vision_onnx_path, device, gpu_mem_limit)?;
        let text_loaded = build_session(
            text_onnx_path,
            device,
            &SessionPolicy::text_embed(),
            false,
        )?;
        check_gemma_text_session(&text_loaded.session)?;
        let text = text_loaded.session;
        Ok(Self {
            vision,
            text: RwLock::new(text),
            dim: truncate_dim,
            model_id: model_id.to_string(),
            used_cuda,
            precision,
        })
    }

    /// Embed one already-resized RGB image: row-major `[h, w, 3]` uint8
    /// where `(h, w)` is its own [`gemma_vision_target_size`] at the
    /// default budget (Pillow bicubic resize on the caller side). One
    /// image per call — shapes vary with aspect ratio, so batching buys
    /// nothing (matches both the text path and the Jina vision slot).
    pub fn encode_image(&self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
        check_gemma_target_size(h, w)?;
        let (patches, positions, real) = gemma_patches(rgb, h, w)?;
        let max_patches = GEMMA_MAX_PATCHES;

        let pv_arr = Array2::from_shape_vec(
            (max_patches, GEMMA_PATCH * GEMMA_PATCH * 3),
            patches,
        )
        .map_err(|e| anyhow!("pixel_values shape: {e}"))?
        .insert_axis(ndarray::Axis(0))
        .to_owned();
        let pos_arr = Array2::from_shape_vec(
            (max_patches, 2),
            positions.iter().flat_map(|p| p.iter().copied()).collect(),
        )
        .map_err(|e| anyhow!("pixel_position_ids shape: {e}"))?
        .insert_axis(ndarray::Axis(0))
        .to_owned();

        // Tower run: dynamic rows out (K pooled soft tokens).
        let features: Array2<f32> = {
            let mut guard =
                self.vision.write().map_err(|_| anyhow!("vision session lock poisoned"))?;
            let pv_t = oe(ort::value::TensorRef::from_array_view(&pv_arr))?;
            let pos_t = oe(ort::value::TensorRef::from_array_view(&pos_arr))?;
            let outputs = oe(guard.run(ort::inputs! {
                "pixel_values" => pv_t,
                "pixel_position_ids" => pos_t
            }))?;
            oe(outputs["image_features"].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow!("expected [K,512] image_features: {e}"))?
        };
        let k = features.shape()[0];
        if features.shape()[1] != 512 {
            return Err(anyhow!(
                "image_features width is {}, expected 512",
                features.shape()[1]
            ));
        }
        // K must be exactly the pooled real-patch count: the tower pools
        // 3×3 over real patches only, so anything else means the export or
        // the host math drifted apart.
        if k == 0 || k > GEMMA_SOFT_BUDGET || k * GEMMA_POOL * GEMMA_POOL != real {
            return Err(anyhow!(
                "vision tower returned {k} soft tokens for {real} real patches \
                 (expected {}/9) — export/host mismatch",
                real
            ));
        }

        // Fusion ids: [bos, boi, image × K, eoi, eos], mask all ones.
        let mut ids = Vec::with_capacity(k + 4);
        ids.push(GEMMA_BOS_ID);
        ids.push(GEMMA_BOI_ID);
        ids.extend(std::iter::repeat_n(GEMMA_IMAGE_ID, k));
        ids.push(GEMMA_EOI_ID);
        ids.push(GEMMA_EOS_ID);
        let seq = ids.len();
        let ids_arr = Array2::from_shape_vec((1, seq), ids)
            .map_err(|e| anyhow!("input_ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec((1, seq), vec![1i64; seq])
            .map_err(|e| anyhow!("attention_mask shape: {e}"))?;
        let empty = Array2::<f32>::zeros((0, 512));

        let embedding = {
            let mut guard =
                self.text.write().map_err(|_| anyhow!("text session lock poisoned"))?;
            let ids_t = oe(ort::value::TensorRef::from_array_view(&ids_arr))?;
            let mask_t = oe(ort::value::TensorRef::from_array_view(&mask_arr))?;
            let img_t = oe(ort::value::TensorRef::from_array_view(&features))?;
            let empty_v = oe(ort::value::TensorRef::from_array_view(&empty))?;
            let empty_a = oe(ort::value::TensorRef::from_array_view(&empty))?;
            let outputs = oe(guard.run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "image_features" => img_t,
                "video_features" => empty_v,
                "audio_features" => empty_a
            }))?;
            oe(outputs["sentence_embedding"].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow!("expected [1,768] sentence_embedding: {e}"))?
        };
        let pooled = embedding.slice(ndarray::s![0, ..]).to_owned();
        Ok(l2_truncate(&pooled, self.dim))
    }

    /// Configured output dimension (post-Matryoshka, same ladder as text).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Model id — or the vision ONNX path when opened via `open_files`.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Whether the sessions actually run on CUDA.
    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    /// Weight precision the slot was opened with (`auto` already resolved
    /// against the landed device). Explicit files report `fp32`.
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_target_sizes_match_processor() {
        // Values from transformers Gemma4ImageProcessor (patch 16, pool 3,
        // budget 280): 240² upscales to 768²; 640×480 and 4000×3000 both
        // land 912×672; tall 100×2000 lands 144×3552.
        assert_eq!(gemma_vision_target_size(240, 240, 280).unwrap(), (768, 768));
        assert_eq!(gemma_vision_target_size(640, 480, 280).unwrap(), (912, 672));
        assert_eq!(gemma_vision_target_size(480, 640, 280).unwrap(), (672, 912));
        assert_eq!(gemma_vision_target_size(100, 2000, 280).unwrap(), (144, 3552));
        assert_eq!(gemma_vision_target_size(4000, 3000, 280).unwrap(), (912, 672));
        assert_eq!(gemma_vision_target_size(32, 32, 280).unwrap(), (768, 768));
        assert!(gemma_vision_target_size(0, 240, 280).is_err());
        assert!(gemma_vision_target_size(240, 240, 0).is_err());
    }

    #[test]
    fn gemma_patchify_order_and_positions() {
        // 48×48 → 3×3 patches; patch (0,0) holds pixels (0..16, 0..16).
        let h = 48usize;
        let w = 48usize;
        let mut rgb = vec![0u8; h * w * 3];
        rgb[(5 * w + 7) * 3] = 255; // R at (y=5, x=7)
        rgb[(5 * w + 7) * 3 + 1] = 128;
        let (patches, positions, real) = gemma_patches(&rgb, h, w).unwrap();
        assert_eq!(real, 9);
        assert_eq!(positions[0], [0, 0]);
        assert_eq!(positions[2], [2, 0]);
        assert_eq!(positions[3], [0, 1]);
        // (kh=5, kw=7, c=0): 5*48 + 7*3 + 0 = 261 → 1.0; c=1 → 128/255.
        assert!((patches[261] - 1.0).abs() < 1e-6);
        assert!((patches[262] - 128.0 / 255.0).abs() < 1e-6);
        assert_eq!(patches.len(), GEMMA_MAX_PATCHES * 768);
        // Padding rows are zeros.
        assert!(patches[9 * 768..].iter().all(|&v| v == 0.0));
        assert!(positions[9..].iter().all(|p| *p == [-1, -1]));
    }

    #[test]
    fn gemma_fusion_id_layout() {
        // Mirrors the encode_image id construction for K=256 (240² image):
        // [bos, boi, image × 256, eoi, eos] = 260 tokens.
        let k = 256usize;
        let mut ids = vec![GEMMA_BOS_ID, GEMMA_BOI_ID];
        ids.extend(std::iter::repeat_n(GEMMA_IMAGE_ID, k));
        ids.extend([GEMMA_EOI_ID, GEMMA_EOS_ID]);
        assert_eq!(ids.len(), 260);
        assert_eq!(&ids[..3], &[2, 255999, 258880]);
        assert_eq!(&ids[258..], &[258882, 1]);
    }

    #[test]
    fn gemma_open_rejects_unknown_ids() {
        let r = GemmaVision::open(
            "someone/elsewhere",
            None,
            None,
            512,
            DeviceReq::Cpu,
            Precision::Fp32,
            None,
        );
        assert!(r.is_err());
    }
}
