//! The Jina v5 vision contract: dynamic-grid image embeddings (Phase 6).
//!
//! One graph (vision tower + merger + EuroBERT text tower +
//! last-token pool + L2 norm) with host-computed grid inputs, so any
//! aspect ratio and resolution works. Exact port of the spike's `grid.py`
//! (numpy), itself verified bit-identical to transformers on 43,431 sizes
//! and all 3,319 reachable grids — the fixture suite in
//! `fixtures/vision/` pins this port at bitwise equality.
//!
//! Split at the resize (deliberate): okfgraph decodes + converts to RGB +
//! Pillow-bicubic-resizes to [`vision_target_size`]; embroider takes the
//! resized bytes and does normalisation, patchify, host tensors, prompt
//! ids, then runs the graph. Porting PIL's resampler was not worth the
//! parity risk. Arrays that are not a valid target size are rejected —
//! silently accepting them would compute vectors for a different contract.
//!
//! Image vectors share the text-nano vector space (`text_partner` in the
//! registry): only comparable against text-nano text vectors.

use crate::acquire::{
    artifact_for, fetch_tokenizer_file, hf_client, lookup_model, parse_owner_name,
    VISION_NANO_MODEL,
};
use crate::error::{oe, Result};
use crate::jina::{check_truncate_dim_for, l2_truncate, load_tokenizer};
use crate::policy::{resolve_provider_names, DeviceReq, Precision, SessionPolicy};
use crate::providers::apply_providers_with_arena;
use anyhow::anyhow;
use ndarray::Array2;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Resize factor: patch 16 × spatial-merge 2.
pub const VISION_FACTOR: u32 = 32;
/// Minimum pixels (512²): smaller images are upscaled into the range.
pub const VISION_MIN_PIXELS: u64 = 262144;
/// Maximum pixels (1,280 merged tokens): larger images are downscaled.
pub const VISION_MAX_PIXELS: u64 = 1310720;
/// Absolute aspect-ratio limit; beyond it the processor refuses.
pub const VISION_MAX_ASPECT: f64 = 200.0;
/// Side of the learned square position table (48×48 = 2304 embeddings).
pub const VISION_POS_SIDE: i64 = 48;
/// Native width of the vision output (same as text-nano).
pub const VISION_NATIVE_DIM: usize = 768;
/// Prompt framing around the `<image>` run: `<|im_start|>user\n` … `<|im_end|>\n`
/// plus special tokens tokenize to 15 non-image tokens, so
/// `seq = image_tokens + 15` (max 1280 + 15).
pub const VISION_PROMPT_OVERHEAD: usize = 15;

/// Banker's rounding (round-half-to-even) for non-negative values — the
/// exact semantics of Python's `round()`, which `smart_resize` relies on.
/// Rust's `f64::round` rounds half away from zero and would mistarget
/// every dimension ≡ 16 (mod 32) (e.g. 16 → 32 instead of 0). Quotients
/// here are exact binary fractions (integer / 32), so the `.5` test is
/// exact — no epsilon.
fn round_half_even(x: f64) -> f64 {
    debug_assert!(x >= 0.0);
    let f = x.floor();
    let d = x - f;
    if d < 0.5 {
        f
    } else if d > 0.5 {
        f + 1.0
    } else if (f as u64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// Resize target for an `(h, w)` image: exact port of transformers'
/// qwen2_vl `smart_resize` (factor 32, pixel range, aspect preserved).
/// Errors on aspect ratios above 200:1, mirroring the processor.
pub fn vision_target_size(h: u32, w: u32) -> Result<(u32, u32)> {
    if h == 0 || w == 0 {
        return Err(anyhow!("image dimensions must be nonzero, got {h}x{w}"));
    }
    if (h.max(w) as f64) / (h.min(w) as f64) > VISION_MAX_ASPECT {
        return Err(anyhow!(
            "absolute aspect ratio must be smaller than 200, got {}x{}",
            h,
            w
        ));
    }
    let f = VISION_FACTOR as f64;
    let mut hb = round_half_even(h as f64 / f) as u32 * VISION_FACTOR;
    let mut wb = round_half_even(w as f64 / f) as u32 * VISION_FACTOR;
    let area = hb as u64 * wb as u64;
    if area > VISION_MAX_PIXELS {
        // Same op order as the reference: (h*w)/max → sqrt → h/beta/factor.
        let beta = ((h as f64 * w as f64) / VISION_MAX_PIXELS as f64).sqrt();
        hb = ((h as f64 / beta / f).floor() as u32).max(1) * VISION_FACTOR;
        wb = ((w as f64 / beta / f).floor() as u32).max(1) * VISION_FACTOR;
    } else if area < VISION_MIN_PIXELS {
        let beta = (VISION_MIN_PIXELS as f64 / (h as f64 * w as f64)).sqrt();
        hb = (h as f64 * beta / f).ceil() as u32 * VISION_FACTOR;
        wb = (w as f64 * beta / f).ceil() as u32 * VISION_FACTOR;
    }
    Ok((hb, wb))
}

/// Reject arrays that are not their own resize target: the caller (Pillow
/// side) must resize first. A quiet accept would embed a different
/// resolution contract under the same model id.
pub(crate) fn check_target_size(h: usize, w: usize) -> Result<(u32, u32)> {
    let h32: u32 = h.try_into().map_err(|_| anyhow!("height {h} out of range"))?;
    let w32: u32 = w.try_into().map_err(|_| anyhow!("width {w} out of range"))?;
    let (rh, rw) = vision_target_size(h32, w32)?;
    if (rh as usize, rw as usize) != (h, w) {
        return Err(anyhow!(
            "image is {h}x{w} but the vision contract needs {rh}x{rw} \
             (Pillow bicubic resize first); refusing rather than embedding \
             a different resolution contract"
        ));
    }
    Ok((rh, rw))
}

/// (row, col) of every patch in spatial-merge-block order — the single
/// traversal behind both `vision_pos_ids` and the interpolation taps.
/// `i` is the patch-linear index: within-block offsets cycle fastest,
/// then block column, then block row.
fn merge_order_rows_cols(patches: usize, blocks_w: usize) -> (Vec<i64>, Vec<i64>) {
    const MERGE: usize = 2;
    let mut rows = Vec::with_capacity(patches);
    let mut cols = Vec::with_capacity(patches);
    for i in 0..patches {
        let in_col = (i % MERGE) as i64;
        let in_row = ((i / MERGE) % MERGE) as i64;
        let block_col = ((i / (MERGE * MERGE)) % blocks_w) as i64;
        let block_row = (i / (MERGE * MERGE * blocks_w)) as i64;
        rows.push(block_row * MERGE as i64 + in_row);
        cols.push(block_col * MERGE as i64 + in_col);
    }
    (rows, cols)
}

/// Rotary (row, col) per patch, merge-block order — port of transformers'
/// `get_vision_position_ids` (spatial_merge_size=2).
pub(crate) fn vision_pos_ids(gh: usize, gw: usize) -> Vec<[i64; 2]> {
    let (rows, cols) = merge_order_rows_cols(gh * gw, gw / 2);
    rows.into_iter().zip(cols).map(|(r, c)| [r, c]).collect()
}

/// Bilinear taps of one axis into the position table — float32 throughout,
/// same op order as torch (`index * (side-1) / clamp(size-1, 1)`).
fn axis_taps(index: &[i64], size: usize) -> (Vec<[i64; 2]>, Vec<[f32; 2]>) {
    let denom = (size.max(2) - 1) as f32;
    let mut taps = Vec::with_capacity(index.len());
    let mut weights = Vec::with_capacity(index.len());
    for &ix in index {
        let src = ix as f32 * (VISION_POS_SIDE as f32 - 1.0) / denom;
        let floor = src.floor();
        let f = floor as i64;
        let t0 = f.clamp(0, VISION_POS_SIDE - 1);
        let t1 = (f + 1).clamp(0, VISION_POS_SIDE - 1);
        // `|src - floor - offset|` with the offset as f32, then 1-dist
        // clipped at 0 — the reference's `clip(1 - dist, 0, None)`.
        let w1 = (1.0 - (src - floor - 1.0).abs()).max(0.0);
        let w0 = (1.0 - (src - floor).abs()).max(0.0);
        taps.push([t0, t1]);
        weights.push([w0, w1]);
    }
    (taps, weights)
}

/// Bilinear taps into the learned 48×48 position table — port of
/// transformers' `get_vision_interpolation_indices_and_weights`
/// (bilinear, align_corners, merge 2). Row-major pairing: each patch maps
/// to `[(h0,w0), (h0,w1), (h1,w0), (h1,w1)]`.
pub(crate) fn interp_taps(
    gh: usize,
    gw: usize,
) -> (Vec<[i64; 4]>, Vec<[f32; 4]>) {
    let (rows, cols) = merge_order_rows_cols(gh * gw, gw / 2);
    let (h_taps, h_w) = axis_taps(&rows, gh);
    let (w_taps, w_w) = axis_taps(&cols, gw);
    let side = VISION_POS_SIDE;
    let mut idx = Vec::with_capacity(rows.len());
    let mut wts = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let [h0, h1] = h_taps[i];
        let [w0, w1] = w_taps[i];
        let [a0, a1] = h_w[i];
        let [b0, b1] = w_w[i];
        idx.push([h0 * side + w0, h0 * side + w1, h1 * side + w0, h1 * side + w1]);
        wts.push([a0 * b0, a0 * b1, a1 * b0, a1 * b1]);
    }
    (idx, wts)
}

/// Normalise + Qwen2-VL patchify an already-resized RGB buffer — port of
/// the spike's `pixel_values` minus the Pillow resize (that half lives
/// with the caller). `rgb` is row-major `[h, w, 3]` uint8; `h`/`w` must
/// be multiples of 16. Output `[gh*gw, 1536]` float32, merge-block order.
///
/// Per-element op order mirrors numpy (`(u/255 - 0.5)/0.5`, 0.5 exact in
/// f32), so output bits equal the reference bit for bit.
pub(crate) fn pixel_values(rgb: &[u8], h: usize, w: usize) -> Result<Array2<f32>> {
    if h % 16 != 0 || w % 16 != 0 {
        return Err(anyhow!(
            "resized dimensions must be multiples of 16 (patch size), got {h}x{w}"
        ));
    }
    if rgb.len() != h * w * 3 {
        return Err(anyhow!(
            "rgb buffer is {} bytes, expected {} for {h}x{w}x3",
            rgb.len(),
            h * w * 3
        ));
    }
    // Transpose chain (batch, bgh, bgw, m_row, m_col, C, T, y, x):
    // innermost varies fastest → offset [p, ((c*2+t)*16+y)*16+x].
    // Temporal dim is a duplicated frame (still images).
    let (rows, cols) = merge_order_rows_cols((h / 16) * (w / 16), w / 16 / 2);
    let mut out = Vec::with_capacity(rows.len() * 1536);
    for p in 0..rows.len() {
        let pr = rows[p] as usize;
        let pc = cols[p] as usize;
        for c in 0..3usize {
            for t in 0..2usize {
                let _ = t; // still image: both temporal slots read one frame
                for y in 0..16usize {
                    for x in 0..16usize {
                        let px = ((pr * 16 + y) * w + (pc * 16 + x)) * 3 + c;
                        let v = (rgb[px] as f32 / 255.0 - 0.5) / 0.5;
                        out.push(v);
                    }
                }
            }
        }
    }
    Array2::from_shape_vec((rows.len(), 1536), out)
        .map_err(|e| anyhow!("pixel_values shape: {e}"))
}

/// Load a vision tokenizer: truncation permanently off. The shipped
/// `tokenizer.json` carries a 512-truncation in-file (the text path
/// overrides it explicitly); an image prompt runs up to 1280 + 15 tokens
/// and must never be cut — the spike's `grid.py` calls `no_truncation()`
/// for the same reason. `encode_image` re-checks the outcome via the
/// `seq == image_tokens + 15` guard.
fn load_vision_tokenizer(tok_path: &Path) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer = load_tokenizer(tok_path, None)?;
    tokenizer
        .with_truncation(None)
        .map_err(|e| anyhow!("truncation disable failed: {e}"))?;
    Ok(tokenizer)
}

/// Prompt token ids: `<|im_start|>user\n` + `<image>`×k + `<|im_end|>\n`
/// with special tokens (see [`load_vision_tokenizer`] for why truncation
/// is off).
pub(crate) fn prompt_ids(
    tokenizer: &tokenizers::Tokenizer,
    n_image_tokens: usize,
) -> Result<Vec<i64>> {
    let text = format!(
        "<|im_start|>user\n{}<|im_end|>\n",
        "<image>".repeat(n_image_tokens)
    );
    let enc = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("prompt tokenization failed: {e}"))?;
    Ok(enc.get_ids().iter().map(|&x| x as i64).collect())
}

/// Resolve the effective precision: `Auto` follows the landed device
/// (CUDA → fp16, CPU → fp32). Explicit fp16 on a CPU-resolved session is
/// a hard error — unlike the text path, where it merely runs slow, the
/// vision fp16 graph *stalls* on CPU (no fast fp16 kernels; observed
/// hang, not slowness). Fail fast instead of hanging a batch import.
pub(crate) fn resolve_vision_precision(
    precision: Precision,
    used_cuda: bool,
) -> Result<Precision> {
    let resolved = precision.resolve(used_cuda);
    if resolved == Precision::Fp16 && !used_cuda {
        return Err(anyhow!(
            "vision fp16 weights on CPU stall (no fast fp16 kernels) — \
             pass precision='fp32' (or 'auto') for CPU sessions"
        ));
    }
    Ok(resolved)
}

pub struct JinaV5Vision {
    session: RwLock<Session>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
    model_id: String,
    used_cuda: bool,
    precision: Precision,
}

/// Build + contract-check a vision session from an ONNX file on disk.
/// Vision-slot policy: ORT defaults (NOT the text `text_embed()` tuning —
/// bobine pins that split with a test), arena on, and on CUDA
/// `arena_extend_strategy=kSameAsRequested` (varying shapes grew the
/// default arena +8.4 GB fp32 / +3.4 GB fp16 in the spike). `gpu_mem_limit`
/// caps CUDA arena bytes when set. CPU sessions run arena-off like the
/// text path (RSS over speed for batch imports).
fn build_vision_session(
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
    // Arena rides on for CUDA (default allocator), off for CPU.
    builder = apply_providers_with_arena(builder, &provider_names, used_cuda);
    let session = oe(builder.commit_from_file(onnx_path))
        .map_err(|e| anyhow!("loading {}: {e:#}", onnx_path.display()))?;

    // Exact vision contract: the six host+prompt inputs and the single
    // `sentence_embedding` output. Anything else is not the dynamic-grid
    // graph (e.g. a text export passed by mistake) — fail naming it.
    for need in [
        "input_ids",
        "attention_mask",
        "pixel_values",
        "vision_pos_ids",
        "interp_indices",
        "interp_weights",
    ] {
        if !session.inputs().iter().any(|o| o.name() == need) {
            let have: Vec<&str> =
                session.inputs().iter().map(|o| o.name()).collect();
            return Err(anyhow!(
                "vision ONNX export lacks required input '{need}' (has {have:?}) — \
                 not the dynamic-grid graph"
            ));
        }
    }
    if !session.outputs().iter().any(|o| o.name() == "sentence_embedding") {
        let have: Vec<&str> =
            session.outputs().iter().map(|o| o.name()).collect();
        return Err(anyhow!(
            "vision ONNX export lacks the 'sentence_embedding' output (has {have:?})"
        ));
    }
    Ok((RwLock::new(session), used_cuda))
}

impl JinaV5Vision {
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
        // Registered ids validate against their own ladder (vision shares
        // the nano 768 ladder); unknown ids keep the frozen legacy check.
        match lookup_model(model_id) {
            Some(spec) => {
                check_truncate_dim_for(truncate_dim, spec.native_dim, spec.ladder)?;
            }
            None => check_truncate_dim_for(
                truncate_dim,
                VISION_NATIVE_DIM,
                &[32, 64, 128, 256, 512, 768],
            )?,
        }

        // Resolve the device BEFORE acquisition (same discipline as the
        // text path): precision follows the landed device, and explicit
        // fp16-on-CPU fails here — before any download.
        let (_, used_cuda) = resolve_provider_names(device);
        let precision = resolve_vision_precision(precision, used_cuda)?;
        let artifact = artifact_for(model_id, precision);

        // ---- acquisition (per-artifact file layout) ----
        let (owner, name) = parse_owner_name(artifact.repo)?;
        let client = hf_client(cache_dir.clone())?;
        let repo = client.model(owner, name);
        let rev = revision.map(str::to_string);
        let onnx_path = repo
            .download_file()
            .filename(artifact.file.to_string())
            .maybe_revision(rev.clone())
            .send()
            .map_err(|e| anyhow!("{} missing for '{}': {e}", artifact.file, artifact.repo))?;
        match artifact.sidecar() {
            Some(sidecar) if repo
                .download_file()
                .filename(sidecar.clone())
                .maybe_revision(rev)
                .send()
                .is_err() =>
            {
                return Err(anyhow!(
                    "vision weights sidecar {sidecar} missing for '{}': \
                     external-data weights cannot run without it",
                    artifact.repo
                ));
            }
            _ => {}
        }
        // ---- tokenizer: truncation permanently off (see load_vision_tokenizer) ----
        let tok_path = fetch_tokenizer_file(artifact.repo, revision, cache_dir)?;
        let tokenizer = load_vision_tokenizer(&tok_path)?;

        let (session, used_cuda) = build_vision_session(&onnx_path, device, gpu_mem_limit)?;
        Ok(Self {
            session,
            tokenizer,
            dim: truncate_dim,
            model_id: model_id.to_string(),
            used_cuda,
            precision,
        })
    }

    /// Load from explicit local files — no network access. The
    /// external-data sidecar must sit next to `onnx_path` (ORT resolves it
    /// relative to the model file). Precision selection does not apply
    /// here (the files are what they are); the reported precision reads
    /// `fp32` and callers that need FP16 weights pass the FP16 files
    /// explicitly (mirrors `JinaV5::open_files`).
    pub fn open_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        truncate_dim: usize,
        device: DeviceReq,
        gpu_mem_limit: Option<u64>,
    ) -> Result<Self> {
        let ladder = lookup_model(VISION_NANO_MODEL)
            .map(|s| s.ladder)
            .unwrap_or(&[32, 64, 128, 256, 512, 768]);
        check_truncate_dim_for(truncate_dim, VISION_NATIVE_DIM, ladder)?;
        if !onnx_path.is_file() {
            return Err(anyhow!("onnx model not found: {}", onnx_path.display()));
        }
        if !tokenizer_path.is_file() {
            return Err(anyhow!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            ));
        }
        let tokenizer = load_vision_tokenizer(tokenizer_path)?;
        let (session, used_cuda) = build_vision_session(onnx_path, device, gpu_mem_limit)?;
        Ok(Self {
            session,
            tokenizer,
            dim: truncate_dim,
            model_id: onnx_path.display().to_string(),
            used_cuda,
            precision: Precision::Fp32,
        })
    }

    /// Embed one already-resized RGB image: row-major `[h, w, 3]` uint8
    /// where `(h, w)` is its own [`vision_target_size`] (Pillow bicubic on
    /// the caller side). One image per call — batching variable-shape
    /// images buys nothing (matches the text path's sequential `encode_many`).
    pub fn encode_image(&self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
        check_target_size(h, w)?;
        let gh = h / 16;
        let gw = w / 16;
        let patches = gh * gw;

        let pv = pixel_values(rgb, h, w)?;
        let pos = vision_pos_ids(gh, gw);
        let (idx, wts) = interp_taps(gh, gw);
        let ids = prompt_ids(&self.tokenizer, patches / 4)?;
        let seq = ids.len();
        if seq != patches / 4 + VISION_PROMPT_OVERHEAD {
            return Err(anyhow!(
                "prompt tokenized to {seq} tokens, expected {} ({} image + {}) — \
                 wrong tokenizer for the vision contract",
                patches / 4 + VISION_PROMPT_OVERHEAD,
                patches / 4,
                VISION_PROMPT_OVERHEAD
            ));
        }
        // Bind the feed arrays first: TensorRef borrows them across run().
        let ids_arr = Array2::from_shape_vec((1, seq), ids)
            .map_err(|e| anyhow!("input_ids shape: {e}"))?;
        let mask_arr = Array2::from_shape_vec((1, seq), vec![1i64; seq])
            .map_err(|e| anyhow!("attention_mask shape: {e}"))?;
        let pos_arr = Array2::from_shape_vec(
            (patches, 2),
            pos.iter().flat_map(|p| *p).collect(),
        )
        .map_err(|e| anyhow!("vision_pos_ids shape: {e}"))?;
        let idx_arr = Array2::from_shape_vec(
            (patches, 4),
            idx.iter().flat_map(|p| *p).collect(),
        )
        .map_err(|e| anyhow!("interp_indices shape: {e}"))?;
        let wts_arr = Array2::from_shape_vec(
            (patches, 4),
            wts.iter().flat_map(|p| *p).collect(),
        )
        .map_err(|e| anyhow!("interp_weights shape: {e}"))?;

        let embedding = {
            let mut guard =
                self.session.write().map_err(|_| anyhow!("session lock poisoned"))?;
            let ids_t = oe(ort::value::TensorRef::from_array_view(&ids_arr))?;
            let mask_t = oe(ort::value::TensorRef::from_array_view(&mask_arr))?;
            let pv_t = oe(ort::value::TensorRef::from_array_view(&pv))?;
            let pos_t = oe(ort::value::TensorRef::from_array_view(&pos_arr))?;
            let idx_t = oe(ort::value::TensorRef::from_array_view(&idx_arr))?;
            let wts_t = oe(ort::value::TensorRef::from_array_view(&wts_arr))?;
            let outputs = oe(guard.run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => mask_t,
                "pixel_values" => pv_t,
                "vision_pos_ids" => pos_t,
                "interp_indices" => idx_t,
                "interp_weights" => wts_t
            }))?;
            oe(outputs["sentence_embedding"].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow!("expected [1,768] output: {e}"))?
        };
        let pooled = embedding.slice(ndarray::s![0, ..]).to_owned();
        Ok(l2_truncate(&pooled, self.dim))
    }

    /// Configured output dimension (post-Matryoshka, same ladder as text-nano).
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

    /// Weight precision the session was opened with (`auto` already
    /// resolved against the landed device). Explicit files report `fp32`.
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    fn fixture(name: &str) -> serde_json::Value {
        let full = format!("{}/fixtures/vision/{}", env!("CARGO_MANIFEST_DIR"), name);
        let text =
            std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{full}: {e}"));
        serde_json::from_str(&text).expect("fixture parses")
    }

    fn as_i64_grid(v: &serde_json::Value) -> Vec<Vec<i64>> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
            })
            .collect()
    }

    fn as_f32_grid(v: &serde_json::Value) -> Vec<Vec<f32>> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap() as f32)
                    .collect()
            })
            .collect()
    }

    // ---- resize contract -----------------------------------------------------

    #[test]
    fn target_sizes_match_reference() {
        let cases = fixture("target_sizes.json");
        assert!(!cases.as_array().unwrap().is_empty());
        for c in cases.as_array().unwrap() {
            let h = c["in"][0].as_u64().unwrap() as u32;
            let w = c["in"][1].as_u64().unwrap() as u32;
            match vision_target_size(h, w) {
                Ok((rh, rw)) => {
                    assert_eq!([rh as u64, rw as u64], [c["out"][0].as_u64().unwrap(), c["out"][1].as_u64().unwrap()],
                        "{h}x{w}");
                }
                Err(_) => assert!(c["out"].is_null(), "{h}x{w} should resize"),
            }
        }
    }

    #[test]
    fn target_size_rejects_extreme_aspect() {
        // 3201:16 > 200:1 both ways round.
        assert!(vision_target_size(3201, 16).is_err());
        assert!(vision_target_size(16, 3201).is_err());
        assert!(vision_target_size(0, 64).is_err());
    }

    #[test]
    fn check_target_size_rejects_unresized() {
        // 100x100 is not its own target (upscales to 512x512).
        let e = check_target_size(100, 100).unwrap_err().to_string();
        assert!(e.contains("Pillow"), "{e}");
        assert!(check_target_size(512, 512).is_ok());
    }

    // ---- host tensors, bit-identical -----------------------------------------

    #[test]
    fn host_tensors_match_reference_all_grids() {
        let recs = fixture("host_tensors.json");
        let recs = recs.as_array().unwrap();
        assert!(recs.len() >= 30, "sweep coverage");
        for r in recs {
            let gh = r["grid"][0].as_u64().unwrap() as usize;
            let gw = r["grid"][1].as_u64().unwrap() as usize;
            let pos = vision_pos_ids(gh, gw);
            let want_pos = as_i64_grid(&r["pos"]);
            assert_eq!(pos.len(), want_pos.len(), "{gh}x{gw} rows");
            for (got, want) in pos.iter().zip(&want_pos) {
                assert_eq!(&got[..], &want[..], "{gh}x{gw}");
            }
            let (idx, wts) = interp_taps(gh, gw);
            let want_idx = as_i64_grid(&r["idx"]);
            let want_wts = as_f32_grid(&r["wts"]);
            assert_eq!(idx.len(), want_idx.len(), "{gh}x{gw}");
            for (got, want) in idx.iter().zip(&want_idx) {
                assert_eq!(&got[..], &want[..], "{gh}x{gw}");
            }
            for (got, want) in wts.iter().zip(&want_wts) {
                // Bitwise: f64 fixture values are the exact f32 values.
                let got_bits = got.map(f32::to_bits);
                let want_bits: [u32; 4] =
                    [want[0].to_bits(), want[1].to_bits(), want[2].to_bits(), want[3].to_bits()];
                assert_eq!(got_bits, want_bits, "{gh}x{gw}");
            }
        }
    }

    // ---- pixel pipeline ------------------------------------------------------

    #[test]
    fn pixel_values_match_reference() {
        let cases = fixture("pixel_cases.json");
        for c in cases.as_array().unwrap() {
            let h = c["size"][0].as_u64().unwrap() as usize;
            let w = c["size"][1].as_u64().unwrap() as usize;
            let raw = base64::engine::general_purpose::STANDARD
                .decode(c["rgb_b64"].as_str().unwrap())
                .unwrap();
            let pv = pixel_values(&raw, h, w).unwrap();
            let gh = c["grid"][0].as_u64().unwrap() as usize;
            let gw = c["grid"][1].as_u64().unwrap() as usize;
            assert_eq!(pv.dim(), (gh * gw, 1536));
            let want: Vec<f32> = c["pixel_values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect();
            assert_eq!(want.len(), gh * gw * 1536);
            for (k, &v) in pv.iter().enumerate() {
                assert_eq!(v.to_bits(), want[k].to_bits(), "case {h}x{w} flat [{k}]");
            }
            // Same grids through the shared host-tensor path.
            let pos = vision_pos_ids(gh, gw);
            let want_pos = as_i64_grid(&c["pos"]);
            for (got, want) in pos.iter().zip(&want_pos) {
                assert_eq!(&got[..], &want[..]);
            }
        }
    }

    #[test]
    fn pixel_values_rejects_bad_shapes() {
        assert!(pixel_values(&vec![0u8; 30 * 30 * 3], 30, 30).is_err()); // not %16
        assert!(pixel_values(&vec![0u8; 10], 32, 32).is_err()); // short buffer
    }

    // ---- precision policy ----------------------------------------------------

    #[test]
    fn auto_follows_device_fp16_explicit_on_cpu_fails() {
        assert_eq!(resolve_vision_precision(Precision::Auto, true).unwrap(), Precision::Fp16);
        assert_eq!(resolve_vision_precision(Precision::Auto, false).unwrap(), Precision::Fp32);
        assert_eq!(resolve_vision_precision(Precision::Fp32, true).unwrap(), Precision::Fp32);
        // Explicit fp16 on CPU would stall (observed hang) — fail, don't hang.
        assert!(resolve_vision_precision(Precision::Fp16, false).is_err());
        assert!(resolve_vision_precision(Precision::Fp16, true).is_ok());
    }

    #[test]
    fn open_rejects_bad_dims_before_io() {
        let e = JinaV5Vision::open(
            VISION_NANO_MODEL, None, None, 1024, DeviceReq::Cpu, Precision::Fp32, None,
        ).err().expect("open should fail").to_string();
        assert!(e.contains("1..=768"), "{e}");
        // ...as does the explicit-fp16-on-CPU refusal (before any download).
        let e = JinaV5Vision::open(
            VISION_NANO_MODEL, None, None, 512, DeviceReq::Cpu, Precision::Fp16, None,
        ).err().expect("open should fail").to_string();
        assert!(e.contains("stall"), "{e}");
    }
}
