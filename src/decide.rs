//! The typed-decision contract (GLiClass uni-encoder).
//!
//! State + typed questions in, one calibrated distribution per question
//! out — single forward pass, no per-label NLI forwards. Verified artifact
//! (plan `docs/typed-decision-plan.md` §8, ORT 1.29-safe opset 14,
//! torch↔ONNX exact match on 8 probes at pinned revision `21edefaf`):
//!
//! - `knowledgator/gliclass-small-v1.0` (English single/multi-label,
//!   144M, DeBERTa-v3 encoder, 2-input graph — no token_type_ids)
//!
//! Session building reuses the shared [`crate::jina::build_session`]
//! (the graph declares `logits`, so the output preference lands without
//! special-casing). Input construction is the decider half and part of
//! the contract: `state + (<<LABEL>>option)* + <<SEP>> + instructions`
//! (`prompt_first=false`), truncate-state-side-first — the options are
//! the question and are never cut to fit the state.

use crate::acquire::{
    artifact_for_task, fetch_tokenizer_file, hf_client, lookup_task, parse_owner_name,
};
use crate::error::{oe, Result};
use crate::jina::{build_session, count_tokens_in, load_tokenizer};
use crate::policy::{resolve_provider_names, DeviceReq, Precision, SessionPolicy};
use anyhow::anyhow;
use ndarray::Array2;
use ort::session::Session;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// Default sequence cap: the DeBERTa-v3 positional ceiling and the
/// pipeline default. `open()` defaults to the registered ceiling,
/// explicit files default here.
pub const DECIDE_DEFAULT_MAX: usize = 512;
/// Absolute cap for explicit files (largest ceiling on the roadmap —
/// Laya and the GLiClass v3/v4 graphs all sit at or under 512; 4096
/// leaves headroom without inviting silent garbage from 512-position
/// models). Explicit files are caller-asserted; without a registry entry
/// there is no per-model ceiling to validate against.
pub const DECIDE_ABSOLUTE_MAX: usize = 4096;

/// Validate an optional sequence cap against one task model's positional
/// ceiling before any I/O: None selects the model's full context.
/// Returns the effective limit.
pub(crate) fn check_decide_limit_for(
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

/// A typed question over a state. `Choice` is single-label (softmax over
/// the options — exactly one wins); `Score` is multi-label (sigmoid per
/// option, native unthresholded scores — thresholds are consumer policy).
/// `Noul` (none-of-the-above handling) is declared so callers can ask,
/// but no registered model serves it yet — it fails loudly at decide
/// time instead of degrading into a threshold nobody fitted.
pub enum Question {
    Choice { instructions: String, options: Vec<String> },
    Score { instructions: String, options: Vec<String> },
    Noul { instructions: String, options: Vec<String> },
}

impl Question {
    fn kind(&self) -> &'static str {
        match self {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        }
    }

    fn options(&self) -> &[String] {
        match self {
            Question::Choice { options, .. }
            | Question::Score { options, .. }
            | Question::Noul { options, .. } => options,
        }
    }

    fn instructions(&self) -> &str {
        match self {
            Question::Choice { instructions, .. }
            | Question::Score { instructions, .. }
            | Question::Noul { instructions, .. } => instructions,
        }
    }
}

/// One answered question: the winning option plus the full native
/// distribution in option order (softmax for `Choice`, sigmoid for
/// `Score` — never thresholded; the consumer decides what counts).
pub struct Decision {
    pub label: String,
    pub probabilities: Vec<(String, f32)>,
}

/// Numerically stable softmax (max-subtracted) — the `Choice` activation.
pub(crate) fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// Logistic sigmoid — the `Score` activation (native, unthresholded).
pub(crate) fn sigmoid(logits: &[f32]) -> Vec<f32> {
    logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
}

pub struct Decider {
    session: RwLock<Session>,
    /// Truncation-free handle: state and label-block length budgets are
    /// measured here, then the surviving ids are concatenated manually —
    /// the labels side is never truncated, so the pair-tokenizer
    /// `OnlySecond` pattern (which cuts the second span) does not apply.
    plain: tokenizers::Tokenizer,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    max_len: usize,
    model_id: String,
    prompt_first: bool,
    used_cuda: bool,
    precision: Precision,
    feed_token_type_ids: bool,
    output_name: String,
}

fn special_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| {
        anyhow!("GLiClass tokenizer lacks required special token '{token}'")
    })
}

/// Assemble one input sequence: `[CLS] + state + label-block + [SEP]`.
/// Pure function — the layout half of the contract, layout-pinned by the
/// e2e (manual concat must equal a single recipe encode).
pub(crate) fn build_sequence(
    state_ids: &[u32],
    label_ids: &[u32],
    cls_id: u32,
    sep_id: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(state_ids.len() + label_ids.len() + 2);
    out.push(cls_id);
    out.extend_from_slice(state_ids);
    out.extend_from_slice(label_ids);
    out.push(sep_id);
    out
}

/// Labels-first variant (`prompt_first: true` — the v4-edge recipe):
/// `[CLS] + label-block + state + [SEP]`. The label block (options +
/// SEP + instructions) is built identically; only its position moves.
/// Truncation discipline is unchanged: the state keeps its head in
/// both layouts, the question is never cut.
pub(crate) fn build_sequence_labels_first(
    label_ids: &[u32],
    state_ids: &[u32],
    cls_id: u32,
    sep_id: u32,
) -> Vec<u32> {
    let mut out = Vec::with_capacity(state_ids.len() + label_ids.len() + 2);
    out.push(cls_id);
    out.extend_from_slice(label_ids);
    out.extend_from_slice(state_ids);
    out.push(sep_id);
    out
}

/// Truncate-state-side-first: how many state ids survive given the label
/// block and the cap. Errors when the options alone (plus specials) fill
/// the window — the question is never cut to fit the state.
pub(crate) fn truncate_state_to(state_len: usize, label_len: usize, cap: usize) -> Result<usize> {
    let room = cap.checked_sub(label_len + 2).filter(|&r| r > 0).ok_or_else(|| {
        anyhow!(
            "options encode to {label_len} tokens plus 2 specials, filling the \
             {cap}-token window with no room for the state (state is {state_len} \
             tokens) — shorten the options or raise max_length"
        )
    })?;
    // Short states pass through uncut; only over-long states are truncated
    // (to their head — the B0 recipe order).
    Ok(room.min(state_len))
}

impl Decider {
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
        // ceiling; unknown ids fall back to the decide standard.
        let ceiling =
            lookup_task(model_id).map(|s| s.max_tokens).unwrap_or(DECIDE_DEFAULT_MAX);
        let max_length = check_decide_limit_for(max_length, ceiling)?;

        // Same device→precision ordering as JinaV5/Reranker: precision
        // follows the device the session will actually land on.
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
        // No sidecar probe: the spike artifact is B0-verified single-file
        // (ORT loads it standalone); a future external-data export fails
        // loudly at session load instead.
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
        // ---- tokenizer (no truncation policy — truncation is manual,
        // state-side-first, per call) ----
        let tok_path = fetch_tokenizer_file(artifact.repo, revision, cache_dir)?;
        // Registry layout; unknown ids take the text-first standard.
        let prompt_first = lookup_task(model_id).map(|s| s.prompt_first).unwrap_or(false);
        Self::from_files(&onnx_path, &tok_path, model_id, device, max_length, prompt_first, precision, cpu_arena)
    }

    /// Load from explicit local files — no network access. An external-data
    /// sidecar must sit next to `onnx_path` (ORT resolves it relative to the
    /// model file). Missing files fail before any tokenizer or session work;
    /// precision reads `fp32` (the files are what they are). Explicit
    /// files carry no registry metadata, so the input layout is
    /// caller-specified: false = text-first (small-v1.0 recipe), true =
    /// labels-first (v4-edge recipe) — a wrong layout scores garbage at
    /// parity, never an error, so there is no silent default to inherit.
    pub fn open_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        device: DeviceReq,
        max_length: Option<usize>,
        prompt_first: bool,
        cpu_arena: bool,
    ) -> Result<Self> {
        let max_length = match max_length {
            None => DECIDE_DEFAULT_MAX,
            Some(n) if (1..=DECIDE_ABSOLUTE_MAX).contains(&n) => n,
            Some(n) => {
                return Err(anyhow!(
                    "max_length must be within 1..={DECIDE_ABSOLUTE_MAX} for explicit files, got {n}"
                ))
            }
        };
        Self::from_files(
            onnx_path,
            tokenizer_path,
            &onnx_path.display().to_string(),
            device,
            max_length,
            prompt_first,
            Precision::Fp32,
            cpu_arena,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        model_id: &str,
        device: DeviceReq,
        max_length: usize,
        prompt_first: bool,
        precision: Precision,
        cpu_arena: bool,
    ) -> Result<Self> {
        if !onnx_path.is_file() {
            return Err(anyhow!("onnx model not found: {}", onnx_path.display()));
        }
        if !tokenizer_path.is_file() {
            return Err(anyhow!(
                "tokenizer file not found: {}",
                tokenizer_path.display()
            ));
        }
        let plain = load_tokenizer(tokenizer_path, None)?;
        let cls_id = special_id(&plain, "[CLS]")?;
        let sep_id = special_id(&plain, "[SEP]")?;
        let pad_id = special_id(&plain, "[PAD]").unwrap_or(0);
        let loaded =
            build_session(onnx_path, device, &SessionPolicy::decide_pair(), cpu_arena)?;
        Ok(Self {
            session: RwLock::new(loaded.session),
            plain,
            cls_id,
            sep_id,
            pad_id,
            max_len: max_length,
            model_id: model_id.to_string(),
            prompt_first,
            used_cuda: loaded.used_cuda,
            precision,
            feed_token_type_ids: loaded.feed_token_type_ids,
            output_name: loaded.output_name,
        })
    }

    /// Answer questions over one state — one padded session run for the
    /// whole batch (order-preserving). The single-forward property that
    /// beats per-label NLI: N questions = 1 run, not N (and not N×labels).
    pub fn decide(&self, state: &str, questions: &[Question]) -> Result<Vec<Decision>> {
        if questions.is_empty() {
            return Ok(Vec::new());
        }
        let items: Vec<(&str, &Question)> =
            questions.iter().map(|q| (state, q)).collect();
        self.decide_many(&items)
    }

    /// Answer (state, question) items in one padded session run.
    /// Order-preserving: output[i] answers items[i].
    pub fn decide_many(&self, items: &[(&str, &Question)]) -> Result<Vec<Decision>> {
        if items.is_empty() {
            return Ok(Vec::new());
        }
        // ---- validation first (no partial I/O, no partial runs) ----
        for (i, (_, q)) in items.iter().enumerate() {
            if matches!(q, Question::Noul { .. }) {
                return Err(anyhow!(
                    "item {i}: Noul questions are not served by this model \
                     (no registered Decide model handles none-of-the-above yet) — \
                     express abstention as a consumer-side threshold instead"
                ));
            }
            if q.options().is_empty() {
                return Err(anyhow!("item {i}: {} question needs at least one option", q.kind()));
            }
            if q.options().iter().any(|o| o.is_empty()) {
                return Err(anyhow!("item {i}: options must not be empty strings"));
            }
        }
        // ---- input construction (contract: truncate-state-side-first) ----
        let mut seqs: Vec<Vec<u32>> = Vec::with_capacity(items.len());
        let mut counts: Vec<usize> = Vec::with_capacity(items.len());
        for (state, q) in items {
            let label_text = format!(
                "{}<<SEP>>{}",
                q.options().iter().map(|o| format!("<<LABEL>>{o}")).collect::<String>(),
                q.instructions()
            );
            let state_ids = self
                .plain
                .encode(*state, false)
                .map_err(|e| anyhow!("state tokenization failed: {e}"))?;
            let label_ids = self
                .plain
                .encode(label_text.as_str(), false)
                .map_err(|e| anyhow!("options tokenization failed: {e}"))?;
            let keep = truncate_state_to(
                state_ids.len(),
                label_ids.len(),
                self.max_len,
            )?;
            let state_keep = &state_ids.get_ids()[..keep];
            seqs.push(if self.prompt_first {
                build_sequence_labels_first(
                    label_ids.get_ids(),
                    state_keep,
                    self.cls_id,
                    self.sep_id,
                )
            } else {
                build_sequence(state_keep, label_ids.get_ids(), self.cls_id, self.sep_id)
            });
            counts.push(q.options().len());
        }
        let t = seqs.iter().map(|s| s.len()).max().unwrap_or(0);
        let n = seqs.len();
        let mut ids = Vec::with_capacity(n * t);
        let mut mask = Vec::with_capacity(n * t);
        for s in &seqs {
            ids.extend(s.iter().map(|&x| x as i64));
            mask.extend(std::iter::repeat_n(1i64, s.len()));
            ids.extend(std::iter::repeat_n(self.pad_id as i64, t - s.len()));
            mask.extend(std::iter::repeat_n(0i64, t - s.len()));
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
                // No registered Decide graph declares token_type_ids
                // today; the branch exists so a future export is fed, not
                // refused (the rerank seam, reused).
                let tids = Array2::from_shape_vec((n, t), vec![0i64; n * t])
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
                .map_err(|e| anyhow!("expected [B,C] logits: {e}"))?
        };
        if logits.shape()[0] != n {
            return Err(anyhow!(
                "logits batch {:?} does not match {n} decided items",
                logits.shape()
            ));
        }
        // ---- readout: first C_i logits per row (batched rows pad to the
        // batch max — B0-verified: row prefixes equal single runs
        // exactly, tail slots are padding garbage and unread). ----
        let mut out = Vec::with_capacity(n);
        for (i, (_, q)) in items.iter().enumerate() {
            let c = counts[i];
            if logits.shape()[1] < c {
                return Err(anyhow!(
                    "item {i}: graph returned {} class slots, need {c} for the options",
                    logits.shape()[1]
                ));
            }
            let row: Vec<f32> = logits.row(i).iter().take(c).copied().collect();
            let probs = match q {
                Question::Choice { .. } => softmax(&row),
                // Sigmoid, native and unthresholded — the consumer fits
                // thresholds (the B0 multi-label lesson: raw sigmoids are
                // systematically over-confident, so nothing here pretends
                // otherwise).
                Question::Score { .. } => sigmoid(&row),
                Question::Noul { .. } => unreachable!("rejected in validation"),
            };
            let (best, _) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .expect("non-empty options");
            out.push(Decision {
                label: q.options()[best].clone(),
                probabilities: q
                    .options()
                    .iter()
                    .cloned()
                    .zip(probs)
                    .collect(),
            });
        }
        Ok(out)
    }

    /// Token count without special tokens — the same helper JinaV5 uses
    /// (shared [`count_tokens_in`]), measured through the truncation-free
    /// handle so long states report their true length. The label-block
    /// overhead is NOT included — size windows against the cap with the
    /// options encoded separately, not this number alone.
    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        count_tokens_in(&self.plain, text)
    }

    /// Effective sequence cap (truncation ceiling for decides).
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Input layout: false = text-first (small-v1.0 recipe), true =
    /// labels-first (v4-edge recipe). Resolved from the task registry;
    /// explicit files default to text-first.
    pub fn prompt_first(&self) -> bool {
        self.prompt_first
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
    fn decide_limit_defaults_to_ceiling() {
        use crate::acquire::GLICLASS_SMALL_MODEL;
        let spec = lookup_task(GLICLASS_SMALL_MODEL).unwrap();
        assert_eq!(check_decide_limit_for(None, spec.max_tokens).unwrap(), 512);
        assert_eq!(check_decide_limit_for(Some(128), spec.max_tokens).unwrap(), 128);
        assert!(check_decide_limit_for(Some(0), spec.max_tokens).is_err());
        assert!(check_decide_limit_for(Some(513), spec.max_tokens).is_err());
    }

    #[test]
    fn softmax_sums_to_one_and_argmaxes() {
        let p = softmax(&[13.48, 13.52, 9.34]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[1] > p[0] && p[0] > p[2]);
        // Uniform input stays uniform.
        for x in softmax(&[0.0, 0.0, 0.0]) {
            assert!((x - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn sigmoid_midpoint_and_monotone() {
        assert!((sigmoid(&[0.0])[0] - 0.5).abs() < 1e-6);
        let p = sigmoid(&[-1.4, 13.4]);
        assert!(p[0] < 0.5 && p[1] > 0.99);
    }

    #[test]
    fn build_sequence_layout() {
        // [CLS] + state + labels + [SEP] — the B0 recipe order.
        assert_eq!(build_sequence(&[11, 12], &[21, 22, 23], 101, 102), vec![101, 11, 12, 21, 22, 23, 102]);
        assert_eq!(build_sequence(&[], &[21], 101, 102), vec![101, 21, 102]);
    }

    #[test]
    fn build_sequence_labels_first_layout() {
        // [CLS] + labels + state + [SEP] — the v4-edge recipe order.
        assert_eq!(build_sequence_labels_first(&[21, 22], &[11, 12], 101, 102), vec![101, 21, 22, 11, 12, 102]);
        assert_eq!(build_sequence_labels_first(&[21], &[], 101, 102), vec![101, 21, 102]);
    }

    #[test]
    fn task_registry_carries_layout() {
        use crate::acquire::{GLICLASS_SMALL_MODEL, GLICLASS_V4_EDGE_MODEL};
        assert!(!lookup_task(GLICLASS_SMALL_MODEL).unwrap().prompt_first);
        let v4 = lookup_task(GLICLASS_V4_EDGE_MODEL).unwrap();
        assert!(v4.prompt_first);
        assert_eq!(v4.max_tokens, 2048);
    }

    #[test]
    fn truncate_state_keeps_short_states_whole() {
        assert_eq!(truncate_state_to(10, 20, 512).unwrap(), 10);
        assert_eq!(truncate_state_to(400, 20, 512).unwrap(), 400);
        // Over-long states cut to head room after labels + specials.
        assert_eq!(truncate_state_to(600, 20, 512).unwrap(), 490);
    }

    #[test]
    fn truncate_state_never_cuts_options() {
        let e = truncate_state_to(10, 510, 512).err().expect("should fail").to_string();
        assert!(e.contains("no room for the state"), "{e}");
        // Empty states are legitimate (labels-only input still scores).
        assert_eq!(truncate_state_to(0, 5, 512).unwrap(), 0);
    }

    #[test]
    fn open_files_rejects_missing_model_before_anything() {
        let e = Decider::open_files(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
            DeviceReq::Cpu,
            None,
            false,
            false,
        )
        .err()
        .expect("open_files should fail")
        .to_string();
        assert!(e.contains("onnx model not found"), "{e}");
    }

    #[test]
    fn open_files_rejects_bad_limit_before_io() {
        for bad in [Some(0usize), Some(DECIDE_ABSOLUTE_MAX + 1)] {
            let e = Decider::open_files(
                Path::new("/nonexistent/model.onnx"),
                Path::new("/nonexistent/tokenizer.json"),
                DeviceReq::Cpu,
                bad,
                false,
                false,
            )
            .err()
            .expect("open_files should fail")
            .to_string();
            assert!(e.contains("1..=4096"), "{bad:?}: {e}");
        }
    }

    #[test]
    fn open_rejects_out_of_range_limit_before_io() {
        use crate::acquire::GLICLASS_SMALL_MODEL;
        let e = Decider::open(
            GLICLASS_SMALL_MODEL,
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
        let e = Decider::open(
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
