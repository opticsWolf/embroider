//! Slot-layout typed decisions: serving Mapika/decider-2b.
//!
//! GLiClass splices options into the token stream (2 inputs, label
//! text). Laya gathers marker slots (5 inputs). decider-2b is a third
//! contract: a Qwen3.5-2B decoder where each question is one token row —
//! `Context:` + state, then `\n\nQuestion: …\nOptions:` with lettered
//! options and a trailing `Answer: (` whose `(` slot is scored. 2 inputs
//! (`input_ids`, `slot_pos`), 1 output (`label_logits` [rows, 255]); no
//! attention mask (causal; right padding never reaches earlier
//! positions). This module ports
//! `ollaya_convert.families.decider.layout` 1:1 — render, lettered/wide
//! pieces, isolated yes/no score rows, row batching, option logits —
//! plus the checkpoint temperatures from `calibration.json`, so
//! embroider serves the `opticsWolf/decider-2b-fp32-onnx` weightless
//! export bit-faithfully instead of approximating it.
//!
//! Question shapes mirror `LayaQuestion` on purpose (Choice with
//! descriptions, level Scores, Noul) but the type is its own: decider's
//! noul carries **both** criteria (`true` and `false` descriptions —
//! every S1MB noul ships both), where Laya's has one statement. Sharing
//! the enum would silently drop the false side on the whole noul bench.
//! Same reason `Decider::open` refuses this id: a slot graph fed to the
//! 2-input GLiClass path would fail late instead of routing here.
//!
//! CUDA runs the **parallel executor** (`SessionPolicy::decider2b`): the
//! sequential memory planner aliases a shape buffer on this graph (ORT
//! #4829 class — a chunk-scan Reshape reads a zeroed dim and the run
//! fails deterministically on the first batch), and the C-level
//! `enable_mem_reuse=false` that fixes it is not reachable from the C
//! API (no setter; the `session.enable_mem_reuse` config key is not
//! honored — both probed). Parallel execution is parity-identical on
//! CPU + CUDA (e2e-pinned) — but only with ORT's default thread pools:
//! an explicit `inter_op_num_threads=1` re-triggers the corruption
//! under the parallel executor (probed), so the policy sets no thread
//! counts. The memory-pattern switch is NOT an
//! equivalent (probed: still corrupt).
//!
//! Deliberately NOT ported: anything the checkpoint doesn't ship —
//! fitted binning maps (consumer-side, as with Laya), `predict_long`
//! window scans (the caller windows + aggregates; per-question decode
//! here is already order-preserving so scans compose).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use ndarray::Array2;

use crate::acquire::{
    artifact_for_task, fetch_repo_file, lookup_task, TaskKind, DECIDER2B_MODEL,
};
use crate::error::oe;
use crate::jina::{count_tokens_in, load_tokenizer};
use crate::laya::{answer_confidence, confidence_from_probs, softmax_stable, temperature_rejected};
use crate::policy::{resolve_provider_names, DeviceReq, Precision, SessionPolicy};
use crate::providers::apply_providers_with_arena;

/// One decider-2b option: a label with an optional description (rendered
/// as `label: description`, the description side of decider's criteria
/// dict). Labels are the answer keys and must be distinct.
#[derive(Clone, Debug, PartialEq)]
pub struct Decider2BOption {
    pub label: String,
    pub description: Option<String>,
}

impl Decider2BOption {
    pub fn plain(label: &str) -> Self {
        Self { label: label.to_string(), description: None }
    }
}

/// A decider-2b question. Shapes mirror `LayaQuestion` deliberately —
/// Choice: one distribution over options; Score: one distribution over
/// LEVELS (expected index in `.value`); Noul: P(true) for the pair of
/// criteria (the rerank primitive) — but noul carries **both** sides:
/// `true_criterion` / `false_criterion` render as `yes: …` / `no: …`
/// (bare `yes` / `no` when empty), exactly like decider's own criteria
/// dict. Options render in label order (no `option_order`: the reference
/// layout has no such concept - a caller that wants position de-biasing
/// permutes the options vec itself before calling).
#[derive(Clone, Debug, PartialEq)]
pub enum Decider2BQuestion {
    Choice {
        instructions: String,
        options: Vec<Decider2BOption>,
    },
    Score {
        instructions: String,
        levels: Vec<String>,
    },
    Noul {
        instructions: String,
        true_criterion: String,
        false_criterion: String,
    },
}

/// One answered decider-2b question. Native f32 throughout (presentation
/// rounding is the caller's; goldens pin full precision). No escalation
/// head exists on this graph, so there is no `act_probability` — the
/// tuple is five wide, not six.
#[derive(Clone, Debug, PartialEq)]
pub struct Decider2BDecision {
    /// Choice: winning label. Score: top level's text. Noul:
    /// "true"/"false".
    pub label: String,
    /// Choice: per-option (label, p). Score: per-level (text, p). Noul:
    /// `[("false", p0), ("true", p1)]`. Caller order (unpermuted).
    pub probabilities: Vec<(String, f32)>,
    /// Score only: the expected level index (`Σ i·pᵢ`).
    pub value: Option<f32>,
    /// Noul: max(p) (the two-sided certainty). Elsewhere the
    /// Shannon-concentration confidence (1.0 for k < 2).
    pub confidence: f32,
    /// Calibrated confidence: max(p) — what the temperatures were fitted on.
    pub answer_confidence: f32,
}

/// Usable-temperature clamp for this checkpoint: [0.05, 5.0]
/// (`temperature_range` in calibration.json) — NOT Laya's [0.5, 5.0]:
/// the fitted noul temperature IS 0.05 and must survive the clamp.
/// NaN/inf/non-number entries read 1.0 (unfitted bucket).
pub(crate) fn clamp_decider2b_temperature(t: f64) -> f64 {
    if t.is_nan() || t.is_infinite() {
        return 1.0;
    }
    t.clamp(0.05, 5.0)
}

/// Checkpoint temperatures as loaded: the (choice, score, noul) triple
/// after the [0.05, 5.0] clamp, plus the load-time rejection list (the
/// agent's load-time warnings, queryable instead of printed).
#[derive(Clone, Debug)]
pub struct Decider2BTemperatures {
    pub triple: [f64; 3],
    pub rejections: Vec<String>,
}

impl Decider2BTemperatures {
    fn load(cfg: &serde_json::Value) -> Result<Self> {
        let raw_triple = cfg.get("temperature").and_then(|v| v.as_array()).ok_or_else(|| {
            anyhow!("calibration.json has no 'temperature' triple - not a decider-2b checkpoint")
        })?;
        if raw_triple.len() != 3 {
            return Err(anyhow!(
                "calibration.json 'temperature' must have 3 entries, got {}",
                raw_triple.len()
            ));
        }
        let mut triple = [1.0f64; 3];
        let mut rejections = Vec::new();
        for (i, raw) in raw_triple.iter().enumerate() {
            let applied = clamp_decider2b_temperature(raw.as_f64().unwrap_or(f64::NAN));
            triple[i] = applied;
            if temperature_rejected(raw, applied) {
                rejections.push(format!("temperature[{i}]={raw} -> {applied}"));
            }
        }
        Ok(Self { triple, rejections })
    }
}

fn qtype_of(q: &Decider2BQuestion) -> usize {
    match q {
        Decider2BQuestion::Choice { .. } => 0,
        Decider2BQuestion::Score { .. } => 1,
        Decider2BQuestion::Noul { .. } => 2,
    }
}

/// Width at/below which options render as lettered text (`(A) …`);
/// wider questions address one label token per option instead
/// (`decider-slots-v1`: `NARROW = 10`).
const NARROW: usize = 10;
const LETTERS: &[u8] = b"ABCDEFGHIJ";
/// Isolated yes/no score rows (`ISOLATED`): the level proposes, the
/// pair disposes. Thepiece adds the `\n\nQuestion:` prefix — this is
/// only the `{q}` body (`_piece` is shared with the listwise path).
const ISOLATED_Q: &str = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?";
/// Row-batch token budget (`ort_rows.run_rows`: `budget = 8192`) and the
/// graph's sequence multiple (positions are implicit `0..seq-1`).
const ROW_BUDGET: usize = 8192;
const SEQ_MULTIPLE: usize = 64;

/// Strip a leading `N: ` level number (`strip_level_number`): the
/// isolated template takes the raw level text, and numbered legends
/// (`0: low`) would double-number it.
pub(crate) fn strip_level_number(text: &str) -> String {
    let t = text.trim_start();
    let mut rest = t;
    if rest.starts_with('-') {
        rest = &rest[1..];
    }
    let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits > 0 {
        let after = &rest[digits..];
        if let Some(stripped) = after.strip_prefix(':') {
            return stripped.trim_start().to_string();
        }
    }
    text.to_string()
}

/// Render option texts in label-index order (`render_question`):
/// - Choice: `label` or `label: description`.
/// - Score listwise (`isolated_levels=false` only): `i: description`.
/// - Noul: `no` / `no: {false}` then `yes` / `yes: {true}`.
fn render_decider_options(q: &Decider2BQuestion) -> Vec<String> {
    match q {
        Decider2BQuestion::Choice { options, .. } => options
            .iter()
            .map(|o| match &o.description {
                Some(d) if !d.is_empty() => format!("{}: {d}", o.label),
                _ => o.label.clone(),
            })
            .collect(),
        Decider2BQuestion::Score { levels, .. } => levels
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{i}: {c}"))
            .collect(),
        Decider2BQuestion::Noul { true_criterion, false_criterion, .. } => vec![
            if false_criterion.is_empty() {
                "no".to_string()
            } else {
                format!("no: {false_criterion}")
            },
            if true_criterion.is_empty() {
                "yes".to_string()
            } else {
                format!("yes: {true_criterion}")
            },
        ],
    }
}

/// Reject a question that cannot be answered, naming it. Validation
/// fires before any tokenizer or session work; caps come from the
/// checkpoint config (`max_options`, `max_levels`), not the code.
fn validate_decider2b_question(
    i: usize,
    q: &Decider2BQuestion,
    max_options: usize,
    max_levels: usize,
) -> Result<()> {
    let instructions = match q {
        Decider2BQuestion::Choice { instructions, .. }
        | Decider2BQuestion::Score { instructions, .. }
        | Decider2BQuestion::Noul { instructions, .. } => instructions,
    };
    if instructions.trim().is_empty() {
        return Err(anyhow!(
            "question {i}: 'instructions' must not be empty; add the text the model should answer"
        ));
    }
    match q {
        Decider2BQuestion::Choice { options, .. } => {
            if options.len() < 2 {
                return Err(anyhow!(
                    "question {i}: a choice question needs at least 2 options, got {}",
                    options.len()
                ));
            }
            if options.len() > max_options {
                return Err(anyhow!(
                    "question {i}: {} options exceed the {max_options} label slots this checkpoint serves",
                    options.len()
                ));
            }
            let mut seen = HashSet::new();
            for o in options {
                if !seen.insert(o.label.clone()) {
                    return Err(anyhow!(
                        "question {i}: duplicate choice label {:?}; labels are the answer keys and must be distinct",
                        o.label
                    ));
                }
            }
        }
        Decider2BQuestion::Score { levels, .. } => {
            if levels.len() < 2 {
                return Err(anyhow!(
                    "question {i}: a score question needs at least 2 levels, got {}",
                    levels.len()
                ));
            }
            if levels.len() > max_levels {
                return Err(anyhow!(
                    "question {i}: {} levels exceed the {max_levels} isolated rows this checkpoint serves",
                    levels.len()
                ));
            }
        }
        Decider2BQuestion::Noul { .. } => {}
    }
    Ok(())
}

/// One lettered/wide question piece: `\n\nQuestion: …\nOptions:` +
/// options + `\nAnswer: (` (`_piece`). Narrow questions letter inline;
/// wide ones address one label token per option. Returns ids + the slot
/// (the `(` token: its next-token logits over label ids are the option
/// logits) + the option count.
fn build_piece(
    tok: &tokenizers::Tokenizer,
    question: &str,
    options: &[String],
    label_ids: &[u32],
) -> Result<(Vec<u32>, usize)> {
    let head_text = format!("\n\nQuestion: {question}\nOptions:");
    if options.len() <= NARROW {
        let mut text = head_text;
        for (j, o) in options.iter().enumerate() {
            text.push_str(&format!("\n({}) {o}", LETTERS[j] as char));
        }
        text.push_str("\nAnswer: (");
        let ids = tok
            .encode(text.as_str(), false)
            .map_err(|e| anyhow!("question encode: {e}"))?;
        let mut v: Vec<u32> = ids.get_ids().to_vec();
        let slot = v.len() - 1;
        Ok((v, slot))
    } else {
        let mut v: Vec<u32> = tok
            .encode(head_text.as_str(), false)
            .map_err(|e| anyhow!("question encode: {e}"))?
            .get_ids()
            .to_vec();
        let open: Vec<u32> = tok
            .encode("\n(", false)
            .map_err(|e| anyhow!("open-paren encode: {e}"))?
            .get_ids()
            .to_vec();
        for (j, o) in options.iter().enumerate() {
            let label_id = *label_ids.get(j).ok_or_else(|| {
                anyhow!("wide question needs label id {j} ({} options)", options.len())
            })?;
            v.extend_from_slice(&open);
            v.push(label_id);
            let rest: Vec<u32> = tok
                .encode(format!(") {o}").as_str(), false)
                .map_err(|e| anyhow!("option encode: {e}"))?
                .get_ids()
                .to_vec();
            v.extend_from_slice(&rest);
        }
        let tail: Vec<u32> = tok
            .encode("\nAnswer: (", false)
            .map_err(|e| anyhow!("answer-tail encode: {e}"))?
            .get_ids()
            .to_vec();
        v.extend_from_slice(&tail);
        let slot = v.len() - 1;
        Ok((v, slot))
    }
}

/// One encoded row: token ids + the answer slot + the option count.
/// Score questions with `isolated_levels` become one yes/no row per
/// level; everything else is one row.
struct EncRow {
    ids: Vec<u32>,
    slot: usize,
    k: usize,
}

/// Isolated yes/no row for one level (`encode` score branch).
fn build_isolated_row(
    tok: &tokenizers::Tokenizer,
    ctx: &[u32],
    instructions: &str,
    level: &str,
    label_ids: &[u32],
    max_ctx: usize,
) -> Result<EncRow> {
    let q = ISOLATED_Q
        .replace("{q}", instructions)
        .replace("{level}", &strip_level_number(level));
    let (mut piece, slot_in_piece) = build_piece(tok, &q, &["no".to_string(), "yes".to_string()], label_ids)?;
    let mut ids = Vec::with_capacity(ctx.len().min(max_ctx) + piece.len());
    ids.extend_from_slice(&ctx[..ctx.len().min(max_ctx)]);
    let slot = ids.len() + slot_in_piece;
    ids.append(&mut piece);
    Ok(EncRow { ids, slot, k: 2 })
}

/// Encode one (state, question) pair into rows (`encode`): shared
/// `Context:` prefix truncated to the checkpoint ceiling, then the
/// piece(s). Plan rows index the shared row list.
fn encode_pair(
    tok: &tokenizers::Tokenizer,
    ctx: &[u32],
    q: &Decider2BQuestion,
    label_ids: &[u32],
    max_ctx: usize,
    isolated_levels: bool,
) -> Result<(Vec<EncRow>, PlanItem)> {
    let instructions = match q {
        Decider2BQuestion::Choice { instructions, .. }
        | Decider2BQuestion::Score { instructions, .. }
        | Decider2BQuestion::Noul { instructions, .. } => instructions,
    };
    match q {
        Decider2BQuestion::Score { levels, .. } if isolated_levels => {
            let mut rows = Vec::with_capacity(levels.len());
            let mut idx = Vec::with_capacity(levels.len());
            // rows accumulate in the caller; indices are relative here
            // and rebased by the caller.
            for level in levels {
                rows.push(build_isolated_row(tok, ctx, instructions, level, label_ids, max_ctx)?);
                idx.push(rows.len() - 1);
            }
            Ok((
                rows,
                PlanItem { qtype: 1, kind: RowKind::Iso, rows: idx, k: levels.len() },
            ))
        }
        _ => {
            let rendered = render_decider_options(q);
            let (mut piece, slot_in_piece) = build_piece(tok, instructions, &rendered, label_ids)?;
            let mut ids = Vec::with_capacity(ctx.len().min(max_ctx) + piece.len());
            ids.extend_from_slice(&ctx[..ctx.len().min(max_ctx)]);
            let slot = ids.len() + slot_in_piece;
            ids.append(&mut piece);
            let k = rendered.len();
            Ok((
                vec![EncRow { ids, slot, k }],
                PlanItem { qtype: qtype_of(q), kind: RowKind::List, rows: vec![0], k },
            ))
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum RowKind {
    List,
    Iso,
}

struct PlanItem {
    qtype: usize,
    kind: RowKind,
    rows: Vec<usize>,
    k: usize,
}

/// Per-question option logits from the graph output `label_logits`
/// [rows, 255] (`option_logits`): listwise rows read the first-k label
/// logits; isolated rows combine each yes/no pair into
/// `log_sigmoid((yes − no) / T_row)` so `softmax(z)` is the upstream
/// normalised per-level fit (the score calibration temperature is 1.0
/// because `T_row` is already inside `z`).
fn option_logits(item: &PlanItem, outs: &[Vec<f32>], row_temp: f64) -> Vec<f64> {
    match item.kind {
        RowKind::List => {
            let r = item.rows[0];
            outs[r].iter().take(item.k).map(|&x| x as f64).collect()
        }
        RowKind::Iso => item
            .rows
            .iter()
            .map(|&r| {
                let d = (outs[r][1] as f64 - outs[r][0] as f64) / row_temp;
                if d > -30.0 {
                    -(1.0 + (-d).exp()).ln()
                } else {
                    d - (1.0 + d.exp()).ln()
                }
            })
            .collect(),
    }
}

/// Pack row indices into ort_rows batches: length-sorted, each batch
/// holding while `L2 * count <= budget` with `L2` the 64-ceiling.
/// `L` is the batch's padded width.
pub(crate) fn pack_row_batches(lens: &[usize]) -> Vec<(Vec<usize>, usize)> {
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| lens[i]);
    let mut out = Vec::new();
    let mut i = 0;
    while i < order.len() {
        let l0 = lens[order[i]].div_ceil(SEQ_MULTIPLE) * SEQ_MULTIPLE;
        let mut l = l0;
        let mut j = i + 1;
        while j < order.len() {
            let l2 = lens[order[j]].div_ceil(SEQ_MULTIPLE) * SEQ_MULTIPLE;
            if l2 * (j + 1 - i) > ROW_BUDGET {
                break;
            }
            l = l2;
            j += 1;
        }
        out.push((order[i..j].to_vec(), l));
        i = j;
    }
    out
}

/// Decode one question from its row outputs (`_decode_answers` core:
/// temperature → softmax → slot→caller unpermute → per-type shape).
/// `order` is the choice slot permutation (None = identity).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_decider2b_row(
    logits: &[f64],
    q: &Decider2BQuestion,
    triple: &[f64; 3],
) -> Decider2BDecision {
    let qt = qtype_of(q);
    let k = logits.len();
    let t = triple[qt];
    let z: Vec<f64> = logits.iter().map(|&x| x / t).collect();
    let p = softmax_stable(&z);
    let pf: Vec<f32> = p.iter().map(|&x| x as f32).collect();
    // Noul reports max(P(true), P(false)) — the two-sided certainty —
    // not the entropy concentration.
    let confidence = if qt == 2 {
        p[1].max(1.0 - p[1]) as f32
    } else {
        confidence_from_probs(&p) as f32
    };
    let answer_confidence = answer_confidence(&p) as f32;
    match q {
        Decider2BQuestion::Choice { options, .. } => {
            let win = pf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            Decider2BDecision {
                label: options.get(win).map(|o| o.label.clone()).unwrap_or_default(),
                probabilities: options
                    .iter()
                    .zip(pf.iter())
                    .map(|(o, &v)| (o.label.clone(), v))
                    .collect(),
                value: None,
                confidence,
                answer_confidence,
            }
        }
        Decider2BQuestion::Score { levels, .. } => {
            let value: f64 = p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
            let win = pf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            Decider2BDecision {
                label: levels.get(win).cloned().unwrap_or_default(),
                probabilities: levels
                    .iter()
                    .zip(pf.iter())
                    .map(|(l, &v)| (l.clone(), v))
                    .collect(),
                value: Some(value as f32),
                confidence,
                answer_confidence,
            }
        }
        Decider2BQuestion::Noul { .. } => {
            let label = if p[1] >= p[0] { "true" } else { "false" }.to_string();
            Decider2BDecision {
                label,
                probabilities: vec![("false".to_string(), pf[0]), ("true".to_string(), pf[1])],
                value: None,
                confidence,
                answer_confidence,
            }
        }
    }
}

fn check_decider2b_limit_for(max_length: Option<usize>, ceiling: usize) -> Result<usize> {
    match max_length {
        None => Ok(ceiling),
        Some(n) if (1..=ceiling).contains(&n) => Ok(n),
        Some(n) => Err(anyhow!(
            "max_length must be within 1..={ceiling} for this checkpoint, got {n}"
        )),
    }
}

/// The checkpoint contract: option/label ceilings, the isolated-row
/// temperature and label alphabet from `decision.json`, plus the
/// clamped per-qtype temperatures from `calibration.json`.
#[derive(Clone, Debug)]
pub struct Decider2BConfig {
    pub max_ctx: usize,
    pub max_options: usize,
    pub min_options: usize,
    pub max_levels: usize,
    pub isolated_levels: bool,
    pub row_temp: f64,
    pub label_ids: Vec<u32>,
    pub pad_id: u32,
    pub temps: Decider2BTemperatures,
}

impl Decider2BConfig {
    fn load(decision: &serde_json::Value, calibration: &serde_json::Value) -> Result<Self> {
        let layout = decision.get("layout").and_then(|v| v.as_str()).unwrap_or("");
        if layout != "decider-slots-v1" {
            return Err(anyhow!(
                "decision.json layout={layout:?} is not the decider-slots-v1 contract this runtime serves"
            ));
        }
        let get_usize = |key: &str| {
            decision
                .get(key)
                .and_then(|v| v.as_u64())
                .filter(|&v| v > 0)
                .ok_or_else(|| anyhow!("decision.json has no positive '{key}'"))
                .map(|v| v as usize)
        };
        let max_ctx = get_usize("max_ctx_tokens")?;
        let max_options = get_usize("max_options")?;
        let min_options = decision
            .get("min_options")
            .and_then(|v| v.as_u64())
            .unwrap_or(2) as usize;
        let max_levels = get_usize("max_levels")?;
        let isolated_levels = decision
            .get("isolated_levels")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let row_temp = decision
            .get("isolated_row_temperature")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite() && *v > 0.0)
            .ok_or_else(|| anyhow!("decision.json has no positive 'isolated_row_temperature'"))?;
        let label_ids: Vec<u32> = decision
            .get("labels")
            .and_then(|l| l.get("ids"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("decision.json has no 'labels.ids' array"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .filter(|&n| n <= u32::MAX as u64)
                    .ok_or_else(|| anyhow!("decision.json label id {v} is not a token id"))
                    .map(|n| n as u32)
            })
            .collect::<Result<_>>()?;
        if label_ids.len() < max_options {
            return Err(anyhow!(
                "decision.json carries {} label ids for {max_options} option slots",
                label_ids.len()
            ));
        }
        let pad_id = decision
            .get("special_tokens")
            .and_then(|s| s.get("pad"))
            .and_then(|v| v.as_u64())
            .filter(|&n| n <= u32::MAX as u64)
            .ok_or_else(|| anyhow!("decision.json has no 'special_tokens.pad' token id"))? as u32;
        let temps = Decider2BTemperatures::load(calibration)?;
        Ok(Self {
            max_ctx,
            max_options,
            min_options,
            max_levels,
            isolated_levels,
            row_temp,
            label_ids,
            pad_id,
            temps,
        })
    }
}

pub struct Decider2B {
    session: std::sync::RwLock<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    cfg: Decider2BConfig,
    max_len: usize,
    model_id: String,
    used_cuda: bool,
}

impl Decider2B {
    /// Open the registered decider-2b checkpoint (`Mapika/decider-2b`,
    /// served from the `opticsWolf/decider-2b-fp32-onnx` self-export).
    /// Fetches graph + vendored safetensors + tokenizer +
    /// `decision.json` + `calibration.json` (the checkpoint truth:
    /// ceilings, label alphabet, temperatures) from the artifact repo.
    /// Unknown ids are refused — this export doesn't follow the legacy
    /// file layout, so guessing would fetch garbage. Non-decider-2b
    /// registry ids are refused with a pointer at `Decider`.
    pub fn open(
        model_id: &str,
        revision: Option<&str>,
        cache_dir: Option<PathBuf>,
        device: DeviceReq,
        max_length: Option<usize>,
        precision: Precision,
        cpu_arena: bool,
    ) -> Result<Self> {
        let spec = lookup_task(model_id).ok_or_else(|| {
            anyhow!(
                "unknown decider-2b model '{model_id}'; the registered checkpoint is Mapika/decider-2b (explicit files via open_files)"
            )
        })?;
        if spec.kind != TaskKind::Decide2B {
            return Err(anyhow!(
                "'{model_id}' is a {} model, served by Decider/Reranker/LayaDecider — Decider2B serves the slot-layout Mapika/decider-2b checkpoint only",
                spec.kind.as_str()
            ));
        }
        let (_, used_cuda) = resolve_provider_names(device);
        let precision = precision.resolve(used_cuda);
        if precision != Precision::Fp32 {
            eprintln!(
                "embroider: decider-2b ships fp32 weights only — precision '{}' falls back to the fp32 artifact (the documented unlisted-pair rule, not a measured tier)",
                precision.as_str()
            );
        }
        let artifact = artifact_for_task(model_id, precision);

        // ---- acquisition (artifact repo holds all five files) ----
        let onnx_path =
            fetch_repo_file(artifact.repo, artifact.file, revision, cache_dir.clone())?;
        if let Some(sidecar) = artifact.sidecar() {
            match fetch_repo_file(artifact.repo, &sidecar, revision, cache_dir.clone()) {
                Ok(_) => {}
                Err(e) => eprintln!("embroider: no {sidecar} sidecar; assuming inline weights ({e})"),
            }
        }
        let tok_path =
            fetch_repo_file(artifact.repo, "tokenizer.json", revision, cache_dir.clone())?;
        let decision_path =
            fetch_repo_file(artifact.repo, "decision.json", revision, cache_dir.clone())?;
        let calib_path =
            fetch_repo_file(artifact.repo, "calibration.json", revision, cache_dir)?;
        Self::from_files(
            &onnx_path,
            &tok_path,
            &decision_path,
            &calib_path,
            model_id,
            device,
            check_decider2b_limit_for(max_length, spec.max_tokens)?,
            cpu_arena,
        )
    }

    /// Load from explicit local files — no network access. The configs
    /// carry the checkpoint truth (ceilings, label alphabet,
    /// temperatures); they are required, not optional —
    /// caller-supplied temperatures would silently fork the calibration.
    pub fn open_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        decision_path: &Path,
        calibration_path: &Path,
        device: DeviceReq,
        max_length: Option<usize>,
        cpu_arena: bool,
    ) -> Result<Self> {
        // Read the ceiling before the session: a caller cap above the
        // checkpoint context fails here, not pages into a bench.
        let decision_text = std::fs::read_to_string(decision_path).map_err(|e| {
            anyhow!("decider-2b decision.json not found: {}: {e}", decision_path.display())
        })?;
        let decision: serde_json::Value = serde_json::from_str(&decision_text).map_err(|e| {
            anyhow!("decider-2b decision.json parses: {}: {e}", decision_path.display())
        })?;
        let ceiling = decision
            .get("max_ctx_tokens")
            .and_then(|v| v.as_u64())
            .filter(|&v| v > 0)
            .ok_or_else(|| anyhow!("decision.json has no positive 'max_ctx_tokens'"))? as usize;
        let max_length = check_decider2b_limit_for(max_length, ceiling)?;
        Self::from_files(
            onnx_path,
            tokenizer_path,
            decision_path,
            calibration_path,
            &onnx_path.display().to_string(),
            device,
            max_length,
            cpu_arena,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        decision_path: &Path,
        calibration_path: &Path,
        model_id: &str,
        device: DeviceReq,
        max_length: usize,
        cpu_arena: bool,
    ) -> Result<Self> {
        if !onnx_path.is_file() {
            return Err(anyhow!("onnx model not found: {}", onnx_path.display()));
        }
        if !tokenizer_path.is_file() {
            return Err(anyhow!("tokenizer file not found: {}", tokenizer_path.display()));
        }
        let decision_text = std::fs::read_to_string(decision_path).map_err(|e| {
            anyhow!("decider-2b decision.json not found: {}: {e}", decision_path.display())
        })?;
        let decision: serde_json::Value = serde_json::from_str(&decision_text).map_err(|e| {
            anyhow!("decider-2b decision.json parses: {}: {e}", decision_path.display())
        })?;
        let calib_text = std::fs::read_to_string(calibration_path).map_err(|e| {
            anyhow!("decider-2b calibration.json not found: {}: {e}", calibration_path.display())
        })?;
        let calibration: serde_json::Value =
            serde_json::from_str(&calib_text).map_err(|e| {
                anyhow!("decider-2b calibration.json parses: {}: {e}", calibration_path.display())
            })?;
        let cfg = Decider2BConfig::load(&decision, &calibration)?;
        for r in &cfg.temps.rejections {
            eprintln!("embroider: decider-2b temperature clamped [{r}] — treat the bucket as uncalibrated");
        }
        let tok = load_tokenizer(tokenizer_path, None)?;

        // Own session build (not the shared `build_session`): this graph
        // declares (input_ids, slot_pos) with no attention mask, and
        // runs the parallel executor (see the module docs).
        let (provider_names, used_cuda) = resolve_provider_names(device);
        let mut builder = oe(ort::session::Session::builder())?;
        builder = SessionPolicy::decider2b().apply(builder)?;
        builder = apply_providers_with_arena(builder, &provider_names, cpu_arena);
        let session = oe(builder.commit_from_file(onnx_path))
            .map_err(|e| anyhow!("loading {}: {e:#}", onnx_path.display()))?;
        let have: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
        for need in ["input_ids", "slot_pos"] {
            if !have.iter().any(|&n| n == need) {
                return Err(anyhow!(
                    "ONNX export lacks required input '{need}' (has {have:?}) — not a decider-2b slot-layout graph"
                ));
            }
        }
        if !session.outputs().iter().any(|o| o.name() == "label_logits") {
            let outs: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
            return Err(anyhow!(
                "ONNX export lacks required output 'label_logits' (has {outs:?}) — not a decider-2b slot-layout graph"
            ));
        }
        Ok(Self {
            session: std::sync::RwLock::new(session),
            tokenizer: tok,
            cfg,
            max_len: max_length,
            model_id: model_id.to_string(),
            used_cuda,
        })
    }
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn used_cuda(&self) -> bool {
        self.used_cuda
    }

    /// Effective sequence cap.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Checkpoint temperatures after the [0.05, 5.0] clamp.
    pub fn temperatures(&self) -> [f64; 3] {
        self.cfg.temps.triple
    }

    /// Load-time clamp rejections - buckets to treat as uncalibrated.
    pub fn temperature_rejections(&self) -> &[String] {
        &self.cfg.temps.rejections
    }

    /// Token count of a state without specials (the truncation-free
    /// handle, same as `Decider::count_tokens`).
    pub fn count_tokens(&self, state: &str) -> Result<usize> {
        count_tokens_in(&self.tokenizer, state)
    }

    /// Build the input rows for (state, questions) without running
    /// the graph: per question, its rows as `(input_ids, slot, k)`.
    /// The inspection seam - lets the e2e pin the Rust construction
    /// against the reference layout id-for-id.
    pub fn encode_rows(
        &self,
        state: &str,
        questions: &[Decider2BQuestion],
    ) -> Result<Vec<(Vec<u32>, usize, usize)>> {
        for (i, q) in questions.iter().enumerate() {
            validate_decider2b_question(i, q, self.cfg.max_options, self.cfg.max_levels)?;
        }
        let full = self
            .tokenizer
            .encode(format!("Context:\n{state}").as_str(), false)
            .map_err(|e| anyhow!("state encode: {e}"))?;
        let ctx = full.get_ids().to_vec();
        let mut out = Vec::new();
        for q in questions {
            let (new_rows, _) = encode_pair(
                &self.tokenizer,
                &ctx,
                q,
                &self.cfg.label_ids,
                self.max_len,
                self.cfg.isolated_levels,
            )?;
            for nr in new_rows {
                out.push((nr.ids, nr.slot, nr.k));
            }
        }
        Ok(out)
    }

    /// Answer questions over one state - one row per question through
    /// batched session runs, order-preserving.
    pub fn decide(
        &self,
        state: &str,
        questions: &[Decider2BQuestion],
    ) -> Result<Vec<Decider2BDecision>> {
        let refs: Vec<(&str, &Decider2BQuestion)> =
            questions.iter().map(|q| (state, q)).collect();
        self.decide_many(&refs)
    }

    /// Answer (state, question) pairs - rows from every pair share the
    /// batcher, one padded session run per batch, order-preserving.
    pub fn decide_many(
        &self,
        items: &[(&str, &Decider2BQuestion)],
    ) -> Result<Vec<Decider2BDecision>> {
        for (i, (_, q)) in items.iter().enumerate() {
            validate_decider2b_question(i, q, self.cfg.max_options, self.cfg.max_levels)?;
        }
        let ctx_each: Vec<Vec<u32>> = items
            .iter()
            .map(|(state, _)| {
                let full = self
                    .tokenizer
                    .encode(format!("Context:\n{state}").as_str(), false)
                    .map_err(|e| anyhow!("state encode: {e}"))?;
                Ok(full.get_ids().to_vec())
            })
            .collect::<Result<_>>()?;
        // Encode: rows accumulate globally, plan items rebase their row
        // indices onto the shared list.
        struct Row {
            ids: Vec<u32>,
            slot: usize,
        }
        let mut rows: Vec<Row> = Vec::new();
        let mut plans: Vec<(PlanItem, usize)> =
            Vec::with_capacity(items.len());
        for (qi, (_, q)) in items.iter().enumerate() {
            let base = rows.len();
            let (new_rows, mut plan) = encode_pair(
                &self.tokenizer,
                &ctx_each[qi],
                q,
                &self.cfg.label_ids,
                self.max_len,
                self.cfg.isolated_levels,
            )
            .map_err(|e| anyhow!("question {qi}: {e:#}"))?;
            for r in &mut plan.rows {
                *r += base;
            }
            for nr in new_rows {
                rows.push(Row { ids: nr.ids, slot: nr.slot });
            }
            plans.push((plan, qi));
        }

        // Collate: ort_rows packing, right-pad with the pad id (any id
        // reaches no model - causal, padding trails the slot).
        let lens: Vec<usize> = rows.iter().map(|r| r.ids.len()).collect();
        let batches = pack_row_batches(&lens);
        let mut outs: Vec<Vec<f32>> = vec![Vec::new(); rows.len()];
        {
            let mut guard =
                self.session.write().map_err(|_| anyhow!("session lock poisoned"))?;
            for (idx, l) in &batches {
                let mut ids = Vec::with_capacity(idx.len() * l);
                let mut slots = Vec::with_capacity(idx.len());
                for &r in idx {
                    ids.extend(rows[r].ids.iter().map(|&x| x as i64));
                    ids.extend(
                        std::iter::repeat(self.cfg.pad_id as i64)
                            .take(l - rows[r].ids.len()),
                    );
                    slots.push(rows[r].slot as i64);
                }
                let ids = Array2::from_shape_vec((idx.len(), *l), ids)
                    .map_err(|e| anyhow!("input shape: {e}"))?;
                let slots = ndarray::Array1::from_vec(slots);
                let ids_t = oe(ort::value::TensorRef::from_array_view(&ids))?;
                let slots_t = oe(ort::value::TensorRef::from_array_view(&slots))?;
                let outputs = oe(guard.run(ort::inputs! {
                    "input_ids" => ids_t,
                    "slot_pos" => slots_t
                }))?;
                let logits = oe(outputs["label_logits"].try_extract_array::<f32>())?
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| anyhow!("expected [rows, 255] label_logits: {e}"))?;
                if logits.shape() != [idx.len(), 255] {
                    return Err(anyhow!(
                        "label_logits shape {:?} does not match {} rows x 255 labels",
                        logits.shape(),
                        idx.len()
                    ));
                }
                for (b, &r) in idx.iter().enumerate() {
                    outs[r] = logits.row(b).iter().copied().collect();
                }
            }
        }
        let mut out = Vec::with_capacity(items.len());
        for (plan, qi) in &plans {
            let logits = option_logits(plan, &outs, self.cfg.row_temp);
            let (_, q) = items[*qi];
            out.push(decode_decider2b_row(&logits, q, &self.cfg.temps.triple));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_level_number_removes_numeric_prefix() {
        assert_eq!(strip_level_number("0: low"), "low");
        assert_eq!(strip_level_number("  12: answer text "), "answer text ");
        assert_eq!(strip_level_number("-3: neg"), "neg");
        assert_eq!(strip_level_number("plain level"), "plain level");
        assert_eq!(strip_level_number("A: lettered"), "A: lettered");
        assert_eq!(strip_level_number("10x: not a number prefix"), "10x: not a number prefix");
    }

    #[test]
    fn clamp_floor_is_checkpoint_range() {
        // The fitted noul temperature IS 0.05: Laya's 0.5 floor would
        // silently corrupt it. Range from calibration.json.
        assert_eq!(clamp_decider2b_temperature(2.465), 2.465);
        assert_eq!(clamp_decider2b_temperature(0.05), 0.05);
        assert_eq!(clamp_decider2b_temperature(0.049), 0.05);
        assert_eq!(clamp_decider2b_temperature(9.0), 5.0);
        assert_eq!(clamp_decider2b_temperature(f64::NAN), 1.0);
        assert_eq!(clamp_decider2b_temperature(f64::INFINITY), 1.0);
        let raw = serde_json::json!(0.05);
        assert!(!temperature_rejected(&raw, 0.05));
        assert!(temperature_rejected(&raw, 0.5));
    }

    #[test]
    fn batch_packing_matches_ort_rows() {
        // ort_rows: sorted by len, extend while L2*(count) <= 8192.
        assert_eq!(pack_row_batches(&[100, 100, 100]), vec![(vec![0, 1, 2], 128)]);
        // One long row forces its own batch at its own ceiling.
        let got = pack_row_batches(&[100, 9000]);
        assert_eq!(got.len(), 2);
        assert_eq!(got[1], (vec![1], 9024));
        // Full-budget boundary: sorted, packed tight.
        let lens = vec![8000, 100, 100];
        let got = pack_row_batches(&lens);
        assert_eq!(got[0].0, vec![1, 2]);
        assert_eq!(got[0].1, 128);
        assert_eq!(got[1].0, vec![0]);
    }

    #[test]
    fn isolated_option_logits_are_log_sigmoids() {
        // Two levels, row_temp 2.833 (the checkpoint value): z_j =
        // log_sigmoid((yes-no)/T). Softmax over z == normalised p_yes.
        let item = PlanItem { qtype: 1, kind: RowKind::Iso, rows: vec![0, 1], k: 2 };
        let outs = vec![vec![0.0f32; 255], vec![0.0f32; 255]];
        let mut o0 = outs[0].clone();
        let mut o1 = outs[1].clone();
        o0[0] = -1.0;
        o0[1] = 1.0;
        o1[0] = 0.5;
        o1[1] = -0.5;
        let z = option_logits(&item, &[o0, o1], 2.833);
        assert_eq!(z.len(), 2);
        let s0 = 1.0 / (1.0 + (-(2.0f64) / 2.833).exp());
        let s1 = 1.0 / (1.0 + (-(-1.0f64) / 2.833).exp());
        assert!((z[0] - s0.ln()).abs() < 1e-12, "{}", z[0]);
        assert!((z[1] - s1.ln()).abs() < 1e-12, "{}", z[1]);
        let item = PlanItem { qtype: 0, kind: RowKind::List, rows: vec![0], k: 3 };
        let mut oo = vec![0.0f32; 255];
        oo[0] = 1.0;
        oo[1] = 2.0;
        oo[2] = 3.0;
        assert_eq!(option_logits(&item, &[oo], 1.0), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn billing_style_temperature_inversion() {
        // parity3 billing: triple [2.465, 1.0, 0.05]; invert a known
        // distribution through the choice temperature and recover it.
        let t = 2.465f64;
        let want = [0.9231f64, 0.048f64, 0.0289f64];
        let logits: Vec<f64> = want.iter().map(|&p| p.ln() * t).collect();
        let q = Decider2BQuestion::Choice {
            instructions: "route".to_string(),
            options: vec![
                Decider2BOption::plain("billing"),
                Decider2BOption::plain("technical support"),
                Decider2BOption::plain("sales"),
            ],
        };
        let d = decode_decider2b_row(&logits, &q, &[t, 1.0, 0.05]);
        assert_eq!(d.label, "billing");
        for (i, (_, p)) in d.probabilities.iter().enumerate() {
            assert!((*p as f64 - want[i]).abs() < 1e-4, "{p} vs {}", want[i]);
        }
    }

    #[test]
    fn noul_confidence_is_two_sided() {
        let q = Decider2BQuestion::Noul {
            instructions: "holds?".to_string(),
            true_criterion: "yes".to_string(),
            false_criterion: String::new(),
        };
        // Bare false renders "no", true renders "yes: yes".
        assert_eq!(
            render_decider_options(&q),
            vec!["no".to_string(), "yes: yes".to_string()]
        );
        let d = decode_decider2b_row(&[-1.0, 1.0], &q, &[2.465, 1.0, 0.05]);
        assert_eq!(d.label, "true");
        assert!(d.confidence > 0.99, "{}", d.confidence);
    }

    #[test]
    fn validation_fires_before_io() {
        let bad = Decider2BQuestion::Choice {
            instructions: "  ".to_string(),
            options: vec![Decider2BOption::plain("a"), Decider2BOption::plain("b")],
        };
        assert!(validate_decider2b_question(0, &bad, 255, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("must not be empty"));
        let bad = Decider2BQuestion::Choice {
            instructions: "x".to_string(),
            options: vec![Decider2BOption::plain("a")],
        };
        assert!(validate_decider2b_question(1, &bad, 255, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("at least 2"));
        let bad = Decider2BQuestion::Choice {
            instructions: "x".to_string(),
            options: vec![Decider2BOption::plain("a"), Decider2BOption::plain("a")],
        };
        assert!(validate_decider2b_question(2, &bad, 255, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate"));
        let bad = Decider2BQuestion::Score { instructions: "x".to_string(), levels: vec![] };
        assert!(validate_decider2b_question(3, &bad, 255, 10)
            .err()
            .unwrap()
            .to_string()
            .contains("at least 2"));
        let ok = Decider2BQuestion::Noul {
            instructions: "x".to_string(),
            true_criterion: String::new(),
            false_criterion: String::new(),
        };
        assert!(validate_decider2b_question(4, &ok, 255, 10).is_ok());
    }

    #[test]
    fn open_files_rejects_missing_config_before_session() {
        let e = Decider2B::open_files(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
            Path::new("/nonexistent/decision.json"),
            Path::new("/nonexistent/calibration.json"),
            DeviceReq::Cpu,
            None,
            false,
        )
        .err()
        .expect("open_files should fail")
        .to_string();
        assert!(e.contains("decision.json not found"), "{e}");
    }
}
