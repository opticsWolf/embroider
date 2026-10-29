//! Laya typed-decision serving: the marker-layout contract.
//!
//! GLiClass splices options into the token stream (`<<LABEL>>` text, 2
//! inputs). Laya is a different architecture: the state goes in as plain
//! tokens and the questions arrive as **marker positions** — 5 inputs
//! (`input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`),
//! 2 outputs (`logits`, `act_logits`). The decision head gathers hidden
//! states at the marker slots and applies fitted per-bucket temperatures.
//! This module ports `laya.common.build_sequence` (+ head budget,
//! truncation, temperature math) and `laya.onnx_agent.ONNXAgent`'s
//! collation/decode 1:1, so embroider serves the `opticsWolf/laya-*`
//! self-exports bit-faithfully instead of approximating them.
//!
//! Sequence (per question row): `[CLS] <type> question: <ins> [SEP]
//! [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, markers at the
//! `[MASK]` slots, attention 1 over real tokens, `k_cols >= 2` (the
//! exported head takes `topk(2)` — one-option questions still collate
//! two marker columns, the second masked). Temperature:
//! `t_scale = temperature_by_options[temp_bucket(qtype, k)]` with the
//! `[0.5, 5.0]` clamp (rejections recorded, see
//! [`LayaDecider::temperature_rejections`]) — including the
//! `choice:11+` 0.1006 entry the clamp refuses.
//!
//! Deliberately NOT ported (consumer/policy territory, as in laya
//! itself): `predict_long` window scans (the caller windows + aggregates),
//! A consumer-fitted [`BinningMap`] overlay is supported (no checkpoint
//! ships one — see `LayaDecider::load_binning_map`); `lang_temperatures`
//! overrides, hooks/serve/structured are not. Score here is
//! Laya's score — a distribution over LEVELS with an expected value —
//! not GLiClass's multi-label sigmoid: the two must never share a
//! question type, so [`LayaQuestion`] is its own enum.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use ndarray::Array2;
use tokenizers::Tokenizer;

use crate::acquire::{artifact_for_task, fetch_repo_file, lookup_task, TaskKind, LAYA_ABSOLUTE_MAX};
use crate::error::oe;
use crate::jina::{build_session, count_tokens_in, load_tokenizer, special_id};
use crate::policy::{resolve_provider_names, DeviceReq, Precision, SessionPolicy};

/// One Laya option: a label with an optional description (rendered as
/// `label: description`, the description side of laya's criteria dict).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaOption {
    pub label: String,
    pub description: Option<String>,
}

impl LayaOption {
    pub fn plain(label: &str) -> Self {
        Self { label: label.to_string(), description: None }
    }
}

/// A Laya question. Shapes mirror `laya.Agent.system_one`'s qdefs exactly:
/// - `Choice`: one distribution over options (labels + optional
///   descriptions), winner takes all.
/// - `Score`: one distribution over LEVELS (ordered best-to-worst or
///   worst-to-best — the caller defines the order); the answer is the
///   expected level index, the label the top level's text.
/// - `Noul`: P(true) for one statement (the rerank primitive).
/// `option_order` permutes option slots (scored in slot order, reported
/// in caller order); it must be an exact permutation or the question is
/// refused — a silent misindex would attach probabilities to the wrong
/// options.
#[derive(Clone, Debug, PartialEq)]
pub enum LayaQuestion {
    Choice {
        instructions: String,
        options: Vec<LayaOption>,
        option_order: Option<Vec<usize>>,
    },
    Score {
        instructions: String,
        levels: Vec<String>,
    },
    Noul {
        instructions: String,
        statement: String,
    },
}

/// One answered Laya question. Native f32 throughout (no `round(4)` —
/// presentation rounding is the caller's; goldens pin full precision).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaDecision {
    /// Choice: winning label. Score: top level's text. Noul: "true"/"false".
    pub label: String,
    /// Choice: per-option. Score: per-level (level text). Noul:
    /// `[("false", p0), ("true", p1)]`. Caller order (unpermuted).
    pub probabilities: Vec<(String, f32)>,
    /// Score only: the expected level index (`Σ i·pᵢ`).
    pub value: Option<f32>,
    /// Shannon-concentration confidence (1.0 for k < 2 — no entropy scale).
    pub confidence: f32,
    /// Calibrated confidence: max(p) — what the temperatures were fitted on.
    pub answer_confidence: f32,
    /// Escalation head: softmax(act_logits)[0] (feeds escalation, not answers).
    pub act_probability: f32,
}

/// Laya's integer question types (`laya.common.QTYPES`).
fn qtype_of(q: &LayaQuestion) -> usize {
    match q {
        LayaQuestion::Choice { .. } => 0,
        LayaQuestion::Score { .. } => 1,
        LayaQuestion::Noul { .. } => 2,
    }
}

fn qtype_name(qt: usize) -> &'static str {
    match qt {
        0 => "choice",
        1 => "score",
        _ => "noul",
    }
}

/// Bucket key: `{type}:{size}` with sizes 2 / 3-5 / 6-10 / 11+
/// (`laya.common.temp_bucket`).
pub(crate) fn temp_bucket(qtype: usize, k: usize) -> String {
    let size = if k <= 2 { "2" } else if k <= 5 { "3-5" } else if k <= 10 { "6-10" } else { "11+" };
    format!("{}:{size}", qtype_name(qtype))
}

/// Usable-temperature clamp `[0.5, 5.0]`, NaN/inf/non-number → 1.0, bools
/// refused (a bool is not a number — `laya.common.clamp_temperature`).
pub(crate) fn clamp_temperature(t: f64) -> f64 {
    if t.is_nan() || t.is_infinite() {
        return 1.0;
    }
    t.clamp(0.5, 5.0)
}

/// Whether a raw checkpoint entry is not the number the clamp applied
/// (`laya.common.temperature_rejected` — bool/non-number aware).
pub(crate) fn temperature_rejected(raw: &serde_json::Value, applied: f64) -> bool {
    match raw.as_f64() {
        Some(v) => v != applied,
        None => true,
    }
}

/// Render option texts in label-index order (`laya.common.render_options`).
/// Noul is always the [false, true] pair; the statement is the true
/// criterion, the false side the stock denial (an empty statement falls
/// back to the stock affirmation, exactly as laya does).
fn render_laya_options(q: &LayaQuestion) -> Vec<String> {
    match q {
        LayaQuestion::Choice { options, .. } => options
            .iter()
            .map(|o| match &o.description {
                Some(d) if !d.is_empty() => format!("{}: {d}", o.label),
                _ => o.label.clone(),
            })
            .collect(),
        LayaQuestion::Score { levels, .. } => levels
            .iter()
            .enumerate()
            .map(|(i, c)| format!("level {i}: {c}"))
            .collect(),
        LayaQuestion::Noul { statement, .. } => vec![
            "false: no, the statement does not hold".to_string(),
            if statement.is_empty() {
                "true: yes, the statement holds".to_string()
            } else {
                format!("true: {statement}")
            },
        ],
    }
}

/// Reject a question that cannot be answered, naming it (the
/// `Agent._check_question` role, adapted to the list shape: `question
/// {i}`). Validation fires before any tokenizer or session work.
fn validate_laya_question(i: usize, q: &LayaQuestion) -> Result<()> {
    let instructions = match q {
        LayaQuestion::Choice { instructions, .. }
        | LayaQuestion::Score { instructions, .. }
        | LayaQuestion::Noul { instructions, .. } => instructions,
    };
    if instructions.trim().is_empty() {
        return Err(anyhow!(
            "question {i}: 'instructions' must not be empty; add the text the model should answer"
        ));
    }
    match q {
        LayaQuestion::Choice { options, option_order, .. } => {
            if options.is_empty() {
                return Err(anyhow!(
                    "question {i}: a choice question needs at least one option"
                ));
            }
            if options.len() > 256 {
                return Err(anyhow!(
                    "question {i}: {} options exceed the 256 option slots the exported graph bakes in (k≤256)",
                    options.len()
                ));
            }
            let mut seen = std::collections::HashSet::new();
            for o in options {
                if !seen.insert(o.label.clone()) {
                    return Err(anyhow!(
                        "question {i}: duplicate choice label {:?}; labels are the answer keys and must be distinct",
                        o.label
                    ));
                }
            }
            if let Some(order) = option_order {
                let mut sorted = order.clone();
                sorted.sort_unstable();
                let expect: Vec<usize> = (0..options.len()).collect();
                if sorted != expect {
                    return Err(anyhow!(
                        "question {i}: option_order must be an exact permutation of 0..{}, got {order:?}",
                        options.len()
                    ));
                }
            }
        }
        LayaQuestion::Score { levels, .. } => {
            if levels.is_empty() {
                return Err(anyhow!(
                    "question {i}: a score question needs at least one level"
                ));
            }
            if levels.len() > 256 {
                return Err(anyhow!(
                    "question {i}: {} levels exceed the 256 option slots the exported graph bakes in (k≤256)",
                    levels.len()
                ));
            }
        }
        LayaQuestion::Noul { .. } => {}
    }
    Ok(())
}

/// Pure head-budget math (`build_head`'s middle): total option-token
/// lengths + head ceiling → kept lengths, the `per_option` cap (or
/// `None`), and the remaining instruction budget. Factored out so the
/// budget is unit-pinned without a tokenizer.
pub(crate) fn truncate_options_to_budget(
    opt_lens: &[usize],
    head_max_len: usize,
) -> (Vec<usize>, Option<usize>, usize) {
    let mut kept: Vec<usize> = opt_lens.to_vec();
    let mut per_option = None;
    let mut opt_budget = head_max_len.saturating_sub(kept.iter().sum());
    if opt_budget < 16 {
        let per = std::cmp::max(4, head_max_len.saturating_sub(16) / std::cmp::max(1, kept.len()));
        per_option = Some(per);
        for k in kept.iter_mut() {
            *k = (*k).min(per);
        }
        opt_budget = head_max_len.saturating_sub(kept.iter().sum());
    }
    (kept, per_option, opt_budget)
}

/// The question half of a row: `[CLS] <type> question: <ins> [SEP]
/// [MASK] opt0 ... [SEP]` + marker positions (`laya.common.build_head`).
/// Returns head ids, markers (slot order), and the option count.
fn build_laya_head(
    tok: &Tokenizer,
    q: &LayaQuestion,
    cls_id: u32,
    sep_id: u32,
    mask_id: u32,
    mask_token: &str,
    head_max_len: usize,
) -> Result<(Vec<u32>, Vec<usize>, usize)> {
    let qt = qtype_of(q);
    let instructions = match q {
        LayaQuestion::Choice { instructions, .. }
        | LayaQuestion::Score { instructions, .. }
        | LayaQuestion::Noul { instructions, .. } => instructions,
    };
    let ins_clean = instructions.replace(mask_token, " ");
    let head_text = format!("{} question: {ins_clean}", qtype_name(qt));
    let head_ids_full = tok
        .encode(head_text.as_str(), false)
        .map_err(|e| anyhow!("instruction encode: {e}"))?;
    let head_ids_full = head_ids_full.get_ids();

    let rendered = render_laya_options(q);
    let order: Vec<usize> = match q {
        LayaQuestion::Choice { options, option_order, .. } => match option_order {
            Some(o) => o.clone(),
            None => (0..options.len()).collect(),
        },
        _ => (0..rendered.len()).collect(),
    };
    // Cap at the tokenizer: first 48 tokens, exactly the `truncation=True,
    // max_length=48` the agent passes (no stride involved).
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(order.len());
    for s in &order {
        let text = format!(" {}", rendered[*s].replace(mask_token, " "));
        let enc = tok.encode(text.as_str(), false).map_err(|e| anyhow!("option encode: {e}"))?;
        let ids: Vec<u32> = enc.get_ids().iter().take(48).copied().collect();
        let mut with_marker = Vec::with_capacity(ids.len() + 1);
        with_marker.push(mask_id);
        with_marker.extend(ids);
        opt_ids.push(with_marker);
    }
    let lens: Vec<usize> = opt_ids.iter().map(|o| o.len()).collect();
    let (kept, _per_option, opt_budget) = truncate_options_to_budget(&lens, head_max_len);
    for (o, k) in opt_ids.iter_mut().zip(kept.iter()) {
        o.truncate(*k);
    }
    let head_keep = std::cmp::max(8, opt_budget).min(head_ids_full.len());
    let mut ids = Vec::with_capacity(2 + head_keep + opt_ids.iter().map(|o| o.len()).sum::<usize>() + 1);
    ids.push(cls_id);
    ids.extend_from_slice(&head_ids_full[..head_keep]);
    ids.push(sep_id);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for o in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(o);
    }
    ids.push(sep_id);
    Ok((ids, markers, rendered.len()))
}

/// One full input row: head + state + `[SEP]`, state truncated from the
/// right to the room the head leaves (`laya.common.build_sequence` with
/// `truncate_left=false` — states here are strings, never turn lists).
/// Markers past `max_len` are dropped; losing any is an error naming both
/// knobs (the agent's message, verbatim in spirit).
fn build_laya_sequence(
    head_ids: &[u32],
    markers: &[usize],
    state_ids: &[u32],
    sep_id: u32,
    max_len: usize,
    head_max_len: usize,
    n_opts: usize,
    qi: usize,
) -> Result<(Vec<u32>, Vec<usize>)> {
    let room = max_len.saturating_sub(head_ids.len() + 1);
    let take = room.min(state_ids.len());
    let mut ids = Vec::with_capacity(head_ids.len() + take + 1);
    ids.extend_from_slice(head_ids);
    ids.extend_from_slice(&state_ids[..take]);
    ids.push(sep_id);
    ids.truncate(max_len);
    let kept: Vec<usize> = markers.iter().copied().filter(|&m| m < max_len).collect();
    if kept.len() != n_opts {
        return Err(anyhow!(
            "question {qi}: only {} of its {n_opts} option markers fit in max_len={max_len} with \
             head_max_len={head_max_len} spent on the question; lower head_max_len, raise max_len, \
             or use fewer options",
            kept.len()
        ));
    }
    Ok((ids, kept))
}

/// Stable softmax in f64 (the numpy the agent divides in is float64 —
/// keep the port on the same arithmetic, cast at the boundary).
pub(crate) fn softmax_stable(xs: &[f64]) -> Vec<f64> {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let es: Vec<f64> = xs.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = es.iter().sum();
    es.iter().map(|&e| e / s).collect()
}

/// Normalized Shannon-concentration confidence
/// (`laya.common.confidence_from_probs`): k < 2 has no entropy scale and
/// reads 1.0 — maximum concentration, not a calibrated claim.
pub(crate) fn confidence_from_probs(p: &[f64]) -> f64 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f64 = p
        .iter()
        .map(|&x| {
            let c = x.clamp(1e-12, 1.0);
            -(c * c.ln())
        })
        .sum();
    let score = 1.0 - ent / (k as f64).ln();
    score.clamp(0.0, 1.0)
}

/// Calibrated confidence: max(p) — what the temperatures fit
/// (`laya.common.answer_confidence`).
pub(crate) fn answer_confidence(p: &[f64]) -> f64 {
    if p.is_empty() {
        return 1.0;
    }
    let m = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m.clamp(0.0, 1.0)
}

/// Checkpoint temperatures as loaded: clamped triple + clamped
/// per-bucket map + the rejection list (the agent's load-time warnings,
/// queryable instead of printed).
#[derive(Clone, Debug)]
pub struct LayaTemperatures {
    pub triple: [f64; 3],
    pub by_options: HashMap<String, f64>,
    pub rejections: Vec<String>,
}

impl LayaTemperatures {
    fn load(cfg: &serde_json::Value) -> Result<Self> {
        let raw_triple = cfg.get("temperature").and_then(|v| v.as_array()).ok_or_else(|| {
            anyhow!("rl_agent_config.json has no 'temperature' triple — not a Laya checkpoint")
        })?;
        if raw_triple.len() != 3 {
            return Err(anyhow!(
                "rl_agent_config.json 'temperature' must have 3 entries, got {}",
                raw_triple.len()
            ));
        }
        let mut triple = [1.0f64; 3];
        let mut rejections = Vec::new();
        for (i, raw) in raw_triple.iter().enumerate() {
            let applied = clamp_temperature(raw.as_f64().unwrap_or(f64::NAN));
            triple[i] = applied;
            if temperature_rejected(raw, applied) {
                rejections.push(format!("temperature[{i}]={raw} -> {applied}"));
            }
        }
        let mut by_options = HashMap::new();
        if let Some(map) = cfg.get("temperature_by_options").and_then(|v| v.as_object()) {
            for (k, raw) in map {
                let applied = clamp_temperature(raw.as_f64().unwrap_or(f64::NAN));
                if temperature_rejected(raw, applied) {
                    rejections.push(format!("{k}={raw} -> {applied}"));
                }
                by_options.insert(k.clone(), applied);
            }
        }
        Ok(Self { triple, by_options, rejections })
    }

    fn scale(&self, qtype: usize, k: usize) -> f64 {
        let bucket = temp_bucket(qtype, k);
        self.by_options.get(&bucket).copied().unwrap_or(self.triple[qtype])
    }
}

/// Recalibrate one `answer_confidence` through a fitted binning map
/// (`laya.calibrate.apply_binning_map`): full pass-through when the map
/// has no entry for the bucket, so a bucket the map was not fit for
/// keeps its temperature-scaled value instead of a wrong one.
pub(crate) fn apply_binning_map(confidence: f64, bucket: &str, map: &BinningMap) -> f64 {
    let entry = match map.get(bucket) {
        Some(e) => e,
        None => return confidence,
    };
    let b = (confidence * entry.bins as f64) as usize;
    entry.values[b.min(entry.bins - 1)]
}

/// A fitted histogram-binning overlay: bucket key (`temp_bucket` form)
/// → bin count + per-bin recalibrated confidences.
#[derive(Clone, Debug, Default)]
pub struct BinningMap {
    entries: HashMap<String, BinEntry>,
}

#[derive(Clone, Debug)]
struct BinEntry {
    bins: usize,
    values: Vec<f64>,
}

impl BinningMap {
    /// Parse a binning map from JSON: either the bare map
    /// (`{"<bucket>": {"bins": n, "values": [...]}}`) or a laya
    /// calibration payload carrying `binning_map` (temperatures in the
    /// payload are ignored — checkpoint truth comes from
    /// `rl_agent_config.json`). Shape-checked before anything loads:
    /// `bins` must be an integer ≥ 1 (bools refused — `true` is not 1),
    /// `values` exactly `bins` finite numbers in [0, 1]. A bad *binning
    /// value* is refused, never clamped — nothing recalibrates a
    /// confidence the way an out-of-range value would.
    pub fn parse(json: &serde_json::Value) -> Result<Self> {
        let obj = json.as_object().ok_or_else(|| {
            anyhow!("binning map JSON must be an object")
        })?;
        let map_obj = match obj.get("binning_map") {
            Some(nested) => nested.as_object().ok_or_else(|| {
                anyhow!("calibration payload 'binning_map' must be an object")
            })?,
            None => obj,
        };
        let mut entries = HashMap::new();
        for (bucket, entry) in map_obj {
            let eobj = entry.as_object().ok_or_else(|| {
                anyhow!("binning entry {bucket:?} must be an object")
            })?;
            let bins_v = eobj.get("bins").ok_or_else(|| {
                anyhow!("binning entry {bucket:?} has no 'bins'")
            })?;
            // Bool is not an integer here: `bins: true` used to install
            // a 1-bin map in silence.
            let bins = match bins_v {
                serde_json::Value::Number(n) => n.as_u64().filter(|&b| b >= 1).ok_or_else(|| {
                    anyhow!("binning entry {bucket:?} 'bins' must be an integer >= 1")
                })? as usize,
                _ => {
                    return Err(anyhow!(
                        "binning entry {bucket:?} 'bins' must be an integer >= 1"
                    ))
                }
            };
            let values_v = eobj.get("values").ok_or_else(|| {
                anyhow!("binning entry {bucket:?} has no 'values'")
            })?;
            let arr = values_v.as_array().ok_or_else(|| {
                anyhow!("binning entry {bucket:?} 'values' must be a list")
            })?;
            if arr.len() != bins {
                return Err(anyhow!(
                    "binning entry {bucket:?} has {} values for {bins} bins",
                    arr.len()
                ));
            }
            let mut values = Vec::with_capacity(bins);
            for v in arr {
                let f = v.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                    anyhow!("binning entry {bucket:?} values must be finite numbers")
                })?;
                if !(0.0..=1.0).contains(&f) {
                    return Err(anyhow!(
                        "binning entry {bucket:?} value {f} outside [0, 1] — refused, not clamped"
                    ));
                }
                values.push(f);
            }
            entries.insert(bucket.clone(), BinEntry { bins, values });
        }
        Ok(Self { entries })
    }

    fn get(&self, bucket: &str) -> Option<&BinEntry> {
        self.entries.get(bucket)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn buckets(&self) -> Vec<String> {
        let mut v: Vec<String> = self.entries.keys().cloned().collect();
        v.sort();
        v
    }
}

/// Sliding-window spans over an `n`-token state: `size`-token windows
/// stepping by `step`, budged to the token stream (`(i, min(i+size, n))`)
/// — the `Agent.predict_long` loop exactly, including its short final
/// window. Span-identity with the agent is what lets a golden pin the
/// scan; do not "improve" this into tail-aligned windows without
/// re-pinning. Contract: `1 <= step <= size` (enforced by
/// `predict_long` before calling).
pub(crate) fn plan_laya_spans(n: usize, size: usize, step: usize) -> Vec<(usize, usize)> {
    debug_assert!(step >= 1 && step <= size && size >= 1);
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        out.push((i, (i + size).min(n)));
        if i + size >= n {
            break;
        }
        i += step;
    }
    out
}

/// Which window decides a question: noul reads P(true) — evidence
/// anywhere wins; choice/score read `answer_confidence` so a localized
/// signal is not out-voted by neutral windows. First window wins ties
/// (strict `>`, matching Python `max` — `max_by` would take the last).
/// `stats` is per-window `(answer_confidence, p_true)`.
pub(crate) fn select_laya_window(qtype: usize, stats: &[(f64, f64)]) -> usize {
    let mut best = 0;
    for (i, &(conf, p_true)) in stats.iter().enumerate() {
        let score = if qtype == 2 { p_true } else { conf };
        let best_score = if qtype == 2 { stats[best].1 } else { stats[best].0 };
        if score > best_score {
            best = i;
        }
    }
    best
}

/// One long-document answer: the deciding window's decision plus the
/// token span that decided it. The probability is that window's, not a
/// calibrated whole-document number (same caveat as laya).
#[derive(Clone, Debug)]
pub struct LayaLongDecision {
    pub decision: LayaDecision,
    /// Token span into the tokenized state (no specials).
    pub window_start: usize,
    pub window_end: usize,
    /// How many windows the scan ran.
    pub windows: usize,
}

/// Decode one model row into an answer (the `_decode_answers` core:
/// temperature → softmax → slot→caller unpermute → per-type shape).
/// `logits`/`act` are the row's first-k / 2-wide slices; `order` the slot
/// permutation (None = identity).
pub(crate) fn decode_laya_row(
    logits: &[f32],
    act_logits: &[f32],
    q: &LayaQuestion,
    order: Option<&[usize]>,
    temps: &LayaTemperatures,
    binning: Option<&BinningMap>,
) -> LayaDecision {
    let qt = qtype_of(q);
    let k = logits.len();
    let t = temps.scale(qt, k);
    let z: Vec<f64> = logits.iter().map(|&x| (x as f64) / t).collect();
    let mut p = softmax_stable(&z);
    if let Some(ord) = order {
        if ord.len() == p.len() {
            let mut canon = vec![0.0; p.len()];
            for (s, &o) in ord.iter().enumerate() {
                canon[o] = p[s];
            }
            p = canon;
        }
    }
    let pf: Vec<f32> = p.iter().map(|&x| x as f32).collect();
    // Noul reports max(P(true), P(false)) — the two-sided certainty —
    // not the entropy concentration (the agent's noul branch never
    // calls `confidence_from_probs`).
    let confidence = if qt == 2 {
        p[1].max(1.0 - p[1]) as f32
    } else {
        confidence_from_probs(&p) as f32
    };
    // The runtime bins before anyone reads `answer_confidence`: a
    // fitted map remaps the max(p) here, an absent bucket passes
    // through untouched.
    let raw_conf = answer_confidence(&p);
    let bucket = temp_bucket(qt, k);
    let answer_confidence = match binning {
        Some(m) => apply_binning_map(raw_conf, &bucket, m) as f32,
        None => raw_conf as f32,
    };
    let act = softmax_stable(&[act_logits[0] as f64, act_logits[1] as f64])[0] as f32;

    match q {
        LayaQuestion::Choice { options, .. } => {
            let win = pf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            LayaDecision {
                label: options.get(win).map(|o| o.label.clone()).unwrap_or_default(),
                probabilities: options.iter().zip(pf.iter()).map(|(o, &v)| (o.label.clone(), v)).collect(),
                value: None,
                confidence,
                answer_confidence,
                act_probability: act,
            }
        }
        LayaQuestion::Score { levels, .. } => {
            let value: f64 = p.iter().enumerate().map(|(i, &x)| i as f64 * x).sum();
            let win = pf
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            LayaDecision {
                label: levels.get(win).cloned().unwrap_or_default(),
                probabilities: levels.iter().zip(pf.iter()).map(|(l, &v)| (l.clone(), v)).collect(),
                value: Some(value as f32),
                confidence,
                answer_confidence,
                act_probability: act,
            }
        }
        LayaQuestion::Noul { .. } => {
            let label = if p[1] >= p[0] { "true" } else { "false" }.to_string();
            LayaDecision {
                label,
                probabilities: vec![("false".to_string(), pf[0]), ("true".to_string(), pf[1])],
                value: None,
                confidence,
                answer_confidence,
                act_probability: act,
            }
        }
    }
}

fn check_laya_limit_for(max_length: Option<usize>, ceiling: usize) -> Result<usize> {
    match max_length {
        None => Ok(ceiling),
        Some(n) if (1..=LAYA_ABSOLUTE_MAX).contains(&n) && n <= ceiling => Ok(n),
        Some(n) => Err(anyhow!(
            "max_length must be within 1..={ceiling} for this checkpoint (traced bound {LAYA_ABSOLUTE_MAX}), got {n}"
        )),
    }
}

pub struct LayaDecider {
    session: std::sync::RwLock<ort::session::Session>,
    tokenizer: Tokenizer,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    mask_id: u32,
    max_len: usize,
    head_max_len: usize,
    model_id: String,
    temps: LayaTemperatures,
    binning: Option<BinningMap>,
    used_cuda: bool,
}

impl LayaDecider {
    /// Open a registered Laya checkpoint (`convaiinnovations/laya-*` ids,
    /// served from the `opticsWolf/laya-*-fp32-onnx` self-exports).
    /// Fetches graph + sidecar + tokenizer + `rl_agent_config.json` (the
    /// checkpoint truth: ceilings, temperatures, layout) from the
    /// artifact repo. Unknown ids are refused — Laya exports don't
    /// follow the legacy file layout, so guessing would fetch garbage.
    /// Non-Laya registry ids are refused with a pointer at `Decider`.
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
                "unknown Laya model '{model_id}'; registered checkpoints are convaiinnovations/laya-typed-decisions, \
                 convaiinnovations/laya and convaiinnovations/laya-multilingual (explicit files via open_files)"
            )
        })?;
        if spec.kind != TaskKind::DecideLaya {
            return Err(anyhow!(
                "'{model_id}' is a {} model, served by Decider/Reranker — LayaDecider serves the marker-layout \
                 convaiinnovations/laya-* checkpoints only",
                spec.kind.as_str()
            ));
        }
        let ceiling = spec.max_tokens.min(LAYA_ABSOLUTE_MAX);
        let max_length = check_laya_limit_for(max_length, ceiling)?;

        let (_, used_cuda) = resolve_provider_names(device);
        let precision = precision.resolve(used_cuda);
        if precision == Precision::Fp16 && !used_cuda {
            eprintln!(
                "embroider: FP16 weights on CPU — this runs >40x slower than FP32-CPU \
                 (emulated half-precision kernels). Explicit request honoured; pass \
                 precision='fp32' (or 'auto') for CPU sessions."
            );
        }
        let artifact = artifact_for_task(model_id, precision);

        // ---- acquisition (artifact repo holds all four files) ----
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
        let cfg_path =
            fetch_repo_file(artifact.repo, "rl_agent_config.json", revision, cache_dir)?;
        Self::from_files(&onnx_path, &tok_path, &cfg_path, model_id, device, max_length, cpu_arena)
    }

    /// Load from explicit local files — no network access. The config
    /// carries the checkpoint truth (ceilings, temperatures, layout
    /// check); it is required, not optional — caller-supplied
    /// temperatures would silently fork the calibration.
    pub fn open_files(
        onnx_path: &Path,
        tokenizer_path: &Path,
        config_path: &Path,
        device: DeviceReq,
        max_length: Option<usize>,
        cpu_arena: bool,
    ) -> Result<Self> {
        let cfg_text = std::fs::read_to_string(config_path)
            .map_err(|e| anyhow!("laya config not found: {}: {e}", config_path.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)
            .map_err(|e| anyhow!("laya config parses: {}: {e}", config_path.display()))?;
        let cfg_max = cfg
            .get("max_len")
            .and_then(|v| v.as_u64())
            .filter(|&v| v > 0)
            .ok_or_else(|| anyhow!("laya config has no positive 'max_len'"))? as usize;
        let ceiling = cfg_max.min(LAYA_ABSOLUTE_MAX);
        let max_length = check_laya_limit_for(max_length, ceiling)?;
        Self::from_files(
            onnx_path,
            tokenizer_path,
            config_path,
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
        config_path: &Path,
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
        let cfg_text = std::fs::read_to_string(config_path)
            .map_err(|e| anyhow!("laya config not found: {}: {e}", config_path.display()))?;
        let cfg: serde_json::Value = serde_json::from_str(&cfg_text)
            .map_err(|e| anyhow!("laya config parses: {}: {e}", config_path.display()))?;
        match cfg.get("option_layout").and_then(|v| v.as_str()) {
            None | Some("sequential") => {}
            Some(other) => {
                return Err(anyhow!(
                    "option_layout={other:?} is not supported by the ONNX runtime; load this checkpoint with laya.Agent"
                ))
            }
        }
        let head_max_len = cfg
            .get("head_max_len")
            .and_then(|v| v.as_u64())
            .filter(|&v| v > 0)
            .ok_or_else(|| anyhow!("laya config has no positive 'head_max_len'"))? as usize;
        let temps = LayaTemperatures::load(&cfg)?;
        for r in &temps.rejections {
            eprintln!("embroider: laya temperature clamped [{r}] — treat the bucket as uncalibrated");
        }

        let tok = load_tokenizer(tokenizer_path, None)?;
        let cls_id = special_id(&tok, "[CLS]")?;
        let sep_id = special_id(&tok, "[SEP]")?;
        let pad_id = special_id(&tok, "[PAD]").unwrap_or(0);
        let mask_id = special_id(&tok, "[MASK]")
            .map_err(|_| anyhow!("tokenizer has no [MASK] — the marker layout needs a mask token"))?;
        let loaded = build_session(onnx_path, device, &SessionPolicy::laya(), cpu_arena)?;
        for need in ["marker_pos", "marker_mask", "qtype"] {
            if !loaded.session.inputs().iter().any(|o| o.name() == need) {
                let have: Vec<&str> = loaded.session.inputs().iter().map(|o| o.name()).collect();
                return Err(anyhow!(
                    "ONNX export lacks required input '{need}' (has {have:?}) — not a Laya marker-layout graph"
                ));
            }
        }
        Ok(Self {
            session: std::sync::RwLock::new(loaded.session),
            tokenizer: tok,
            cls_id,
            sep_id,
            pad_id,
            mask_id,
            max_len: max_length,
            head_max_len,
            model_id: model_id.to_string(),
            temps,
            binning: None,
            used_cuda: loaded.used_cuda,
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

    /// Checkpoint temperatures after the [0.5, 5.0] clamp.
    pub fn temperatures(&self) -> [f64; 3] {
        self.temps.triple
    }

    /// Clamped per-bucket entries the checkpoint ships.
    pub fn temperature_by_options(&self) -> Vec<(String, f64)> {
        let mut v: Vec<(String, f64)> =
            self.temps.by_options.iter().map(|(k, &t)| (k.clone(), t)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    /// Load-time clamp rejections (`choice:11+` et al) — buckets to
    /// treat as uncalibrated.
    pub fn temperature_rejections(&self) -> &[String] {
        &self.temps.rejections
    }

    /// Token count of a state without specials (the truncation-free
    /// handle, same as `Decider::count_tokens`).
    pub fn count_tokens(&self, state: &str) -> Result<usize> {
        count_tokens_in(&self.tokenizer, state)
    }

    /// Install a fitted histogram-binning overlay onto `answer_confidence`
    /// (the `binning_map` role). No checkpoint ships one — the map is
    /// fitted by the consumer on held-out data for its own buckets, then
    /// loaded here so every later `decide` reports binned `answer_conf`.
    /// Buckets the map does not cover pass through untouched; the entropy
    /// `confidence` is never binned (same as laya — the bins are fit on
    /// max(p), not on entropy). Replaces any previously installed map.
    pub fn load_binning_map(&mut self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("binning map not found: {}: {e}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| anyhow!("binning map parses: {}: {e}", path.display()))?;
        let map = BinningMap::parse(&json)?;
        if map.is_empty() {
            return Err(anyhow!("binning map at {} is empty — nothing to install", path.display()));
        }
        self.binning = Some(map);
        Ok(())
    }

    /// Whether a binning overlay is installed.
    pub fn has_binning_map(&self) -> bool {
        self.binning.is_some()
    }

    /// Build one input row: `(input_ids, marker_positions)`.
    /// The inspection seam — lets callers (and the e2e) verify the Rust
    /// construction against the agent's row without running the graph.
    pub fn encode_row(&self, state: &str, question: &LayaQuestion) -> Result<(Vec<u32>, Vec<usize>)> {
        let (head_ids, markers, n_opts) = build_laya_head(
            &self.tokenizer,
            question,
            self.cls_id,
            self.sep_id,
            self.mask_id,
            "[MASK]",
            self.head_max_len,
        )?;
        let state_ids = self
            .tokenizer
            .encode(state.replace("[MASK]", " ").as_str(), false)
            .map_err(|e| anyhow!("state encode: {e}"))?;
        build_laya_sequence(
            &head_ids,
            &markers,
            state_ids.get_ids(),
            self.sep_id,
            self.max_len,
            self.head_max_len,
            n_opts,
            0,
        )
    }

    /// Answer questions over one state — one padded session run for the
    /// whole batch, order-preserving.
    pub fn decide(&self, state: &str, questions: &[LayaQuestion]) -> Result<Vec<LayaDecision>> {
        let refs: Vec<(&str, &LayaQuestion)> = questions.iter().map(|q| (state, q)).collect();
        self.decide_many(&refs)
    }

    /// Answer (state, question) pairs — one padded session run.
    pub fn decide_many(&self, items: &[(&str, &LayaQuestion)]) -> Result<Vec<LayaDecision>> {
        for (i, (_, q)) in items.iter().enumerate() {
            validate_laya_question(i, q)?;
        }
        // Encode the shared head once per question shape, then per state.
        struct Row {
            ids: Vec<u32>,
            markers: Vec<usize>,
            qtype: i64,
            order: Option<Vec<usize>>,
            qi: usize,
        }
        let mut rows: Vec<Row> = Vec::with_capacity(items.len());
        for (qi, (state, q)) in items.iter().enumerate() {
            let (ids, kept) = self.encode_row(state, q).map_err(|e| {
                anyhow!("question {qi}: {e:#}")
            })?;
            let order = match q {
                LayaQuestion::Choice { option_order, .. } => option_order.clone(),
                _ => None,
            };
            rows.push(Row { ids, markers: kept, qtype: qtype_of(q) as i64, order, qi });
        }

        // Collate: pad ids, marker columns to max(kmax, 2) — the topk(2)
        // floor the exported head bakes in.
        let n = rows.len();
        let l = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let kmax = rows.iter().map(|r| r.markers.len()).max().unwrap_or(0);
        let k_cols = kmax.max(2);
        let mut ids = Vec::with_capacity(n * l);
        let mut att = Vec::with_capacity(n * l);
        let mut mpos = Vec::with_capacity(n * k_cols);
        let mut mmask = Vec::with_capacity(n * k_cols);
        let mut qtypes = Vec::with_capacity(n);
        for r in &rows {
            ids.extend(r.ids.iter().map(|&x| x as i64));
            ids.extend(std::iter::repeat(self.pad_id as i64).take(l - r.ids.len()));
            att.extend(std::iter::repeat(1i64).take(r.ids.len()));
            att.extend(std::iter::repeat(0i64).take(l - r.ids.len()));
            mpos.extend(r.markers.iter().map(|&m| m as i64));
            mpos.extend(std::iter::repeat(0i64).take(k_cols - r.markers.len()));
            mmask.extend(std::iter::repeat(true).take(r.markers.len()));
            mmask.extend(std::iter::repeat(false).take(k_cols - r.markers.len()));
            qtypes.push(r.qtype);
        }
        let ids = Array2::from_shape_vec((n, l), ids).map_err(|e| anyhow!("input shape: {e}"))?;
        let att = Array2::from_shape_vec((n, l), att).map_err(|e| anyhow!("mask shape: {e}"))?;
        let mpos =
            Array2::from_shape_vec((n, k_cols), mpos).map_err(|e| anyhow!("marker shape: {e}"))?;
        let mmask =
            Array2::from_shape_vec((n, k_cols), mmask).map_err(|e| anyhow!("marker mask shape: {e}"))?;
        let qtypes = ndarray::Array1::from_vec(qtypes);

        let (logits, act) = {
            let mut guard = self.session.write().map_err(|_| anyhow!("session lock poisoned"))?;
            let ids_t = oe(ort::value::TensorRef::from_array_view(&ids))?;
            let att_t = oe(ort::value::TensorRef::from_array_view(&att))?;
            let mpos_t = oe(ort::value::TensorRef::from_array_view(&mpos))?;
            let mmask_t = oe(ort::value::TensorRef::from_array_view(&mmask))?;
            let qt_t = oe(ort::value::TensorRef::from_array_view(&qtypes))?;
            let outputs = oe(guard.run(ort::inputs! {
                "input_ids" => ids_t,
                "attention_mask" => att_t,
                "marker_pos" => mpos_t,
                "marker_mask" => mmask_t,
                "qtype" => qt_t
            }))?;
            let logits = oe(outputs["logits"].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow!("expected [B,K] logits: {e}"))?;
            let act = oe(outputs["act_logits"].try_extract_array::<f32>())?
                .to_owned()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| anyhow!("expected [B,2] act_logits: {e}"))?;
            (logits, act)
        };
        if logits.shape() != [n, k_cols] {
            return Err(anyhow!(
                "logits shape {:?} does not match {n} rows x {k_cols} marker columns",
                logits.shape()
            ));
        }
        if act.shape() != [n, 2] {
            return Err(anyhow!("act_logits shape {:?} does not match {n} rows", act.shape()));
        }
        let mut out = Vec::with_capacity(n);
        for (r, row) in rows.iter().enumerate() {
            let k = row.markers.len();
            let lrow: Vec<f32> = logits.row(r).iter().take(k).copied().collect();
            let arow: Vec<f32> = act.row(r).iter().copied().collect();
            let (_, q) = items[row.qi];
            out.push(decode_laya_row(&lrow, &arow, q, row.order.as_deref(), &self.temps, self.binning.as_ref()));
        }
        Ok(out)
    }

    /// Answer questions over a state longer than the sequence cap: slide
    /// token windows over the state, score every window through one
    /// shared [`LayaDecider::decide_many`] run, and aggregate per
    /// question — noul takes the strongest window's P(true) (evidence
    /// anywhere), choice/score take the most-confident window's answer
    /// (a localized signal out-votes neutral windows). The returned
    /// probability is the deciding window's, not a calibrated
    /// whole-document number.
    ///
    /// `window`/`stride` adapt the scan to the checkpoint: the default
    /// window is the state budget the config leaves
    /// (`max(64, max_len - head_max_len - 8)`, so a 512-ctxt base scans
    /// narrower than the 1024-ctxt checkpoints) clamped to the room the
    /// questions leave, with 50%-overlap steps by default. A window
    /// wider than the room is clamped with a warning (its tail would
    /// reach no model); a stride past the effective window fails loudly
    /// (tokens between windows would be read by none). A state that
    /// fits the window takes the fast path — one plain `decide`,
    /// identical output, `windows: 1`.
    pub fn predict_long(
        &self,
        state: &str,
        questions: &[LayaQuestion],
        window: Option<usize>,
        stride: Option<usize>,
    ) -> Result<Vec<LayaLongDecision>> {
        for (i, q) in questions.iter().enumerate() {
            validate_laya_question(i, q)?;
        }
        if questions.is_empty() {
            return Ok(vec![]);
        }
        let state_ids = self
            .tokenizer
            .encode(state.replace("[MASK]", " ").as_str(), false)
            .map_err(|e| anyhow!("state encode: {e}"))?;
        let state_ids = state_ids.get_ids();
        // The room is the tightest question's: windows are one shared
        // row list, so a window sized for the roomiest question would be
        // cut short for the tightest one.
        let mut room = usize::MAX;
        for q in questions {
            let (head_ids, _, _) = build_laya_head(
                &self.tokenizer,
                q,
                self.cls_id,
                self.sep_id,
                self.mask_id,
                "[MASK]",
                self.head_max_len,
            )?;
            room = room.min(self.max_len.saturating_sub(head_ids.len() + 1));
        }
        if room == 0 {
            return Err(anyhow!(
                "predict_long: the questions' options fill the whole sequence (max_len={}, \
                 head_max_len={}), leaving no room for the state; no window can carry any of it",
                self.max_len, self.head_max_len
            ));
        }
        let requested = match window {
            Some(n) if n >= 1 => n,
            _ => std::cmp::max(64, self.max_len.saturating_sub(self.head_max_len + 8)),
        };
        let mut size = requested.min(room);
        if size < requested {
            eprintln!(
                "embroider: predict_long window={requested} is wider than the {room} state tokens these \
                 questions leave inside max_len={}, so every window would be truncated on the way to the \
                 model; scanning with window={size} instead",
                self.max_len
            );
        }
        if size == 0 {
            size = 1;
        }
        let mut step = match stride {
            Some(0) => return Err(anyhow!("stride must be >= 1, got 0")),
            Some(n) => n,
            None => std::cmp::max(1, size / 2),
        };
        if step > size {
            if size < requested && stride.map(|s| s <= requested).unwrap_or(false) {
                // The caller's stride fit the window they asked for — the
                // clamp moved the goalposts, so the stride follows it
                // instead of failing on a self-consistent pair.
                eprintln!(
                    "embroider: predict_long stride was a 50% step for the window={requested} asked for, but \
                     the window was reduced to {size} to fit the room; scanning with stride={} instead",
                    std::cmp::max(1, size / 2)
                );
                step = std::cmp::max(1, size / 2);
            } else {
                return Err(anyhow!(
                    "predict_long: stride={step} steps past the {size}-token window, so {} tokens between every \
                     pair of windows would be read by no window at all; pass stride <= {size}",
                    step - size
                ));
            }
        }
        // Fast path: the whole state fits the window, so `decide` reads
        // it whole — identical output, one window counted.
        if state_ids.len() <= size {
            let decisions = self.decide(state, questions)?;
            return Ok(decisions
                .into_iter()
                .map(|decision| LayaLongDecision {
                    decision,
                    window_start: 0,
                    window_end: state_ids.len(),
                    windows: 1,
                })
                .collect());
        }
        let spans = plan_laya_spans(state_ids.len(), size, step);
        // Decode–re-encode round-trips exactly; every chunk flows through
        // the standard row seam, so each window is scored identically to
        // a direct `decide` on the same text.
        let mut chunks: Vec<String> = Vec::with_capacity(spans.len());
        for &(s, e) in &spans {
            chunks.push(
                self.tokenizer
                    .decode(&state_ids[s..e], true)
                    .map_err(|e| anyhow!("chunk decode failed: {e}"))?,
            );
        }
        let mut items: Vec<(&str, &LayaQuestion)> = Vec::with_capacity(spans.len() * questions.len());
        for c in &chunks {
            for q in questions {
                items.push((c.as_str(), q));
            }
        }
        let flat = self.decide_many(&items)?;
        let nq = questions.len();
        let mut out = Vec::with_capacity(nq);
        for (qi, q) in questions.iter().enumerate() {
            let stats: Vec<(f64, f64)> = spans
                .iter()
                .enumerate()
                .map(|(w, _)| {
                    let d = &flat[w * nq + qi];
                    let p_true = d
                        .probabilities
                        .iter()
                        .find(|(o, _)| o == "true")
                        .map(|(_, p)| *p as f64)
                        .unwrap_or(0.0);
                    (d.answer_confidence as f64, p_true)
                })
                .collect();
            let win = select_laya_window(qtype_of(q), &stats);
            let (ws, we) = spans[win];
            out.push(LayaLongDecision {
                decision: flat[win * nq + qi].clone(),
                window_start: ws,
                window_end: we,
                windows: spans.len(),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_keys_match_laya() {
        assert_eq!(temp_bucket(0, 1), "choice:2");
        assert_eq!(temp_bucket(0, 2), "choice:2");
        assert_eq!(temp_bucket(0, 3), "choice:3-5");
        assert_eq!(temp_bucket(0, 5), "choice:3-5");
        assert_eq!(temp_bucket(0, 6), "choice:6-10");
        assert_eq!(temp_bucket(0, 11), "choice:11+");
        assert_eq!(temp_bucket(1, 4), "score:3-5");
        assert_eq!(temp_bucket(2, 2), "noul:2");
    }

    #[test]
    fn clamp_bounds_and_refusals() {
        assert_eq!(clamp_temperature(1.7601), 1.7601);
        assert_eq!(clamp_temperature(0.1006), 0.5);
        assert_eq!(clamp_temperature(9.0), 5.0);
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
        assert_eq!(clamp_temperature(f64::INFINITY), 1.0);
        let raw = serde_json::json!(0.1006);
        assert!(temperature_rejected(&raw, 0.5));
        assert!(!temperature_rejected(&raw, 0.1006));
        assert!(temperature_rejected(&serde_json::json!(true), 1.0));
        assert!(temperature_rejected(&serde_json::json!("x"), 1.0));
    }

    #[test]
    fn head_budget_math() {
        // Fits: no cap, full instruction room.
        let (kept, per, budget) = truncate_options_to_budget(&[5, 6, 7], 192);
        assert_eq!((kept, per, budget), (vec![5, 6, 7], None, 192 - 18));
        // Over budget: per-option cut, instruction floor 8 via caller.
        let (kept, per, budget) = truncate_options_to_budget(&[100, 100, 100], 192);
        assert_eq!(per, Some(58));
        assert_eq!(kept, vec![58, 58, 58]);
        assert_eq!(budget, 192 - 174);
    }

    #[test]
    fn billing_temperature_reproduces_agent() {
        // §9b: billing softmax(L/1.7601) == [0.7463, 0.2086, 0.0451].
        // Invert: L = log(p)·t, decode must return p.
        let t = 1.7601f64;
        let want: [f64; 3] = [0.7463, 0.2086, 0.0451];
        let logits: Vec<f32> =
            want.iter().map(|&p| (p.ln() * t) as f32).collect();
        let mut by = HashMap::new();
        by.insert("choice:3-5".to_string(), t);
        let temps = LayaTemperatures { triple: [1.0, 1.0, 1.0], by_options: by, rejections: vec![] };
        let q = LayaQuestion::Choice {
            instructions: "route".to_string(),
            options: vec![
                LayaOption::plain("a"),
                LayaOption::plain("b"),
                LayaOption::plain("c"),
            ],
            option_order: None,
        };
        let d = decode_laya_row(&logits, &[0.0, 0.0], &q, None, &temps, None);
        assert_eq!(d.label, "a");
        for (i, (_, p)) in d.probabilities.iter().enumerate() {
            assert!((*p as f64 - want[i]).abs() < 1e-4, "{p} vs {}", want[i]);
        }
    }

    #[test]
    fn score_expected_value_and_unpermute() {
        let mut by = HashMap::new();
        by.insert("score:2".to_string(), 1.0);
        let temps = LayaTemperatures { triple: [1.0, 1.0, 1.0], by_options: by, rejections: vec![] };
        let q = LayaQuestion::Score {
            instructions: "rate".to_string(),
            levels: vec!["low".to_string(), "high".to_string()],
        };
        // Slots hold [high, low] (order [1, 0]): slot 1 carries the big
        // logit, and slot 1 is caller-option 0 ("low").
        let logits = [0.0f32, 2.0];
        let d = decode_laya_row(&logits, &[0.0, 0.0], &q, Some(&[1, 0]), &temps, None);
        assert_eq!(d.label, "low");
        assert_eq!(d.probabilities[0].0, "low");
        assert!(d.probabilities[0].1 > 0.8);
        assert!(d.value.unwrap() < 0.5, "{}", d.value.unwrap());
    }

    #[test]
    fn noul_reads_p_true() {
        let temps = LayaTemperatures { triple: [1.0, 1.0, 1.0], by_options: HashMap::new(), rejections: vec![] };
        let q = LayaQuestion::Noul { instructions: "holds?".to_string(), statement: "s".to_string() };
        let d = decode_laya_row(&[-1.0, 1.0], &[0.5, -0.5], &q, None, &temps, None);
        assert_eq!(d.label, "true");
        assert_eq!(d.probabilities.len(), 2);
        assert!(d.probabilities[1].1 > 0.8);
        assert!(d.act_probability > 0.5 && d.act_probability < 1.0);
    }

    #[test]
    fn validation_fires_before_io() {
        let bad = LayaQuestion::Choice { instructions: "  ".to_string(), options: vec![LayaOption::plain("a")], option_order: None };
        assert!(validate_laya_question(0, &bad).err().unwrap().to_string().contains("must not be empty"));
        let bad = LayaQuestion::Choice { instructions: "x".to_string(), options: vec![], option_order: None };
        assert!(validate_laya_question(1, &bad).err().unwrap().to_string().contains("at least one option"));
        let bad = LayaQuestion::Choice {
            instructions: "x".to_string(),
            options: vec![LayaOption::plain("a"), LayaOption::plain("a")],
            option_order: None,
        };
        assert!(validate_laya_question(2, &bad).err().unwrap().to_string().contains("duplicate"));
        let bad = LayaQuestion::Choice {
            instructions: "x".to_string(),
            options: vec![LayaOption::plain("a"), LayaOption::plain("b")],
            option_order: Some(vec![0, 0]),
        };
        assert!(validate_laya_question(3, &bad).err().unwrap().to_string().contains("permutation"));
        assert!(validate_laya_question(4, &LayaQuestion::Noul { instructions: "x".to_string(), statement: String::new() }).is_ok());
    }

    #[test]
    fn binning_map_parses_and_refuses() {
        // Bare map.
        let m = BinningMap::parse(&serde_json::json!({
            "choice:2": {"bins": 2, "values": [0.1, 0.9]},
        }))
        .expect("parse");
        assert_eq!(apply_binning_map(0.05, "choice:2", &m), 0.1);
        assert_eq!(apply_binning_map(0.5, "choice:2", &m), 0.9);
        assert_eq!(apply_binning_map(1.0, "choice:2", &m), 0.9);
        // Absent bucket passes through.
        assert_eq!(apply_binning_map(0.7, "noul:2", &m), 0.7);
        // Laya calibration payload nesting (temperatures ignored).
        let p = BinningMap::parse(&serde_json::json!({
            "version": 2,
            "temperature": [1.0, 1.0, 1.0],
            "binning_map": {"score:3-5": {"bins": 1, "values": [0.42]}},
        }))
        .expect("payload parse");
        assert_eq!(apply_binning_map(0.99, "score:3-5", &p), 0.42);
        // Refusals: bool bins, zero bins, length mismatch, out-of-range.
        for bad in [
            serde_json::json!({"choice:2": {"bins": true, "values": [0.5]}}),
            serde_json::json!({"choice:2": {"bins": 0, "values": []}}),
            serde_json::json!({"choice:2": {"bins": 2, "values": [0.5]}}),
            serde_json::json!({"choice:2": {"bins": 1, "values": [1.5]}}),
            serde_json::json!({"choice:2": {"bins": 1, "values": [f64::NAN]}}),
            serde_json::json!([1, 2]),
        ] {
            assert!(BinningMap::parse(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn spans_match_agent_loop_and_first_wins_ties() {
        // 10 tokens, size 4, step 2: the agent's short-final-window loop.
        assert_eq!(
            plan_laya_spans(10, 4, 2),
            vec![(0, 4), (2, 6), (4, 8), (6, 10)]
        );
        assert_eq!(plan_laya_spans(3, 8, 4), vec![(0, 3)]);
        // Ties go to the first window (strict >), both rules.
        assert_eq!(select_laya_window(2, &[(0.1, 0.5), (0.9, 0.5)]), 0);
        assert_eq!(select_laya_window(0, &[(0.5, 0.0), (0.5, 0.0), (0.6, 0.0)]), 2);
        assert_eq!(select_laya_window(1, &[(0.2, 0.0), (0.8, 0.0)]), 1);
        assert_eq!(select_laya_window(2, &[(0.1, 0.2), (0.1, 0.9)]), 1);
    }

    #[test]
    fn open_files_rejects_missing_config_before_session() {
        let e = LayaDecider::open_files(
            Path::new("/nonexistent/model.onnx"),
            Path::new("/nonexistent/tokenizer.json"),
            Path::new("/nonexistent/rl_agent_config.json"),
            DeviceReq::Cpu,
            None,
            false,
        )
        .err()
        .expect("open_files should fail")
        .to_string();
        assert!(e.contains("laya config not found"), "{e}");
    }
}
