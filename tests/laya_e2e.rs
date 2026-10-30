//! End-to-end Laya parity: `LayaDecider::open_files` (Rust marker
//! construction + ORT session) vs the vendored golden built by the
//! upstream `laya.ONNXAgent` itself.
//!
//! Ignored by default: needs an ONNX Runtime dylib and the weight files.
//! Run: `cargo test --offline --test laya_e2e -- --ignored` with:
//! - `ORT_DYLIB_PATH` → onnxruntime.dll (1.29.x, the pinned runtime)
//! - `LAYA_E2E_ONNX` → `laya-td-fp32.onnx` (+ `.data` sidecar alongside)
//! - `LAYA_E2E_TOK` → `tokenizer.json`
//! - `LAYA_E2E_CFG` → `rl_agent_config.json`
//!
//! Gates (the port bar): Rust-built input ids equal the agent-built ids
//! exactly (the construction pin); per-golden probabilities, expected
//! values, confidences and act head within abs 1e-6; exact-match labels;
//! the clamp rejection list names `choice:11+`; validation fires before
//! any session work.

use embroider::{DeviceReq, LayaDecider, LayaOption, LayaQuestion};
use serde_json::Value;
use std::path::Path;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn build_question(c: &Value) -> LayaQuestion {
    let instructions = c["instructions"].as_str().expect("instructions").to_string();
    match c["kind"].as_str().expect("kind") {
        "choice" => {
            let options: Vec<LayaOption> = c["options"]
                .as_array()
                .expect("options")
                .iter()
                .map(|o| {
                    let a = o.as_array().expect("option pair");
                    LayaOption {
                        label: a[0].as_str().expect("label").to_string(),
                        description: a[1].as_str().map(|s| s.to_string()),
                    }
                })
                .collect();
            let option_order: Option<Vec<usize>> = c.get("option_order").and_then(|v| {
                v.as_array().map(|a| {
                    a.iter().map(|x| x.as_u64().expect("order") as usize).collect()
                })
            });
            LayaQuestion::Choice { instructions, options, option_order }
        }
        "score" => {
            let levels: Vec<String> = c["levels"]
                .as_array()
                .expect("levels")
                .iter()
                .map(|l| l.as_str().expect("level").to_string())
                .collect();
            LayaQuestion::Score { instructions, levels }
        }
        _ => LayaQuestion::Noul {
            instructions,
            statement: c["statement"].as_str().expect("statement").to_string(),
        },
    }
}

#[test]
#[ignore]
fn laya_e2e_parity_vs_agent_goldens() {
    let (onnx, tok, cfg) = match (env("LAYA_E2E_ONNX"), env("LAYA_E2E_TOK"), env("LAYA_E2E_CFG")) {
        (Some(o), Some(t), Some(c)) => (o, t, c),
        _ => {
            eprintln!("laya e2e skipped: set LAYA_E2E_ONNX/LAYA_E2E_TOK/LAYA_E2E_CFG");
            return;
        }
    };
    let d = LayaDecider::open_files(
        Path::new(&onnx),
        Path::new(&tok),
        Path::new(&cfg),
        DeviceReq::Cpu,
        None,
        false,
    )
    .expect("open_files");
    assert_eq!(d.max_len(), 1024, "td ceiling from config");

    // The clamp pin: the checkpoint ships choice:11+ = 0.1006, which the
    // [0.5, 5.0] clamp refuses — the rejection list must name it.
    let rej = d.temperature_rejections();
    assert!(
        rej.iter().any(|r| r.contains("choice:11+")),
        "rejections name choice:11+: {rej:?}"
    );
    // Applied temps are the clamped triple + map (billing bucket exact).
    let by: std::collections::HashMap<String, f64> =
        d.temperature_by_options().into_iter().collect();
    assert!((by["choice:3-5"] - 1.7601518630981445).abs() < 1e-9, "billing bucket");

    let text = std::fs::read_to_string("fixtures/golden_laya_td.json").expect("golden readable");
    let fx: Value = serde_json::from_str(&text).expect("fixture parses");
    assert_eq!(fx["contract"], "decide-laya");
    let cases = fx["cases"].as_array().expect("cases");

    // One padded session run for the whole batch (mixed kinds, mixed
    // option counts — the k_cols≥2 collation covers the solo row).
    let items: Vec<(String, LayaQuestion)> = cases
        .iter()
        .map(|c| (c["state"].as_str().expect("state").to_string(), build_question(c)))
        .collect();
    let refs: Vec<(&str, &LayaQuestion)> =
        items.iter().map(|(s, q)| (s.as_str(), q)).collect();
    let got = d.decide_many(&refs).expect("decide_many");
    assert_eq!(got.len(), cases.len());

    for (i, c) in cases.iter().enumerate() {
        let k = c["key"].as_str().expect("key");
        let dec = &got[i];
        assert_eq!(dec.label, c["label"].as_str().expect("label"), "{k} label");
        let want: Vec<(String, f64)> = c["probs"]
            .as_array()
            .expect("probs")
            .iter()
            .map(|p| {
                (
                    p[0].as_str().expect("opt").to_string(),
                    p[1].as_f64().expect("prob"),
                )
            })
            .collect();
        assert_eq!(dec.probabilities.len(), want.len(), "{k} arity");
        for (j, (opt, p)) in dec.probabilities.iter().enumerate() {
            assert_eq!(opt, &want[j].0, "{k} order");
            let dd = (*p as f64 - want[j].1).abs();
            assert!(dd <= 1e-6, "{k}[{opt}]: got {p} want {} (|Δ|={dd})", want[j].1);
        }
        match c["kind"].as_str().expect("kind") {
            "score" => {
                let v = c["value"].as_f64().expect("value");
                assert!(
                    (dec.value.expect("score value") as f64 - v).abs() <= 1e-6,
                    "{k} value"
                );
            }
            _ => assert!(dec.value.is_none(), "{k} no value"),
        }
        for (field, have) in [
            ("confidence", dec.confidence as f64),
            ("answer_confidence", dec.answer_confidence as f64),
            ("act_probability", dec.act_probability as f64),
        ] {
            let v = c[field].as_f64().expect(field);
            assert!((have - v).abs() <= 1e-6, "{k} {field}: got {have} want {v}");
        }
        // Construction pin: the Rust-built row equals the agent-built
        // row exactly (ids and marker slots).
        let (ids, markers) = d
            .encode_row(c["state"].as_str().expect("state"), &build_question(c))
            .expect("encode_row");
        let agent_ids: Vec<u32> = c["agent_ids"]
            .as_array()
            .expect("agent ids")
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect();
        assert_eq!(ids, agent_ids, "{k} input ids");
        let agent_markers: Vec<usize> = c["markers"]
            .as_array()
            .expect("agent markers")
            .iter()
            .map(|x| x.as_u64().expect("marker") as usize)
            .collect();
        assert_eq!(markers, agent_markers, "{k} markers");
        // No truncation engaged on any of the seven rows, so the pin is
        // exact, not prefix.
        assert!(ids.len() <= 1024, "{k} len {}", ids.len());
    }

    // Validation fires before any session work.
    let e = d
        .decide("state", &[LayaQuestion::Choice {
            instructions: String::new(),
            options: vec![LayaOption::plain("a")],
            option_order: None,
        }])
        .err()
        .expect("empty instructions should fail")
        .to_string();
    assert!(e.contains("must not be empty"), "{e}");
    let e = d
        .decide("state", &[LayaQuestion::Score { instructions: "x".to_string(), levels: vec![] }])
        .err()
        .expect("empty levels should fail")
        .to_string();
    assert!(e.contains("at least one level"), "{e}");
    let e = d
        .decide(
            "state",
            &[LayaQuestion::Choice {
                instructions: "x".to_string(),
                options: vec![LayaOption::plain("a"), LayaOption::plain("b")],
                option_order: Some(vec![0, 0]),
            }],
        )
        .err()
        .expect("bad order should fail")
        .to_string();
    assert!(e.contains("permutation"), "{e}");
    // Cross-guard: registry Laya ids are refused by Decider.
    let e = embroider::Decider::open(
        embroider::LAYA_TD_MODEL,
        None,
        None,
        DeviceReq::Cpu,
        None,
        embroider::Precision::Auto,
        false,
    )
    .err()
    .expect("Decider must refuse Laya ids")
    .to_string();
    assert!(e.contains("LayaDecider"), "{e}");
}

/// q8 leg (td only — the fixture pins td): same goldens through the
/// weight-only int8 encoder graph (worst measured 0.023, bound 0.03;
/// labels and the construction pin stay exact; temperatures come from
/// the untouched fitted head, so the 1e-9 bucket pins are inherited
/// from the fp32 leg, not re-asserted here).
#[test]
#[ignore]
fn laya_e2e_q8_parity_vs_agent_goldens() {
    let (onnx, tok, cfg) = match (env("LAYA_E2E_Q8_ONNX"), env("LAYA_E2E_TOK"), env("LAYA_E2E_CFG"))
    {
        (Some(o), Some(t), Some(c)) => (o, t, c),
        _ => {
            eprintln!("laya q8 e2e skipped: set LAYA_E2E_Q8_ONNX (+LAYA_E2E_TOK/CFG)");
            return;
        }
    };
    let d = LayaDecider::open_files(
        Path::new(&onnx),
        Path::new(&tok),
        Path::new(&cfg),
        DeviceReq::Cpu,
        None,
        false,
    )
    .expect("open_files");
    let text = std::fs::read_to_string("fixtures/golden_laya_td.json").expect("golden readable");
    let fx: Value = serde_json::from_str(&text).expect("fixture parses");
    let cases = fx["cases"].as_array().expect("cases");
    let items: Vec<(String, LayaQuestion)> = cases
        .iter()
        .map(|c| (c["state"].as_str().expect("state").to_string(), build_question(c)))
        .collect();
    let refs: Vec<(&str, &LayaQuestion)> =
        items.iter().map(|(s, q)| (s.as_str(), q)).collect();
    let got = d.decide_many(&refs).expect("decide_many");
    assert_eq!(got.len(), cases.len());
    for (i, c) in cases.iter().enumerate() {
        let k = c["key"].as_str().expect("key");
        let dec = &got[i];
        assert_eq!(dec.label, c["label"].as_str().expect("label"), "{k} label");
        let want: Vec<(String, f64)> = c["probs"]
            .as_array()
            .expect("probs")
            .iter()
            .map(|p| (p[0].as_str().expect("opt").to_string(), p[1].as_f64().expect("prob")))
            .collect();
        assert_eq!(dec.probabilities.len(), want.len(), "{k} arity");
        for (j, (opt, p)) in dec.probabilities.iter().enumerate() {
            assert_eq!(opt, &want[j].0, "{k} order");
            let dd = (*p as f64 - want[j].1).abs();
            assert!(dd <= 0.03, "{k}[{opt}]: got {p} want {} (|Δ|={dd})", want[j].1);
        }
        // Construction pin stays exact: q8 changes weights, never layout.
        let (ids, _) = d
            .encode_row(c["state"].as_str().expect("state"), &build_question(c))
            .expect("encode_row");
        let agent_ids: Vec<u32> = c["agent_ids"]
            .as_array()
            .expect("agent ids")
            .iter()
            .map(|x| x.as_u64().expect("id") as u32)
            .collect();
        assert_eq!(ids, agent_ids, "{k} input ids");
    }
}

fn open_laya() -> Option<LayaDecider> {
    let (onnx, tok, cfg) = match (env("LAYA_E2E_ONNX"), env("LAYA_E2E_TOK"), env("LAYA_E2E_CFG")) {
        (Some(o), Some(t), Some(c)) => (o, t, c),
        _ => {
            eprintln!("laya e2e skipped: set LAYA_E2E_ONNX/LAYA_E2E_TOK/LAYA_E2E_CFG");
            return None;
        }
    };
    Some(
        LayaDecider::open_files(
            Path::new(&onnx),
            Path::new(&tok),
            Path::new(&cfg),
            DeviceReq::Cpu,
            None,
            false,
        )
        .expect("open_files"),
    )
}

#[test]
#[ignore]
fn laya_e2e_predict_long_matches_agent_scan() {
    let d = match open_laya() {
        Some(d) => d,
        None => return,
    };
    let text = std::fs::read_to_string("fixtures/golden_laya_td.json").expect("golden readable");
    let fx: Value = serde_json::from_str(&text).expect("fixture parses");
    let window = fx["long_window"].as_u64().expect("window") as usize;
    let stride = fx["long_stride"].as_u64().expect("stride") as usize;
    let state = fx["long_state"].as_str().expect("long state").to_string();
    let cases = fx["long_cases"].as_array().expect("long cases");
    // The scan answers rounded to 4 decimals (the agent's presentation
    // rounding, not the Rust core) — pinned at 1e-4.
    let tol = 1e-4;
    let items: Vec<(String, LayaQuestion)> = cases
        .iter()
        .map(|c| (state.clone(), build_question(c)))
        .collect();
    let refs: Vec<(&str, &LayaQuestion)> =
        items.iter().map(|(s, q)| (s.as_str(), q)).collect();
    let questions: Vec<LayaQuestion> = refs.iter().map(|(_, q)| (*q).clone()).collect();
    let got = d.predict_long(&state, &questions, Some(window), Some(stride)).expect("scan");
    assert_eq!(got.len(), cases.len());
    for (i, c) in cases.iter().enumerate() {
        let k = c["key"].as_str().expect("key");
        let l = &got[i];
        let dec = &l.decision;
        assert_eq!(dec.label, c["label"].as_str().expect("label"), "{k} label");
        let want: Vec<(String, f64)> = c["probs"]
            .as_array()
            .expect("probs")
            .iter()
            .map(|p| (p[0].as_str().expect("opt").to_string(), p[1].as_f64().expect("prob")))
            .collect();
        assert_eq!(dec.probabilities.len(), want.len(), "{k} arity");
        for (j, (opt, p)) in dec.probabilities.iter().enumerate() {
            assert_eq!(opt, &want[j].0, "{k} order");
            let dd = (*p as f64 - want[j].1).abs();
            assert!(dd <= tol, "{k}[{opt}]: got {p} want {} (|Δ|={dd})", want[j].1);
        }
        for (field, have) in [
            ("confidence", dec.confidence as f64),
            ("answer_confidence", dec.answer_confidence as f64),
            ("act_probability", dec.act_probability as f64),
        ] {
            let v = c[field].as_f64().expect(field);
            assert!((have - v).abs() <= tol, "{k} {field}: got {have} want {v}");
        }
        // Span-identity with the agent: same window decided, same
        // token span, same window count.
        let w = c["window"].as_object().expect("window");
        assert_eq!(l.window_start, w["token_start"].as_u64().expect("start") as usize, "{k} start");
        assert_eq!(l.window_end, w["token_end"].as_u64().expect("end") as usize, "{k} end");
        assert_eq!(l.windows, w["count"].as_u64().expect("count") as usize, "{k} count");
        assert_eq!(l.windows, c["windows"].as_u64().expect("windows") as usize, "{k} windows");
    }
    // The scan actually selected mid-document windows (not window 0) —
    // the aggregation rule did work, not just the fast path.
    assert!(got.iter().any(|l| l.window_start > 0), "a middle window must decide");
    // A short state takes the fast path: identical to decide, one window.
    let short = d
        .predict_long("The football match ended 2-1.", &questions[..1], Some(window), Some(stride))
        .expect("fast path");
    assert_eq!(short.len(), 1);
    assert_eq!(short[0].windows, 1);
    assert_eq!(short[0].window_start, 0);
}

#[test]
#[ignore]
fn laya_e2e_binning_map_remaps_mechanism() {
    let mut d = match open_laya() {
        Some(d) => d,
        None => return,
    };
    assert!(!d.has_binning_map());
    let sport = LayaQuestion::Choice {
        instructions: "Decide the topic.".to_string(),
        options: vec![LayaOption::plain("sports"), LayaOption::plain("cooking")],
        option_order: None,
    };
    let state = "The football match ended 2-1 after extra time.";
    let before = d.decide(state, &[sport.clone()]).expect("decide")[0].clone();
    // Synthetic 2-bin map (mechanism proof, not a fit): confidence
    // 0.91 lands in bin 1 → 0.85, probabilities untouched.
    let dir = std::env::temp_dir().join("embroider-laya-e2e-binning.json");
    std::fs::write(&dir, r#"{"choice:2": {"bins": 2, "values": [0.15, 0.85]}}"#)
        .expect("write map");
    d.load_binning_map(&dir).expect("load map");
    assert!(d.has_binning_map());
    let after = d.decide(state, &[sport]).expect("decide")[0].clone();
    assert_eq!(after.probabilities, before.probabilities, "probs untouched");
    assert_eq!(after.confidence, before.confidence, "entropy untouched");
    assert!((after.answer_confidence - 0.85).abs() < 1e-6, "binned: {}", after.answer_confidence);
    assert!((before.answer_confidence - 0.9134).abs() < 1e-3, "raw: {}", before.answer_confidence);
    std::fs::remove_file(&dir).ok();
}
