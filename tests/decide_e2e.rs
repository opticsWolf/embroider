//! End-to-end decide parity: `Decider::open_files` (Rust input
//! construction + ORT session) vs the vendored goldens.
//!
//! Ignored by default: needs an ONNX Runtime dylib and the weight files.
//! Run: `cargo test --offline --test decide_e2e -- --ignored` with:
//! - `ORT_DYLIB_PATH` → onnxruntime.dll (1.29.x, the pinned runtime)
//! - small-v1.0: `DECIDE_E2E_ONNX` → `onnx/model.onnx`,
//!   `DECIDE_E2E_TOK` → `tokenizer.json`
//! - v4-edge: `DECIDE_E2E_V4_ONNX` → `onnx/model.onnx`,
//!   `DECIDE_E2E_V4_TOK` → `tokenizer.json`
//!
//! Gates: per-golden probabilities within tolerance (1e-6 small-v1.0,
//! 0.03 v4-edge — the embedded calibrator accumulates fp32 kernel-order
//! drift vs torch, so v4 pins ORT values, not torch parity), exact-match
//! labels, Rust-built input ids equal to a single recipe encode (the
//! layout contract — text-first for small, labels-first for v4), and
//! choice winners outranking every loser.

use embroider::{Decider, DeviceReq, Question};
use serde_json::Value;
use std::path::Path;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn check(onnx: &str, tok: &str, fixture: &str, tol: f64, prompt_first: bool, cap: usize) {
    let d = Decider::open_files(
        Path::new(onnx),
        Path::new(tok),
        DeviceReq::Cpu,
        None,
        prompt_first,
        false,
    )
    .expect("open_files");
    assert_eq!(d.prompt_first(), prompt_first, "caller layout");
    let text =
        std::fs::read_to_string(fixture).expect("golden fixture readable");
    let fx: Value = serde_json::from_str(&text).expect("fixture parses");
    assert_eq!(fx["contract"], "decide-gliclass");

    let cases = fx["cases"].as_array().expect("cases list");
    // One batched run per kind group (choice rows and score rows batch
    // separately — mixed-C rows share the max-slot readout either way,
    // but grouping keeps the test's intent readable).
    for kind in ["choice", "score"] {
        let mut items: Vec<(String, Question)> = Vec::new();
        let mut keys: Vec<&str> = Vec::new();
        for c in cases {
            if c["kind"].as_str().expect("kind") != kind {
                continue;
            }
            let options: Vec<String> = c["options"]
                .as_array()
                .expect("options")
                .iter()
                .map(|o| o.as_str().expect("option").to_string())
                .collect();
            let instructions = c
                .get("instructions")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let q = match kind {
                "choice" => Question::Choice { instructions, options },
                _ => Question::Score { instructions, options },
            };
            items.push((c["state"].as_str().expect("state").to_string(), q));
            keys.push(c["key"].as_str().expect("key"));
        }
        if items.is_empty() {
            continue;
        }
        let refs: Vec<(&str, &Question)> =
            items.iter().map(|(s, q)| (s.as_str(), q)).collect();
        let got = d.decide_many(&refs).expect("decide_many");
        assert_eq!(got.len(), keys.len());
        for (i, k) in keys.iter().enumerate() {
            let c = cases.iter().find(|c| c["key"] == *k).expect("case");
            assert_eq!(got[i].label, c["label"].as_str().expect("label"), "{k} label");
            let want: Vec<f64> = c["probs"]
                .as_array()
                .expect("probs")
                .iter()
                .map(|p| p.as_f64().expect("prob"))
                .collect();
            assert_eq!(got[i].probabilities.len(), want.len(), "{k} arity");
            for (j, (opt, p)) in got[i].probabilities.iter().enumerate() {
                assert_eq!(opt, &c["options"][j].as_str().expect("option"), "{k} order");
                let dd = (*p as f64 - want[j]).abs();
                assert!(dd <= tol, "{k}[{opt}]: got {p} want {} (|Δ|={dd})", want[j]);
            }
        }
        // Semantic pin, not just numeric: the winner outranks every loser.
        for dec in &got {
            let top = dec.probabilities.iter().map(|(_, p)| *p).fold(f32::NEG_INFINITY, f32::max);
            let winner = dec.probabilities.iter().find(|(_, p)| *p == top).expect("winner");
            assert_eq!(&dec.label, &winner.0);
        }
    }

    // Layout contract: the Rust-built ids equal a single recipe encode
    // through the same tokenizer (truncate-state-side-first is a no-op
    // here — all cases fit the cap, so equality is exact).
    let rt = tokenizers::Tokenizer::from_file(tok).expect("tokenizer loads");
    let c = &cases[0];
    let state = c["state"].as_str().expect("state");
    let options: Vec<String> = c["options"]
        .as_array()
        .expect("options")
        .iter()
        .map(|o| o.as_str().expect("option").to_string())
        .collect();
    let instructions = c.get("instructions").and_then(|v| v.as_str()).unwrap_or("");
    let label_text = format!(
        "{}{}",
        options.iter().map(|o| format!("<<LABEL>>{o}")).collect::<String>(),
        format!("<<SEP>>{instructions}")
    );
    let recipe_seq = if prompt_first {
        format!("{label_text}{state}")
    } else {
        format!("{state}{label_text}")
    };
    let recipe = rt.encode(recipe_seq.as_str(), true).expect("recipe encode");
    // Rebuild through the public seam: decide returns no ids, so compare
    // against the documented construction — the count pins the state side.
    let counts = fx["token_counts"].as_object().expect("token_counts");
    for cc in cases {
        let k = cc["key"].as_str().expect("key");
        assert_eq!(
            d.count_tokens(cc["state"].as_str().expect("state")).expect("count"),
            counts[k].as_u64().expect("count") as usize,
            "{k} state tokens"
        );
    }
    // The recipe encode fits the cap whole (no truncation engaged).
    assert!(recipe.len() <= cap, "recipe len {}", recipe.len());
    // Specials present exactly once each at the boundaries.
    let ids = recipe.get_ids();
    assert_eq!(ids.first(), Some(&rt.token_to_id("[CLS]").expect("cls")));
    assert_eq!(ids.last(), Some(&rt.token_to_id("[SEP]").expect("sep")));

    // Validation fires before any session work.
    let e = d
        .decide(state, &[Question::Noul { instructions: String::new(), options: vec!["x".to_string()] }])
        .err()
        .expect("noul should fail")
        .to_string();
    assert!(e.contains("Noul"), "{e}");
    let e = d
        .decide(state, &[Question::Choice { instructions: String::new(), options: vec![] }])
        .err()
        .expect("empty options should fail")
        .to_string();
    assert!(e.contains("at least one option"), "{e}");
    // Over-long options (alone filling the cap) fail instead of cutting
    // the question to fit the state.
    let big: Vec<String> = (0..cap).map(|i| format!("option{i}")).collect();
    let e = d
        .decide(
            "short state",
            &[Question::Choice { instructions: String::new(), options: big }],
        )
        .err()
        .expect("over-long options should fail")
        .to_string();
    assert!(e.contains("no room for the state"), "{e}");
}

#[test]
#[ignore]
fn decide_e2e_parity_vs_goldens() {
    if let (Some(onnx), Some(tok)) = (env("DECIDE_E2E_ONNX"), env("DECIDE_E2E_TOK")) {
        check(&onnx, &tok, "fixtures/golden_decide_gliclass_small.json", 1e-6, false, 512);
    } else {
        eprintln!("decide e2e skipped: set DECIDE_E2E_ONNX/DECIDE_E2E_TOK");
    }
}

#[test]
#[ignore]
fn decide_e2e_v4_parity_vs_goldens() {
    if let (Some(onnx), Some(tok)) = (env("DECIDE_E2E_V4_ONNX"), env("DECIDE_E2E_V4_TOK")) {
        check(&onnx, &tok, "fixtures/golden_decide_gliclass_v4edge.json", 0.03, true, 2048);
    } else {
        eprintln!("decide v4 e2e skipped: set DECIDE_E2E_V4_ONNX/DECIDE_E2E_V4_TOK");
    }
}

/// v4 q8 leg: same goldens through the weight-only int8 graph (worst
/// measured 0.0899 — the 0.1 bound below carries ~10% headroom; labels
/// stay exact). Small has no q8 leg: its build flips 2 knife-edge
/// travel cases, so it stays fp32-only (see the registry comment).
#[test]
#[ignore]
fn decide_e2e_v4_q8_parity_vs_goldens() {
    if let (Some(onnx), Some(tok)) = (env("DECIDE_E2E_V4_Q8_ONNX"), env("DECIDE_E2E_V4_Q8_TOK"))
    {
        check(&onnx, &tok, "fixtures/golden_decide_gliclass_v4edge.json", 0.1, true, 2048);
    } else {
        eprintln!("decide v4 q8 e2e skipped: set DECIDE_E2E_V4_Q8_ONNX/DECIDE_E2E_V4_Q8_TOK");
    }
}
