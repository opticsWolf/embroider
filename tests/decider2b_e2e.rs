//! End-to-end decider-2b parity: `Decider2B::open_files` (Rust slot
//! construction + ORT session) vs the vendored golden built by the
//! reference `DeciderLayout` runner itself.
//!
//! Ignored by default: needs an ONNX Runtime dylib and the weight files.
//! Run: `cargo test --offline --test decider2b_e2e -- --ignored` with:
//! - `ORT_DYLIB_PATH` → onnxruntime.dll (1.29.x, the pinned runtime)
//! - `DECIDER2B_E2E_ONNX` → `model.onnx` (+ `model.safetensors` alongside)
//! - `DECIDER2B_E2E_TOK` → `tokenizer.json`
//! - `DECIDER2B_E2E_DECISION` → `decision.json`
//! - `DECIDER2B_E2E_CALIB` → `calibration.json`
//!
//! Gates (the port bar): Rust-built input ids/slots equal the reference
//! ids exactly (the construction pin); exact-match labels;
//! per-golden probabilities and expected values within abs 5e-4 (above
//! the 1e-4 torch-vs-ORT gap, so the fixture stays torch-consistent);
//! the applied triple equals the checkpoint calibration; validation
//! fires before any session work.

use embroider::{Decider2B, Decider2BOption, Decider2BQuestion, DeviceReq};
use serde_json::Value;
use std::path::Path;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn build_questions(spec: &Value) -> Vec<Decider2BQuestion> {
    if spec["kind"].as_str() == Some("multi") {
        return spec["questions"]
            .as_array()
            .expect("questions")
            .iter()
            .map(build_one)
            .collect();
    }
    vec![build_one(spec)]
}

fn build_one(c: &Value) -> Decider2BQuestion {
    let instructions = c["instructions"].as_str().expect("instructions").to_string();
    match c["kind"].as_str().expect("kind") {
        "choice" => {
            let options: Vec<Decider2BOption> = c["options"]
                .as_array()
                .expect("options")
                .iter()
                .map(|o| {
                    let a = o.as_array().expect("option pair");
                    Decider2BOption {
                        label: a[0].as_str().expect("label").to_string(),
                        description: a[1].as_str().map(|s| s.to_string()),
                    }
                })
                .collect();
            Decider2BQuestion::Choice { instructions, options }
        }
        "score" => {
            let levels: Vec<String> = c["levels"]
                .as_array()
                .expect("levels")
                .iter()
                .map(|l| l.as_str().expect("level").to_string())
                .collect();
            Decider2BQuestion::Score { instructions, levels }
        }
        _ => Decider2BQuestion::Noul {
            instructions,
            true_criterion: c["true_criterion"].as_str().unwrap_or("").to_string(),
            false_criterion: c["false_criterion"].as_str().unwrap_or("").to_string(),
        },
    }
}

#[test]
#[ignore]
fn decider2b_e2e_parity_vs_reference_goldens() {
    run_parity(DeviceReq::Cpu);
}

/// The CUDA leg: same goldens through the parallel executor. This is
/// the ORT #4829 gate — the sequential planner aliases a shape buffer
/// on this graph (chunk-scan Reshape reads a zeroed dim, run fails),
/// so a regression back to sequential fails loudly here, not in a
/// bench. Needs the CUDA execution provider in the loaded ORT.
#[test]
#[ignore]
fn decider2b_e2e_cuda_parity() {
    run_parity(DeviceReq::Cuda);
}

fn run_parity(device: DeviceReq) {
    let (onnx, tok, dec, cal) = match (
        env("DECIDER2B_E2E_ONNX"),
        env("DECIDER2B_E2E_TOK"),
        env("DECIDER2B_E2E_DECISION"),
        env("DECIDER2B_E2E_CALIB"),
    ) {
        (Some(o), Some(t), Some(d), Some(c)) => (o, t, d, c),
        _ => {
            eprintln!(
                "decider2b e2e skipped: set \
                 DECIDER2B_E2E_ONNX/TOK/DECISION/CALIB"
            );
            return;
        }
    };
    let d = Decider2B::open_files(
        Path::new(&onnx),
        Path::new(&tok),
        Path::new(&dec),
        Path::new(&cal),
        device,
        None,
        false,
    )
    .expect("open_files");
    assert_eq!(d.max_len(), 32768, "ceiling from decision.json");

    // The clamp pin: the checkpoint ships (2.465, 1.0, 0.05), all inside
    // the [0.05, 5.0] range — nothing rejected, triple exact.
    assert!(d.temperature_rejections().is_empty(), "{:?}", d.temperature_rejections());
    let t = d.temperatures();
    for (got, want) in t.iter().zip([2.465f64, 1.0, 0.05]) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    let text =
        std::fs::read_to_string("fixtures/golden_decider2b.json").expect("golden readable");
    let fx: Value = serde_json::from_str(&text).expect("fixture parses");
    let cases = fx["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 8);

    for c in cases {
        let key = c["key"].as_str().expect("key");
        let state = c["state"].as_str().expect("state");
        let qs = build_questions(&c["spec"]);

        // Construction pin: ids, slot and k, row for row.
        let rows = d.encode_rows(state, &qs).expect("encode_rows");
        let want_rows = c["rows"].as_array().expect("rows");
        assert_eq!(rows.len(), want_rows.len(), "{key} row count");
        for (i, (ids, slot, k)) in rows.iter().enumerate() {
            let want_ids: Vec<u32> = want_rows[i]["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|x| x.as_u64().expect("id") as u32)
                .collect();
            assert_eq!(ids, &want_ids, "{key} row {i} ids");
            assert_eq!(*slot, want_rows[i]["slot"].as_u64().expect("slot") as usize, "{key} row {i} slot");
            assert_eq!(*k, want_rows[i]["k"].as_u64().expect("k") as usize, "{key} row {i} k");
        }

        // Answer pin: labels exact, probs/values at 5e-4.
        let got = d.decide(state, &qs).expect("decide");
        let want_ans = c["answers"].as_array().expect("answers");
        assert_eq!(got.len(), want_ans.len(), "{key} answer count");
        for (dec, want) in got.iter().zip(want_ans) {
            assert_eq!(dec.label, want["label"].as_str().expect("label"), "{key} label");
            let want_p: Vec<(String, f64)> = want["probs"]
                .as_array()
                .expect("probs")
                .iter()
                .map(|p| (p[0].as_str().expect("opt").to_string(), p[1].as_f64().expect("prob")))
                .collect();
            assert_eq!(dec.probabilities.len(), want_p.len(), "{key} arity");
            for (j, (opt, p)) in dec.probabilities.iter().enumerate() {
                assert_eq!(opt, &want_p[j].0, "{key} order");
                let dd = (*p as f64 - want_p[j].1).abs();
                assert!(dd <= 5e-4, "{key}[{opt}]: got {p} want {} (|Δ|={dd})", want_p[j].1);
            }
            match &want["value"] {
                Value::Null => assert!(dec.value.is_none(), "{key} value none"),
                Value::Number(n) => {
                    let v = n.as_f64().expect("value");
                    assert!(
                        (dec.value.expect("value") as f64 - v).abs() <= 5e-4,
                        "{key} value"
                    );
                }
                other => panic!("{key} bad value shape: {other}"),
            }
        }
    }
}
