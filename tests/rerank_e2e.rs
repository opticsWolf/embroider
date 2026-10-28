//! End-to-end rerank parity: `Reranker::open_files` (Rust pair encoding +
//! ORT session) vs the vendored goldens from the A0 verification spike.
//!
//! Ignored by default: needs an ONNX Runtime dylib and the weight files.
//! Run: `cargo test --offline --test rerank_e2e -- --ignored` with:
//! - `ORT_DYLIB_PATH` → onnxruntime.dll (1.29.x, the pinned runtime)
//! - `RERANK_E2E_ONNX` → MiniLM-L-6 `onnx/model.onnx`
//! - `RERANK_E2E_TOK` → MiniLM-L-6 `tokenizer.json`
//! - `RERANK_MM_E2E_ONNX` / `RERANK_MM_E2E_TOK` → the mMARCO pair
//!   (each pair optional: set one to test one model)
//!
//! Gates (the spike-plan bar): abs diff ≤ 1e-6 per golden logit, and
//! relevant outranks irrelevant per language.

use embroider::{DeviceReq, Reranker};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn check(onnx: &str, tok: &str, fixture: &str) {
    let r = Reranker::open_files(
        Path::new(onnx),
        Path::new(tok),
        DeviceReq::Cpu,
        None,
        false,
    )
    .expect("open_files");
    let text =
        std::fs::read_to_string(fixture).expect("golden fixture readable");
    let fx: Value = serde_json::from_str(&text).expect("fixture parses");
    assert_eq!(fx["contract"], "rerank-cross-encoder");

    let pairs = fx["pairs"].as_object().expect("pairs map");
    let mut batch: Vec<(String, String)> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    for c in fx["cases"].as_array().expect("cases list") {
        let k = c["pair"].as_str().expect("pair key").to_string();
        let p = &pairs[&k];
        batch.push((
            p["query"].as_str().expect("query").to_string(),
            p["doc"].as_str().expect("doc").to_string(),
        ));
        keys.push(k);
    }
    let refs: Vec<(&str, &str)> = batch
        .iter()
        .map(|(q, d)| (q.as_str(), d.as_str()))
        .collect();
    let got = r.score_many(&refs).expect("score_many");

    let mut by_key: HashMap<&str, f64> = HashMap::new();
    for (i, c) in fx["cases"].as_array().unwrap().iter().enumerate() {
        let want = c["score"].as_f64().expect("score");
        let d = (got[i] as f64 - want).abs();
        assert!(
            d <= 1e-6,
            "{}: got {} want {want} (|Δ|={d})",
            keys[i],
            got[i]
        );
        by_key.insert(keys[i].as_str(), got[i] as f64);
    }
    // Semantic pin, not just numeric: ranking order holds per language.
    assert!(
        by_key["en_relevant"] > by_key["en_irrelevant"],
        "{by_key:?}"
    );
    assert!(
        by_key["de_relevant"] > by_key["de_irrelevant"],
        "{by_key:?}"
    );
}

#[test]
#[ignore]
fn rerank_e2e_parity_vs_goldens() {
    let mut ran = false;
    if let (Some(onnx), Some(tok)) = (env("RERANK_E2E_ONNX"), env("RERANK_E2E_TOK")) {
        check(&onnx, &tok, "fixtures/golden_rerank_minilm.json");
        ran = true;
    }
    if let (Some(onnx), Some(tok)) = (env("RERANK_MM_E2E_ONNX"), env("RERANK_MM_E2E_TOK")) {
        check(&onnx, &tok, "fixtures/golden_rerank_mmarco.json");
        ran = true;
    }
    if !ran {
        eprintln!(
            "rerank e2e skipped: set RERANK_E2E_ONNX/RERANK_E2E_TOK \
             and/or RERANK_MM_E2E_ONNX/RERANK_MM_E2E_TOK"
        );
    }
}
