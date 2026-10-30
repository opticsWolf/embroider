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
//! - `RERANK_E2E_Q8_ONNX` / `RERANK_E2E_Q8_TOK`,
//!   `RERANK_MM_E2E_Q8_ONNX` / `RERANK_MM_E2E_Q8_TOK` → the q8 pair
//!   (MiniLM worst 0.161, mMARCO worst 0.679 — q8 legs pin those bounds)
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

fn check(onnx: &str, tok: &str, fixture: &str, tol: f64) {
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
            d <= tol,
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

    // Token counts pin the Rust tokenizer against the Python reference
    // (same revision, no specials) — chunking budgets build on these.
    let counts = fx["token_counts"].as_object().expect("token_counts");
    for (i, k) in keys.iter().enumerate() {
        let (q, d) = (&batch[i].0, &batch[i].1);
        assert_eq!(
            r.count_tokens(q).expect("count"),
            counts[k]["query"].as_u64().expect("query count") as usize,
            "{k} query"
        );
        assert_eq!(
            r.count_tokens(d).expect("count"),
            counts[k]["doc"].as_u64().expect("doc count") as usize,
            "{k} doc"
        );
    }

    // score_long on a single-window pair must equal score_one on the same
    // text (decode–re-encode round-trips exactly; the same pair seam runs).
    for (i, k) in keys.iter().enumerate() {
        let (q, d) = (batch[i].0.as_str(), batch[i].1.as_str());
        let long = r
            .score_long(q, d, Some(512), Some(256))
            .expect("score_long single-window");
        let d = (long as f64 - got[i] as f64).abs();
        assert!(d <= 1e-6, "{k}: score_long {long} vs score {got:?} (|Δ|={d})");
    }

    // A document past the cap chunks and max-pools instead of truncating:
    // repeat the irrelevant doc ~30x (well past 512 tokens) and check the
    // long path runs, stays finite, and still prefers the relevant doc.
    let long_doc = batch[1].1.repeat(30);
    assert!(r.count_tokens(&long_doc).expect("count") > 512);
    let long_score = r
        .score_long(&batch[1].0, &long_doc, None, None)
        .expect("score_long multi-window");
    assert!(long_score.is_finite(), "{long_score}");
    // MaxP sanity: the long irrelevant doc must not outrank the short
    // relevant one for the same query.
    assert!(
        by_key["en_relevant"] > long_score as f64,
        "relevant {} vs long-irrelevant {long_score}",
        by_key["en_relevant"]
    );

    // Validation fires before any chunking work.
    let e = r
        .score_long("q", "d", Some(0), None)
        .err()
        .expect("window 0 should fail")
        .to_string();
    assert!(e.contains("1..="), "{e}");
    let e = r
        .score_long("q", "d", None, Some(0))
        .err()
        .expect("stride 0 should fail")
        .to_string();
    assert!(e.contains(">= 1"), "{e}");
}

#[test]
#[ignore]
fn rerank_e2e_parity_vs_goldens() {
    let mut ran = false;
    if let (Some(onnx), Some(tok)) = (env("RERANK_E2E_ONNX"), env("RERANK_E2E_TOK")) {
        check(&onnx, &tok, "fixtures/golden_rerank_minilm.json", 1e-6);
        ran = true;
    }
    if let (Some(onnx), Some(tok)) = (env("RERANK_MM_E2E_ONNX"), env("RERANK_MM_E2E_TOK")) {
        check(&onnx, &tok, "fixtures/golden_rerank_mmarco.json", 1e-6);
        ran = true;
    }
    if !ran {
        eprintln!(
            "rerank e2e skipped: set RERANK_E2E_ONNX/RERANK_E2E_TOK \
             and/or RERANK_MM_E2E_ONNX/RERANK_MM_E2E_TOK"
        );
    }
}

/// q8 legs: same goldens through the weight-only int8 graphs. Bounds
/// are the measured fp32-vs-q8 worst diffs (MiniLM 0.161, mMARCO 0.679)
/// with headroom; rank pins stay exact (order is what ships).
#[test]
#[ignore]
fn rerank_e2e_q8_parity_vs_goldens() {
    let mut ran = false;
    if let (Some(onnx), Some(tok)) = (env("RERANK_E2E_Q8_ONNX"), env("RERANK_E2E_Q8_TOK")) {
        check(&onnx, &tok, "fixtures/golden_rerank_minilm.json", 0.2);
        ran = true;
    }
    if let (Some(onnx), Some(tok)) = (env("RERANK_MM_E2E_Q8_ONNX"), env("RERANK_MM_E2E_Q8_TOK"))
    {
        // 0.9 bounds the shipped-path worst (0.864 on en_relevant;
        // Python-side approx measured 0.679 — truncation differs).
        // The rank pins inside check() stay exact: order is the contract.
        check(&onnx, &tok, "fixtures/golden_rerank_mmarco.json", 0.9);
        ran = true;
    }
    if !ran {
        eprintln!(
            "rerank q8 e2e skipped: set RERANK_E2E_Q8_ONNX/RERANK_E2E_Q8_TOK \
             and/or RERANK_MM_E2E_Q8_ONNX/RERANK_MM_E2E_Q8_TOK"
        );
    }
}
