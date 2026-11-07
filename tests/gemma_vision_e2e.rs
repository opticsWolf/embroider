//! End-to-end Gemma vision parity: `GemmaVision` (Rust host math +
//! two ORT sessions) vs the sentence-transformers goldens.
//!
//! Ignored by default: needs an ONNX Runtime dylib and the weight files.
//! Run: `cargo test --offline --test gemma_vision_e2e -- --ignored` with:
//! - `ORT_DYLIB_PATH` → onnxruntime.dll (1.31.x, the pinned runtime)
//! - `GEMMA_VISION_E2E_VISION` → fp32 `vision_encoder.onnx` (+ sidecar)
//! - `GEMMA_VISION_E2E_TEXT` → fp32 `model.onnx` (+ sidecar)
//! - `GEMMA_VISION_E2E_TOK` → `tokenizer.json`
//! - `GEMMA_VISION_E2E_VISION_FP16` / `GEMMA_VISION_E2E_TEXT_FP16` → fp16 pair
//!   (CUDA-only leg: the vision tower NaNs under CPU-emulated fp16)
//! - `GEMMA_VISION_E2E_CUDA=1` to run on CUDA (default CPU)
//!
//! Gates: cos ≥ 0.9999 fp32 vs ST goldens (768d), fp16 ≥ 0.999, plus a
//! cross-modal rank pin (red image closer to red text than gradient text).

use embroider::{gemma_default_target_size, DeviceReq, GemmaVision};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        d += *x as f64 * *y as f64;
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (d / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

#[test]
#[ignore]
fn gemma_vision_e2e_parity_vs_st() {
    let (vision, text, tok) = match (
        env("GEMMA_VISION_E2E_VISION"),
        env("GEMMA_VISION_E2E_TEXT"),
        env("GEMMA_VISION_E2E_TOK"),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!(
                "gemma vision e2e skipped: set GEMMA_VISION_E2E_VISION/TEXT/TOK"
            );
            return;
        }
    };
    let device =
        if env("GEMMA_VISION_E2E_CUDA").as_deref() == Some("1") { DeviceReq::Cuda } else { DeviceReq::Cpu };
    let gold: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/gemmavision/e2e_cases.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(gold["contract"], "gemma-vision");

    let v = GemmaVision::open_files(
        std::path::Path::new(&vision),
        std::path::Path::new(&text),
        std::path::Path::new(&tok),
        768,
        device,
        None,
    )
    .unwrap();
    println!("used_cuda={} precision={}", v.used_cuda(), v.precision().as_str());

    let mut got: Vec<(&str, Vec<f32>)> = Vec::new();
    for c in gold["cases"].as_array().unwrap() {
        let stem = c["stem"].as_str().unwrap();
        let path = format!(
            "{}/fixtures/gemmavision/{}.png",
            env!("CARGO_MANIFEST_DIR"),
            stem
        );
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        // Fixture PNGs are stored at valid Gemma target sizes already.
        assert_eq!(
            gemma_default_target_size(h as u32, w as u32).unwrap(),
            (h as u32, w as u32),
            "{stem}"
        );
        let want: Vec<f32> = c["st_768"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let e = v.encode_image(img.as_raw(), h, w).unwrap();
        assert_eq!(e.len(), 768);
        let n: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3, "{stem} norm {n}");
        let c32 = cos(&e, &want);
        println!("{stem}: cos fp32={c32:.5}");
        assert!(c32 >= 0.9999, "{stem} fp32 {c32}");
        got.push((stem, e));
    }

    // Registry path: `open()` resolves the vision + text-partner artifacts,
    // verifies the tokenizer, and encodes identically to `open_files`.
    let r = embroider::GemmaVision::open(
        "google/embeddinggemma-2-vision",
        None,
        None,
        768,
        device,
        embroider::Precision::Fp32,
        None,
    )
    .unwrap();
    assert_eq!(r.model_id(), "google/embeddinggemma-2-vision");
    let img = image::open(format!(
        "{}/fixtures/gemmavision/red768.png",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
    .to_rgb8();
    let er = r.encode_image(img.as_raw(), 768, 768).unwrap();
    let red_want: Vec<f32> = gold["cases"].as_array().unwrap()[0]["st_768"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    assert!(cos(&er, &red_want) >= 0.9999, "registry open() parity");

    // Cross-modal rank pin: the red image sits closer to the red caption
    // than to the gradient caption (unified space, not just parity).
    let red = got.iter().find(|(s, _)| *s == "red768").unwrap().1.clone();
    let red_txt: Vec<f32> = gold["texts"]["a solid red square"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let grad_txt: Vec<f32> = gold["texts"]["a smooth horizontal gradient from black to bright"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let c_red = cos(&red, &red_txt);
    let c_grad = cos(&red, &grad_txt);
    println!("cross-modal: red/red={c_red:.4} red/grad={c_grad:.4}");
    assert!(c_red > c_grad, "rank pin broken: {c_red} vs {c_grad}");
}

#[test]
#[ignore]
fn gemma_vision_e2e_fp16_parity() {
    let (vision, text, tok) = match (
        env("GEMMA_VISION_E2E_VISION_FP16"),
        env("GEMMA_VISION_E2E_TEXT_FP16"),
        env("GEMMA_VISION_E2E_TOK"),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!(
                "gemma vision fp16 e2e skipped: set GEMMA_VISION_E2E_VISION_FP16/TEXT_FP16 (+TOK)"
            );
            return;
        }
    };
    let device =
        if env("GEMMA_VISION_E2E_CUDA").as_deref() == Some("1") { DeviceReq::Cuda } else {
            eprintln!("gemma vision fp16 e2e skipped: vision-tower fp16 needs CUDA (GEMMA_VISION_E2E_CUDA=1)");
            return;
        };
    let gold: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/gemmavision/e2e_cases.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let v = GemmaVision::open_files(
        std::path::Path::new(&vision),
        std::path::Path::new(&text),
        std::path::Path::new(&tok),
        768,
        device,
        None,
    )
    .unwrap();
    println!("used_cuda={} precision={}", v.used_cuda(), v.precision().as_str());
    for c in gold["cases"].as_array().unwrap() {
        let stem = c["stem"].as_str().unwrap();
        let path = format!(
            "{}/fixtures/gemmavision/{}.png",
            env!("CARGO_MANIFEST_DIR"),
            stem
        );
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let want: Vec<f32> = c["st_768"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let e = v.encode_image(img.as_raw(), h, w).unwrap();
        let c16 = cos(&e, &want);
        println!("{stem}: cos fp16={c16:.5}");
        assert!(c16 >= 0.999, "{stem} fp16 {c16}");
    }
}
