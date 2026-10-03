//! End-to-end vision parity: `JinaV5Vision` (Rust host math + ORT session)
//! vs the torch-native goldens from the verification spike.
//!
//! Ignored by default: needs an ONNX Runtime dylib, the vision weight
//! files, and a GPU for sane runtimes (fp32-CPU is ~6 s/image).
//! Run: `cargo test --offline --test vision_e2e -- --ignored` with:
//! - `ORT_DYLIB_PATH` → onnxruntime.dll (1.29.x)
//! - `VISION_E2E_FP32` → fp32 `model.onnx` (+ sidecar next to it)
//! - `VISION_E2E_FP16` → fp16 `model.onnx` (+ sidecar next to it)
//! - `VISION_E2E_TOK`  → omni `tokenizer.json`
//! - `VISION_E2E_CUDA=1` to run on CUDA (default CPU)
//!
//! Gates (the 6.1 bar): fp32 cos ≥ 0.9999, fp16 cos ≥ 0.999 vs torch.

use embroider::{vision_target_size, DeviceReq, JinaV5Vision};

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        d += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

#[test]
#[ignore]
fn vision_e2e_parity_vs_torch() {
    let (fp32, fp16, tok) = match (env("VISION_E2E_FP32"), env("VISION_E2E_FP16"), env("VISION_E2E_TOK")) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => {
            eprintln!("vision e2e skipped: set VISION_E2E_FP32/FP16/TOK");
            return;
        }
    };
    let device =
        if env("VISION_E2E_CUDA").as_deref() == Some("1") { DeviceReq::Cuda } else { DeviceReq::Cpu };

    let cases: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/vision/e2e_cases.json"
        ))
        .unwrap(),
    )
    .unwrap();

    let v32 = JinaV5Vision::open_files(
        std::path::Path::new(&fp32),
        std::path::Path::new(&tok),
        768,
        device,
        None,
    )
    .unwrap();
    let v16 = JinaV5Vision::open_files(
        std::path::Path::new(&fp16),
        std::path::Path::new(&tok),
        768,
        device,
        None,
    )
    .unwrap();
    println!("used_cuda fp32={} fp16={}", v32.used_cuda(), v16.used_cuda());

    for c in cases.as_array().unwrap() {
        let stem = c["stem"].as_str().unwrap();
        let path = format!(
            "{}/fixtures/vision/e2e/{}.png",
            env!("CARGO_MANIFEST_DIR"),
            stem
        );
        let img = image::open(&path).unwrap().to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        // Fixture PNGs are stored at valid target sizes already.
        assert_eq!(vision_target_size(h as u32, w as u32).unwrap(), (h as u32, w as u32), "{stem}");
        let want: Vec<f32> = c["torch_768"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let e32 = v32.encode_image(img.as_raw(), h, w).unwrap();
        let e16 = v16.encode_image(img.as_raw(), h, w).unwrap();
        let c32 = cos(&e32, &want);
        let c16 = cos(&e16, &want);
        let cxx = cos(&e32, &e16);
        println!("{stem}: cos fp32={c32:.5} fp16={c16:.5} fp32/fp16={cxx:.5}");
        assert!(c32 >= 0.9999, "{stem} fp32 {c32}");
        assert!(c16 >= 0.999, "{stem} fp16 {c16}");
    }
}
