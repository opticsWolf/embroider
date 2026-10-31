//! HuggingFace acquisition (blocking, cached).
//!
//! The `hf-hub` pattern both consumers used to reimplement: validated
//! `owner/name` parsing, client construction with an optional cache-dir
//! override, and the cheap tokenizer-only fetch that lets token counting
//! work without opening an ONNX session (lazy encoder lifecycle).

use crate::error::Result;
use crate::policy::Precision;
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, RwLock};

/// Default FP32 weights (Jina's official optimum export).
pub const FP32_TEXT_MODEL: &str = "jinaai/jina-embeddings-v5-text-small-retrieval";
/// FP16 mirror of the same export (same graph + contract, weights cast
/// FP32→FP16, IO kept FP32). Published alongside the quant-spike results;
/// CUDA-only by design (FP16 on CPU runs >40x slower than FP32-CPU).
pub const FP16_TEXT_MODEL: &str =
    "opticsWolf/jina-embeddings-v5-text-small-retrieval-onnx-fp16";
/// Nano tier (EuroBERT, 239M, 768-dim, 8K ctx). Jina ships official ONNX
/// in-repo (fp32 + fp16 + dynamic-int8) — no mirror needed. Measured
/// (nano probe): fp32/fp16 faithful @0.99994, int8 keeps rank @0.99980
/// with zero top-5 flips; q4/q4f16 killed @0.957 with 5/5 pos-1 flips.
pub const NANO_TEXT_MODEL: &str = "jinaai/jina-embeddings-v5-text-nano-retrieval";
/// Vision repo: the dynamic-grid omni-nano export (Phase 6). One graph
/// (vision tower + merger + EuroBERT text tower) with host-computed grid
/// inputs; sidecars use the `model.onnx.data` layout, hence the explicit
/// sidecar override below. Weights CC BY-NC 4.0 (Jina), same as the text
/// models — see the model card for attribution and the changes statement.
pub const VISION_NANO_REPO: &str =
    "opticsWolf/jina-embeddings-v5-omni-nano-retrieval-onnx";
/// Canonical vision model id. Short (not `owner/name`): the id names the
/// contract, the spec's artifacts name where the files live.
pub const VISION_NANO_MODEL: &str = "jina-v5-omni-nano-retrieval-vision";
/// Rerank spike ids (plan `docs/rerank-typed-decision-plan.md` §A0): the
/// English MiniLM-L-6 and its multilingual mMARCO sibling. Owner/name ids
/// name the contract directly; the spec's artifacts name the file layout.
pub const RERANK_MINILM_MODEL: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";
pub const RERANK_MMARCO_MODEL: &str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";
/// Decide spike id (plan `docs/typed-decision-plan.md` §8): GLiClass
/// small-v1.0, lowercased canonical id. Uni-encoder labels-spliced input
/// through the same `open_files` seam as the rerank spikes.
pub const GLICLASS_SMALL_MODEL: &str = "knowledgator/gliclass-small-v1.0";
pub const GLICLASS_V4_EDGE_MODEL: &str = "knowledgator/gliclass-v4-edge";

/// One downloadable artifact: the repo holding it plus the model file
/// inside that repo. The external-data sidecar is derived (same stem +
/// `.onnx_data`); unknown layouts warn-and-continue at fetch time.
#[derive(Clone, Copy, Debug)]
pub struct Artifact<'a> {
    pub repo: &'a str,
    pub file: &'a str,
    /// Explicit sidecar file inside the repo. `None` derives the legacy
    /// optimum layout (`onnx/model_fp16.onnx` → `onnx/model_fp16.onnx_data`);
    /// `Some` names it verbatim (the vision export's `model.onnx.data`).
    pub sidecar: Option<&'a str>,
    /// Pinned commit SHA the artifact was parity-gated at. `open()` uses
    /// it when the caller passes no revision (caller `Some` always wins);
    /// `None` means floating latest — allowed only while a pin is being
    /// established, never as a steady state (structural test below).
    pub revision: Option<&'a str>,
}

impl Artifact<'_> {
    /// Sidecar filename, or `None` for non-`.onnx` files (inline weights
    /// assumed, no sidecar).
    pub fn sidecar(self) -> Option<String> {
        if let Some(explicit) = self.sidecar {
            return Some(explicit.to_string());
        }
        self.file
            .strip_suffix(".onnx")
            .map(|stem| format!("{stem}.onnx_data"))
    }
}

/// Static model contract: everything `open()` needs beyond the id.
/// Ladders are the official Matryoshka levels (small tops at 1024,
/// nano at 768); `max_tokens` is the positional ceiling (Qwen3 32768,
/// EuroBERT 8192). Precision maps list only MEASURED artifacts —
/// unlisted (precision, model) pairs fall back to the fp32 artifact
/// rather than fetching unvalidated weights.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    pub id: &'static str,
    pub native_dim: usize,
    pub max_tokens: usize,
    pub ladder: &'static [usize],
    pub fp32: Artifact<'static>,
    pub fp16: Option<Artifact<'static>>,
    pub int8: Option<Artifact<'static>>,
    /// Text model sharing this model's vector space, if any. Vision
    /// specs set it (image vectors are only comparable to their text
    /// partner's); text specs leave it `None`.
    pub text_partner: Option<&'static str>,
}

pub const TEXT_SMALL: ModelSpec = ModelSpec {
    id: FP32_TEXT_MODEL,
    native_dim: 1024,
    max_tokens: 32768,
    ladder: &[32, 64, 128, 256, 512, 768, 1024],
    fp32: Artifact { repo: FP32_TEXT_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("6856e76bb72982e58de0620458a4e8b3614da340") },
    fp16: Some(Artifact { repo: FP16_TEXT_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("b79c38889bee6a84dd7b36c0df206afa420f4f9f") }),
    int8: None, // killed in the quant spike @0.93 drift — never shipped
    text_partner: None,
};

pub const TEXT_NANO: ModelSpec = ModelSpec {
    id: NANO_TEXT_MODEL,
    native_dim: 768,
    max_tokens: 8192,
    ladder: &[32, 64, 128, 256, 512, 768],
    fp32: Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("ac5d898c8d382b17167c33e5c8af644a3519b47d") },
    fp16: Some(Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model_fp16.onnx", sidecar: None, revision: Some("ac5d898c8d382b17167c33e5c8af644a3519b47d") }),
    int8: Some(Artifact { repo: NANO_TEXT_MODEL, file: "onnx/model_quantized.onnx", sidecar: None, revision: Some("ac5d898c8d382b17167c33e5c8af644a3519b47d") }),
    text_partner: None,
};

/// Vision contract: dynamic-grid omni-nano (Phase 6). Same 768-dim
/// Matryoshka ladder as text-nano, its `text_partner` — image vectors
/// only compare against text-nano vectors. Prompt fits ≤1295 tokens;
/// `max_tokens` carries the shared EuroBERT positional ceiling (8192).
/// fp16 recipe: ORT converter, keep_io_types, Pow/ReduceMean/Sqrt/
/// Reciprocal/Cos/Sin kept fp32, duplicate Casts deduped. No int8/q4
/// (community q4f16 killed @0.954 drift). Pinned artifact commit + per-file
/// sha256 live in the repo's `manifest.json`.
pub const VISION_NANO: ModelSpec = ModelSpec {
    id: VISION_NANO_MODEL,
    native_dim: 768,
    max_tokens: 8192,
    ladder: &[32, 64, 128, 256, 512, 768],
    fp32: Artifact {
        repo: VISION_NANO_REPO,
        file: "fp32/model.onnx",
        sidecar: Some("fp32/model.onnx.data"),
        revision: Some("95c7be07d81ded60cd105727ed9ba2d04003692d"),
    },
    fp16: Some(Artifact {
        repo: VISION_NANO_REPO,
        file: "fp16/model.onnx",
        sidecar: Some("fp16/model.onnx.data"),
        revision: Some("95c7be07d81ded60cd105727ed9ba2d04003692d"),
    }),
    int8: None,
    text_partner: Some(NANO_TEXT_MODEL),
};

/// Builtin registry. The default id stays `TEXT_SMALL.id` — adding models
/// never moves existing graphs (different weights = different vector space).
pub fn builtin_models() -> &'static [ModelSpec] {
    &[TEXT_SMALL, TEXT_NANO, VISION_NANO]
}

/// Look up a canonical id in the registry. Unknown ids (custom repos,
/// omni-small, mirrors) return `None` and take the legacy path: the id
/// itself is the fp32 repo with the default `onnx/model.onnx` layout.
pub fn lookup_model(model_id: &str) -> Option<ModelSpec> {
    builtin_models().iter().find(|m| m.id == model_id).copied()
}

/// What a task model does. Embedding specs carry vector-space metadata
/// (`native_dim`, `ladder`); task specs don't — scores and decisions
/// have no dimension to truncate, only a positional ceiling.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskKind {
    Rerank,
    Decide,
    /// Marker-layout typed decisions (Laya): same question shapes as
    /// `Decide`, different input contract (marker positions, not label
    /// text) — served by `LayaDecider`, refused by `Decider`.
    DecideLaya,
    /// Slot-layout typed decisions (decider-2b): same question shapes as
    /// `Decide`, decoder input contract (`input_ids` + `slot_pos`, no
    /// attention mask) — served by `Decider2B`, refused by `Decider`.
    Decide2B,
}

impl TaskKind {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskKind::Rerank => "rerank",
            TaskKind::Decide => "decide",
            TaskKind::DecideLaya => "decide-laya",
            TaskKind::Decide2B => "decide-2b",
        }
    }

    /// Parse a registry `kind` string (file or API). Unknown kinds are
    /// refused with a pointer at the layer-2 boundary: a new input
    /// contract needs a Rust loader, not a registry row.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "rerank" => Ok(TaskKind::Rerank),
            "decide" => Ok(TaskKind::Decide),
            "decide-laya" | "decidelaya" | "laya" => Ok(TaskKind::DecideLaya),
            "decide-2b" | "decide2b" | "decider2b" => Ok(TaskKind::Decide2B),
            other => Err(anyhow!(
                "unknown task kind '{other}': expected 'rerank', 'decide', \
                 'decide-laya' or 'decide-2b' — a new input contract needs a \
                 Rust loader (see docs/model-registry-draft.md layer 2), not a registry row"
            )),
        }
    }
}

/// Absolute per-kind ceiling for registered `max_tokens`: no overlay
/// entry may exceed the largest builtin of its kind (session, policy
/// and export bounds assume it — e.g. `LAYA_ABSOLUTE_MAX` is baked
/// into the traced exports). Builtins keep their own ceilings.
pub fn kind_ceiling(kind: TaskKind) -> usize {
    match kind {
        TaskKind::Rerank => 512,
        TaskKind::Decide => 2048,
        TaskKind::DecideLaya => 1024,
        TaskKind::Decide2B => 32768,
    }
}

/// Required config files beyond graph + tokenizer, per kind. Callers
/// may extend, never shrink (validated at registration).
pub fn kind_extra_files(kind: TaskKind) -> &'static [&'static str] {
    match kind {
        TaskKind::Rerank | TaskKind::Decide => &[],
        TaskKind::DecideLaya => &["rl_agent_config.json"],
        TaskKind::Decide2B => &["decision.json", "calibration.json"],
    }
}

/// Static contract for a non-embedding model: rerankers (`Rerank`) and
/// typed-decision models (`Decide`). Same artifact mechanics as
/// [`ModelSpec`] (fp32 default, measured precisions only — unlisted
/// (precision, model) pairs fall back to the fp32 artifact), minus the
/// vector-space fields task outputs don't have.
#[derive(Clone, Copy, Debug)]
pub struct TaskSpec {
    pub id: &'static str,
    pub kind: TaskKind,
    pub max_tokens: usize,
    /// Input layout: false = text-first (`state + labels + [SEP]`,
    /// small-v1.0 recipe), true = labels-first (`labels + [SEP] +
    /// prompt? + state`, v4-edge recipe). Part of the contract — a
    /// wrong layout scores garbage at parity, never an error.
    pub prompt_first: bool,
    pub fp32: Artifact<'static>,
    pub fp16: Option<Artifact<'static>>,
    pub int8: Option<Artifact<'static>>,
    /// Weight-only int8 tier (measured, 0 flips — see `Precision::Q8`).
    /// Unlisted pairs fall back to fp32 (same rule as fp16/int8).
    pub q8: Option<Artifact<'static>>,
}

/// Rerank spike, English (plan §A0): MiniLM-L-6 cross-encoder, 22.7M.
/// In-repo ONNX (`onnx/model.onnx`) + `tokenizer.json`; only the fp32
/// tier is listed until a precision is measured (the quant-spike rule).
pub const RERANK_MINILM: TaskSpec = TaskSpec {
    id: RERANK_MINILM_MODEL,
    kind: TaskKind::Rerank,
    max_tokens: 512,
    prompt_first: false,
    fp32: Artifact { repo: RERANK_MINILM_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("233902d25c440f23af6f7d6e94d2946bac0bee0a") },
    fp16: None,
    int8: None,
    q8: Some(Artifact {
        repo: "opticsWolf/ms-marco-MiniLM-L-6-v2-q8-onnx",
        file: "model_q8.onnx",
        sidecar: Some("model_q8.onnx.data"),
        revision: Some("98b39029a16a2c3f4e881119755a6d24cea51b10"),
    }),
};

/// Rerank spike, multilingual (plan §A0): mMARCO MiniLMv2-L12-H384, 118M.
/// Same `onnx/model.onnx` in-repo layout (not the O1–O4 optimum
/// variants); XLM-R arch + SentencePiece tokenizer exercise the loader's
/// second family through the same `open_files` seam.
pub const RERANK_MMARCO: TaskSpec = TaskSpec {
    id: RERANK_MMARCO_MODEL,
    kind: TaskKind::Rerank,
    max_tokens: 512,
    prompt_first: false,
    fp32: Artifact { repo: RERANK_MMARCO_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("1427fd652930e4ba29e8149678df786c240d8825") },
    fp16: None,
    int8: None,
    q8: Some(Artifact {
        repo: "opticsWolf/mmarco-mMiniLMv2-L12-H384-v1-q8-onnx",
        file: "model_q8.onnx",
        sidecar: Some("model_q8.onnx.data"),
        revision: Some("4230967c48f473f9941f6aa3aa86c6e0d9a80fbc"),
    }),
};

/// Decide spike, English single/multi-label (plan §8): GLiClass-small
/// v1.0, 144M. In-repo ONNX (`onnx/model.onnx`) + root `tokenizer.json`
/// at pinned revision `21edefaf`; only the fp32 tier is listed until a
/// precision is measured (the quant-spike rule — the in-repo int8 waits
/// for its own harness, same as the rerank int8). 512 is the DeBERTa-v3
/// positional ceiling and the pipeline default.
pub const GLICLASS_SMALL: TaskSpec = TaskSpec {
    id: GLICLASS_SMALL_MODEL,
    kind: TaskKind::Decide,
    max_tokens: 512,
    prompt_first: false,
    fp32: Artifact { repo: GLICLASS_SMALL_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("21edefaf7951f68c68c505f9139ba536d3b448f7") },
    fp16: None,
    int8: None,
    // No q8: the weight-only build flips 2 knife-edge travel cases
    // (0.11-0.25 drift, block 32, head excluded) — the 0-flip gate holds.
    q8: None,
};

/// Decide, English single/multi-label + instruction prompts (plan
/// §8b): GLiClass v4-edge, 440M, DeBERTa-v3-class encoder. Labels-first
/// recipe (`prompt_first: true` — labels + SEP + prompt + text, the
/// v4 training order), 2048-token ceiling per the model card's usage,
/// in-repo ONNX (`onnx/model.onnx`) + root `tokenizer.json` at pinned
/// revision `885af1af`; q8 tier at `opticsWolf/gliclass-v4-edge-q8-onnx`
/// (worst 0.090, 0 flips — the dynamic-int8 route stays rejected, the
/// weight-only route passed). The graph
/// embeds the held-out-fitted calibrator (B0-proven: ORT matches the
/// calibrated branch, mismatches raw by 7.9), so goldens pin ORT
/// values at a 0.03 tolerance, not 1e-6.
pub const GLICLASS_V4_EDGE: TaskSpec = TaskSpec {
    id: GLICLASS_V4_EDGE_MODEL,
    kind: TaskKind::Decide,
    max_tokens: 2048,
    prompt_first: true,
    fp32: Artifact { repo: GLICLASS_V4_EDGE_MODEL, file: "onnx/model.onnx", sidecar: None, revision: Some("885af1af8a07edf4ed34df83ca0d995ed8201283") },
    fp16: None,
    int8: None,
    q8: Some(Artifact {
        repo: "opticsWolf/gliclass-v4-edge-q8-onnx",
        file: "model_q8.onnx",
        sidecar: Some("model_q8.onnx.data"),
        revision: Some("b1fd271a14b8bd69c70b494e167d494c9b0e5235"),
    }),
};

/// Laya checkpoints (plan §9b): native typed questions (choice with
/// descriptions, level-scores, noul) over the marker layout. Canonical
/// ids are upstream (`convaiinnovations/laya-*`); the fp32 artifacts are
/// the `opticsWolf/laya-*-fp32-onnx` self-exports (upstream ships no
/// ONNX — torch `dynamo=True`, opset 18, explicit dynamic dims
/// batch/seq≤1024/k≤256). Served by `LayaDecider` (`src/laya.rs`);
/// `prompt_first` is the GLiClass-layout axis and does not apply (the
/// marker layout is intrinsic). fp32 only — fp16 measured (drifts
/// 1.5–4.3e-3, CUDA-only relief) and wires up as a serving variant on
/// its own parity gate; int8 rejected on three arms (dynamic 2.8–5.4
/// with flips, static 9997 on the −1e4 mask sentinel, encoder-only
/// still 2.85 with no CPU speedup).
///
/// - typed-decisions @ `1a793eb` (421M ModernBERT-large, max_len 1024 /
///   head 256, fitted per-bucket temps): the classifier-duty checkpoint.
/// - base English @ `55cf4c4e` (512/192): near-chance zero-shot per
///   Ollaya — export faithful, not for classifier duty.
/// - multilingual @ `e4e9ddf2` (mmBERT-base, 1024/256): fastest on CPU.

/// Upstream id: the typed-decisions checkpoint (classifier duty).
pub const LAYA_TD_MODEL: &str = "convaiinnovations/laya-typed-decisions";
/// Upstream id: the base English checkpoint.
pub const LAYA_BASE_MODEL: &str = "convaiinnovations/laya";
/// Upstream id: the multilingual checkpoint.
pub const LAYA_MULTI_MODEL: &str = "convaiinnovations/laya-multilingual";

/// Traced sequence bound all three Laya exports bake in (dynamicDims
/// seq≤1024): explicit `max_length` can never exceed it.
pub const LAYA_ABSOLUTE_MAX: usize = 1024;
/// Upstream id: Mapika's Qwen3.5-2B decoder with the slot_logits head
/// (decider-ai 1.10.0, v12 @ `7614a6c7`). Served by `Decider2B`
/// (`src/decider2b.rs`): the 7.6MB graph plus the 3.76GB vendored
/// safetensors, constructed Ollaya-side from the torch checkpoint
/// (dynamo structurally hung). fp32 only — no precision story exists
/// for this export.
pub const DECIDER2B_MODEL: &str = "Mapika/decider-2b";
/// `open_files` default ceiling (registry ids resolve their own).
pub const LAYA_DEFAULT_MAX: usize = 1024;

pub const LAYA_TD: TaskSpec = TaskSpec {
    id: LAYA_TD_MODEL,
    kind: TaskKind::DecideLaya,
    max_tokens: 1024,
    prompt_first: false,
    fp32: Artifact {
        repo: "opticsWolf/laya-typed-decisions-fp32-onnx",
        file: "laya-td-fp32.onnx",
        sidecar: Some("laya-td-fp32.onnx.data"),
        revision: Some("fa9cac92fbb1914b3ac0f4a2a8b2ec244444db98"),
    },
    fp16: None,
    int8: None,
    q8: Some(Artifact {
        repo: "opticsWolf/laya-typed-decisions-fp32-onnx",
        file: "laya-td-q8.onnx",
        sidecar: Some("laya-td-q8.onnx.data"),
        revision: Some("fa9cac92fbb1914b3ac0f4a2a8b2ec244444db98"),
    }),
};

pub const LAYA_BASE: TaskSpec = TaskSpec {
    id: LAYA_BASE_MODEL,
    kind: TaskKind::DecideLaya,
    max_tokens: 512,
    prompt_first: false,
    fp32: Artifact {
        repo: "opticsWolf/laya-english-fp32-onnx",
        file: "laya-base-fp32.onnx",
        sidecar: Some("laya-base-fp32.onnx.data"),
        revision: Some("26dbfc689b8cdaf30cb5870206c463a492da2f26"),
    },
    fp16: None,
    int8: None,
    q8: Some(Artifact {
        repo: "opticsWolf/laya-english-fp32-onnx",
        file: "laya-base-q8.onnx",
        sidecar: Some("laya-base-q8.onnx.data"),
        revision: Some("26dbfc689b8cdaf30cb5870206c463a492da2f26"),
    }),
};

pub const LAYA_MULTI: TaskSpec = TaskSpec {
    id: LAYA_MULTI_MODEL,
    kind: TaskKind::DecideLaya,
    max_tokens: 1024,
    prompt_first: false,
    fp32: Artifact {
        repo: "opticsWolf/laya-multilingual-fp32-onnx",
        file: "laya-multi-fp32.onnx",
        sidecar: Some("laya-multi-fp32.onnx.data"),
        revision: Some("a20896963b9bd0216ba33b20d699bc9e1a716734"),
    },
    fp16: None,
    int8: None,
    q8: Some(Artifact {
        repo: "opticsWolf/laya-multilingual-fp32-onnx",
        file: "laya-multi-q8.onnx",
        sidecar: Some("laya-multi-q8.onnx.data"),
        revision: Some("a20896963b9bd0216ba33b20d699bc9e1a716734"),
    }),
};

/// Slot-layout typed decisions (plan `docs/typed-decision-plan.md` §10):
/// decider-2b, Qwen3.5-2B decoder + slot head, 32k context. Two inputs
/// (`input_ids`, `slot_pos`), one output (`label_logits` [rows, 255]);
/// no attention mask (causal; right padding never reaches earlier
/// positions). The `slot_pos` rows plus per-qtype temperatures in
/// `decision.json`/`calibration.json` are the contract — `open()`
/// fetches both sidecars alongside the graph + vendored safetensors +
/// tokenizer. fp32 only (no precision story); CUDA sessions run the
/// parallel executor (the sequential planner aliases a shape buffer on
/// this graph — ORT #4829 class — corrupting a chunk-scan Reshape;
/// `enable_mem_reuse=false` is not reachable from the C API, parallel
/// execution is the sanctioned equivalent, parity-pinned in e2e).
pub const DECIDER2B: TaskSpec = TaskSpec {
    id: DECIDER2B_MODEL,
    kind: TaskKind::Decide2B,
    max_tokens: 32768,
    prompt_first: false,
    fp32: Artifact {
        repo: "opticsWolf/decider-2b-fp32-onnx",
        file: "model.onnx",
        sidecar: Some("model.safetensors"),
        revision: Some("a951b9fe4647f8d6defb4ff0c66b7511960b101d"),
    },
    fp16: None,
    int8: None,
    // q8 graph still reads embeddings/head/norms from model.safetensors —
    // Decider2B::open fetches it alongside the q8 sidecar (see open).
    q8: Some(Artifact {
        repo: "opticsWolf/decider-2b-fp32-onnx",
        file: "model_q8.onnx",
        sidecar: Some("model_q8.onnx.data"),
        revision: Some("a951b9fe4647f8d6defb4ff0c66b7511960b101d"),
    }),
};

/// Builtin task registry. Adding entries never moves existing graphs —
/// different models, different outputs, no shared vector space.
pub fn builtin_tasks() -> &'static [TaskSpec] {
    &[
        RERANK_MINILM,
        RERANK_MMARCO,
        GLICLASS_SMALL,
        GLICLASS_V4_EDGE,
        LAYA_TD,
        LAYA_BASE,
        LAYA_MULTI,
        DECIDER2B,
    ]
}

/// Look up a canonical id in the task registry. Unknown ids return
/// `None` and take the legacy path (same convention as
/// [`lookup_model`]).
pub fn lookup_task(model_id: &str) -> Option<TaskSpec> {
    builtin_tasks().iter().find(|m| m.id == model_id).copied()
}

/// Where a resolved spec came from: frozen consts or the overlay.
/// Surfaced in `available_models()` so consumers can key off it
/// without parsing ids.
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RegistrySource {
    Builtin,
    Registered,
}

impl RegistrySource {
    pub fn as_str(self) -> &'static str {
        match self {
            RegistrySource::Builtin => "builtin",
            RegistrySource::Registered => "registered",
        }
    }
}

/// Machine-readable provenance (JSON has no comments, so the drift
/// numbers and gate notes this repo keeps in comments become fields).
/// Serialized verbatim into `available_models()`; `open()` warns from
/// `measured: false`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub measured: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worst_drift: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flips: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl Provenance {
    /// Builtin tiers: measured by construction (the parity gates).
    pub fn measured() -> Self {
        Self { measured: true, worst_drift: None, flips: None, harness: None, notes: None }
    }

    /// Overlay default: unmeasured until the caller claims otherwise
    /// (flagged-allow policy — the claim rides the entry, the warning
    /// rides `open()`).
    pub fn unclaimed() -> Self {
        Self { measured: false, worst_drift: None, flips: None, harness: None, notes: None }
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self::unclaimed()
    }
}

/// Owned twin of [`Artifact`]: the same acquisition contract with heap
/// strings, so registry files and API calls can build it at runtime.
/// `sidecar` follows [`Artifact::sidecar`] semantics.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredArtifact {
    pub repo: String,
    pub file: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidecar: Option<String>,
    /// Pinned commit SHA the tier was gated at. `open()` uses it when
    /// the caller passes no revision (caller `Some` always wins).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

/// Owned twin of [`TaskSpec`]: everything `open()` needs beyond the id,
/// buildable at runtime. `kind` is an existing [`TaskKind`] only —
/// unknown kinds are refused at parse (layer-2 boundary).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredTask {
    pub id: String,
    pub kind: TaskKind,
    pub max_tokens: usize,
    #[serde(default)]
    pub prompt_first: bool,
    pub fp32: RegisteredArtifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp16: Option<RegisteredArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub int8: Option<RegisteredArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub q8: Option<RegisteredArtifact>,
    #[serde(default)]
    pub provenance: Provenance,
    /// Assigned by the registration front door (`"api"` or
    /// `"path#index"`), always overwritten there, and named in
    /// `open()` warnings.
    #[serde(default)]
    pub registered_from: String,
    /// Caller-supplied config files beyond the kind defaults.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_files: Vec<String>,
}

/// Owned twin of [`ModelSpec`], minus the q8 tier (no embed q8 exists).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegisteredModel {
    pub id: String,
    pub native_dim: usize,
    pub max_tokens: usize,
    pub ladder: Vec<usize>,
    pub fp32: RegisteredArtifact,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp16: Option<RegisteredArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub int8: Option<RegisteredArtifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_partner: Option<String>,
    #[serde(default)]
    pub provenance: Provenance,
    #[serde(default)]
    pub registered_from: String,
}

/// Absolute ceiling for registered embed `max_tokens` (no kind axis on
/// the embed side; builtins keep their own ceilings).
const EMBED_ABSOLUTE_MAX: usize = 32768;

/// Runtime overlay: validated entries only (see `validate_task` /
/// `validate_model`). Builtins are consulted first and can never be
/// shadowed — the frozen contract means a builtin id resolves
/// identically in every process.
static TASK_OVERLAY: LazyLock<RwLock<HashMap<String, RegisteredTask>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));
static MODEL_OVERLAY: LazyLock<RwLock<HashMap<String, RegisteredModel>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

fn validate_artifact(where_: &str, a: &RegisteredArtifact) -> Result<()> {
    parse_owner_name(&a.repo).map_err(|e| anyhow!("{where_}: bad repo: {e}"))?;
    if !a.file.ends_with(".onnx") {
        return Err(anyhow!("{where_}: artifact file must end in .onnx, got '{}'", a.file));
    }
    if let Some(sc) = a.sidecar.as_deref() {
        if !(sc.ends_with(".data") || sc.ends_with("_data") || sc.ends_with(".safetensors")) {
            return Err(anyhow!(
                "{where_}: sidecar must end in .data, _data or .safetensors, got '{sc}'"
            ));
        }
    }
    if let Some(rev) = a.revision.as_deref() {
        if rev.trim().is_empty() || rev.chars().any(char::is_whitespace) {
            return Err(anyhow!("{where_}: revision must be a non-empty ref/SHA, got '{rev}'"));
        }
    }
    Ok(())
}

/// Validate a task entry before it touches the overlay: id shape, kind
/// ceiling, artifact layouts, kind file contract. No I/O — validation
/// is data, downloading is `open()`'s job. `prompt_first` on a
/// non-`Decide` kind warns (the marker/slot layouts are intrinsic —
/// the flag would be a lie) but registers: warn, don't refuse.
pub fn validate_task(entry: &RegisteredTask) -> Result<()> {
    // Overlay ids must be `owner/name` — no short-id exception (the
    // builtin vision short id is grandfathered, not a pattern).
    parse_owner_name(&entry.id)
        .map_err(|e| anyhow!("task id must be 'owner/name', got '{}': {e}", entry.id))?;
    let ceiling = kind_ceiling(entry.kind);
    if entry.max_tokens == 0 || entry.max_tokens > ceiling {
        return Err(anyhow!(
            "task '{}': max_tokens {} outside 1..={} for kind '{}'",
            entry.id, entry.max_tokens, ceiling, entry.kind.as_str()
        ));
    }
    validate_artifact(&format!("task '{}' fp32", entry.id), &entry.fp32)?;
    for (tier, art) in [("fp16", &entry.fp16), ("int8", &entry.int8), ("q8", &entry.q8)]
        .into_iter()
        .filter_map(|(t, a)| a.as_ref().map(|a| (t, a)))
    {
        validate_artifact(&format!("task '{}' {tier}", entry.id), art)?;
    }
    if entry.prompt_first && entry.kind != TaskKind::Decide {
        eprintln!(
            "embroider: task '{}' sets prompt_first on kind '{}' — the marker/slot \
             layouts are intrinsic, the flag has no effect and is ignored",
            entry.id, entry.kind.as_str()
        );
    }
    for need in kind_extra_files(entry.kind) {
        if !entry.extra_files.iter().any(|f| f == need) {
            return Err(anyhow!(
                "task '{}': kind '{}' requires '{need}' in extra_files (defaults apply — \
                 extend, never shrink)",
                entry.id, entry.kind.as_str()
            ));
        }
    }
    // The q8 fetch seam for the slot layout reads the weights file out
    // of extra_files (see Decider2B::open) — require it up front.
    if entry.kind == TaskKind::Decide2B
        && !entry.extra_files.iter().any(|f| f.ends_with(".safetensors"))
    {
        return Err(anyhow!(
            "task '{}': kind 'decide-2b' requires a .safetensors weights file in extra_files \
             (embeddings/head/norms live outside the q8 graph)",
            entry.id
        ));
    }
    Ok(())
}

/// Validate an embed entry: id shape, positive dims, a sane ladder
/// (non-empty, strictly ascending, within `native_dim`), artifact
/// layouts. `text_partner` may forward-reference an id registered
/// later — no existence check.
pub fn validate_model(entry: &RegisteredModel) -> Result<()> {
    parse_owner_name(&entry.id)
        .map_err(|e| anyhow!("model id must be 'owner/name', got '{}': {e}", entry.id))?;
    if entry.native_dim == 0 {
        return Err(anyhow!("model '{}': native_dim must be positive", entry.id));
    }
    if entry.max_tokens == 0 || entry.max_tokens > EMBED_ABSOLUTE_MAX {
        return Err(anyhow!(
            "model '{}': max_tokens {} outside 1..={EMBED_ABSOLUTE_MAX}",
            entry.id, entry.max_tokens
        ));
    }
    if entry.ladder.is_empty()
        || entry.ladder.windows(2).any(|w| w[0] >= w[1])
        || entry.ladder.iter().any(|&d| d == 0 || d > entry.native_dim)
    {
        return Err(anyhow!(
            "model '{}': ladder must be non-empty, strictly ascending and within native_dim {}",
            entry.id, entry.native_dim
        ));
    }
    validate_artifact(&format!("model '{}' fp32", entry.id), &entry.fp32)?;
    for (tier, art) in [("fp16", &entry.fp16), ("int8", &entry.int8)]
        .into_iter()
        .filter_map(|(t, a)| a.as_ref().map(|a| (t, a)))
    {
        validate_artifact(&format!("model '{}' {tier}", entry.id), art)?;
    }
    Ok(())
}

/// Register a validated task entry. Refuses builtin ids and duplicate
/// registrations loudly — shadowing would fork the frozen contract.
/// `registered_from` defaults to `"api"` when the caller leaves it
/// empty (the file loader stamps `"path#index"`).
pub fn register_task(mut entry: RegisteredTask) -> Result<RegisteredTask> {
    validate_task(&entry)?;
    if lookup_task(&entry.id).is_some() {
        return Err(anyhow!(
            "cannot register task '{}': a builtin model owns this id — mirrors take their own ids \
             (the opticsWolf/*-q8-onnx convention)",
            entry.id
        ));
    }
    if entry.registered_from.is_empty() {
        entry.registered_from = "api".to_string();
    }
    let mut overlay = TASK_OVERLAY.write().map_err(|e| anyhow!("task registry lock: {e}"))?;
    if overlay.contains_key(&entry.id) {
        return Err(anyhow!("task '{}' is already registered", entry.id));
    }
    overlay.insert(entry.id.clone(), entry.clone());
    Ok(entry)
}

/// Register a validated embed entry (same shadowing rule as tasks).
pub fn register_model(mut entry: RegisteredModel) -> Result<RegisteredModel> {
    validate_model(&entry)?;
    if lookup_model(&entry.id).is_some() {
        return Err(anyhow!(
            "cannot register model '{}': a builtin model owns this id — mirrors take their own ids",
            entry.id
        ));
    }
    if entry.registered_from.is_empty() {
        entry.registered_from = "api".to_string();
    }
    let mut overlay = MODEL_OVERLAY.write().map_err(|e| anyhow!("model registry lock: {e}"))?;
    if overlay.contains_key(&entry.id) {
        return Err(anyhow!("model '{}' is already registered", entry.id));
    }
    overlay.insert(entry.id.clone(), entry.clone());
    Ok(entry)
}

/// Remove an overlay task entry. Test/correction seam (statics persist
/// across tests in one process); builtins are not removable.
pub fn unregister_task(model_id: &str) -> bool {
    TASK_OVERLAY.write().map(|mut o| o.remove(model_id).is_some()).unwrap_or(false)
}

/// Remove an overlay embed entry (same test/correction seam).
pub fn unregister_model(model_id: &str) -> bool {
    MODEL_OVERLAY.write().map(|mut o| o.remove(model_id).is_some()).unwrap_or(false)
}

/// Owned acquisition artifact: the same contract as [`Artifact`] with
/// heap strings, so builtin consts and overlay entries resolve through
/// one type. `revision` is the tier pin `open()` defaults to when the
/// caller passes none.
#[derive(Clone, Debug)]
pub struct ResolvedArtifact {
    pub repo: String,
    pub file: String,
    pub sidecar: Option<String>,
    pub revision: Option<String>,
}

impl ResolvedArtifact {
    /// Sidecar filename, same rule as [`Artifact::sidecar`].
    pub fn sidecar(&self) -> Option<String> {
        if let Some(explicit) = self.sidecar.as_deref() {
            return Some(explicit.to_string());
        }
        self.file.strip_suffix(".onnx").map(|stem| format!("{stem}.onnx_data"))
    }
}

impl From<Artifact<'static>> for ResolvedArtifact {
    fn from(a: Artifact<'static>) -> Self {
        Self {
            repo: a.repo.to_string(),
            file: a.file.to_string(),
            sidecar: a.sidecar.map(str::to_string),
            revision: a.revision.map(str::to_string),
        }
    }
}

impl From<&RegisteredArtifact> for ResolvedArtifact {
    fn from(a: &RegisteredArtifact) -> Self {
        Self {
            repo: a.repo.clone(),
            file: a.file.clone(),
            sidecar: a.sidecar.clone(),
            revision: a.revision.clone(),
        }
    }
}

/// Owned resolution of a task id: builtins convert, overlay clones.
/// `open()` paths take this instead of `lookup_task` so registered
/// models flow through the same seams (ceilings, artifacts, sidecars).
#[derive(Clone, Debug)]
pub struct ResolvedTask {
    pub kind: TaskKind,
    pub max_tokens: usize,
    pub prompt_first: bool,
    pub fp32: ResolvedArtifact,
    pub fp16: Option<ResolvedArtifact>,
    pub int8: Option<ResolvedArtifact>,
    pub q8: Option<ResolvedArtifact>,
    pub extra_files: Vec<String>,
    pub source: RegistrySource,
    pub measured: bool,
    /// `"builtin"`, `"api"` or `"path#index"` — named in warnings.
    pub registered_from: String,
}

/// Warn when a session runs caller-supplied weights: registered +
/// unmeasured + a non-fp32 tier. fp32 is byte-exact serving (no drift
/// claim involved), so the safe default stays quiet; quant tiers carry
/// the caller's gate, stated here with the entry's attribution.
fn warn_if_unmeasured(source: RegistrySource, measured: bool, from: &str, model_id: &str, precision: Precision) {
    if source == RegistrySource::Registered && !measured && precision != Precision::Fp32 {
        eprintln!(
            "embroider: '{model_id}' at precision '{}' is caller-supplied (registered from {from}, measured: false) — \
             the drift is the caller's gate, not a parity pin",
            precision.as_str()
        );
    }
}

/// Task-side unmeasured-tier warning (see above).
pub fn warn_if_unmeasured_task(spec: &ResolvedTask, model_id: &str, precision: Precision) {
    warn_if_unmeasured(spec.source, spec.measured, &spec.registered_from, model_id, precision);
}

/// Embed-side unmeasured-tier warning (same policy as tasks).
pub fn warn_if_unmeasured_model(spec: &ResolvedModel, model_id: &str, precision: Precision) {
    warn_if_unmeasured(spec.source, spec.measured, &spec.registered_from, model_id, precision);
}

/// Owned resolution of an embed id (builtins convert, overlay clones).
#[derive(Clone, Debug)]
pub struct ResolvedModel {
    pub native_dim: usize,
    pub max_tokens: usize,
    pub ladder: Vec<usize>,
    pub fp32: ResolvedArtifact,
    pub fp16: Option<ResolvedArtifact>,
    pub int8: Option<ResolvedArtifact>,
    pub text_partner: Option<String>,
    pub source: RegistrySource,
    pub measured: bool,
    /// `"builtin"`, `"api"` or `"path#index"` — named in warnings.
    pub registered_from: String,
}

/// Resolve a task id: builtins first, overlay second, `None` for fully
/// unknown ids (the legacy path).
pub fn resolve_task(model_id: &str) -> Option<ResolvedTask> {
    if let Some(spec) = lookup_task(model_id) {
        return Some(ResolvedTask {
            kind: spec.kind,
            max_tokens: spec.max_tokens,
            prompt_first: spec.prompt_first,
            fp32: spec.fp32.into(),
            fp16: spec.fp16.map(ResolvedArtifact::from),
            int8: spec.int8.map(ResolvedArtifact::from),
            q8: spec.q8.map(ResolvedArtifact::from),
            extra_files: kind_extra_files(spec.kind).iter().map(|s| s.to_string()).collect(),
            source: RegistrySource::Builtin,
            measured: true,
            registered_from: "builtin".to_string(),
        });
    }
    TASK_OVERLAY
        .read()
        .ok()
        .and_then(|o| o.get(model_id).cloned())
        .map(|e| ResolvedTask {
            kind: e.kind,
            max_tokens: e.max_tokens,
            prompt_first: e.prompt_first,
            fp32: ResolvedArtifact::from(&e.fp32),
            fp16: e.fp16.as_ref().map(ResolvedArtifact::from),
            int8: e.int8.as_ref().map(ResolvedArtifact::from),
            q8: e.q8.as_ref().map(ResolvedArtifact::from),
            // Kind defaults always apply — the entry may only extend.
            extra_files: {
                let mut files: Vec<String> =
                    kind_extra_files(e.kind).iter().map(|s| s.to_string()).collect();
                for f in e.extra_files {
                    if !files.contains(&f) {
                        files.push(f);
                    }
                }
                files
            },
            source: RegistrySource::Registered,
            measured: e.provenance.measured,
            registered_from: e.registered_from.clone(),
        })
}

/// Resolve an embed id: builtins first, overlay second, `None` for
/// fully unknown ids.
pub fn resolve_model(model_id: &str) -> Option<ResolvedModel> {
    if let Some(spec) = lookup_model(model_id) {
        return Some(ResolvedModel {
            native_dim: spec.native_dim,
            max_tokens: spec.max_tokens,
            ladder: spec.ladder.to_vec(),
            fp32: spec.fp32.into(),
            fp16: spec.fp16.map(ResolvedArtifact::from),
            int8: spec.int8.map(ResolvedArtifact::from),
            text_partner: spec.text_partner.map(str::to_string),
            source: RegistrySource::Builtin,
            measured: true,
            registered_from: "builtin".to_string(),
        });
    }
    MODEL_OVERLAY
        .read()
        .ok()
        .and_then(|o| o.get(model_id).cloned())
        .map(|e| ResolvedModel {
            native_dim: e.native_dim,
            max_tokens: e.max_tokens,
            ladder: e.ladder.clone(),
            fp32: ResolvedArtifact::from(&e.fp32),
            fp16: e.fp16.as_ref().map(ResolvedArtifact::from),
            int8: e.int8.as_ref().map(ResolvedArtifact::from),
            text_partner: e.text_partner.clone(),
            source: RegistrySource::Registered,
            measured: e.provenance.measured,
            registered_from: e.registered_from.clone(),
        })
}

/// Registry file envelope: `{"registry_version": 1, "tasks": [...],
/// "models": [...]}`. The version is required — v2 files fail loudly
/// instead of mis-parsing. Entries validate through the identical
/// `validate_*` as API registration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    registry_version: u32,
    #[serde(default)]
    tasks: Vec<RegisteredTask>,
    #[serde(default)]
    models: Vec<RegisteredModel>,
}

/// What one `load_registry_file` call added.
#[derive(Clone, Debug, Serialize)]
pub struct RegistrySummary {
    pub path: String,
    pub tasks_registered: usize,
    pub models_registered: usize,
}

/// Load a registry file into the overlay: parse (serde, strict),
/// check the version, validate + register each entry with
/// `registered_from` stamped `"path#index"`. No auto-discovery, no
/// cwd sniffing — the caller names the file (or sets
/// `EMBROIDER_REGISTRY_FILE`). Entries are data: a bad file fails
/// here, before any download.
pub fn load_registry_file(path: &Path) -> Result<RegistrySummary> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("registry file '{}' unreadable: {e}", path.display()))?;
    let file: RegistryFile = serde_json::from_str(&text)
        .map_err(|e| anyhow!("registry file '{}' parses: {e}", path.display()))?;
    if file.registry_version != 1 {
        return Err(anyhow!(
            "registry file '{}': registry_version {} unsupported (this embroider reads version 1) — \
             refusing to mis-parse a newer schema",
            path.display(), file.registry_version
        ));
    }
    let mut tasks_registered = 0;
    for (i, mut entry) in file.tasks.into_iter().enumerate() {
        entry.registered_from = format!("{}#{i}", path.display());
        register_task(entry)?;
        tasks_registered += 1;
    }
    let mut models_registered = 0;
    for (i, mut entry) in file.models.into_iter().enumerate() {
        entry.registered_from = format!("{}#{i}", path.display());
        register_model(entry)?;
        models_registered += 1;
    }
    Ok(RegistrySummary {
        path: path.display().to_string(),
        tasks_registered,
        models_registered,
    })
}

/// Snapshot of the task overlay (sorted by id, for `available_models`
/// and `export_registry`).
pub fn registered_tasks() -> Vec<RegisteredTask> {
    let mut tasks: Vec<RegisteredTask> = TASK_OVERLAY
        .read()
        .map(|o| o.values().cloned().collect())
        .unwrap_or_default();
    tasks.sort_by(|a, b| a.id.cmp(&b.id));
    tasks
}

/// Snapshot of the embed overlay (sorted by id).
pub fn registered_models() -> Vec<RegisteredModel> {
    let mut models: Vec<RegisteredModel> = MODEL_OVERLAY
        .read()
        .map(|o| o.values().cloned().collect())
        .unwrap_or_default();
    models.sort_by(|a, b| a.id.cmp(&b.id));
    models
}

/// Export the overlay (registered entries only — builtins cannot be
/// re-registered, so a merged export would not round-trip) as a
/// versioned registry file. Deterministic: entries sorted by id.
pub fn export_registry() -> Result<String> {
    let tasks = registered_tasks();
    let models = registered_models();
    serde_json::to_string_pretty(&serde_json::json!({
        "registry_version": 1,
        "tasks": tasks,
        "models": models,
    }))
    .map_err(|e| anyhow!("registry export serializes: {e}"))
}

/// Select the acquisition artifact for a (task model, precision) request.
/// `Auto` must already be resolved via `Precision::resolve` before
/// calling (auto never yields `Int8`/`Q8`). Unlisted pairs fall back to
/// the spec fp32 artifact; unknown ids fall back to the legacy layout.
pub fn artifact_for_task(model_id: &str, precision: Precision) -> ResolvedArtifact {
    match resolve_task(model_id) {
        Some(spec) => match precision {
            Precision::Fp16 => spec.fp16.unwrap_or(spec.fp32),
            Precision::Int8 => spec.int8.unwrap_or(spec.fp32),
            Precision::Q8 => spec.q8.unwrap_or(spec.fp32),
            _ => spec.fp32,
        },
        None => ResolvedArtifact {
            repo: model_id.to_string(),
            file: "onnx/model.onnx".to_string(),
            sidecar: None,
            revision: None,
        },
    }
}

/// Select the acquisition artifact for a (model, precision) request.
/// `Auto` must already be resolved via `Precision::resolve` before
/// calling (auto never yields `Int8`). Unlisted pairs fall back to the
/// spec fp32 artifact; unknown ids fall back to the legacy layout.
pub fn artifact_for(model_id: &str, precision: Precision) -> ResolvedArtifact {
    match resolve_model(model_id) {
        Some(spec) => match precision {
            Precision::Fp16 => spec.fp16.unwrap_or(spec.fp32),
            Precision::Int8 => spec.int8.unwrap_or(spec.fp32),
            _ => spec.fp32,
        },
        None => ResolvedArtifact {
            repo: model_id.to_string(),
            file: "onnx/model.onnx".to_string(),
            sidecar: None,
            revision: None,
        },
    }
}

/// Legacy repo-only selection (frozen contract: only the default id maps
/// to the FP16 mirror; everything else is the identity). Prefer
/// `artifact_for`, which additionally resolves the in-repo file layout
/// (official multi-file repos) and the int8 tier.
pub fn repo_for_precision(model_id: &str, precision: Precision) -> &str {
    if model_id == FP32_TEXT_MODEL && precision == Precision::Fp16 {
        FP16_TEXT_MODEL
    } else {
        model_id
    }
}

/// Split an `owner/name` model id — validated before any network access.
pub fn parse_owner_name(model_id: &str) -> Result<(String, String)> {
    model_id
        .split_once('/')
        .map(|(owner, name)| (owner.to_string(), name.to_string()))
        .ok_or_else(|| anyhow!("model_id must be 'owner/name', got '{model_id}'"))
}

/// Build an HF client with an optional cache-directory override.
pub(crate) fn hf_client(cache_dir: Option<PathBuf>) -> Result<hf_hub::HFClientSync> {
    if let Some(dir) = cache_dir {
        Ok(hf_hub::HFClientBuilder::new()
            .cache_dir(dir)
            .build()
            .map(hf_hub::HFClientSync::from_inner)
            .map_err(|e| anyhow!("{e}"))?
            .map_err(|e| anyhow!("{e}"))?)
    } else {
        Ok(hf_hub::HFClientSync::new().map_err(|e| anyhow!("{e}"))?)
    }
}

/// Offline answer to "is this model already cached, and where?" — built
/// by [`cache_info`] without a session, a download, or a device probe.
#[derive(Clone, Debug)]
pub struct CacheReport {
    /// Model id exactly as passed in.
    pub model_id: String,
    /// Artifact repo actually used — may differ from `model_id` (the
    /// default id at fp16 resolves to the FP16 mirror).
    pub repo: String,
    /// Resolved precision tier for registry lookups (`cache_info` sets
    /// `Some`; `auto` is refused). `None` for the generic
    /// [`cache_info_files`] lookup, which has no precision tier.
    pub precision: Option<Precision>,
    /// Effective hub cache directory (override or env-resolved default).
    pub cache_dir: PathBuf,
    /// One entry per probed file — the artifact model file, its external-
    /// data sidecar when the layout has one, and `tokenizer.json`.
    /// `None` = not present in this cache.
    pub files: BTreeMap<String, Option<PathBuf>>,
    /// True when `open()` would succeed offline: model file + tokenizer
    /// both present (a missing sidecar only warns at open, so it keeps
    /// `cached` true).
    pub cached: bool,
    /// Snapshot directory holding the present files
    /// (`<cache>/models--{owner}--{name}/snapshots/<sha>`), when known.
    pub snapshot_path: Option<PathBuf>,
    /// Sum of on-disk sizes of the present files (snapshot pointers are
    /// followed to their blobs).
    pub disk_usage_bytes: u64,
}

/// Offline cache inspection for an arbitrary (repo, files) pair.
///
/// The generic core underneath [`cache_info`]: looks each named file up
/// in the hub cache — offline only, by construction (every lookup runs
/// with `local_files_only`, so it can never touch the network, never
/// build a session, never probe the device). `files` pairs each
/// repo-relative filename with whether it is required: every entry
/// appears in `report.files`, and `cached` is true when all *required*
/// files are present. `repo` must be `owner/name` and at least one file
/// is required — both are validated before any I/O (an empty lookup
/// would report `cached` vacuously true).
///
/// This is the surface non-registry layouts use (bobine's converter
/// models): same offline mechanics, same [`CacheReport`] contract —
/// `model_id` echoes `repo`, and `precision` is `None` (no tier here).
/// Any failure reads as "absent": this report is diagnostic, and
/// `open()` re-fails loudly where it matters.
pub fn cache_info_files(
    repo: &str,
    files: &[(&str, bool)],
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
) -> Result<CacheReport> {
    if files.is_empty() {
        return Err(anyhow!(
            "cache_info_files needs at least one file to look up; pass \
             [(filename, required)] pairs"
        ));
    }
    let (owner, name) = parse_owner_name(repo)?;
    let repo_handle = hf_client(cache_dir.clone())?.model(owner, name);
    let rev = revision.map(str::to_string);
    // The crate joins the repo-relative filename (may contain '/') onto the
    // snapshot dir; re-walking the components re-separates them natively
    // (Windows: `...\onnx/model.onnx` → `...\onnx\model.onnx`) without
    // resolving symlinks — canonicalize would land on the blob, not the
    // snapshot pointer.
    fn cleaned(p: &Path) -> PathBuf {
        p.components().collect::<PathBuf>()
    }
    // The crate's own offline resolution (refs → snapshots/<sha>/<file>,
    // `.no_exist` aware). Any failure reads as "absent".
    let lookup = |filename: &str| {
        repo_handle
            .download_file()
            .filename(filename.to_string())
            .maybe_revision(rev.clone())
            .local_files_only(true)
            .send()
            .ok()
            .map(|p| cleaned(&p))
    };
    let mut file_map: BTreeMap<String, Option<PathBuf>> = BTreeMap::new();
    let mut cached = true;
    for (filename, required) in files {
        let path = lookup(filename);
        if *required && path.is_none() {
            cached = false;
        }
        file_map.insert(filename.to_string(), path);
    }
    let present: Vec<&PathBuf> = file_map.values().flatten().collect();
    // The <sha> dir a found file lives in: first ancestor whose parent is
    // the snapshots/ folder. None when nothing (or no snapshot-rooted
    // file) was found.
    let snapshot_path = present
        .iter()
        .find_map(|p| {
            p.ancestors()
                .find(|a| a.parent().is_some_and(|par| par.file_name().is_some_and(|n| n == "snapshots")))
        })
        .map(|p| p.to_path_buf());
    let disk_usage_bytes = present
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();

    Ok(CacheReport {
        model_id: repo.to_string(),
        repo: repo.to_string(),
        precision: None,
        cache_dir: cache_dir.unwrap_or_else(hf_hub::resolve_cache_dir),
        files: file_map,
        cached,
        snapshot_path,
        disk_usage_bytes,
    })
}

/// Offline cache inspection for one (model, precision) artifact.
///
/// Resolves the artifact exactly like `JinaV5::open` / `JinaV5Vision::open`
/// (`lookup_model` → `artifact_for`; task ids resolve via `lookup_task`
/// → `artifact_for_task`), then delegates the lookup to the
/// generic [`cache_info_files`] core (model file + `tokenizer.json`
/// required, sidecar optional — matching `open()`'s warn-and-continue).
/// The report carries the id as passed plus its resolved tier.
///
/// `precision` must be explicit: `auto` needs the landed device, and a
/// guessed tier would report the wrong repo (fp32 official vs fp16
/// mirror). Bad legacy ids fail in `parse_owner_name` before any I/O;
/// registered ids skip that check (the vision id is deliberately short).
pub fn cache_info(
    model_id: &str,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
    precision: Precision,
) -> Result<CacheReport> {
    if precision == Precision::Auto {
        return Err(anyhow!(
            "cache_info never probes the device, so precision='auto' cannot be resolved; \
             pass an explicit precision ('fp32', 'fp16' or 'int8')"
        ));
    }
    let resolved = resolve_task(model_id);
    // Task ids resolve through the task tiers (rerank/decide/laya/2b);
    // everything else is an embed id or a legacy repo path.
    let artifact = if resolved.is_some() {
        artifact_for_task(model_id, precision)
    } else {
        artifact_for(model_id, precision)
    };
    // Caller revision wins; otherwise the tier pin (builtins and
    // registered tiers alike); floating latest only when nothing pins.
    let rev = revision.map(str::to_string).or(artifact.revision.clone());
    if resolve_model(model_id).is_none() && resolved.is_none() {
        parse_owner_name(&artifact.repo)?;
    }

    let sidecar = artifact.sidecar();
    let mut files: Vec<(String, bool)> = Vec::with_capacity(6);
    files.push((artifact.file.clone(), true));
    if let Some(sc) = sidecar {
        files.push((sc, false));
    }
    files.push(("tokenizer.json".to_string(), true));
    // Kind contract files ride the resolved extra_files (kind defaults
    // for builtins, validated extension for registered entries). The
    // slot-layout weights file follows the sidecar rule instead: open()
    // requires it for q8, warns-and-continues otherwise — so cache_info
    // requires it for q8 only (precision-aware, same as the artifact).
    if let Some(spec) = resolved.as_ref() {
        for f in &spec.extra_files {
            if matches!(spec.kind, TaskKind::DecideLaya | TaskKind::Decide2B) {
                let required = !f.ends_with(".safetensors") || precision == Precision::Q8;
                files.push((f.clone(), required));
            }
        }
    }
    let refs: Vec<(&str, bool)> = files.iter().map(|(f, r)| (f.as_str(), *r)).collect();
    let mut rep = cache_info_files(&artifact.repo, &refs, rev.as_deref(), cache_dir)?;
    // The generic core echoes the repo; the registry surface reports the
    // id as passed plus its resolved tier.
    rep.model_id = model_id.to_string();
    rep.precision = Some(precision);
    Ok(rep)
}

/// Fetch one named file from any repo — the general acquisition path
/// (`tokenizer.json` and `rl_agent_config.json` ride it; the graph
/// itself rides the artifact downloader in each `open`).
pub fn fetch_repo_file(
    repo_id: &str,
    filename: &str,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    let (owner, name) = parse_owner_name(repo_id)?;
    let client = hf_client(cache_dir)?;
    let repo = client.model(owner, name);
    repo.download_file()
        .filename(filename.to_string())
        .maybe_revision(revision.map(str::to_string))
        .send()
        .map_err(|e| anyhow!("{filename} missing for '{repo_id}': {e}"))
}

/// Fetch only `tokenizer.json` — the cheap acquisition path that lets token
/// counting work without opening the ONNX session.
pub fn fetch_tokenizer_file(
    model_id: &str,
    revision: Option<&str>,
    cache_dir: Option<PathBuf>,
) -> Result<PathBuf> {
    let (owner, name) = parse_owner_name(model_id)?;
    let client = hf_client(cache_dir)?;
    let repo = client.model(owner, name);
    repo.download_file()
        .filename("tokenizer.json".to_string())
        .maybe_revision(revision.map(str::to_string))
        .send()
        .map_err(|e| anyhow!("tokenizer.json missing for '{model_id}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_owner_name_splits_once() {
        let (owner, name) = parse_owner_name("jinaai/some-model").unwrap();
        assert_eq!(owner, "jinaai");
        assert_eq!(name, "some-model");
    }

    #[test]
    fn parse_owner_name_rejects_bare_id() {
        let err = parse_owner_name("no-slash").unwrap_err().to_string();
        assert!(err.contains("model_id must be 'owner/name'"), "{err}");
    }

    #[test]
    fn repo_selection_maps_default_id_to_fp16() {
        assert_eq!(
            repo_for_precision(FP32_TEXT_MODEL, Precision::Fp16),
            FP16_TEXT_MODEL
        );
        assert_eq!(
            repo_for_precision(FP32_TEXT_MODEL, Precision::Fp32),
            FP32_TEXT_MODEL
        );
    }

    #[test]
    fn q8_artifact_resolution_lists_measured_tiers_only() {
        // Each shipped q8 resolves to its own repo/file/sidecar.
        let cases = [
            (RERANK_MINILM_MODEL, "opticsWolf/ms-marco-MiniLM-L-6-v2-q8-onnx", "model_q8.onnx"),
            (RERANK_MMARCO_MODEL, "opticsWolf/mmarco-mMiniLMv2-L12-H384-v1-q8-onnx", "model_q8.onnx"),
            (GLICLASS_V4_EDGE_MODEL, "opticsWolf/gliclass-v4-edge-q8-onnx", "model_q8.onnx"),
            (LAYA_TD_MODEL, "opticsWolf/laya-typed-decisions-fp32-onnx", "laya-td-q8.onnx"),
            (LAYA_BASE_MODEL, "opticsWolf/laya-english-fp32-onnx", "laya-base-q8.onnx"),
            (LAYA_MULTI_MODEL, "opticsWolf/laya-multilingual-fp32-onnx", "laya-multi-q8.onnx"),
            (DECIDER2B_MODEL, "opticsWolf/decider-2b-fp32-onnx", "model_q8.onnx"),
        ];
        for (id, repo, file) in cases {
            let a = artifact_for_task(id, Precision::Q8);
            assert_eq!(a.repo, repo, "{id}");
            assert_eq!(a.file, file, "{id}");
            assert!(a.sidecar().is_some(), "{id}");
        }
        // Unlisted pairs fall back to fp32 (small failed the 0-flip
        // gate; fp16/int8 tiers never existed for these models).
        let small = artifact_for_task(GLICLASS_SMALL_MODEL, Precision::Q8);
        assert_eq!(small.repo, GLICLASS_SMALL_MODEL);
        assert_eq!(small.file, "onnx/model.onnx");
        let fp16 = artifact_for_task(RERANK_MINILM_MODEL, Precision::Fp16);
        assert_eq!(fp16.file, "onnx/model.onnx");
        // Unknown ids take the legacy layout.
        let legacy = artifact_for_task("someone/custom", Precision::Q8);
        assert_eq!(legacy.file, "onnx/model.onnx");
    }

    #[test]
    fn repo_selection_never_redirects_explicit_ids() {
        // Omni tower, custom repos, mirrors: precision is moot, the id wins.
        for id in [
            "jinaai/jina-embeddings-v5-omni-small-retrieval",
            "someone/custom-embed",
        ] {
            assert_eq!(repo_for_precision(id, Precision::Fp16), id);
            assert_eq!(repo_for_precision(id, Precision::Auto), id);
        }
    }

    #[test]
    fn artifact_sidecar_derives_from_stem() {
        let a = Artifact { repo: "x/y", file: "onnx/model_fp16.onnx", sidecar: None, revision: None };
        assert_eq!(a.sidecar().as_deref(), Some("onnx/model_fp16.onnx_data"));
        let b = Artifact { repo: "x/y", file: "model.bin", sidecar: None, revision: None };
        assert_eq!(b.sidecar(), None);
    }

    #[test]
    fn artifact_sidecar_explicit_override_wins() {
        let a = Artifact {
            repo: VISION_NANO_REPO,
            file: "fp32/model.onnx",
            sidecar: Some("fp32/model.onnx.data"),
            revision: Some("95c7be07d81ded60cd105727ed9ba2d04003692d"),
        };
        assert_eq!(a.sidecar().as_deref(), Some("fp32/model.onnx.data"));
    }

    #[test]
    fn registry_vision_contract() {
        let spec = lookup_model(VISION_NANO_MODEL).expect("vision registered");
        assert_eq!(spec.native_dim, 768);
        assert_eq!(spec.ladder, &[32, 64, 128, 256, 512, 768]);
        assert_eq!(spec.text_partner, Some(NANO_TEXT_MODEL));
        assert!(TEXT_SMALL.text_partner.is_none());
        assert!(TEXT_NANO.text_partner.is_none());
        let fp32 = artifact_for(VISION_NANO_MODEL, Precision::Fp32);
        assert_eq!((fp32.repo.as_str(), fp32.file.as_str()), (VISION_NANO_REPO, "fp32/model.onnx"));
        assert_eq!(fp32.sidecar().as_deref(), Some("fp32/model.onnx.data"));
        let fp16 = artifact_for(VISION_NANO_MODEL, Precision::Fp16);
        assert_eq!((fp16.repo.as_str(), fp16.file.as_str()), (VISION_NANO_REPO, "fp16/model.onnx"));
        // Unlisted pair (vision has no int8) falls back to fp32.
        let int8 = artifact_for(VISION_NANO_MODEL, Precision::Int8);
        assert_eq!((int8.repo.as_str(), int8.file.as_str()), (VISION_NANO_REPO, "fp32/model.onnx"));
    }

    #[test]
    fn registry_resolves_nano_artifacts() {
        let spec = lookup_model(NANO_TEXT_MODEL).expect("nano registered");
        assert_eq!(spec.native_dim, 768);
        assert_eq!(spec.max_tokens, 8192);
        assert_eq!(spec.ladder, &[32, 64, 128, 256, 512, 768]);
        let fp16 = artifact_for(NANO_TEXT_MODEL, Precision::Fp16);
        assert_eq!((fp16.repo.as_str(), fp16.file.as_str()), (NANO_TEXT_MODEL, "onnx/model_fp16.onnx"));
        let int8 = artifact_for(NANO_TEXT_MODEL, Precision::Int8);
        assert_eq!((int8.repo.as_str(), int8.file.as_str()), (NANO_TEXT_MODEL, "onnx/model_quantized.onnx"));
        // Unlisted pair (small has no int8) falls back to fp32, never to junk.
        let small_int8 = artifact_for(FP32_TEXT_MODEL, Precision::Int8);
        assert_eq!((small_int8.repo.as_str(), small_int8.file.as_str()), (FP32_TEXT_MODEL, "onnx/model.onnx"));
    }

    #[test]
    fn registry_unknown_id_takes_legacy_layout() {
        assert!(lookup_model("someone/custom-embed").is_none());
        let a = artifact_for("someone/custom-embed", Precision::Fp16);
        assert_eq!((a.repo.as_str(), a.file.as_str()), ("someone/custom-embed", "onnx/model.onnx"));
    }

    #[test]
    fn task_registry_spike_entries() {
        let mini = lookup_task(RERANK_MINILM_MODEL).expect("minilm registered");
        assert_eq!(mini.kind, TaskKind::Rerank);
        assert_eq!(mini.max_tokens, 512);
        // Unmeasured tiers stay unlisted (the quant-spike rule).
        assert!(mini.fp16.is_none() && mini.int8.is_none());
        let mm = lookup_task(RERANK_MMARCO_MODEL).expect("mmarco registered");
        assert_eq!(mm.kind, TaskKind::Rerank);
        assert_eq!(mm.max_tokens, 512);
        let fp32 = artifact_for_task(RERANK_MINILM_MODEL, Precision::Fp32);
        assert_eq!((fp32.repo.as_str(), fp32.file.as_str()), (RERANK_MINILM_MODEL, "onnx/model.onnx"));
        // Unlisted pairs fall back to fp32, never to junk.
        for p in [Precision::Fp16, Precision::Int8, Precision::Auto] {
            let a = artifact_for_task(RERANK_MMARCO_MODEL, p);
            assert_eq!((a.repo.as_str(), a.file.as_str()), (RERANK_MMARCO_MODEL, "onnx/model.onnx"));
        }
        // Unknown ids take the legacy layout.
        assert!(lookup_task("someone/custom-embed").is_none());
        let a = artifact_for_task("someone/custom-rerank", Precision::Fp16);
        assert_eq!((a.repo.as_str(), a.file.as_str()), ("someone/custom-rerank", "onnx/model.onnx"));
        // The registries don't leak into each other.
        assert!(lookup_model(RERANK_MINILM_MODEL).is_none());
        assert!(lookup_task(FP32_TEXT_MODEL).is_none());
        let gli = lookup_task(GLICLASS_SMALL_MODEL).expect("gliclass registered");
        assert_eq!(gli.kind, TaskKind::Decide);
        assert_eq!(gli.max_tokens, 512);
        assert!(gli.fp16.is_none() && gli.int8.is_none());
        let fp32 = artifact_for_task(GLICLASS_SMALL_MODEL, Precision::Fp32);
        assert_eq!((fp32.repo.as_str(), fp32.file.as_str()), (GLICLASS_SMALL_MODEL, "onnx/model.onnx"));
        assert_eq!(TaskKind::Rerank.as_str(), "rerank");
        assert_eq!(TaskKind::Decide.as_str(), "decide");
    }

    #[test]
    fn cache_info_resolves_task_ids_offline() {
        let empty = std::env::temp_dir().join("embroider-cache-info-task-empty");
        let _ = std::fs::remove_dir_all(&empty);
        let rep =
            cache_info(RERANK_MMARCO_MODEL, None, Some(empty), Precision::Fp32).unwrap();
        assert_eq!(rep.model_id, RERANK_MMARCO_MODEL);
        assert_eq!(rep.repo, RERANK_MMARCO_MODEL);
        assert_eq!(rep.precision, Some(Precision::Fp32));
        assert!(!rep.cached);
        assert_eq!(rep.files.get("onnx/model.onnx"), Some(&None));
        assert_eq!(rep.files.get("tokenizer.json"), Some(&None));
    }

    // ---- cache_info: offline artifact resolution -----------------------------

    #[test]
    fn cache_info_resolves_registered_artifacts_offline() {
        // Point every lookup at an empty temp cache: resolution alone is
        // under test, nothing is present, and no real cache is touched.
        let empty = std::env::temp_dir().join("embroider-cache-info-empty");
        let _ = std::fs::remove_dir_all(&empty);

        // Default id + fp16 → the mirror repo; precision echoed resolved.
        let rep = cache_info(FP32_TEXT_MODEL, None, Some(empty.clone()), Precision::Fp16)
            .expect("offline resolution must not fail");
        assert_eq!(rep.repo, FP16_TEXT_MODEL);
        assert_eq!(rep.precision, Some(Precision::Fp16));
        assert_eq!(rep.cache_dir, empty);
        assert!(!rep.cached);
        assert_eq!(rep.files.get("onnx/model.onnx"), Some(&None));
        assert!(rep.snapshot_path.is_none());
        assert_eq!(rep.disk_usage_bytes, 0);

        // fp32 (what precision=None documents) stays on the official repo.
        let rep = cache_info(FP32_TEXT_MODEL, None, Some(empty.clone()), Precision::Fp32).unwrap();
        assert_eq!(rep.repo, FP32_TEXT_MODEL);

        // Multi-file repos list their sidecar (vision layout is explicit).
        let rep = cache_info(VISION_NANO_MODEL, None, Some(empty), Precision::Fp32).unwrap();
        assert_eq!(rep.repo, VISION_NANO_REPO);
        assert!(rep.files.contains_key("fp32/model.onnx"));
        assert!(rep.files.contains_key("fp32/model.onnx.data"));
        assert!(rep.files.contains_key("tokenizer.json"));
    }

    #[test]
    fn cache_info_rejects_auto_and_bad_ids_before_io() {
        let err = cache_info(
            FP32_TEXT_MODEL,
            None,
            Some(std::env::temp_dir()),
            Precision::Auto,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("precision='auto'"), "{err}");

        // Bare legacy id: parse_owner_name fires before any I/O.
        let err = cache_info("no-slash", None, None, Precision::Fp32)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model_id must be 'owner/name'"), "{err}");
    }

    #[test]
    fn cache_info_fake_cache_hit_partial_and_full_miss() {
        // No network is possible (local_files_only lookups); the unroutable
        // endpoint would fail any accidental request loudly anyway.
        const ENDPOINT: &str = "HF_ENDPOINT";
        let saved = std::env::var(ENDPOINT).ok();
        std::env::set_var(ENDPOINT, "http://127.0.0.1:1");

        let root =
            std::env::temp_dir().join(format!("embroider-cache-fake-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo_folder = "models--jinaai--jina-embeddings-v5-text-nano-retrieval";
        let sha = "0123456789abcdef0123456789abcdef01234567";
        let snap = root.join(repo_folder).join("snapshots").join(sha);
        std::fs::create_dir_all(snap.join("onnx")).unwrap();
        std::fs::create_dir_all(root.join(repo_folder).join("refs")).unwrap();
        std::fs::write(root.join(repo_folder).join("refs").join("main"), format!("{sha}\n"))
            .unwrap();

        // ---- hit: model + tokenizer cached, sidecar absent (like open()) ----
        let model_bytes = vec![0u8; 512];
        let tok_bytes = vec![0u8; 64];
        std::fs::write(snap.join("onnx/model.onnx"), &model_bytes).unwrap();
        std::fs::write(snap.join("tokenizer.json"), &tok_bytes).unwrap();

        let rep = cache_info(NANO_TEXT_MODEL, Some(sha), Some(root.clone()), Precision::Fp32).unwrap();
        assert!(rep.cached, "{rep:?}");
        assert_eq!(
            rep.files.get("onnx/model.onnx").unwrap().as_deref(),
            Some(snap.join("onnx/model.onnx").as_path())
        );
        // Missing sidecar is OK — open() only warns (inline weights).
        assert_eq!(rep.files.get("onnx/model.onnx_data"), Some(&None));
        assert!(rep.files.get("tokenizer.json").unwrap().is_some());
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, (model_bytes.len() + tok_bytes.len()) as u64);

        // ---- partial miss: tokenizer gone → open() would fail offline ----
        std::fs::remove_file(snap.join("tokenizer.json")).unwrap();
        let rep = cache_info(NANO_TEXT_MODEL, Some(sha), Some(root.clone()), Precision::Fp32).unwrap();
        assert!(!rep.cached);
        assert!(rep.files.get("onnx/model.onnx").unwrap().is_some());
        assert!(rep.files.get("tokenizer.json").unwrap().is_none());
        // Snapshot root still derivable from the file that IS present.
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, model_bytes.len() as u64);

        // ---- pinned revision: a commit hash resolves without refs/ ----
        std::fs::write(snap.join("tokenizer.json"), &tok_bytes).unwrap();
        let rep = cache_info(NANO_TEXT_MODEL, Some(sha), Some(root.clone()), Precision::Fp32)
            .unwrap();
        assert!(rep.cached);

        // ---- full miss: fresh cache dir, raises nothing, everything None ----
        let fresh = root.join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let rep = cache_info(NANO_TEXT_MODEL, Some(sha), Some(fresh), Precision::Fp32).unwrap();
        assert!(!rep.cached);
        assert!(rep.files.values().all(|p| p.is_none()));
        assert!(rep.snapshot_path.is_none());
        assert_eq!(rep.disk_usage_bytes, 0);

        match saved {
            Some(v) => std::env::set_var(ENDPOINT, v),
            None => std::env::remove_var(ENDPOINT),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- cache_info_files: the generic (repo, files) surface -------------

    /// Seed a fake hub-cache repo: refs/main + snapshots/<sha>/<files>.
    fn seed_fake_repo(root: &std::path::Path, folder: &str, blobs: &[(&str, &[u8])]) -> PathBuf {
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        let repo_dir = root.join(folder);
        std::fs::create_dir_all(repo_dir.join("refs")).unwrap();
        std::fs::write(repo_dir.join("refs").join("main"), format!("{sha}\n")).unwrap();
        let snap = repo_dir.join("snapshots").join(sha);
        for (name, data) in blobs {
            let p = snap.join(name);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, data).unwrap();
        }
        snap
    }

    #[test]
    fn cache_info_files_hit_partial_and_full_miss() {
        const ENDPOINT: &str = "HF_ENDPOINT";
        let saved = std::env::var(ENDPOINT).ok();
        std::env::set_var(ENDPOINT, "http://127.0.0.1:1");

        let root =
            std::env::temp_dir().join(format!("embroider-cache-files-fake-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let det = vec![0u8; 256];
        let rec = vec![0u8; 128];
        let snap = seed_fake_repo(
            &root,
            "models--org--converter",
            &[("det.onnx", &det), ("nested/rec.onnx", &rec)],
        );

        // ---- hit: required present, optional missing → still cached ----
        let rep = cache_info_files(
            "org/converter",
            &[("det.onnx", true), ("nested/rec.onnx", true), ("extra.onnx", false)],
            None,
            Some(root.clone()),
        )
        .unwrap();
        assert_eq!(rep.model_id, "org/converter");
        assert_eq!(rep.repo, "org/converter");
        assert_eq!(rep.precision, None); // generic lookup has no tier
        assert!(rep.cached, "{rep:?}");
        assert!(rep.files.get("det.onnx").unwrap().is_some());
        assert_eq!(rep.files.get("extra.onnx"), Some(&None));
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, (det.len() + rec.len()) as u64);

        // ---- partial: a required file gone → not cached, snapshot kept ----
        std::fs::remove_file(snap.join("nested/rec.onnx")).unwrap();
        let rep = cache_info_files(
            "org/converter",
            &[("det.onnx", true), ("nested/rec.onnx", true)],
            None,
            Some(root.clone()),
        )
        .unwrap();
        assert!(!rep.cached);
        assert!(rep.files.get("det.onnx").unwrap().is_some());
        assert_eq!(rep.snapshot_path.as_deref(), Some(snap.as_path()));
        assert_eq!(rep.disk_usage_bytes, det.len() as u64);

        // ---- full miss: fresh dir raises nothing, everything None ----
        let fresh = root.join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let rep = cache_info_files(
            "org/converter",
            &[("det.onnx", true)],
            None,
            Some(fresh),
        )
        .unwrap();
        assert!(!rep.cached);
        assert!(rep.files.values().all(|p| p.is_none()));
        assert!(rep.snapshot_path.is_none());
        assert_eq!(rep.disk_usage_bytes, 0);

        match saved {
            Some(v) => std::env::set_var(ENDPOINT, v),
            None => std::env::remove_var(ENDPOINT),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_info_files_pinned_revision() {
        let root =
            std::env::temp_dir().join(format!("embroider-cache-files-pin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sha = "abcdef0123456789abcdef0123456789abcdef01";
        seed_fake_repo(&root, "models--org--pinned", &[("w.onnx", &[1u8; 16])]);
        // Commit hash resolves without consulting refs/.
        let rep = cache_info_files("org/pinned", &[("w.onnx", true)], Some(sha), Some(root.clone()))
            .unwrap();
        assert!(rep.cached);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn cache_info_files_rejects_empty_and_bad_repo_before_io() {
        let err = cache_info_files("org/converter", &[], None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least one file"), "{err}");

        let err = cache_info_files("no-slash", &[("w.onnx", true)], None, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("model_id must be 'owner/name'"), "{err}");
    }

    #[test]
    fn cache_info_delegates_to_the_generic_core() {
        // Structural pin: the registry surface must stay exactly the
        // generic core plus id/tier patching — filenames, cached-ness,
        // snapshot and usage cannot drift apart.
        let root =
            std::env::temp_dir().join(format!("embroider-cache-deleg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let snap = seed_fake_repo(
            &root,
            "models--jinaai--jina-embeddings-v5-text-nano-retrieval",
            &[("onnx/model.onnx", &[9u8; 64]), ("tokenizer.json", &[7u8; 8])],
        );
        // The fake repo pins refs/main at the fixture sha: pass it
        // explicitly (None now means the spec pin, which the fake cache
        // lacks — that default is covered by the pin structural test).
        let via_registry = cache_info(
            NANO_TEXT_MODEL,
            Some("abcdef0123456789abcdef0123456789abcdef01"),
            Some(root.clone()),
            Precision::Fp32,
        )
        .unwrap();
        let via_generic = cache_info_files(
            NANO_TEXT_MODEL,
            &[
                ("onnx/model.onnx", true),
                ("onnx/model.onnx_data", false),
                ("tokenizer.json", true),
            ],
            None,
            Some(root.clone()),
        )
        .unwrap();
        assert_eq!(via_registry.files, via_generic.files);
        assert_eq!(via_registry.cached, via_generic.cached);
        assert_eq!(via_registry.snapshot_path, via_generic.snapshot_path);
        assert_eq!(via_registry.disk_usage_bytes, via_generic.disk_usage_bytes);
        // ...plus the registry-only patching:
        assert_eq!(via_registry.model_id, NANO_TEXT_MODEL);
        assert_eq!(via_registry.precision, Some(Precision::Fp32));
        assert_eq!(via_generic.model_id, NANO_TEXT_MODEL); // repo echoed
        assert_eq!(via_generic.precision, None);
        assert_eq!(via_registry.snapshot_path.as_deref(), Some(snap.as_path()));
        let _ = std::fs::remove_dir_all(&root);
    }
    fn test_task_entry(repo: &str) -> RegisteredTask {
        RegisteredTask {
            id: "test-org/test-reranker".to_string(),
            kind: TaskKind::Rerank,
            max_tokens: 256,
            prompt_first: false,
            fp32: RegisteredArtifact {
                repo: repo.to_string(),
                file: "onnx/model.onnx".to_string(),
                sidecar: None,
                revision: None,
            },
            fp16: None,
            int8: None,
            q8: None,
            provenance: Provenance::unclaimed(),
            registered_from: String::new(),
            extra_files: vec![],
        }
    }

    #[test]
    fn task_kind_parse_accepts_aliases() {
        assert_eq!(TaskKind::parse("rerank").unwrap(), TaskKind::Rerank);
        assert_eq!(TaskKind::parse("DECIDE").unwrap(), TaskKind::Decide);
        assert_eq!(TaskKind::parse("decide-laya").unwrap(), TaskKind::DecideLaya);
        assert_eq!(TaskKind::parse("laya").unwrap(), TaskKind::DecideLaya);
        assert_eq!(TaskKind::parse("decide-2b").unwrap(), TaskKind::Decide2B);
        assert_eq!(TaskKind::parse("decider2b").unwrap(), TaskKind::Decide2B);
        assert!(TaskKind::parse("embed").is_err());
        assert!(TaskKind::parse("cross-encoder").is_err());
    }

    #[test]
    fn register_task_round_trip_resolves_and_serves() {
        let id = "test-org/test-reranker";
        unregister_task(id);
        let entry = test_task_entry("test-org/test-reranker-onnx");
        let back = register_task(entry).expect("register");
        assert_eq!(back.registered_from, "api");
        let spec = resolve_task(id).expect("resolves");
        assert_eq!(spec.kind, TaskKind::Rerank);
        assert_eq!(spec.max_tokens, 256);
        assert_eq!(spec.source, RegistrySource::Registered);
        assert!(!spec.measured);
        assert_eq!(spec.registered_from, "api");
        // Unlisted tiers fall back to fp32 (the same rule as builtins).
        assert_eq!(artifact_for_task(id, Precision::Q8).file, "onnx/model.onnx");
        assert_eq!(
            artifact_for_task(id, Precision::Fp32).repo,
            "test-org/test-reranker-onnx"
        );
        // Duplicate refused; builtin shadow refused.
        assert!(register_task(test_task_entry("test-org/test-reranker-onnx")).is_err());
        let mut shadow = test_task_entry("x/y");
        shadow.id = RERANK_MINILM_MODEL.to_string();
        assert!(register_task(shadow).is_err());
        assert!(unregister_task(id));
        assert!(resolve_task(id).is_none());
        // Legacy path untouched: unknown ids still resolve None.
        assert!(resolve_task("someone/else").is_none());
    }

    #[test]
    fn register_task_validation_is_loud_before_io() {
        // Bad id shape.
        let mut e = test_task_entry("test-org/x-onnx");
        e.id = "not-an-owner-name".to_string();
        assert!(register_task(e).is_err());
        // Over the kind ceiling (rerank caps at 512).
        let mut e = test_task_entry("test-org/x-onnx");
        e.id = "test-org/too-long".to_string();
        e.max_tokens = 1024;
        assert!(register_task(e).is_err());
        // Non-onnx graph file.
        let mut e = test_task_entry("test-org/x-onnx");
        e.id = "test-org/bad-file".to_string();
        e.fp32.file = "model.safetensors".to_string();
        assert!(register_task(e).is_err());
        // Laya kind without its config file.
        let mut e = test_task_entry("test-org/x-onnx");
        e.id = "test-org/bare-laya".to_string();
        e.kind = TaskKind::DecideLaya;
        e.max_tokens = 512;
        assert!(register_task(e).is_err());
        // Decide-2b kind without the weights file.
        let mut e = test_task_entry("test-org/x-onnx");
        e.id = "test-org/bare-2b".to_string();
        e.kind = TaskKind::Decide2B;
        e.max_tokens = 512;
        e.extra_files = vec!["decision.json".to_string(), "calibration.json".to_string()];
        assert!(register_task(e).is_err());
        // prompt_first on a marker layout warns but registers.
        let mut e = test_task_entry("test-org/x-onnx");
        e.id = "test-org/warned-laya".to_string();
        e.kind = TaskKind::DecideLaya;
        e.max_tokens = 512;
        e.prompt_first = true;
        e.extra_files = vec!["rl_agent_config.json".to_string()];
        assert!(register_task(e).is_ok());
        assert!(unregister_task("test-org/warned-laya"));
    }

    #[test]
    fn register_model_validation_and_resolution() {
        let id = "test-org/test-embed";
        unregister_model(id);
        let entry = RegisteredModel {
            id: id.to_string(),
            native_dim: 768,
            max_tokens: 8192,
            ladder: vec![32, 128, 768],
            fp32: RegisteredArtifact {
                repo: "test-org/test-embed-onnx".to_string(),
                file: "onnx/model.onnx".to_string(),
                sidecar: None,
                revision: None,
            },
            fp16: None,
            int8: None,
            text_partner: None,
            provenance: Provenance::unclaimed(),
            registered_from: String::new(),
        };
        register_model(entry).expect("register");
        let spec = resolve_model(id).expect("resolves");
        assert_eq!(spec.native_dim, 768);
        assert_eq!(spec.ladder, vec![32, 128, 768]);
        assert_eq!(spec.source, RegistrySource::Registered);
        assert!(unregister_model(id));
        // Bad ladders refused: empty, unsorted, over native_dim.
        for bad in [vec![], vec![128, 32], vec![32, 1024]] {
            let mut e = RegisteredModel {
                id: "test-org/bad-ladder".to_string(),
                native_dim: 768,
                max_tokens: 512,
                ladder: bad,
                fp32: RegisteredArtifact {
                    repo: "test-org/x".to_string(),
                    file: "onnx/model.onnx".to_string(),
                    sidecar: None,
                    revision: None,
                },
                fp16: None,
                int8: None,
                text_partner: None,
                provenance: Provenance::unclaimed(),
                registered_from: String::new(),
            };
            let _ = &mut e;
            assert!(validate_model(&e).is_err());
        }
        // Builtin shadow refused.
        let mut e = RegisteredModel {
            id: NANO_TEXT_MODEL.to_string(),
            native_dim: 768,
            max_tokens: 8192,
            ladder: vec![768],
            fp32: RegisteredArtifact {
                repo: "test-org/x".to_string(),
                file: "onnx/model.onnx".to_string(),
                sidecar: None,
                revision: None,
            },
            fp16: None,
            int8: None,
            text_partner: None,
            provenance: Provenance::unclaimed(),
            registered_from: String::new(),
        };
        let _ = &mut e;
        assert!(register_model(e).is_err());
    }

    #[test]
    fn registry_file_round_trip() {
        // Strict schema: unknown fields and versions fail loudly.
        assert!(serde_json::from_str::<RegistryFile>("{\"registry_version\": 1}").is_ok());
        assert!(serde_json::from_str::<RegistryFile>(
            "{\"registry_version\": 2, \"tasks\": [], \"models\": []}"
        )
        .is_ok()); // parses; the version gate lives in load_registry_file
        assert!(serde_json::from_str::<RegistryFile>(
            "{\"registry_version\": 1, \"tasks\": [], \"oops\": []}"
        )
        .is_err());
        let dir = std::env::temp_dir().join("embroider-registry-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("extra.json");
        std::fs::write(
            &path,
            r#"{"registry_version": 1,
                "tasks": [{"id": "test-org/file-reranker", "kind": "rerank", "max_tokens": 128,
                           "fp32": {"repo": "test-org/file-reranker-onnx", "file": "onnx/model.onnx"}}],
                "models": []}"#,
        )
        .unwrap();
        unregister_task("test-org/file-reranker");
        let summary = load_registry_file(&path).expect("load");
        assert_eq!(summary.tasks_registered, 1);
        assert_eq!(summary.models_registered, 0);
        let spec = resolve_task("test-org/file-reranker").expect("resolves");
        assert_eq!(spec.registered_from, format!("{}#0", path.display()));
        // Export round-trips through the loader (overlay only).
        let exported = export_registry().expect("export");
        unregister_task("test-org/file-reranker");
        let summary = load_registry_file(&path).expect("reload");
        assert_eq!(summary.tasks_registered, 1);
        assert!(exported.contains("test-org/file-reranker"));
        assert!(unregister_task("test-org/file-reranker"));
        std::fs::remove_file(&path).ok();
        // Wrong version refused before any registration.
        std::fs::write(&path, r#"{"registry_version": 2, "tasks": [], "models": []}"#).unwrap();
        assert!(load_registry_file(&path).is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn builtin_artifacts_carry_revision_pins() {
        // Structural: fetching defaults to the parity-gated pin, never
        // floating latest (decision #4 — pins transcribed from comments).
        for m in builtin_models() {
            assert!(m.fp32.revision.is_some(), "{} fp32 unpinned", m.id);
            for (tier, art) in [("fp16", m.fp16), ("int8", m.int8)].into_iter().filter_map(|(t, a)| a.map(|a| (t, a))) {
                assert!(art.revision.is_some(), "{} {tier} unpinned", m.id);
            }
        }
        for t in builtin_tasks() {
            assert!(t.fp32.revision.is_some(), "{} fp32 unpinned", t.id);
            for (tier, art) in
                [("fp16", t.fp16), ("int8", t.int8), ("q8", t.q8)].into_iter().filter_map(|(t, a)| {
                    a.map(|a| (t, a))
                })
            {
                assert!(art.revision.is_some(), "{} {tier} unpinned", t.id);
            }
        }
    }

}

