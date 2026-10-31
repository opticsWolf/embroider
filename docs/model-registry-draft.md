# Model registry surface — draft

Status: **shipped in 0.12.0** (layer 1 + revision pins; layer 2 deferred as designed). Problem: every model is a
`pub const` in `src/acquire.rs` (`TEXT_SMALL`, `RERANK_MINILM`,
`GLICLASS_SMALL`, …) with `&'static` fields; `lookup_model` /
`lookup_task` search the builtin slices only. Unknown ids get the
legacy path — fp32-only, default `onnx/model.onnx` layout, no kind
metadata. Consequences today:

- A custom cross-encoder works only if its layout matches a builtin
  seam by accident; a custom GLiClass cannot declare `prompt_first`
  (wrong layout scores garbage, never an error).
- Custom Laya / Decider-2B checkpoints are unservable: `cache_info`
  special-cases their sidecars per kind (`acquire.rs` ~667/673), and
  `open()` resolves those sidecars from the hardcoded spec.
- Contributors adding a model touch Rust consts + a release. There is
  no data-only path.

## Non-goals

New *input contracts* stay code. The four loaders (rerank pair seam,
GLiClass label splice, Laya markers, decider-2b slots) are behavior,
not data — a genuinely new type still needs a `TaskKind` variant, a
loader, goldens, and a release. This draft covers new *models under
known kinds* (layer 1) and sketches where a loader plug-in would dock
(layer 2, explicitly deferred).

## Layer 1: spec registration (this draft)

### Owned spec types

`TaskSpec` / `ModelSpec` keep their `&'static` consts (frozen
contract, zero-cost builtins). Registration uses owned twins:

```rust
pub struct RegisteredArtifact {
    pub repo: String,
    pub file: String,
    pub sidecar: Option<String>, // same semantics as Artifact::sidecar
    pub revision: Option<String>, // overlay can do better than builtins here
}

/// Machine-readable provenance (JSON has no comments, so the drift
/// numbers and gate notes this repo keeps in comments become fields).
/// Serialized verbatim into `available_models()`; `open()` warns from
/// `measured: false`.
#[derive(Serialize, Deserialize)]
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

pub struct RegisteredTask {
    pub id: String,              // must be `owner/name`
    pub kind: TaskKind,           // existing kinds only
    pub max_tokens: usize,        // clamped to the kind ceiling
    pub prompt_first: bool,       // meaningful for Decide only (warn otherwise)
    pub fp32: RegisteredArtifact,
    pub fp16: Option<RegisteredArtifact>,
    pub int8: Option<RegisteredArtifact>,
    pub q8: Option<RegisteredArtifact>,
    pub provenance: Provenance,  // measured flag + drift/claims, see policy
    pub registered_from: String, // file path + line, or "api" — named in open() warnings
    /// Extra required files beyond graph + tokenizer, per kind:
    /// Laya → [`rl_agent_config.json`], Decider2B → [`decision.json`,
    /// `calibration.json`, `model.safetensors`]. Defaults come from the
    /// kind; the caller may extend, never shrink.
    pub extra_files: Vec<String>,
}
```

Embed side mirrors it (`RegisteredModel`: `native_dim`, `ladder`,
`text_partner`, same artifact ladder). One registration call per side;
"various types" = any of the existing kinds, embed included.

### The overlay

```rust
static TASK_OVERLAY: RwLock<HashMap<String, RegisteredTask>>
static MODEL_OVERLAY: RwLock<HashMap<String, RegisteredModel>>
```

- Lookups check builtins first, overlay second. Unchanged fallback
  order afterwards (legacy path for fully unknown ids stays).
- **Shadowing refused loudly**: registering an id equal to a builtin
  (or an already-registered) id is an error, not an override — the
  frozen contract means `cross-encoder/ms-marco-MiniLM-L-6-v2` resolves
  identically in every process. Mirrors get their own ids (the
  `opticsWolf/…-q8-onnx` convention already proves this shape).
- `artifact_for_task` / `artifact_for` keep their exact fallback rule
  (unlisted precision → fp32); they just accept an owned spec too.
- `available_models()` merges both layers and gains a `source` column
  (`builtin` / `registered`) plus `measured` (`true` / `false`).
  okfgraph's dispatch can key off `kind` + `source` without parsing ids.

### Validation at registration (before any I/O)

1. `id` parses as `owner/name` (`parse_owner_name`, existing).
2. `kind` parses (`rerank` / `decide` / `decide-laya` / `decide-2b` /
   `embed`); unknown kinds refused — that is the layer-2 boundary,
   and the error says so with a pointer.
3. Artifact files end in `.onnx`; sidecar naming follows
   `Artifact::sidecar` rules.
4. `max_tokens` within the kind ceiling (512 rerank/GLiClass-small,
   2048 v4-shape, 1024 Laya, 32768 decider-2b — today's consts become
   a `kind_ceiling()` function so both paths share them).
5. `prompt_first=true` on a non-`Decide` kind warns (Laya/decider-2b
   layouts are intrinsic; the flag would be a lie).
6. Optional `probe=True`: offline `cache_info` check that the files
   are already cached — registration succeeds regardless (registration
   is data, not download), but the report tells the caller what `open()`
   would need.

### The unmeasured-tier policy: flagged-allow (decided)

Background: the `ModelSpec` rule — *precision maps list only
MEASURED artifacts, unlisted pairs fall back to fp32 rather than
fetching unvalidated weights* — was earned the hard way
(dynamic-int8 drift 2.8–9997 on the −1e4 mask sentinel, bf16 killed,
fp16 adopted only as a measured CUDA-serving variant, small-q8
refused for 2 flips). The fallback-to-fp32 is a safety property, not
a preference.

Strict (fp32-only overlay) was considered and rejected: the legacy
unknown-id path already serves fp32, so strict mode would deliver
only kind metadata (ceilings, `prompt_first`, sidecars) — real value,
but it would make the exact quantize-gate-serve workflow this repo
ran six times impossible without a source fork. A lab that measured
its own `myorg/my-decider-q8` must be able to serve it.

Flagged-allow holds because serving an unmeasured overlay tier
requires **two explicit actions, no silent path**: (1) registering
the tier (file or API call), and (2) requesting it by name at every
`open()` — `Auto` never resolves to a non-fp32 tier on any device,
and additionally **never resolves to an overlay non-fp32 tier even
when that tier is the only one listed**. No default, no fallback,
no auto-resolution can land on unvalidated weights.

Three mechanisms, all structural:

- `provenance.measured: false` recorded at registration, surfaced in
  `available_models()`, and `open()` warns in the fp16-on-CPU class.
- `registered_from: String` captured at registration (file path +
  line, or `"api"`) and named in the warning — `"q8 tier for X is
  caller-supplied (registered from models/extra.json:14, measured:
  false)"` is actionable; a generic warning is wallpaper.
- `Auto` resolves to overlay fp32 freely (safe default, same as
  builtins) but to overlay non-fp32 tiers never — one rule in the
  resolution logic, closing the "registry file makes q8 the
  default" hole.

Supply-chain bound: a hostile registry file cannot shadow builtins
(refused at registration) and cannot make anything default (rule
above). Blast radius is ids the attacker invented, served only to a
caller who loaded the attacker's file and requested them by name.
House rule preserved by scoping: *we* ship only measured tiers;
*callers* register unmeasured ones explicitly, attributed.

### Persistence: serde JSON surface, not hidden state

The owned spec types are the file format — `#[derive(Serialize,
Deserialize)]` with `#[serde(deny_unknown_fields)]` (typos fail
loudly, house style), parsed once at load and validated through the
identical `validate()` as API registration. One struct, three uses:
file loading, `export_registry()` round-trip/debug, and the
`available_models()` merge.

JSON, not TOML: `serde_json` is already in the tree (zero new deps —
TOML/YAML crates would violate the offline locked build), Python
writes it with stdlib `json.dump` (TOML writing needs a third-party
lib, and registry files are generated far more than hand-authored),
and every metadata artifact in this stack is already JSON
(`tokenizer.json`, `decision.json`, `manifest.json`, `runtime.json`).

```python
embroider.load_registry_file("models/extra.json")  # once, process-global
```

plus `EMBROIDER_REGISTRY_FILE` env as the same call without code
(logged at first `open()` when set). No auto-discovery, no cwd
sniffing — the offline/no-hidden-state contract holds.

JSON has no comments, and this repo documents provenance in comments
(pins, drift numbers, gate notes) — so provenance becomes a
first-class field instead of a comment, machine-readable and
surfacable in `available_models()`:

```json
{
  "registry_version": 1,
  "tasks": [
    {
      "id": "myorg/my-reranker",
      "kind": "rerank",
      "max_tokens": 512,
      "prompt_first": false,
      "fp32": {"repo": "myorg/my-reranker-onnx", "file": "onnx/model.onnx"},
      "q8": {"repo": "myorg/my-reranker-q8-onnx", "file": "model_q8.onnx",
               "sidecar": "model_q8.onnx.data"},
      "revision": "abc123",
      "provenance": {"measured": false,
                       "notes": "self-quant, bench pending"}
    }
  ],
  "models": []
}
```

`registry_version` is required from day one: v2 files fail loudly
instead of mis-parsing. `provenance.measured: false` is the
flagged-allow policy made auditable — caller-supplied tiers carry
their claim (or lack of one) in the file, and `open()` warns from
the same field.

### Python surface

```python
embroider.register_task(
    id="myorg/my-reranker",
    kind="rerank",
    max_tokens=512,
    fp32={"repo": "myorg/my-reranker-onnx", "file": "onnx/model.onnx"},
    q8={"repo": "myorg/my-reranker-q8-onnx", "file": "model_q8.onnx",
        "sidecar": "model_q8.onnx.data"},
    revision="abc123",
    measured=False,
)
r = Reranker.open("myorg/my-reranker", precision="q8")  # overlay resolves
```

`register_model(...)` likewise for embeds (`native_dim`, `ladder`,
`text_partner`). Both return the validated entry (echo, testable).
Both refuse builtin ids with the registry path in the message.

## Layer 2: kind plug-ins (deferred, sketched)

A `TaskLoader` trait (`open_files`-shape: resolve files → session →
typed batch call) that downstream Rust crates implement; `TaskKind`
gains a `Custom(&'static str)` variant or a parallel registry
`kind → loader`. Python-defined *behavior* (a per-batch scoring
callback) is deliberately excluded: it would run Python in the hot
loop, break the offline determinism story, and void every golden. New
contracts stay Rust; the registry makes them instantly servable once
they exist (registering a model of a freshly added kind needs no
further acquire.rs change — that is the layer-1/layer-2 handshake to
protect: `kind_ceiling()` and `extra_files` defaults must be
loader-owned, not match statements in acquire.rs).

## Migration (mechanical, no behavior change)

1. `kind_ceiling()`, `kind_extra_files()` extracted from today's consts.
2. Owned twins + overlays + `validate()`; `lookup_*` check overlay.
3. `artifact_for_*` generic over borrowed-vs-owned (or a tiny
   `AsSpec` view trait).
4. `available_models` merges + `source`/`measured` columns.
5. `register_task` / `register_model` / `load_registry_file` Python
   bindings; unit test: register → lookup → artifact resolution →
   duplicate-refused; e2e stays `open_files` (already the seam).
6. Docs: README registry section, compat table gains a "registered"
   row shape, CHANGELOG minor.

### Builtin revision pins through the same machinery (decided: yes)

Current state contradicts the repo's own invariant. The registry
says "adding entries never moves existing graphs" — but builtins
pin revisions only in *comments* (`233902d`, `21edefa`, `885af1af`,
…) while fetching `None` → HF latest. An upstream file amendment
moves every fresh cache dir to different bytes while e2e stays green
(env-pathed files, never registry-resolved): users diverge, tests
don't notice. The frozen contract currently rests on upstream repos
never changing — hope, not engineering.

The change is mechanical, a transcription not archaeology:

1. `Artifact` consts gain `revision: Option<&'static str>`,
   populated with the SHAs already sitting in the comments.
2. `open()` / `cache_info` revision threading becomes "caller
   `None` → spec revision; caller `Some` → wins." Explicit always
   beats default; every current caller is preserved.
3. A unit test asserts every builtin spec has `Some(revision)` — the
   next added model can't rot the policy by omission (same trick as
   the 0-flip gate: structural, not conventional).

Pin format is the commit SHA, never a tag or branch (tags float).
Cost is a bounded bump chore on deliberate upstream updates: a const
edit plus golden regen — the exact workflow the repo already runs
per model — replacing silent drift with a reviewable, golden-pinned
update. For inference weights, immutability-by-default with explicit
upgrades is the correct polarity.

Our own `opticsWolf/*` delta commits (model_q8 landing on
decider-2b-fp32-onnx as a new commit) don't break this: the fp32
files are content-identical across those commits. But each shippers'
commit wants a pin review — does the pinned SHA still contain every
file the spec references? Running builtins through the same
`validate()` as the overlay catches that at test time instead of at
a user's first offline `open()`. One validator, two front doors,
builtins included.

## Questions for the maintainer

1. Shadowing: refuse (draft) vs allow-behind-a-flag (mirror override)?
2. File loading: env var, explicit call, or both (draft says both)?
3. ~~Unmeasured tiers~~ — decided: flagged-allow (§above).
4. ~~Builtin revision pins~~ — decided: yes (§above).
