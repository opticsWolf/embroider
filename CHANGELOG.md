# Changelog

All notable changes to `embroider` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

## [0.1.1] — 2026-09-13

### Fixed
- PyPI wheel interpreter matrix: 0.1.0 shipped only the runners' default
  interpreters (cp312/cp314). The release workflow now builds the proven
  3-platform × Python 3.11/3.12/3.13 matrix (`--interpreter` per OKFgraph's
  okf-embed release job), and `requires-python` is pinned to `>=3.11` to
  match what wheels are actually built for.

## [0.1.0] — 2026-09-13

### Added
- Initial release — clean move from OKFgraph's `rust/okf-embed`
  (github.com/opticsWolf/OKFgraph). Jina v5 text-embedding contract
  (`JinaV5`: prefix → tokenize@8192 → last-token pooling → L2 → Matryoshka
  truncate → re-normalise; `JinaTokenizer` for session-free exact counts),
  lazy session lifecycle, air-gapped `open_files`, cross-provider ONNX
  Runtime plumbing (`providers`/`probe`/`policy`/`acquire`/`diag`),
  opt-in PyO3 `extension-module` for wheels, pure-Rust rlib for crates.io.
  21 pure unit tests.
