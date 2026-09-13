//! Error plumbing.
//!
//! anyhow carries everything. ort errors are stringified at every boundary:
//! `ort::Error` carries occurrences of non-Send/Sync payloads, so it cannot
//! travel in `anyhow::Error` directly (ort::Error is generic over the failing
//! stage; accept any Display).

/// Crate-wide result alias.
pub type Result<T> = anyhow::Result<T>;

/// Stringify any Display-able error into an anyhow error.
pub(crate) fn oe<T, E: std::fmt::Display>(r: std::result::Result<T, E>) -> Result<T> {
    r.map_err(|e| anyhow::anyhow!("{e}"))
}
