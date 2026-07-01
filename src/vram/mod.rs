//! # VRAM Management
//!
//! GPU VRAM tracking, GGUF catalog metadata, and llama.cpp-backed sizing.
//!
//! ## Key Components
//! - `estimator.rs`: lightweight GGUF metadata reader for model management
//! - `llama_fit.rs`: `llama-fit-params` integration for authoritative sizing
//! - `tracker.rs`: `VRAMTracker` reading live GPU VRAM usage
//!
//! ## Features
//! - GGUF metadata parsing with support for hybrid architectures
//! - Scan/import metadata without depending on stale GGUF crates
//! - Runtime placement sizing delegated to llama.cpp

pub mod estimator;
pub mod llama_fit;
pub mod tracker;
