//! # VRAM Management
//!
//! GPU VRAM tracking and estimation for admission control and eviction.
//! 
//! ## Key Components
//! - `estimator.rs`: `VramEstimate` from GGUF metadata + context size
//! - `tracker.rs`: `VRAMTracker` reading AMD GPU sysfs for live VRAM usage
//!
//! ## Features
//! - GGUF metadata parsing with support for hybrid architectures
//! - Sliding-window attention handling
//! - Mixed quantization support (q4_0, q8_0, fp16)
//! - 10% runtime overhead buffer

pub mod estimator;
pub mod tracker;
