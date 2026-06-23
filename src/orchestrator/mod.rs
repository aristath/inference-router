//! # Model Orchestration
//!
//! Core system for managing model lifecycles, VRAM admission, and GPU allocation.
//!
//! ## Key Components
//! - `engine.rs`: Main `Orchestrator` struct with mutable state and locking
//! - `allocation.rs`: Pure GPU allocation logic (greedy best-fit with 5% headroom)
//! - `eviction.rs`: Eviction scoring formula (`ln(idle + 1) + 1 / log2(gib + 1)`)
//!
//! ## Architecture
//! - Two-level locking: global `admission` mutex + per-model `load_guards`
//! - Detached load tasks via `tokio::spawn` for cancellation safety
//! - Dirty-flag persistence with 5s reconcile loop
//! - One process per loaded model (no sharing, no hot-swap)

pub mod allocation;
pub mod engine;
pub mod eviction;

pub use engine::{AppState, LoadError, MutationError, Orchestrator, StopError};
