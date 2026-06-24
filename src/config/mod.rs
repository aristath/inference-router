//! # Configuration Management
//!
//! Handles all persistent configuration for models, binary presets, and application settings.
//!
//! ## Key Components
//! - `model.rs`: `ModelConfig` with 30+ fields covering sampling, GPU allocation, speculative decoding
//! - `preset.rs`: `BinaryPreset` for reusable inference-server binary paths
//! - `settings.rs`: `AppSettings` for loop guard and runtime configuration
//! - `store.rs`: `JsonStore<T>` for atomic file persistence with temp-then-rename safety
//!
//! All configuration is stored in `~/.config/inference-router/` as JSON files:
//! - `models.json`: All model definitions
//! - `presets.json`: Binary path presets
//! - `aliases.json`: Model name aliases
//! - `settings.json`: Application settings

pub mod alias;
pub mod backend;
pub mod gpu_tags;
pub mod model;
pub mod perf;
pub mod preset;
pub mod settings;
pub mod store;

pub use alias::*;
pub use backend::*;
pub use gpu_tags::*;
pub use model::*;
pub use perf::*;
pub use preset::*;
pub use settings::*;
pub use store::*;
