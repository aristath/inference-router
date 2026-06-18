//! # Dashboard UI
//!
//! HTML templates and data structures for the browser dashboard.
//!
//! ## Key Components
//! - `templates.rs`: Askama template types (`DashboardTemplate`, `DashboardFragmentTemplate`)
//! - `assets.rs`: embedded htmx + idiomorph served at `/assets/*`
//! - HTML templates in `templates/` directory
//!
//! ## Features
//! - Live updates via htmx polling + idiomorph DOM morphing (no full-table
//!   re-render, so the model table no longer flashes on each tick)
//! - Server-authoritative sort/filter (the browser sends sort/dir/q; Rust
//!   renders the ordered, filtered rows)
//! - Model CRUD interface
//! - GPU/CPU/RAM monitoring
//! - Binary preset management

pub mod assets;
pub mod templates;
