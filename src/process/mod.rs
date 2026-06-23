//! # Backend Process Management
//!
//! Spawns, monitors, and kills inference-server processes (llama-server, vLLM).
//!
//! ## Key Components
//! - `manager.rs`: `ProcessManager` with PID tracking and health checking
//! - `RequestGuard`: RAII handle for active request counting
//!
//! ## Features
//! - Automatic port allocation (ephemeral or configured range)
//! - Process health monitoring
//! - Graceful shutdown via SIGKILL on drop
//! - Active request tracking for load balancing

pub mod manager;
