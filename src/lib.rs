//! # rustarmic
//!
//! High-performance AArch64 → x86_64 dynamic recompiler.
//!
//! Design notes:
//! - Vec-based SSA (no linked lists, no per-instruction allocation).
//! - Single-pass optimizer.
//! - Multi-block CFG; direct-branch chaining.
//! - Assumes immutable guest code (no SMC tracking).

pub mod error;
pub mod arch;
pub mod ir;
pub mod frontend;
pub mod optimizer;
pub mod backend;
pub mod jit;
pub mod util;

pub use error::{Error, Result};
pub use jit::{Jit, JitConfig, ExitReason, Memory, CpuContext};
