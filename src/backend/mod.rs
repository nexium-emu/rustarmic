//! x86_64 backend.
//!
//! The backend is intentionally narrow for the first milestone:
//!
//! - **Register allocator**: linear-scan over a small callee-saved GPR pool
//!   (RBX, R12, R13, R14) with stack spill slots as fallback. Honors
//!   per-op clobber masks and coalesces Identity / two-address binops with
//!   their source register when the source dies at the op.
//!
//! - **Code emission**: `iced-x86`'s `CodeAssembler` builds an instruction
//!   stream that the `BlockEncoder` then resolves into a `Vec<u8>` with
//!   patched RIP-relative offsets. The result is copied into an executable
//!   page by [`crate::jit::code_cache`].
//!
//! - **ABI**: the JIT entrypoint takes a single argument — the `CpuContext`
//!   pointer — in `rcx` (Win64) / `rdi` (SysV). The dispatcher saves it into
//!   `r15` (callee-saved) so all subsequent code can address guest state as
//!   `[r15 + offset]`. We use `rax`/`rcx`/`rdx`/`r8`/`r9`/`r10`/`r11` as
//!   scratch within an armlet.

pub mod abi;
pub mod clobbers;
pub mod operand;
pub mod regalloc;
pub mod isel;
pub mod prologue;
pub mod emit;

pub use emit::{emit_block, ChainSite, EmittedBlock};
