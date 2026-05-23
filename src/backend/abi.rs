//! Host calling convention helpers.
//!
//! We support both Win64 and SysV AMD64. The JIT entry-point signature is
//! `extern "sysv64" fn(*mut CpuContext) -> u64` on Linux/macOS and
//! `extern "win64"  fn(*mut CpuContext) -> u64` on Windows. The dispatcher
//! returns the next guest PC to resume at, or `u64::MAX` to signal exit.

use iced_x86::code_asm::*;

/// Host register holding the `CpuContext*` while JITted code runs.
/// Callee-saved on both ABIs.
pub const CTX_REG: AsmRegister64 = r15;

/// First-argument register for the JIT entry.
#[cfg(target_os = "windows")]
pub const ARG0_REG: AsmRegister64 = rcx;
#[cfg(not(target_os = "windows"))]
pub const ARG0_REG: AsmRegister64 = rdi;

/// Generic scratch register #1 — used by isel for the LHS of most ops.
pub const SCRATCH1: AsmRegister64 = rax;
/// Generic scratch register #2 — used by isel for the RHS / shift count.
pub const SCRATCH2: AsmRegister64 = r10;
/// Generic scratch register #3 — used for addresses and temporaries.
pub const SCRATCH3: AsmRegister64 = r11;

/// Callee-saved registers we touch in the prologue and restore in the epilogue.
pub const CALLEE_SAVED: &[AsmRegister64] = &[rbx, rbp, r12, r13, r14, r15];
