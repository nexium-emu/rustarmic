use iced_x86::code_asm::*;

pub const CTX_REG: AsmRegister64 = r15;

#[cfg(target_os = "windows")]
pub const ARG0_REG: AsmRegister64 = rcx;
#[cfg(not(target_os = "windows"))]
pub const ARG0_REG: AsmRegister64 = rdi;

pub const SCRATCH0: AsmRegister64 = rax;
pub const SCRATCH1: AsmRegister64 = rsi;
pub const SCRATCH2: AsmRegister64 = rdi;
pub const SCRATCH3: AsmRegister64 = rdx;

pub const CALLEE_SAVED: &[AsmRegister64] = &[rbx, rbp, rsi, rdi, r12, r13, r14, r15];
