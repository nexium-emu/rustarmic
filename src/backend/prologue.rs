//! Emit the per-block prologue/epilogue and the dispatcher tail.

use iced_x86::code_asm::*;

use crate::backend::abi::{ARG0_REG, CTX_REG};
use crate::error::Result;

/// Emit the prologue:
/// - save callee-saved registers we use,
/// - move the `CpuContext*` arg into `CTX_REG` (= r15),
/// - reserve the frame for value slots.
pub fn emit_prologue(asm: &mut CodeAssembler, frame_bytes: i32) -> Result<()> {
    asm.push(rbp)?;
    asm.mov(rbp, rsp)?;
    asm.push(rbx)?;
    asm.push(r12)?;
    asm.push(r13)?;
    asm.push(r14)?;
    asm.push(r15)?;
    asm.mov(CTX_REG, ARG0_REG)?;
    if frame_bytes > 0 {
        asm.sub(rsp, frame_bytes as i32)?;
    }
    Ok(())
}

/// Emit the epilogue. Assumes the next-PC value is already in `rax`.
pub fn emit_epilogue(asm: &mut CodeAssembler, frame_bytes: i32) -> Result<()> {
    if frame_bytes > 0 {
        asm.add(rsp, frame_bytes as i32)?;
    }
    asm.pop(r15)?;
    asm.pop(r14)?;
    asm.pop(r13)?;
    asm.pop(r12)?;
    asm.pop(rbx)?;
    asm.pop(rbp)?;
    asm.ret()?;
    Ok(())
}
