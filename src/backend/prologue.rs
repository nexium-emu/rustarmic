use iced_x86::code_asm::*;

use crate::backend::abi::{ARG0_REG, CTX_REG};
use crate::backend::operand::xmm;
use crate::backend::regalloc::Allocation;
use crate::error::Result;

pub fn emit_prologue(asm: &mut CodeAssembler, alloc: &Allocation) -> Result<()> {
    asm.push(rbp)?;
    asm.mov(rbp, rsp)?;
    asm.push(rbx)?;
    asm.push(r12)?;
    asm.push(r13)?;
    asm.push(r14)?;
    asm.push(r15)?;
    asm.mov(CTX_REG, ARG0_REG)?;
    let frame_bytes = alloc.frame_bytes();
    if frame_bytes > 0 {
        asm.sub(rsp, frame_bytes as i32)?;
    }
    for x in alloc.iter_used_xmms() {
        let off = alloc.xmm_save_offset(x);
        asm.movdqu(xmmword_ptr(rbp - off), xmm(x))?;
    }
    Ok(())
}

pub fn emit_epilogue(asm: &mut CodeAssembler, alloc: &Allocation) -> Result<()> {
    for x in alloc.iter_used_xmms() {
        let off = alloc.xmm_save_offset(x);
        asm.movdqu(xmm(x), xmmword_ptr(rbp - off))?;
    }
    let frame_bytes = alloc.frame_bytes();
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
