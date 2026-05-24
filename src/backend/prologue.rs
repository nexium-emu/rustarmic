//! Per-block prologue/epilogue are now empty — the shared host→JIT thunk
//! (see [`crate::backend::thunk`]) does the callee-saved push/pop and the
//! XMM6..XMM15 save/restore once per `Jit::run` iteration. Blocks just run
//! the body, then `ret` (or `jmp` to the next block via a chain patch).

use iced_x86::code_asm::*;

use crate::backend::regalloc::Allocation;
use crate::error::Result;

pub fn emit_prologue(_asm: &mut CodeAssembler, _alloc: &Allocation) -> Result<()> {
    Ok(())
}

pub fn emit_epilogue(asm: &mut CodeAssembler, _alloc: &Allocation) -> Result<()> {
    asm.ret().map_err(|e| crate::error::Error::Backend(e.to_string()))?;
    Ok(())
}
