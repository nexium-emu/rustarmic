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
