//! Top-level "compile a block" entry point.

use iced_x86::code_asm::*;

use crate::backend::abi::CTX_REG;
use crate::backend::isel::{emit_armlet, emit_cond_check_byte};
use crate::backend::prologue::{emit_epilogue, emit_prologue};
use crate::backend::reg_alloc::Allocation;
use crate::error::Result;
use crate::ir::{Block, Terminal};
use crate::jit::context::cpu_offsets;

pub struct EmittedBlock {
    pub code: Vec<u8>,
}

const BITNESS: u32 = 64;

pub fn emit_block(block: &Block) -> Result<EmittedBlock> {
    let alloc = Allocation::build(block);
    let mut asm = CodeAssembler::new(BITNESS)?;

    emit_prologue(&mut asm, alloc.frame_bytes)?;

    for i in 0..block.code.len() {
        emit_armlet(&mut asm, block, &alloc, i)?;
    }

    emit_terminator(&mut asm, block, &alloc)?;
    emit_epilogue(&mut asm, alloc.frame_bytes)?;

    let bytes = asm.assemble(0)?;
    Ok(EmittedBlock { code: bytes })
}

fn emit_terminator(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation) -> Result<()> {
    match block.terminal {
        Terminal::Invalid => {
            asm.mov(rax, block.end_pc as i64)?;
        }
        Terminal::LinkBlock { next_pc } |
        Terminal::DirectBranch { target_pc: next_pc, link: _ } => {
            asm.mov(rax, next_pc as i64)?;
        }
        Terminal::ConditionalBranch { cond_nzcv: _, cond_code, taken_pc, not_taken_pc } => {
            asm.movzx(r10d, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
            emit_cond_check_byte(asm, crate::arch::Cond::from_bits(cond_code))?;
            asm.test(al, al)?;
            asm.mov(rax, not_taken_pc as i64)?;
            asm.mov(rcx, taken_pc as i64)?;
            asm.cmovne(rax, rcx)?;
        }
        Terminal::CompareBranchZero { value, inverse, taken_pc, not_taken_pc } => {
            let loc = alloc.loc(value);
            asm.mov(rcx, qword_ptr(rbp - loc.stack_offset))?;
            asm.test(rcx, rcx)?;
            asm.mov(rax, not_taken_pc as i64)?;
            asm.mov(rcx, taken_pc as i64)?;
            if inverse {
                asm.cmovne(rax, rcx)?;
            } else {
                asm.cmove(rax, rcx)?;
            }
        }
        Terminal::TestBranchBit { value, bit, inverse, taken_pc, not_taken_pc } => {
            let loc = alloc.loc(value);
            asm.mov(rcx, qword_ptr(rbp - loc.stack_offset))?;
            asm.bt(rcx, bit as u32 as i32)?;
            asm.mov(rax, not_taken_pc as i64)?;
            asm.mov(rcx, taken_pc as i64)?;
            if inverse {
                asm.cmovc(rax, rcx)?;
            } else {
                asm.cmovnc(rax, rcx)?;
            }
        }
        Terminal::IndirectBranch { target, link: _, is_ret: _ } => {
            let loc = alloc.loc(target);
            asm.mov(rax, qword_ptr(rbp - loc.stack_offset))?;
        }
        Terminal::Exception { kind, imm: _ } => {
            let kind_bits: u64 = match kind {
                crate::ir::block::ExceptionKind::Svc => 0xE000_0000_0000_0001,
                crate::ir::block::ExceptionKind::Brk => 0xE000_0000_0000_0002,
                crate::ir::block::ExceptionKind::Hvc => 0xE000_0000_0000_0003,
                crate::ir::block::ExceptionKind::UnknownInst => 0xE000_0000_0000_00FF,
            };
            asm.mov(rax, kind_bits as i64)?;
        }
    }
    Ok(())
}
