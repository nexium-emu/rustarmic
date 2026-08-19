use iced_x86::BlockEncoderOptions;
use iced_x86::code_asm::*;

use crate::arch::Cond;
use crate::backend::abi::CTX_REG;
use crate::backend::isel::{emit_armlet, emit_cond_check_byte};
use crate::backend::operand::load64;
use crate::backend::prologue::{emit_epilogue, emit_prologue};
use crate::backend::regalloc::{ALLOCATABLE_GPRS, compute_live_ranges, linear_scan};
use crate::error::{Error, Result};
use crate::ir::block::ExceptionKind;
use crate::ir::{Block, Terminal};
use crate::jit::context::cpu_offsets;

pub struct EmittedBlock {
    pub code: Vec<u8>,
    pub chains: Vec<ChainSite>,
    pub body_offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct ChainSite {
    pub patch_offset: u32,
    pub target_pc: u64,
}

const BITNESS: u32 = 64;

pub fn emit_block(block: &Block) -> Result<EmittedBlock> {
    let ranges = compute_live_ranges(block);
    let alloc = linear_scan(block, &ranges, ALLOCATABLE_GPRS);

    let mut asm = CodeAssembler::new(BITNESS)?;

    emit_prologue(&mut asm, &alloc)?;

    let mut body_label = asm.create_label();
    asm.set_label(&mut body_label)?;
    asm.nop()?;

    for (vr, _) in block.iter_live() {
        emit_armlet(&mut asm, block, &alloc, vr.as_usize())?;
    }

    let mut chain_specs: Vec<(CodeLabel, u64)> = Vec::new();
    let mut epilogue_label = asm.create_label();

    match block.terminal {
        Terminal::Invalid => {
            asm.mov(rax, block.end_pc as i64)?;
        }
        Terminal::LinkBlock { next_pc }
        | Terminal::DirectBranch {
            target_pc: next_pc,
            link: _,
        } => {
            let lbl = emit_patch_site(&mut asm, next_pc)?;
            chain_specs.push((lbl, next_pc));
        }
        Terminal::ConditionalBranch {
            cond_nzcv: _,
            cond_code,
            taken_pc,
            not_taken_pc,
        } => {
            asm.movzx(edx, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
            emit_cond_check_byte(&mut asm, Cond::from_bits(cond_code))?;
            asm.test(al, al)?;
            let mut taken_label = asm.create_label();
            asm.jnz(taken_label)?;
            emit_two_way_patches(
                &mut asm,
                &mut chain_specs,
                &mut taken_label,
                &mut epilogue_label,
                taken_pc,
                not_taken_pc,
            )?;
        }
        Terminal::CompareBranchZero {
            value,
            inverse,
            taken_pc,
            not_taken_pc,
        } => {
            load64(&mut asm, &alloc, value, rcx)?;
            asm.test(rcx, rcx)?;
            let mut taken_label = asm.create_label();
            if inverse {
                asm.jnz(taken_label)?;
            } else {
                asm.jz(taken_label)?;
            }
            emit_two_way_patches(
                &mut asm,
                &mut chain_specs,
                &mut taken_label,
                &mut epilogue_label,
                taken_pc,
                not_taken_pc,
            )?;
        }
        Terminal::TestBranchBit {
            value,
            bit,
            inverse,
            taken_pc,
            not_taken_pc,
        } => {
            load64(&mut asm, &alloc, value, rcx)?;
            asm.bt(rcx, bit as u32 as i32)?;
            let mut taken_label = asm.create_label();
            if inverse {
                asm.jc(taken_label)?;
            } else {
                asm.jnc(taken_label)?;
            }
            emit_two_way_patches(
                &mut asm,
                &mut chain_specs,
                &mut taken_label,
                &mut epilogue_label,
                taken_pc,
                not_taken_pc,
            )?;
        }
        Terminal::IndirectBranch {
            target,
            link: _,
            is_ret: _,
        } => {
            load64(&mut asm, &alloc, target, rax)?;
        }
        Terminal::Exception { kind, imm } => {
            let kind_byte: u64 = match kind {
                ExceptionKind::Svc => 0x01,
                ExceptionKind::Brk => 0x02,
                ExceptionKind::Hvc => 0x03,
                ExceptionKind::UnknownInst => 0xFF,
            };
            let exit_value: u64 = 0xE000_0000_0000_0000 | ((imm as u64 & 0xFFFF) << 8) | kind_byte;
            asm.mov(rcx, block.end_pc as i64)?;
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::pc() as i32), rcx)?;
            asm.mov(rax, exit_value as i64)?;
        }
    }

    asm.nop()?;
    asm.set_label(&mut epilogue_label)?;
    emit_epilogue(&mut asm, &alloc)?;

    let result = asm
        .assemble_options(0, BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS)
        .map_err(|e| Error::Backend(e.to_string()))?;

    let mut chains = Vec::with_capacity(chain_specs.len());
    for (lbl, target_pc) in chain_specs {
        let off = result
            .label_ip(&lbl)
            .map_err(|e| Error::Backend(e.to_string()))?;
        chains.push(ChainSite {
            patch_offset: off as u32,
            target_pc,
        });
    }
    let body_offset = result
        .label_ip(&body_label)
        .map_err(|e| Error::Backend(e.to_string()))? as u32;

    let code = result.inner.code_buffer;
    Ok(EmittedBlock {
        code,
        chains,
        body_offset,
    })
}

fn emit_patch_site(asm: &mut CodeAssembler, fallback_pc: u64) -> Result<CodeLabel> {
    let mut label = asm.create_label();
    asm.set_label(&mut label)?;
    asm.db(&[0xE9])?;
    asm.dd(&[0u32])?;
    asm.mov(rax, fallback_pc as i64)?;
    Ok(label)
}

fn emit_patch_site_at_label(
    asm: &mut CodeAssembler,
    label: &mut CodeLabel,
    fallback_pc: u64,
) -> Result<()> {
    asm.set_label(label)?;
    asm.db(&[0xE9])?;
    asm.dd(&[0u32])?;
    asm.mov(rax, fallback_pc as i64)?;
    Ok(())
}

fn emit_two_way_patches(
    asm: &mut CodeAssembler,
    chain_specs: &mut Vec<(CodeLabel, u64)>,
    taken_label: &mut CodeLabel,
    epilogue_label: &mut CodeLabel,
    taken_pc: u64,
    not_taken_pc: u64,
) -> Result<()> {
    let lbl_nt = emit_patch_site(asm, not_taken_pc)?;
    chain_specs.push((lbl_nt, not_taken_pc));
    asm.jmp(*epilogue_label)?;

    emit_patch_site_at_label(asm, taken_label, taken_pc)?;
    chain_specs.push((*taken_label, taken_pc));
    Ok(())
}
