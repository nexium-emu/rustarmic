//! Drive a translation pass over a guest PC range.

use crate::error::{Error, Result};
use crate::frontend::decoder::{classify, DecodeClass};
use crate::frontend::translate;
use crate::ir::{Block, IrEmitter, Terminal};

#[derive(Clone, Copy, Debug)]
pub struct TranslateOptions {
    /// Maximum number of guest instructions to fold into a single block.
    pub max_insts: u32,
    /// Whether to follow direct branches across blocks (multi-block).
    pub multiblock: bool,
}

impl Default for TranslateOptions {
    fn default() -> Self {
        Self { max_insts: 64, multiblock: true }
    }
}

/// Outcome of translating a single guest instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstStatus {
    /// Continue translating; PC advances by 4.
    Continue,
    /// Block terminator was emitted; stop translating.
    Terminator,
}

/// Translate a single 32-bit guest instruction into Armlets.
pub fn translate_instruction(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let cls = classify(inst);
    match cls {
        DecodeClass::DataProcImm   => translate::data_proc_imm::translate(em, inst),
        DecodeClass::DataProcReg   => translate::data_proc_reg::translate(em, inst),
        DecodeClass::BranchSysExc  => translate::branch::translate(em, inst),
        DecodeClass::LoadStore     => translate::load_store::translate(em, inst),
        DecodeClass::DataProcFloat => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
        DecodeClass::Sve | DecodeClass::Sme | DecodeClass::Reserved =>
            Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
}

/// Translate an entire basic block starting at `start_pc`.
pub fn translate_block(
    start_pc: u64,
    fetch: &mut dyn FnMut(u64) -> Option<u32>,
    opts: TranslateOptions,
) -> Result<Block> {
    let mut block = Block::new(start_pc);
    let mut pc = start_pc;

    for _ in 0..opts.max_insts {
        let inst = fetch(pc).ok_or(Error::GuestMemory { addr: pc })?;
        let mut em = IrEmitter::new(&mut block, pc);

        match translate_instruction(&mut em, inst)? {
            InstStatus::Continue => {
                pc = pc.wrapping_add(4);
                block.cycles = block.cycles.saturating_add(1);
            }
            InstStatus::Terminator => {
                block.cycles = block.cycles.saturating_add(1);
                block.end_pc = pc.wrapping_add(4);
                return Ok(block);
            }
        }
    }

    // Budget exhausted — emit a fall-through link to next PC.
    block.end_pc = pc;
    block.terminal = Terminal::LinkBlock { next_pc: pc };
    Ok(block)
}
