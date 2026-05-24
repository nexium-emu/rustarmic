use disarm64::decoder;
use disarm64::decoder::Operation;

use crate::error::{Error, Result};
use crate::frontend::translate;
use crate::ir::{Block, IrEmitter, Terminal};

#[derive(Clone, Copy, Debug)]
pub struct TranslateOptions {
    pub max_insts: u32,
    pub multiblock: bool,
}

impl Default for TranslateOptions {
    fn default() -> Self {
        Self { max_insts: 64, multiblock: true }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstStatus {
    Continue,
    Terminator,
}

pub fn translate_instruction(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let opcode = decoder::decode(inst)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: inst })?;

    match opcode.operation {
        Operation::MOVEWIDE(insn)         => translate::movewide::translate(em, insn),
        Operation::ADDSUB_IMM(insn)       => translate::addsub_imm::translate(em, insn),
        Operation::ADDSUB_SHIFT(insn)     => translate::addsub_shift::translate(em, insn),
        Operation::ADDSUB_CARRY(insn)     => translate::addsub_carry::translate(em, insn),
        Operation::ADDSUB_EXT(insn)       => translate::addsub_ext::translate(em, insn),
        Operation::LOG_IMM(insn)          => translate::log_imm::translate(em, insn),
        Operation::LOG_SHIFT(insn)        => translate::log_shift::translate(em, insn),
        Operation::EXCEPTION(insn)        => translate::exception::translate(em, insn),
        Operation::LDST_POS(insn)         => translate::ldst_pos::translate(em, insn),
        Operation::LDST_IMM9(insn)        => translate::ldst_imm9::translate(em, insn),
        Operation::LDST_UNSCALED(insn)    => translate::ldst_unscaled::translate(em, insn),
        Operation::LDST_REGOFF(insn)      => translate::ldst_regoff::translate(em, insn),
        Operation::LDSTEXCL(insn)         => translate::ldstexcl::translate(em, insn),
        Operation::LSE_ATOMIC(insn)       => translate::lse_atomic::translate(em, insn),
        Operation::IC_SYSTEM(insn)        => translate::ic_system::translate(em, insn),
        Operation::LDSTPAIR_OFF(insn)     => translate::ldstpair_off::translate(em, insn),
        Operation::LDSTPAIR_INDEXED(insn) => translate::ldstpair_indexed::translate(em, insn),
        Operation::BRANCH_IMM(insn)       => translate::branch_imm::translate(em, insn),
        Operation::BRANCH_REG(insn)       => translate::branch_reg::translate(em, insn),
        Operation::CONDBRANCH(insn)       => translate::condbranch::translate(em, insn),
        Operation::COMPBRANCH(insn)       => translate::compbranch::translate(em, insn),
        Operation::TESTBRANCH(insn)       => translate::testbranch::translate(em, insn),
        Operation::BITFIELD(insn)         => translate::bitfield::translate(em, insn),
        Operation::CONDSEL(insn)          => translate::condsel::translate(em, insn),
        Operation::PCRELADDR(insn)        => translate::pcreladdr::translate(em, insn),
        Operation::EXTRACT(insn)          => translate::extract::translate(em, insn),
        Operation::CONDCMP_IMM(insn)      => translate::condcmp_imm::translate(em, insn),
        Operation::CONDCMP_REG(insn)      => translate::condcmp_reg::translate(em, insn),
        Operation::DP_1SRC(insn)          => translate::dp_1src::translate(em, insn),
        Operation::DP_2SRC(insn)          => translate::dp_2src::translate(em, insn),
        Operation::DP_3SRC(insn)          => translate::dp_3src::translate(em, insn),
        Operation::FLOATDP1(insn)         => translate::fp_dp1::translate(em, insn),
        Operation::FLOATDP2(insn)         => translate::fp_dp2::translate(em, insn),
        Operation::FLOATDP3(insn)         => translate::fp_dp3::translate(em, insn),
        Operation::FLOATCMP(insn)         => translate::fp_cmp::translate(em, insn),
        Operation::FLOATCCMP(insn)        => translate::fp_ccmp::translate(em, insn),
        Operation::FLOATSEL(insn)         => translate::fp_sel::translate(em, insn),
        Operation::FLOATIMM(insn)         => translate::fp_imm::translate(em, insn),
        Operation::FLOAT2INT(insn)        => translate::fp_conv::translate(em, insn),
        _ => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
}

pub fn translate_block_into(
    block: &mut Block,
    start_pc: u64,
    fetch: &mut dyn FnMut(u64) -> Option<u32>,
    opts: TranslateOptions,
) -> Result<()> {
    block.reset(start_pc);
    let mut pc = start_pc;

    for _ in 0..opts.max_insts {
        let inst = fetch(pc).ok_or(Error::GuestMemory { addr: pc })?;
        let mut em = IrEmitter::new(block, pc);

        match translate_instruction(&mut em, inst)? {
            InstStatus::Continue => {
                pc = pc.wrapping_add(4);
                block.cycles = block.cycles.saturating_add(1);
            }
            InstStatus::Terminator => {
                block.cycles = block.cycles.saturating_add(1);
                block.end_pc = pc.wrapping_add(4);
                return Ok(());
            }
        }
    }

    block.end_pc = pc;
    block.terminal = Terminal::LinkBlock { next_pc: pc };
    Ok(())
}
