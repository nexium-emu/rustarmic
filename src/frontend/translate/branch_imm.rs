use disarm64::decoder::BRANCH_IMM;

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: BRANCH_IMM) -> Result<InstStatus> {
    use BRANCH_IMM::*;
    let (raw, link) = match insn {
        B_ADDR_PCREL26(i)  => (i.0, false),
        BL_ADDR_PCREL26(i) => (i.0, true),
    };
    let imm26 = bits(raw, 0, 26);
    let offset = sign_extend(imm26 as u64, 26) << 2;
    let target = em.current_pc.wrapping_add(offset as u64);
    em.branch(target, link);
    Ok(InstStatus::Terminator)
}
