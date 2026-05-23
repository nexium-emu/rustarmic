use disarm64::decoder::CONDBRANCH;

use crate::arch::Cond;
use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: CONDBRANCH) -> Result<InstStatus> {
    use CONDBRANCH::*;
    let raw = match insn {
        B__ADDR_PCREL19(i)  => i.0,
        BC__ADDR_PCREL19(i) => i.0,
    };
    let cond = Cond::from_bits(bits(raw, 0, 4) as u8);
    let imm19 = bits(raw, 5, 19);
    let offset = sign_extend(imm19 as u64, 19) << 2;
    let target = em.current_pc.wrapping_add(offset as u64);
    em.branch_cond(cond, target);
    Ok(InstStatus::Terminator)
}
