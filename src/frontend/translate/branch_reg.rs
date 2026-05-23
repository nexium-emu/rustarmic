use disarm64::decoder::BRANCH_REG;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: BRANCH_REG) -> Result<InstStatus> {
    use BRANCH_REG::*;
    let (raw, link, is_ret) = match insn {
        BR_Rn(i)  => (i.0, false, false),
        BLR_Rn(i) => (i.0, true,  false),
        RET_Rn(i) => (i.0, false, true),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    let rn = bits(raw, 5, 5) as u8;
    let target = em.get_x(rn);
    em.branch_indirect(target, link, is_ret);
    Ok(InstStatus::Terminator)
}
