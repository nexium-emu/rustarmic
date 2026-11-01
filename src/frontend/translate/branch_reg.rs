use disarm64::decoder::BRANCH_REG;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: BRANCH_REG) -> Result<InstStatus> {
    use BRANCH_REG::*;
    let rn = |raw: u32| bits(raw, 5, 5) as u8;
    let (rn, link, is_ret) = match insn {
        BR_Rn(i) => (rn(i.0), false, false),
        BRAA_Rn_Rd_SP(i) => (rn(i.0), false, false),
        BRAB_Rn_Rd_SP(i) => (rn(i.0), false, false),
        BRAAZ_Rn(i) => (rn(i.0), false, false),
        BRABZ_Rn(i) => (rn(i.0), false, false),
        BLR_Rn(i) => (rn(i.0), true, false),
        BLRAA_Rn_Rd_SP(i) => (rn(i.0), true, false),
        BLRAB_Rn_Rd_SP(i) => (rn(i.0), true, false),
        BLRAAZ_Rn(i) => (rn(i.0), true, false),
        BLRABZ_Rn(i) => (rn(i.0), true, false),
        RET_Rn(i) => (rn(i.0), false, true),
        RETAA(_) | RETAB(_) => (30, false, true),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };
    let target = em.get_x(rn);
    em.branch_indirect(target, link, is_ret);
    Ok(InstStatus::Terminator)
}
