use disarm64::decoder::PCRELADDR;

use crate::error::Result;
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: PCRELADDR) -> Result<InstStatus> {
    use PCRELADDR::*;
    let (raw, is_adrp) = match insn {
        ADR_Rd_ADDR_PCREL21(i) => (i.0, false),
        ADRP_Rd_ADDR_ADRP(i) => (i.0, true),
    };
    let immlo = bits(raw, 29, 2);
    let immhi = bits(raw, 5, 19);
    let rd = bits(raw, 0, 5) as u8;
    let _ = bit(raw, 31);

    let raw_imm = ((immhi as u64) << 2) | (immlo as u64);
    let offset = sign_extend(raw_imm, 21);

    let (base, final_off) = if is_adrp {
        (em.current_pc & !0xFFF, offset << 12)
    } else {
        (em.current_pc, offset)
    };
    let target = base.wrapping_add(final_off as u64);

    let c = em.const_u64(target);
    em.set_x(rd, c);
    Ok(InstStatus::Continue)
}
