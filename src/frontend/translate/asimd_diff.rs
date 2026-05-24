//! ASIMD "different"-form widening ops. Coverage so far: SADDL/SADDL2 and
//! UADDL/UADDL2 (widening signed/unsigned add). SSUBL/USUBL/MULL etc. share
//! the same shape and can slot in here as needed.

use disarm64::decoder::ASIMDDIFF;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDDIFF) -> Result<InstStatus> {
    use ASIMDDIFF::*;
    let (raw, signed) = match insn {
        SADDL_Vd_Vn_Vm(i)  => (i.0, true),
        SADDL2_Vd_Vn_Vm(i) => (i.0, true),
        UADDL_Vd_Vn_Vm(i)  => (i.0, false),
        UADDL2_Vd_Vn_Vm(i) => (i.0, false),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    let high_half = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);  // 00=B->H, 01=H->S, 10=S->D
    let rm   = bits(raw, 16, 5) as u8;
    let rn   = bits(raw, 5,  5) as u8;
    let rd   = bits(raw, 0,  5) as u8;
    if size > 2 {
        return Err(Error::Decode { pc: em.current_pc, opcode: raw });
    }
    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);
    let result = if signed { em.vec_saddl(vn, vm, size, high_half) }
                 else      { em.vec_uaddl(vn, vm, size, high_half) };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
