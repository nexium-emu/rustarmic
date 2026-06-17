use disarm64::decoder::ASIMDALL;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDALL) -> Result<InstStatus> {
    use ASIMDALL::*;
    match insn {
        ADDV_Fd_Vn(i) => translate_addv(em, i.0),
        _ => Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    }
}

fn translate_addv(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let rn   = bits(raw, 5, 5) as u8;
    let rd   = bits(raw, 0, 5) as u8;

    if !(q && size == 2) {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let vn = em.get_v_q(rn);
    let sum = em.vec_addv32(vn);
    em.set_v_s(rd, sum);
    Ok(InstStatus::Continue)
}
