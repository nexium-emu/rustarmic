//! Floating-point data-processing (1 source) — `FMOV`, `FNEG`, `FABS`,
//! `FSQRT`, etc. Phase 1 covers `FMOV Vd, Vn` only; arithmetic unaries
//! follow in later phases.

use disarm64::decoder::FLOATDP1;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATDP1) -> Result<InstStatus> {
    use FLOATDP1::*;
    let raw = match insn {
        FMOV_Fd_Fn(i)              => i.0,
        FMOV_Fd_S_S_Fn_S_S(i)      => i.0,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let ptype = bits(raw, 22, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    match ptype {
        0b00 => {  // single
            let v = em.get_v_s(rn);
            em.set_v_s(rd, v);
        }
        0b01 => {  // double
            let v = em.get_v_d(rn);
            em.set_v_d(rd, v);
        }
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    }
    Ok(InstStatus::Continue)
}
