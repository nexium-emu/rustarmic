//! ASIMD "misc"-form single-operand ops (Vd, Vn).
//!
//! Coverage so far:
//!   - NEG Vd.<T>, Vn.<T>: per-lane two's-complement negation
//!   - ABS Vd.<T>, Vn.<T>: per-lane absolute value (8/16/32-bit lanes only;
//!     64-bit lane ABS needs SSE emulation, deferred)
//!   - NOT Vd.<T>, Vn.<T> (aka MVN): bitwise complement

use disarm64::decoder::ASIMDMISC;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Neg, Abs, Not }

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDMISC) -> Result<InstStatus> {
    use ASIMDMISC::*;
    let (raw, kind) = match insn {
        NEG_Vd_Vn(i) => (i.0, Kind::Neg),
        ABS_Vd_Vn(i) => (i.0, Kind::Abs),
        NOT_Vd_Vn(i) => (i.0, Kind::Not),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2); // 00=B, 01=H, 10=S, 11=D
    let rn   = bits(raw, 5, 5) as u8;
    let rd   = bits(raw, 0, 5) as u8;

    let vn = em.get_v_q(rn);
    let result = match kind {
        Kind::Neg => em.vec_neg(vn, size, q),
        Kind::Abs => {
            if size == 3 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            em.vec_abs(vn, size, q)
        }
        Kind::Not => em.vec_not(vn, q),
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
