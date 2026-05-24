//! ASIMD "same"-form three-operand vector ops (Vd, Vn, Vm with matching shapes).
//!
//! Covered so far:
//!   - bitwise: AND, ORR, EOR, BIC, ORN (also gives MOV V.16B via ORR Vd,Vn,Vn)
//!   - integer add/sub: ADD, SUB across 8B/16B/4H/8H/2S/4S/2D lanes
//! Compare, multiply, min/max, etc. come in later phases.

use disarm64::decoder::ASIMDSAME;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind {
    Add, Sub,
    And, Orr, Eor, Bic, Orn,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDSAME) -> Result<InstStatus> {
    use ASIMDSAME::*;
    let (raw, kind) = match insn {
        ADD_Vd_Vn_Vm(i) => (i.0, Kind::Add),
        SUB_Vd_Vn_Vm(i) => (i.0, Kind::Sub),
        AND_Vd_Vn_Vm(i) => (i.0, Kind::And),
        ORR_Vd_Vn_Vm(i) => (i.0, Kind::Orr),
        EOR_Vd_Vn_Vm(i) => (i.0, Kind::Eor),
        BIC_Vd_Vn_Vm(i) => (i.0, Kind::Bic),
        ORN_Vd_Vn_Vm(i) => (i.0, Kind::Orn),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2); // 00=B, 01=H, 10=S, 11=D
    let rm   = bits(raw, 16, 5) as u8;
    let rn   = bits(raw, 5,  5) as u8;
    let rd   = bits(raw, 0,  5) as u8;

    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);

    let result = match kind {
        Kind::Add => em.vec_add(vn, vm, size, q),
        Kind::Sub => em.vec_sub(vn, vm, size, q),
        Kind::And => em.vec_and(vn, vm, q),
        Kind::Orr => em.vec_orr(vn, vm, q),
        Kind::Eor => em.vec_eor(vn, vm, q),
        Kind::Bic => em.vec_bic(vn, vm, q),
        Kind::Orn => em.vec_orn(vn, vm, q),
    };

    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
