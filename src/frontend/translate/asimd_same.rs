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
    Add, Sub, Mul,
    And, Orr, Eor, Bic, Orn,
    CmEq, CmGt, CmGe, CmHi, CmHs,
    Bit, Bif, Bsl,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDSAME) -> Result<InstStatus> {
    use ASIMDSAME::*;
    let (raw, kind) = match insn {
        ADD_Vd_Vn_Vm(i) => (i.0, Kind::Add),
        SUB_Vd_Vn_Vm(i) => (i.0, Kind::Sub),
        MUL_Vd_Vn_Vm(i) => (i.0, Kind::Mul),
        AND_Vd_Vn_Vm(i) => (i.0, Kind::And),
        ORR_Vd_Vn_Vm(i) => (i.0, Kind::Orr),
        EOR_Vd_Vn_Vm(i) => (i.0, Kind::Eor),
        BIC_Vd_Vn_Vm(i) => (i.0, Kind::Bic),
        ORN_Vd_Vn_Vm(i) => (i.0, Kind::Orn),
        CMEQ_Vd_Vn_Vm(i) => (i.0, Kind::CmEq),
        CMGT_Vd_Vn_Vm(i) => (i.0, Kind::CmGt),
        CMGE_Vd_Vn_Vm(i) => (i.0, Kind::CmGe),
        CMHI_Vd_Vn_Vm(i) => (i.0, Kind::CmHi),
        CMHS_Vd_Vn_Vm(i) => (i.0, Kind::CmHs),
        BIT_Vd_Vn_Vm(i) => (i.0, Kind::Bit),
        BIF_Vd_Vn_Vm(i) => (i.0, Kind::Bif),
        BSL_Vd_Vn_Vm(i) => (i.0, Kind::Bsl),
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
        Kind::Mul => {
            // Only 16/32-bit lanes are wired through SSE today; surface a clear
            // error for 8/64-bit lane MUL until we add the decomposition.
            if size != 1 && size != 2 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            em.vec_mul(vn, vm, size, q)
        }
        Kind::And => em.vec_and(vn, vm, q),
        Kind::Orr => em.vec_orr(vn, vm, q),
        Kind::Eor => em.vec_eor(vn, vm, q),
        Kind::Bic => em.vec_bic(vn, vm, q),
        Kind::Orn => em.vec_orn(vn, vm, q),
        Kind::CmEq => em.vec_cmeq(vn, vm, size, q),
        Kind::CmGt => em.vec_cmgt(vn, vm, size, q),
        Kind::CmGe => em.vec_cmge(vn, vm, size, q),
        Kind::CmHi => {
            if size == 0 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            em.vec_cmhi(vn, vm, size, q)
        }
        Kind::CmHs => {
            if size == 0 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            em.vec_cmhs(vn, vm, size, q)
        }
        Kind::Bit | Kind::Bif | Kind::Bsl => {
            // These read Vd as their third source, so fetch it now.
            let vd_prev = em.get_v_q(rd);
            match kind {
                Kind::Bit => em.vec_bit(vd_prev, vn, vm, q),
                Kind::Bif => em.vec_bif(vd_prev, vn, vm, q),
                Kind::Bsl => em.vec_bsl(vd_prev, vn, vm, q),
                _ => unreachable!(),
            }
        }
    };

    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
