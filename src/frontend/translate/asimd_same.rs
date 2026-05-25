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
    Smin, Smax, Umin, Umax,
    // FP per-lane ops. disarm64 groups every FP vector form (2S/4S/2D) into a
    // single `_V_2S_` enum variant, so we only need one `Kind` per op and
    // decode the (q, sz) bits at translate time.
    FAdd, FSub, FMul, FDiv, FMax, FMin,
    FCmEq, FCmGt, FCmGe,
    FMla, FMls,
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
        SMIN_Vd_Vn_Vm(i) => (i.0, Kind::Smin),
        SMAX_Vd_Vn_Vm(i) => (i.0, Kind::Smax),
        UMIN_Vd_Vn_Vm(i) => (i.0, Kind::Umin),
        UMAX_Vd_Vn_Vm(i) => (i.0, Kind::Umax),
        FADD_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FAdd),
        FADD_Vd_Vn_Vm(i)                => (i.0, Kind::FAdd),
        FSUB_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FSub),
        FSUB_Vd_Vn_Vm(i)                => (i.0, Kind::FSub),
        FMUL_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMul),
        FMUL_Vd_Vn_Vm(i)                => (i.0, Kind::FMul),
        FDIV_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FDiv),
        FDIV_Vd_Vn_Vm(i)                => (i.0, Kind::FDiv),
        FMAX_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMax),
        FMAX_Vd_Vn_Vm(i)                => (i.0, Kind::FMax),
        FMIN_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMin),
        FMIN_Vd_Vn_Vm(i)                => (i.0, Kind::FMin),
        FCMEQ_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FCmEq),
        FCMEQ_Vd_Vn_Vm(i)                => (i.0, Kind::FCmEq),
        FCMGT_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FCmGt),
        FCMGT_Vd_Vn_Vm(i)                => (i.0, Kind::FCmGt),
        FCMGE_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FCmGe),
        FCMGE_Vd_Vn_Vm(i)                => (i.0, Kind::FCmGe),
        FMLA_Vd_V_2S_Vn_V_2S_Vm_V_2S(i)  => (i.0, Kind::FMla),
        FMLA_Vd_Vn_Vm(i)                 => (i.0, Kind::FMla),
        FMLS_Vd_V_2S_Vn_V_2S_Vm_V_2S(i)  => (i.0, Kind::FMls),
        FMLS_Vd_Vn_Vm(i)                 => (i.0, Kind::FMls),
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
        Kind::CmHi => em.vec_cmhi(vn, vm, size, q),
        Kind::CmHs => em.vec_cmhs(vn, vm, size, q),
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
        Kind::Smin | Kind::Smax | Kind::Umin | Kind::Umax => {
            // 64-bit lane min/max is unsupported (no PMINSQ/PMAXSQ pre-AVX-512).
            if size == 3 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            match kind {
                Kind::Smin => em.vec_smin(vn, vm, size, q),
                Kind::Smax => em.vec_smax(vn, vm, size, q),
                Kind::Umin => em.vec_umin(vn, vm, size, q),
                Kind::Umax => em.vec_umax(vn, vm, size, q),
                _ => unreachable!(),
            }
        }
        Kind::FAdd | Kind::FSub | Kind::FMul | Kind::FDiv | Kind::FMax | Kind::FMin
        | Kind::FCmEq | Kind::FCmGt | Kind::FCmGe => {
            // sz bit at 22 selects single (0) vs double (1).
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            match kind {
                Kind::FAdd  => em.vec_fadd (vn, vm, double, q),
                Kind::FSub  => em.vec_fsub (vn, vm, double, q),
                Kind::FMul  => em.vec_fmul (vn, vm, double, q),
                Kind::FDiv  => em.vec_fdiv (vn, vm, double, q),
                Kind::FMax  => em.vec_fmax (vn, vm, double, q),
                Kind::FMin  => em.vec_fmin (vn, vm, double, q),
                Kind::FCmEq => em.vec_fcmeq(vn, vm, double, q),
                Kind::FCmGt => em.vec_fcmgt(vn, vm, double, q),
                Kind::FCmGe => em.vec_fcmge(vn, vm, double, q),
                _ => unreachable!(),
            }
        }
        Kind::FMla | Kind::FMls => {
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            let vd_prev = em.get_v_q(rd);
            match kind {
                Kind::FMla => em.vec_fmla(vd_prev, vn, vm, double, q),
                Kind::FMls => em.vec_fmls(vd_prev, vn, vm, double, q),
                _ => unreachable!(),
            }
        }
    };

    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
