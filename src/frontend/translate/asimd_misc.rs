use disarm64::decoder::ASIMDMISC;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind {
    Neg, Abs, Not, FNeg, FAbs, FSqrt, Xtn, Rev16, Rev32, Rev64,
    FRintN, FRintM, FRintP, FRintZ, FRintA, FRintX,
    CmEq0, CmGt0, CmGe0, CmLe0, CmLt0,
    FCmEq0, FCmGt0, FCmGe0, FCmLe0, FCmLt0,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDMISC) -> Result<InstStatus> {
    use ASIMDMISC::*;
    let (raw, kind) = match insn {
        NEG_Vd_Vn(i)   => (i.0, Kind::Neg),
        ABS_Vd_Vn(i)   => (i.0, Kind::Abs),
        NOT_Vd_Vn(i)   => (i.0, Kind::Not),
        FNEG_Vd_Vn(i)  => (i.0, Kind::FNeg),
        FABS_Vd_Vn(i)  => (i.0, Kind::FAbs),
        FSQRT_Vd_Vn(i) => (i.0, Kind::FSqrt),
        XTN_Vd_Vn(i)   => (i.0, Kind::Xtn),
        XTN2_Vd_Vn(i)  => (i.0, Kind::Xtn),
        REV16_Vd_Vn(i) => (i.0, Kind::Rev16),
        REV32_Vd_Vn(i) => (i.0, Kind::Rev32),
        REV64_Vd_Vn(i) => (i.0, Kind::Rev64),
        FRINTN_Vd_Vn(i) => (i.0, Kind::FRintN),
        FRINTM_Vd_Vn(i) => (i.0, Kind::FRintM),
        FRINTP_Vd_Vn(i) => (i.0, Kind::FRintP),
        FRINTZ_Vd_Vn(i) => (i.0, Kind::FRintZ),
        FRINTA_Vd_Vn(i) => (i.0, Kind::FRintA),
        FRINTX_Vd_Vn(i) => (i.0, Kind::FRintX),
        FRINTI_Vd_Vn(i) => (i.0, Kind::FRintX),
        CMEQ_Vd_Vn_IMM0(i) => (i.0, Kind::CmEq0),
        CMGT_Vd_Vn_IMM0(i) => (i.0, Kind::CmGt0),
        CMGE_Vd_Vn_IMM0(i) => (i.0, Kind::CmGe0),
        CMLE_Vd_Vn_IMM0(i) => (i.0, Kind::CmLe0),
        CMLT_Vd_Vn_IMM0(i) => (i.0, Kind::CmLt0),
        FCMEQ_Vd_Vn_FPIMM0(i) => (i.0, Kind::FCmEq0),
        FCMGT_Vd_Vn_FPIMM0(i) => (i.0, Kind::FCmGt0),
        FCMGE_Vd_Vn_FPIMM0(i) => (i.0, Kind::FCmGe0),
        FCMLE_Vd_Vn_FPIMM0(i) => (i.0, Kind::FCmLe0),
        FCMLT_Vd_Vn_FPIMM0(i) => (i.0, Kind::FCmLt0),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
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
        Kind::FNeg | Kind::FAbs | Kind::FSqrt => {
            let double = bit(raw, 22) == 1;
            match kind {
                Kind::FNeg  => em.vec_fneg(vn, double, q),
                Kind::FAbs  => em.vec_fabs(vn, double, q),
                Kind::FSqrt => em.vec_fsqrt(vn, double, q),
                _ => unreachable!(),
            }
        }
        Kind::Xtn => {
            if size > 2 {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            if q {
                let vd_prev = em.get_v_q(rd);
                em.vec_xtn2(vd_prev, vn, size + 1)
            } else {
                em.vec_xtn(vn, size + 1)
            }
        }
        Kind::FRintN | Kind::FRintM | Kind::FRintP | Kind::FRintZ
        | Kind::FRintA | Kind::FRintX => {
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            match kind {
                Kind::FRintN => em.vec_frintn(vn, double, q),
                Kind::FRintM => em.vec_frintm(vn, double, q),
                Kind::FRintP => em.vec_frintp(vn, double, q),
                Kind::FRintZ => em.vec_frintz(vn, double, q),
                Kind::FRintA => em.vec_frinta(vn, double, q),
                Kind::FRintX => em.vec_frintx(vn, double, q),
                _ => unreachable!(),
            }
        }
        Kind::Rev16 | Kind::Rev32 | Kind::Rev64 => {
            let max_src = match kind {
                Kind::Rev16 => 1,
                Kind::Rev32 => 2,
                Kind::Rev64 => 3,
                _ => unreachable!(),
            };
            if size >= max_src {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            let container_log2 = match kind {
                Kind::Rev16 => 1,
                Kind::Rev32 => 2,
                Kind::Rev64 => 3,
                _ => unreachable!(),
            };
            em.vec_rev(vn, size, container_log2, q)
        }
        Kind::CmEq0 | Kind::CmGt0 | Kind::CmGe0 | Kind::CmLe0 | Kind::CmLt0 => {
            let zlo = em.const_u64(0);
            let zhi = em.const_u64(0);
            let zero = em.vec_build_q(zlo, zhi);
            match kind {
                Kind::CmEq0 => em.vec_cmeq(vn, zero, size, q),
                Kind::CmGt0 => em.vec_cmgt(vn, zero, size, q),
                Kind::CmGe0 => em.vec_cmge(vn, zero, size, q),
                Kind::CmLe0 => em.vec_cmge(zero, vn, size, q),
                Kind::CmLt0 => em.vec_cmgt(zero, vn, size, q),
                _ => unreachable!(),
            }
        }
        Kind::FCmEq0 | Kind::FCmGt0 | Kind::FCmGe0 | Kind::FCmLe0 | Kind::FCmLt0 => {
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            let zlo = em.const_u64(0);
            let zhi = em.const_u64(0);
            let zero = em.vec_build_q(zlo, zhi);
            match kind {
                Kind::FCmEq0 => em.vec_fcmeq(vn, zero, double, q),
                Kind::FCmGt0 => em.vec_fcmgt(vn, zero, double, q),
                Kind::FCmGe0 => em.vec_fcmge(vn, zero, double, q),
                Kind::FCmLe0 => em.vec_fcmge(zero, vn, double, q),
                Kind::FCmLt0 => em.vec_fcmgt(zero, vn, double, q),
                _ => unreachable!(),
            }
        }
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
