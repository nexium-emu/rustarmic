use disarm64::decoder::FLOATDP1;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty, ValueRef};
use crate::util::bits::bits;

#[derive(Clone, Copy)]
enum Kind {
    Mov,
    Neg,
    Abs,
    Sqrt,
    RintA,
    RintM,
    RintP,
    Fcvt,
    /// FRINT{N,Z,X,I}; I rounds like X, in the current mode.
    Rint(u8),
}

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATDP1) -> Result<InstStatus> {
    use FLOATDP1::*;
    let (raw, kind) = match insn {
        FMOV_Fd_Fn(i) => (i.0, Kind::Mov),
        FMOV_Fd_S_S_Fn_S_S(i) => (i.0, Kind::Mov),
        FNEG_Fd_Fn(i) => (i.0, Kind::Neg),
        FNEG_Fd_S_S_Fn_S_S(i) => (i.0, Kind::Neg),
        FABS_Fd_Fn(i) => (i.0, Kind::Abs),
        FABS_Fd_S_S_Fn_S_S(i) => (i.0, Kind::Abs),
        FSQRT_Fd_Fn(i) => (i.0, Kind::Sqrt),
        FSQRT_Fd_S_S_Fn_S_S(i) => (i.0, Kind::Sqrt),
        FRINTA_Fd_Fn(i) => (i.0, Kind::RintA),
        FRINTA_Fd_S_S_Fn_S_S(i) => (i.0, Kind::RintA),
        FRINTM_Fd_Fn(i) => (i.0, Kind::RintM),
        FRINTM_Fd_S_S_Fn_S_S(i) => (i.0, Kind::RintM),
        FRINTP_Fd_Fn(i) => (i.0, Kind::RintP),
        FRINTP_Fd_S_S_Fn_S_S(i) => (i.0, Kind::RintP),
        FCVT_Fd_Fn(i) => (i.0, Kind::Fcvt),
        FRINTN_Fd_Fn(i) => (i.0, Kind::Rint(b'n')),
        FRINTZ_Fd_Fn(i) => (i.0, Kind::Rint(b'z')),
        FRINTX_Fd_Fn(i) => (i.0, Kind::Rint(b'x')),
        FRINTI_Fd_Fn(i) => (i.0, Kind::Rint(b'x')),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let ptype = bits(raw, 22, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    if matches!(kind, Kind::Fcvt) {
        let opc = bits(raw, 15, 2);
        return translate_fcvt(em, ptype, opc, rn, rd);
    }

    let is_double = match ptype {
        0b00 => false,
        0b01 => true,
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };

    if let Kind::Rint(mode) = kind {
        let v = em.get_v_q(rn);
        let r = round_lanes(em, v, mode, is_double, false);
        if is_double {
            let lo = em.vec_extract_lo64(r);
            em.set_v_d(rd, lo);
        } else {
            let s = em.vec_extract_u32(r, 0);
            em.set_v_s(rd, s);
        }
        return Ok(InstStatus::Continue);
    }

    let src = if is_double {
        em.get_v_d(rn)
    } else {
        em.get_v_s(rn)
    };

    let result = match kind {
        Kind::RintA => round_to_nearest(em, src, is_double, true),
        Kind::RintM => round_to_minus_infinity(em, src, is_double),
        Kind::RintP => round_to_plus_infinity(em, src, is_double),
        Kind::Mov => src,
        Kind::Neg => {
            let (op, ty) = if is_double {
                (Op::Fneg64, Ty::U64)
            } else {
                (Op::Fneg32, Ty::U32)
            };
            em.push(Armlet::new(op, ty).with_args(&[src]))
        }
        Kind::Abs => {
            let (op, ty) = if is_double {
                (Op::Fabs64, Ty::U64)
            } else {
                (Op::Fabs32, Ty::U32)
            };
            em.push(Armlet::new(op, ty).with_args(&[src]))
        }
        Kind::Sqrt => {
            let (op, ty) = if is_double {
                (Op::Fsqrt64, Ty::U64)
            } else {
                (Op::Fsqrt32, Ty::U32)
            };
            em.push(Armlet::new(op, ty).with_args(&[src]))
        }
        Kind::Fcvt | Kind::Rint(_) => unreachable!("handled before this match"),
    };

    if is_double {
        em.set_v_d(rd, result);
    } else {
        em.set_v_s(rd, result);
    }
    Ok(InstStatus::Continue)
}

// Explicit rounding, independent of the FPCR rounding-mode field.
// Like the existing packed rounding backend, this currently provides
// default-FPCR values rather than complete DN/FZ/FPSR arithmetic emulation.
pub(super) fn round_to_minus_infinity(
    em: &mut IrEmitter<'_>,
    src: ValueRef,
    is_double: bool,
) -> ValueRef {
    round_scalar(em, src, is_double, ScalarRounding::MinusInfinity)
}

pub(super) fn round_to_plus_infinity(
    em: &mut IrEmitter<'_>,
    src: ValueRef,
    is_double: bool,
) -> ValueRef {
    round_scalar(em, src, is_double, ScalarRounding::PlusInfinity)
}

pub(super) fn round_to_nearest(
    em: &mut IrEmitter<'_>,
    src: ValueRef,
    is_double: bool,
    ties_away: bool,
) -> ValueRef {
    round_scalar(
        em,
        src,
        is_double,
        if ties_away {
            ScalarRounding::NearestAway
        } else {
            ScalarRounding::NearestEven
        },
    )
}

enum ScalarRounding {
    MinusInfinity,
    PlusInfinity,
    NearestEven,
    NearestAway,
}

fn round_scalar(
    em: &mut IrEmitter<'_>,
    src: ValueRef,
    is_double: bool,
    rounding: ScalarRounding,
) -> ValueRef {
    let src = if is_double {
        src
    } else {
        // A U32 SSA spill need not have zero in its unused upper word.
        let mask = em.const_u64(u32::MAX as u64);
        em.and(src, mask, RegSize::X)
    };
    let zero = em.const_u64(0);
    let vector = em.vec_build_q(src, zero);
    let rounded = match rounding {
        ScalarRounding::MinusInfinity => em.vec_frintm(vector, is_double, false),
        ScalarRounding::PlusInfinity => em.vec_frintp(vector, is_double, false),
        ScalarRounding::NearestEven => em.vec_frintn(vector, is_double, false),
        ScalarRounding::NearestAway => em.vec_frinta(vector, is_double, false),
    };
    if is_double {
        em.vec_extract_lo64(rounded)
    } else {
        em.vec_extract_u32(rounded, 0)
    }
}

fn translate_fcvt(
    em: &mut IrEmitter<'_>,
    src_ptype: u32,
    dst_opc: u32,
    rn: u8,
    rd: u8,
) -> Result<InstStatus> {
    match (src_ptype, dst_opc) {
        (0b00, 0b01) => {
            let src = em.get_v_s(rn);
            let r = em.push(Armlet::new(Op::FcvtSD, Ty::U64).with_args(&[src]));
            em.set_v_d(rd, r);
        }
        (0b01, 0b00) => {
            let src = em.get_v_d(rn);
            let r = em.push(Armlet::new(Op::FcvtDS, Ty::U32).with_args(&[src]));
            em.set_v_s(rd, r);
        }
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    }
    Ok(InstStatus::Continue)
}

/// Rounds the lanes of `v` to integral values: `mode` is one of the
/// FRINT suffixes n / p / m / z / a / x.
pub(crate) fn round_lanes(
    em: &mut IrEmitter<'_>,
    v: crate::ir::ValueRef,
    mode: u8,
    double: bool,
    q: bool,
) -> crate::ir::ValueRef {
    match mode {
        b'n' => em.vec_frintn(v, double, q),
        b'p' => em.vec_frintp(v, double, q),
        b'm' => em.vec_frintm(v, double, q),
        b'z' => em.vec_frintz(v, double, q),
        b'a' => em.vec_frinta(v, double, q),
        _ => em.vec_frintx(v, double, q),
    }
}
