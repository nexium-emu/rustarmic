use disarm64::decoder::FLOATDP1;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

#[derive(Clone, Copy)]
enum Kind {
    Mov,
    Neg,
    Abs,
    Sqrt,
    Fcvt,
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
        FCVT_Fd_Fn(i) => (i.0, Kind::Fcvt),
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

    let src = if is_double {
        em.get_v_d(rn)
    } else {
        em.get_v_s(rn)
    };

    let result = match kind {
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
        Kind::Fcvt => unreachable!("handled before this match"),
    };

    if is_double {
        em.set_v_d(rd, result);
    } else {
        em.set_v_s(rd, result);
    }
    Ok(InstStatus::Continue)
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
