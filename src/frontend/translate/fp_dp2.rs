//! Floating-point data-processing (2 source) — `FADD`, `FSUB`, `FMUL`,
//! `FDIV`. Scalar S/D for now; FMAX/FMIN/FNMUL/half-precision later.

use disarm64::decoder::FLOATDP2;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty, ValueRef};
use crate::util::bits::bits;

#[derive(Clone, Copy)]
enum Kind { Add, Sub, Mul, Div }

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATDP2) -> Result<InstStatus> {
    use FLOATDP2::*;
    let (raw, kind) = match insn {
        FADD_Fd_Fn_Fm(i)             => (i.0, Kind::Add),
        FADD_Fd_S_S_Fn_S_S_Fm_S_S(i) => (i.0, Kind::Add),
        FSUB_Fd_Fn_Fm(i)             => (i.0, Kind::Sub),
        FSUB_Fd_S_S_Fn_S_S_Fm_S_S(i) => (i.0, Kind::Sub),
        FMUL_Fd_Fn_Fm(i)             => (i.0, Kind::Mul),
        FMUL_Fd_S_S_Fn_S_S_Fm_S_S(i) => (i.0, Kind::Mul),
        FDIV_Fd_Fn_Fm(i)             => (i.0, Kind::Div),
        FDIV_Fd_S_S_Fn_S_S_Fm_S_S(i) => (i.0, Kind::Div),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let ptype = bits(raw, 22, 2);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    match ptype {
        0b00 => {
            let a = em.get_v_s(rn);
            let b = em.get_v_s(rm);
            let r = fbin(em, a, b, kind, false);
            em.set_v_s(rd, r);
        }
        0b01 => {
            let a = em.get_v_d(rn);
            let b = em.get_v_d(rm);
            let r = fbin(em, a, b, kind, true);
            em.set_v_d(rd, r);
        }
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    }
    Ok(InstStatus::Continue)
}

fn fbin(em: &mut IrEmitter<'_>, a: ValueRef, b: ValueRef, k: Kind, is_double: bool) -> ValueRef {
    let (op, ty) = match (k, is_double) {
        (Kind::Add, false) => (Op::Fadd32, Ty::U32),
        (Kind::Add, true)  => (Op::Fadd64, Ty::U64),
        (Kind::Sub, false) => (Op::Fsub32, Ty::U32),
        (Kind::Sub, true)  => (Op::Fsub64, Ty::U64),
        (Kind::Mul, false) => (Op::Fmul32, Ty::U32),
        (Kind::Mul, true)  => (Op::Fmul64, Ty::U64),
        (Kind::Div, false) => (Op::Fdiv32, Ty::U32),
        (Kind::Div, true)  => (Op::Fdiv64, Ty::U64),
    };
    em.push(Armlet::new(op, ty).with_args(&[a, b]))
}
