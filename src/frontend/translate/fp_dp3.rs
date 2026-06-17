use disarm64::decoder::FLOATDP3;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty, ValueRef};
use crate::util::bits::bits;

#[derive(Clone, Copy)]
enum Kind { Fmadd, Fmsub, Fnmadd, Fnmsub }

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATDP3) -> Result<InstStatus> {
    use FLOATDP3::*;
    let (raw, kind) = match insn {
        FMADD_Fd_Fn_Fm_Fa(i)                  => (i.0, Kind::Fmadd),
        FMADD_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(i)  => (i.0, Kind::Fmadd),
        FMSUB_Fd_Fn_Fm_Fa(i)                  => (i.0, Kind::Fmsub),
        FMSUB_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(i)  => (i.0, Kind::Fmsub),
        FNMADD_Fd_Fn_Fm_Fa(i)                 => (i.0, Kind::Fnmadd),
        FNMADD_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(i) => (i.0, Kind::Fnmadd),
        FNMSUB_Fd_Fn_Fm_Fa(i)                 => (i.0, Kind::Fnmsub),
        FNMSUB_Fd_S_S_Fn_S_S_Fm_S_S_Fa_S_S(i) => (i.0, Kind::Fnmsub),
    };

    let ptype = bits(raw, 22, 2);
    let rm = bits(raw, 16, 5) as u8;
    let ra = bits(raw, 10, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let is_double = match ptype {
        0b00 => false,
        0b01 => true,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    };

    let n_val = if is_double { em.get_v_d(rn) } else { em.get_v_s(rn) };
    let m_val = if is_double { em.get_v_d(rm) } else { em.get_v_s(rm) };
    let a_val = if is_double { em.get_v_d(ra) } else { em.get_v_s(ra) };

    let prod = fmul(em, n_val, m_val, is_double);
    let result = match kind {
        Kind::Fmadd  => fadd(em, a_val, prod, is_double),
        Kind::Fmsub  => fsub(em, a_val, prod, is_double),
        Kind::Fnmadd => {
            let sum = fadd(em, a_val, prod, is_double);
            fneg(em, sum, is_double)
        }
        Kind::Fnmsub => fsub(em, prod, a_val, is_double),
    };

    if is_double { em.set_v_d(rd, result); } else { em.set_v_s(rd, result); }
    Ok(InstStatus::Continue)
}

fn fmul(em: &mut IrEmitter<'_>, a: ValueRef, b: ValueRef, is_double: bool) -> ValueRef {
    let (op, ty) = if is_double { (Op::Fmul64, Ty::U64) } else { (Op::Fmul32, Ty::U32) };
    em.push(Armlet::new(op, ty).with_args(&[a, b]))
}

fn fadd(em: &mut IrEmitter<'_>, a: ValueRef, b: ValueRef, is_double: bool) -> ValueRef {
    let (op, ty) = if is_double { (Op::Fadd64, Ty::U64) } else { (Op::Fadd32, Ty::U32) };
    em.push(Armlet::new(op, ty).with_args(&[a, b]))
}

fn fsub(em: &mut IrEmitter<'_>, a: ValueRef, b: ValueRef, is_double: bool) -> ValueRef {
    let (op, ty) = if is_double { (Op::Fsub64, Ty::U64) } else { (Op::Fsub32, Ty::U32) };
    em.push(Armlet::new(op, ty).with_args(&[a, b]))
}

fn fneg(em: &mut IrEmitter<'_>, v: ValueRef, is_double: bool) -> ValueRef {
    let (op, ty) = if is_double { (Op::Fneg64, Ty::U64) } else { (Op::Fneg32, Ty::U32) };
    em.push(Armlet::new(op, ty).with_args(&[v]))
}
