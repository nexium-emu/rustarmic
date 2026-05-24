//! Floating-point data-processing (1 source) — `FMOV`, `FNEG`, `FABS`,
//! `FSQRT`, etc. Phase 1 covers `FMOV Vd, Vn` only; arithmetic unaries
//! follow in later phases.

use disarm64::decoder::FLOATDP1;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

#[derive(Clone, Copy)]
enum Kind { Mov, Neg, Abs, Sqrt }

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATDP1) -> Result<InstStatus> {
    use FLOATDP1::*;
    let (raw, kind) = match insn {
        FMOV_Fd_Fn(i)             => (i.0, Kind::Mov),
        FMOV_Fd_S_S_Fn_S_S(i)     => (i.0, Kind::Mov),
        FNEG_Fd_Fn(i)             => (i.0, Kind::Neg),
        FNEG_Fd_S_S_Fn_S_S(i)     => (i.0, Kind::Neg),
        FABS_Fd_Fn(i)             => (i.0, Kind::Abs),
        FABS_Fd_S_S_Fn_S_S(i)     => (i.0, Kind::Abs),
        FSQRT_Fd_Fn(i)            => (i.0, Kind::Sqrt),
        FSQRT_Fd_S_S_Fn_S_S(i)    => (i.0, Kind::Sqrt),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let ptype = bits(raw, 22, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let is_double = match ptype {
        0b00 => false,
        0b01 => true,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    };

    let src = if is_double { em.get_v_d(rn) } else { em.get_v_s(rn) };
    let size = if is_double { RegSize::X } else { RegSize::W };

    let result = match kind {
        Kind::Mov => src,
        Kind::Neg => {
            let sign = if is_double {
                em.const_u64(0x8000_0000_0000_0000)
            } else {
                em.const_u32(0x8000_0000)
            };
            em.eor(src, sign, size)
        }
        Kind::Abs => {
            let mask = if is_double {
                em.const_u64(0x7FFF_FFFF_FFFF_FFFF)
            } else {
                em.const_u32(0x7FFF_FFFF)
            };
            em.and(src, mask, size)
        }
        Kind::Sqrt => {
            let (op, ty) = if is_double { (Op::Fsqrt64, Ty::U64) } else { (Op::Fsqrt32, Ty::U32) };
            em.push(Armlet::new(op, ty).with_args(&[src]))
        }
    };

    if is_double { em.set_v_d(rd, result); }
    else { em.set_v_s(rd, result); }
    Ok(InstStatus::Continue)
}
