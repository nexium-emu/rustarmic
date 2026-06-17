use disarm64::decoder::FLOATCMP;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATCMP) -> Result<InstStatus> {
    use FLOATCMP::*;
    let (raw, against_zero) = match insn {
        FCMP_Fn_Fm(i)             => (i.0, false),
        FCMP_Fn_S_S_Fm_S_S(i)     => (i.0, false),
        FCMPE_Fn_Fm(i)            => (i.0, false),
        FCMPE_Fn_S_S_Fm_S_S(i)    => (i.0, false),
        FCMP_Fn_FPIMM0(i)         => (i.0, true),
        FCMP_Fn_S_S_FPIMM0(i)     => (i.0, true),
        FCMPE_Fn_FPIMM0(i)        => (i.0, true),
        FCMPE_Fn_S_S_FPIMM0(i)    => (i.0, true),
    };

    let ptype = bits(raw, 22, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;

    let (op, ty) = match ptype {
        0b00 => (Op::Fcmp32, Ty::U32),
        0b01 => (Op::Fcmp64, Ty::U64),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    };

    let lhs = if ptype == 0b00 { em.get_v_s(rn) } else { em.get_v_d(rn) };
    let rhs = if against_zero {
        em.const_u64(0)
    } else if ptype == 0b00 {
        em.get_v_s(rm)
    } else {
        em.get_v_d(rm)
    };

    em.push(Armlet::new(op, ty).with_args(&[lhs, rhs]));
    Ok(InstStatus::Continue)
}
