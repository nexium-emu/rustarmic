use disarm64::decoder::FLOATCCMP;

use crate::arch::Cond;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATCCMP) -> Result<InstStatus> {
    use FLOATCCMP::*;
    let raw = match insn {
        FCCMP_Fn_Fm_NZCV_COND(i)            => i.0,
        FCCMP_Fn_S_S_Fm_S_S_NZCV_COND(i)    => i.0,
        FCCMPE_Fn_Fm_NZCV_COND(i)           => i.0,
        FCCMPE_Fn_S_S_Fm_S_S_NZCV_COND(i)   => i.0,
    };

    let ptype = bits(raw, 22, 2);
    let rm    = bits(raw, 16, 5) as u8;
    let cond  = bits(raw, 12, 4) as u8;
    let rn    = bits(raw, 5, 5) as u8;
    let nzcv4 = bits(raw, 0, 4);

    let (fcmp_op, ty) = match ptype {
        0b00 => (Op::Fcmp32, Ty::U32),
        0b01 => (Op::Fcmp64, Ty::U64),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    };

    let cur_nzcv = em.get_nzcv();

    let lhs = if ptype == 0b00 { em.get_v_s(rn) } else { em.get_v_d(rn) };
    let rhs = if ptype == 0b00 { em.get_v_s(rm) } else { em.get_v_d(rm) };
    em.push(Armlet::new(fcmp_op, Ty::Void).with_args(&[lhs, rhs]).with_imm(ty as u64));

    let fcmp_nzcv = em.get_nzcv();
    let fallback  = em.const_u32(nzcv4 as u32);

    let chosen = em.push(Armlet::new(Op::Csel32, Ty::U32)
        .with_args(&[fcmp_nzcv, fallback, cur_nzcv])
        .with_imm(Cond::from_bits(cond) as u64));

    em.set_nzcv(chosen);
    Ok(InstStatus::Continue)
}
