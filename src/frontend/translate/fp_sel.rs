use disarm64::decoder::FLOATSEL;

use crate::arch::Cond;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATSEL) -> Result<InstStatus> {
    use FLOATSEL::*;
    let raw = match insn {
        FCSEL_Fd_Fn_Fm_COND(i) => i.0,
        FCSEL_Fd_S_S_Fn_S_S_Fm_S_S_COND(i) => i.0,
    };

    let ptype = bits(raw, 22, 2);
    let rm = bits(raw, 16, 5) as u8;
    let cond = bits(raw, 12, 4) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let (op, ty) = match ptype {
        0b00 => (Op::Csel32, Ty::U32),
        0b01 => (Op::Csel64, Ty::U64),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };

    let taken = if ptype == 0b00 {
        em.get_v_s(rn)
    } else {
        em.get_v_d(rn)
    };
    let not_taken = if ptype == 0b00 {
        em.get_v_s(rm)
    } else {
        em.get_v_d(rm)
    };
    let nzcv = em.get_nzcv();

    let result = em.push(
        Armlet::new(op, ty)
            .with_args(&[taken, not_taken, nzcv])
            .with_imm(Cond::from_bits(cond) as u64),
    );

    if ptype == 0b00 {
        em.set_v_s(rd, result);
    } else {
        em.set_v_d(rd, result);
    }
    Ok(InstStatus::Continue)
}
