use disarm64::decoder::{CRYPTOSHA2, CRYPTOSHA3};

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::bits;

pub fn translate_two(em: &mut IrEmitter<'_>, insn: CRYPTOSHA2) -> Result<InstStatus> {
    let raw = match insn {
        CRYPTOSHA2::SHA256SU0_Vd_Vn(i) => i.0,
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    emit(em, raw, Op::Sha256Schedule0, false)
}

pub fn translate_three(em: &mut IrEmitter<'_>, insn: CRYPTOSHA3) -> Result<InstStatus> {
    let (raw, op) = match insn {
        CRYPTOSHA3::SHA256H_Fd_Fn_Vm(i) => (i.0, Op::Sha256Hash),
        CRYPTOSHA3::SHA256H2_Fd_Fn_Vm(i) => (i.0, Op::Sha256Hash2),
        CRYPTOSHA3::SHA256SU1_Vd_Vn_Vm(i) => (i.0, Op::Sha256Schedule1),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };
    emit(em, raw, op, true)
}

fn emit(em: &mut IrEmitter<'_>, raw: u32, op: Op, three: bool) -> Result<InstStatus> {
    let rd = bits(raw, 0, 5) as u8;
    let state = em.get_v_q(rd);
    let source = em.get_v_q(bits(raw, 5, 5) as u8);
    let mut a = Armlet::new(op, Ty::U128).with_args(&[state, source]);
    if three {
        a.args[2] = em.get_v_q(bits(raw, 16, 5) as u8);
    }
    let result = em.push(a);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
