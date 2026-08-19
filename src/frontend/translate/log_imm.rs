use disarm64::decoder::LOG_IMM;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, decode_bit_masks};

#[derive(Clone, Copy)]
enum LogOp {
    And,
    Or,
    Eor,
    Ands,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: LOG_IMM) -> Result<InstStatus> {
    use LOG_IMM::*;
    let (raw, op) = match insn {
        AND_Rd_SP_Rn_LIMM(i) => (i.0, LogOp::And),
        ORR_Rd_SP_Rn_LIMM(i) => (i.0, LogOp::Or),
        EOR_Rd_SP_Rn_LIMM(i) => (i.0, LogOp::Eor),
        ANDS_Rd_Rn_LIMM(i) => (i.0, LogOp::Ands),
    };

    let sf = bit(raw, 31);
    let n = bit(raw, 22);
    let immr = bits(raw, 16, 6);
    let imms = bits(raw, 10, 6);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let width = if sf == 1 { 64 } else { 32 };
    if sf == 0 && n != 0 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let imm = decode_bit_masks(n, imms, immr, width).ok_or(Error::Decode {
        pc: em.current_pc,
        opcode: raw,
    })?;

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let sp_form_dst = !matches!(op, LogOp::Ands);
    let a = em.get_gpr(rn, size);
    let b = em.const_u64(imm);

    let result = match op {
        LogOp::And | LogOp::Ands => em.and(a, b, size),
        LogOp::Or => em.or(a, b, size),
        LogOp::Eor => em.eor(a, b, size),
    };

    if matches!(op, LogOp::Ands) {
        let zero = em.const_u64(0);
        em.subs(result, zero, size);
    }

    em.set_x_or_sp(rd, result, sp_form_dst);
    Ok(InstStatus::Continue)
}
