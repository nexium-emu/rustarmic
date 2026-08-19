use disarm64::decoder::LOG_SHIFT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum LogOp {
    And,
    Or,
    Eor,
    Ands,
    Bic,
    Orn,
    Eon,
    Bics,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: LOG_SHIFT) -> Result<InstStatus> {
    use LOG_SHIFT::*;
    let (raw, op) = match insn {
        AND_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::And),
        ORR_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Or),
        EOR_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Eor),
        ANDS_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Ands),
        BIC_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Bic),
        ORN_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Orn),
        EON_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Eon),
        BICS_Rd_Rn_Rm_SFT(i) => (i.0, LogOp::Bics),
    };

    let sf = bit(raw, 31);
    let shift = bits(raw, 22, 2);
    let rm = bits(raw, 16, 5) as u8;
    let imm6 = bits(raw, 10, 6);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    if sf == 0 && imm6 >= 32 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }

    let a = em.get_gpr(rn, size);
    let mut b = em.get_gpr(rm, size);

    if imm6 != 0 {
        let amt = em.const_u64(imm6 as u64);
        b = match shift {
            0b00 => em.lsl(b, amt, size),
            0b01 => em.lsr(b, amt, size),
            0b10 => em.asr(b, amt, size),
            0b11 => em.ror(b, amt, size),
            _ => unreachable!(),
        };
    }

    let inverted = matches!(op, LogOp::Bic | LogOp::Orn | LogOp::Eon | LogOp::Bics);
    if inverted {
        let all_ones = em.const_u64(if sf == 1 { !0u64 } else { 0xFFFF_FFFFu64 });
        b = em.eor(b, all_ones, size);
    }

    let result = match op {
        LogOp::And | LogOp::Ands | LogOp::Bic | LogOp::Bics => em.and(a, b, size),
        LogOp::Or | LogOp::Orn => em.or(a, b, size),
        LogOp::Eor | LogOp::Eon => em.eor(a, b, size),
    };

    if matches!(op, LogOp::Ands | LogOp::Bics) {
        let zero = em.const_u64(0);
        em.subs(result, zero, size);
    }

    em.set_x(rd, result);
    Ok(InstStatus::Continue)
}
