use disarm64::decoder::MOVEWIDE;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: MOVEWIDE) -> Result<InstStatus> {
    use MOVEWIDE::*;
    let (kind, raw) = match insn {
        MOVZ_Rd_HALF(i) => (Kind::Z, i.0),
        MOVN_Rd_HALF(i) => (Kind::N, i.0),
        MOVK_Rd_HALF(i) => (Kind::K, i.0),
    };

    let sf = bit(raw, 31);
    let hw = bits(raw, 21, 2);
    let imm16 = bits(raw, 5, 16);
    let rd = bits(raw, 0, 5) as u8;

    if sf == 0 && hw >= 2 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let shift = hw * 16;
    let imm_shifted = (imm16 as u64) << shift;

    match kind {
        Kind::Z => {
            let c = em.const_u64(imm_shifted);
            em.set_gpr(rd, c, size);
        }
        Kind::N => {
            let mut value = !imm_shifted;
            if sf == 0 {
                value &= 0xFFFF_FFFF;
            }
            let c = em.const_u64(value);
            em.set_gpr(rd, c, size);
        }
        Kind::K => {
            let prev = em.get_gpr(rd, size);
            let mask = !((0xFFFFu64) << shift);
            let mask_c = em.const_u64(if sf == 0 { mask & 0xFFFF_FFFF } else { mask });
            let cleared = em.and(prev, mask_c, size);
            let imm_c = em.const_u64(imm_shifted);
            let merged = em.or(cleared, imm_c, size);
            em.set_gpr(rd, merged, size);
        }
    }
    Ok(InstStatus::Continue)
}

enum Kind {
    Z,
    N,
    K,
}
