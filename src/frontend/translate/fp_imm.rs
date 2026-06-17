use disarm64::decoder::FLOATIMM;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::bits;

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOATIMM) -> Result<InstStatus> {
    use FLOATIMM::*;
    let raw = match insn {
        FMOV_Fd_FPIMM(i)         => i.0,
        FMOV_Fd_S_S_FPIMM(i)     => i.0,
    };

    let ptype = bits(raw, 22, 2);
    let imm8  = bits(raw, 13, 8) as u8;
    let rd    = bits(raw, 0, 5) as u8;

    match ptype {
        0b00 => {
            let bits32 = expand_imm32(imm8);
            let v = em.const_u32(bits32);
            em.set_v_s(rd, v);
        }
        0b01 => {
            let bits64 = expand_imm64(imm8);
            let v = em.const_u64(bits64);
            em.set_v_d(rd, v);
        }
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    }
    Ok(InstStatus::Continue)
}

fn expand_imm64(imm8: u8) -> u64 {
    let a = ((imm8 >> 7) & 1) as u64;
    let b = ((imm8 >> 6) & 1) as u64;
    let cdefgh = (imm8 & 0x3F) as u64;
    let b_rep8 = if b == 1 { 0xFFu64 } else { 0 };
    (a << 63) | ((b ^ 1) << 62) | (b_rep8 << 54) | (cdefgh << 48)
}

fn expand_imm32(imm8: u8) -> u32 {
    let a = ((imm8 >> 7) & 1) as u32;
    let b = ((imm8 >> 6) & 1) as u32;
    let cdefgh = (imm8 & 0x3F) as u32;
    let b_rep5 = if b == 1 { 0x1F } else { 0 };
    (a << 31) | ((b ^ 1) << 30) | (b_rep5 << 25) | (cdefgh << 19)
}
