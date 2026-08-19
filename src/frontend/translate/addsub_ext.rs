use disarm64::decoder::ADDSUB_EXT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ADDSUB_EXT) -> Result<InstStatus> {
    use ADDSUB_EXT::*;
    let (raw, sub, set_flags) = match insn {
        ADD_Rd_SP_Rn_SP_Rm_EXT(i) => (i.0, false, false),
        ADDS_Rd_Rn_SP_Rm_EXT(i) => (i.0, false, true),
        SUB_Rd_SP_Rn_SP_Rm_EXT(i) => (i.0, true, false),
        SUBS_Rd_Rn_SP_Rm_EXT(i) => (i.0, true, true),
    };

    let sf = bit(raw, 31);
    let rm = bits(raw, 16, 5) as u8;
    let option_ = bits(raw, 13, 3);
    let imm3 = bits(raw, 10, 3);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    if imm3 > 4 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let sp_form = !set_flags;
    let a = em.get_x_or_sp(rn, sp_form);

    let mut b = em.get_x(rm);
    let (extracted_width, signed) = match option_ {
        0b000 => (8, false),
        0b001 => (16, false),
        0b010 => (32, false),
        0b011 => (64, false),
        0b100 => (8, true),
        0b101 => (16, true),
        0b110 => (32, true),
        0b111 => (64, true),
        _ => unreachable!(),
    };

    if extracted_width < 64 {
        let mask = (1u64 << extracted_width) - 1;
        let mask_c = em.const_u64(mask);
        b = em.and(b, mask_c, RegSize::X);
        if signed {
            let shl = em.const_u64((64 - extracted_width) as u64);
            let sh1 = em.lsl(b, shl, RegSize::X);
            let shl2 = em.const_u64((64 - extracted_width) as u64);
            b = em.asr(sh1, shl2, RegSize::X);
        }
    }
    if imm3 != 0 {
        let amt = em.const_u64(imm3 as u64);
        b = em.lsl(b, amt, RegSize::X);
    }

    if size == RegSize::W {
        let mask_c = em.const_u64(0xFFFF_FFFF);
        b = em.and(b, mask_c, RegSize::X);
    }

    if set_flags {
        let result = if sub {
            em.subs(a, b, size)
        } else {
            em.adds(a, b, size)
        };
        em.set_x(rd, result);
    } else {
        let result = if sub {
            em.sub(a, b, size)
        } else {
            em.add(a, b, size)
        };
        em.set_x_or_sp(rd, result, sp_form);
    }
    Ok(InstStatus::Continue)
}
