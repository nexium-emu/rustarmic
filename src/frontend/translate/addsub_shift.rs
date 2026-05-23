use disarm64::decoder::ADDSUB_SHIFT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ADDSUB_SHIFT) -> Result<InstStatus> {
    use ADDSUB_SHIFT::*;
    let (raw, sub, set_flags) = match insn {
        ADD_Rd_Rn_Rm_SFT(i)  => (i.0, false, false),
        ADDS_Rd_Rn_Rm_SFT(i) => (i.0, false, true),
        SUB_Rd_Rn_Rm_SFT(i)  => (i.0, true,  false),
        SUBS_Rd_Rn_Rm_SFT(i) => (i.0, true,  true),
    };

    let sf    = bit(raw, 31);
    let shift = bits(raw, 22, 2);
    let rm    = bits(raw, 16, 5) as u8;
    let imm6  = bits(raw, 10, 6);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;

    if shift == 0b11 {
        return Err(Error::Decode { pc: em.current_pc, opcode: raw });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    if sf == 0 && imm6 >= 32 {
        return Err(Error::Decode { pc: em.current_pc, opcode: raw });
    }

    let a = em.get_gpr(rn, size);
    let mut b = em.get_gpr(rm, size);
    if imm6 != 0 {
        let amt = em.const_u64(imm6 as u64);
        b = match shift {
            0b00 => em.lsl(b, amt, size),
            0b01 => em.lsr(b, amt, size),
            0b10 => em.asr(b, amt, size),
            _ => unreachable!(),
        };
    }

    if set_flags {
        let result = if sub { em.subs(a, b, size) } else { em.adds(a, b, size) };
        em.set_x(rd, result);
    } else {
        let result = if sub { em.sub(a, b, size) } else { em.add(a, b, size) };
        em.set_x(rd, result);
    }
    Ok(InstStatus::Continue)
}
