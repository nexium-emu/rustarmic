use disarm64::decoder::EXTRACT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: EXTRACT) -> Result<InstStatus> {
    use EXTRACT::*;
    let raw = match insn {
        EXTR_Rd_Rn_Rm_IMMS(i) => i.0,
    };
    let sf = bit(raw, 31);
    let n = bit(raw, 22);
    let rm = bits(raw, 16, 5) as u8;
    let imms = bits(raw, 10, 6);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    if sf != n {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let width = if sf == 1 { 64u32 } else { 32 };
    if imms >= width {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }

    let result = if rn == rm {
        let v = em.get_gpr(rn, size);
        let amt = em.const_u64(imms as u64);
        em.ror(v, amt, size)
    } else {
        let hi = em.get_gpr(rn, size);
        let lo = em.get_gpr(rm, size);
        let hi_shift = em.const_u64((width - imms) as u64);
        let lo_shift = em.const_u64(imms as u64);
        let hi_part = em.lsl(hi, hi_shift, size);
        let lo_part = em.lsr(lo, lo_shift, size);
        em.or(hi_part, lo_part, size)
    };
    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}
