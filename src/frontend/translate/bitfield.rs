use disarm64::decoder::BITFIELD;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Ubfm, Sbfm, Bfm }

pub fn translate(em: &mut IrEmitter<'_>, insn: BITFIELD) -> Result<InstStatus> {
    use BITFIELD::*;
    let (raw, kind) = match insn {
        UBFM_Rd_Rn_IMMR_IMMS(i) => (i.0, Kind::Ubfm),
        SBFM_Rd_Rn_IMMR_IMMS(i) => (i.0, Kind::Sbfm),
        BFM_Rd_Rn_IMMR_IMMS(i)  => (i.0, Kind::Bfm),
    };

    let sf   = bit(raw, 31);
    let n    = bit(raw, 22);
    let immr = bits(raw, 16, 6);
    let imms = bits(raw, 10, 6);
    let rn   = bits(raw, 5, 5) as u8;
    let rd   = bits(raw, 0, 5) as u8;

    if sf != n {
        return Err(Error::Decode { pc: em.current_pc, opcode: raw });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let width = if sf == 1 { 64u32 } else { 32 };

    let src = em.get_gpr(rn, size);
    let r_amt = em.const_u64(immr as u64);
    let rotated = em.ror(src, r_amt, size);

    let mask_bits = imms + 1;
    let mask = if mask_bits >= width { (!0u64) >> (64 - width) } else { (1u64 << mask_bits) - 1 };
    let mask_c = em.const_u64(mask);
    let bot = em.and(rotated, mask_c, size);

    let result = match kind {
        Kind::Ubfm => bot,
        Kind::Sbfm => {
            let high_bit = if imms < immr { width - 1 } else { imms - immr };
            let shl_amt = (width - 1 - high_bit) as u64;
            let amt_c = em.const_u64(shl_amt);
            let shifted_l = em.lsl(bot, amt_c, size);
            let amt2 = em.const_u64(shl_amt);
            em.asr(shifted_l, amt2, size)
        }
        Kind::Bfm => {
            let dst_prev = em.get_gpr(rd, size);
            let clear_mask = !mask;
            let clear_c = em.const_u64(if sf == 0 { clear_mask & 0xFFFF_FFFF } else { clear_mask });
            let cleared = em.and(dst_prev, clear_c, size);
            em.or(cleared, bot, size)
        }
    };

    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}
