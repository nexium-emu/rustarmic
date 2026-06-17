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

    let (bot, low_mask_bits): (crate::ir::ValueRef, u32) = if imms >= immr {
        let extract_bits = imms - immr + 1;
        let amt = em.const_u64(immr as u64);
        let shifted = em.lsr(src, amt, size);
        let mask = if extract_bits >= width { (!0u64) >> (64 - width) }
                   else { (1u64 << extract_bits) - 1 };
        let mask_c = em.const_u64(mask);
        (em.and(shifted, mask_c, size), extract_bits)
    } else {
        let extract_bits = imms + 1;
        let mask = (1u64 << extract_bits) - 1;
        let mask_c = em.const_u64(mask);
        let masked = em.and(src, mask_c, size);
        let shift = (width - immr) as u64;
        let shift_c = em.const_u64(shift);
        (em.lsl(masked, shift_c, size), 0)
    };

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
            let mask = if imms >= immr {
                let bits_n = low_mask_bits;
                if bits_n >= width { (!0u64) >> (64 - width) }
                else { (1u64 << bits_n) - 1 }
            } else {
                let bits_n = imms + 1;
                let base_mask = (1u64 << bits_n) - 1;
                base_mask.wrapping_shl(width - immr)
            };
            let clear_mask = !mask;
            let clear_c = em.const_u64(if sf == 0 { clear_mask & 0xFFFF_FFFF } else { clear_mask });
            let cleared = em.and(dst_prev, clear_c, size);
            em.or(cleared, bot, size)
        }
    };

    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}
