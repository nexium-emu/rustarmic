//! ASIMD "shift by immediate" form. Coverage: SHL, USHR, SSHR.
//!
//! The element width is encoded in immh (bits 22:19):
//!   immh = 0001        → 8-bit lanes
//!   immh = 001x        → 16-bit lanes
//!   immh = 01xx        → 32-bit lanes
//!   immh = 1xxx        → 64-bit lanes
//! The shift amount is decoded from immh:immb (7 bits):
//!   SHL:  amt = (immh:immb) - lane_bits
//!   SHR:  amt = 2*lane_bits - (immh:immb)

use disarm64::decoder::ASIMDSHF;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Shl, Ushr, Sshr }

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDSHF) -> Result<InstStatus> {
    use ASIMDSHF::*;
    let (raw, kind) = match insn {
        SHL_Vd_Vn_IMM_VLSL(i)   => (i.0, Kind::Shl),
        USHR_Vd_Vn_IMM_VLSR(i)  => (i.0, Kind::Ushr),
        SSHR_Vd_Vn_IMM_VLSR(i)  => (i.0, Kind::Sshr),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q     = bit(raw, 30) == 1;
    let immh  = bits(raw, 19, 4);
    let immb  = bits(raw, 16, 3);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;

    if immh == 0 {
        return Err(Error::Decode { pc: em.current_pc, opcode: raw });
    }
    // Lane log2 byte size from leading bit of immh: bit 3=64, bit 2=32, bit 1=16, bit 0=8.
    let lane_log2 = if (immh & 0b1000) != 0 { 3 }
                    else if (immh & 0b0100) != 0 { 2 }
                    else if (immh & 0b0010) != 0 { 1 }
                    else { 0 };
    let lane_bits = 8u32 << lane_log2;
    let immhb = (immh << 3) | immb; // 7-bit value
    let shift = match kind {
        Kind::Shl  => immhb.wrapping_sub(lane_bits),
        Kind::Ushr | Kind::Sshr => (2 * lane_bits).wrapping_sub(immhb),
    };

    let vn = em.get_v_q(rn);
    let result = match kind {
        Kind::Shl  => em.vec_shl_imm (vn, lane_log2, shift, q),
        Kind::Ushr => em.vec_ushr_imm(vn, lane_log2, shift, q),
        Kind::Sshr => em.vec_sshr_imm(vn, lane_log2, shift, q),
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
