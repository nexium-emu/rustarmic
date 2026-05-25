//! ASIMD "misc"-form single-operand ops (Vd, Vn).
//!
//! Coverage so far:
//!   - NEG Vd.<T>, Vn.<T>: per-lane two's-complement negation
//!   - ABS Vd.<T>, Vn.<T>: per-lane absolute value (8/16/32-bit lanes only;
//!     64-bit lane ABS needs SSE emulation, deferred)
//!   - NOT Vd.<T>, Vn.<T> (aka MVN): bitwise complement

use disarm64::decoder::ASIMDMISC;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind { Neg, Abs, Not, FNeg, FAbs, FSqrt, Xtn, Rev16, Rev32, Rev64 }

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDMISC) -> Result<InstStatus> {
    use ASIMDMISC::*;
    let (raw, kind) = match insn {
        NEG_Vd_Vn(i)   => (i.0, Kind::Neg),
        ABS_Vd_Vn(i)   => (i.0, Kind::Abs),
        NOT_Vd_Vn(i)   => (i.0, Kind::Not),
        FNEG_Vd_Vn(i)  => (i.0, Kind::FNeg),
        FABS_Vd_Vn(i)  => (i.0, Kind::FAbs),
        FSQRT_Vd_Vn(i) => (i.0, Kind::FSqrt),
        XTN_Vd_Vn(i)   => (i.0, Kind::Xtn),
        REV16_Vd_Vn(i) => (i.0, Kind::Rev16),
        REV32_Vd_Vn(i) => (i.0, Kind::Rev32),
        REV64_Vd_Vn(i) => (i.0, Kind::Rev64),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let q    = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2); // 00=B, 01=H, 10=S, 11=D
    let rn   = bits(raw, 5, 5) as u8;
    let rd   = bits(raw, 0, 5) as u8;

    let vn = em.get_v_q(rn);
    let result = match kind {
        Kind::Neg => em.vec_neg(vn, size, q),
        Kind::Abs => {
            if size == 3 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            em.vec_abs(vn, size, q)
        }
        Kind::Not => em.vec_not(vn, q),
        Kind::FNeg | Kind::FAbs | Kind::FSqrt => {
            // FP misc ops use bit 22 as the sz flag (0 = single, 1 = double).
            let double = bit(raw, 22) == 1;
            match kind {
                Kind::FNeg  => em.vec_fneg(vn, double, q),
                Kind::FAbs  => em.vec_fabs(vn, double, q),
                Kind::FSqrt => em.vec_fsqrt(vn, double, q),
                _ => unreachable!(),
            }
        }
        Kind::Xtn => {
            // XTN narrows by one: B->… wait XTN takes the wider lane source
            // (size bits indicate the DESTINATION lane width). For
            // XTN.<Td>, Vn.<Ts>, source lane Ts = Td * 2 in bits.
            // `size` field at 22 says dst lane: 00=B, 01=H, 10=S; we pass
            // src_lane_log2 = size + 1 to the IR helper.
            if size > 2 {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            // Q=1 (XTN2) preserves the low half of Vd — we don't support
            // that yet; only the XTN (Q=0) form, which zeroes the upper 64.
            if q {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            em.vec_xtn(vn, size + 1)
        }
        Kind::Rev16 | Kind::Rev32 | Kind::Rev64 => {
            // ARM encodes the byte-level reversal granularity in `size`. The
            // outer container is implied by the mnemonic. Reject invalid
            // (mnemonic, size) combinations: REV16 only takes B, REV32 takes
            // B or H, REV64 takes B/H/S.
            let max_src = match kind {
                Kind::Rev16 => 1,
                Kind::Rev32 => 2,
                Kind::Rev64 => 3,
                _ => unreachable!(),
            };
            if size >= max_src {
                return Err(Error::Decode { pc: em.current_pc, opcode: raw });
            }
            let container_log2 = match kind {
                Kind::Rev16 => 1,
                Kind::Rev32 => 2,
                Kind::Rev64 => 3,
                _ => unreachable!(),
            };
            em.vec_rev(vn, size, container_log2, q)
        }
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
