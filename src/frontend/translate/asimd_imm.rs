//! Advanced SIMD modified-immediate ops:
//!   MOVI / MVNI / BIC#imm / ORR#imm  (integer immediates)
//!   FMOV Vd, #fimm                    (FP immediate)
//!
//! Encoding (vector form):
//!     0 Q op 0 1111 0 0 0 0 0 a b c cmode o2 1 d e f g h Rd
//!     ^^   ^^                            ^^   ^^^^^^^^^
//!     bit 30=Q, bit 29=op (0=MOVI/ORR/MOV/FMOV, 1=MVNI/BIC)
//!     abc = bits 18..16, cmode = bits 15..12, o2 = bit 11
//!     defgh = bits 9..5, Rd = bits 4..0
//!     imm8 = abc:defgh
//!
//! cmode/op together select one of ~16 expansion patterns (yuzu's
//! `AdvSIMDExpandImm`). Strategy: decode imm8 + cmode + op to a 64-bit
//! "per-half" value, then either replicate to 128 bits for Q=1 or write
//! only the low 64 bits for Q=0. For BIC#imm and ORR#imm we additionally
//! combine with the prior Vd contents.

use disarm64::decoder::ASIMDIMM;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action { Movi, Mvni, BicImm, OrrImm, FmovS, FmovD, FmovH }

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDIMM) -> Result<InstStatus> {
    use ASIMDIMM::*;
    let (raw, action) = match insn {
        // MOVI: every flavour ends up in the same 64-bit "expand imm" pipeline
        // because the cmode field already tells AdvSIMDExpandImm which shape
        // (LSL, MSL, replicate-byte, replicate-half) to apply.
        MOVI_Vd_SIMD_IMM(i)               => (i.0, Action::Movi),
        MOVI_Vd_SIMD_IMM_SFT(i)           => (i.0, Action::Movi),
        MOVI_Vd_V_8B_SIMD_IMM_SFT_LSL(i)  => (i.0, Action::Movi),
        MOVI_Vd_V_4H_SIMD_IMM_SFT_LSL(i)  => (i.0, Action::Movi),
        MOVI_Vd_V_2S_SIMD_IMM_SFT_MSL(i)  => (i.0, Action::Movi),

        // MVNI = bitwise NOT of the expanded immediate, same expansion rules.
        MVNI_Vd_SIMD_IMM_SFT(i)           => (i.0, Action::Mvni),
        MVNI_Vd_V_4H_SIMD_IMM_SFT_LSL(i)  => (i.0, Action::Mvni),
        MVNI_Vd_V_2S_SIMD_IMM_SFT_MSL(i)  => (i.0, Action::Mvni),

        BIC_Vd_SIMD_IMM_SFT(i)            => (i.0, Action::BicImm),
        BIC_Vd_V_4H_SIMD_IMM_SFT_LSL(i)   => (i.0, Action::BicImm),

        ORR_Vd_SIMD_IMM_SFT(i)            => (i.0, Action::OrrImm),
        ORR_Vd_V_4H_SIMD_IMM_SFT_LSL(i)   => (i.0, Action::OrrImm),

        FMOV_Vd_SIMD_FPIMM(i)             => (i.0, Action::FmovS),
        FMOV_Vd_V_2D_SIMD_FPIMM(i)        => (i.0, Action::FmovD),
        FMOV_Vd_V_4H_SIMD_FPIMM(i)        => (i.0, Action::FmovH),

        // The lone scalar form: MOVI Dd, #imm64 (cmode=1110, op=1).
        // disarm64 puts that here; we handle it the same way as the Q=0
        // vector form below.
        MOVI_Sd_SIMD_IMM(i)               => (i.0, Action::Movi),
    };

    let q     = bit(raw, 30) == 1;
    let op    = bit(raw, 29);
    let abc   = bits(raw, 16, 3);
    let cmode = bits(raw, 12, 4);
    let defgh = bits(raw, 5, 5);
    let rd    = bits(raw, 0, 5) as u8;
    let imm8  = (abc << 5) | defgh;

    let lane64 = match action {
        Action::Movi   => adv_simd_expand_imm(imm8, cmode, op)?,
        Action::Mvni   => !adv_simd_expand_imm(imm8, cmode, op)?,
        Action::BicImm | Action::OrrImm => adv_simd_expand_imm(imm8, cmode, op)?,

        // FP immediate forms encode an 8-bit imm directly into a single-
        // precision float; the same bits get widened to double in the D form.
        Action::FmovS => vfp_expand_imm32(imm8) as u64 | ((vfp_expand_imm32(imm8) as u64) << 32),
        Action::FmovD => vfp_expand_imm64(imm8),
        Action::FmovH => {
            // FP16 FMOV vector immediate — needs an FP16 widen path that the
            // current IR doesn't have. Surface clearly so we know it's wanted.
            return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
        }
    };

    let lo = em.const_u64(lane64);
    let hi = em.const_u64(if q { lane64 } else { 0 });
    let imm_q = em.vec_build_q(lo, hi);

    let result = match action {
        Action::Movi | Action::Mvni | Action::FmovS | Action::FmovD => imm_q,
        Action::OrrImm => {
            let vd_prev = em.get_v_q(rd);
            em.vec_orr(vd_prev, imm_q, q)
        }
        Action::BicImm => {
            // BIC #imm = Vd AND NOT imm. The IR's vec_bic is `vn & ~vm`, so
            // we pass (Vd_prev, imm_q) and get the right semantics.
            let vd_prev = em.get_v_q(rd);
            em.vec_bic(vd_prev, imm_q, q)
        }
        Action::FmovH => unreachable!(),
    };

    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

/// ARM ARM `AdvSIMDExpandImm(op, cmode, imm8)`.
///
/// Returns the 64-bit value to replicate into each half of the V register.
/// Mirrors yuzu's translate_arithmetic.cpp implementation; see Arm ARM C7.2.
fn adv_simd_expand_imm(imm8: u32, cmode: u32, op: u32) -> Result<u64> {
    let imm8 = imm8 as u64;

    // cmode bits 3..1 (high three bits) select the major family; bit 0
    // varies by family. We follow the Arm ARM case structure exactly.
    let result: u64 = match cmode >> 1 {
        // cmode = 0xx0: replicate imm8 into each 32-bit lane (LSL #(8*cmode_hi)).
        0b000 => replicate_lane(imm8, 32),
        0b001 => replicate_lane(imm8 << 8, 32),
        0b010 => replicate_lane(imm8 << 16, 32),
        0b011 => replicate_lane(imm8 << 24, 32),

        // cmode = 10xx: replicate imm8 into each 16-bit lane (LSL #(8*cmode_lo)).
        0b100 => replicate_lane(imm8, 16),
        0b101 => replicate_lane(imm8 << 8, 16),

        // cmode = 110x: shifted MSL form (imm8:ones), 32-bit replicate.
        0b110 => {
            if cmode & 1 == 0 {
                replicate_lane((imm8 << 8)  | 0xFF,    32)
            } else {
                replicate_lane((imm8 << 16) | 0xFFFF,  32)
            }
        }

        // cmode = 111x: byte/FP/64-bit special encodings.
        0b111 => match (cmode & 1, op) {
            // cmode=1110, op=0: replicate imm8 into every byte.
            (0, 0) => replicate_lane(imm8, 8),

            // cmode=1110, op=1: 64-bit imm built by setting each byte to 0xFF
            // if the corresponding bit of imm8 is 1, else 0x00. Used by
            // MOVI Dd, #imm and MOVI Vd.2D, #imm.
            (0, 1) => {
                let mut v: u64 = 0;
                for i in 0..8 {
                    if (imm8 >> i) & 1 != 0 {
                        v |= 0xFFu64 << (i * 8);
                    }
                }
                v
            }

            // cmode=1111, op=0: FMOV-from-imm8 (single precision), replicate.
            (1, 0) => {
                let f = vfp_expand_imm32(imm8 as u32) as u64;
                f | (f << 32)
            }

            // cmode=1111, op=1: FMOV.2D imm form — same 8-bit-to-double
            // encoding, broadcast as one 64-bit lane.
            (1, 1) => vfp_expand_imm64(imm8 as u32),

            _ => unreachable!("cmode 111x op outside 0/1"),
        },
        _ => unreachable!("cmode >> 1 outside 0..7"),
    };
    Ok(result)
}

/// Replicate `value` (low `lane_bits`) across a 64-bit lane.
fn replicate_lane(value: u64, lane_bits: u32) -> u64 {
    let mask = if lane_bits >= 64 { !0 } else { (1u64 << lane_bits) - 1 };
    let v = value & mask;
    let mut out: u64 = 0;
    let mut shift = 0;
    while shift < 64 {
        out |= v << shift;
        shift += lane_bits;
    }
    out
}

/// `VFPExpandImm(imm8, 32)` — decode the 8-bit FMOV-imm encoding into a
/// single-precision float bit pattern.
///
/// Layout: imm8 = abcdefgh.
/// result = sign:exp:mantissa where
///     sign     = a                          (1 bit)
///     exp[7]   = !b                         (top bit inverted)
///     exp[6..2]= bbbbb                      (replicate b)
///     exp[1..0]= cd                         (2 bits)
///     mantissa = efgh:000_0000_0000_0000_0000 (4 bits left-padded)
fn vfp_expand_imm32(imm8: u32) -> u32 {
    let a = (imm8 >> 7) & 1;
    let b = (imm8 >> 6) & 1;
    let cd = (imm8 >> 4) & 0b11;
    let efgh = imm8 & 0b1111;
    let exp_top = if b == 0 { 1 } else { 0 };
    let exp = (exp_top << 7) | (if b == 0 { 0 } else { 0b11111 } << 2) | cd;
    (a << 31) | (exp << 23) | (efgh << 19)
}

/// Same as above but expands to 64-bit (double precision) layout.
fn vfp_expand_imm64(imm8: u32) -> u64 {
    let a = (imm8 >> 7) & 1;
    let b = (imm8 >> 6) & 1;
    let cd = (imm8 >> 4) & 0b11;
    let efgh = imm8 & 0b1111;
    let exp_top = if b == 0 { 1 } else { 0 };
    let exp = (exp_top << 10) | (if b == 0 { 0 } else { 0b11111111 } << 2) | cd;
    ((a as u64) << 63) | ((exp as u64) << 52) | ((efgh as u64) << 48)
}
