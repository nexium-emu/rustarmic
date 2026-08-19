use disarm64::decoder::ASIMDIMM;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Action {
    Movi,
    Mvni,
    BicImm,
    OrrImm,
    FmovS,
    FmovD,
    FmovH,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDIMM) -> Result<InstStatus> {
    use ASIMDIMM::*;
    let (raw, action) = match insn {
        MOVI_Vd_SIMD_IMM(i) => (i.0, Action::Movi),
        MOVI_Vd_SIMD_IMM_SFT(i) => (i.0, Action::Movi),
        MOVI_Vd_V_8B_SIMD_IMM_SFT_LSL(i) => (i.0, Action::Movi),
        MOVI_Vd_V_4H_SIMD_IMM_SFT_LSL(i) => (i.0, Action::Movi),
        MOVI_Vd_V_2S_SIMD_IMM_SFT_MSL(i) => (i.0, Action::Movi),

        MVNI_Vd_SIMD_IMM_SFT(i) => (i.0, Action::Mvni),
        MVNI_Vd_V_4H_SIMD_IMM_SFT_LSL(i) => (i.0, Action::Mvni),
        MVNI_Vd_V_2S_SIMD_IMM_SFT_MSL(i) => (i.0, Action::Mvni),

        BIC_Vd_SIMD_IMM_SFT(i) => (i.0, Action::BicImm),
        BIC_Vd_V_4H_SIMD_IMM_SFT_LSL(i) => (i.0, Action::BicImm),

        ORR_Vd_SIMD_IMM_SFT(i) => (i.0, Action::OrrImm),
        ORR_Vd_V_4H_SIMD_IMM_SFT_LSL(i) => (i.0, Action::OrrImm),

        FMOV_Vd_SIMD_FPIMM(i) => (i.0, Action::FmovS),
        FMOV_Vd_V_2D_SIMD_FPIMM(i) => (i.0, Action::FmovD),
        FMOV_Vd_V_4H_SIMD_FPIMM(i) => (i.0, Action::FmovH),

        MOVI_Sd_SIMD_IMM(i) => (i.0, Action::Movi),
    };

    let q = bit(raw, 30) == 1;
    let op = bit(raw, 29);
    let abc = bits(raw, 16, 3);
    let cmode = bits(raw, 12, 4);
    let defgh = bits(raw, 5, 5);
    let rd = bits(raw, 0, 5) as u8;
    let imm8 = (abc << 5) | defgh;

    let lane64 = match action {
        Action::Movi => adv_simd_expand_imm(imm8, cmode, op)?,
        Action::Mvni => !adv_simd_expand_imm(imm8, cmode, op)?,
        Action::BicImm | Action::OrrImm => adv_simd_expand_imm(imm8, cmode, op)?,

        Action::FmovS => vfp_expand_imm32(imm8) as u64 | ((vfp_expand_imm32(imm8) as u64) << 32),
        Action::FmovD => vfp_expand_imm64(imm8),
        Action::FmovH => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            });
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
            let vd_prev = em.get_v_q(rd);
            em.vec_bic(vd_prev, imm_q, q)
        }
        Action::FmovH => unreachable!(),
    };

    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

fn adv_simd_expand_imm(imm8: u32, cmode: u32, op: u32) -> Result<u64> {
    let imm8 = imm8 as u64;

    let result: u64 = match cmode >> 1 {
        0b000 => replicate_lane(imm8, 32),
        0b001 => replicate_lane(imm8 << 8, 32),
        0b010 => replicate_lane(imm8 << 16, 32),
        0b011 => replicate_lane(imm8 << 24, 32),

        0b100 => replicate_lane(imm8, 16),
        0b101 => replicate_lane(imm8 << 8, 16),

        0b110 => {
            if cmode & 1 == 0 {
                replicate_lane((imm8 << 8) | 0xFF, 32)
            } else {
                replicate_lane((imm8 << 16) | 0xFFFF, 32)
            }
        }

        0b111 => match (cmode & 1, op) {
            (0, 0) => replicate_lane(imm8, 8),

            (0, 1) => {
                let mut v: u64 = 0;
                for i in 0..8 {
                    if (imm8 >> i) & 1 != 0 {
                        v |= 0xFFu64 << (i * 8);
                    }
                }
                v
            }

            (1, 0) => {
                let f = vfp_expand_imm32(imm8 as u32) as u64;
                f | (f << 32)
            }

            (1, 1) => vfp_expand_imm64(imm8 as u32),

            _ => unreachable!("cmode 111x op outside 0/1"),
        },
        _ => unreachable!("cmode >> 1 outside 0..7"),
    };
    Ok(result)
}

fn replicate_lane(value: u64, lane_bits: u32) -> u64 {
    let mask = if lane_bits >= 64 {
        !0
    } else {
        (1u64 << lane_bits) - 1
    };
    let v = value & mask;
    let mut out: u64 = 0;
    let mut shift = 0;
    while shift < 64 {
        out |= v << shift;
        shift += lane_bits;
    }
    out
}

fn vfp_expand_imm32(imm8: u32) -> u32 {
    let a = (imm8 >> 7) & 1;
    let b = (imm8 >> 6) & 1;
    let cd = (imm8 >> 4) & 0b11;
    let efgh = imm8 & 0b1111;
    let exp_top = if b == 0 { 1 } else { 0 };
    let exp = (exp_top << 7) | (if b == 0 { 0 } else { 0b11111 } << 2) | cd;
    (a << 31) | (exp << 23) | (efgh << 19)
}

fn vfp_expand_imm64(imm8: u32) -> u64 {
    let a = (imm8 >> 7) & 1;
    let b = (imm8 >> 6) & 1;
    let cd = (imm8 >> 4) & 0b11;
    let efgh = imm8 & 0b1111;
    let exp_top = if b == 0 { 1 } else { 0 };
    let exp = (exp_top << 10) | (if b == 0 { 0 } else { 0b11111111 } << 2) | cd;
    ((a as u64) << 63) | ((exp as u64) << 52) | ((efgh as u64) << 48)
}
