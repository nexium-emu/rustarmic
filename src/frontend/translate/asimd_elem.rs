use disarm64::decoder::ASIMDELEM;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum FpKind {
    FMla,
    FMls,
    FMul,
    FMulx,
}

#[derive(Clone, Copy)]
enum IntKind {
    Mul,
    Mla,
    Mls,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDELEM) -> Result<InstStatus> {
    use ASIMDELEM::*;
    match insn {
        FMLA_Vd_Vn_Em(i) => translate_fp(em, i.0, FpKind::FMla),
        FMLS_Vd_Vn_Em(i) => translate_fp(em, i.0, FpKind::FMls),
        FMUL_Vd_Vn_Em(i) => translate_fp(em, i.0, FpKind::FMul),
        FMULX_Vd_Vn_Em(i) => translate_fp(em, i.0, FpKind::FMulx),

        FMLA_Vd_Vn_Em16(i) => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: i.0,
        }),
        FMLS_Vd_Vn_Em16(i) => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: i.0,
        }),
        FMUL_Vd_Vn_Em16(i) => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: i.0,
        }),
        FMULX_Vd_Vn_Em16(i) => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: i.0,
        }),

        MUL_Vd_Vn_Em16(i) => translate_int(em, i.0, IntKind::Mul),
        MLA_Vd_Vn_Em16(i) => translate_int(em, i.0, IntKind::Mla),
        MLS_Vd_Vn_Em16(i) => translate_int(em, i.0, IntKind::Mls),

        other => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: raw_of(&other),
        }),
    }
}

fn raw_of(insn: &ASIMDELEM) -> u32 {
    use ASIMDELEM::*;
    match insn {
        FCMLA_Vd_Vn_Em_IMM_ROT2(i) => i.0,
        FMLAL2_Vd_V_4S_Vn_V_4H_Em16_S_H(i) => i.0,
        FMLAL2_Vd_Vn_Em16(i) => i.0,
        FMLAL_Vd_V_4S_Vn_V_4H_Em16_S_H(i) => i.0,
        FMLAL_Vd_Vn_Em16(i) => i.0,
        FMLA_Vd_Vn_Em(i) => i.0,
        FMLA_Vd_Vn_Em16(i) => i.0,
        FMLSL2_Vd_V_4S_Vn_V_4H_Em16_S_H(i) => i.0,
        FMLSL2_Vd_Vn_Em16(i) => i.0,
        FMLSL_Vd_V_4S_Vn_V_4H_Em16_S_H(i) => i.0,
        FMLSL_Vd_Vn_Em16(i) => i.0,
        FMLS_Vd_Vn_Em(i) => i.0,
        FMLS_Vd_Vn_Em16(i) => i.0,
        FMULX_Vd_Vn_Em(i) => i.0,
        FMULX_Vd_Vn_Em16(i) => i.0,
        FMUL_Vd_Vn_Em(i) => i.0,
        FMUL_Vd_Vn_Em16(i) => i.0,
        MLA_Vd_Vn_Em16(i) => i.0,
        MLS_Vd_Vn_Em16(i) => i.0,
        MUL_Vd_Vn_Em16(i) => i.0,
        SMLAL2_Vd_Vn_Em16(i) => i.0,
        SMLAL_Vd_Vn_Em16(i) => i.0,
        SMLSL2_Vd_Vn_Em16(i) => i.0,
        SMLSL_Vd_Vn_Em16(i) => i.0,
        SMULL2_Vd_Vn_Em16(i) => i.0,
        SMULL_Vd_Vn_Em16(i) => i.0,
        SQDMLAL2_Vd_Vn_Em16(i) => i.0,
        SQDMLAL_Vd_Vn_Em16(i) => i.0,
        SQDMLSL2_Vd_Vn_Em16(i) => i.0,
        SQDMLSL_Vd_Vn_Em16(i) => i.0,
        SQDMULH_Vd_Vn_Em16(i) => i.0,
        SQDMULL2_Vd_Vn_Em16(i) => i.0,
        SQDMULL_Vd_Vn_Em16(i) => i.0,
        SQRDMLAH_Vd_Vn_Em16(i) => i.0,
        SQRDMLSH_Vd_Vn_Em16(i) => i.0,
        SQRDMULH_Vd_Vn_Em16(i) => i.0,
        UMLAL2_Vd_Vn_Em16(i) => i.0,
        UMLAL_Vd_Vn_Em16(i) => i.0,
        UMLSL2_Vd_Vn_Em16(i) => i.0,
        UMLSL_Vd_Vn_Em16(i) => i.0,
        UMULL2_Vd_Vn_Em16(i) => i.0,
        UMULL_Vd_Vn_Em16(i) => i.0,
    }
}

fn translate_fp(em: &mut IrEmitter<'_>, raw: u32, kind: FpKind) -> Result<InstStatus> {
    let q = bit(raw, 30) == 1;
    let sz = bit(raw, 22);
    let l = bit(raw, 21);
    let m_b = bit(raw, 20);
    let h = bit(raw, 11);
    let rm_low4 = bits(raw, 16, 4);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    if sz == 1 && !q {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }

    let (idx, rm) = if sz == 0 {
        let idx = (h << 1) | l;
        let rm = (m_b << 4) | rm_low4;
        (idx, rm as u8)
    } else {
        let idx = h;
        let rm = (l << 4) | rm_low4;
        let _ = m_b;
        (idx, rm as u8)
    };

    let vm = em.get_v_q(rm);
    let broadcast = if sz == 0 {
        let scalar = em.vec_extract_u32(vm, idx);
        em.vec_dup_gpr(scalar, 2, q)
    } else {
        let scalar = if idx == 0 {
            em.vec_extract_lo64(vm)
        } else {
            em.vec_extract_hi64(vm)
        };
        em.vec_dup_gpr(scalar, 3, q)
    };

    let vn = em.get_v_q(rn);
    let double = sz == 1;
    let result = match kind {
        FpKind::FMla => {
            let vd_prev = em.get_v_q(rd);
            em.vec_fmla(vd_prev, vn, broadcast, double, q)
        }
        FpKind::FMls => {
            let vd_prev = em.get_v_q(rd);
            em.vec_fmls(vd_prev, vn, broadcast, double, q)
        }
        FpKind::FMul => em.vec_fmul(vn, broadcast, double, q),
        FpKind::FMulx => em.vec_fmul(vn, broadcast, double, q),
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

fn translate_int(em: &mut IrEmitter<'_>, raw: u32, kind: IntKind) -> Result<InstStatus> {
    let q = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let l = bit(raw, 21);
    let m_b = bit(raw, 20);
    let h = bit(raw, 11);
    let rm_low4 = bits(raw, 16, 4);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let (lane_log2, idx, rm) = match size {
        0b01 => {
            let idx = (h << 2) | (l << 1) | m_b;
            (1u32, idx, rm_low4 as u8)
        }
        0b10 => {
            let idx = (h << 1) | l;
            let rm = (m_b << 4) | rm_low4;
            (2u32, idx, rm as u8)
        }
        _ => {
            return Err(Error::Decode {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };

    let vm = em.get_v_q(rm);
    let scalar = match lane_log2 {
        1 => em.vec_extract_u16(vm, idx),
        2 => em.vec_extract_u32(vm, idx),
        _ => unreachable!(),
    };
    let broadcast = em.vec_dup_gpr(scalar, lane_log2, q);
    let vn = em.get_v_q(rn);

    let result = match kind {
        IntKind::Mul => em.vec_mul(vn, broadcast, lane_log2, q),
        IntKind::Mla => {
            let prod = em.vec_mul(vn, broadcast, lane_log2, q);
            let vd_prev = em.get_v_q(rd);
            em.vec_add(vd_prev, prod, lane_log2, q)
        }
        IntKind::Mls => {
            let prod = em.vec_mul(vn, broadcast, lane_log2, q);
            let vd_prev = em.get_v_q(rd);
            em.vec_sub(vd_prev, prod, lane_log2, q)
        }
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
