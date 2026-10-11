use disarm64::decoder::ASIMDSAME;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind {
    Add,
    Sub,
    Ushl,
    Sshl,
    Mul,
    Mla,
    Mls,
    And,
    Orr,
    Eor,
    Bic,
    Orn,
    CmEq,
    CmGt,
    CmGe,
    CmHi,
    CmHs,
    Bit,
    Bif,
    Bsl,
    Smin,
    Smax,
    Umin,
    Umax,
    Addp,
    Smaxp,
    Sminp,
    Umaxp,
    Uminp,
    FAdd,
    FSub,
    FMul,
    FDiv,
    FMax,
    FMin,
    FCmEq,
    FCmGt,
    FCmGe,
    FMla,
    FMls,
    FAddp,
    FMaxp,
    FMinp,
}

fn translate_minmax_number(em: &mut IrEmitter<'_>, raw: u32, minimum: bool, pairwise: bool) -> Result<InstStatus> {
    let double = bit(raw, 22) != 0;
    let q = bit(raw, 30) != 0;
    if double && !q {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let mut n = em.get_v_q(rn);
    let mut m = em.get_v_q(rm);
    if pairwise {
        let original_n = n;
        let original_m = m;
        let size = if double { 3 } else { 2 };
        n = em.vec_uzp1(original_n, original_m, size, q);
        m = em.vec_uzp2(original_n, original_m, size, q);
    }
    let zero = em.const_u64(0);
    let mut result = em.vec_build_q(zero, zero);
    for lane in 0..if double || !q { 2 } else { 4 } {
        let extract = |em: &mut IrEmitter<'_>, source| {
            if double {
                if lane == 0 { em.vec_extract_lo64(source) } else { em.vec_extract_hi64(source) }
            } else { em.vec_extract_u32(source, lane) }
        };
        let a = extract(em, n);
        let b = extract(em, m);
        let value = em.push(Armlet::new(
            if double { Op::FpMinMaxNum64 } else { Op::FpMinMaxNum32 },
            if double { Ty::U64 } else { Ty::U32 },
        ).with_args(&[a, b]).with_imm(u64::from(minimum)));
        result = em.vec_ins_gpr(result, value, if double { 3 } else { 2 }, lane, q);
    }
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

pub(super) fn translate_saturating_binary(
    em: &mut IrEmitter<'_>, raw: u32, scalar: bool, unsigned: bool, subtract: bool,
) -> Result<InstStatus> {
    let size = bits(raw, 22, 2);
    let q = bit(raw, 30) != 0;
    if !scalar && size == 3 && !q {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let n = em.get_v_q(bits(raw, 5, 5) as u8);
    let m = em.get_v_q(bits(raw, 16, 5) as u8);
    let values = [em.vec_extract_lo64(n), em.vec_extract_hi64(n),
                  em.vec_extract_lo64(m), em.vec_extract_hi64(m)];
    let mode = u64::from(size) | (u64::from(unsigned) << 2) | (u64::from(subtract) << 3)
        | (u64::from(q) << 4) | (u64::from(scalar) << 5);
    let result = em.push(Armlet::new(Op::SaturatingBinary, Ty::U128)
        .with_args(&values).with_imm(mode));
    em.set_v_q(bits(raw, 0, 5) as u8, result);
    Ok(InstStatus::Continue)
}

pub(super) fn translate_absolute_compare(
    em: &mut IrEmitter<'_>, raw: u32, scalar: bool, greater: bool,
) -> Result<InstStatus> {
    let double = bit(raw, 22) != 0;
    let q = bit(raw, 30) != 0;
    if !scalar && double && !q {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let n = em.get_v_q(bits(raw, 5, 5) as u8);
    let m = em.get_v_q(bits(raw, 16, 5) as u8);
    let zero = em.const_u64(0);
    let mut result = em.vec_build_q(zero, zero);
    for lane in 0..if scalar { 1 } else if double || !q { 2 } else { 4 } {
        let extract = |em: &mut IrEmitter<'_>, value| {
            if double {
                if lane == 0 { em.vec_extract_lo64(value) } else { em.vec_extract_hi64(value) }
            } else { em.vec_extract_u32(value, lane) }
        };
        let a = extract(em, n);
        let b = extract(em, m);
        let value = em.push(Armlet::new(
            if double { Op::FpAbsoluteCompare64 } else { Op::FpAbsoluteCompare32 },
            if double { Ty::U64 } else { Ty::U32 },
        ).with_args(&[a, b]).with_imm(u64::from(greater) | (u64::from(double) << 1)));
        result = em.vec_ins_gpr(result, value, if double { 3 } else { 2 }, lane, true);
    }
    em.set_v_q(bits(raw, 0, 5) as u8, result);
    Ok(InstStatus::Continue)
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDSAME) -> Result<InstStatus> {
    use ASIMDSAME::*;
    let (raw, kind) = match insn {
        SQADD_Vd_Vn_Vm(i) => return translate_saturating_binary(em, i.0, false, false, false),
        UQADD_Vd_Vn_Vm(i) => return translate_saturating_binary(em, i.0, false, true, false),
        SQSUB_Vd_Vn_Vm(i) => return translate_saturating_binary(em, i.0, false, false, true),
        UQSUB_Vd_Vn_Vm(i) => return translate_saturating_binary(em, i.0, false, true, true),
        FACGE_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return translate_absolute_compare(em, i.0, false, false),
        FACGT_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return translate_absolute_compare(em, i.0, false, true),
        FMINNM_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return translate_minmax_number(em, i.0, true, false),
        FMAXNM_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return translate_minmax_number(em, i.0, false, false),
        CMTST_Vd_Vn_Vm(i) => return translate_cmtst(em, i.0, false),
        FRSQRTS_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return super::asimd_recip::translate(em, i.0, false, true),
        FRECPS_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return super::asimd_recip::translate_reciprocal(em, i.0, false, true),
        ADD_Vd_Vn_Vm(i) => (i.0, Kind::Add),
        SUB_Vd_Vn_Vm(i) => (i.0, Kind::Sub),
        USHL_Vd_Vn_Vm(i) => (i.0, Kind::Ushl),
        SSHL_Vd_Vn_Vm(i) => (i.0, Kind::Sshl),
        MUL_Vd_Vn_Vm(i) => (i.0, Kind::Mul),
        MLA_Vd_Vn_Vm(i) => (i.0, Kind::Mla),
        MLS_Vd_Vn_Vm(i) => (i.0, Kind::Mls),
        AND_Vd_Vn_Vm(i) => (i.0, Kind::And),
        ORR_Vd_Vn_Vm(i) => (i.0, Kind::Orr),
        EOR_Vd_Vn_Vm(i) => (i.0, Kind::Eor),
        BIC_Vd_Vn_Vm(i) => (i.0, Kind::Bic),
        ORN_Vd_Vn_Vm(i) => (i.0, Kind::Orn),
        CMEQ_Vd_Vn_Vm(i) => (i.0, Kind::CmEq),
        CMGT_Vd_Vn_Vm(i) => (i.0, Kind::CmGt),
        CMGE_Vd_Vn_Vm(i) => (i.0, Kind::CmGe),
        CMHI_Vd_Vn_Vm(i) => (i.0, Kind::CmHi),
        CMHS_Vd_Vn_Vm(i) => (i.0, Kind::CmHs),
        BIT_Vd_Vn_Vm(i) => (i.0, Kind::Bit),
        BIF_Vd_Vn_Vm(i) => (i.0, Kind::Bif),
        BSL_Vd_Vn_Vm(i) => (i.0, Kind::Bsl),
        SMIN_Vd_Vn_Vm(i) => (i.0, Kind::Smin),
        SMAX_Vd_Vn_Vm(i) => (i.0, Kind::Smax),
        UMIN_Vd_Vn_Vm(i) => (i.0, Kind::Umin),
        UMAX_Vd_Vn_Vm(i) => (i.0, Kind::Umax),
        ADDP_Vd_Vn_Vm(i) => (i.0, Kind::Addp),
        SMAXP_Vd_Vn_Vm(i) => (i.0, Kind::Smaxp),
        SMINP_Vd_Vn_Vm(i) => (i.0, Kind::Sminp),
        UMAXP_Vd_Vn_Vm(i) => (i.0, Kind::Umaxp),
        UMINP_Vd_Vn_Vm(i) => (i.0, Kind::Uminp),
        FADDP_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FAddp),
        FMAXP_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMaxp),
        FMINP_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMinp),
        FMAXNMP_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return translate_minmax_number(em, i.0, false, true),
        FMINNMP_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return translate_minmax_number(em, i.0, true, true),
        FADD_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FAdd),
        FSUB_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FSub),
        FABD_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => return super::asimd_recip::translate_absolute_difference(em, i.0, false),
        FMUL_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMul),
        FDIV_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FDiv),
        FMAX_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMax),
        FMIN_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMin),
        FCMEQ_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FCmEq),
        FCMGT_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FCmGt),
        FCMGE_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FCmGe),
        FMLA_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMla),
        FMLS_Vd_V_2S_Vn_V_2S_Vm_V_2S(i) => (i.0, Kind::FMls),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let q = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let vn = em.get_v_q(rn);
    let vm = em.get_v_q(rm);

    let result = match kind {
        Kind::Add => em.vec_add(vn, vm, size, q),
        Kind::Sub => em.vec_sub(vn, vm, size, q),
        Kind::Ushl | Kind::Sshl => {
            if size == 3 && !q {
                return Err(Error::Decode {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
            em.vec_variable_shift(vn, vm, size, q, matches!(kind, Kind::Sshl))
        }
        Kind::Mul => {
            if size == 3 {
                return Err(Error::Unsupported {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
            em.vec_mul(vn, vm, size, q)
        }
        Kind::Mla | Kind::Mls => {
            if size == 3 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
            // Capture the old accumulator as well as both original full
            // sources before any destination write, including all aliases.
            let accumulator = em.get_v_q(rd);
            let product = em.vec_mul(vn, vm, size, q);
            if matches!(kind, Kind::Mls) {
                em.vec_sub(accumulator, product, size, q)
            } else {
                em.vec_add(accumulator, product, size, q)
            }
        }
        Kind::And => em.vec_and(vn, vm, q),
        Kind::Orr => em.vec_orr(vn, vm, q),
        Kind::Eor => em.vec_eor(vn, vm, q),
        Kind::Bic => em.vec_bic(vn, vm, q),
        Kind::Orn => em.vec_orn(vn, vm, q),
        Kind::CmEq => em.vec_cmeq(vn, vm, size, q),
        Kind::CmGt => em.vec_cmgt(vn, vm, size, q),
        Kind::CmGe => em.vec_cmge(vn, vm, size, q),
        Kind::CmHi => em.vec_cmhi(vn, vm, size, q),
        Kind::CmHs => em.vec_cmhs(vn, vm, size, q),
        Kind::Bit | Kind::Bif | Kind::Bsl => {
            let vd_prev = em.get_v_q(rd);
            match kind {
                Kind::Bit => em.vec_bit(vd_prev, vn, vm, q),
                Kind::Bif => em.vec_bif(vd_prev, vn, vm, q),
                Kind::Bsl => em.vec_bsl(vd_prev, vn, vm, q),
                _ => unreachable!(),
            }
        }
        Kind::Smin | Kind::Smax | Kind::Umin | Kind::Umax => match kind {
            Kind::Smin => em.vec_smin(vn, vm, size, q),
            Kind::Smax => em.vec_smax(vn, vm, size, q),
            Kind::Umin => em.vec_umin(vn, vm, size, q),
            Kind::Umax => em.vec_umax(vn, vm, size, q),
            _ => unreachable!(),
        },
        Kind::Addp | Kind::Smaxp | Kind::Sminp | Kind::Umaxp | Kind::Uminp => {
            let u1 = em.vec_uzp1(vn, vm, size, q);
            let u2 = em.vec_uzp2(vn, vm, size, q);
            match kind {
                Kind::Addp => em.vec_add(u1, u2, size, q),
                Kind::Smaxp => em.vec_smax(u1, u2, size, q),
                Kind::Sminp => em.vec_smin(u1, u2, size, q),
                Kind::Umaxp => em.vec_umax(u1, u2, size, q),
                Kind::Uminp => em.vec_umin(u1, u2, size, q),
                _ => unreachable!(),
            }
        }
        Kind::FAddp | Kind::FMaxp | Kind::FMinp => {
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
            let lane = if double { 3 } else { 2 };
            let u1 = em.vec_uzp1(vn, vm, lane, q);
            let u2 = em.vec_uzp2(vn, vm, lane, q);
            match kind {
                Kind::FAddp => em.vec_fadd(u1, u2, double, q),
                Kind::FMaxp => em.vec_fmax(u1, u2, double, q),
                Kind::FMinp => em.vec_fmin(u1, u2, double, q),
                _ => unreachable!(),
            }
        }
        Kind::FAdd
        | Kind::FSub
        | Kind::FMul
        | Kind::FDiv
        | Kind::FMax
        | Kind::FMin
        | Kind::FCmEq
        | Kind::FCmGt
        | Kind::FCmGe => {
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
            match kind {
                Kind::FAdd => em.vec_fadd(vn, vm, double, q),
                Kind::FSub => em.vec_fsub(vn, vm, double, q),
                Kind::FMul => em.vec_fmul(vn, vm, double, q),
                Kind::FDiv => em.vec_fdiv(vn, vm, double, q),
                Kind::FMax => em.vec_fmax(vn, vm, double, q),
                Kind::FMin => em.vec_fmin(vn, vm, double, q),
                Kind::FCmEq => em.vec_fcmeq(vn, vm, double, q),
                Kind::FCmGt => em.vec_fcmgt(vn, vm, double, q),
                Kind::FCmGe => em.vec_fcmge(vn, vm, double, q),
                _ => unreachable!(),
            }
        }
        Kind::FMla | Kind::FMls => {
            let double = bit(raw, 22) == 1;
            if double && !q {
                return Err(Error::Decode {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
            let vd_prev = em.get_v_q(rd);
            match kind {
                Kind::FMla => em.vec_fmla(vd_prev, vn, vm, double, q),
                Kind::FMls => em.vec_fmls(vd_prev, vn, vm, double, q),
                _ => unreachable!(),
            }
        }
    };

    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

// Arm CMTST: each lane is all ones exactly when the original operand AND
// is nonzero. Integer comparisons do not change NZCV or floating status.
pub(super) fn translate_cmtst(em: &mut IrEmitter<'_>, raw: u32, scalar: bool) -> Result<InstStatus> {
    let size = bits(raw, 22, 2);
    let q = !scalar && bit(raw, 30) != 0;
    if (scalar && size != 3) || (!scalar && size == 3 && !q) {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let n = em.get_v_q(rn);
    let m = em.get_v_q(rm);
    let masked = em.vec_and(n, m, q);
    let z = em.const_u64(0);
    let zero = em.vec_build_q(z, z);
    let equal_zero = em.vec_cmeq(masked, zero, size, q);
    let result = em.vec_not(equal_zero, q);
    // Q0 and scalar D both write only the low64 bits and clear the upper64.
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
