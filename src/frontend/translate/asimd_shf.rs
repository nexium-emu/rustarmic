use disarm64::decoder::{ASIMDSHF, ASISDSHF};

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

#[derive(Clone, Copy)]
enum Kind {
    Shl,
    Ushr,
    Sshr,
    Shrn,
    Shrn2,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDSHF) -> Result<InstStatus> {
    use ASIMDSHF::*;
    let (raw, kind) = match insn {
        SLI_Vd_Vn_IMM_VLSL(i) => return translate_insert(em, i.0, false, true),
        SRI_Vd_Vn_IMM_VLSR(i) => return translate_insert(em, i.0, false, false),
        SSHLL_Vd_Vn_IMM_VLSL(i) => return translate_widen_shift(em, i.0, true),
        SSHLL2_Vd_Vn_IMM_VLSL(i) => return translate_widen_shift(em, i.0, true),
        USHLL_Vd_Vn_IMM_VLSL(i) => return translate_widen_shift(em, i.0, false),
        USHLL2_Vd_Vn_IMM_VLSL(i) => return translate_widen_shift(em, i.0, false),
        SSRA_Vd_Vn_IMM_VLSR(i) => return translate_accumulate(em, i.0, false, true),
        USRA_Vd_Vn_IMM_VLSR(i) => return translate_accumulate(em, i.0, false, false),
        SRSHR_Vd_Vn_IMM_VLSR(i) => return translate_right(em, i.0, false, true, true, false),
        URSHR_Vd_Vn_IMM_VLSR(i) => return translate_right(em, i.0, false, false, true, false),
        SRSRA_Vd_Vn_IMM_VLSR(i) => return translate_right(em, i.0, false, true, true, true),
        URSRA_Vd_Vn_IMM_VLSR(i) => return translate_right(em, i.0, false, false, true, true),
        FCVTZS_Vd_Vn_IMM_VLSR(i) => {
            return super::asimd_convert::translate_fixed_float_to_int(em, i.0, false, false);
        }
        FCVTZS_Vd_V_2S_Vn_V_2S_IMM_VLSR_V_2S(i) => {
            return super::asimd_convert::translate_fixed_float_to_int(em, i.0, false, false);
        }
        FCVTZU_Vd_Vn_IMM_VLSR(i) => {
            return super::asimd_convert::translate_fixed_float_to_int(em, i.0, false, true);
        }
        FCVTZU_Vd_V_2S_Vn_V_2S_IMM_VLSR_V_2S(i) => {
            return super::asimd_convert::translate_fixed_float_to_int(em, i.0, false, true);
        }
        SHL_Vd_Vn_IMM_VLSL(i) => (i.0, Kind::Shl),
        USHR_Vd_Vn_IMM_VLSR(i) => (i.0, Kind::Ushr),
        SSHR_Vd_Vn_IMM_VLSR(i) => (i.0, Kind::Sshr),
        SHRN_Vd_Vn_IMM_VLSR(i) => (i.0, Kind::Shrn),
        SHRN2_Vd_Vn_IMM_VLSR(i) => (i.0, Kind::Shrn2),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let q = bit(raw, 30) == 1;
    let immh = bits(raw, 19, 4);
    let immb = bits(raw, 16, 3);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    if immh == 0 {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let lane_log2 = if (immh & 0b1000) != 0 {
        3
    } else if (immh & 0b0100) != 0 {
        2
    } else if (immh & 0b0010) != 0 {
        1
    } else {
        0
    };
    let lane_bits = 8u32 << lane_log2;
    let immhb = (immh << 3) | immb;
    let shift = match kind {
        Kind::Shl => immhb.wrapping_sub(lane_bits),
        Kind::Ushr | Kind::Sshr | Kind::Shrn | Kind::Shrn2 => (2 * lane_bits).wrapping_sub(immhb),
    };

    let vn = em.get_v_q(rn);
    let result = match kind {
        Kind::Shl => em.vec_shl_imm(vn, lane_log2, shift, q),
        Kind::Ushr => em.vec_ushr_imm(vn, lane_log2, shift, q),
        Kind::Sshr => em.vec_sshr_imm(vn, lane_log2, shift, q),
        Kind::Shrn | Kind::Shrn2 => {
            let src_log2 = lane_log2 + 1;
            let shifted = em.vec_ushr_imm(vn, src_log2, shift, true);
            if matches!(kind, Kind::Shrn2) {
                let vd_prev = em.get_v_q(rd);
                em.vec_xtn2(vd_prev, shifted, src_log2)
            } else {
                em.vec_xtn(shifted, src_log2)
            }
        }
    };
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

pub fn translate_scalar(em: &mut IrEmitter<'_>, insn: ASISDSHF) -> Result<InstStatus> {
    match insn {
        ASISDSHF::SLI_Sd_Sn_IMM_VLSL(i) => translate_insert(em, i.0, true, true),
        ASISDSHF::SRI_Sd_Sn_IMM_VLSR(i) => translate_insert(em, i.0, true, false),
        ASISDSHF::SSRA_Sd_Sn_IMM_VLSR(i) => translate_accumulate(em, i.0, true, true),
        ASISDSHF::USRA_Sd_Sn_IMM_VLSR(i) => translate_accumulate(em, i.0, true, false),
        ASISDSHF::SRSHR_Sd_Sn_IMM_VLSR(i) => translate_right(em, i.0, true, true, true, false),
        ASISDSHF::URSHR_Sd_Sn_IMM_VLSR(i) => translate_right(em, i.0, true, false, true, false),
        ASISDSHF::SRSRA_Sd_Sn_IMM_VLSR(i) => translate_right(em, i.0, true, true, true, true),
        ASISDSHF::URSRA_Sd_Sn_IMM_VLSR(i) => translate_right(em, i.0, true, false, true, true),
        ASISDSHF::SSHR_Sd_Sn_IMM_VLSR(i) => translate_right(em, i.0, true, true, false, false),
        ASISDSHF::USHR_Sd_Sn_IMM_VLSR(i) => translate_right(em, i.0, true, false, false, false),
        ASISDSHF::SHL_Sd_Sn_IMM_VLSL(i) => {
            if bits(i.0, 19, 4) & 8 == 0 {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: i.0 });
            }
            let source = em.get_v_d(bits(i.0, 5, 5) as u8);
            let amount = em.const_u64((bits(i.0, 16, 7) - 64) as u64);
            let result = em.lsl(source, amount, RegSize::X);
            em.set_v_d(bits(i.0, 0, 5) as u8, result);
            Ok(InstStatus::Continue)
        }
        ASISDSHF::FCVTZS_Sd_Sn_IMM_VLSR(i) => {
            super::asimd_convert::translate_fixed_float_to_int(em, i.0, true, false)
        }
        ASISDSHF::FCVTZS_Sd_S_S_Sn_S_S_IMM_VLSR_S_S(i) => {
            super::asimd_convert::translate_fixed_float_to_int(em, i.0, true, false)
        }
        ASISDSHF::FCVTZU_Sd_Sn_IMM_VLSR(i) => {
            super::asimd_convert::translate_fixed_float_to_int(em, i.0, true, true)
        }
        ASISDSHF::FCVTZU_Sd_S_S_Sn_S_S_IMM_VLSR_S_S(i) => {
            super::asimd_convert::translate_fixed_float_to_int(em, i.0, true, true)
        }
        _ => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: 0,
        }),
    }
}

fn translate_insert(
    em: &mut IrEmitter<'_>, raw: u32, scalar: bool, left: bool,
) -> Result<InstStatus> {
    let immh = bits(raw, 19, 4);
    let q = bit(raw, 30) != 0;
    if immh == 0 || (scalar && immh & 8 == 0) || (!scalar && immh & 8 != 0 && !q) {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let size = 31 - immh.leading_zeros();
    let width = 8u32 << size;
    let immediate = bits(raw, 16, 7);
    let shift = if left { immediate - width } else { 2 * width - immediate };
    let lane_mask = u64::MAX >> (64 - width);
    let preserved_mask = if left {
        (1u64 << shift) - 1
    } else if shift == 64 {
        u64::MAX
    } else {
        lane_mask ^ (lane_mask >> shift)
    };
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    if scalar {
        let source = em.get_v_d(rn);
        let old = em.get_v_d(rd);
        let shifted = if !left && shift == 64 {
            em.const_u64(0)
        } else {
            let amount = em.const_u64(shift as u64);
            if left { em.lsl(source, amount, RegSize::X) }
            else { em.lsr(source, amount, RegSize::X) }
        };
        let mask = em.const_u64(preserved_mask);
        let retained = em.and(old, mask, RegSize::X);
        let result = em.or(shifted, retained, RegSize::X);
        em.set_v_d(rd, result);
    } else {
        let source = em.get_v_q(rn);
        let old = em.get_v_q(rd);
        let shifted = if left { em.vec_shl_imm(source, size, shift, q) }
            else { em.vec_ushr_imm(source, size, shift, q) };
        let mask = em.const_u64(preserved_mask);
        let mask = em.vec_dup_gpr(mask, size, q);
        let retained = em.vec_and(old, mask, q);
        let result = em.vec_orr(shifted, retained, q);
        em.set_v_q(rd, result);
    }
    Ok(InstStatus::Continue)
}

// Arm SSRA/USRA (ISA111181/2025-09_ASL1): each lane adds a truncated
// arithmetic/logical right shift, with modular element-width accumulation.
fn translate_accumulate(
    em: &mut IrEmitter<'_>, raw: u32, scalar: bool, signed: bool,
) -> Result<InstStatus> {
    translate_right(em, raw, scalar, signed, false, true)
}

// Compute rounding without overflowing the original lane: floor(value/2^s)
// plus the last discarded bit equals floor((value + 2^(s-1))/2^s), including
// signed negatives and s == width. Accumulation is modular, not saturating.
fn translate_right(
    em: &mut IrEmitter<'_>, raw: u32, scalar: bool, signed: bool,
    rounded: bool, accumulate: bool,
) -> Result<InstStatus> {
    let immh = bits(raw, 19, 4);
    let q = bit(raw, 30) != 0;
    if immh == 0 || (scalar && immh & 8 == 0) || (!scalar && immh & 8 != 0 && !q) {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let size = 31 - immh.leading_zeros();
    let width = 8u32 << size;
    let shift = 2 * width - bits(raw, 16, 7);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    if scalar {
        let source = em.get_v_d(rn);
        let accumulator = if accumulate { Some(em.get_v_d(rd)) } else { None };
        let mut shifted = if signed {
            // Scalar host shifts mask a count of64; Arm requires sign fill.
            let amount = em.const_u64(shift.min(63) as u64);
            em.asr(source, amount, RegSize::X)
        } else if shift == 64 {
            em.const_u64(0)
        } else {
            let amount = em.const_u64(shift as u64);
            em.lsr(source, amount, RegSize::X)
        };
        if rounded {
            let amount = em.const_u64((shift - 1) as u64);
            let discarded = em.lsr(source, amount, RegSize::X);
            let one = em.const_u64(1);
            let round_bit = em.and(discarded, one, RegSize::X);
            shifted = em.add(shifted, round_bit, RegSize::X);
        }
        let result = if let Some(accumulator) = accumulator {
            em.add(accumulator, shifted, RegSize::X)
        } else { shifted };
        em.set_v_d(rd, result);
    } else {
        // Both original operands are captured before an aliased write.
        let source = em.get_v_q(rn);
        let accumulator = if accumulate { Some(em.get_v_q(rd)) } else { None };
        let mut shifted = if signed {
            em.vec_sshr_imm(source, size, shift, q)
        } else {
            em.vec_ushr_imm(source, size, shift, q)
        };
        if rounded {
            let discarded = em.vec_ushr_imm(source, size, shift - 1, q);
            let one = em.const_u64(1);
            let ones = em.vec_dup_gpr(one, size, q);
            let round_bit = em.vec_and(discarded, ones, q);
            shifted = em.vec_add(shifted, round_bit, size, q);
        }
        let result = if let Some(accumulator) = accumulator {
            em.vec_add(accumulator, shifted, size, q)
        } else { shifted };
        em.set_v_q(rd, result);
    }
    Ok(InstStatus::Continue)
}

// Arm SSHLL/USHLL (ISA111181/2025-09_ASL1): widen the selected64-bit
// source half, then shift each double-width result by0..source_width-1.
fn translate_widen_shift(em: &mut IrEmitter<'_>, raw: u32, signed: bool) -> Result<InstStatus> {
    let immh = bits(raw, 19, 4);
    if immh == 0 || immh & 8 != 0 {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let size = 31 - immh.leading_zeros();
    let width = 8u32 << size;
    let shift = bits(raw, 16, 7) - width;
    let upper = bit(raw, 30) != 0;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    // An aliased destination must not clear the half being extracted.
    let source = em.get_v_q(rn);
    let z = em.const_u64(0);
    let zero = em.vec_build_q(z, z);
    let widened = if signed {
        em.vec_saddl(source, zero, size, upper)
    } else {
        em.vec_uaddl(source, zero, size, upper)
    };
    let result = em.vec_shl_imm(widened, size + 1, shift, true);
    // The2 suffix selects the SOURCE upper half; both forms replace all128
    // destination bits (unlike narrowing2 instructions' preserved low half).
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
