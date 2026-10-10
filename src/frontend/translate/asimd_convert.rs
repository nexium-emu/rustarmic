use disarm64::decoder::ASISDMISC;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty, ValueRef};
use crate::util::bits::{bit, bits};

fn convert(
    em: &mut IrEmitter<'_>,
    value: ValueRef,
    double: bool,
    unsigned: bool,
    float_to_int: bool,
) -> ValueRef {
    let (op, ty) = match (double, unsigned, float_to_int) {
        (false, false, false) => (Op::ScvtfWS, Ty::U32),
        (true, false, false) => (Op::ScvtfXD, Ty::U64),
        (false, true, false) => (Op::UcvtfWS, Ty::U32),
        (true, true, false) => (Op::UcvtfXD, Ty::U64),
        (false, false, true) => (Op::FcvtZsSW, Ty::U32),
        (true, false, true) => (Op::FcvtZsDX, Ty::U64),
        (false, true, true) => (Op::FcvtZuSW, Ty::U32),
        (true, true, true) => (Op::FcvtZuDX, Ty::U64),
    };
    em.push(Armlet::new(op, ty).with_args(&[value]))
}

pub fn translate_scalar(em: &mut IrEmitter<'_>, insn: ASISDMISC) -> Result<InstStatus> {
    let (raw, unsigned, float_to_int) = match insn {
        ASISDMISC::FRSQRTE_Sd_Sn(i) => return super::asimd_recip::translate(em, i.0, true, false),
        ASISDMISC::FRECPE_Sd_Sn(i) => {
            return super::asimd_recip::translate_reciprocal(em, i.0, true, false);
        }
        ASISDMISC::SQXTN_Sd_Sn(i) => return super::asimd_narrow::translate(em, i.0, true, 0),
        ASISDMISC::UQXTN_Sd_Sn(i) => return super::asimd_narrow::translate(em, i.0, true, 1),
        ASISDMISC::SQXTUN_Sd_Sn(i) => return super::asimd_narrow::translate(em, i.0, true, 2),
        ASISDMISC::SCVTF_Sd_Sn(i) => (i.0, false, false),
        ASISDMISC::UCVTF_Sd_Sn(i) => (i.0, true, false),
        ASISDMISC::FCVTZS_Sd_Sn(i) => (i.0, false, true),
        ASISDMISC::FCVTZU_Sd_Sn(i) => (i.0, true, true),
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };
    let double = bit(raw, 22) != 0;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    // Both SIMD scalar conversion directions read Vn, never a general register.
    let value = if double {
        em.get_v_d(rn)
    } else {
        em.get_v_s(rn)
    };
    let result = convert(em, value, double, unsigned, float_to_int);
    if double {
        em.set_v_d(rd, result);
    } else {
        em.set_v_s(rd, result);
    }
    Ok(InstStatus::Continue)
}

fn pack_singles(em: &mut IrEmitter<'_>, low: ValueRef, high: ValueRef) -> ValueRef {
    // Scalar conversion results are U32 bit patterns. Their spill locations may
    // retain upper bits, so explicitly zero-extend before 64-bit packing.
    let mask = em.const_u64(u32::MAX as u64);
    let low = em.and(low, mask, RegSize::X);
    let high = em.and(high, mask, RegSize::X);
    let shift = em.const_u64(32);
    let high = em.lsl(high, shift, RegSize::X);
    em.or(low, high, RegSize::X)
}

pub fn translate_vector(em: &mut IrEmitter<'_>, raw: u32, unsigned: bool) -> Result<InstStatus> {
    translate_vector_common(em, raw, unsigned, false)
}

pub fn translate_vector_float_to_int(
    em: &mut IrEmitter<'_>,
    raw: u32,
    unsigned: bool,
) -> Result<InstStatus> {
    translate_vector_common(em, raw, unsigned, true)
}

fn translate_vector_common(
    em: &mut IrEmitter<'_>,
    raw: u32,
    unsigned: bool,
    float_to_int: bool,
) -> Result<InstStatus> {
    let double = bit(raw, 22) != 0;
    let q = bit(raw, 30) != 0;
    if double && !q {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let source = em.get_v_q(rn);
    let (low, high) = if double {
        let low = em.vec_extract_lo64(source);
        let high = em.vec_extract_hi64(source);
        (
            convert(em, low, true, unsigned, float_to_int),
            convert(em, high, true, unsigned, float_to_int),
        )
    } else {
        let mut lanes = Vec::with_capacity(if q { 4 } else { 2 });
        for lane in 0..if q { 4 } else { 2 } {
            let value = em.vec_extract_u32(source, lane);
            lanes.push(convert(em, value, false, unsigned, float_to_int));
        }
        let low = pack_singles(em, lanes[0], lanes[1]);
        let high = if q {
            pack_singles(em, lanes[2], lanes[3])
        } else {
            em.const_u64(0)
        };
        (low, high)
    };
    let result = em.vec_build_q(low, high);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

/// Fixed-point FCVTZS/FCVTZU S/D lanes. Values follow the existing saturating
/// converters; complete arithmetic FPSR handling remains their existing gap.
pub fn translate_fixed_float_to_int(
    em: &mut IrEmitter<'_>,
    raw: u32,
    scalar: bool,
    unsigned: bool,
) -> Result<InstStatus> {
    let immh = bits(raw, 19, 4);
    let double = immh & 8 != 0;
    let q = bit(raw, 30) != 0;
    // Only S/D are implemented; zero, byte/halfword/FP16, and vector .1D
    // encodings must not accidentally enter the 32-bit conversion path.
    if (!double && immh & 4 == 0) || (!scalar && double && !q) {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let width = if double { 64 } else { 32 };
    let fracbits = 2 * width - ((immh << 3) | bits(raw, 16, 3));
    if !(1..=width).contains(&fracbits) {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let source = em.get_v_q(rn);
    let scale = if double {
        em.const_u64((1023 + fracbits as u64) << 52)
    } else {
        em.const_u32((127 + fracbits) << 23)
    };
    let lanes = if scalar {
        1
    } else if double || !q {
        2
    } else {
        4
    };
    let mut results = Vec::with_capacity(lanes);
    for lane in 0..lanes {
        let value = if double {
            if lane == 0 {
                em.vec_extract_lo64(source)
            } else {
                em.vec_extract_hi64(source)
            }
        } else {
            em.vec_extract_u32(source, lane as u32)
        };
        // Positive power-of-two scaling is exact until overflow. Infinity
        // then saturates to the same integer endpoint as the exact result.
        let (op, ty) = if double {
            (Op::Fmul64, Ty::U64)
        } else {
            (Op::Fmul32, Ty::U32)
        };
        let scaled = em.push(Armlet::new(op, ty).with_args(&[value, scale]));
        results.push(convert(em, scaled, double, unsigned, true));
    }
    if scalar {
        if double {
            em.set_v_d(rd, results[0]);
        } else {
            em.set_v_s(rd, results[0]);
        }
    } else {
        let (low, high) = if double {
            (results[0], results[1])
        } else {
            let low = pack_singles(em, results[0], results[1]);
            let high = if q {
                pack_singles(em, results[2], results[3])
            } else {
                em.const_u64(0)
            };
            (low, high)
        };
        let result = em.vec_build_q(low, high);
        em.set_v_q(rd, result);
    }
    Ok(InstStatus::Continue)
}
