use disarm64::decoder::ASISDSAME;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty, ValueRef};
use crate::util::bits::{bit, bits};

pub fn translate_scalar_step(em: &mut IrEmitter<'_>, insn: ASISDSAME) -> Result<InstStatus> {
    match insn {
        ASISDSAME::SQADD_Sd_Sn_Sm(i) => super::asimd_same::translate_saturating_binary(em, i.0, true, false, false),
        ASISDSAME::UQADD_Sd_Sn_Sm(i) => super::asimd_same::translate_saturating_binary(em, i.0, true, true, false),
        ASISDSAME::SQSUB_Sd_Sn_Sm(i) => super::asimd_same::translate_saturating_binary(em, i.0, true, false, true),
        ASISDSAME::UQSUB_Sd_Sn_Sm(i) => super::asimd_same::translate_saturating_binary(em, i.0, true, true, true),
        ASISDSAME::FACGE_Sd_S_S_Sn_S_S_Sm_S_S(i) => super::asimd_same::translate_absolute_compare(em, i.0, true, false),
        ASISDSAME::FACGT_Sd_S_S_Sn_S_S_Sm_S_S(i) => super::asimd_same::translate_absolute_compare(em, i.0, true, true),
        ASISDSAME::ADD_Sd_Sn_Sm(i) => translate_scalar_integer(em, i.0, false),
        ASISDSAME::SUB_Sd_Sn_Sm(i) => translate_scalar_integer(em, i.0, true),
        ASISDSAME::CMTST_Sd_Sn_Sm(i) => super::asimd_same::translate_cmtst(em, i.0, true),
        ASISDSAME::FRSQRTS_Sd_S_S_Sn_S_S_Sm_S_S(i) => translate(em, i.0, true, true),
        ASISDSAME::FRECPS_Sd_S_S_Sn_S_S_Sm_S_S(i) => translate_reciprocal(em, i.0, true, true),
        ASISDSAME::FABD_Sd_S_S_Sn_S_S_Sm_S_S(i) => translate_absolute_difference(em, i.0, true),
        _ => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: 0,
        }),
    }
}

fn translate_scalar_integer(em: &mut IrEmitter<'_>, raw: u32, subtract: bool) -> Result<InstStatus> {
    if bits(raw, 22, 2) != 3 { return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }); }
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let n = em.get_v_d(rn);
    let m = em.get_v_d(rm);
    let result = if subtract { em.sub(n, m, RegSize::X) } else { em.add(n, m, RegSize::X) };
    em.set_v_d(rd, result);
    Ok(InstStatus::Continue)
}

pub fn translate_absolute_difference(
    em: &mut IrEmitter<'_>,
    raw: u32,
    scalar: bool,
) -> Result<InstStatus> {
    let double = bit(raw, 22) != 0;
    let q = bit(raw, 30) != 0;
    if !scalar && double && !q {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let op = if double {
        Op::FabsDiff64
    } else {
        Op::FabsDiff32
    };
    let ty = if double { Ty::U64 } else { Ty::U32 };
    // Snapshot both original full sources, so aliases never corrupt a later
    // lane and NaN selection sees unmodified N/M rather than a negated M.
    let n = em.get_v_q(rn);
    let m = em.get_v_q(rm);
    let mut values = Vec::new();
    for lane in 0..if scalar {
        1
    } else if double || !q {
        2
    } else {
        4
    } {
        let extract = |em: &mut IrEmitter<'_>, source| {
            if double {
                if lane == 0 {
                    em.vec_extract_lo64(source)
                } else {
                    em.vec_extract_hi64(source)
                }
            } else {
                em.vec_extract_u32(source, lane)
            }
        };
        let n = extract(em, n);
        let m = extract(em, m);
        values.push(em.push(Armlet::new(op, ty).with_args(&[n, m])));
    }
    if scalar {
        if double {
            em.set_v_d(rd, values[0]);
        } else {
            em.set_v_s(rd, values[0]);
        }
    } else {
        let (low, high) = if double {
            (values[0], values[1])
        } else {
            let low = pack(em, values[0], values[1]);
            let high = if q {
                pack(em, values[2], values[3])
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

fn operation(
    em: &mut IrEmitter<'_>,
    n: ValueRef,
    m: Option<ValueRef>,
    double: bool,
    reciprocal: bool,
) -> ValueRef {
    let op = match (m.is_some(), double, reciprocal) {
        (false, false, false) => Op::FrsqrtEstimate32,
        (false, true, false) => Op::FrsqrtEstimate64,
        (true, false, false) => Op::FrsqrtStep32,
        (true, true, false) => Op::FrsqrtStep64,
        (false, false, true) => Op::FrecipEstimate32,
        (false, true, true) => Op::FrecipEstimate64,
        (true, false, true) => Op::FrecipStep32,
        (true, true, true) => Op::FrecipStep64,
    };
    let armlet = Armlet::new(op, if double { Ty::U64 } else { Ty::U32 });
    em.push(if let Some(m) = m {
        armlet.with_args(&[n, m])
    } else {
        armlet.with_args(&[n])
    })
}

fn pack(em: &mut IrEmitter<'_>, low: ValueRef, high: ValueRef) -> ValueRef {
    let mask = em.const_u64(u32::MAX as u64);
    let low = em.and(low, mask, RegSize::X);
    let high = em.and(high, mask, RegSize::X);
    let shift = em.const_u64(32);
    let high = em.lsl(high, shift, RegSize::X);
    em.or(low, high, RegSize::X)
}

pub fn translate(em: &mut IrEmitter<'_>, raw: u32, scalar: bool, step: bool) -> Result<InstStatus> {
    translate_common(em, raw, scalar, step, false)
}

pub fn translate_reciprocal(
    em: &mut IrEmitter<'_>,
    raw: u32,
    scalar: bool,
    step: bool,
) -> Result<InstStatus> {
    translate_common(em, raw, scalar, step, true)
}

fn translate_common(
    em: &mut IrEmitter<'_>,
    raw: u32,
    scalar: bool,
    step: bool,
    reciprocal: bool,
) -> Result<InstStatus> {
    let double = bit(raw, 22) != 0;
    let q = bit(raw, 30) != 0;
    if !scalar && double && !q {
        return Err(Error::Decode {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rm = bits(raw, 16, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    if scalar {
        let n = if double {
            em.get_v_d(rn)
        } else {
            em.get_v_s(rn)
        };
        let m = step.then(|| {
            if double {
                em.get_v_d(rm)
            } else {
                em.get_v_s(rm)
            }
        });
        let result = operation(em, n, m, double, reciprocal);
        if double {
            em.set_v_d(rd, result);
        } else {
            em.set_v_s(rd, result);
        }
        return Ok(InstStatus::Continue);
    }
    // Snapshot both whole sources before any destination write, including all
    // Rd==Rn/Rm and V31 aliases. Q0 always clears the high half.
    let n = em.get_v_q(rn);
    let m = step.then(|| em.get_v_q(rm));
    let mut lanes = Vec::with_capacity(if double || !q { 2 } else { 4 });
    for lane in 0..if double || !q { 2 } else { 4 } {
        let extract = |em: &mut IrEmitter<'_>, v| {
            if double {
                if lane == 0 {
                    em.vec_extract_lo64(v)
                } else {
                    em.vec_extract_hi64(v)
                }
            } else {
                em.vec_extract_u32(v, lane)
            }
        };
        let n = extract(em, n);
        let m = m.map(|v| extract(em, v));
        lanes.push(operation(em, n, m, double, reciprocal));
    }
    let (low, high) = if double {
        (lanes[0], lanes[1])
    } else {
        let low = pack(em, lanes[0], lanes[1]);
        let high = if q {
            pack(em, lanes[2], lanes[3])
        } else {
            em.const_u64(0)
        };
        (low, high)
    };
    let result = em.vec_build_q(low, high);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
