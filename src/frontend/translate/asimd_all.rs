use disarm64::decoder::ASIMDALL;

use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDALL) -> Result<InstStatus> {
    use ASIMDALL::*;
    match insn {
        ADDV_Fd_Vn(i) => translate_addv(em, i.0),
        UADDLV_Fd_Vn(i) => translate_uaddlv(em, i.0),
        SMINV_Fd_Vn(i) => super::asimd_int_reduce::translate(em, i.0, true, false),
        SMAXV_Fd_Vn(i) => super::asimd_int_reduce::translate(em, i.0, true, true),
        UMINV_Fd_Vn(i) => super::asimd_int_reduce::translate(em, i.0, false, false),
        UMAXV_Fd_Vn(i) => super::asimd_int_reduce::translate(em, i.0, false, true),
        FMAXV_Fd_S_S_Vn_V_4S(i) => translate_fp_reduce(em, i.0, 0),
        FMINV_Fd_S_S_Vn_V_4S(i) => translate_fp_reduce(em, i.0, 1),
        FMAXNMV_Fd_S_S_Vn_V_4S(i) => translate_fp_reduce(em, i.0, 2),
        FMINNMV_Fd_S_S_Vn_V_4S(i) => translate_fp_reduce(em, i.0, 3),
        _ => Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: 0,
        }),
    }
}

fn translate_fp_reduce(em: &mut IrEmitter<'_>, raw: u32, mode: u64) -> Result<InstStatus> {
    // The baseline single-precision reduction has exactly four lanes.
    // Q=0 and size.D are reserved; FP16 enums remain explicitly unsupported.
    if bit(raw, 30) == 0 || bit(raw, 22) != 0 {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
    }
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let source = em.get_v_q(rn);
    let low = em.vec_extract_lo64(source);
    let high = em.vec_extract_hi64(source);
    let result = em.push(Armlet::new(Op::FpMinMaxReduce32, Ty::U32)
        .with_args(&[low, high]).with_imm(mode));
    em.set_v_s(rd, result);
    Ok(InstStatus::Continue)
}

fn translate_uaddlv(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    // UADDLV B/H forms cover the common byte/halfword reductions and fit in
    // the existing scalar W Armlet path.  D-lane reductions need a 64-bit
    // widening accumulator and are kept as a precise unsupported stop until
    // that lowering is added, rather than silently truncating the sum.
    if size > 1 {
        return Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: raw,
        });
    }

    let lanes = (if q { 16 } else { 8 }) >> size;
    let vn = em.get_v_q(rn);
    let mut sum = em.const_u32(0);
    for lane in 0..lanes {
        let value = if size == 0 {
            em.vec_extract_u8(vn, lane)
        } else {
            em.vec_extract_u16(vn, lane)
        };
        sum = em.add(sum, value, crate::arch::RegSize::W);
    }
    em.set_v_s(rd, sum);
    Ok(InstStatus::Continue)
}

fn translate_addv(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q = bit(raw, 30) == 1;
    let size = bits(raw, 22, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    if !(q && size == 2) {
        return Err(Error::Unsupported {
            pc: em.current_pc,
            opcode: raw,
        });
    }
    let vn = em.get_v_q(rn);
    let sum = em.vec_addv32(vn);
    em.set_v_s(rd, sum);
    Ok(InstStatus::Continue)
}
