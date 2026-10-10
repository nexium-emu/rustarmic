use disarm64::decoder::FLOAT2INT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{Armlet, IrEmitter, Op, Ty};
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: FLOAT2INT) -> Result<InstStatus> {
    use FLOAT2INT::*;
    let raw = match insn {
        FCVTZS_Rd_Fn(i) => i.0,
        FCVTZS_Rd_W_Fn_S_D(i) => i.0,
        FCVTMS_Rd_Fn(i) => i.0,
        FCVTMS_Rd_W_Fn_S_D(i) => i.0,
        FCVTMU_Rd_Fn(i) => i.0,
        FCVTMU_Rd_W_Fn_S_D(i) => i.0,
        FCVTNS_Rd_Fn(i) => i.0,
        FCVTNS_Rd_W_Fn_S_D(i) => i.0,
        FCVTNU_Rd_Fn(i) => i.0,
        FCVTNU_Rd_W_Fn_S_D(i) => i.0,
        FCVTAS_Rd_Fn(i) => i.0,
        FCVTAS_Rd_W_Fn_S_D(i) => i.0,
        FCVTAU_Rd_Fn(i) => i.0,
        FCVTAU_Rd_W_Fn_S_D(i) => i.0,
        FCVTPS_Rd_Fn(i) => i.0,
        FCVTPS_Rd_W_Fn_S_D(i) => i.0,
        FCVTPU_Rd_Fn(i) => i.0,
        FCVTPU_Rd_W_Fn_S_D(i) => i.0,
        FCVTZU_Rd_Fn(i) => i.0,
        FCVTZU_Rd_W_Fn_S_D(i) => i.0,
        SCVTF_Fd_Rn(i) => i.0,
        SCVTF_Fd_S_D_Rn_W(i) => i.0,
        UCVTF_Fd_Rn(i) => i.0,
        UCVTF_Fd_S_D_Rn_W(i) => i.0,
        FMOV_Fd_Rn(i) => i.0,
        FMOV_Fd_S_S_Rn_W(i) => i.0,
        FMOV_Rd_Fn(i) => i.0,
        FMOV_Rd_W_Fn_S_S(i) => i.0,
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let sf = bit(raw, 31);
    let ptype = bits(raw, 22, 2);
    let rmode = bits(raw, 19, 2);
    let opcode = bits(raw, 16, 3);
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;

    let dst_is_x = sf == 1;
    let src_is_x = sf == 1;
    let is_double = match ptype {
        0b00 => false,
        0b01 => true,
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };

    match (rmode, opcode) {
        (0b11, 0b000 | 0b001) => return translate_float_to_int(em, raw, 0),
        (0b00, 0b000 | 0b001 | 0b100 | 0b101) | (0b01 | 0b10, 0b000 | 0b001) => {
            let src = if is_double {
                em.get_v_d(rn)
            } else {
                em.get_v_s(rn)
            };
            // Integral rounding is exact in the source format. The existing
            // saturating truncate converters then give the integer value,
            // including NaNs, negative unsigned input and range endpoints.
            // Complete guest arithmetic FPSR handling remains a backend gap.
            let src = match (rmode, opcode) {
                (0b01, _) => super::fp_dp1::round_to_plus_infinity(em, src, is_double),
                (0b10, _) => super::fp_dp1::round_to_minus_infinity(em, src, is_double),
                (0b00, 0b000 | 0b001) => super::fp_dp1::round_to_nearest(em, src, is_double, false),
                (0b00, 0b100 | 0b101) => super::fp_dp1::round_to_nearest(em, src, is_double, true),
                _ => unreachable!(),
            };
            let unsigned = opcode & 1 != 0;
            let (op, ty) = match (is_double, dst_is_x, unsigned) {
                (false, false, false) => (Op::FcvtZsSW, Ty::U32),
                (false, true, false) => (Op::FcvtZsSX, Ty::U64),
                (true, false, false) => (Op::FcvtZsDW, Ty::U32),
                (true, true, false) => (Op::FcvtZsDX, Ty::U64),
                (false, false, true) => (Op::FcvtZuSW, Ty::U32),
                (false, true, true) => (Op::FcvtZuSX, Ty::U64),
                (true, false, true) => (Op::FcvtZuDW, Ty::U32),
                (true, true, true) => (Op::FcvtZuDX, Ty::U64),
            };
            let result = em.push(Armlet::new(op, ty).with_args(&[src]));
            em.set_gpr(rd, result, if dst_is_x { RegSize::X } else { RegSize::W });
        }
        (0b00, 0b010) => {
            let size = if src_is_x { RegSize::X } else { RegSize::W };
            let src = em.get_gpr(rn, size);
            let (op, ty) = match (is_double, src_is_x) {
                (false, false) => (Op::ScvtfWS, Ty::U32),
                (false, true) => (Op::ScvtfXS, Ty::U32),
                (true, false) => (Op::ScvtfWD, Ty::U64),
                (true, true) => (Op::ScvtfXD, Ty::U64),
            };
            let r = em.push(Armlet::new(op, ty).with_args(&[src]));
            if is_double {
                em.set_v_d(rd, r);
            } else {
                em.set_v_s(rd, r);
            }
        }

        (0b00, 0b011) => {
            let size = if src_is_x { RegSize::X } else { RegSize::W };
            let src = em.get_gpr(rn, size);
            let (op, ty) = match (is_double, src_is_x) {
                (false, false) => (Op::UcvtfWS, Ty::U32),
                (false, true) => (Op::UcvtfXS, Ty::U32),
                (true, false) => (Op::UcvtfWD, Ty::U64),
                (true, true) => (Op::UcvtfXD, Ty::U64),
            };
            let r = em.push(Armlet::new(op, ty).with_args(&[src]));
            if is_double {
                em.set_v_d(rd, r);
            } else {
                em.set_v_s(rd, r);
            }
        }

        (0b00, 0b111) => {
            let size = if src_is_x { RegSize::X } else { RegSize::W };
            let src = em.get_gpr(rn, size);
            if is_double {
                em.set_v_d(rd, src);
            } else {
                em.set_v_s(rd, src);
            }
        }

        (0b00, 0b110) => {
            let src = if is_double {
                em.get_v_d(rn)
            } else {
                em.get_v_s(rn)
            };
            let size = if dst_is_x { RegSize::X } else { RegSize::W };
            em.set_gpr(rd, src, size);
        }

        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    }

    Ok(InstStatus::Continue)
}

pub(super) fn translate_float_to_int(
    em: &mut IrEmitter<'_>,
    raw: u32,
    fracbits: u32,
) -> Result<InstStatus> {
    let is_double = match bits(raw, 22, 2) {
        0 => false,
        1 => true,
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: raw,
            })
        }
    };
    let dst_is_x = bit(raw, 31) != 0;
    let unsigned = bit(raw, 16) != 0;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let mut src = if is_double {
        em.get_v_d(rn)
    } else {
        em.get_v_s(rn)
    };
    if fracbits != 0 {
        // Scaling by a positive power of two is exact until overflow. An
        // overflowed infinity has the same saturated integer result as the
        // exact scaled value, so no intermediate rounding changes the result.
        let (op, ty, scale) = if is_double {
            (
                Op::Fmul64,
                Ty::U64,
                em.const_u64((1023 + fracbits as u64) << 52),
            )
        } else {
            (Op::Fmul32, Ty::U32, em.const_u32((127 + fracbits) << 23))
        };
        src = em.push(Armlet::new(op, ty).with_args(&[src, scale]));
    }
    let (op, ty) = match (is_double, dst_is_x, unsigned) {
        (false, false, false) => (Op::FcvtZsSW, Ty::U32),
        (false, true, false) => (Op::FcvtZsSX, Ty::U64),
        (true, false, false) => (Op::FcvtZsDW, Ty::U32),
        (true, true, false) => (Op::FcvtZsDX, Ty::U64),
        (false, false, true) => (Op::FcvtZuSW, Ty::U32),
        (false, true, true) => (Op::FcvtZuSX, Ty::U64),
        (true, false, true) => (Op::FcvtZuDW, Ty::U32),
        (true, true, true) => (Op::FcvtZuDX, Ty::U64),
    };
    let result = em.push(Armlet::new(op, ty).with_args(&[src]));
    em.set_gpr(rd, result, if dst_is_x { RegSize::X } else { RegSize::W });
    Ok(InstStatus::Continue)
}
