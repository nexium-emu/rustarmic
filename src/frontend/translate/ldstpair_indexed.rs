use disarm64::decoder::LDSTPAIR_INDEXED;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: LDSTPAIR_INDEXED) -> Result<InstStatus> {
    use LDSTPAIR_INDEXED::*;
    let (raw, is_load, is_fp) = match insn {
        STP_Rt_W_Rt2_W_ADDR_SIMM7_S_S(i) => (i.0, false, false),
        LDP_Rt_W_Rt2_W_ADDR_SIMM7_S_S(i) => (i.0, true, false),
        STP_Ft_S_S_Ft2_S_S_ADDR_SIMM7_S_S(i) => (i.0, false, true),
        LDP_Ft_S_S_Ft2_S_S_ADDR_SIMM7_S_S(i) => (i.0, true, true),
        LDPSW_Rt_X_Rt2_X_ADDR_SIMM7_S_S(i) => {
            let raw = i.0;
            let mode = bits(raw, 23, 3);
            let idx = if mode == 0b001 {
                super::ldstpair_off::IdxMode::Post
            } else {
                super::ldstpair_off::IdxMode::Pre
            };
            return super::ldstpair_off::ldpsw(em, raw, idx);
        }
        _ => {
            return Err(Error::Unsupported {
                pc: em.current_pc,
                opcode: 0,
            });
        }
    };

    let opc = bits(raw, 30, 2);
    let mode = bits(raw, 23, 3);
    let imm7 = bits(raw, 15, 7);
    let rt2 = bits(raw, 10, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;

    let (scale, size_bytes, size_kind) = if is_fp {
        match opc {
            0b00 => (2u32, 4u32, RegSize::W),
            0b01 => (3, 8, RegSize::X),
            0b10 => (4, 16, RegSize::X),
            _ => {
                return Err(Error::Unsupported {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
        }
    } else {
        match opc {
            0b00 => (2u32, 4u32, RegSize::W),
            0b10 => (3, 8, RegSize::X),
            _ => {
                return Err(Error::Unsupported {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
        }
    };
    let offset = sign_extend(imm7 as u64, 7) << scale;

    let base = em.get_x_or_sp(rn, true);
    let off = em.const_u64(offset as u64);
    let writeback_addr = em.add(base, off, RegSize::X);

    let access_addr = match mode {
        0b001 => base,
        _ => writeback_addr,
    };
    let one = em.const_u64(size_bytes as u64);
    let access_addr2 = em.add(access_addr, one, RegSize::X);

    if is_load {
        if is_fp && size_bytes == 16 {
            fp_load_q(em, rt, access_addr);
            fp_load_q(em, rt2, access_addr2);
        } else {
            let lo = em.load(access_addr, size_bytes);
            let hi = em.load(access_addr2, size_bytes);
            if is_fp {
                if size_bytes == 8 {
                    em.set_v_d(rt, lo);
                    em.set_v_d(rt2, hi);
                } else {
                    em.set_v_s(rt, lo);
                    em.set_v_s(rt2, hi);
                }
            } else {
                em.set_gpr(rt, lo, size_kind);
                em.set_gpr(rt2, hi, size_kind);
            }
        }
    } else {
        if is_fp && size_bytes == 16 {
            fp_store_q(em, rt, access_addr);
            fp_store_q(em, rt2, access_addr2);
        } else {
            let (lo, hi) = if is_fp {
                if size_bytes == 8 {
                    (em.get_v_d(rt), em.get_v_d(rt2))
                } else {
                    (em.get_v_s(rt), em.get_v_s(rt2))
                }
            } else {
                (em.get_gpr(rt, size_kind), em.get_gpr(rt2, size_kind))
            };
            em.store(access_addr, lo, size_bytes);
            em.store(access_addr2, hi, size_bytes);
        }
    }

    em.set_x_or_sp(rn, writeback_addr, true);
    Ok(InstStatus::Continue)
}

fn fp_load_q(em: &mut IrEmitter<'_>, rt: u8, addr: crate::ir::ValueRef) {
    let lo = em.load(addr, 8);
    let eight = em.const_u64(8);
    let addr_hi = em.add(addr, eight, RegSize::X);
    let hi = em.load(addr_hi, 8);
    let q = em.vec_build_q(lo, hi);
    em.set_v_q(rt, q);
}

fn fp_store_q(em: &mut IrEmitter<'_>, rt: u8, addr: crate::ir::ValueRef) {
    let q = em.get_v_q(rt);
    let lo = em.vec_extract_lo64(q);
    let hi = em.vec_extract_hi64(q);
    em.store(addr, lo, 8);
    let eight = em.const_u64(8);
    let addr_hi = em.add(addr, eight, RegSize::X);
    em.store(addr_hi, hi, 8);
}
