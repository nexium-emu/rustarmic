use disarm64::decoder::LDST_REGOFF;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

enum Kind {
    LoadU,
    LoadS,
    Store,
    FpLoad,
    FpStore,
}

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_REGOFF) -> Result<InstStatus> {
    use LDST_REGOFF::*;
    let (raw, kind, target_x) = match insn {
        LDR_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadU, true),
        LDRB_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadU, false),
        LDRH_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadU, false),
        LDRSB_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadS, bit(i.0, 22) != 0),
        LDRSH_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadS, bit(i.0, 22) != 0),
        LDRSW_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadS, true),
        STR_Rt_ADDR_REGOFF(i) => (i.0, Kind::Store, true),
        STRB_Rt_ADDR_REGOFF(i) => (i.0, Kind::Store, false),
        STRH_Rt_ADDR_REGOFF(i) => (i.0, Kind::Store, false),
        LDR_Ft_ADDR_REGOFF(i) => (i.0, Kind::FpLoad, false),
        STR_Ft_ADDR_REGOFF(i) => (i.0, Kind::FpStore, false),
        PRFM_PRFOP_ADDR_REGOFF(_) => return Ok(InstStatus::Continue),
    };

    let size = bits(raw, 30, 2);
    let rm = bits(raw, 16, 5) as u8;
    let option_ = bits(raw, 13, 3);
    let s_bit = bit(raw, 12);
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;

    let q_form = matches!(kind, Kind::FpLoad | Kind::FpStore) && size == 0 && bit(raw, 23) == 1;
    let bytes = if q_form { 16 } else { 1u32 << size };
    let base = em.get_x_or_sp(rn, true);

    let mut off = em.get_x(rm);
    let (extracted_width, signed) = match option_ {
        0b010 => (32, false),
        0b011 => (64, false),
        0b110 => (32, true),
        0b111 => (64, true),
        _ => {
            return Err(Error::Decode {
                pc: em.current_pc,
                opcode: raw,
            });
        }
    };
    if extracted_width < 64 {
        let mask = (1u64 << extracted_width) - 1;
        let mask_c = em.const_u64(mask);
        off = em.and(off, mask_c, RegSize::X);
        if signed {
            let shl = em.const_u64((64 - extracted_width) as u64);
            let s1 = em.lsl(off, shl, RegSize::X);
            let shl2 = em.const_u64((64 - extracted_width) as u64);
            off = em.asr(s1, shl2, RegSize::X);
        }
    }
    if s_bit == 1 {
        let shamt_v = if q_form { 4 } else { size };
        let shamt = em.const_u64(shamt_v as u64);
        off = em.lsl(off, shamt, RegSize::X);
    }

    let addr = em.add(base, off, RegSize::X);

    match kind {
        Kind::Store => {
            let val_size = if size == 3 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            em.store(addr, val, bytes);
        }
        Kind::LoadU => {
            let v = em.load(addr, bytes);
            if size <= 2 {
                em.set_w(rt, v);
            } else {
                em.set_x(rt, v);
            }
            let _ = target_x;
        }
        Kind::LoadS => {
            let v = em.load(addr, bytes);
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let shl2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, shl2, RegSize::X);
            if target_x {
                em.set_x(rt, sx);
            } else {
                em.set_w(rt, sx);
            }
        }
        Kind::FpLoad => {
            if bytes == 16 {
                let lo = em.load(addr, 8);
                let eight = em.const_u64(8);
                let addr_hi = em.add(addr, eight, RegSize::X);
                let hi = em.load(addr_hi, 8);
                let q = em.vec_build_q(lo, hi);
                em.set_v_q(rt, q);
            } else if bytes == 8 {
                let v = em.load(addr, bytes);
                em.set_v_d(rt, v);
            } else if bytes == 4 {
                let v = em.load(addr, bytes);
                em.set_v_s(rt, v);
            } else {
                return Err(Error::Unsupported {
                    pc: em.current_pc,
                    opcode: raw,
                });
            }
        }
        Kind::FpStore => {
            if bytes == 16 {
                let q = em.get_v_q(rt);
                let lo = em.vec_extract_lo64(q);
                let hi = em.vec_extract_hi64(q);
                em.store(addr, lo, 8);
                let eight = em.const_u64(8);
                let addr_hi = em.add(addr, eight, RegSize::X);
                em.store(addr_hi, hi, 8);
            } else {
                let v = if bytes == 8 {
                    em.get_v_d(rt)
                } else if bytes == 4 {
                    em.get_v_s(rt)
                } else {
                    return Err(Error::Unsupported {
                        pc: em.current_pc,
                        opcode: raw,
                    });
                };
                em.store(addr, v, bytes);
            }
        }
    }
    Ok(InstStatus::Continue)
}
