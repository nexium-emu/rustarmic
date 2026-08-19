use disarm64::decoder::LDST_POS;

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

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_POS) -> Result<InstStatus> {
    use LDST_POS::*;
    let (raw, kind, target_x) = match insn {
        LDR_Rt_ADDR_UIMM12(i) => (i.0, Kind::LoadU, true),
        LDRB_Rt_ADDR_UIMM12(i) => (i.0, Kind::LoadU, false),
        LDRH_Rt_ADDR_UIMM12(i) => (i.0, Kind::LoadU, false),
        LDRSB_Rt_ADDR_UIMM12(i) => (i.0, Kind::LoadS, true),
        LDRSH_Rt_ADDR_UIMM12(i) => (i.0, Kind::LoadS, true),
        LDRSW_Rt_ADDR_UIMM12(i) => (i.0, Kind::LoadS, true),
        STR_Rt_ADDR_UIMM12(i) => (i.0, Kind::Store, true),
        STRB_Rt_ADDR_UIMM12(i) => (i.0, Kind::Store, false),
        STRH_Rt_ADDR_UIMM12(i) => (i.0, Kind::Store, false),
        LDR_Ft_ADDR_UIMM12(i) => (i.0, Kind::FpLoad, false),
        STR_Ft_ADDR_UIMM12(i) => (i.0, Kind::FpStore, false),
        PRFM_PRFOP_ADDR_UIMM12(_) => return Ok(InstStatus::Continue),
    };

    let size = bits(raw, 30, 2);
    let imm12 = bits(raw, 10, 12);
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;

    let q_form = matches!(kind, Kind::FpLoad | Kind::FpStore) && size == 0 && bit(raw, 23) == 1;
    let bytes = if q_form { 16 } else { 1u32 << size };
    let offset = (imm12 as u64) * (bytes as u64);

    let base = em.get_x_or_sp(rn, true);
    let off = em.const_u64(offset);
    let addr = em.add(base, off, RegSize::X);

    match kind {
        Kind::Store => {
            let val_size = if size == 3 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            em.store(addr, val, bytes);
        }
        Kind::LoadU => {
            let v = em.load(addr, bytes);
            if size <= 2 && !target_x {
                em.set_w(rt, v);
            } else if size <= 2 {
                em.set_w(rt, v);
            } else {
                em.set_x(rt, v);
            }
        }
        Kind::LoadS => {
            let v = em.load(addr, bytes);
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let s2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, s2, RegSize::X);
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
