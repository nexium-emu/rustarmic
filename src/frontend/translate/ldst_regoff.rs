use disarm64::decoder::LDST_REGOFF;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

enum Kind { LoadU, LoadS, Store, FpLoad, FpStore }

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_REGOFF) -> Result<InstStatus> {
    use LDST_REGOFF::*;
    let (raw, kind, target_x) = match insn {
        LDR_Rt_ADDR_REGOFF(i)   => (i.0, Kind::LoadU, true),
        LDRB_Rt_ADDR_REGOFF(i)  => (i.0, Kind::LoadU, false),
        LDRH_Rt_ADDR_REGOFF(i)  => (i.0, Kind::LoadU, false),
        LDRSB_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadS, true),
        LDRSH_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadS, true),
        LDRSW_Rt_ADDR_REGOFF(i) => (i.0, Kind::LoadS, true),
        STR_Rt_ADDR_REGOFF(i)   => (i.0, Kind::Store, true),
        STRB_Rt_ADDR_REGOFF(i)  => (i.0, Kind::Store, false),
        STRH_Rt_ADDR_REGOFF(i)  => (i.0, Kind::Store, false),
        LDR_Ft_ADDR_REGOFF(i)   => (i.0, Kind::FpLoad, false),
        STR_Ft_ADDR_REGOFF(i)   => (i.0, Kind::FpStore, false),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let size    = bits(raw, 30, 2);
    let rm      = bits(raw, 16, 5) as u8;
    let option_ = bits(raw, 13, 3);
    let s_bit   = bit(raw, 12);
    let rn      = bits(raw, 5, 5) as u8;
    let rt      = bits(raw, 0, 5) as u8;

    let bytes = 1u32 << size;
    let base  = em.get_x_or_sp(rn, true);

    let mut off = em.get_x(rm);
    let (extracted_width, signed) = match option_ {
        0b010 => (32, false),
        0b011 => (64, false),
        0b110 => (32, true),
        0b111 => (64, true),
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: raw }),
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
        let shamt = em.const_u64(size as u64);
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
            if size <= 2 { em.set_w(rt, v); } else { em.set_x(rt, v); }
            let _ = target_x;
        }
        Kind::LoadS => {
            let v = em.load(addr, bytes);
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let shl2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, shl2, RegSize::X);
            if target_x { em.set_x(rt, sx); } else { em.set_w(rt, sx); }
        }
        Kind::FpLoad => {
            let v = em.load(addr, bytes);
            if bytes == 8 { em.set_v_d(rt, v); }
            else if bytes == 4 { em.set_v_s(rt, v); }
            else {
                return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
            }
        }
        Kind::FpStore => {
            let v = if bytes == 8 { em.get_v_d(rt) }
                    else if bytes == 4 { em.get_v_s(rt) }
                    else {
                        return Err(Error::Unsupported { pc: em.current_pc, opcode: raw });
                    };
            em.store(addr, v, bytes);
        }
    }
    Ok(InstStatus::Continue)
}
