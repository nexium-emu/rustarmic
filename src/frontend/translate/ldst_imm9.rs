use disarm64::decoder::LDST_IMM9;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

enum Kind { LoadU, LoadS, Store, FpLoad, FpStore }

pub fn translate(em: &mut IrEmitter<'_>, insn: LDST_IMM9) -> Result<InstStatus> {
    use LDST_IMM9::*;
    let (raw, kind, target_x) = match insn {
        LDR_Rt_ADDR_SIMM9(i)   => (i.0, Kind::LoadU, true),
        LDRB_Rt_ADDR_SIMM9(i)  => (i.0, Kind::LoadU, false),
        LDRH_Rt_ADDR_SIMM9(i)  => (i.0, Kind::LoadU, false),
        LDRSB_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        LDRSH_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        LDRSW_Rt_ADDR_SIMM9(i) => (i.0, Kind::LoadS, true),
        STR_Rt_ADDR_SIMM9(i)   => (i.0, Kind::Store, true),
        STRB_Rt_ADDR_SIMM9(i)  => (i.0, Kind::Store, false),
        STRH_Rt_ADDR_SIMM9(i)  => (i.0, Kind::Store, false),
        LDR_Ft_ADDR_SIMM9(i)   => (i.0, Kind::FpLoad, false),
        STR_Ft_ADDR_SIMM9(i)   => (i.0, Kind::FpStore, false),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let size  = bits(raw, 30, 2);
    let imm9  = bits(raw, 12, 9);
    let mode  = bits(raw, 10, 2);
    let rn    = bits(raw, 5, 5) as u8;
    let rt    = bits(raw, 0, 5) as u8;

    let bytes  = 1u32 << size;
    let offset = sign_extend(imm9 as u64, 9);

    let base = em.get_x_or_sp(rn, true);
    let off  = em.const_u64(offset as u64);
    let effective_addr = em.add(base, off, RegSize::X);

    let access_addr = match mode {
        0b01 => base,
        0b11 => effective_addr,
        _    => effective_addr,
    };

    match kind {
        Kind::Store => {
            let val_size = if size == 3 { RegSize::X } else { RegSize::W };
            let val = em.get_gpr(rt, val_size);
            em.store(access_addr, val, bytes);
        }
        Kind::LoadU => {
            let v = em.load(access_addr, bytes);
            if size <= 2 { em.set_w(rt, v); } else { em.set_x(rt, v); }
            let _ = target_x;
        }
        Kind::LoadS => {
            let v = em.load(access_addr, bytes);
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let shl2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, shl2, RegSize::X);
            if target_x { em.set_x(rt, sx); } else { em.set_w(rt, sx); }
        }
        Kind::FpLoad => {
            let v = em.load(access_addr, bytes);
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
            em.store(access_addr, v, bytes);
        }
    }

    if mode == 0b01 || mode == 0b11 {
        em.set_x_or_sp(rn, effective_addr, true);
    }
    Ok(InstStatus::Continue)
}
