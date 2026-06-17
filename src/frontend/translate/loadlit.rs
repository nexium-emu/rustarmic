use disarm64::decoder::LOADLIT;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: LOADLIT) -> Result<InstStatus> {
    use LOADLIT::*;
    let raw = match insn {
        LDR_Rt_ADDR_PCREL19(i)     => i.0,
        LDRSW_Rt_ADDR_PCREL19(i)   => i.0,
        LDR_Ft_ADDR_PCREL19(i)     => i.0,
        PRFM_PRFOP_ADDR_PCREL19(_) => return Ok(InstStatus::Continue),
    };

    let opc   = bits(raw, 30, 2);
    let v     = bit(raw, 26);
    let imm19 = bits(raw, 5, 19);
    let rt    = bits(raw, 0, 5) as u8;

    let offset = sign_extend((imm19 as u64) << 2, 21);
    let addr_val = em.current_pc.wrapping_add(offset as u64);
    let addr = em.const_u64(addr_val);

    if v == 1 {
        match opc {
            0b00 => { let val = em.load(addr, 4); em.set_v_s(rt, val); }
            0b01 => { let val = em.load(addr, 8); em.set_v_d(rt, val); }
            0b10 => {
                let lo = em.load(addr, 8);
                let eight = em.const_u64(8);
                let addr_hi = em.add(addr, eight, RegSize::X);
                let hi = em.load(addr_hi, 8);
                let q = em.vec_build_q(lo, hi);
                em.set_v_q(rt, q);
            }
            _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
        }
    } else {
        match opc {
            0b00 => { let val = em.load(addr, 4); em.set_w(rt, val); }
            0b01 => { let val = em.load(addr, 8); em.set_x(rt, val); }
            0b10 => {
                let val = em.load(addr, 4);
                let shl = em.const_u64(32);
                let s1 = em.lsl(val, shl, RegSize::X);
                let shl2 = em.const_u64(32);
                let sx = em.asr(s1, shl2, RegSize::X);
                em.set_x(rt, sx);
            }
            _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
        }
    }

    Ok(InstStatus::Continue)
}
