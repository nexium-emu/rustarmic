use disarm64::decoder::LDSTPAIR_OFF;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: LDSTPAIR_OFF) -> Result<InstStatus> {
    use LDSTPAIR_OFF::*;
    let (raw, is_load, is_fp) = match insn {
        STP_Rt_Rt2_ADDR_SIMM7(i)                  => (i.0, false, false),
        LDP_Rt_Rt2_ADDR_SIMM7(i)                  => (i.0, true,  false),
        STP_Ft_Ft2_ADDR_SIMM7(i)                  => (i.0, false, true),
        LDP_Ft_Ft2_ADDR_SIMM7(i)                  => (i.0, true,  true),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let opc  = bits(raw, 30, 2);
    let imm7 = bits(raw, 15, 7);
    let rt2  = bits(raw, 10, 5) as u8;
    let rn   = bits(raw, 5, 5) as u8;
    let rt   = bits(raw, 0, 5) as u8;

    let (scale, size_bytes, size_kind) = if is_fp {
        // FP: opc 00=S(4), 01=D(8), 10=Q(16, not yet supported).
        match opc {
            0b00 => (2u32, 4u32, RegSize::W),
            0b01 => (3,    8,    RegSize::X),
            _    => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
        }
    } else {
        match opc {
            0b00 => (2u32, 4u32, RegSize::W),
            0b10 => (3,    8,    RegSize::X),
            _    => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
        }
    };
    let offset = sign_extend(imm7 as u64, 7) << scale;

    let base = em.get_x_or_sp(rn, true);
    let off  = em.const_u64(offset as u64);
    let access_addr  = em.add(base, off, RegSize::X);
    let one          = em.const_u64(size_bytes as u64);
    let access_addr2 = em.add(access_addr, one, RegSize::X);

    if is_load {
        let lo = em.load(access_addr,  size_bytes);
        let hi = em.load(access_addr2, size_bytes);
        if is_fp {
            if size_bytes == 8 { em.set_v_d(rt, lo); em.set_v_d(rt2, hi); }
            else               { em.set_v_s(rt, lo); em.set_v_s(rt2, hi); }
        } else {
            em.set_gpr(rt,  lo, size_kind);
            em.set_gpr(rt2, hi, size_kind);
        }
    } else {
        let (lo, hi) = if is_fp {
            if size_bytes == 8 { (em.get_v_d(rt), em.get_v_d(rt2)) }
            else               { (em.get_v_s(rt), em.get_v_s(rt2)) }
        } else {
            (em.get_gpr(rt, size_kind), em.get_gpr(rt2, size_kind))
        };
        em.store(access_addr,  lo, size_bytes);
        em.store(access_addr2, hi, size_bytes);
    }
    Ok(InstStatus::Continue)
}
