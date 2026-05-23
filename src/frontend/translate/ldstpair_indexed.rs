use disarm64::decoder::LDSTPAIR_INDEXED;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, insn: LDSTPAIR_INDEXED) -> Result<InstStatus> {
    use LDSTPAIR_INDEXED::*;
    let (raw, is_load) = match insn {
        STP_Rt_W_Rt2_W_ADDR_SIMM7_S_S(i) => (i.0, false),
        LDP_Rt_W_Rt2_W_ADDR_SIMM7_S_S(i) => (i.0, true),
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: 0 }),
    };

    let opc  = bits(raw, 30, 2);
    let mode = bits(raw, 23, 3); // 001=post, 011=pre
    let imm7 = bits(raw, 15, 7);
    let rt2  = bits(raw, 10, 5) as u8;
    let rn   = bits(raw, 5, 5) as u8;
    let rt   = bits(raw, 0, 5) as u8;

    let (scale, size_bytes, size_kind) = match opc {
        0b00 => (2u32, 4u32, RegSize::W),
        0b10 => (3,    8,    RegSize::X),
        _    => return Err(Error::Unsupported { pc: em.current_pc, opcode: raw }),
    };
    let offset = sign_extend(imm7 as u64, 7) << scale;

    let base = em.get_x_or_sp(rn, true);
    let off  = em.const_u64(offset as u64);
    let writeback_addr = em.add(base, off, RegSize::X);

    let access_addr = match mode {
        0b001 => base,
        _     => writeback_addr,
    };
    let one          = em.const_u64(size_bytes as u64);
    let access_addr2 = em.add(access_addr, one, RegSize::X);

    if is_load {
        let lo = em.load(access_addr,  size_bytes);
        let hi = em.load(access_addr2, size_bytes);
        em.set_gpr(rt,  lo, size_kind);
        em.set_gpr(rt2, hi, size_kind);
    } else {
        let lo = em.get_gpr(rt,  size_kind);
        let hi = em.get_gpr(rt2, size_kind);
        em.store(access_addr,  lo, size_bytes);
        em.store(access_addr2, hi, size_bytes);
    }

    em.set_x_or_sp(rn, writeback_addr, true);
    Ok(InstStatus::Continue)
}
