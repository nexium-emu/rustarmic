//! Load/store — top4 = x1x0. Initial coverage: unsigned immediate offset,
//! pre/post-indexed, register-offset, and load/store pair.

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    // Distinguish sub-classes via bits 28..23.
    let bit29 = bit(inst, 29);
    let bit28 = bit(inst, 28);
    let bit26 = bit(inst, 26);
    let bit24 = bit(inst, 24);

    if bit28 == 1 && bit29 == 1 && bit26 == 0 {
        // Load/store register (unsigned immediate offset)
        return ldst_unsigned_imm(em, inst);
    }
    if bit28 == 1 && bit29 == 0 && bit24 == 0 {
        // Load/store register (immediate post/pre-indexed or unscaled)
        return ldst_reg_imm_variants(em, inst);
    }
    if bit29 == 0 && bit28 == 1 && bits(inst, 21, 1) == 1 && bits(inst, 10, 2) == 0b10 {
        return ldst_register_offset(em, inst);
    }
    if bit26 == 0 && bits(inst, 27, 3) == 0b101 {
        return ldst_pair(em, inst);
    }
    Err(Error::Unsupported { pc: em.current_pc, opcode: inst })
}

fn size_to_bytes(size: u32) -> u32 { 1 << size }

fn ldst_unsigned_imm(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let size = bits(inst, 30, 2);
    let v    = bit(inst, 26); // 1 = SIMD/FP
    let opc  = bits(inst, 22, 2);
    let imm12 = bits(inst, 10, 12);
    let rn   = bits(inst, 5, 5) as u8;
    let rt   = bits(inst, 0, 5) as u8;

    if v == 1 {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: inst });
    }

    let bytes = size_to_bytes(size);
    let offset = (imm12 as u64) * (bytes as u64);

    let base = em.get_x_or_sp(rn, true);
    let off = em.const_u64(offset);
    let addr = em.add(base, off, RegSize::X);

    match opc {
        0b00 => {
            // STR
            let val = em.get_gpr(rt, if size == 3 { RegSize::X } else { RegSize::W });
            em.store(addr, val, bytes);
        }
        0b01 => {
            // LDR (unsigned)
            let v = em.load(addr, bytes);
            if size <= 2 {
                em.set_w(rt, v);
            } else {
                em.set_x(rt, v);
            }
        }
        0b10 | 0b11 => {
            // LDRSW / LDRSB / LDRSH (sign-extending). We model as load-then-sign-extend
            // by shifting left+ASR — the optimizer will collapse for constant widths.
            let v = em.load(addr, bytes);
            let target_size = if opc == 0b10 { RegSize::X } else { RegSize::W };
            let width_bits = (bytes * 8) as u64;
            let shl = em.const_u64(64 - width_bits);
            let s1 = em.lsl(v, shl, RegSize::X);
            let s2 = em.const_u64(64 - width_bits);
            let sx = em.asr(s1, s2, RegSize::X);
            em.set_gpr(rt, sx, target_size);
        }
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
    }
    Ok(InstStatus::Continue)
}

fn ldst_reg_imm_variants(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let size = bits(inst, 30, 2);
    let v    = bit(inst, 26);
    let opc  = bits(inst, 22, 2);
    let imm9 = bits(inst, 12, 9);
    let mode = bits(inst, 10, 2); // 00=unscaled,01=post,11=pre
    let rn   = bits(inst, 5, 5) as u8;
    let rt   = bits(inst, 0, 5) as u8;

    if v == 1 {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: inst });
    }
    if mode == 0b10 {
        // unprivileged variants — treat as plain unscaled for user-mode emu.
    }

    let bytes = size_to_bytes(size);
    let offset = sign_extend(imm9 as u64, 9);

    let base = em.get_x_or_sp(rn, true);
    let off  = em.const_u64(offset as u64);
    let effective_addr = em.add(base, off, RegSize::X);

    // Pre-indexed: writeback before access; Post-indexed: writeback after.
    let access_addr = match mode {
        0b01 => base,            // post-index: access at old base
        0b11 => effective_addr,  // pre-index: access at base+off
        _    => effective_addr,  // unscaled: access at base+off, no writeback
    };

    match opc {
        0b00 => {
            let val = em.get_gpr(rt, if size == 3 { RegSize::X } else { RegSize::W });
            em.store(access_addr, val, bytes);
        }
        0b01 => {
            let v = em.load(access_addr, bytes);
            if size <= 2 { em.set_w(rt, v); } else { em.set_x(rt, v); }
        }
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }

    if mode == 0b01 || mode == 0b11 {
        em.set_x_or_sp(rn, effective_addr, true);
    }
    Ok(InstStatus::Continue)
}

fn ldst_register_offset(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let size = bits(inst, 30, 2);
    let v    = bit(inst, 26);
    let opc  = bits(inst, 22, 2);
    let rm   = bits(inst, 16, 5) as u8;
    let option_ = bits(inst, 13, 3);
    let s_bit = bit(inst, 12);
    let rn   = bits(inst, 5, 5) as u8;
    let rt   = bits(inst, 0, 5) as u8;

    if v == 1 {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: inst });
    }

    let bytes = size_to_bytes(size);
    let base = em.get_x_or_sp(rn, true);

    // Extend Rm per option_, then optionally shift by `size`.
    let mut off = em.get_x(rm);
    let (extracted_width, signed) = match option_ {
        0b010 => (32, false),    // UXTW
        0b011 => (64, false),    // LSL / UXTX
        0b110 => (32, true),     // SXTW
        0b111 => (64, true),     // SXTX
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
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

    match opc {
        0b00 => {
            let val = em.get_gpr(rt, if size == 3 { RegSize::X } else { RegSize::W });
            em.store(addr, val, bytes);
        }
        0b01 => {
            let v = em.load(addr, bytes);
            if size <= 2 { em.set_w(rt, v); } else { em.set_x(rt, v); }
        }
        _ => return Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
    Ok(InstStatus::Continue)
}

fn ldst_pair(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let opc = bits(inst, 30, 2);
    let v   = bit(inst, 26);
    let l   = bit(inst, 22);
    let imm7 = bits(inst, 15, 7);
    let rt2 = bits(inst, 10, 5) as u8;
    let rn  = bits(inst, 5, 5) as u8;
    let rt  = bits(inst, 0, 5) as u8;
    let mode = bits(inst, 23, 3); // 010=offset,011=pre-index,001=post-index

    if v == 1 {
        return Err(Error::Unsupported { pc: em.current_pc, opcode: inst });
    }
    let (scale, size_bytes, size_kind) = match opc {
        0b00 => (2u32, 4u32, RegSize::W),
        0b10 => (3,    8,    RegSize::X),
        0b01 => return Err(Error::Unsupported { pc: em.current_pc, opcode: inst }), // LDPSW
        _    => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
    };
    let offset = sign_extend(imm7 as u64, 7) << scale;

    let base = em.get_x_or_sp(rn, true);
    let off  = em.const_u64(offset as u64);
    let writeback_addr = em.add(base, off, RegSize::X);

    let access_addr = match mode {
        0b001 => base,            // post-index
        0b011 | 0b010 => writeback_addr, // pre-index or plain offset
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
    };

    let one = em.const_u64(size_bytes as u64);
    let access_addr2 = em.add(access_addr, one, RegSize::X);

    if l == 1 {
        let lo = em.load(access_addr, size_bytes);
        let hi = em.load(access_addr2, size_bytes);
        em.set_gpr(rt,  lo, size_kind);
        em.set_gpr(rt2, hi, size_kind);
    } else {
        let lo = em.get_gpr(rt,  size_kind);
        let hi = em.get_gpr(rt2, size_kind);
        em.store(access_addr,  lo, size_bytes);
        em.store(access_addr2, hi, size_bytes);
    }

    if mode == 0b001 || mode == 0b011 {
        em.set_x_or_sp(rn, writeback_addr, true);
    }
    Ok(InstStatus::Continue)
}
