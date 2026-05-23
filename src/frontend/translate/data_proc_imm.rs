//! Data processing — immediate (top4 = 100x).

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits, decode_bit_masks, sign_extend};

pub fn translate(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    // op0 = inst[25:23] within the imm group (after the top 100x match)
    let op0 = bits(inst, 23, 3);
    match op0 {
        0b000 | 0b001 => pc_rel_addressing(em, inst),
        0b010 | 0b011 => add_sub_imm(em, inst),
        0b100         => logical_imm(em, inst),
        0b101         => move_wide_imm(em, inst),
        0b110         => bitfield(em, inst),
        0b111         => extract(em, inst),
        _             => Err(Error::Unsupported { pc: em.current_pc, opcode: inst }),
    }
}

/// ADR / ADRP — PC-relative addressing.
fn pc_rel_addressing(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let op = bit(inst, 31);
    let immlo = bits(inst, 29, 2);
    let immhi = bits(inst, 5, 19);
    let rd = bits(inst, 0, 5) as u8;

    let raw = ((immhi as u64) << 2) | (immlo as u64);
    let offset = sign_extend(raw, 21);

    let base = if op == 0 {
        em.current_pc
    } else {
        // ADRP: PC[63:12] || 0..0 (12 bits), offset shifted left by 12.
        em.current_pc & !0xFFF
    };
    let final_off = if op == 0 { offset } else { offset << 12 };
    let target = base.wrapping_add(final_off as u64);

    let c = em.const_u64(target);
    em.set_x(rd, c);
    Ok(InstStatus::Continue)
}

/// ADD/SUB (immediate), optionally setting flags.
fn add_sub_imm(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf   = bit(inst, 31);
    let op_  = bit(inst, 30); // 0=ADD, 1=SUB
    let s    = bit(inst, 29); // S — set flags
    let sh   = bit(inst, 22); // shift imm12 left by 12
    let imm12 = bits(inst, 10, 12);
    let rn   = bits(inst, 5, 5) as u8;
    let rd   = bits(inst, 0, 5) as u8;

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let mut imm = imm12 as u64;
    if sh == 1 { imm <<= 12; }
    if sf == 0 { imm &= 0xFFFF_FFFF; }

    // SP-form encoding 31 reads/writes SP when S==0.
    let sp_form = s == 0;
    let a = em.get_x_or_sp(rn, sp_form);
    let b = em.const_u64(imm);

    if s == 1 {
        let (result, flag) = if op_ == 0 { em.adds(a, b, size) } else { em.subs(a, b, size) };
        em.set_nzcv(flag);
        em.set_gpr(rd, result, size);
    } else {
        let result = if op_ == 0 { em.add(a, b, size) } else { em.sub(a, b, size) };
        em.set_x_or_sp(rd, result, sp_form);
    }
    Ok(InstStatus::Continue)
}

/// AND/ORR/EOR/ANDS — logical (immediate). Uses the N:immr:imms bit-mask encoding.
fn logical_imm(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf   = bit(inst, 31);
    let opc  = bits(inst, 29, 2); // 00=AND,01=ORR,10=EOR,11=ANDS
    let n    = bit(inst, 22);
    let immr = bits(inst, 16, 6);
    let imms = bits(inst, 10, 6);
    let rn   = bits(inst, 5, 5) as u8;
    let rd   = bits(inst, 0, 5) as u8;

    let width = if sf == 1 { 64 } else { 32 };
    if sf == 0 && n != 0 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }
    let imm = decode_bit_masks(n, imms, immr, width)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: inst })?;

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let sp_form_dst = opc != 0b11; // ANDS writes XZR/WZR (no SP form)
    let a = em.get_gpr(rn, size);
    let b = em.const_u64(imm);

    let result = match opc {
        0b00 | 0b11 => em.and(a, b, size),
        0b01        => em.or(a, b, size),
        0b10        => em.eor(a, b, size),
        _ => unreachable!(),
    };

    if opc == 0b11 {
        // ANDS — compute flags from result via a (result & result)==result; we model it
        // as `subs result, 0` for now to set NZ flags. C and V are cleared.
        let zero = em.const_u64(0);
        let (_, flag) = em.subs(result, zero, size);
        em.set_nzcv(flag);
    }

    em.set_x_or_sp(rd, result, sp_form_dst);
    Ok(InstStatus::Continue)
}

/// MOVN/MOVZ/MOVK — move-wide immediate.
fn move_wide_imm(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf  = bit(inst, 31);
    let opc = bits(inst, 29, 2); // 00=MOVN, 10=MOVZ, 11=MOVK
    let hw  = bits(inst, 21, 2);
    let imm16 = bits(inst, 5, 16);
    let rd  = bits(inst, 0, 5) as u8;

    if sf == 0 && hw >= 2 {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }

    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let shift = hw * 16;
    let imm_shifted = (imm16 as u64) << shift;

    match opc {
        0b00 => {
            // MOVN: NOT(imm16 << hw*16)
            let mut value = !imm_shifted;
            if sf == 0 { value &= 0xFFFF_FFFF; }
            let c = em.const_u64(value);
            em.set_gpr(rd, c, size);
        }
        0b10 => {
            // MOVZ
            let c = em.const_u64(imm_shifted);
            em.set_gpr(rd, c, size);
        }
        0b11 => {
            // MOVK: clear that 16-bit field then OR in imm16<<shift.
            let prev = em.get_gpr(rd, size);
            let mask = !((0xFFFFu64) << shift);
            let mask_c = em.const_u64(if sf == 0 { mask & 0xFFFF_FFFF } else { mask });
            let cleared = em.and(prev, mask_c, size);
            let imm_c = em.const_u64(imm_shifted);
            let merged = em.or(cleared, imm_c, size);
            em.set_gpr(rd, merged, size);
        }
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
    }
    Ok(InstStatus::Continue)
}

/// SBFM/UBFM/BFM — bitfield. We lower SBFM/UBFM into mask+shift sequences so
/// the const-folder can collapse common idioms (UXTW, SXTW, LSL/LSR/ASR-imm).
fn bitfield(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf   = bit(inst, 31);
    let opc  = bits(inst, 29, 2); // 00=SBFM, 01=BFM, 10=UBFM
    let n    = bit(inst, 22);
    let immr = bits(inst, 16, 6);
    let imms = bits(inst, 10, 6);
    let rn   = bits(inst, 5, 5) as u8;
    let rd   = bits(inst, 0, 5) as u8;

    if sf != n {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let width = if sf == 1 { 64u32 } else { 32 };

    // Common helper: rotate Rn right by immr, then mask low (imms+1) bits.
    let src = em.get_gpr(rn, size);
    let r_amt = em.const_u64(immr as u64);
    let rotated = em.ror(src, r_amt, size);

    let mask_bits = imms + 1;
    let mask = if mask_bits >= width { (!0u64) >> (64 - width) } else { (1u64 << mask_bits) - 1 };
    let mask_c = em.const_u64(mask);
    let bot = em.and(rotated, mask_c, size);

    let result = match opc {
        0b10 => bot, // UBFM
        0b00 => {
            // SBFM — sign-extend from bit `imms - immr + width` if imms < immr (wrap), else from `imms - immr`.
            // For the lowered version: shift left to position the sign bit at MSB, then arithmetic-shift right.
            let high_bit = if imms < immr {
                // wrap case (handles ASR-imm and shifts)
                width - 1
            } else {
                imms - immr
            };
            let shl_amt = (width - 1 - high_bit) as u64;
            let amt_c = em.const_u64(shl_amt);
            let shifted_l = em.lsl(bot, amt_c, size);
            let amt2 = em.const_u64(shl_amt);
            em.asr(shifted_l, amt2, size)
        }
        0b01 => {
            // BFM — merges bot into Rd preserving bits outside the field.
            let dst_prev = em.get_gpr(rd, size);
            let clear_mask = !mask;
            let clear_c = em.const_u64(if sf == 0 { clear_mask & 0xFFFF_FFFF } else { clear_mask });
            let cleared = em.and(dst_prev, clear_c, size);
            em.or(cleared, bot, size)
        }
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: inst }),
    };

    em.set_gpr(rd, result, size);
    Ok(InstStatus::Continue)
}

/// EXTR — extract from a pair (also used by ROR-immediate).
fn extract(em: &mut IrEmitter<'_>, inst: u32) -> Result<InstStatus> {
    let sf   = bit(inst, 31);
    let n    = bit(inst, 22);
    let rm   = bits(inst, 16, 5) as u8;
    let imms = bits(inst, 10, 6);
    let rn   = bits(inst, 5, 5) as u8;
    let rd   = bits(inst, 0, 5) as u8;

    if sf != n {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }
    let size = if sf == 1 { RegSize::X } else { RegSize::W };
    let width = if sf == 1 { 64u32 } else { 32 };
    if imms >= width {
        return Err(Error::Decode { pc: em.current_pc, opcode: inst });
    }

    if rn == rm {
        // EXTR Rd, Rn, Rn, #lsb  ==  ROR Rd, Rn, #lsb
        let v = em.get_gpr(rn, size);
        let amt = em.const_u64(imms as u64);
        let res = em.ror(v, amt, size);
        em.set_gpr(rd, res, size);
    } else {
        let hi = em.get_gpr(rn, size);
        let lo = em.get_gpr(rm, size);
        // (hi << (width - imms)) | (lo >> imms)
        let hi_shift = em.const_u64((width - imms) as u64);
        let lo_shift = em.const_u64(imms as u64);
        let hi_part = em.lsl(hi, hi_shift, size);
        let lo_part = em.lsr(lo, lo_shift, size);
        let res = em.or(hi_part, lo_part, size);
        em.set_gpr(rd, res, size);
    }
    Ok(InstStatus::Continue)
}
