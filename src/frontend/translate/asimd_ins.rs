use disarm64::decoder::ASIMDINS;

use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::IrEmitter;
use crate::util::bits::{bit, bits};

pub fn translate(em: &mut IrEmitter<'_>, insn: ASIMDINS) -> Result<InstStatus> {
    use ASIMDINS::*;
    match insn {
        DUP_Vd_Rn(i)   => translate_dup_from_gpr(em, i.0),
        DUP_Vd_En(i)   => translate_dup_from_element(em, i.0),
        UMOV_Rd_En(i)  => translate_umov(em, i.0),
        SMOV_Rd_En(i)  => translate_smov(em, i.0),
        INS_Ed_Rn(i)   => translate_ins_from_gpr(em, i.0),
        INS_Ed_En(i)   => translate_ins_from_element(em, i.0),
    }
}

fn decode_imm5(imm5: u32) -> Option<(u32, u32)> {
    if imm5 & 0b00001 != 0 { return Some((0, (imm5 >> 1) & 0xF)); }
    if imm5 & 0b00010 != 0 { return Some((1, (imm5 >> 2) & 0x7)); }
    if imm5 & 0b00100 != 0 { return Some((2, (imm5 >> 3) & 0x3)); }
    if imm5 & 0b01000 != 0 { return Some((3, (imm5 >> 4) & 0x1)); }
    None
}

fn translate_dup_from_gpr(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q     = bit(raw, 30) == 1;
    let imm5  = bits(raw, 16, 5);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let (lane_log2, _) = decode_imm5(imm5)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: raw })?;
    let gpr_size = if lane_log2 == 3 { RegSize::X } else { RegSize::W };
    let gpr_val = em.get_gpr(rn, gpr_size);
    let result = em.vec_dup_gpr(gpr_val, lane_log2, q);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

fn translate_dup_from_element(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q     = bit(raw, 30) == 1;
    let imm5  = bits(raw, 16, 5);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let (lane_log2, src_lane) = decode_imm5(imm5)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: raw })?;

    let src_q = em.get_v_q(rn);
    let scalar = match lane_log2 {
        0 => em.vec_extract_u8(src_q, src_lane),
        1 => em.vec_extract_u16(src_q, src_lane),
        2 => em.vec_extract_u32(src_q, src_lane),
        3 => match src_lane {
            0 => em.vec_extract_lo64(src_q),
            1 => em.vec_extract_hi64(src_q),
            _ => return Err(Error::Decode { pc: em.current_pc, opcode: raw }),
        },
        _ => unreachable!(),
    };
    let result = em.vec_dup_gpr(scalar, lane_log2, q);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

fn translate_umov(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q     = bit(raw, 30) == 1;
    let imm5  = bits(raw, 16, 5);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let (lane_log2, lane) = decode_imm5(imm5)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: raw })?;

    let src_q = em.get_v_q(rn);
    match lane_log2 {
        0 => { let v = em.vec_extract_u8(src_q, lane);  em.set_w(rd, v); }
        1 => { let v = em.vec_extract_u16(src_q, lane); em.set_w(rd, v); }
        2 => { let v = em.vec_extract_u32(src_q, lane); em.set_w(rd, v); }
        3 => {
            if !q { return Err(Error::Decode { pc: em.current_pc, opcode: raw }); }
            let v = match lane {
                0 => em.vec_extract_lo64(src_q),
                1 => em.vec_extract_hi64(src_q),
                _ => return Err(Error::Decode { pc: em.current_pc, opcode: raw }),
            };
            em.set_x(rd, v);
        }
        _ => unreachable!(),
    }
    Ok(InstStatus::Continue)
}

fn translate_smov(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let q     = bit(raw, 30) == 1;
    let imm5  = bits(raw, 16, 5);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let (lane_log2, lane) = decode_imm5(imm5)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: raw })?;

    let src_q = em.get_v_q(rn);
    let raw_val = match lane_log2 {
        0 => em.vec_extract_u8(src_q, lane),
        1 => em.vec_extract_u16(src_q, lane),
        2 => em.vec_extract_u32(src_q, lane),
        _ => return Err(Error::Decode { pc: em.current_pc, opcode: raw }),
    };
    let lane_bits = 8u64 << lane_log2;
    let dst_bits = if q { 64 } else { 32 };
    let dst_size = if q { RegSize::X } else { RegSize::W };
    let shl = em.const_u64(dst_bits - lane_bits);
    let s1 = em.lsl(raw_val, shl, dst_size);
    let shr = em.const_u64(dst_bits - lane_bits);
    let sx = em.asr(s1, shr, dst_size);
    em.set_gpr(rd, sx, dst_size);
    Ok(InstStatus::Continue)
}

fn translate_ins_from_gpr(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let imm5  = bits(raw, 16, 5);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let (lane_log2, lane) = decode_imm5(imm5)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: raw })?;
    let gpr_size = if lane_log2 == 3 { RegSize::X } else { RegSize::W };
    let gpr_val = em.get_gpr(rn, gpr_size);
    let vd_prev = em.get_v_q(rd);
    let result = em.vec_ins_gpr(vd_prev, gpr_val, lane_log2, lane, true);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}

fn translate_ins_from_element(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let imm5  = bits(raw, 16, 5);
    let imm4  = bits(raw, 11, 4);
    let rn    = bits(raw, 5, 5) as u8;
    let rd    = bits(raw, 0, 5) as u8;
    let (lane_log2, dst_lane) = decode_imm5(imm5)
        .ok_or(Error::Decode { pc: em.current_pc, opcode: raw })?;
    let src_lane = imm4 >> lane_log2;

    let src_q = em.get_v_q(rn);
    let scalar = match lane_log2 {
        0 => em.vec_extract_u8(src_q, src_lane),
        1 => em.vec_extract_u16(src_q, src_lane),
        2 => em.vec_extract_u32(src_q, src_lane),
        3 => match src_lane {
            0 => em.vec_extract_lo64(src_q),
            1 => em.vec_extract_hi64(src_q),
            _ => return Err(Error::Decode { pc: em.current_pc, opcode: raw }),
        },
        _ => unreachable!(),
    };
    let vd_prev = em.get_v_q(rd);
    let result = em.vec_ins_gpr(vd_prev, scalar, lane_log2, dst_lane, true);
    em.set_v_q(rd, result);
    Ok(InstStatus::Continue)
}
