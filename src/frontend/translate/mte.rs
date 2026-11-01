use crate::arch::RegSize;
use crate::error::{Error, Result};
use crate::frontend::translator::InstStatus;
use crate::ir::{IrEmitter, ValueRef};
use crate::util::bits::{bit, bits, sign_extend};

const TAG_SHIFT: u64 = 56;
const TAG_GRANULE: u64 = 16;

fn without_tag(em: &mut IrEmitter<'_>, value: ValueRef) -> ValueRef {
    let mask = em.const_u64(!(0xF << TAG_SHIFT));
    em.and(value, mask, RegSize::X)
}

fn sign_extend_56(em: &mut IrEmitter<'_>, value: ValueRef) -> ValueRef {
    let shift = em.const_u64(64 - TAG_SHIFT);
    let high = em.lsl(value, shift, RegSize::X);
    let shift = em.const_u64(64 - TAG_SHIFT);
    em.asr(high, shift, RegSize::X)
}

fn unsupported(em: &IrEmitter<'_>) -> Result<InstStatus> {
    Err(Error::Unsupported {
        pc: em.current_pc,
        opcode: 0,
    })
}

pub fn tag_store(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let opc = bits(raw, 22, 2);
    let mode = bits(raw, 10, 2);
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let offset = (sign_extend(bits(raw, 12, 9) as u64, 9) as u64) << 4;
    if opc == 0b01 && mode == 0b00 {
        let value = em.get_x(rt);
        let value = without_tag(em, value);
        em.set_x(rt, value);
        return Ok(InstStatus::Continue);
    }
    if mode == 0b00 {
        return unsupported(em);
    }
    let base = em.get_x_or_sp(rn, true);
    let delta = em.const_u64(offset);
    let offset_address = em.add(base, delta, RegSize::X);
    let address = if mode == 0b01 { base } else { offset_address };
    let zeroed = match opc {
        0b01 => TAG_GRANULE,
        0b11 => 2 * TAG_GRANULE,
        _ => 0,
    };
    let zero = em.const_u64(0);
    for chunk in 0..zeroed / 8 {
        let delta = em.const_u64(chunk * 8);
        let at = em.add(address, delta, RegSize::X);
        em.store(at, zero, 8);
    }
    if mode != 0b10 {
        em.set_x_or_sp(rn, offset_address, true);
    }
    Ok(InstStatus::Continue)
}

pub fn store_pair(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let mode = bits(raw, 23, 2);
    let rt2 = bits(raw, 10, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rt = bits(raw, 0, 5) as u8;
    let offset = (sign_extend(bits(raw, 15, 7) as u64, 7) as u64) << 4;
    let first = em.get_x(rt);
    let second = em.get_x(rt2);
    let base = em.get_x_or_sp(rn, true);
    let delta = em.const_u64(offset);
    let offset_address = em.add(base, delta, RegSize::X);
    let address = if mode == 0b01 { base } else { offset_address };
    em.store(address, first, 8);
    let eight = em.const_u64(8);
    let high = em.add(address, eight, RegSize::X);
    em.store(high, second, 8);
    if mode != 0b10 {
        em.set_x_or_sp(rn, offset_address, true);
    }
    Ok(InstStatus::Continue)
}

pub fn data_processing(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let set_flags = bit(raw, 29) == 1;
    let rm = bits(raw, 16, 5) as u8;
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    match (bits(raw, 10, 6), set_flags) {
        (0b000100, false) => {
            let address = em.get_x_or_sp(rn, true);
            let tagged = without_tag(em, address);
            em.set_x_or_sp(rd, tagged, true);
        }
        (0b000101, false) => {
            let address = em.get_x_or_sp(rn, true);
            let shift = em.const_u64(TAG_SHIFT);
            let tag = em.lsr(address, shift, RegSize::X);
            let mask = em.const_u64(0xF);
            let tag = em.and(tag, mask, RegSize::X);
            let one = em.const_u64(1);
            let tag_bit = em.lsl(one, tag, RegSize::X);
            let excluded = em.get_x(rm);
            let result = em.or(excluded, tag_bit, RegSize::X);
            em.set_x(rd, result);
        }
        (0b000000, _) => {
            let first = em.get_x_or_sp(rn, true);
            let first = sign_extend_56(em, first);
            let second = em.get_x_or_sp(rm, true);
            let second = sign_extend_56(em, second);
            let result = if set_flags {
                em.subs(first, second, RegSize::X)
            } else {
                em.sub(first, second, RegSize::X)
            };
            em.set_x(rd, result);
        }
        _ => return unsupported(em),
    }
    Ok(InstStatus::Continue)
}

pub fn add_tagged(em: &mut IrEmitter<'_>, raw: u32) -> Result<InstStatus> {
    let rn = bits(raw, 5, 5) as u8;
    let rd = bits(raw, 0, 5) as u8;
    let base = em.get_x_or_sp(rn, true);
    let delta = em.const_u64((bits(raw, 16, 6) as u64) << 4);
    let moved = if bit(raw, 30) == 1 {
        em.sub(base, delta, RegSize::X)
    } else {
        em.add(base, delta, RegSize::X)
    };
    let tagged = without_tag(em, moved);
    em.set_x_or_sp(rd, tagged, true);
    Ok(InstStatus::Continue)
}
