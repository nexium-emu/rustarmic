#[inline(always)]
pub const fn bits(value: u32, lsb: u32, len: u32) -> u32 {
    (value >> lsb) & ((1u32 << len) - 1)
}

#[inline(always)]
pub const fn bit(value: u32, idx: u32) -> u32 {
    (value >> idx) & 1
}

#[inline(always)]
pub const fn sign_extend(value: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((value << shift) as i64) >> shift
}

pub fn decode_bit_masks(n: u32, imms: u32, immr: u32, width: u32) -> Option<u64> {
    unsafe { core::hint::assert_unchecked(width == 32 || width == 64) };
    let combined = (n << 6) | (!imms & 0x3f);
    let len = 31u32.checked_sub(combined.leading_zeros())?;
    if len < 1 {
        return None;
    }
    if width < (1u32 << len) {
        return None;
    }

    let levels = (1u32 << len) - 1;
    let s = imms & levels;
    let r = immr & levels;

    if s == levels {
        return None;
    }

    let esize = 1u32 << len;
    unsafe { core::hint::assert_unchecked(esize <= width) };
    unsafe { core::hint::assert_unchecked(esize.is_power_of_two() && esize >= 2) };

    let welem: u64 = (1u64 << (s + 1)) - 1;
    let welem_rotated: u64 = if r == 0 {
        welem
    } else {
        let mask = if esize == 64 {
            u64::MAX
        } else {
            (1u64 << esize) - 1
        };
        ((welem >> r) | (welem << (esize - r))) & mask
    };

    let mut out: u64 = 0;
    let mut filled = 0u32;
    let chunk = if esize == 64 {
        u64::MAX
    } else {
        (1u64 << esize) - 1
    };
    while filled < width {
        out |= (welem_rotated & chunk) << filled;
        filled += esize;
    }
    if width == 32 {
        out &= 0xFFFF_FFFF;
    }
    Some(out)
}
