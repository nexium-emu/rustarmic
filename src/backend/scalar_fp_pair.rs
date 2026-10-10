use crate::jit::context::CpuContext;

fn shift_jam(value: u64, distance: u32) -> u64 {
    if distance == 0 { value }
    else if distance < 64 { (value >> distance) | u64::from(value << (64 - distance) != 0) }
    else { u64::from(value != 0) }
}

fn add(ctx: &mut CpuContext, mut n: u64, mut m: u64, fraction_bits: u32) -> u64 {
    let sign = 1u64 << (if fraction_bits == 52 { 63 } else { 31 });
    let infinity = if fraction_bits == 52 { 0x7ff0_0000_0000_0000 } else { 0x7f80_0000 };
    let exponent_limit = infinity >> fraction_bits;
    let hidden = 1u64 << fraction_bits;
    let fraction = hidden - 1;
    let rounding = (ctx.fpcr >> 22) & 3;
    if n & !sign == infinity || m & !sign == infinity {
        if n & !sign == infinity && m & !sign == infinity && (n ^ m) & sign != 0 {
            ctx.fpsr |= 1;
            return infinity | (hidden >> 1);
        }
        return if n & !sign == infinity { n } else { m };
    }
    if n & !sign < m & !sign { std::mem::swap(&mut n, &mut m); }
    let result_sign = n & sign;
    let mut exponent = ((n & infinity) >> fraction_bits).max(1);
    let m_exponent = ((m & infinity) >> fraction_bits).max(1);
    let n_sig = ((n & fraction) | if n & infinity != 0 { hidden } else { 0 }) << 3;
    let m_sig = ((m & fraction) | if m & infinity != 0 { hidden } else { 0 }) << 3;
    let m_sig = shift_jam(m_sig, (exponent - m_exponent) as u32);
    let same_sign = (n ^ m) & sign == 0;
    let mut significand = if same_sign { n_sig + m_sig } else { n_sig - m_sig };
    if significand == 0 {
        return if same_sign { result_sign } else if rounding == 2 { sign } else { 0 };
    }
    if significand >= hidden << 4 {
        significand = shift_jam(significand, 1);
        exponent += 1;
    }
    while significand < hidden << 3 && exponent > 1 {
        significand <<= 1;
        exponent -= 1;
    }
    let tiny = exponent == 1 && significand < hidden << 3;
    if tiny && ctx.fpcr & (1 << 24) != 0 {
        ctx.fpsr |= 1 << 3;
        return result_sign;
    }
    let remainder = significand & 7;
    let mut rounded = significand >> 3;
    let increment = match rounding {
        0 => remainder > 4 || (remainder == 4 && rounded & 1 != 0),
        1 => result_sign == 0 && remainder != 0,
        2 => result_sign != 0 && remainder != 0,
        _ => false,
    };
    rounded += u64::from(increment);
    if rounded >= hidden << 1 {
        rounded >>= 1;
        exponent += 1;
    }
    if exponent >= exponent_limit {
        ctx.fpsr |= (1 << 2) | (1 << 4);
        let to_infinity = rounding == 0 || (rounding == 1 && result_sign == 0)
            || (rounding == 2 && result_sign != 0);
        return result_sign | if to_infinity { infinity } else { infinity - 1 };
    }
    if remainder != 0 {
        ctx.fpsr |= 1 << 4;
        if tiny { ctx.fpsr |= 1 << 3; }
    }
    if rounded < hidden { exponent = 0; }
    result_sign | (exponent << fraction_bits) | (rounded & fraction)
}

fn pair(ctx: &mut CpuContext, mut n: u64, mut m: u64, mode: u64, double: bool) -> u64 {
    let (sign, infinity, quiet, fraction_bits) = if double {
        (1u64 << 63, 0x7ff0_0000_0000_0000, 1u64 << 51, 52)
    } else {
        n &= u32::MAX as u64;
        m &= u32::MAX as u64;
        (1u64 << 31, 0x7f80_0000, 1u64 << 22, 23)
    };
    if ctx.fpcr & (1 << 24) != 0 {
        for value in [&mut n, &mut m] {
            if *value & infinity == 0 && *value & !sign != 0 {
                *value &= sign;
                ctx.fpsr |= 1 << 7;
            }
        }
    }
    let nan = |value: u64| value & !sign > infinity;
    for value in [n, m] {
        if nan(value) && value & quiet == 0 {
            ctx.fpsr |= 1;
            return if ctx.fpcr & (1 << 25) != 0 { infinity | quiet } else { value | quiet };
        }
    }
    if nan(n) || nan(m) {
        if mode >= 3 && nan(n) != nan(m) { return if nan(n) { m } else { n }; }
        return if ctx.fpcr & (1 << 25) != 0 { infinity | quiet }
            else { if nan(n) { n } else { m } };
    }
    if mode == 0 { return add(ctx, n, m, fraction_bits); }
    let minimum = u64::from(mode == 2 || mode == 4);
    if double { unsafe { super::fp_minmax_number::double(ctx, n, m, minimum) } }
    else { unsafe { super::fp_minmax_number::single(ctx, n, m, minimum) } }
}

pub(super) unsafe extern "C" fn single(ctx: *mut CpuContext, n: u64, m: u64, mode: u64) -> u64 {
    pair(unsafe { &mut *ctx }, n, m, mode, false)
}

pub(super) unsafe extern "C" fn double(ctx: *mut CpuContext, n: u64, m: u64, mode: u64) -> u64 {
    pair(unsafe { &mut *ctx }, n, m, mode, true)
}
