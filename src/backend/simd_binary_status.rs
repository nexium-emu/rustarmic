//! Baseline Arm SatQ and FPAbs+FPCompareGE/GT for AdvSIMD binary operations.
//! Arm ISA111181 SQADD/UQADD/SQSUB/UQSUB and FACGE/FACGT. Integer arithmetic
//! cannot overflow i128 for inputs up to 64 bits. FP comparisons inspect exact
//! IEEE encodings, implement FZ input flushing and sticky IOC/IDC, and signal
//! invalid for quiet as well as signaling NaNs. No FEAT_AFP alternate behavior
//! or enabled synchronous FP exception traps are implemented here.

use crate::jit::context::CpuContext;

pub(super) unsafe extern "C" fn saturating_binary(
    ctx: *mut CpuContext,
    n_low: u64,
    n_high: u64,
    m_low: u64,
    m_high: u64,
    mode: u64,
) {
    let width = 8u32 << (mode & 3);
    let unsigned = mode & 4 != 0;
    let subtract = mode & 8 != 0;
    let q = mode & 16 != 0;
    let scalar = mode & 32 != 0;
    let n = n_low as u128 | ((n_high as u128) << 64);
    let m = m_low as u128 | ((m_high as u128) << 64);
    let mask = (1u128 << width) - 1;
    let minimum = if unsigned { 0 } else { -(1i128 << (width - 1)) };
    let maximum = if unsigned { mask as i128 } else { (1i128 << (width - 1)) - 1 };
    let mut result = 0u128;
    let mut saturated = false;
    for lane in 0..if scalar { 1 } else { (if q { 128 } else { 64 }) / width } {
        let value = |input: u128| {
            let raw = (input >> (lane * width)) & mask;
            if unsigned { raw as i128 }
            else { ((raw << (128 - width)) as i128) >> (128 - width) }
        };
        let a = value(n);
        let b = value(m);
        let mathematical = if subtract { a - b } else { a + b };
        let clamped = mathematical.clamp(minimum, maximum);
        saturated |= mathematical != clamped;
        result |= (clamped as u128 & mask) << (lane * width);
    }
    let ctx = unsafe { &mut *ctx };
    ctx.io_value = [result as u64, (result >> 64) as u64];
    if saturated { ctx.fpsr |= 1 << 27; }
}

pub(super) unsafe extern "C" fn absolute_compare(
    ctx: *mut CpuContext,
    n: u64,
    m: u64,
    mode: u64,
) -> u64 {
    let double = mode & 2 != 0;
    let greater = mode & 1 != 0;
    let (sign, exponent, fraction, ones) = if double {
        (1u64 << 63, 0x7ff0_0000_0000_0000, 0x000f_ffff_ffff_ffff, u64::MAX)
    } else {
        (1u64 << 31, 0x7f80_0000, 0x007f_ffff, u32::MAX as u64)
    };
    let ctx = unsafe { &mut *ctx };
    let mut a = n & (ones ^ sign);
    let mut b = m & (ones ^ sign);
    if ctx.fpcr & (1 << 24) != 0 {
        for value in [&mut a, &mut b] {
            if *value & exponent == 0 && *value & fraction != 0 {
                *value = 0;
                ctx.fpsr |= 1 << 7;
            }
        }
    }
    if a > exponent || b > exponent {
        ctx.fpsr |= 1;
        return 0;
    }
    if if greater { a > b } else { a >= b } { ones } else { 0 }
}
