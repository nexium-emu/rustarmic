//! Exact baseline Arm FPMinNum/FPMaxNum selections, without host FP arithmetic.
//! Arm A64 ISA111181 and 100076 D6.64/D6.70: one quiet NaN prefers the
//! number, but signaling NaNs retain N-before-M priority and raise IOC.
//! Implements FPCR.DN/FZ and sticky IOC/IDC, not alternate FEAT_AFP behavior
//! or enabled synchronous FP exception traps. Rounding mode is immaterial.
use crate::jit::context::CpuContext;

pub(super) unsafe extern "C" fn single(ctx: *mut CpuContext, n: u64, m: u64, minimum: u64) -> u64 {
    let ctx = unsafe { &mut *ctx };
    let mut status = 0;
    let result = super::fp_reduce::pair(
        n as u32,
        m as u32,
        minimum != 0,
        true,
        ctx.fpcr,
        &mut status,
    );
    ctx.fpsr |= status;
    result as u64
}

pub(super) unsafe extern "C" fn double(
    ctx: *mut CpuContext,
    mut n: u64,
    mut m: u64,
    minimum: u64,
) -> u64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const INFINITY: u64 = 0x7ff0_0000_0000_0000;
    const FRACTION: u64 = 0x000f_ffff_ffff_ffff;
    const QUIET: u64 = 0x0008_0000_0000_0000;
    let ctx = unsafe { &mut *ctx };
    if ctx.fpcr & (1 << 24) != 0 {
        for value in [&mut n, &mut m] {
            if *value & INFINITY == 0 && *value & FRACTION != 0 {
                *value &= SIGN;
                ctx.fpsr |= 1 << 7;
            }
        }
    }
    let n_nan = n & !SIGN > INFINITY;
    let m_nan = m & !SIGN > INFINITY;
    let n_signaling = n_nan && n & QUIET == 0;
    let m_signaling = m_nan && m & QUIET == 0;
    if n_signaling || m_signaling {
        ctx.fpsr |= 1;
        return if ctx.fpcr & (1 << 25) != 0 {
            INFINITY | QUIET
        } else {
            (if n_signaling { n } else { m }) | QUIET
        };
    }
    if n_nan || m_nan {
        if n_nan != m_nan {
            return if n_nan { m } else { n };
        }
        return if ctx.fpcr & (1 << 25) != 0 {
            INFINITY | QUIET
        } else {
            n
        };
    }
    if n & !SIGN == 0 && m & !SIGN == 0 {
        return if minimum != 0 {
            (n | m) & SIGN
        } else {
            (n & m) & SIGN
        };
    }
    let key = |value: u64| {
        if value & SIGN != 0 {
            !value
        } else {
            value ^ SIGN
        }
    };
    if (key(n) < key(m)) == (minimum != 0) {
        n
    } else {
        m
    }
}
