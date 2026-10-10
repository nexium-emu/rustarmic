//! FABD's scoped Arm NaN selection; ordinary finite subtraction remains in the
//! existing target-precision native arithmetic path, not a global FSUB rewrite.
use crate::jit::context::CpuContext;

fn special(ctx: &mut CpuContext, n: u64, m: u64, double: bool) -> u64 {
    let (sign, infinity, quiet) = if double {
        (1 << 63, 0x7ff0_0000_0000_0000, 1 << 51)
    } else {
        (1 << 31, 0x7f80_0000, 1 << 22)
    };
    let nan = |bits: u64| bits & !sign > infinity;
    // FPSub processes the original N/M operands, signaling first and N before
    // M; FPAbs clears sign only AFTER this selection. Do not negate M first.
    for value in [n, m] {
        if nan(value) && value & quiet == 0 {
            ctx.fpsr |= 1; // IOC is sticky for a signaling NaN.
            return if ctx.fpcr & (1 << 25) != 0 {
                infinity | quiet
            } else {
                (value | quiet) & !sign
            };
        }
    }
    for value in [n, m] {
        if nan(value) {
            return if ctx.fpcr & (1 << 25) != 0 {
                infinity | quiet
            } else {
                value & !sign
            };
        }
    }
    // Same-sign infinity subtraction is invalid before absolute value.
    if n & !sign == infinity && m == n {
        ctx.fpsr |= 1;
        return infinity | quiet;
    }
    // Defensive numeric fallback. Directed rounding, FZ processing and finite
    // OFC/UFC/IXC retain the broader existing arithmetic backend limitations.
    if double {
        (f64::from_bits(n) - f64::from_bits(m)).to_bits() & !sign
    } else {
        (f32::from_bits(n as u32) - f32::from_bits(m as u32)).to_bits() as u64 & !sign
    }
}

pub(super) unsafe extern "C" fn single(ctx: *mut CpuContext, n: u64, m: u64) -> u64 {
    special(
        unsafe { &mut *ctx },
        n as u32 as u64,
        m as u32 as u64,
        false,
    )
}
pub(super) unsafe extern "C" fn double(ctx: *mut CpuContext, n: u64, m: u64) -> u64 {
    special(unsafe { &mut *ctx }, n, m, true)
}
