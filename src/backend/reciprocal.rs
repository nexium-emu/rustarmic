//! Arm's quantized reciprocal / reciprocal-square-root estimates and steps.
//! Algorithms follow Arm FPRecipEstimate/RecipEstimate and FPRecipStepFused,
//! also implemented in Dynarmic's 0BSD common/math_util.cpp and common/fp/op.
//! Estimates are architectural integer quantizers, not host RCP/RSQRT.
use crate::jit::context::CpuContext;

const IOC: u32 = 1;
const DZC: u32 = 2;
const OFC: u32 = 1 << 2;
const UFC: u32 = 1 << 3;
const IXC: u32 = 1 << 4;
const IDC: u32 = 1 << 7;

const fn estimate_table() -> [u8; 512] {
    let mut table = [0; 512];
    let mut i = 128usize;
    while i < 512 {
        let a = if i < 256 { 2 * i + 1 } else { 2 * (i | 1) } as u64;
        let mut b = 512u64;
        while a * (b + 1) * (b + 1) < (1 << 28) {
            b += 1;
        }
        table[i] = ((b + 1) / 2) as u8;
        i += 1;
    }
    table
}
const ESTIMATE: [u8; 512] = estimate_table();

const fn reciprocal_table() -> [u8; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let midpoint = 2 * (i as u64 + 256) + 1;
        // Both divisions are architectural integer round-to-nearest stages.
        // The implicit ninth result bit is restored by exponent assembly.
        table[i] = (((1 << 19) / midpoint + 1) / 2) as u8;
        i += 1;
    }
    table
}
const RECIPROCAL_ESTIMATE: [u8; 256] = reciprocal_table();

#[derive(Clone, Copy)]
struct Format {
    fraction: u32,
    bias: i32,
    infinity: u64,
    sign: u64,
}
const S: Format = Format {
    fraction: 23,
    bias: 127,
    infinity: 0x7f80_0000,
    sign: 1 << 31,
};
const D: Format = Format {
    fraction: 52,
    bias: 1023,
    infinity: 0x7ff0_0000_0000_0000,
    sign: 1 << 63,
};
impl Format {
    fn fraction_mask(self) -> u64 {
        (1 << self.fraction) - 1
    }
    fn quiet(self) -> u64 {
        1 << (self.fraction - 1)
    }
    fn default_nan(self) -> u64 {
        self.infinity | self.quiet()
    }
    fn nan(self, value: u64) -> bool {
        value & !self.sign > self.infinity
    }
    fn signaling(self, value: u64) -> bool {
        self.nan(value) && value & self.quiet() == 0
    }
}

fn input(ctx: &mut CpuContext, bits: u64, f: Format) -> u64 {
    if ctx.fpcr & (1 << 24) != 0 && bits & !f.sign != 0 && bits & !f.sign < 1 << f.fraction {
        ctx.fpsr |= IDC;
        bits & f.sign
    } else {
        bits
    }
}

fn process_nan(ctx: &mut CpuContext, bits: u64, f: Format) -> u64 {
    if f.signaling(bits) {
        ctx.fpsr |= IOC;
    }
    if ctx.fpcr & (1 << 25) != 0 {
        f.default_nan()
    } else {
        bits | f.quiet()
    }
}

fn estimate(ctx: &mut CpuContext, bits: u64, f: Format) -> u64 {
    let bits = input(ctx, bits, f);
    let magnitude = bits & !f.sign;
    if f.nan(bits) {
        return process_nan(ctx, bits, f);
    }
    if magnitude == 0 {
        ctx.fpsr |= DZC;
        return f.infinity | (bits & f.sign);
    }
    if bits & f.sign != 0 {
        ctx.fpsr |= IOC;
        return f.default_nan();
    }
    if magnitude == f.infinity {
        return 0;
    }
    let exponent_field = (bits >> f.fraction) as i32;
    let mut exponent = exponent_field - f.bias;
    let mut significand = bits & f.fraction_mask();
    if exponent_field == 0 {
        exponent = 1 - f.bias;
        while significand & (1 << f.fraction) == 0 {
            significand <<= 1;
            exponent -= 1;
        }
    } else {
        significand |= 1 << f.fraction;
    }
    let scaled = significand >> (f.fraction - if exponent & 1 == 0 { 7 } else { 8 });
    let result_exponent = (-(exponent + 1) >> 1) + f.bias;
    ((result_exponent as u64) << f.fraction)
        | ((ESTIMATE[scaled as usize] as u64) << (f.fraction - 8))
}

fn reciprocal_estimate(ctx: &mut CpuContext, bits: u64, f: Format) -> u64 {
    let bits = input(ctx, bits, f);
    let sign = bits & f.sign;
    let magnitude = bits & !f.sign;
    if f.nan(bits) {
        return process_nan(ctx, bits, f);
    }
    if magnitude == f.infinity {
        return sign;
    }
    if magnitude == 0 {
        ctx.fpsr |= DZC;
        return sign | f.infinity;
    }
    let exponent_field = (magnitude >> f.fraction) as i32;
    let mut exponent = exponent_field - f.bias;
    let mut significand = magnitude & f.fraction_mask();
    if exponent_field == 0 {
        exponent = 1 - f.bias;
        while significand & (1 << f.fraction) == 0 {
            significand <<= 1;
            exponent -= 1;
        }
    } else {
        significand |= 1 << f.fraction;
    }
    // Strict |x| < 2^-128 (S) / 2^-1024 (D), before quantization.
    if exponent < -f.bias - 1 {
        ctx.fpsr |= OFC | IXC;
        let mode = (ctx.fpcr >> 22) & 3;
        let overflow_to_infinity =
            mode == 0 || (mode == 1 && sign == 0) || (mode == 2 && sign != 0);
        return sign
            | if overflow_to_infinity {
                f.infinity
            } else {
                f.infinity - 1
            };
    }
    // FZ flushes an estimated subnormal result with UFC, not IXC. With FZ
    // disabled, the architecture directly constructs its subnormal estimate.
    if ctx.fpcr & (1 << 24) != 0 && exponent >= f.bias - 1 {
        ctx.fpsr |= UFC;
        return sign;
    }
    let scaled = significand >> (f.fraction - 8);
    let mut fraction = (RECIPROCAL_ESTIMATE[(scaled - 256) as usize] as u64) << (f.fraction - 8);
    let mut result_exponent = -(exponent + 1) + f.bias;
    if result_exponent <= 0 {
        fraction |= 1 << f.fraction;
        fraction >>= 1 - result_exponent;
        result_exponent = 0;
    }
    sign | ((result_exponent as u64) << f.fraction) | fraction
}

fn step(ctx: &mut CpuContext, n: u64, m: u64, f: Format, square_root: bool) -> u64 {
    // Arm negates operand1 before signaling-first NaN selection, including its
    // sign bit. Preserve this ordering rather than relying on host FMA NaNs.
    let n = input(ctx, n ^ f.sign, f);
    let m = input(ctx, m, f);
    for value in [n, m] {
        if f.signaling(value) {
            return process_nan(ctx, value, f);
        }
    }
    for value in [n, m] {
        if f.nan(value) {
            return process_nan(ctx, value, f);
        }
    }
    let na = n & !f.sign;
    let ma = m & !f.sign;
    let constant = if square_root {
        ((f.bias as u64) << f.fraction) | (1 << (f.fraction - 1))
    } else {
        ((f.bias + 1) as u64) << f.fraction
    };
    if (na == f.infinity && ma == 0) || (ma == f.infinity && na == 0) {
        return constant;
    }
    if na == f.infinity || ma == f.infinity {
        return f.infinity | ((n ^ m) & f.sign);
    }
    // FRECPS is ONE target-precision fused operation: 2 + (-N)*M.
    // As for FRSQRTS below, this finite path currently provides default RNE,
    // not directed FPCR rounding or full arithmetic FPSR OFC/UFC/IXC flags.
    // DN, input FZ/IDC, and signaling-NaN IOC are handled above for both steps.
    if !square_root {
        return if f.fraction == 23 {
            f32::from_bits(n as u32)
                .mul_add(f32::from_bits(m as u32), 2.0)
                .to_bits() as u64
        } else {
            f64::from_bits(n).mul_add(f64::from_bits(m), 2.0).to_bits()
        };
    }
    // Exactly halve the greater-magnitude operand while it remains normal,
    // then perform ONE target-precision FMA. This computes (3-N*M)/2 without
    // premature overflow or double rounding. If both exponent fields <=1,
    // their product is far below half an ULP of 1.5: default RNE returns1.5.
    // Directed FPCR rounding / complete arithmetic FPSR flags remain outside
    // this backend's current FP contract (as for its other arithmetic ops).
    let (a, b) = if na >= ma { (n, m) } else { (m, n) };
    if a & !f.sign < 2 << f.fraction {
        return constant;
    }
    let half = a - (1 << f.fraction);
    if f.fraction == 23 {
        f32::from_bits(half as u32)
            .mul_add(f32::from_bits(b as u32), 1.5)
            .to_bits() as u64
    } else {
        f64::from_bits(half)
            .mul_add(f64::from_bits(b), 1.5)
            .to_bits()
    }
}

pub(super) unsafe extern "C" fn estimate_s(ctx: *mut CpuContext, n: u64) -> u64 {
    estimate(unsafe { &mut *ctx }, n as u32 as u64, S)
}
pub(super) unsafe extern "C" fn estimate_d(ctx: *mut CpuContext, n: u64) -> u64 {
    estimate(unsafe { &mut *ctx }, n, D)
}
pub(super) unsafe extern "C" fn step_s(ctx: *mut CpuContext, n: u64, m: u64) -> u64 {
    step(
        unsafe { &mut *ctx },
        n as u32 as u64,
        m as u32 as u64,
        S,
        true,
    )
}
pub(super) unsafe extern "C" fn step_d(ctx: *mut CpuContext, n: u64, m: u64) -> u64 {
    step(unsafe { &mut *ctx }, n, m, D, true)
}

pub(super) unsafe extern "C" fn reciprocal_estimate_s(ctx: *mut CpuContext, n: u64) -> u64 {
    reciprocal_estimate(unsafe { &mut *ctx }, n as u32 as u64, S)
}
pub(super) unsafe extern "C" fn reciprocal_estimate_d(ctx: *mut CpuContext, n: u64) -> u64 {
    reciprocal_estimate(unsafe { &mut *ctx }, n, D)
}
pub(super) unsafe extern "C" fn reciprocal_step_s(ctx: *mut CpuContext, n: u64, m: u64) -> u64 {
    step(
        unsafe { &mut *ctx },
        n as u32 as u64,
        m as u32 as u64,
        S,
        false,
    )
}
pub(super) unsafe extern "C" fn reciprocal_step_d(ctx: *mut CpuContext, n: u64, m: u64) -> u64 {
    step(unsafe { &mut *ctx }, n, m, D, false)
}
