//! Soft floating-point support for the instruction helpers: unpacking,
//! Arm NaN propagation, FPCR-controlled rounding, float <-> integer
//! conversions and the reciprocal (square root) estimates.
//!
//! Arithmetic itself uses host IEEE operations (round to nearest even, as
//! the rest of the JIT does); everything that depends on a rounding mode,
//! on Arm-specific NaN rules or on FPCR.FZ / FPCR.DN is modelled here.

/// FPSR cumulative exception bits.
pub const IOC: u32 = 1 << 0;
pub const DZC: u32 = 1 << 1;
pub const OFC: u32 = 1 << 2;
pub const UFC: u32 = 1 << 3;
pub const IXC: u32 = 1 << 4;
pub const IDC: u32 = 1 << 7;
/// FPSR.QC (saturation).
pub const QC: u32 = 1 << 27;

/// Execution state shared by every helper: FPCR in, FPSR accumulated.
pub struct St {
    pub fpcr: u32,
    pub fpsr: u32,
}

impl St {
    pub fn fz(&self) -> bool {
        self.fpcr & (1 << 24) != 0
    }
    pub fn dn(&self) -> bool {
        self.fpcr & (1 << 25) != 0
    }
    pub fn rmode(&self) -> Round {
        match (self.fpcr >> 22) & 3 {
            0 => Round::TieEven,
            1 => Round::PosInf,
            2 => Round::NegInf,
            _ => Round::Zero,
        }
    }
    pub fn sat(&mut self) {
        self.fpsr |= QC;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Round {
    TieEven,
    PosInf,
    NegInf,
    Zero,
    TieAway,
    Odd,
}

/// A binary floating-point format.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fmt {
    H,
    S,
    D,
}

impl Fmt {
    pub const fn bits(self) -> u32 {
        match self {
            Fmt::H => 16,
            Fmt::S => 32,
            Fmt::D => 64,
        }
    }
    pub const fn ebits(self) -> u32 {
        match self {
            Fmt::H => 5,
            Fmt::S => 8,
            Fmt::D => 11,
        }
    }
    pub const fn fbits(self) -> u32 {
        self.bits() - 1 - self.ebits()
    }
    pub const fn bias(self) -> i32 {
        (1 << (self.ebits() - 1)) - 1
    }
    pub const fn emax_field(self) -> u64 {
        (1 << self.ebits()) - 1
    }
    pub const fn sign_bit(self) -> u64 {
        1 << (self.bits() - 1)
    }
    pub const fn frac_mask(self) -> u64 {
        (1 << self.fbits()) - 1
    }
    pub const fn default_nan(self) -> u64 {
        (self.emax_field() << self.fbits()) | (1 << (self.fbits() - 1))
    }
    pub const fn inf(self, sign: bool) -> u64 {
        ((sign as u64) << (self.bits() - 1)) | (self.emax_field() << self.fbits())
    }
    pub const fn zero(self, sign: bool) -> u64 {
        (sign as u64) << (self.bits() - 1)
    }
    pub const fn max_normal(self, sign: bool) -> u64 {
        ((sign as u64) << (self.bits() - 1))
            | ((self.emax_field() - 1) << self.fbits())
            | self.frac_mask()
    }
    pub const fn exp_field(self, x: u64) -> u64 {
        (x >> self.fbits()) & self.emax_field()
    }
    pub const fn sign(self, x: u64) -> bool {
        (x >> (self.bits() - 1)) & 1 != 0
    }
    pub fn is_nan(self, x: u64) -> bool {
        self.exp_field(x) == self.emax_field() && x & self.frac_mask() != 0
    }
    pub fn is_snan(self, x: u64) -> bool {
        self.is_nan(x) && x & (1 << (self.fbits() - 1)) == 0
    }
    pub fn quiet(self, x: u64) -> u64 {
        x | (1 << (self.fbits() - 1))
    }
    pub fn is_denormal(self, x: u64) -> bool {
        self.exp_field(x) == 0 && x & self.frac_mask() != 0
    }
    pub fn is_zero(self, x: u64) -> bool {
        x & !self.sign_bit() & mask(self.bits()) == 0
    }
    pub fn is_inf(self, x: u64) -> bool {
        self.exp_field(x) == self.emax_field() && x & self.frac_mask() == 0
    }
    pub fn abs(self, x: u64) -> u64 {
        x & !self.sign_bit() & mask(self.bits())
    }
    pub fn neg(self, x: u64) -> u64 {
        (x ^ self.sign_bit()) & mask(self.bits())
    }
}

pub const fn mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Zero,
    Normal,
    Inf,
    QNaN,
    SNaN,
}

/// An unpacked operand: `value = (-1)^sign * mant * 2^exp`.
#[derive(Clone, Copy, Debug)]
pub struct Unpacked {
    pub class: Class,
    pub sign: bool,
    pub mant: u64,
    pub exp: i32,
}

/// FPUnpack: classifies `x`, flushing denormal inputs when FPCR.FZ is set.
pub fn unpack(f: Fmt, x: u64, st: &mut St) -> Unpacked {
    let sign = f.sign(x);
    let e = f.exp_field(x);
    let frac = x & f.frac_mask();
    let fb = f.fbits() as i32;
    if e == 0 {
        if frac == 0 || (st.fz() && f != Fmt::H) {
            if frac != 0 {
                st.fpsr |= IDC;
            }
            return Unpacked { class: Class::Zero, sign, mant: 0, exp: 0 };
        }
        return Unpacked {
            class: Class::Normal,
            sign,
            mant: frac,
            exp: 1 - f.bias() - fb,
        };
    }
    if e == f.emax_field() {
        let class = if frac == 0 {
            Class::Inf
        } else if frac >> (fb - 1) & 1 == 1 {
            Class::QNaN
        } else {
            Class::SNaN
        };
        return Unpacked { class, sign, mant: frac, exp: 0 };
    }
    Unpacked {
        class: Class::Normal,
        sign,
        mant: frac | (1 << fb),
        exp: e as i32 - f.bias() - fb,
    }
}

/// FPProcessNaN for one operand.
pub fn process_nan(f: Fmt, x: u64, st: &mut St) -> u64 {
    if f.is_snan(x) {
        st.fpsr |= IOC;
    }
    if st.dn() { f.default_nan() } else { f.quiet(x) }
}

/// FPProcessNaNs: the first signalling NaN, else the first quiet NaN.
pub fn process_nans(f: Fmt, ops: &[u64], st: &mut St) -> Option<u64> {
    if let Some(&x) = ops.iter().find(|&&x| f.is_snan(x)) {
        return Some(process_nan(f, x, st));
    }
    ops.iter()
        .find(|&&x| f.is_nan(x))
        .map(|&x| process_nan(f, x, st))
}

/// Flushes a denormal input to a signed zero under FPCR.FZ.
pub fn flush_in(f: Fmt, x: u64, st: &mut St) -> u64 {
    if f != Fmt::H && st.fz() && f.is_denormal(x) {
        st.fpsr |= IDC;
        f.zero(f.sign(x))
    } else {
        x
    }
}

/// Post-processes a host arithmetic result: Arm's default NaN for
/// invalid operations and FPCR.FZ flushing of denormal results.
pub fn finish(f: Fmt, r: u64, st: &mut St) -> u64 {
    if f.is_nan(r) {
        st.fpsr |= IOC;
        return f.default_nan();
    }
    if st.fz() && f.is_denormal(r) {
        st.fpsr |= UFC;
        return f.zero(f.sign(r));
    }
    r
}

/// Host operations on raw bits.
pub fn host2(f: Fmt, a: u64, b: u64, op: fn(f64, f64) -> f64, op32: fn(f32, f32) -> f32) -> u64 {
    match f {
        Fmt::S => op32(f32::from_bits(a as u32), f32::from_bits(b as u32)).to_bits() as u64,
        _ => op(f64::from_bits(a), f64::from_bits(b)).to_bits(),
    }
}

/// A binary arithmetic operation with Arm NaN rules.
pub fn arith2(
    f: Fmt,
    a: u64,
    b: u64,
    st: &mut St,
    op: fn(f64, f64) -> f64,
    op32: fn(f32, f32) -> f32,
) -> u64 {
    if let Some(n) = process_nans(f, &[a, b], st) {
        return n;
    }
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    let r = host2(f, a, b, op, op32);
    finish(f, r, st)
}

pub fn fadd(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    arith2(f, a, b, st, |x, y| x + y, |x, y| x + y)
}
pub fn fsub(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    arith2(f, a, b, st, |x, y| x - y, |x, y| x - y)
}
pub fn fmul(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    arith2(f, a, b, st, |x, y| x * y, |x, y| x * y)
}
pub fn fdiv(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    arith2(f, a, b, st, |x, y| x / y, |x, y| x / y)
}
pub fn fabd(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    // FPAbs(FPSub(a, b)): the sign is cleared even on a NaN result.
    let d = fsub(f, a, b, st);
    f.abs(d)
}

/// FMULX: FMUL except that infinity times zero is +/-2.0.
pub fn fmulx(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    if let Some(n) = process_nans(f, &[a, b], st) {
        return n;
    }
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    if (f.is_inf(a) && f.is_zero(b)) || (f.is_zero(a) && f.is_inf(b)) {
        let sign = f.sign(a) ^ f.sign(b);
        return two(f, sign);
    }
    finish(f, host2(f, a, b, |x, y| x * y, |x, y| x * y), st)
}

fn two(f: Fmt, sign: bool) -> u64 {
    f.zero(sign) | ((f.bias() as u64 + 1) << f.fbits())
}

fn from_f64(f: Fmt, v: f64) -> u64 {
    match f {
        Fmt::S => (v as f32).to_bits() as u64,
        _ => v.to_bits(),
    }
}

/// FPMulAdd: `addend + a * b` with one rounding.
pub fn fmuladd(f: Fmt, addend: u64, a: u64, b: u64, st: &mut St) -> u64 {
    let inf_times_zero = |a: u64, b: u64| {
        (f.is_inf(a) && f.is_zero(flush_quiet(f, b, st_fz(st)))) || (f.is_zero(flush_quiet(f, a, st_fz(st))) && f.is_inf(b))
    };
    let invalid_qnan_case = f.is_nan(addend) && !f.is_snan(addend) && inf_times_zero(a, b);
    if let Some(n) = process_nans(f, &[addend, a, b], st) {
        if invalid_qnan_case && !f.is_snan(a) && !f.is_snan(b) {
            st.fpsr |= IOC;
            return f.default_nan();
        }
        return n;
    }
    let c = flush_in(f, addend, st);
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    let r = match f {
        Fmt::S => f32::from_bits(a as u32)
            .mul_add(f32::from_bits(b as u32), f32::from_bits(c as u32))
            .to_bits() as u64,
        _ => f64::from_bits(a)
            .mul_add(f64::from_bits(b), f64::from_bits(c))
            .to_bits(),
    };
    finish(f, r, st)
}

fn st_fz(st: &St) -> bool {
    st.fz()
}

fn flush_quiet(f: Fmt, x: u64, fz: bool) -> u64 {
    if fz && f != Fmt::H && f.is_denormal(x) {
        f.zero(f.sign(x))
    } else {
        x
    }
}

/// FRECPS: `2.0 - a * b`, with infinity times zero giving 2.0.
pub fn frecps(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    // FPRecipStepFused negates op1 before NaN propagation.
    if let Some(n) = process_nans(f, &[f.neg(a), b], st) {
        return n;
    }
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    if (f.is_inf(a) && f.is_zero(b)) || (f.is_zero(a) && f.is_inf(b)) {
        return two(f, false);
    }
    let r = match f {
        Fmt::S => (-f32::from_bits(a as u32))
            .mul_add(f32::from_bits(b as u32), 2.0)
            .to_bits() as u64,
        _ => (-f64::from_bits(a)).mul_add(f64::from_bits(b), 2.0).to_bits(),
    };
    finish(f, r, st)
}

/// FRSQRTS: `(3.0 - a * b) / 2`, with infinity times zero giving 1.5.
pub fn frsqrts(f: Fmt, a: u64, b: u64, st: &mut St) -> u64 {
    if let Some(n) = process_nans(f, &[f.neg(a), b], st) {
        return n;
    }
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    if (f.is_inf(a) && f.is_zero(b)) || (f.is_zero(a) && f.is_inf(b)) {
        return from_f64(f, 1.5);
    }
    // (3 - a*b) / 2 with one rounding, computed as 1.5 - (a/2)*b with a
    // fused multiply-add.  Halving an operand is exact unless it is tiny
    // (exponent field below 2); halve the other one then, and if both are
    // tiny the product cannot affect the rounded 1.5.
    let halvable = |x: u64| f.exp_field(x) >= 2;
    let (x, y) = if halvable(a) {
        (half(f, a), b)
    } else if halvable(b) {
        (a, half(f, b))
    } else {
        return from_f64(f, 1.5);
    };
    let r = match f {
        Fmt::S => (-f32::from_bits(x as u32))
            .mul_add(f32::from_bits(y as u32), 1.5)
            .to_bits() as u64,
        _ => (-f64::from_bits(x)).mul_add(f64::from_bits(y), 1.5).to_bits(),
    };
    finish(f, r, st)
}

/// `x / 2` for a value whose exponent field is at least 2 (exact).
fn half(f: Fmt, x: u64) -> u64 {
    if f.is_inf(x) {
        return x;
    }
    x - (1 << f.fbits())
}

/// FPMax / FPMin (`max` selects which), with FMAXNM / FMINNM number
/// preference when `nm`.
pub fn fmaxmin(f: Fmt, a: u64, b: u64, max: bool, nm: bool, st: &mut St) -> u64 {
    let (mut a, mut b) = (a, b);
    if nm {
        // A single quiet NaN loses to any other operand: it becomes the
        // infinity the comparison never picks (-inf for max, +inf for min).
        let qa = f.is_nan(a) && !f.is_snan(a);
        let qb = f.is_nan(b) && !f.is_snan(b);
        if qa && !qb {
            a = f.inf(max);
        } else if !qa && qb {
            b = f.inf(max);
        }
    }
    if let Some(n) = process_nans(f, &[a, b], st) {
        return n;
    }
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    let ua = unpack(f, a, st);
    let ub = unpack(f, b, st);
    if ua.class == Class::Zero && ub.class == Class::Zero && ua.sign != ub.sign {
        // max(+0, -0) = +0, min(+0, -0) = -0
        return f.zero(!max);
    }
    let gt = fgt(f, a, b);
    let pick_a = if max { gt } else { !gt };
    let r = if pick_a { a } else { b };
    // Results are flushed like arithmetic results.
    if st.fz() && f.is_denormal(r) {
        st.fpsr |= UFC;
        return f.zero(f.sign(r));
    }
    r
}

/// Ordered "a > b" on non-NaN values.
pub fn fgt(f: Fmt, a: u64, b: u64) -> bool {
    match f {
        Fmt::S => f32::from_bits(a as u32) > f32::from_bits(b as u32),
        _ => f64::from_bits(a) > f64::from_bits(b),
    }
}

/// FPCompareEQ / GE / GT (unordered compares are false).
pub fn fcmp(f: Fmt, a: u64, b: u64, kind: Cmp, st: &mut St) -> bool {
    if f.is_nan(a) || f.is_nan(b) {
        if kind != Cmp::Eq || f.is_snan(a) || f.is_snan(b) {
            st.fpsr |= IOC;
        }
        return false;
    }
    let a = flush_in(f, a, st);
    let b = flush_in(f, b, st);
    let (x, y) = match f {
        Fmt::S => (f32::from_bits(a as u32) as f64, f32::from_bits(b as u32) as f64),
        _ => (f64::from_bits(a), f64::from_bits(b)),
    };
    match kind {
        Cmp::Eq => x == y,
        Cmp::Ge => x >= y,
        Cmp::Gt => x > y,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ge,
    Gt,
}

/// Rounds `(-1)^sign * mant * 2^exp` (mant != 0) to format `f`.
pub fn round_to(f: Fmt, sign: bool, mant: u128, exp: i32, mode: Round, st: &mut St) -> u64 {
    debug_assert!(mant != 0);
    let fb = f.fbits() as i32;
    let emin = 1 - f.bias();
    let msb = 127 - mant.leading_zeros() as i32;
    let e = msb + exp;
    let tiny = e < emin;
    if tiny && st.fz() && f != Fmt::H {
        st.fpsr |= UFC;
        return f.zero(sign);
    }
    // Scale so that the significand has fb fraction bits.
    let mut scale = if tiny { emin - fb } else { e - fb };
    let shift = scale - exp;
    let (mut sig, half, sticky) = if shift <= 0 {
        (mant << (-shift) as u32, false, false)
    } else if shift >= 128 {
        (0u128, false, true)
    } else {
        let s = shift as u32;
        let kept = mant >> s;
        let rem = mant & ((1u128 << s) - 1);
        let halfbit = 1u128 << (s - 1);
        (kept, rem & halfbit != 0, rem & (halfbit - 1) != 0)
    };
    let inexact = half || sticky;
    let round_up = match mode {
        Round::TieEven => half && (sticky || sig & 1 == 1),
        Round::TieAway => half,
        Round::PosInf => inexact && !sign,
        Round::NegInf => inexact && sign,
        Round::Zero | Round::Odd => false,
    };
    if round_up {
        sig += 1;
    }
    if mode == Round::Odd && inexact {
        sig |= 1;
    }
    if sig >> (fb + 1) != 0 {
        sig >>= 1;
        scale += 1;
    }
    if inexact {
        st.fpsr |= IXC;
        if tiny {
            st.fpsr |= UFC;
        }
    }
    if sig == 0 {
        return f.zero(sign);
    }
    let biased = if sig >> fb == 0 {
        0i64
    } else {
        (scale + fb + f.bias()) as i64
    };
    if biased >= f.emax_field() as i64 {
        st.fpsr |= OFC | IXC;
        let to_inf = match mode {
            Round::TieEven | Round::TieAway => true,
            Round::PosInf => !sign,
            Round::NegInf => sign,
            Round::Zero | Round::Odd => false,
        };
        return if to_inf { f.inf(sign) } else { f.max_normal(sign) };
    }
    f.zero(sign) | ((biased as u64) << fb) | (sig as u64 & f.frac_mask())
}

/// FPConvert between formats (FCVT, FCVTL, FCVTN, FCVTXN).
pub fn convert(from: Fmt, to: Fmt, x: u64, mode: Round, st: &mut St) -> u64 {
    let u = unpack(from, x, st);
    match u.class {
        Class::SNaN | Class::QNaN => {
            if u.class == Class::SNaN {
                st.fpsr |= IOC;
            }
            if st.dn() {
                return to.default_nan();
            }
            // Keep the top fraction bits, quieted.
            let frac = x & from.frac_mask();
            let tf = to.fbits() as i32;
            let ff = from.fbits() as i32;
            let nf = if tf >= ff { frac << (tf - ff) } else { frac >> (ff - tf) };
            to.inf(u.sign) | nf | (1 << (tf - 1))
        }
        Class::Inf => to.inf(u.sign),
        Class::Zero => to.zero(u.sign),
        Class::Normal => round_to(to, u.sign, u.mant as u128, u.exp, mode, st),
    }
}

/// FPToFixed: `x * 2^fbits` rounded with `mode` and saturated to a
/// `width`-bit signed or unsigned integer.
pub fn to_fixed(
    f: Fmt,
    x: u64,
    fbits: u32,
    unsigned: bool,
    width: u32,
    mode: Round,
    st: &mut St,
) -> u64 {
    let u = unpack(f, x, st);
    let (max, min): (i128, i128) = if unsigned {
        ((1i128 << width) - 1, 0)
    } else {
        ((1i128 << (width - 1)) - 1, -(1i128 << (width - 1)))
    };
    let val: i128 = match u.class {
        Class::QNaN | Class::SNaN => {
            st.fpsr |= IOC;
            return 0;
        }
        Class::Inf => {
            st.fpsr |= IOC;
            return (if u.sign { min } else { max }) as u64 & mask(width);
        }
        Class::Zero => 0,
        Class::Normal => {
            let e = u.exp + fbits as i32;
            let mag: i128 = if e >= 0 {
                if e > 70 {
                    i128::MAX
                } else {
                    (u.mant as i128) << e
                }
            } else {
                let s = (-e) as u32;
                let (kept, half, sticky) = if s >= 100 {
                    (0i128, false, true)
                } else {
                    let m = u.mant as i128;
                    let kept = m >> s;
                    let rem = m & ((1i128 << s) - 1);
                    let halfbit = 1i128 << (s - 1);
                    (kept, rem & halfbit != 0, rem & (halfbit - 1) != 0)
                };
                let inexact = half || sticky;
                if inexact {
                    st.fpsr |= IXC;
                }
                let up = match mode {
                    Round::TieEven => half && (sticky || kept & 1 == 1),
                    Round::TieAway => half,
                    Round::PosInf => inexact && !u.sign,
                    Round::NegInf => inexact && u.sign,
                    Round::Zero | Round::Odd => false,
                };
                kept + up as i128
            };
            if u.sign { -mag } else { mag }
        }
    };
    if val > max {
        st.fpsr |= IOC;
        return max as u64 & mask(width);
    }
    if val < min {
        st.fpsr |= IOC;
        return min as u64 & mask(width);
    }
    val as u64 & mask(width)
}

/// FixedToFP: `x / 2^fbits` for a `width`-bit integer.
pub fn from_fixed(f: Fmt, x: u64, fbits: u32, unsigned: bool, width: u32, mode: Round, st: &mut St) -> u64 {
    let x = x & mask(width);
    let (sign, mag) = if unsigned {
        (false, x as u128)
    } else {
        let v = ((x << (64 - width)) as i64) >> (64 - width);
        (v < 0, v.unsigned_abs() as u128)
    };
    if mag == 0 {
        return f.zero(false);
    }
    round_to(f, sign, mag, -(fbits as i32), mode, st)
}

/// FPRoundInt (FRINT*): rounds to an integral value in format `f`.
pub fn round_int(f: Fmt, x: u64, mode: Round, exact: bool, st: &mut St) -> u64 {
    let u = unpack(f, x, st);
    match u.class {
        Class::QNaN | Class::SNaN => process_nan(f, x, st),
        Class::Inf => x,
        Class::Zero => f.zero(u.sign),
        Class::Normal => {
            if u.exp >= 0 {
                return x;
            }
            let s = (-u.exp) as u32;
            let m = u.mant as u128;
            let (kept, half, sticky) = if s >= 100 {
                (0u128, false, m != 0)
            } else {
                let kept = m >> s;
                let rem = m & ((1u128 << s) - 1);
                let halfbit = 1u128 << (s - 1);
                (kept, rem & halfbit != 0, rem & (halfbit - 1) != 0)
            };
            let inexact = half || sticky;
            let up = match mode {
                Round::TieEven => half && (sticky || kept & 1 == 1),
                Round::TieAway => half,
                Round::PosInf => inexact && !u.sign,
                Round::NegInf => inexact && u.sign,
                Round::Zero | Round::Odd => false,
            };
            let r = kept + up as u128;
            if exact && inexact {
                st.fpsr |= IXC;
            }
            if r == 0 {
                return f.zero(u.sign);
            }
            round_to(f, u.sign, r, 0, Round::TieEven, st)
        }
    }
}

/// RecipEstimate on a 9-bit input in [256, 511].
fn recip_estimate(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1u64 << 19) / a;
    b.div_ceil(2)
}

/// RecipSqrtEstimate on a 9-bit input in [128, 511].
fn recip_sqrt_estimate(a: u64) -> u64 {
    let a = if a < 256 {
        // 0.25 <= a < 0.5, in units of 1/512 rounded to nearest
        a * 2 + 1
    } else {
        // 0.5 <= a < 1.0: drop the bottom bit, units of 1/256 rounded
        (((a >> 1) << 1) + 1) * 2
    };
    let mut b = 512u64;
    while a * (b + 1) * (b + 1) < (1u64 << 28) {
        b += 1;
    }
    b.div_ceil(2)
}

/// FRECPE (FPRecipEstimate).
pub fn frecpe(f: Fmt, x: u64, st: &mut St) -> u64 {
    let u = unpack(f, x, st);
    match u.class {
        Class::QNaN | Class::SNaN => return process_nan(f, x, st),
        Class::Inf => return f.zero(u.sign),
        Class::Zero => {
            st.fpsr |= DZC;
            return f.inf(u.sign);
        }
        Class::Normal => {}
    }
    let abs = f.abs(x);
    // abs(value) < 2^-(bias+1): the reciprocal overflows.
    let tiny = match f {
        Fmt::S => abs < 0x0020_0000, // 2^-128
        _ => abs < 0x0004_0000_0000_0000, // 2^-1024
    };
    if tiny {
        st.fpsr |= OFC | IXC;
        let to_inf = match st.rmode() {
            Round::TieEven => true,
            Round::PosInf => !u.sign,
            Round::NegInf => u.sign,
            _ => false,
        };
        return if to_inf { f.inf(u.sign) } else { f.max_normal(u.sign) };
    }
    let huge = match f {
        Fmt::S => abs >= 0x7E80_0000, // 2^126
        _ => abs >= 0x7FD0_0000_0000_0000, // 2^1022
    };
    if st.fz() && huge {
        st.fpsr |= UFC;
        return f.zero(u.sign);
    }
    let (mut fraction, mut exp): (u64, i64) = match f {
        Fmt::S => ((x & 0x7F_FFFF) << 29, ((x >> 23) & 0xFF) as i64),
        _ => (x & 0xF_FFFF_FFFF_FFFF, ((x >> 52) & 0x7FF) as i64),
    };
    if exp == 0 {
        if (fraction >> 51) & 1 == 0 {
            exp = -1;
            fraction = (fraction << 2) & 0xF_FFFF_FFFF_FFFF;
        } else {
            fraction = (fraction << 1) & 0xF_FFFF_FFFF_FFFF;
        }
    }
    let scaled = (1 << 8) | ((fraction >> 44) & 0xFF);
    let mut result_exp = match f {
        Fmt::S => 253 - exp,
        _ => 2045 - exp,
    };
    let estimate = recip_estimate(scaled);
    let mut fraction = (estimate & 0xFF) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    match f {
        Fmt::S => {
            ((u.sign as u64) << 31) | (((result_exp as u64) & 0xFF) << 23) | ((fraction >> 29) & 0x7F_FFFF)
        }
        _ => ((u.sign as u64) << 63) | (((result_exp as u64) & 0x7FF) << 52) | (fraction & 0xF_FFFF_FFFF_FFFF),
    }
}

/// FRSQRTE (FPRSqrtEstimate).
pub fn frsqrte(f: Fmt, x: u64, st: &mut St) -> u64 {
    let u = unpack(f, x, st);
    match u.class {
        Class::QNaN | Class::SNaN => return process_nan(f, x, st),
        Class::Zero => {
            st.fpsr |= DZC;
            return f.inf(u.sign);
        }
        _ if u.sign => {
            st.fpsr |= IOC;
            return f.default_nan();
        }
        Class::Inf => return f.zero(false),
        Class::Normal => {}
    }
    let (mut fraction, mut exp): (u64, i64) = match f {
        Fmt::S => ((x & 0x7F_FFFF) << 29, ((x >> 23) & 0xFF) as i64),
        _ => (x & 0xF_FFFF_FFFF_FFFF, ((x >> 52) & 0x7FF) as i64),
    };
    if exp == 0 {
        while (fraction >> 51) & 1 == 0 {
            fraction = (fraction << 1) & 0xF_FFFF_FFFF_FFFF;
            exp -= 1;
        }
        fraction = (fraction << 1) & 0xF_FFFF_FFFF_FFFF;
    }
    let scaled = if exp & 1 == 0 {
        (1 << 8) | ((fraction >> 44) & 0xFF)
    } else {
        (1 << 7) | ((fraction >> 45) & 0x7F)
    };
    let result_exp = match f {
        Fmt::S => (380 - exp).div_euclid(2),
        _ => (3068 - exp).div_euclid(2),
    };
    let estimate = recip_sqrt_estimate(scaled);
    match f {
        Fmt::S => (((result_exp as u64) & 0xFF) << 23) | ((estimate & 0xFF) << 15),
        _ => (((result_exp as u64) & 0x7FF) << 52) | ((estimate & 0xFF) << 44),
    }
}

/// FRECPX.
pub fn frecpx(f: Fmt, x: u64, st: &mut St) -> u64 {
    if f.is_nan(x) {
        return process_nan(f, x, st);
    }
    let e = f.exp_field(x);
    let ne = if e == 0 {
        f.emax_field() - 1
    } else {
        !e & f.emax_field()
    };
    f.zero(f.sign(x)) | (ne << f.fbits())
}

/// FSQRT.
pub fn fsqrt(f: Fmt, x: u64, st: &mut St) -> u64 {
    if f.is_nan(x) {
        return process_nan(f, x, st);
    }
    let x = flush_in(f, x, st);
    if f.is_zero(x) {
        return x;
    }
    if f.sign(x) {
        st.fpsr |= IOC;
        return f.default_nan();
    }
    let r = match f {
        Fmt::S => f32::from_bits(x as u32).sqrt().to_bits() as u64,
        _ => f64::from_bits(x).sqrt().to_bits(),
    };
    finish(f, r, st)
}

/// Unsigned reciprocal estimate (URECPE) on a 32-bit element.
pub fn urecpe(x: u32) -> u32 {
    if x >> 31 == 0 {
        return 0xFFFF_FFFF;
    }
    let est = recip_estimate(((x >> 23) & 0x1FF) as u64);
    (est as u32) << 23
}

/// Unsigned reciprocal square root estimate (URSQRTE).
pub fn ursqrte(x: u32) -> u32 {
    if x >> 30 == 0 {
        return 0xFFFF_FFFF;
    }
    let est = recip_sqrt_estimate(((x >> 23) & 0x1FF) as u64);
    (est as u32) << 23
}
