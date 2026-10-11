//! Advanced SIMD (scalar and vector) instruction semantics for the
//! out-of-line helpers.  Every entry point returns `None` for encodings it
//! does not implement; that decision depends on the encoding only, so the
//! frontend can ask with zero operands at translation time.

use super::fp::{self, Cmp, Fmt, Round, St};

#[inline]
pub fn emask(esize: u32) -> u64 {
    fp::mask(esize)
}

#[inline]
pub fn get(v: u128, i: u32, esize: u32) -> u64 {
    ((v >> (i * esize)) as u64) & emask(esize)
}

#[inline]
pub fn set(v: &mut u128, i: u32, esize: u32, x: u64) {
    let m = (emask(esize) as u128) << (i * esize);
    *v = (*v & !m) | (((x & emask(esize)) as u128) << (i * esize));
}

#[inline]
fn sx(x: u64, esize: u32) -> i128 {
    (((x << (64 - esize)) as i64) >> (64 - esize)) as i128
}

#[inline]
fn zx(x: u64, esize: u32) -> i128 {
    (x & emask(esize)) as i128
}

fn ext(x: u64, esize: u32, unsigned: bool) -> i128 {
    if unsigned { zx(x, esize) } else { sx(x, esize) }
}

fn sat(v: i128, esize: u32, unsigned: bool, st: &mut St) -> u64 {
    let (lo, hi) = if unsigned {
        (0i128, (1i128 << esize) - 1)
    } else {
        (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    };
    if v > hi {
        st.sat();
        hi as u64 & emask(esize)
    } else if v < lo {
        st.sat();
        lo as u64 & emask(esize)
    } else {
        v as u64 & emask(esize)
    }
}

fn ones(b: bool, esize: u32) -> u64 {
    if b { emask(esize) } else { 0 }
}

fn fmt_of(sz: u32) -> Fmt {
    if sz == 1 { Fmt::D } else { Fmt::S }
}

fn bits(x: u32, lsb: u32, len: u32) -> u32 {
    (x >> lsb) & ((1 << len) - 1)
}

fn bit(x: u32, i: u32) -> u32 {
    (x >> i) & 1
}

/// Clears everything above `datasize` bits.
fn clip(v: u128, datasize: u32) -> u128 {
    if datasize >= 128 { v } else { v & ((1u128 << datasize) - 1) }
}

/// Integer shift by a signed register amount (SSHL / USHL / SRSHL /
/// URSHL / SQSHL / UQSHL / SQRSHL / UQRSHL).
fn shl_reg(a: u64, b: u64, esize: u32, unsigned: bool, round: bool, saturate: bool, st: &mut St) -> u64 {
    let shift = (b as u8) as i8 as i32;
    let x = ext(a, esize, unsigned);
    if shift >= 0 {
        let s = shift as u32;
        if saturate {
            if x == 0 {
                return 0;
            }
            if s >= esize {
                return sat(if x < 0 { i128::MIN } else { i128::MAX }, esize, unsigned, st);
            }
            return sat(x << s, esize, unsigned, st);
        }
        if s >= esize {
            return 0;
        }
        return ((x << s) as u64) & emask(esize);
    }
    let s = ((-shift) as u32).min(esize + 1);
    let r = if round { (x + (1i128 << (s - 1))) >> s } else { x >> s };
    if saturate { sat(r, esize, unsigned, st) } else { r as u64 & emask(esize) }
}

fn pmul8(a: u64, b: u64) -> u64 {
    let mut r = 0u64;
    for i in 0..8 {
        if (b >> i) & 1 == 1 {
            r ^= a << i;
        }
    }
    r & 0xFF
}

fn pmull(a: u64, b: u64, esize: u32) -> u128 {
    let mut r = 0u128;
    for i in 0..esize {
        if (b >> i) & 1 == 1 {
            r ^= (a as u128) << i;
        }
    }
    r
}

fn sqdmulh(a: u64, b: u64, esize: u32, round: bool, st: &mut St) -> u64 {
    let p = 2 * sx(a, esize) * sx(b, esize) + if round { 1i128 << (esize - 1) } else { 0 };
    sat(p >> esize, esize, false, st)
}

/// SQRDMLAH / SQRDMLSH element.
fn sqrdmlah(d: u64, a: u64, b: u64, esize: u32, sub: bool, st: &mut St) -> u64 {
    let p = sx(a, esize) * sx(b, esize);
    let p = if sub { -p } else { p };
    let r = (sx(d, esize) << esize) + 2 * p + (1i128 << (esize - 1));
    sat(r >> esize, esize, false, st)
}

/// Integer "three same" element operation; `None` when not an integer op.
fn int3(opcode: u32, u: bool, size: u32, a: u64, b: u64, d: u64, st: &mut St) -> Option<u64> {
    let esize = 8u32 << size;
    let (xa, xb) = (ext(a, esize, u), ext(b, esize, u));
    let m = emask(esize);
    Some(match opcode {
        0b00000 => ((xa + xb) >> 1) as u64 & m,
        0b00001 => sat(xa + xb, esize, u, st),
        0b00010 => ((xa + xb + 1) >> 1) as u64 & m,
        0b00100 => ((xa - xb) >> 1) as u64 & m,
        0b00101 => sat(xa - xb, esize, u, st),
        0b00110 => ones(xa > xb, esize),
        0b00111 => ones(xa >= xb, esize),
        0b01000 => shl_reg(a, b, esize, u, false, false, st),
        0b01001 => shl_reg(a, b, esize, u, false, true, st),
        0b01010 => shl_reg(a, b, esize, u, true, false, st),
        0b01011 => shl_reg(a, b, esize, u, true, true, st),
        0b01100 => (if xa >= xb { a } else { b }) & m,
        0b01101 => (if xa <= xb { a } else { b }) & m,
        0b01110 => (xa - xb).unsigned_abs() as u64 & m,
        0b01111 => d.wrapping_add((xa - xb).unsigned_abs() as u64) & m,
        0b10000 => {
            if u {
                a.wrapping_sub(b) & m
            } else {
                a.wrapping_add(b) & m
            }
        }
        0b10001 => {
            if u {
                ones(a & m == b & m, esize)
            } else {
                ones(a & b & m != 0, esize)
            }
        }
        0b10010 => {
            let p = a.wrapping_mul(b);
            if u { d.wrapping_sub(p) & m } else { d.wrapping_add(p) & m }
        }
        0b10011 => {
            if u {
                if size != 0 {
                    return None;
                }
                pmul8(a, b)
            } else {
                a.wrapping_mul(b) & m
            }
        }
        0b10110 => {
            if size == 0 || size == 3 {
                return None;
            }
            sqdmulh(a, b, esize, u, st)
        }
        _ => return None,
    })
}

/// Pairwise integer reduction of two elements.
fn int_pair(opcode: u32, u: bool, esize: u32, a: u64, b: u64) -> Option<u64> {
    let (xa, xb) = (ext(a, esize, u), ext(b, esize, u));
    Some(match (opcode, u) {
        (0b10100, _) => if xa >= xb { a } else { b },
        (0b10101, _) => if xa <= xb { a } else { b },
        (0b10111, false) => a.wrapping_add(b) & emask(esize),
        _ => return None,
    })
}

/// Floating-point "three same" element operation (`a` = size<1>).
#[allow(clippy::too_many_arguments)]
fn fp3(opcode: u32, u: bool, hi: bool, f: Fmt, a: u64, b: u64, d: u64, st: &mut St) -> Option<u64> {
    let es = f.bits();
    Some(match (u, hi, opcode) {
        (false, false, 0b11000) => fp::fmaxmin(f, a, b, true, true, st),
        (false, true, 0b11000) => fp::fmaxmin(f, a, b, false, true, st),
        (false, false, 0b11001) => fp::fmuladd(f, d, a, b, st),
        (false, true, 0b11001) => fp::fmuladd(f, d, f.neg(a), b, st),
        (false, false, 0b11010) => fp::fadd(f, a, b, st),
        (false, true, 0b11010) => fp::fsub(f, a, b, st),
        (false, false, 0b11011) => fp::fmulx(f, a, b, st),
        (false, false, 0b11100) => ones(fp::fcmp(f, a, b, Cmp::Eq, st), es),
        (false, false, 0b11110) => fp::fmaxmin(f, a, b, true, false, st),
        (false, true, 0b11110) => fp::fmaxmin(f, a, b, false, false, st),
        (false, false, 0b11111) => fp::frecps(f, a, b, st),
        (false, true, 0b11111) => fp::frsqrts(f, a, b, st),
        (true, true, 0b11010) => fp::fabd(f, a, b, st),
        (true, false, 0b11011) => fp::fmul(f, a, b, st),
        (true, false, 0b11100) => ones(fp::fcmp(f, a, b, Cmp::Ge, st), es),
        (true, true, 0b11100) => ones(fp::fcmp(f, a, b, Cmp::Gt, st), es),
        (true, false, 0b11101) => ones(fp::fcmp(f, f.abs(a), f.abs(b), Cmp::Ge, st), es),
        (true, true, 0b11101) => ones(fp::fcmp(f, f.abs(a), f.abs(b), Cmp::Gt, st), es),
        (true, false, 0b11111) => fp::fdiv(f, a, b, st),
        _ => return None,
    })
}

/// Pairwise floating-point operation (`u` must be set in the vector form).
fn fp_pair(opcode: u32, hi: bool, f: Fmt, a: u64, b: u64, st: &mut St) -> Option<u64> {
    Some(match (hi, opcode) {
        (false, 0b11000) => fp::fmaxmin(f, a, b, true, true, st),
        (true, 0b11000) => fp::fmaxmin(f, a, b, false, true, st),
        (false, 0b11010) => fp::fadd(f, a, b, st),
        (false, 0b11110) => fp::fmaxmin(f, a, b, true, false, st),
        (true, 0b11110) => fp::fmaxmin(f, a, b, false, false, st),
        _ => return None,
    })
}

/// AdvSIMD three same, vector (`scalar == false`) or scalar.
pub fn three_same(insn: u32, n: u128, m: u128, d: u128, scalar: bool, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1 || scalar;
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let opcode = bits(insn, 11, 5);
    let datasize = if q { 128 } else { 64 };

    if opcode == 0b00011 {
        if scalar {
            return None;
        }
        // Logical ops (size selects the operation).
        let r = match (u, size) {
            (false, 0) => n & m,
            (false, 1) => n & !m,
            (false, 2) => n | m,
            (false, _) => n | !m,
            (true, 0) => n ^ m,
            (true, 1) => (d & n) | (!d & m),  // BSL: d selects
            (true, 2) => (d & !m) | (n & m),  // BIT
            (true, _) => (d & m) | (n & !m),  // BIF
        };
        return Some(clip(r, datasize));
    }

    if opcode >= 0b11000 {
        // Floating point; size<0> is the precision, size<1> the variant.
        let sz = size & 1;
        let hi = size >> 1 == 1;
        if sz == 1 && !q {
            return None;
        }
        let f = fmt_of(sz);
        let es = f.bits();
        if scalar {
            let ok = matches!(
                (u, hi, opcode),
                (false, false, 0b11011)
                    | (false, false, 0b11100)
                    | (false, false, 0b11111)
                    | (false, true, 0b11111)
                    | (true, false, 0b11100)
                    | (true, false, 0b11101)
                    | (true, true, 0b11010)
                    | (true, true, 0b11100)
                    | (true, true, 0b11101)
            );
            if !ok {
                return None;
            }
            let r = fp3(opcode, u, hi, f, get(n, 0, es), get(m, 0, es), get(d, 0, es), st)?;
            return Some(r as u128);
        }
        let elements = datasize / es;
        if u && matches!((hi, opcode), (false, 0b11000) | (true, 0b11000) | (false, 0b11010) | (false, 0b11110) | (true, 0b11110)) {
            // Pairwise over the concatenation n:m.
            let mut r = 0u128;
            for i in 0..elements {
                let (src, j) = if i < elements / 2 { (n, 2 * i) } else { (m, 2 * (i - elements / 2)) };
                let v = fp_pair(opcode, hi, f, get(src, j, es), get(src, j + 1, es), st)?;
                set(&mut r, i, es, v);
            }
            return Some(r);
        }
        fp3(opcode, u, hi, f, 0, 0, 0, &mut St { fpcr: 0, fpsr: 0 })?;
        let mut r = 0u128;
        for i in 0..elements {
            let v = fp3(opcode, u, hi, f, get(n, i, es), get(m, i, es), get(d, i, es), st)?;
            set(&mut r, i, es, v);
        }
        return Some(r);
    }

    let esize = 8u32 << size;
    if scalar {
        let ok = match opcode {
            0b00001 | 0b00101 | 0b01001 | 0b01011 => true,
            0b00110 | 0b00111 | 0b01000 | 0b01010 | 0b10000 | 0b10001 => size == 3,
            0b10110 => size == 1 || size == 2,
            _ => false,
        };
        if !ok {
            return None;
        }
        let r = int3(opcode, u, size, get(n, 0, esize), get(m, 0, esize), get(d, 0, esize), st)?;
        return Some(r as u128);
    }
    if size == 3 && !q {
        return None;
    }
    if matches!(opcode, 0b10100 | 0b10101 | 0b10111) {
        if size == 3 && opcode != 0b10111 {
            return None;
        }
        let elements = datasize / esize;
        let mut r = 0u128;
        for i in 0..elements {
            let (src, j) = if i < elements / 2 { (n, 2 * i) } else { (m, 2 * (i - elements / 2)) };
            let v = int_pair(opcode, u, esize, get(src, j, esize), get(src, j + 1, esize))?;
            set(&mut r, i, esize, v);
        }
        return Some(r);
    }
    // Ops without a 64-bit element form.
    if size == 3 && matches!(opcode, 0b00000 | 0b00010 | 0b00100 | 0b01100 | 0b01101 | 0b01110 | 0b01111 | 0b10010 | 0b10011) {
        return None;
    }
    int3(opcode, u, size, 0, 0, 0, &mut St { fpcr: 0, fpsr: 0 })?;
    let elements = datasize / esize;
    let mut r = 0u128;
    for i in 0..elements {
        let v = int3(opcode, u, size, get(n, i, esize), get(m, i, esize), get(d, i, esize), st)?;
        set(&mut r, i, esize, v);
    }
    Some(r)
}

fn rbit(x: u64, esize: u32) -> u64 {
    x.reverse_bits() >> (64 - esize)
}

fn cls(x: u64, esize: u32) -> u64 {
    let s = sx(x, esize) as i64;
    let v = if s < 0 { !s } else { s } as u64;
    (v.leading_zeros() - (64 - esize)) as u64 - 1
}

fn clz(x: u64, esize: u32) -> u64 {
    let v = x & emask(esize);
    (v.leading_zeros() - (64 - esize)) as u64
}

/// Rounding mode of an FCVT{N,M,A,P,Z}{S,U} from its opcode bits
/// (`o2` = size<1>, `o1` = opcode<0>), and of FRINT{N,M,P,Z,A}.
fn cvt_round(o2: u32, o1: u32) -> Round {
    match (o2, o1) {
        (0, 0) => Round::TieEven,
        (0, 1) => Round::NegInf,
        (1, 0) => Round::PosInf,
        _ => Round::Zero,
    }
}

/// AdvSIMD two-register miscellaneous, vector or scalar.
pub fn two_reg_misc(insn: u32, n: u128, d: u128, scalar: bool, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1;
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let opcode = bits(insn, 12, 5);
    let datasize = if q || scalar { 128 } else { 64 };
    let esize = 8u32 << size;

    // Narrowing ops write half a register: the upper half for the "2"
    // forms (Q = 1), keeping the lower half of Vd.
    let narrow = |r: u64, st_d: u128| -> u128 {
        if scalar {
            r as u128
        } else if q {
            (st_d & u64::MAX as u128) | ((r as u128) << 64)
        } else {
            r as u128
        }
    };

    // Floating-point part of the space.
    let is_fp = opcode >= 0b11000 || (size >> 1 == 1 && matches!(opcode, 0b01100..=0b01111))
        || matches!(opcode, 0b10110 | 0b10111);
    if is_fp && !matches!(opcode, 0b10110 | 0b10111) {
        let sz = size & 1;
        let hi = size >> 1;
        let f = fmt_of(sz);
        let es = f.bits();
        if !scalar && sz == 1 && !q {
            return None;
        }
        let elements = if scalar { 1 } else { datasize / es };
        let ident = (u, hi, opcode);
        // URECPE / URSQRTE are integer ops in the FP column.
        if matches!(ident, (false, 1, 0b11100) | (true, 1, 0b11100)) {
            if scalar || sz == 1 {
                return None;
            }
            let mut r = 0u128;
            for i in 0..elements {
                let x = get(n, i, 32) as u32;
                set(&mut r, i, 32, if u { fp::ursqrte(x) } else { fp::urecpe(x) } as u64);
            }
            return Some(r);
        }
        let ok = match ident {
            (_, 0, 0b11010) | (_, 0, 0b11011) | (_, 0, 0b11100) | (_, 0, 0b11101) => true,
            (_, 1, 0b11010) | (_, 1, 0b11011) => true,
            (_, 1, 0b01100) | (_, 1, 0b01101) => true,
            (false, 1, 0b01110) => true,
            (_, 1, 0b11101) => true,
            (false, 1, 0b11111) => scalar,
            (_, 1, 0b01111) | (true, 1, 0b11111) => !scalar,
            (_, 0, 0b11000) | (_, 0, 0b11001) | (false, 1, 0b11000) | (false, 1, 0b11001) | (true, 1, 0b11001) => !scalar,
            _ => false,
        };
        if !ok {
            return None;
        }
        let mut r = 0u128;
        for i in 0..elements {
            let x = get(n, i, es);
            let v = match ident {
                (_, 0, 0b11010) | (_, 0, 0b11011) | (_, 1, 0b11010) | (_, 1, 0b11011) => {
                    let mode = cvt_round(hi, opcode & 1);
                    fp::to_fixed(f, x, 0, u, es, mode, st)
                }
                (_, 0, 0b11100) => fp::to_fixed(f, x, 0, u, es, Round::TieAway, st),
                (_, 0, 0b11101) => fp::from_fixed(f, x, 0, u, es, st.rmode(), st),
                (false, 1, 0b01100) => ones(fp::fcmp(f, x, 0, Cmp::Gt, st), es),
                (true, 1, 0b01100) => ones(fp::fcmp(f, x, 0, Cmp::Ge, st), es),
                (false, 1, 0b01101) => ones(fp::fcmp(f, x, 0, Cmp::Eq, st), es),
                (true, 1, 0b01101) => ones(fp::fcmp(f, 0, x, Cmp::Ge, st), es),
                (false, 1, 0b01110) => ones(fp::fcmp(f, 0, x, Cmp::Gt, st), es),
                (false, 1, 0b11101) => fp::frecpe(f, x, st),
                (true, 1, 0b11101) => fp::frsqrte(f, x, st),
                (false, 1, 0b11111) => fp::frecpx(f, x, st),
                (false, 1, 0b01111) => f.abs(x),
                (true, 1, 0b01111) => f.neg(x),
                (true, 1, 0b11111) => fp::fsqrt(f, x, st),
                (false, 0, 0b11000) => fp::round_int(f, x, Round::TieEven, false, st),
                (false, 0, 0b11001) => fp::round_int(f, x, Round::NegInf, false, st),
                (false, 1, 0b11000) => fp::round_int(f, x, Round::PosInf, false, st),
                (false, 1, 0b11001) => fp::round_int(f, x, Round::Zero, false, st),
                (true, 0, 0b11000) => fp::round_int(f, x, Round::TieAway, false, st),
                (true, 0, 0b11001) => {
                    let m = st.rmode();
                    fp::round_int(f, x, m, true, st)
                }
                (true, 1, 0b11001) => {
                    let m = st.rmode();
                    fp::round_int(f, x, m, false, st)
                }
                _ => return None,
            };
            set(&mut r, i, es, v);
        }
        return Some(r);
    }

    match (u, opcode) {
        // FCVTN / FCVTN2 / FCVTXN / FCVTXN2: narrow each element.
        (_, 0b10110) => {
            if size >> 1 == 1 {
                return None;
            }
            let sz = size & 1;
            if u && sz == 0 {
                return None;
            }
            let (from, to, mode) = match (u, sz) {
                (false, 0) => (Fmt::S, Fmt::H, st.rmode()),
                (false, _) => (Fmt::D, Fmt::S, st.rmode()),
                (true, _) => (Fmt::D, Fmt::S, Round::Odd),
            };
            if scalar && !u {
                return None;
            }
            let elements = if scalar { 1 } else { 64 / to.bits() };
            let mut r = 0u64;
            for i in 0..elements {
                let x = get(n, i, from.bits());
                let v = fp::convert(from, to, x, mode, st);
                r |= (v & emask(to.bits())) << (i * to.bits());
            }
            return Some(narrow(r, d));
        }
        // FCVTL / FCVTL2: widen the lower or upper half.
        (false, 0b10111) => {
            if scalar || size >> 1 == 1 {
                return None;
            }
            let (from, to) = if size & 1 == 0 { (Fmt::H, Fmt::S) } else { (Fmt::S, Fmt::D) };
            let half = if q { (n >> 64) as u64 } else { n as u64 };
            let elements = 64 / from.bits();
            let mut r = 0u128;
            for i in 0..elements {
                let x = get(half as u128, i, from.bits());
                let v = fp::convert(from, to, x, st.rmode(), st);
                set(&mut r, i, to.bits(), v);
            }
            return Some(r);
        }
        _ => {}
    }

    // Integer part.
    if scalar {
        let ok = match (u, opcode) {
            (_, 0b00011) | (_, 0b00111) => true,
            (_, 0b01000) | (_, 0b01001) | (false, 0b01010) | (_, 0b01011) => size == 3,
            (_, 0b10100) | (true, 0b10010) => size != 3,
            _ => false,
        };
        if !ok {
            return None;
        }
    }
    let elements = if scalar { 1 } else { datasize / esize };
    let mut r = 0u128;
    match (u, opcode) {
        (false, 0b00000) | (true, 0b00000) | (false, 0b00001) => {
            // REV64 / REV32 / REV16: reverse elements within containers.
            let container = match (u, opcode) {
                (false, 0b00000) => 64,
                (true, 0b00000) => 32,
                _ => 16,
            };
            if esize >= container {
                return None;
            }
            let per = container / esize;
            for i in 0..elements {
                let base = i / per * per;
                let j = base + (per - 1 - (i - base));
                set(&mut r, i, esize, get(n, j, esize));
            }
        }
        (_, 0b00010) | (_, 0b00110) => {
            // SADDLP / UADDLP / SADALP / UADALP
            if size == 3 {
                return None;
            }
            let pairs = datasize / (2 * esize);
            for i in 0..pairs {
                let s = ext(get(n, 2 * i, esize), esize, u) + ext(get(n, 2 * i + 1, esize), esize, u);
                let acc = if opcode == 0b00110 { get(d, i, 2 * esize) as i128 } else { 0 };
                set(&mut r, i, 2 * esize, (s + acc) as u64);
            }
        }
        (_, 0b00011) => {
            // SUQADD (u = 0): signed + unsigned(d)... operands: Vd += Vn.
            if size == 3 && !q && !scalar {
                return None;
            }
            for i in 0..elements {
                let x = get(n, i, esize);
                let acc = get(d, i, esize);
                let v = if u {
                    // USQADD: unsigned accumulator plus signed operand.
                    sat(zx(acc, esize) + sx(x, esize), esize, true, st)
                } else {
                    sat(sx(acc, esize) + zx(x, esize), esize, false, st)
                };
                set(&mut r, i, esize, v);
            }
        }
        (_, 0b00100) => {
            if size == 3 {
                return None;
            }
            for i in 0..elements {
                let x = get(n, i, esize);
                set(&mut r, i, esize, if u { clz(x, esize) } else { cls(x, esize) });
            }
        }
        (false, 0b00101) => {
            if size != 0 {
                return None;
            }
            for i in 0..elements {
                set(&mut r, i, 8, (get(n, i, 8)).count_ones() as u64);
            }
        }
        (true, 0b00101) => match size {
            0 => r = clip(!n, datasize),
            1 => {
                for i in 0..datasize / 8 {
                    set(&mut r, i, 8, rbit(get(n, i, 8), 8));
                }
            }
            _ => return None,
        },
        (_, 0b00111) => {
            if size == 3 && !q && !scalar {
                return None;
            }
            for i in 0..elements {
                let x = sx(get(n, i, esize), esize);
                let v = if u { -x } else { x.abs() };
                set(&mut r, i, esize, sat(v, esize, false, st));
            }
        }
        (_, 0b01000) | (_, 0b01001) | (false, 0b01010) | (_, 0b01011) => {
            if size == 3 && !q && !scalar {
                return None;
            }
            for i in 0..elements {
                let x = sx(get(n, i, esize), esize);
                let v = match (u, opcode) {
                    (false, 0b01000) => ones(x > 0, esize),
                    (true, 0b01000) => ones(x >= 0, esize),
                    (false, 0b01001) => ones(x == 0, esize),
                    (true, 0b01001) => ones(x <= 0, esize),
                    (false, 0b01010) => ones(x < 0, esize),
                    (false, 0b01011) => x.unsigned_abs() as u64,
                    _ => (-x) as u64,
                };
                set(&mut r, i, esize, v);
            }
        }
        (_, 0b10010) | (_, 0b10100) => {
            // XTN / SQXTUN / SQXTN / UQXTN: narrow 2*esize -> esize.
            if size == 3 {
                return None;
            }
            if opcode == 0b10010 && !u && scalar {
                return None;
            }
            let elems = if scalar { 1 } else { 64 / esize };
            let mut out = 0u64;
            for i in 0..elems {
                let x = get(n, i, 2 * esize);
                let v = match (u, opcode) {
                    (false, 0b10010) => x & emask(esize),
                    (true, 0b10010) => sat(sx(x, 2 * esize), esize, true, st),
                    (false, _) => sat(sx(x, 2 * esize), esize, false, st),
                    (true, _) => sat(zx(x, 2 * esize), esize, true, st),
                };
                out |= v << (i * esize);
            }
            return Some(narrow(out, d));
        }
        (true, 0b10011) => {
            // SHLL / SHLL2
            if size == 3 {
                return None;
            }
            let half = if q { n >> 64 } else { n & u64::MAX as u128 };
            for i in 0..64 / esize {
                set(&mut r, i, 2 * esize, get(half, i, esize) << esize);
            }
        }
        _ => return None,
    }
    Some(r)
}

/// AdvSIMD across lanes.
pub fn across_lanes(insn: u32, n: u128, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1;
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let opcode = bits(insn, 12, 5);
    let datasize = if q { 128 } else { 64 };
    if matches!(opcode, 0b01100 | 0b01111) {
        // FMAXNMV / FMINNMV / FMAXV / FMINV (single precision, 4S only).
        if !u || size & 1 == 1 || !q {
            return None;
        }
        let max = size >> 1 == 0;
        let nm = opcode == 0b01100;
        fn reduce(v: &[u64], max: bool, nm: bool, st: &mut St) -> u64 {
            if v.len() == 1 {
                return v[0];
            }
            let h = v.len() / 2;
            let a = reduce(&v[..h], max, nm, st);
            let b = reduce(&v[h..], max, nm, st);
            fp::fmaxmin(Fmt::S, a, b, max, nm, st)
        }
        let e: Vec<u64> = (0..4).map(|i| get(n, i, 32)).collect();
        return Some(reduce(&e, max, nm, st) as u128);
    }
    let esize = 8u32 << size;
    if size == 3 || (size == 2 && !q) {
        return None;
    }
    let elements = datasize / esize;
    let vals: Vec<i128> = (0..elements).map(|i| ext(get(n, i, esize), esize, u)).collect();
    let r: u128 = match (u, opcode) {
        (_, 0b00011) => {
            let s: i128 = vals.iter().sum();
            (s as u128) & emask(2 * esize) as u128
        }
        (_, 0b01010) => *vals.iter().max().unwrap() as u128 & emask(esize) as u128,
        (_, 0b11010) => *vals.iter().min().unwrap() as u128 & emask(esize) as u128,
        (false, 0b11011) => {
            let s: i128 = vals.iter().sum();
            (s as u128) & emask(esize) as u128
        }
        _ => return None,
    };
    Some(r)
}

/// AdvSIMD three different (widening / narrowing / long forms).
pub fn three_different(insn: u32, n: u128, m: u128, d: u128, scalar: bool, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1;
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let opcode = bits(insn, 12, 4);
    if scalar && (u || !matches!(opcode, 0b1001 | 0b1011 | 0b1101) || !(size == 1 || size == 2)) {
        return None;
    }
    let esize = 8u32 << size;
    if opcode == 0b1110 {
        // PMULL / PMULL2 (8-bit, or 64-bit with the crypto extension).
        if u {
            return None;
        }
        let (hn, hm) = if q { (n >> 64, m >> 64) } else { (n, m) };
        return match size {
            0 => {
                let mut r = 0u128;
                for i in 0..8 {
                    set(&mut r, i, 16, pmull(get(hn, i, 8), get(hm, i, 8), 8) as u64);
                }
                Some(r)
            }
            3 => Some(pmull(hn as u64, hm as u64, 64)),
            _ => None,
        };
    }
    if size == 3 {
        return None;
    }
    if (opcode == 0b1001 || opcode == 0b1011 || opcode == 0b1101) && (u || size == 0) {
        return None;
    }
    let half = |v: u128| if q && !scalar { v >> 64 } else { v & u64::MAX as u128 };
    let elements = if scalar { 1 } else { 64 / esize };
    let mut r = 0u128;
    match opcode {
        0b0100 | 0b0110 => {
            // ADDHN / RADDHN / SUBHN / RSUBHN: wide inputs, narrow result.
            let mut out = 0u64;
            for i in 0..elements {
                let a = get(n, i, 2 * esize) as u128;
                let b = get(m, i, 2 * esize) as u128;
                let mut s = if opcode == 0b0100 { a.wrapping_add(b) } else { a.wrapping_sub(b) };
                if u {
                    s = s.wrapping_add(1u128 << (esize - 1));
                }
                out |= ((s >> esize) as u64 & emask(esize)) << (i * esize);
            }
            return Some(if q {
                (d & u64::MAX as u128) | ((out as u128) << 64)
            } else {
                out as u128
            });
        }
        _ => {}
    }
    for i in 0..elements {
        let wide_n = matches!(opcode, 0b0001 | 0b0011);
        let a = if wide_n {
            ext(get(n, i, 2 * esize), 2 * esize, u)
        } else {
            ext(get(half(n), i, esize), esize, u)
        };
        let b = ext(get(half(m), i, esize), esize, u);
        let acc = ext(get(d, i, 2 * esize), 2 * esize, u);
        let w = 2 * esize;
        let v = match opcode {
            0b0000 | 0b0001 => (a + b) as u64,
            0b0010 | 0b0011 => (a - b) as u64,
            0b0101 => (acc + (a - b).abs()) as u64,
            0b0111 => (a - b).unsigned_abs() as u64,
            0b1000 => (acc + a * b) as u64,
            0b1010 => (acc - a * b) as u64,
            0b1100 => (a * b) as u64,
            0b1001 => {
                let p = sat(2 * a * b, w, false, st);
                sat(sx(get(d, i, w), w) + sx(p, w), w, false, st)
            }
            0b1011 => {
                let p = sat(2 * a * b, w, false, st);
                sat(sx(get(d, i, w), w) - sx(p, w), w, false, st)
            }
            0b1101 => sat(2 * a * b, w, false, st),
            _ => return None,
        };
        set(&mut r, i, w, v);
    }
    Some(r)
}

/// AdvSIMD shift by immediate, vector or scalar.
pub fn shift_imm(insn: u32, n: u128, d: u128, scalar: bool, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1;
    let u = bit(insn, 29) == 1;
    let immh = bits(insn, 19, 4);
    let immb = bits(insn, 16, 3);
    let opcode = bits(insn, 11, 5);
    if immh == 0 {
        return None;
    }
    let hsb = 31 - immh.leading_zeros();
    let esize = 8u32 << hsb;
    let imm = (immh << 3) | immb;
    let datasize = if q || scalar { 128 } else { 64 };
    let rshift = 2 * esize - imm;
    let lshift = imm - esize;
    let narrowing = matches!(opcode, 0b10000..=0b10011);
    if narrowing || opcode == 0b10100 {
        if esize == 64 {
            return None;
        }
    } else if !scalar && esize == 64 && !q {
        return None;
    }
    let plain_ok = matches!((u, opcode), (_, 0b00000) | (_, 0b00010) | (_, 0b00100) | (_, 0b00110) | (true, 0b01000) | (_, 0b01010));
    if scalar && plain_ok && esize != 64 {
        return None;
    }
    let elements = if scalar { 1 } else { datasize / esize };
    let mut r = 0u128;
    match (u, opcode) {
        (_, 0b00000) | (_, 0b00010) | (_, 0b00100) | (_, 0b00110) => {
            let round = opcode & 0b00100 != 0;
            let acc = opcode & 0b00010 != 0;
            for i in 0..elements {
                let x = ext(get(n, i, esize), esize, u);
                let mut v = if round { (x + (1i128 << (rshift - 1))) >> rshift } else { x >> rshift } as u64;
                if acc {
                    v = v.wrapping_add(get(d, i, esize));
                }
                set(&mut r, i, esize, v & emask(esize));
            }
        }
        (true, 0b01000) => {
            // SRI
            for i in 0..elements {
                let x = get(n, i, esize);
                let mask = if rshift >= esize { 0 } else { emask(esize) >> rshift };
                let sh = if rshift >= 64 { 0 } else { x >> rshift };
                let v = (get(d, i, esize) & !mask) | (sh & mask);
                set(&mut r, i, esize, v);
            }
        }
        (_, 0b01010) => {
            // SHL / SLI
            for i in 0..elements {
                let x = get(n, i, esize);
                let sh = (x << lshift) & emask(esize);
                let v = if u {
                    let mask = (emask(esize) << lshift) & emask(esize);
                    (get(d, i, esize) & !mask) | sh
                } else {
                    sh
                };
                set(&mut r, i, esize, v);
            }
        }
        (true, 0b01100) | (_, 0b01110) => {
            // SQSHLU / SQSHL / UQSHL (immediate)
            let src_unsigned = u && opcode == 0b01110;
            let dst_unsigned = u;
            for i in 0..elements {
                let x = ext(get(n, i, esize), esize, src_unsigned);
                set(&mut r, i, esize, sat(x << lshift, esize, dst_unsigned, st));
            }
        }
        (_, 0b10000) | (_, 0b10001) | (_, 0b10010) | (_, 0b10011) => {
            // SHRN / RSHRN / SQSHRUN / SQRSHRUN / SQSHRN / UQSHRN / SQRSHRN / UQRSHRN
            let round = opcode & 1 == 1;
            let (src_unsigned, dst_unsigned, saturating) = match (u, opcode >> 1) {
                (false, 0b1000) => (true, true, false), // SHRN / RSHRN (bit pattern only)
                (true, 0b1000) => (false, true, true),  // SQSHRUN / SQRSHRUN
                (false, _) => (false, false, true),     // SQSHRN / SQRSHRN
                (true, _) => (true, true, true),        // UQSHRN / UQRSHRN
            };
            if scalar && !saturating {
                return None;
            }
            let elems = if scalar { 1 } else { 64 / esize };
            let mut out = 0u64;
            for i in 0..elems {
                let x = ext(get(n, i, 2 * esize), 2 * esize, src_unsigned);
                let v = if round { (x + (1i128 << (rshift - 1))) >> rshift } else { x >> rshift };
                let v = if saturating { sat(v, esize, dst_unsigned, st) } else { v as u64 & emask(esize) };
                out |= v << (i * esize);
            }
            return Some(if scalar {
                out as u128
            } else if q {
                (d & u64::MAX as u128) | ((out as u128) << 64)
            } else {
                out as u128
            });
        }
        (_, 0b10100) => {
            // SSHLL / USHLL (SXTL / UXTL)
            if scalar {
                return None;
            }
            let half = if q { n >> 64 } else { n & u64::MAX as u128 };
            for i in 0..64 / esize {
                let x = ext(get(half, i, esize), esize, u);
                set(&mut r, i, 2 * esize, (x << lshift) as u64);
            }
        }
        (_, 0b11100) | (_, 0b11111) => {
            // SCVTF / UCVTF / FCVTZS / FCVTZU (fixed point)
            if esize < 32 {
                return None;
            }
            let f = if esize == 64 { Fmt::D } else { Fmt::S };
            for i in 0..elements {
                let x = get(n, i, esize);
                let v = if opcode == 0b11100 {
                    fp::from_fixed(f, x, rshift, u, esize, st.rmode(), st)
                } else {
                    fp::to_fixed(f, x, rshift, u, esize, Round::Zero, st)
                };
                set(&mut r, i, esize, v);
            }
        }
        _ => return None,
    }
    Some(r)
}

/// AdvSIMD (scalar) x indexed element.
pub fn by_element(insn: u32, n: u128, m: u128, d: u128, scalar: bool, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1;
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let l = bit(insn, 21);
    let mbit = bit(insn, 20);
    let opcode = bits(insn, 12, 4);
    let h = bit(insn, 11);
    let datasize = if q || scalar { 128 } else { 64 };

    let fp_op = matches!((u, opcode), (false, 0b0001) | (false, 0b0101) | (false, 0b1001) | (true, 0b1001));
    if fp_op {
        if size >> 1 == 0 {
            return None;
        }
        let sz = size & 1;
        if sz == 1 && (l == 1 || (!q && !scalar)) {
            return None;
        }
        let f = fmt_of(sz);
        let es = f.bits();
        let index = if sz == 1 { h } else { (h << 1) | l };
        let e2 = get(m, index, es);
        let elements = if scalar { 1 } else { datasize / es };
        let mut r = 0u128;
        for i in 0..elements {
            let e1 = get(n, i, es);
            let acc = get(d, i, es);
            let v = match (u, opcode) {
                (false, 0b0001) => fp::fmuladd(f, acc, e1, e2, st),
                (false, 0b0101) => fp::fmuladd(f, acc, f.neg(e1), e2, st),
                (false, _) => fp::fmul(f, e1, e2, st),
                _ => fp::fmulx(f, e1, e2, st),
            };
            set(&mut r, i, es, v);
        }
        return Some(r);
    }

    // Integer forms.
    let index = match size {
        1 => (h << 2) | (l << 1) | mbit,
        2 => (h << 1) | l,
        _ => return None,
    };
    let esize = 8u32 << size;
    let e2 = get(m, index, esize);
    if scalar {
        let ok = matches!((u, opcode), (false, 0b0011) | (false, 0b0111) | (false, 0b1011) | (false, 0b1100) | (false, 0b1101) | (true, 0b1101) | (true, 0b1111));
        if !ok {
            return None;
        }
    }
    let elements = if scalar { 1 } else { datasize / esize };
    let mut r = 0u128;
    match (u, opcode) {
        (true, 0b0000) | (true, 0b0100) | (false, 0b1000) | (false, 0b1100) | (false, 0b1101) | (true, 0b1101) | (true, 0b1111) => {
            for i in 0..elements {
                let e1 = get(n, i, esize);
                let acc = get(d, i, esize);
                let v = match (u, opcode) {
                    (true, 0b0000) => acc.wrapping_add(e1.wrapping_mul(e2)),
                    (true, 0b0100) => acc.wrapping_sub(e1.wrapping_mul(e2)),
                    (false, 0b1000) => e1.wrapping_mul(e2),
                    (false, 0b1100) => sqdmulh(e1, e2, esize, false, st),
                    (false, 0b1101) => sqdmulh(e1, e2, esize, true, st),
                    (true, 0b1101) => sqrdmlah(acc, e1, e2, esize, false, st),
                    _ => sqrdmlah(acc, e1, e2, esize, true, st),
                };
                set(&mut r, i, esize, v & emask(esize));
            }
        }
        (_, 0b0010) | (_, 0b0110) | (_, 0b1010) | (false, 0b0011) | (false, 0b0111) | (false, 0b1011) => {
            let w = 2 * esize;
            let half = if q && !scalar { n >> 64 } else { n & u64::MAX as u128 };
            let elems = if scalar { 1 } else { 64 / esize };
            let sgn = matches!(opcode, 0b0011 | 0b0111 | 0b1011);
            let uu = u && !sgn;
            let b = ext(e2, esize, uu);
            for i in 0..elems {
                let a = ext(get(half, i, esize), esize, uu);
                let acc = get(d, i, w);
                let v = match opcode {
                    0b0010 => (ext(acc, w, uu) + a * b) as u64,
                    0b0110 => (ext(acc, w, uu) - a * b) as u64,
                    0b1010 => (a * b) as u64,
                    0b0011 => {
                        let p = sat(2 * a * b, w, false, st);
                        sat(sx(acc, w) + sx(p, w), w, false, st)
                    }
                    0b0111 => {
                        let p = sat(2 * a * b, w, false, st);
                        sat(sx(acc, w) - sx(p, w), w, false, st)
                    }
                    _ => sat(2 * a * b, w, false, st),
                };
                set(&mut r, i, w, v);
            }
        }
        (_, 0b1110) => {
            // SDOT / UDOT (by element)
            if size != 2 || scalar {
                return None;
            }
            let idx = (h << 1) | l;
            let rm = get(m, idx, 32);
            for i in 0..datasize / 32 {
                let mut acc = get(d, i, 32) as i64;
                for b in 0..4 {
                    let x = ext(get(n, 4 * i + b, 8), 8, u) as i64;
                    let y = ext((rm >> (8 * b)) & 0xFF, 8, u) as i64;
                    acc = acc.wrapping_add(x * y);
                }
                set(&mut r, i, 32, acc as u64);
            }
        }
        _ => return None,
    }
    Some(r)
}

/// AdvSIMD three same extra: SQRDMLAH / SQRDMLSH, SDOT / UDOT.
pub fn three_same_extra(insn: u32, n: u128, m: u128, d: u128, scalar: bool, st: &mut St) -> Option<u128> {
    let q = bit(insn, 30) == 1;
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let opcode = bits(insn, 11, 4);
    let datasize = if q || scalar { 128 } else { 64 };
    match (u, opcode) {
        (true, 0b0000) | (true, 0b0001) => {
            if size == 0 || size == 3 {
                return None;
            }
            let esize = 8u32 << size;
            let elements = if scalar { 1 } else { datasize / esize };
            let mut r = 0u128;
            for i in 0..elements {
                let v = sqrdmlah(get(d, i, esize), get(n, i, esize), get(m, i, esize), esize, opcode == 1, st);
                set(&mut r, i, esize, v);
            }
            Some(r)
        }
        (_, 0b0010) => {
            if scalar || size != 2 {
                return None;
            }
            let mut r = 0u128;
            for i in 0..datasize / 32 {
                let mut acc = get(d, i, 32) as i64;
                for b in 0..4 {
                    let x = ext(get(n, 4 * i + b, 8), 8, u) as i64;
                    let y = ext(get(m, 4 * i + b, 8), 8, u) as i64;
                    acc = acc.wrapping_add(x * y);
                }
                set(&mut r, i, 32, acc as u64);
            }
            Some(r)
        }
        _ => None,
    }
}

/// AdvSIMD scalar pairwise.
pub fn scalar_pairwise(insn: u32, n: u128, st: &mut St) -> Option<u128> {
    let u = bit(insn, 29) == 1;
    let size = bits(insn, 22, 2);
    let opcode = bits(insn, 12, 5);
    if !u {
        if opcode == 0b11011 && size == 3 {
            return Some((get(n, 0, 64).wrapping_add(get(n, 1, 64))) as u128);
        }
        return None;
    }
    let sz = size & 1;
    let hi = size >> 1 == 1;
    let f = fmt_of(sz);
    let es = f.bits();
    let (a, b) = (get(n, 0, es), get(n, 1, es));
    let r = match (hi, opcode) {
        (false, 0b01100) => fp::fmaxmin(f, a, b, true, true, st),
        (true, 0b01100) => fp::fmaxmin(f, a, b, false, true, st),
        (false, 0b01101) => fp::fadd(f, a, b, st),
        (false, 0b01111) => fp::fmaxmin(f, a, b, true, false, st),
        (true, 0b01111) => fp::fmaxmin(f, a, b, false, false, st),
        _ => return None,
    };
    Some(r as u128)
}

/// Scalar DUP (element): `DUP <V><d>, <Vn>.<T>[<index>]`.
pub fn scalar_dup(insn: u32, n: u128) -> Option<u128> {
    if bit(insn, 29) != 0 || bits(insn, 11, 4) != 0 {
        return None;
    }
    let imm5 = bits(insn, 16, 5);
    if imm5 & 0xF == 0 {
        return None;
    }
    let size = imm5.trailing_zeros();
    let esize = 8u32 << size;
    let index = imm5 >> (size + 1);
    Some(get(n, index, esize) as u128)
}
