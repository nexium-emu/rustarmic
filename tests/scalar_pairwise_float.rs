use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const STICKY: u32 = 0x0800_0002;
const DN: u32 = 1 << 25;
const FZ: u32 = 1 << 24;
const IOC: u32 = 1;
const OFC: u32 = 1 << 2;
const UFC: u32 = 1 << 3;
const IXC: u32 = 1 << 4;
const IDC: u32 = 1 << 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Add,
    Maximum,
    Minimum,
    MaximumNumber,
    MinimumNumber,
}
const KINDS: [Kind; 5] = [
    Kind::Add,
    Kind::Maximum,
    Kind::Minimum,
    Kind::MaximumNumber,
    Kind::MinimumNumber,
];
impl Kind {
    fn word(self, double: bool, rd: u32, rn: u32) -> u32 {
        let base = match self {
            Self::Add => 0x7e30_d800,
            Self::Maximum => 0x7e30_f800,
            Self::Minimum => 0x7eb0_f800,
            Self::MaximumNumber => 0x7e30_c800,
            Self::MinimumNumber => 0x7eb0_c800,
        };
        base | (u32::from(double) << 22) | (rn << 5) | rd
    }
}
struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get(offset as usize / 4).copied()
    }
}
fn compiler() -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        ..JitConfig::default()
    })
    .unwrap()
}
fn properties(double: bool) -> (u32, u64, u64, u64) {
    if double {
        (52, 1 << 63, 0x7ff0_0000_0000_0000, 0x0008_0000_0000_0000)
    } else {
        (23, 1 << 31, 0x7f80_0000, 0x0040_0000)
    }
}
fn source(a: u64, b: u64, double: bool) -> [u64; 2] {
    if double {
        [a, b]
    } else {
        [(a & u32::MAX as u64) | ((b & u32::MAX as u64) << 32), GUARD]
    }
}
fn magnitude(value: u64, double: bool) -> [u64; 34] {
    let (fraction, sign, inf, _) = properties(double);
    let value = value & !sign;
    let exponent = (value & inf) >> fraction;
    let mantissa = (value & ((1 << fraction) - 1)) | if exponent == 0 { 0 } else { 1 << fraction };
    let shift = exponent.saturating_sub(1) as usize;
    let mut result = [0; 34];
    result[shift / 64] = mantissa << (shift % 64);
    if shift % 64 != 0 {
        result[shift / 64 + 1] = mantissa >> (64 - shift % 64);
    }
    result
}
fn compare(a: &[u64; 34], b: &[u64; 34]) -> std::cmp::Ordering {
    a.iter().rev().cmp(b.iter().rev())
}
fn exact_sum(a: u64, b: u64, double: bool) -> ([u64; 34], bool) {
    let (_, sign, _, _) = properties(double);
    let mut left = magnitude(a, double);
    let mut right = magnitude(b, double);
    let mut negative = a & sign != 0;
    if (a ^ b) & sign == 0 {
        let mut carry = 0u128;
        for i in 0..34 {
            let sum = u128::from(left[i]) + u128::from(right[i]) + carry;
            left[i] = sum as u64;
            carry = sum >> 64;
        }
    } else {
        if compare(&left, &right).is_lt() {
            std::mem::swap(&mut left, &mut right);
            negative = b & sign != 0;
        }
        let mut borrow = false;
        for i in 0..34 {
            let (value, first) = left[i].overflowing_sub(right[i]);
            let (value, second) = value.overflowing_sub(u64::from(borrow));
            left[i] = value;
            borrow = first || second;
        }
    }
    (left, negative)
}
fn bit(value: &[u64; 34], position: usize) -> bool {
    value[position / 64] & (1 << (position % 64)) != 0
}
fn below(value: &[u64; 34], position: usize) -> bool {
    value[..position / 64].iter().any(|&word| word != 0)
        || position % 64 != 0 && value[position / 64] & ((1 << (position % 64)) - 1) != 0
}
fn encode_add(a: u64, b: u64, double: bool, fpcr: u32, status: &mut u32) -> u64 {
    let (fraction, sign, inf, _) = properties(double);
    let rounding = (fpcr >> 22) & 3;
    let (sum, negative) = exact_sum(a, b, double);
    let Some(word) = sum.iter().rposition(|&word| word != 0) else {
        return if (a | b) & !sign == 0 && (a ^ b) & sign == 0 {
            a & sign
        } else if rounding == 2 {
            sign
        } else {
            0
        };
    };
    let high = word * 64 + 63 - sum[word].leading_zeros() as usize;
    let result_sign = if negative { sign } else { 0 };
    if high < fraction as usize {
        if fpcr & FZ != 0 {
            *status |= UFC;
            return result_sign;
        }
        return result_sign | sum[0];
    }
    let discarded = high - fraction as usize;
    let offset = discarded % 64;
    let mut retained = sum[discarded / 64] >> offset;
    if offset != 0 {
        retained |= sum[discarded / 64 + 1] << (64 - offset);
    }
    retained &= (1 << (fraction + 1)) - 1;
    let inexact = below(&sum, discarded);
    let increment = match rounding {
        0 => {
            discarded != 0
                && bit(&sum, discarded - 1)
                && (below(&sum, discarded - 1) || retained & 1 != 0)
        }
        1 => !negative && inexact,
        2 => negative && inexact,
        _ => false,
    };
    retained += u64::from(increment);
    let mut exponent = (high - fraction as usize + 1) as u64;
    if retained == 1 << (fraction + 1) {
        retained >>= 1;
        exponent += 1;
    }
    if exponent >= inf >> fraction {
        *status |= OFC | IXC;
        let infinity = rounding == 0 || rounding == 1 && !negative || rounding == 2 && negative;
        return result_sign | if infinity { inf } else { inf - 1 };
    }
    if inexact {
        *status |= IXC;
    }
    result_sign | (exponent << fraction) | (retained & ((1 << fraction) - 1))
}
fn expected(kind: Kind, mut a: u64, mut b: u64, double: bool, fpcr: u32) -> (u64, u32) {
    let (_, sign, inf, quiet) = properties(double);
    let mut status = 0;
    for v in [&mut a, &mut b] {
        if fpcr & FZ != 0 && *v & inf == 0 && *v & !sign != 0 {
            *v &= sign;
            status |= IDC;
        }
    }
    let an = a & !sign > inf;
    let bn = b & !sign > inf;
    let signaling = if an && a & quiet == 0 {
        Some(a)
    } else if bn && b & quiet == 0 {
        Some(b)
    } else {
        None
    };
    if let Some(nan) = signaling {
        return (
            if fpcr & DN != 0 {
                inf | quiet
            } else {
                nan | quiet
            },
            status | IOC,
        );
    }
    if an || bn {
        if matches!(kind, Kind::MaximumNumber | Kind::MinimumNumber) && an != bn {
            return (if an { b } else { a }, status);
        }
        return (
            if fpcr & DN != 0 {
                inf | quiet
            } else if an {
                a
            } else {
                b
            },
            status,
        );
    }
    if kind == Kind::Add {
        if a & !sign == inf || b & !sign == inf {
            if a & !sign == inf && b & !sign == inf && (a ^ b) & sign != 0 {
                return (inf | quiet, status | IOC);
            }
            return (if a & !sign == inf { a } else { b }, status);
        }
        return (encode_add(a, b, double, fpcr, &mut status), status);
    }
    let minimum = matches!(kind, Kind::Minimum | Kind::MinimumNumber);
    if (a | b) & !sign == 0 {
        return (
            if minimum {
                (a | b) & sign
            } else {
                (a & b) & sign
            },
            status,
        );
    }
    let less = if (a ^ b) & sign != 0 {
        a & sign != 0
    } else if a & sign != 0 {
        a & !sign > b & !sign
    } else {
        a & !sign < b & !sign
    };
    (if less == minimum { a } else { b }, status)
}
fn context(fpcr: u32) -> CpuContext {
    let mut c = CpuContext::default();
    c.pc = CODE;
    for r in 0..31 {
        c.x[r] = GUARD.wrapping_add(r as u64 * 137);
    }
    for r in 0..32 {
        c.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
    }
    c.sp = GUARD;
    c.nzcv = 0xb;
    c.fpcr = fpcr;
    c.fpsr = STICKY;
    c.tpidr_el0 = GUARD ^ 1;
    c.tpidrro_el0 = GUARD ^ 2;
    c.exclusive_addr = GUARD ^ 3;
    c.exclusive_value = GUARD ^ 4;
    c.exclusive_size = 8;
    c.io_value = [GUARD ^ 5, GUARD ^ 6];
    c
}
fn guards(c: &CpuContext, x: [u64; 31], v: [[u64; 2]; 32], fpcr: u32, status: u32) {
    assert_eq!(c.x, x);
    assert_eq!(c.v, v);
    assert_eq!(c.sp, GUARD);
    assert_eq!(c.nzcv, 0xb);
    assert_eq!(c.fpcr, fpcr);
    assert_eq!(c.fpsr, STICKY | status);
    assert_eq!(c.tpidr_el0, GUARD ^ 1);
    assert_eq!(c.tpidrro_el0, GUARD ^ 2);
    assert_eq!(c.exclusive_addr, GUARD ^ 3);
    assert_eq!(c.exclusive_value, GUARD ^ 4);
    assert_eq!(c.exclusive_size, 8);
    assert_eq!(c.io_value, [GUARD ^ 5, GUARD ^ 6]);
    assert_eq!(c.mem_fault, 0);
    assert_eq!(c.should_halt, 0);
}
fn check(
    jit: &mut Jit,
    code: &mut Code,
    kind: Kind,
    double: bool,
    rd: usize,
    rn: usize,
    a: u64,
    b: u64,
    fpcr: u32,
) {
    let mut c = context(fpcr);
    c.v[rn] = source(a, b, double);
    let x = c.x;
    let mut v = c.v;
    let (answer, status) = expected(kind, a, b, double, fpcr);
    v[rd] = [answer, 0];
    let original = code.0.clone();
    let outcome = jit.run_bounded(&mut c, code, 1000).unwrap();
    assert_eq!(
        outcome.stop,
        StopReason::Brk(0),
        "{kind:?} double{double} a{a:x} b{b:x} fpcr{fpcr:x}"
    );
    assert_eq!(outcome.retired, 2);
    assert_eq!(code.0, original);
    guards(&c, x, v, fpcr, status);
}
fn classes(double: bool) -> Vec<u64> {
    let (fraction, sign, inf, quiet) = properties(double);
    let one = if double {
        0x3ff0_0000_0000_0000
    } else {
        0x3f80_0000
    };
    vec![
        0,
        sign,
        1,
        sign | 1,
        (1 << fraction) - 1,
        sign | ((1 << fraction) - 1),
        1 << fraction,
        sign | (1 << fraction),
        (1 << fraction) + 1,
        sign | ((1 << fraction) + 1),
        one,
        sign | one,
        one - 1,
        one + 1,
        sign | (one - 1),
        sign | (one + 1),
        inf - 1,
        sign | (inf - 1),
        inf,
        sign | inf,
        inf | quiet | 0x123,
        sign | inf | quiet | 0x456,
        inf | 0x789,
        sign | inf | 0xabc,
    ]
}
fn samples(double: bool) -> Vec<(u64, u64)> {
    let classes = classes(double);
    let mut result = Vec::new();
    for &a in &classes {
        for &b in &classes {
            result.push((a, b));
        }
    }
    let mut seed = 0x85cd_ae87_e30d_8a51u64;
    let mask = if double { u64::MAX } else { u32::MAX as u64 };
    for _ in 0..128 {
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed & mask
        };
        result.push((next(), next()));
    }
    result
}

#[test]
fn pairwise_minmax_all_formats_nan_zero_subnormal_rounding_controls_and_aliases() {
    for double in [false, true] {
        let samples = samples(double);
        for kind in [
            Kind::Maximum,
            Kind::Minimum,
            Kind::MaximumNumber,
            Kind::MinimumNumber,
        ] {
            for (rd, rn) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31), (16, 30)] {
                let mut code = Code(vec![kind.word(double, rd, rn), BRK]);
                let mut jit = compiler();
                for rounding in 0..4 {
                    for controls in [0, DN, FZ, DN | FZ] {
                        for &(a, b) in &samples {
                            check(
                                &mut jit,
                                &mut code,
                                kind,
                                double,
                                rd as usize,
                                rn as usize,
                                a,
                                b,
                                (rounding << 22) | controls,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn pairwise_add_all_rounding_modes_exact_integer_oracle_and_special_controls() {
    for double in [false, true] {
        for (rd, rn) in [(0, 1), (0, 0), (5, 5), (31, 31), (31, 2), (2, 31), (16, 30)] {
            let mut code = Code(vec![Kind::Add.word(double, rd, rn), BRK]);
            let mut jit = compiler();
            for rounding in 0..4 {
                for controls in [0, DN, FZ, DN | FZ] {
                    for (a, b) in samples(double) {
                        check(
                            &mut jit,
                            &mut code,
                            Kind::Add,
                            double,
                            rd as usize,
                            rn as usize,
                            a,
                            b,
                            (rounding << 22) | controls,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn independent_add_oracle_rounding_and_overflow_literals() {
    for double in [false, true] {
        let (_, sign, inf, _) = properties(double);
        let (one, half_ulp) = if double {
            (0x3ff0_0000_0000_0000, 0x3ca0_0000_0000_0000)
        } else {
            (0x3f80_0000, 0x3380_0000)
        };
        assert_eq!(expected(Kind::Add, one, half_ulp, double, 0), (one, IXC));
        assert_eq!(
            expected(Kind::Add, one + 1, half_ulp, double, 0),
            (one + 2, IXC)
        );
        assert_eq!(
            expected(Kind::Add, one, half_ulp, double, 1 << 22),
            (one + 1, IXC)
        );
        assert_eq!(
            expected(Kind::Add, one, one | sign, double, 2 << 22),
            (sign, 0)
        );
        assert_eq!(
            expected(Kind::Add, inf - 1, inf - 1, double, 0),
            (inf, OFC | IXC)
        );
        assert_eq!(
            expected(Kind::Add, inf - 1, inf - 1, double, 3 << 22),
            (inf - 1, OFC | IXC)
        );
        assert_eq!(expected(Kind::Add, 1, 1, double, 0), (2, 0));
        let (fraction, _, _, _) = properties(double);
        assert_eq!(
            expected(
                Kind::Add,
                (1 << fraction) + 1,
                sign | (1 << fraction),
                double,
                FZ
            ),
            (0, UFC)
        );
    }
}

#[test]
fn add_rounding_ties_signed_zero_overflow_and_normal_cancellation_literals() {
    for double in [false, true] {
        let (fraction, sign, inf, _) = properties(double);
        let (one, half_ulp) = if double {
            (0x3ff0_0000_0000_0000, 0x3ca0_0000_0000_0000)
        } else {
            (0x3f80_0000, 0x3380_0000)
        };
        let mut code = Code(vec![Kind::Add.word(double, 31, 31), BRK]);
        let mut jit = compiler();
        for rounding in 0..4 {
            for controls in [0, DN, FZ, DN | FZ] {
                for (a, b) in [
                    (one, half_ulp),
                    (one + 1, half_ulp),
                    (one | sign, half_ulp | sign),
                    (one + 1 | sign, half_ulp | sign),
                    (one, one | sign),
                    (0, sign),
                    (sign, sign),
                    (inf - 1, inf - 1),
                    (sign | (inf - 1), sign | (inf - 1)),
                    ((1 << fraction) + 1, sign | (1 << fraction)),
                    (sign | ((1 << fraction) + 1), 1 << fraction),
                ] {
                    check(
                        &mut jit,
                        &mut code,
                        Kind::Add,
                        double,
                        31,
                        31,
                        a,
                        b,
                        (rounding << 22) | controls,
                    );
                }
            }
        }
    }
}

#[test]
fn exact_mk8d_faddp_source_alias_and_low_two_lane_endianness() {
    assert_eq!(Kind::Add.word(false, 5, 5), 0x7e30_d8a5);
    for double in [false, true] {
        for kind in KINDS {
            for rn in 0..32u32 {
                for rd in [rn, (rn + 11) % 32] {
                    let mut code = Code(vec![kind.word(double, rd, rn), BRK]);
                    let (a, b) = if double {
                        (1.25f64.to_bits(), (-0.5f64).to_bits())
                    } else {
                        (1.25f32.to_bits() as u64, (-0.5f32).to_bits() as u64)
                    };
                    check(
                        &mut compiler(),
                        &mut code,
                        kind,
                        double,
                        rd as usize,
                        rn as usize,
                        a,
                        b,
                        0,
                    );
                }
            }
        }
    }
}

#[test]
fn helper_calls_preserve_live_scalar_vector_integer_values_under_pressure() {
    for double in [false, true] {
        for kind in KINDS {
            let mut c = context(DN | FZ);
            let (_, sign, inf, quiet) = properties(double);
            c.v[31] = source(inf | quiet | 0x123, sign | inf | 0x456, double);
            let mut x = c.x;
            let mut v = c.v;
            let mut program = Vec::new();
            for r in 0..16usize {
                program.push(0x9100_0400 | ((r as u32) << 5) | r as u32);
                program.push(0x4ea0_1c00 | (16 << 16) | ((r as u32) << 5) | r as u32);
                x[r] = x[r].wrapping_add(1);
                v[r] = [v[r][0] | v[16][0], v[r][1] | v[16][1]];
            }
            program.push(kind.word(double, 31, 31));
            let (answer, status) = expected(
                kind,
                inf | quiet | 0x123,
                sign | inf | 0x456,
                double,
                DN | FZ,
            );
            v[31] = [answer, 0];
            for p in 0..8usize {
                let rd = if p == 7 { 16 } else { 24 + p };
                program.push(
                    0x8b00_0000
                        | (((p * 2 + 1) as u32) << 16)
                        | (((p * 2) as u32) << 5)
                        | rd as u32,
                );
                x[rd] = x[p * 2].wrapping_add(x[p * 2 + 1]);
            }
            for r in 0..14usize {
                program
                    .push(0x4ea0_1c00 | ((r as u32) << 16) | ((r as u32) << 5) | (17 + r) as u32);
                v[17 + r] = v[r];
            }
            program.push(BRK);
            let count = program.len() as u64;
            let mut code = Code(program);
            let outcome = compiler().run_bounded(&mut c, &mut code, 1000).unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, count);
            guards(&c, x, v, DN | FZ, status);
        }
    }
}

#[test]
fn overwritten_scalar_results_keep_observable_nan_flags() {
    for double in [false, true] {
        for kind in KINDS {
            let (_, _, inf, _) = properties(double);
            let mut c = context(0);
            c.v[1] = source(inf | 0x123, 0, double);
            let x = c.x;
            let mut v = c.v;
            v[0] = [0; 2];
            let mut code = Code(vec![kind.word(double, 0, 1), 0x1e27_03e0, BRK]);
            let outcome = compiler().run_bounded(&mut c, &mut code, 1000).unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, 3);
            guards(&c, x, v, 0, IOC);
        }
    }
}

#[test]
fn unsupported_half_precision_forms_preserve_state_without_retirement() {
    for kind in KINDS {
        let word = kind.word(false, 0, 1) & !(1 << 29);
        let mut c = context(DN | FZ);
        let x = c.x;
        let v = c.v;
        let mut code = Code(vec![word, BRK]);
        let outcome = compiler().run_bounded(&mut c, &mut code, 1).unwrap();
        assert!(
            matches!(outcome.stop, StopReason::Unsupported(_)),
            "word{word:08x}: {:?}",
            outcome.stop
        );
        assert_eq!(outcome.retired, 0);
        assert_eq!(c.pc, CODE);
        guards(&c, x, v, DN | FZ, 0);
    }
}
