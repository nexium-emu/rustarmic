use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const QC: u32 = 1 << 27;
const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;
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
fn context() -> CpuContext {
    let mut c = CpuContext::default();
    c.pc = CODE;
    c.x.fill(GUARD);
    c.v.fill([GUARD, !GUARD]);
    c.sp = GUARD;
    c.nzcv = 0xb;
    c.fpcr = DN;
    c.fpsr = 0x94;
    c
}
fn run(jit: &mut Jit, words: Vec<u32>, c: &mut CpuContext) {
    jit.invalidate_range(CODE, 0x1000);
    let mut code = Code(words);
    let budget = code.0.len() as u64;
    let result = jit.run_bounded(c, &mut code, budget).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, code.0.len() as u64);
}

// Mathematical addition, including the complemented second operand for SBC.
// No host flags, native ADC/SBB, or implementation's flag packing are used.
fn carry(n: u64, m: u64, width: u32, input: u8, subtract: bool) -> (u64, u8) {
    let modulus = 1u128 << width;
    let mask = modulus - 1;
    let a = n as u128 & mask;
    let b = if subtract {
        (!m as u128) & mask
    } else {
        m as u128 & mask
    };
    let sum = a + b + u128::from(input & 2 != 0);
    let result = sum % modulus;
    let signed = |x: u128| {
        if x >= modulus / 2 {
            x as i128 - modulus as i128
        } else {
            x as i128
        }
    };
    let signed_sum = signed(a) + signed(b) + i128::from(input & 2 != 0);
    let overflow = signed_sum < -(modulus as i128 / 2) || signed_sum >= modulus as i128 / 2;
    (
        result as u64,
        ((result >= modulus / 2) as u8) * 8
            | ((result == 0) as u8) * 4
            | ((sum >= modulus) as u8) * 2
            | overflow as u8,
    )
}

#[test]
fn carry_arithmetic_flags_aliases_zero_register_and_ordinary_flags_preservation() {
    let mut jit = compiler();
    for width in [32, 64] {
        let mask = if width == 64 {
            u64::MAX
        } else {
            u32::MAX as u64
        };
        let values = [
            0,
            1,
            2,
            mask,
            mask - 1,
            1 << (width - 1),
            (1 << (width - 1)) - 1,
            GUARD,
        ];
        for subtract in [false, true] {
            for flags in [false, true] {
                for (rn, rm, rd) in [
                    (0, 1, 2),
                    (0, 1, 0),
                    (0, 1, 1),
                    (31, 1, 2),
                    (0, 31, 2),
                    (0, 1, 31),
                    (30, 30, 30),
                ] {
                    let word = 0x1a00_0000
                        | ((width == 64) as u32) << 31
                        | (subtract as u32) << 30
                        | (flags as u32) << 29
                        | (rm << 16)
                        | (rn << 5)
                        | rd;
                    jit.invalidate_range(CODE, 8);
                    let mut code = Code(vec![word, BRK]);
                    for &n in &values {
                        for &m in &values {
                            for input in [0, 2, 0xd, 0xf] {
                                let mut c = context();
                                if rn != 31 {
                                    c.x[rn as usize] = n;
                                }
                                if rm != 31 {
                                    c.x[rm as usize] = m;
                                }
                                c.nzcv = input;
                                let mut expected = c.x;
                                let (value, nzcv) = carry(
                                    if rn == 31 { 0 } else { c.x[rn as usize] },
                                    if rm == 31 { 0 } else { c.x[rm as usize] },
                                    width,
                                    input,
                                    subtract,
                                );
                                if rd != 31 {
                                    expected[rd as usize] = value;
                                }
                                let result = jit.run_bounded(&mut c, &mut code, 2).unwrap();
                                assert_eq!(result.stop, StopReason::Brk(0), "{word:08x}");
                                assert_eq!(c.x, expected, "{word:08x} n{n:x} m{m:x} c{input:x}");
                                assert_eq!(
                                    c.nzcv,
                                    if flags { nzcv } else { input },
                                    "{word:08x} n{n:x} m{m:x} c{input:x}"
                                );
                                assert_eq!(c.v, [[GUARD, !GUARD]; 32]);
                                assert_eq!((c.sp, c.fpcr, c.fpsr), (GUARD, DN, 0x94));
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn carry_forwarding_chain_and_dead_result_flag_side_effects() {
    let mut jit = compiler();
    let mut c = context();
    c.x[0] = u64::MAX;
    c.x[1] = 1;
    c.x[29] = 0x2000_0000;
    // Set carry in this block, generate carry with discarded ADCS output, then
    // consume the new flags through ordinary ADC and flag-setting SBCS.
    run(
        &mut jit,
        vec![
            0xd51b_421d,
            0xba01_001f,
            0x9a01_0002,
            0xfa01_0043,
            0xd53b_4204,
            BRK,
        ],
        &mut c,
    );
    assert_eq!(c.x[2], 1);
    assert_eq!(c.x[3], 0);
    assert_eq!(c.nzcv, 6);
    assert_eq!(c.x[4], 0x6000_0000);
}

fn lane(v: [u64; 2], index: usize, width: u32) -> u64 {
    let packed = v[0] as u128 | (v[1] as u128) << 64;
    ((packed >> (index as u32 * width)) & ((1u128 << width) - 1)) as u64
}
fn pack(values: impl Iterator<Item = u64>, width: u32) -> [u64; 2] {
    let mut value = 0u128;
    let mask = (1u128 << width) - 1;
    for (i, x) in values.enumerate() {
        value |= (x as u128 & mask) << (i as u32 * width);
    }
    [value as u64, (value >> 64) as u64]
}
fn saturation(
    n: [u64; 2],
    m: [u64; 2],
    width: u32,
    q: bool,
    scalar: bool,
    unsigned: bool,
    sub: bool,
) -> ([u64; 2], bool) {
    let modulus = 1i128 << width;
    let decode = |x: u64| {
        if !unsigned && x as i128 >= modulus / 2 {
            x as i128 - modulus
        } else {
            x as i128
        }
    };
    let (lo, hi) = if unsigned {
        (0, modulus - 1)
    } else {
        (-modulus / 2, modulus / 2 - 1)
    };
    let mut changed = false;
    let count = if scalar {
        1
    } else {
        (if q { 128 } else { 64 }) / width as usize
    };
    let result = pack(
        (0..count).map(|i| {
            let a = decode(lane(n, i, width));
            let b = decode(lane(m, i, width));
            let x = if sub { a - b } else { a + b };
            let clamped = x.clamp(lo, hi);
            changed |= x != clamped;
            clamped as u64
        }),
        width,
    );
    (result, changed)
}

#[test]
fn saturating_all_vector_scalar_widths_aliases_boundaries_and_sticky_qc() {
    let mut jit = compiler();
    for scalar in [false, true] {
        for width in [8, 16, 32, 64] {
            for q in [false, true] {
                if scalar && !q || !scalar && width == 64 && !q {
                    continue;
                }
                for unsigned in [false, true] {
                    for sub in [false, true] {
                        for (rn, rm, rd) in
                            [(0, 1, 2), (0, 1, 0), (0, 1, 1), (31, 0, 31), (1, 1, 1)]
                        {
                            let word = 0x0e20_0c00
                                | (scalar as u32) << 28
                                | (q as u32) << 30
                                | (unsigned as u32) << 29
                                | (sub as u32) << 13
                                | ((width as u32).trailing_zeros() - 3) << 22
                                | rm << 16
                                | rn << 5
                                | rd;
                            jit.invalidate_range(CODE, 8);
                            let mut code = Code(vec![word, BRK]);
                            let modulus = 1u128 << width;
                            let values = [
                                0,
                                1,
                                2,
                                (modulus - 1) as u64,
                                (modulus / 2) as u64,
                                (modulus / 2 - 1) as u64,
                                (modulus / 2 + 1) as u64,
                            ];
                            for rotation in 0..values.len() {
                                for other in 0..values.len() {
                                    for old in [0x94, QC | 0x94] {
                                        let mut c = context();
                                        c.fpsr = old;
                                        c.v[rn as usize] = pack(
                                            (0..128 / width).map(|i| {
                                                values[(i as usize + rotation) % values.len()]
                                            }),
                                            width,
                                        );
                                        c.v[rm as usize] = pack(
                                            (0..128 / width).map(|i| {
                                                values[(i as usize + other) % values.len()]
                                            }),
                                            width,
                                        );
                                        let mut expected = c.v;
                                        let (answer, changed) = saturation(
                                            expected[rn as usize],
                                            expected[rm as usize],
                                            width,
                                            q,
                                            scalar,
                                            unsigned,
                                            sub,
                                        );
                                        expected[rd as usize] = answer;
                                        let result = jit.run_bounded(&mut c, &mut code, 2).unwrap();
                                        assert_eq!(result.stop, StopReason::Brk(0), "{word:08x}");
                                        assert_eq!(c.v, expected, "{word:08x}");
                                        assert_eq!(
                                            c.fpsr,
                                            old | if changed { QC } else { 0 },
                                            "{word:08x}"
                                        );
                                        assert_eq!(
                                            (c.x, c.sp, c.nzcv, c.fpcr),
                                            ([GUARD; 31], GUARD, 0xb, DN)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn compare(n: u64, m: u64, double: bool, greater: bool, fpcr: u32) -> (u64, u32) {
    let sign = if double { 1 << 63 } else { 1 << 31 };
    let inf = if double {
        0x7ff0_0000_0000_0000
    } else {
        0x7f80_0000
    };
    let mut a = n & (sign - 1);
    let mut b = m & (sign - 1);
    let mut status = 0;
    for x in [&mut a, &mut b] {
        if fpcr & FZ != 0
            && *x > 0
            && *x
                < if double {
                    0x0010_0000_0000_0000
                } else {
                    0x0080_0000
                }
        {
            *x = 0;
            status |= 128;
        }
    }
    if a > inf || b > inf {
        return (0, status | 1);
    }
    let (a, b) = if double {
        (f64::from_bits(a), f64::from_bits(b))
    } else {
        (
            f32::from_bits(a as u32) as f64,
            f32::from_bits(b as u32) as f64,
        )
    };
    (
        if if greater { a > b } else { a >= b } {
            if double {
                u64::MAX
            } else {
                u32::MAX as u64
            }
        } else {
            0
        },
        status,
    )
}
#[test]
fn absolute_comparisons_all_forms_specials_aliases_and_scoped_status() {
    let mut jit = compiler();
    for (scalar, double, q) in [
        (false, false, false),
        (false, false, true),
        (false, true, true),
        (true, false, true),
        (true, true, true),
    ] {
        let width = if double { 64 } else { 32 };
        let values: &[u64] = if double {
            &[
                0,
                1,
                0x000f_ffff_ffff_ffff,
                0x0010_0000_0000_0000,
                0x3ff0_0000_0000_0000,
                0xbff0_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0123,
                0xfff0_0000_0000_0456,
            ]
        } else {
            &[
                0,
                1,
                0x007f_ffff,
                0x0080_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x7f7f_ffff,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0123,
                0xff80_0456,
            ]
        };
        for greater in [false, true] {
            for (rn, rm, rd) in [(1, 2, 0), (1, 2, 1), (1, 2, 2), (31, 0, 31), (31, 31, 31)] {
                let word = 0x2e20_ec00
                    | (scalar as u32) << 28
                    | (q as u32) << 30
                    | (double as u32) << 22
                    | (greater as u32) << 23
                    | rm << 16
                    | rn << 5
                    | rd;
                jit.invalidate_range(CODE, 8);
                let mut code = Code(vec![word, BRK]);
                for a in 0..values.len() {
                    for b in 0..values.len() {
                        for fpcr in [0, DN, FZ, DN | FZ] {
                            let mut c = context();
                            c.fpcr = fpcr;
                            c.fpsr = 0x14; // IDC and IOC must be newly raised, not preseeded.
                            c.v[rn as usize] = pack(
                                (0..128 / width)
                                    .map(|i| values[(a + i as usize * 3) % values.len()]),
                                width,
                            );
                            c.v[rm as usize] = pack(
                                (0..128 / width)
                                    .map(|i| values[(b + i as usize * 5) % values.len()]),
                                width,
                            );
                            let mut expected = c.v;
                            let n = expected[rn as usize];
                            let m = expected[rm as usize];
                            let mut status = 0;
                            let result = pack(
                                (0..if scalar {
                                    1
                                } else {
                                    (if q { 128 } else { 64 }) / width
                                })
                                    .map(|i| {
                                        let (r, s) = compare(
                                            lane(n, i as usize, width),
                                            lane(m, i as usize, width),
                                            double,
                                            greater,
                                            fpcr,
                                        );
                                        status |= s;
                                        r
                                    }),
                                width,
                            );
                            expected[rd as usize] = result;
                            let result = jit.run_bounded(&mut c, &mut code, 2).unwrap();
                            assert_eq!(result.stop, StopReason::Brk(0), "{word:08x}");
                            assert_eq!(c.v, expected, "{word:08x} a{a} b{b}");
                            assert_eq!(c.fpsr, 0x14 | status, "{word:08x} a{a} b{b} fpcr{fpcr:x}");
                            assert_eq!(
                                (c.x, c.sp, c.nzcv, c.fpcr),
                                ([GUARD; 31], GUARD, 0xb, fpcr)
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn reserved_vector_1d_saturation_and_absolute_comparison_do_not_retire() {
    let mut jit = compiler();
    for word in [
        0x0ee1_0c02,
        0x2ee1_0c02,
        0x0ee1_2c02,
        0x2ee1_2c02,
        0x2e61_ec02,
        0x2ee1_ec02,
    ] {
        jit.invalidate_range(CODE, 8);
        let mut code = Code(vec![word, BRK]);
        let mut c = context();
        let before = (c.x, c.v, c.sp, c.nzcv, c.fpcr, c.fpsr);
        let result = jit.run_bounded(&mut c, &mut code, 2).unwrap();
        assert!(
            matches!(result.stop, StopReason::Unsupported(_)),
            "{word:08x}: {:?}",
            result.stop
        );
        assert_eq!(result.retired, 0);
        assert_eq!((c.x, c.v, c.sp, c.nzcv, c.fpcr, c.fpsr), before);
    }
}
