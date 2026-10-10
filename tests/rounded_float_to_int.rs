use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;

struct CodeMem(Vec<u32>);
impl Memory for CodeMem {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE_BASE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

fn context() -> CpuContext {
    let mut context = CpuContext::default();
    context.pc = CODE_BASE;
    context.sp = GUARD;
    context.nzcv = 0xb;
    context.x.fill(GUARD);
    for (index, value) in context.v.iter_mut().enumerate() {
        *value = [GUARD ^ index as u64, !GUARD ^ ((index as u64) << 32)];
    }
    context
}

#[derive(Clone, Copy, Debug)]
enum Rounding {
    NearestEven,
    NearestAway,
    Floor,
    Ceil,
    Truncate,
}

impl Rounding {
    fn opcode(self) -> u32 {
        match self {
            Self::NearestEven => 0x1e20_0000,
            Self::NearestAway => 0x1e24_0000,
            Self::Floor => 0x1e30_0000,
            Self::Ceil => 0x1e28_0000,
            Self::Truncate => 0x1e38_0000,
        }
    }
}

// Decode IEEE significand and exponent and round the integer magnitude directly.
// This oracle does not round to an intermediate float or execute a host float
// conversion, so it is independent of the production rounder + FCVTZS/ZU path.
fn integer_result(input: u64, double: bool, x_form: bool, unsigned: bool, mode: Rounding) -> u64 {
    let (fraction_bits, exponent_bits, bias) = if double { (52, 11, 1023) } else { (23, 8, 127) };
    let width = if x_form { 64 } else { 32 };
    let negative = input >> (fraction_bits + exponent_bits) & 1 != 0;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = input >> fraction_bits & exponent_mask;
    let fraction = input & ((1u64 << fraction_bits) - 1);
    if exponent == exponent_mask && fraction != 0 || unsigned && negative {
        return 0;
    }
    let limit = if unsigned {
        (1u128 << width) - 1
    } else {
        (1u128 << (width - 1)) - u128::from(!negative)
    };
    let magnitude = if exponent == exponent_mask {
        limit
    } else {
        let significand = fraction | if exponent == 0 { 0 } else { 1 << fraction_bits };
        let shift = (if exponent == 0 { 1 } else { exponent as i32 }) - bias - fraction_bits;
        if shift >= 0 {
            if shift >= 128 || significand as u128 > (limit >> shift) {
                limit
            } else {
                (significand as u128) << shift
            }
        } else {
            let cut = -shift;
            let (quotient, remainder, half) = if cut >= 64 {
                (0, significand, None)
            } else {
                (
                    significand >> cut,
                    significand & ((1u64 << cut) - 1),
                    Some(1u64 << (cut - 1)),
                )
            };
            let increment = match mode {
                Rounding::Floor => negative && remainder != 0,
                Rounding::Ceil => !negative && remainder != 0,
                Rounding::NearestEven => half
                    .is_some_and(|half| remainder > half || remainder == half && quotient & 1 != 0),
                Rounding::NearestAway => half.is_some_and(|half| remainder >= half),
                Rounding::Truncate => false,
            };
            (quotient as u128 + u128::from(increment)).min(limit)
        }
    } as u64;
    let result = if negative {
        0u64.wrapping_sub(magnitude)
    } else {
        magnitude
    };
    if x_form {
        result
    } else {
        result as u32 as u64
    }
}

fn ceil_bits(input: u64, double: bool) -> u64 {
    let (fraction_bits, exponent_bits, bias) = if double { (52, 11, 1023) } else { (23, 8, 127) };
    let sign = 1u64 << (fraction_bits + exponent_bits);
    let input = input & (sign | (sign - 1));
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = input >> fraction_bits & exponent_mask;
    let fraction = input & ((1u64 << fraction_bits) - 1);
    if exponent == exponent_mask {
        return input
            | if fraction != 0 {
                1 << (fraction_bits - 1)
            } else {
                0
            };
    }
    if exponent == 0 && fraction == 0 {
        return input;
    }
    let power = exponent as i32 - bias;
    if power < 0 {
        return if input & sign != 0 {
            sign
        } else {
            (bias as u64) << fraction_bits
        };
    }
    if power >= fraction_bits {
        return input;
    }
    let cut = fraction_bits - power;
    let mask = (1u64 << cut) - 1;
    (input & !mask)
        + if input & sign == 0 && input & mask != 0 {
            1 << cut
        } else {
            0
        }
}

fn samples(double: bool) -> Vec<u64> {
    let (fraction_bits, bias, infinity, sign) = if double {
        (52, 1023, 0x7ff0_0000_0000_0000, 1u64 << 63)
    } else {
        (23, 127, 0x7f80_0000, 1u64 << 31)
    };
    let mut values = vec![
        0,
        1,
        2,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        infinity - 1,
        infinity,
        infinity | 0x12345,
        infinity | (1 << (fraction_bits - 1)) | 0x12345,
    ];
    for power in [
        -2,
        -1,
        0,
        1,
        fraction_bits - 1,
        fraction_bits,
        30,
        31,
        32,
        62,
        63,
        64,
    ] {
        let value = ((bias + power) as u64) << fraction_bits;
        values.extend([value - 1, value, value + 1]);
    }
    let ties = if double {
        [
            0x3fe0_0000_0000_0000,
            0x3ff8_0000_0000_0000,
            0x4004_0000_0000_0000,
            0x400c_0000_0000_0000,
        ]
    } else {
        [0x3f00_0000, 0x3fc0_0000, 0x4020_0000, 0x4060_0000]
    };
    for tie in ties {
        values.extend([tie - 1, tie, tie + 1]);
    }
    values.sort_unstable();
    values.dedup();
    let mut all = Vec::new();
    for value in values {
        all.extend([value, value | sign]);
    }
    let mut random = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..128 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        all.push(if double { random } else { random as u32 as u64 });
    }
    all
}

fn check_integer_matrix(mode: Rounding) {
    for double in [false, true] {
        for x_form in [false, true] {
            for unsigned in [false, true] {
                for (rd, rn) in [(0, 1), (0, 0), (30, 31), (31, 0), (0, 31)] {
                    let opcode = mode.opcode()
                        | ((double as u32) << 22)
                        | ((x_form as u32) << 31)
                        | ((unsigned as u32) << 16)
                        | (rn << 5)
                        | rd;
                    let mut memory = CodeMem(vec![opcode, BRK]);
                    let mut jit = Jit::new(JitConfig::default()).unwrap();
                    for input in samples(double) {
                        let mut context = context();
                        context.v[rn as usize] = [
                            input | if double { 0 } else { 0xdead_beef_0000_0000 },
                            GUARD,
                        ];
                        let mut expected_x = context.x;
                        if rd != 31 {
                            expected_x[rd as usize] =
                                integer_result(input, double, x_form, unsigned, mode);
                        }
                        let original_v = context.v;
                        assert_eq!(
                            jit.run(&mut context, &mut memory).unwrap(),
                            ExitReason::Brk(0)
                        );
                        assert_eq!(
                            context.x, expected_x,
                            "mode={mode:?} opcode={opcode:#010x} input={input:#018x}"
                        );
                        assert_eq!(context.v, original_v);
                        assert_eq!(context.sp, GUARD);
                        assert_eq!(context.nzcv, 0xb);
                        assert_eq!(context.fpcr, 0);
                        assert_eq!(memory.0, [opcode, BRK]);
                    }
                }
            }
        }
    }
}

#[test]
fn independent_oracles_have_literal_tie_and_boundary_results() {
    for (input, even, away, floor, ceil) in [
        (0x3f00_0000, 0, 1, 0, 1),
        (0x3fc0_0000, 2, 2, 1, 2),
        (0x4020_0000, 2, 3, 2, 3),
        (0x4060_0000, 4, 4, 3, 4),
    ] {
        assert_eq!(
            integer_result(input, false, true, false, Rounding::NearestEven),
            even
        );
        assert_eq!(
            integer_result(input, false, true, false, Rounding::NearestAway),
            away
        );
        assert_eq!(
            integer_result(input, false, true, false, Rounding::Floor),
            floor
        );
        assert_eq!(
            integer_result(input, false, true, false, Rounding::Ceil),
            ceil
        );
        assert_eq!(
            integer_result(input | 0x8000_0000, false, true, false, Rounding::Ceil),
            0u64.wrapping_sub(floor)
        );
    }
    assert_eq!(ceil_bits(0x8000_0001, false), 0x8000_0000);
    assert_eq!(ceil_bits(1, false), 0x3f80_0000);
    assert_eq!(ceil_bits(0xff81_2345, false), 0xffc1_2345);
    assert_eq!(
        ceil_bits(0x3ff8_0000_0000_0000, true),
        0x4000_0000_0000_0000
    );
    assert_eq!(integer_result(1, false, true, true, Rounding::Ceil), 1);
    assert_eq!(
        integer_result(0x8000_0001, false, true, false, Rounding::Floor),
        u64::MAX
    );
    assert_eq!(
        integer_result(
            0x43e0_0000_0000_0000,
            true,
            true,
            false,
            Rounding::NearestEven
        ),
        i64::MAX as u64
    );
    assert_eq!(
        integer_result(0x43e0_0000_0000_0000, true, true, true, Rounding::Ceil),
        1 << 63
    );
    assert_eq!(
        integer_result(0x43f0_0000_0000_0000, true, true, true, Rounding::Ceil),
        u64::MAX
    );
    assert_eq!(
        integer_result(0xc1e0_0000_0010_0000, true, false, false, Rounding::Floor),
        0x8000_0000
    );
}

#[test]
fn nearest_even_gpr_conversions_all_signed_unsigned_widths() {
    check_integer_matrix(Rounding::NearestEven);
}
#[test]
fn nearest_away_gpr_conversions_all_signed_unsigned_widths() {
    check_integer_matrix(Rounding::NearestAway);
}
#[test]
fn floor_gpr_conversions_all_signed_unsigned_widths() {
    check_integer_matrix(Rounding::Floor);
}
#[test]
fn ceil_gpr_conversions_all_signed_unsigned_widths() {
    check_integer_matrix(Rounding::Ceil);
}
#[test]
fn existing_truncate_gpr_conversions_remain_correct() {
    check_integer_matrix(Rounding::Truncate);
}

fn check_scalar_ceil(double: bool) {
    for (rd, rn) in [(0, 1), (0, 0), (31, 31), (31, 0), (0, 31)] {
        let opcode = 0x1e24_c000 | ((double as u32) << 22) | (rn << 5) | rd;
        let mut memory = CodeMem(vec![opcode, BRK]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        for input in samples(double) {
            let mut context = context();
            context.v[rn as usize] = [
                input | if double { 0 } else { 0xdead_beef_0000_0000 },
                GUARD,
            ];
            let mut expected_v = context.v;
            expected_v[rd as usize] = [ceil_bits(input, double), 0];
            let original_x = context.x;
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v, expected_v,
                "opcode={opcode:#010x} input={input:#018x}"
            );
            assert_eq!(context.x, original_x);
            assert_eq!(context.sp, GUARD);
            assert_eq!(context.nzcv, 0xb);
            assert_eq!(memory.0, [opcode, BRK]);
        }
    }
}

#[test]
fn scalar_single_frintp_preserves_negative_zero_and_nan_payload() {
    check_scalar_ceil(false);
}
#[test]
fn scalar_double_frintp_preserves_negative_zero_and_nan_payload() {
    check_scalar_ceil(true);
}

#[test]
fn exact_wonder_fcvtpu_x9_s0() {
    let mut memory = CodeMem(vec![0x9e29_0009, BRK]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    for (input, result) in [
        (0x3fc0_0000, 2),
        (1, 1),
        (0xbfc0_0000, 0),
        (0x5f00_0000, 1 << 63),
        (0x5f80_0000, u64::MAX),
        (0x7f81_2345, 0),
    ] {
        let mut context = context();
        context.v[0] = [0xdead_beef_0000_0000 | input, GUARD];
        let original_v = context.v;
        let mut expected_x = context.x;
        expected_x[9] = result;
        assert_eq!(
            jit.run(&mut context, &mut memory).unwrap(),
            ExitReason::Brk(0)
        );
        assert_eq!(context.x, expected_x);
        assert_eq!(context.v, original_v);
        assert_eq!(context.sp, GUARD);
        assert_eq!(context.nzcv, 0xb);
    }
}

#[test]
fn rounded_conversions_keep_live_values_safe_under_register_pressure() {
    let mut memory = CodeMem(vec![
        0x9100_0484,
        0x9100_04a5,
        0x9100_04c6,
        0x9100_04e7,
        0x9e62_012f, // SCVTF D15,X9 poisons scratch XMM upper S lane.
        0x1e24_c040, // FRINTP S0,S2.
        0x9e29_0003, // FCVTPU X3,S0.
        0x9e20_004c, // FCVTNS X12,S2.
        0x9e25_004d, // FCVTAU X13,S2.
        0x9e31_004e, // FCVTMU X14,S2.
        0x8b05_008a,
        0x8b07_00cb,
        BRK,
    ]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    for input in samples(false) {
        let mut context = context();
        context.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
        context.x[9] = 1;
        context.v[2] = [input | 0xdead_beef_0000_0000, GUARD];
        let mut expected_v = context.v;
        expected_v[0] = [ceil_bits(input, false), 0];
        expected_v[15] = [0x3ff0_0000_0000_0000, 0];
        let mut expected_x = context.x;
        expected_x[3] = integer_result(input, false, true, true, Rounding::Ceil);
        expected_x[12] = integer_result(input, false, true, false, Rounding::NearestEven);
        expected_x[13] = integer_result(input, false, true, true, Rounding::NearestAway);
        expected_x[14] = integer_result(input, false, true, true, Rounding::Floor);
        expected_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
        expected_x[10] = 32;
        expected_x[11] = 72;
        assert_eq!(
            jit.run(&mut context, &mut memory).unwrap(),
            ExitReason::Brk(0)
        );
        assert_eq!(context.v, expected_v, "input={input:#010x}");
        assert_eq!(context.x, expected_x, "input={input:#010x}");
        assert_eq!(context.nzcv, 0xb);
        assert_eq!(context.sp, GUARD);
    }
}
