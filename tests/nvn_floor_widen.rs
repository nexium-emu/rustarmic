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
    context.nzcv = 0xb;
    context.sp = GUARD;
    context.x.fill(GUARD);
    for (index, value) in context.v.iter_mut().enumerate() {
        *value = [GUARD ^ index as u64, !GUARD ^ ((index as u64) << 32)];
    }
    context
}

// Integer-only IEEE oracles. None executes a host floating-point conversion
// or floor operation, nor calls the production translator/backend helpers.
fn widen(input: u32) -> u64 {
    let sign = ((input as u64) & 0x8000_0000) << 32;
    let exponent = (input >> 23) & 0xff;
    let fraction = input & 0x7f_ffff;
    if exponent == 0xff {
        return sign
            | 0x7ff0_0000_0000_0000
            | ((fraction as u64) << 29)
            | if fraction == 0 {
                0
            } else {
                0x0008_0000_0000_0000
            };
    }
    if exponent != 0 {
        return sign | (((exponent as u64) + 896) << 52) | ((fraction as u64) << 29);
    }
    if fraction == 0 {
        return sign;
    }
    let top = 31 - fraction.leading_zeros();
    sign | (((874 + top) as u64) << 52) | (((fraction ^ (1 << top)) as u64) << (52 - top))
}

fn floor(input: u64, double: bool) -> u64 {
    let (fraction_bits, exponent_bits, bias) = if double { (52, 11, 1023) } else { (23, 8, 127) };
    let sign = 1u64 << (fraction_bits + exponent_bits);
    let input = input & (sign | (sign - 1));
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = (input >> fraction_bits) & exponent_mask;
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
        return if input & sign == 0 {
            0
        } else {
            sign | ((bias as u64) << fraction_bits)
        };
    }
    if power >= fraction_bits {
        return input;
    }
    let cut = fraction_bits - power;
    let mask = (1u64 << cut) - 1;
    let integer = input & !mask;
    integer
        + if input & sign != 0 && input & mask != 0 {
            1 << cut
        } else {
            0
        }
}

fn integer_floor(input: u64, double: bool, x_form: bool) -> u64 {
    let (fraction_bits, exponent_bits, bias) = if double { (52, 11, 1023) } else { (23, 8, 127) };
    let width = if x_form { 64 } else { 32 };
    let negative = input >> (fraction_bits + exponent_bits) & 1 != 0;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = input >> fraction_bits & exponent_mask;
    let fraction = input & ((1u64 << fraction_bits) - 1);
    if exponent == exponent_mask && fraction != 0 {
        return 0;
    }
    let limit = (1u128 << (width - 1)) - u128::from(!negative);
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
        } else if shift <= -64 {
            u128::from(negative && significand != 0)
        } else {
            let cut = -shift;
            let quotient = significand >> cut;
            let remainder = significand & ((1u64 << cut) - 1);
            (quotient as u128 + u128::from(negative && remainder != 0)).min(limit)
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
    values.extend([
        ((bias as u64) << fraction_bits) | (3 << (fraction_bits - 2)),
        (((bias + 1) as u64) << fraction_bits) | (3 << (fraction_bits - 2)),
    ]);
    values.sort_unstable();
    values.dedup();
    let mut all = Vec::new();
    for value in values {
        all.extend([value, value | sign]);
    }
    let mut random = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..96 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        all.push(if double { random } else { random as u32 as u64 });
    }
    all
}

fn packed(values: [u32; 4]) -> [u64; 2] {
    [
        values[0] as u64 | ((values[1] as u64) << 32),
        values[2] as u64 | ((values[3] as u64) << 32),
    ]
}

#[test]
fn independent_oracles_have_known_boundary_results() {
    assert_eq!(widen(1), 0x36a0_0000_0000_0000);
    assert_eq!(widen(0x8000_0000), 0x8000_0000_0000_0000);
    assert_eq!(widen(0xff81_2345), 0xfff8_2468_a000_0000);
    assert_eq!(floor(0x8000_0001, false), 0xbf80_0000);
    assert_eq!(floor(0xbff8_0000_0000_0000, true), 0xc000_0000_0000_0000);
    assert_eq!(integer_floor(0x8000_0001, false, true), u64::MAX);
    assert_eq!(integer_floor(0x7fc1_2345, false, false), 0);
    assert_eq!(
        integer_floor(0x43e0_0000_0000_0000, true, true),
        i64::MAX as u64
    );
    assert_eq!(
        integer_floor(0xc1e0_0000_0010_0000, true, false),
        0x8000_0000
    );
}

#[test]
fn vector_widening_first_and_upper_half_snapshot_all_aliases() {
    let values = samples(false);
    for upper in [false, true] {
        for (rd, rn) in [(0, 1), (0, 0), (31, 31), (31, 0), (0, 31)] {
            let opcode = 0x0e61_7800 | ((upper as u32) << 30) | (rn << 5) | rd;
            let mut memory = CodeMem(vec![opcode, BRK]);
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            for index in 0..values.len() {
                let lanes =
                    [0, 1, 2, 3].map(|lane| values[(index + lane * 11) % values.len()] as u32);
                let mut context = context();
                context.v[rn as usize] = packed(lanes);
                let mut expected = context.v;
                let first = if upper { 2 } else { 0 };
                expected[rd as usize] = [widen(lanes[first]), widen(lanes[first + 1])];
                let original_x = context.x;
                assert_eq!(
                    jit.run(&mut context, &mut memory).unwrap(),
                    ExitReason::Brk(0)
                );
                assert_eq!(
                    context.v, expected,
                    "opcode={opcode:#010x} lanes={lanes:08x?}"
                );
                assert_eq!(context.x, original_x);
                assert_eq!(context.sp, GUARD);
                assert_eq!(context.nzcv, 0xb);
            }
        }
    }
}

fn check_scalar_floor(double: bool) {
    for (rd, rn) in [(0, 1), (0, 0), (31, 31), (31, 0), (0, 31)] {
        let opcode = 0x1e25_4000 | ((double as u32) << 22) | (rn << 5) | rd;
        let mut memory = CodeMem(vec![opcode, BRK]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        for input in samples(double) {
            let mut context = context();
            context.v[rn as usize] = [
                input | if double { 0 } else { 0xdead_beef_0000_0000 },
                GUARD,
            ];
            let mut expected = context.v;
            expected[rd as usize] = [floor(input, double), 0];
            let original_x = context.x;
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v, expected,
                "opcode={opcode:#010x} input={input:#018x}"
            );
            assert_eq!(context.x, original_x);
            assert_eq!(context.sp, GUARD);
            assert_eq!(context.nzcv, 0xb);
        }
    }
}

#[test]
fn scalar_single_floor_preserves_signed_zero_and_quiets_nan() {
    check_scalar_floor(false);
}
#[test]
fn scalar_double_floor_preserves_signed_zero_and_quiets_nan() {
    check_scalar_floor(true);
}

#[test]
fn all_four_gpr_floor_conversions_saturate_and_preserve_sources() {
    for double in [false, true] {
        for x_form in [false, true] {
            for (rd, rn) in [(0, 1), (0, 0), (30, 31), (31, 0), (0, 31)] {
                let opcode = 0x1e30_0000
                    | ((double as u32) << 22)
                    | ((x_form as u32) << 31)
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
                        expected_x[rd as usize] = integer_floor(input, double, x_form);
                    }
                    let original_v = context.v;
                    assert_eq!(
                        jit.run(&mut context, &mut memory).unwrap(),
                        ExitReason::Brk(0)
                    );
                    assert_eq!(
                        context.x, expected_x,
                        "opcode={opcode:#010x} input={input:#018x}"
                    );
                    assert_eq!(context.v, original_v);
                    assert_eq!(context.sp, GUARD);
                    assert_eq!(context.nzcv, 0xb);
                }
            }
        }
    }
}

#[test]
fn exact_nvn_words_execute_in_place_widen_then_floor() {
    let mut memory = CodeMem(vec![
        0x0e61_7821,
        0x0e61_7800,
        0x1e65_4000,
        0x9e70_0016,
        0x9e70_0018,
        BRK,
    ]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut context = context();
    context.v[0] = packed([0x4070_0000, 0xc070_0000, 0xdead_beef, 0xfedc_ba98]);
    context.v[1] = packed([0xbe80_0000, 0x3e80_0000, 0xdead_beef, 0xfedc_ba98]);
    let mut expected_v = context.v;
    expected_v[0] = [0x4008_0000_0000_0000, 0];
    expected_v[1] = [0xbfd0_0000_0000_0000, 0x3fd0_0000_0000_0000];
    let mut expected_x = context.x;
    expected_x[22] = 3;
    expected_x[24] = 3;
    assert_eq!(
        jit.run(&mut context, &mut memory).unwrap(),
        ExitReason::Brk(0)
    );
    assert_eq!(context.v, expected_v);
    assert_eq!(context.x, expected_x);
    assert_eq!(context.nzcv, 0xb);
}

#[test]
fn rounding_and_widening_keep_live_values_safe_under_register_pressure() {
    let mut memory = CodeMem(vec![
        0x9100_0484,
        0x9100_04a5,
        0x9100_04c6,
        0x9100_04e7,
        0x9e62_012f, // SCVTF D15,X9 poisons the scratch XMM upper S lane.
        0x4e61_7830, // FCVTL2 V16.2D,V1.4S.
        0x1e25_4040, // FRINTM S0,S2.
        0x9e30_0003, // FCVTMS X3,S0.
        0x8b05_008a,
        0x8b07_00cb,
        BRK,
    ]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    for input in samples(false) {
        let mut context = context();
        context.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
        context.x[9] = 1;
        context.v[1] = packed([0, 0x3f80_0000, input as u32, !(input as u32)]);
        context.v[2] = [input | 0xdead_beef_0000_0000, GUARD];
        let mut expected_v = context.v;
        expected_v[0] = [floor(input, false), 0];
        expected_v[15] = [0x3ff0_0000_0000_0000, 0];
        expected_v[16] = [widen(input as u32), widen(!(input as u32))];
        let mut expected_x = context.x;
        expected_x[3] = integer_floor(input, false, true);
        expected_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
        expected_x[10] = 32;
        expected_x[11] = 72;
        assert_eq!(
            jit.run(&mut context, &mut memory).unwrap(),
            ExitReason::Brk(0)
        );
        assert_eq!(context.v, expected_v, "input={input:#010x}");
        assert_eq!(context.x, expected_x);
        assert_eq!(context.nzcv, 0xb);
    }
}
