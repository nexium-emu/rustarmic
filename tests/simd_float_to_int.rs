use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE_BASE: u64 = 0x1000;
const CANARY: u64 = 0xface_1234_5678_9abc;

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

// Exact integer-only IEEE decode: truncate the significand toward zero, clamp
// to the destination range, and return zero for NaN or negative unsigned input.
// These expected values do not depend on either backend or host float casts.
fn integer(input: u64, double: bool, unsigned: bool) -> u64 {
    let (fraction_bits, exponent_bits, bias, width) = if double {
        (52, 11, 1023, 64)
    } else {
        (23, 8, 127, 32)
    };
    let negative = input >> (fraction_bits + exponent_bits) & 1 != 0;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = input >> fraction_bits & exponent_mask;
    let fraction = input & ((1u64 << fraction_bits) - 1);
    if (exponent == exponent_mask && fraction != 0) || (unsigned && negative) {
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
        } else if shift <= -64 {
            0
        } else {
            ((significand >> -shift) as u128).min(limit)
        }
    } as u64;
    let value = if negative {
        0u64.wrapping_sub(magnitude)
    } else {
        magnitude
    };
    if double { value } else { value as u32 as u64 }
}

fn samples(double: bool) -> Vec<u64> {
    let (fraction_bits, bias, width) = if double {
        (52, 1023, 64)
    } else {
        (23, 127, 32)
    };
    let sign = 1u64 << if double { 63 } else { 31 };
    let infinity = if double {
        0x7ff0_0000_0000_0000
    } else {
        0x7f80_0000
    };
    let mut positive = vec![
        0,
        1,
        (1 << fraction_bits) - 1,
        1 << fraction_bits,
        infinity - 1,
        infinity,
        infinity | (1 << (fraction_bits - 1)) | 0x12345,
        infinity | 0x12345,
        ((bias - 1) as u64) << fraction_bits, // 0.5.
        ((bias as u64) << fraction_bits) | (3 << (fraction_bits - 2)), // 1.75.
        (((bias + 1) as u64) << fraction_bits) | (3 << (fraction_bits - 2)), // 3.5.
    ];
    // Adjacent representable values at truncation, precision, both saturation
    // boundaries, and the unsigned64 signed-conversion split at 2^63.
    for power in [
        0,
        1,
        fraction_bits,
        fraction_bits + 1,
        width - 1,
        width,
        width + 1,
    ] {
        let boundary = ((bias + power) as u64) << fraction_bits;
        positive.extend([boundary - 1, boundary, boundary + 1]);
    }
    positive.sort_unstable();
    positive.dedup();
    let mut all = Vec::new();
    for bits in positive {
        all.extend([bits, bits | sign]);
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

fn input_for(values: &[u64], index: usize, double: bool) -> [u64; 2] {
    if double {
        [values[index], values[(index + 7) % values.len()]]
    } else {
        let mut input = [0; 2];
        for lane in 0..4 {
            input[lane / 2] |= values[(index + lane * 7) % values.len()] << (32 * (lane % 2));
        }
        input
    }
}

fn expected(input: [u64; 2], scalar: bool, double: bool, q: bool, unsigned: bool) -> [u64; 2] {
    let lanes = if scalar {
        1
    } else if double || !q {
        2
    } else {
        4
    };
    let mut output = [0; 2];
    for lane in 0..lanes {
        if double {
            output[lane] = integer(input[lane], true, unsigned);
        } else {
            let bits = (input[lane / 2] >> (32 * (lane % 2))) as u32 as u64;
            output[lane / 2] |= integer(bits, false, unsigned) << (32 * (lane % 2));
        }
    }
    output
}

fn encoding(scalar: bool, double: bool, q: bool, unsigned: bool, rn: u32, rd: u32) -> u32 {
    let base = if scalar {
        0x5ea1_b800
    } else {
        0x0ea1_b800 | (u32::from(q) << 30)
    };
    base | (u32::from(unsigned) << 29) | (u32::from(double) << 22) | (rn << 5) | rd
}

fn check(scalar: bool, double: bool, q: bool, unsigned: bool, pressure: bool) {
    let values = samples(double);
    for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
        let conversion = encoding(scalar, double, q, unsigned, rn, rd);
        let mut words = if pressure {
            vec![
                0x9100_0484,
                0x9100_04a5,
                0x9100_04c6,
                0x9100_04e7, // Keep four integer values live.
                0x9e62_012f, // SCVTF D15,X9: poison host SIMD scratch upper bits.
                conversion,
                0x8b05_008a,
                0x8b07_00cb, // Consume the live values after conversion.
            ]
        } else {
            vec![conversion]
        };
        words.push(0xd420_0000); // BRK #0.
        let length = words.len() as u64;
        let mut memory = CodeMem(words);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        for index in 0..values.len() {
            let input = input_for(&values, index, double);
            let mut context = CpuContext::default();
            context.pc = CODE_BASE;
            context.nzcv = 0xb;
            context.sp = CANARY;
            context.x.fill(CANARY); // SIMD conversions must never read Xn.
            context
                .v
                .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
            context.v[rn as usize] = input;
            if pressure {
                context.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
                context.x[9] = 1;
            }
            let mut wanted_x = context.x;
            let mut wanted_v = context.v;
            wanted_v[rd as usize] = expected(input, scalar, double, q, unsigned);
            if pressure {
                wanted_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
                wanted_x[10] = 32;
                wanted_x[11] = 72;
                wanted_v[15] = [0x3ff0_0000_0000_0000, 0];
            }
            let outcome = jit.run_bounded(&mut context, &mut memory, length).unwrap();
            assert_eq!(
                outcome.stop,
                StopReason::Brk(0),
                "opcode={conversion:#010x}"
            );
            assert_eq!(outcome.retired, length);
            assert_eq!(
                context.v, wanted_v,
                "opcode={conversion:#010x} input={input:016x?} pressure={pressure}"
            );
            assert_eq!(context.x, wanted_x, "GPR source guard");
            assert_eq!(context.sp, CANARY);
            assert_eq!(context.nzcv, 0xb);
        }
    }
    // FPSR exception flags are outside this regression's assertion contract.
}

#[test]
fn exact_wonder_scalar_fcvtzs_and_independent_known_results() {
    assert_eq!(encoding(true, false, false, false, 0, 1), 0x5ea1_b801);
    assert_eq!(integer(0x4070_0000, false, false), 3);
    assert_eq!(integer(0xc070_0000, false, false), (-3i32) as u32 as u64);
    assert_eq!(integer(0x7f80_0000, false, false), i32::MAX as u64);
    assert_eq!(integer(0xff80_0000, false, false), i32::MIN as u32 as u64);
    assert_eq!(integer(0xc070_0000, false, true), 0);
    assert_eq!(integer(0x7fc1_2345, false, false), 0);
    assert_eq!(integer(0x43e0_0000_0000_0000, true, true), 1 << 63);
    assert_eq!(integer(0x43f0_0000_0000_0000, true, true), u64::MAX);
}

#[test]
fn scalar_simd_signed_float_to_int() {
    for double in [false, true] {
        check(true, double, false, false, false);
    }
}

#[test]
fn scalar_simd_unsigned_float_to_int() {
    for double in [false, true] {
        check(true, double, false, true, false);
    }
}

#[test]
fn vector_simd_signed_float_to_int() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        check(false, double, q, false, false);
    }
}

#[test]
fn vector_simd_unsigned_float_to_int() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        check(false, double, q, true, false);
    }
}

#[test]
fn vector_lane_results_pack_correctly_under_register_pressure() {
    for unsigned in [false, true] {
        for (double, q) in [(false, false), (false, true), (true, true)] {
            check(false, double, q, unsigned, true);
        }
    }
}

#[test]
fn reserved_one_double_vector_form_is_rejected_without_retirement() {
    for unsigned in [false, true] {
        let mut memory = CodeMem(vec![
            encoding(false, true, false, unsigned, 1, 0),
            0xd420_0000,
        ]);
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.x.fill(CANARY);
        context.v.fill([CANARY, CANARY]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert!(matches!(outcome.stop, StopReason::Unsupported(_)));
        assert_eq!(outcome.retired, 0);
        assert_eq!(context.x, [CANARY; 31]);
        assert_eq!(context.v, [[CANARY; 2]; 32]);
    }
}
