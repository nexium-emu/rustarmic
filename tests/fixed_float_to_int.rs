use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const CANARY: u64 = 0xface_1234_5678_9abc;

struct CodeMem([u32; 2]);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let offset = addr.checked_sub(CODE_BASE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

// Decode IEEE bits into an exact integer significand and binary exponent.
// The oracle never performs a floating-point operation or native float cast.
fn expected(input: u64, double: bool, x: bool, unsigned: bool, fbits: u32) -> u64 {
    let (fraction_bits, exponent_bits, bias) = if double { (52, 11, 1023) } else { (23, 8, 127) };
    let sign = input >> (fraction_bits + exponent_bits) & 1 != 0;
    let exponent_mask = (1u64 << exponent_bits) - 1;
    let exponent = input >> fraction_bits & exponent_mask;
    let fraction = input & ((1u64 << fraction_bits) - 1);
    if exponent == exponent_mask && fraction != 0 {
        return 0;
    }
    if unsigned && sign {
        return 0;
    }
    let width = if x { 64 } else { 32 };
    let limit = if unsigned {
        (1u128 << width) - 1
    } else if sign {
        1u128 << (width - 1)
    } else {
        (1u128 << (width - 1)) - 1
    };
    let magnitude = if exponent == exponent_mask {
        limit
    } else {
        let significand = if exponent == 0 {
            fraction
        } else {
            fraction | (1u64 << fraction_bits)
        };
        let shift = (if exponent == 0 { 1 } else { exponent as i32 }) - bias - fraction_bits as i32
            + fbits as i32;
        if shift >= 0 {
            if significand == 0 {
                0
            } else if shift >= 128 || significand as u128 > limit >> shift {
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
    let value = if sign {
        0u64.wrapping_sub(magnitude)
    } else {
        magnitude
    };
    if x {
        value
    } else {
        value as u32 as u64
    }
}

fn inputs(double: bool, x: bool, unsigned: bool, fbits: u32) -> Vec<u64> {
    let mut values = if double {
        vec![
            0,
            0x8000_0000_0000_0000,
            1,
            0x8000_0000_0000_0001,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x3fe0_0000_0000_0000,
            0xbfe0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x400e_0000_0000_0000,
            0xc00e_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_1234_5678_9abc,
            0xfff8_1234_5678_9abc,
            0x7ff0_1234_5678_9abc,
            0xfff0_1234_5678_9abc,
        ]
    } else {
        vec![
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f00_0000,
            0xbf00_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x4070_0000,
            0xc070_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xffc1_2345,
            0x7f81_2345,
            0xff81_2345,
        ]
    };
    let width = if x { 64 } else { 32 };
    let fraction_bits = if double { 52 } else { 23 };
    let bias = if double { 1023 } else { 127 };
    let sign = 1u64 << if double { 63 } else { 31 };
    // Probe every immediate's saturation boundary and the unsigned-X split.
    for power in [width - u32::from(!unsigned), 63, 0] {
        let exponent = bias + power as i32 - fbits as i32;
        let boundary = (exponent as u64) << fraction_bits;
        for value in [boundary - 1, boundary, boundary + 1] {
            values.extend([value, value | sign]);
        }
    }
    let mut random = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..128 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        values.push(if double { random } else { random as u32 as u64 });
    }
    values
}

fn check(opcode: u32, rn: usize, rd: usize, double: bool, x: bool, unsigned: bool, fbits: u32) {
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut memory = CodeMem([opcode, BRK]);
    for input in inputs(double, x, unsigned, fbits) {
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.x.fill(CANARY);
        context.sp = CANARY;
        context.nzcv = 0b1010;
        context.v.fill([CANARY, CANARY]);
        context.v[rn] = [
            if double {
                input
            } else {
                input | 0xdead_beef_0000_0000
            },
            CANARY,
        ];
        let vectors = context.v;
        let exit = jit.run(&mut context, &mut memory).unwrap();
        assert_eq!(exit, ExitReason::Brk(0));
        let result = expected(input, double, x, unsigned, fbits);
        for (reg, value) in context.x.iter().enumerate() {
            assert_eq!(
                *value,
                if reg == rd { result } else { CANARY },
                "opcode={opcode:#010x} input={input:#018x} x{reg}"
            );
        }
        assert_eq!(context.sp, CANARY, "Rd=31 is ZR, not SP");
        assert_eq!(context.v, vectors, "all SIMD registers must be preserved");
        assert_eq!(context.nzcv, 0b1010);
    }
}

fn family(unsigned: bool) {
    for double in [false, true] {
        for x in [false, true] {
            let formats =
                (u32::from(x) << 31) | (u32::from(double) << 22) | (u32::from(unsigned) << 16);
            for fbits in 1..=if x { 64 } else { 32 } {
                let opcode = 0x1e18_0000 | formats | ((64 - fbits) << 10) | (1 << 5);
                check(opcode, 1, 0, double, x, unsigned, fbits);
            }
            // Ensure destination ZR and high-numbered SIMD source are decoded.
            let opcode = 0x1e18_0000 | formats | ((64 - 8) << 10) | (31 << 5) | 31;
            check(opcode, 31, 31, double, x, unsigned, 8);
        }
    }
}

#[test]
fn all_fixed_signed_formats_and_immediates() {
    family(false);
}

#[test]
fn all_fixed_unsigned_formats_and_immediates() {
    family(true);
}

#[test]
fn ordinary_unsigned_formats() {
    for double in [false, true] {
        for x in [false, true] {
            let opcode = 0x1e39_0000 | (u32::from(x) << 31) | (u32::from(double) << 22) | (1 << 5);
            check(opcode, 1, 0, double, x, true, 0);
            check(opcode | 31, 1, 31, double, x, true, 0);
        }
    }
}

#[test]
fn exact_wonder_fcvtzu_w9_s2_fbits8() {
    check(0x1e19_e049, 2, 9, false, false, true, 8);
}

#[test]
fn oracle_manual_boundaries() {
    assert_eq!(expected(0x3f80_8000, false, false, true, 8), 257);
    assert_eq!(expected(0xbf80_8000, false, false, true, 8), 0);
    assert_eq!(expected(0xbf80_8000, false, false, false, 8), 0xffff_feff);
    assert_eq!(
        expected(0x43df_ffff_ffff_ffff, true, true, true, 0),
        0x7fff_ffff_ffff_fc00
    );
    assert_eq!(
        expected(0x43e0_0000_0000_0001, true, true, true, 0),
        0x8000_0000_0000_0800
    );
    assert_eq!(
        expected(0x43ef_ffff_ffff_ffff, true, true, true, 0),
        0xffff_ffff_ffff_f800
    );
    assert_eq!(
        expected(0x43f0_0000_0000_0000, true, true, true, 0),
        u64::MAX
    );
    assert_eq!(
        expected(0x3ff0_0000_0000_0000, true, true, true, 64),
        u64::MAX
    );
    assert_eq!(
        expected(0xbfe0_0000_0000_0000, true, true, false, 64),
        i64::MIN as u64
    );
}
