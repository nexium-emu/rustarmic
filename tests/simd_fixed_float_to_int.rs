use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

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

#[derive(Clone, Copy, Debug)]
struct Form {
    double: bool,
    scalar: bool,
    q: bool,
    unsigned: bool,
    fbits: u32,
    rn: u32,
    rd: u32,
}
impl Form {
    fn word(self) -> u32 {
        let width = if self.double { 64 } else { 32 };
        (if self.scalar {
            0x5f00_fc00
        } else {
            0x0f00_fc00
        }) | (u32::from(self.unsigned) << 29)
            | (u32::from(self.q && !self.scalar) << 30)
            | ((2 * width - self.fbits) << 16)
            | (self.rn << 5)
            | self.rd
    }
    fn wanted(self, source: [u64; 2]) -> [u64; 2] {
        let lanes = if self.scalar {
            1
        } else if self.double || !self.q {
            2
        } else {
            4
        };
        let mut result = [0u64; 2];
        for lane in 0..lanes {
            let bits = if self.double {
                source[lane]
            } else {
                (source[lane / 2] >> (lane % 2 * 32)) & 0xffff_ffff
            };
            let integer = expected(bits, self.double, self.double, self.unsigned, self.fbits);
            if self.double {
                result[lane] = integer
            } else {
                result[lane / 2] |= integer << (lane % 2 * 32)
            }
        }
        result
    }
}
fn check(form: Form) -> usize {
    let mut memory = CodeMem([form.word(), BRK]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut cases = 0;
    for input in inputs(form.double, form.double, form.unsigned, form.fbits)
        .into_iter()
        .take(54)
    {
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.x.fill(CANARY);
        context.sp = CANARY;
        context.nzcv = 0xb;
        context.v.fill([CANARY, CANARY]);
        context.fpsr = 1 << 27;
        context.v[form.rn as usize] = if form.double {
            [input, input ^ (1 << 63)]
        } else {
            [
                input | ((input ^ (1 << 31)) << 32),
                ((input ^ 0x0040_0000) & 0xffff_ffff)
                    | (((input ^ 0x8040_0000) & 0xffff_ffff) << 32),
            ]
        };
        let mut wanted = context.v;
        wanted[form.rd as usize] = form.wanted(context.v[form.rn as usize]);
        let run = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
        assert_eq!(
            run.stop,
            StopReason::Brk(0),
            "{form:?} opcode={:#x}",
            form.word()
        );
        assert_eq!(run.retired, 2);
        assert_eq!(context.v, wanted, "{form:?} input={input:#x}");
        assert_eq!(context.x, [CANARY; 31]);
        assert_eq!(context.sp, CANARY);
        assert_eq!(context.nzcv, 0xb);
        assert_eq!(context.fpcr, 0);
        // Arithmetic exception flags remain an existing converter limitation.
        assert_eq!(context.fpsr & (1 << 27), 1 << 27);
        cases += 1;
    }
    cases
}
fn matrix(unsigned: bool) -> usize {
    let mut count = 0;
    for (double, scalar, q) in [
        (false, true, false),
        (true, true, false),
        (false, false, false),
        (false, false, true),
        (true, false, true),
    ] {
        for fbits in 1..=if double { 64 } else { 32 } {
            for (rn, rd) in [(2, 1), (31, 31), (0, 31)] {
                count += check(Form {
                    double,
                    scalar,
                    q,
                    unsigned,
                    fbits,
                    rn,
                    rd,
                });
            }
        }
    }
    count
}
#[test]
fn signed_all_formats_immediates_aliases_and_integer_edges() {
    assert!(matrix(false) > 30_000);
}
#[test]
fn unsigned_all_formats_immediates_aliases_and_integer_edges() {
    assert!(matrix(true) > 30_000);
}
#[test]
fn exact_wonder_fixed_vector_double_fbits4() {
    let form = Form {
        double: true,
        scalar: false,
        q: true,
        unsigned: false,
        fbits: 4,
        rn: 1,
        rd: 1,
    };
    assert_eq!(form.word(), 0x4f7c_fc21);
    check(form);
    check(Form {
        rn: 0,
        rd: 0,
        ..form
    });
}
#[test]
fn reserved_and_unimplemented_fp16_reject_without_mutation() {
    for instruction in [0x0f7c_fc21, 0x0f00_fc21, 0x0f1c_fc21, 0x5f1c_fc21] {
        let mut memory = CodeMem([instruction, BRK]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.v.fill([CANARY, CANARY]);
        context.x.fill(CANARY);
        context.sp = CANARY;
        let run = jit.run_bounded(&mut context, &mut memory, 1).unwrap();
        assert!(
            matches!(
                run.stop,
                StopReason::Unsupported(_)
            ),
            "{instruction:#x}: {:?}",
            run.stop
        );
        assert_eq!(run.retired, 0);
        assert_eq!(context.pc, CODE_BASE);
        assert_eq!(context.v, [[CANARY; 2]; 32]);
        assert_eq!(context.x, [CANARY; 31]);
        assert_eq!(context.sp, CANARY);
    }
}
