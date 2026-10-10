use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
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

// Integer-only round-to-nearest-even oracle, independent of host conversions.
fn reference(value: u64, double: bool) -> u64 {
    if value == 0 {
        return 0;
    }
    let fraction_bits = if double { 52 } else { 23 };
    let bias = if double { 1023 } else { 127 };
    let mut exponent = 63 - value.leading_zeros();
    let mut significand = if exponent <= fraction_bits {
        value << (fraction_bits - exponent)
    } else {
        let shift = exponent - fraction_bits;
        let truncated = value >> shift;
        let remainder = value & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        truncated + u64::from(remainder > half || (remainder == half && truncated & 1 != 0))
    };
    if significand == (1u64 << (fraction_bits + 1)) {
        exponent += 1;
        significand >>= 1;
    }
    ((exponent + bias) as u64) << fraction_bits | (significand & ((1u64 << fraction_bits) - 1))
}

fn inputs() -> Vec<u64> {
    let mut out = vec![
        0,
        1,
        2,
        u32::MAX as u64 - 1,
        u32::MAX as u64,
        u64::MAX - 1,
        u64::MAX,
    ];
    for exponent in 1..64 {
        let power = 1u64 << exponent;
        out.extend([power - 1, power, power + 1]);
        for precision in [23, 52] {
            if exponent > precision {
                let half = 1u64 << (exponent - precision - 1);
                out.extend([power + half - 1, power + half, power + half + 1]);
                out.extend([power + 3 * half - 1, power + 3 * half, power + 3 * half + 1]);
            }
        }
    }
    let mut random = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..256 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        out.push(random);
    }
    out
}

fn check_format(src_is_x: bool, double: bool) {
    let base = 0x1e23_0000 | ((src_is_x as u32) << 31) | ((double as u32) << 22);
    for rn in [8usize, 31] {
        let mut memory = CodeMem([base | ((rn as u32) << 5), 0xd420_0000]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        for value in inputs() {
            let mut context = CpuContext::default();
            context.pc = CODE_BASE;
            context.nzcv = 0xb;
            context.x.fill(0xface_1234_5678_9abc);
            context
                .v
                .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
            if rn != 31 {
                context.x[rn] = value;
            }
            let original_x = context.x;
            let original_v = context.v;
            let input = if rn == 31 {
                0
            } else if src_is_x {
                value
            } else {
                value as u32 as u64
            };
            let expected = reference(input, double);
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v[0],
                [expected, 0],
                "opcode={:#010x}, input={input:#018x}",
                memory.0[0]
            );
            assert_eq!(context.x, original_x);
            assert_eq!(&context.v[1..], &original_v[1..]);
            assert_eq!(context.nzcv, 0xb);
        }
    }
}

#[test]
fn unsigned_w_to_single() {
    check_format(false, false);
}
#[test]
fn unsigned_x_to_single() {
    check_format(true, false);
}
#[test]
fn unsigned_w_to_double() {
    check_format(false, true);
}
#[test]
fn unsigned_x_to_double() {
    check_format(true, true);
}
