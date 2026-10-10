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

fn expected(input: [u64; 2], scalar: bool, double: bool, q: bool, unsigned: bool) -> [u64; 2] {
    let mut out = [0; 2];
    let lanes = if scalar {
        1
    } else if double || !q {
        2
    } else {
        4
    };
    for lane in 0..lanes {
        if double {
            out[lane] = if unsigned {
                (input[lane] as f64).to_bits()
            } else {
                (input[lane] as i64 as f64).to_bits()
            };
        } else {
            let integer = (input[lane / 2] >> (32 * (lane % 2))) as u32;
            let float = if unsigned {
                integer as f32
            } else {
                integer as i32 as f32
            };
            out[lane / 2] |= (float.to_bits() as u64) << (32 * (lane % 2));
        }
    }
    out
}

fn inputs() -> Vec<[u64; 2]> {
    let mut out = vec![
        [0, 0],
        [u64::MAX, u64::MAX],
        [0x8000_0000_7fff_ffff, 0xffff_ffff_0000_0001],
        [0x0100_0001_00ff_ffff, 0x0100_0003_0100_0002],
        [0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff],
        [0x8000_0000_0000_0400, 0x8000_0000_0000_0401],
        [0x8000_0080_0000_0000, 0x8000_0080_0000_0001],
        [0x0020_0000_0000_0001, 0xffdf_ffff_ffff_ffff],
    ];
    let mut random = 0x4d59_5df4_d0f3_3173u64;
    for _ in 0..128 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        let low = random;
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        out.push([low, random]);
    }
    out
}

fn check_form(unsigned: bool, scalar: bool, double: bool, q: bool) {
    let base = if scalar {
        0x5e21_d800
    } else if q {
        0x4e21_d800
    } else {
        0x0e21_d800
    };
    let base = base | ((unsigned as u32) << 29) | ((double as u32) << 22);
    for (rd, rn) in [(0, 0), (0, 1), (31, 31), (31, 1)] {
        let mut memory = CodeMem([base | rd | rn << 5, 0xd420_0000]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        for input in inputs() {
            let mut context = CpuContext::default();
            context.pc = CODE_BASE;
            context.nzcv = 0xb;
            context.x.fill(0xface_1234_5678_9abc); // SIMD sources must not read Xn.
            context
                .v
                .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
            context.v[rn as usize] = input;
            let original_x = context.x;
            let mut expected_v = context.v;
            expected_v[rd as usize] = expected(input, scalar, double, q, unsigned);
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v, expected_v,
                "opcode={:#010x} source={input:016x?}",
                memory.0[0]
            );
            assert_eq!(context.x, original_x);
            assert_eq!(context.nzcv, 0xb);
        }
    }
}

#[test]
fn scalar_simd_unsigned_integer_to_float() {
    for double in [false, true] {
        check_form(true, true, double, false);
    }
}
#[test]
fn scalar_simd_signed_integer_to_float() {
    for double in [false, true] {
        check_form(false, true, double, false);
    }
}
#[test]
fn vector_unsigned_integer_to_float() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        check_form(true, false, double, q);
    }
}
#[test]
fn vector_signed_integer_to_float() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        check_form(false, false, double, q);
    }
}

struct PressureCodeMem(Vec<u32>);
impl Memory for PressureCodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let offset = addr.checked_sub(CODE_BASE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

#[test]
fn single_vector_lanes_zero_extend_when_float_results_spill() {
    for unsigned in [false, true] {
        for q in [false, true] {
            for alias in [false, true] {
                let rn = u32::from(!alias);
                let conversion =
                    0x0e21_d800 | ((unsigned as u32) << 29) | ((q as u32) << 30) | (rn << 5);
                let mut memory = PressureCodeMem(vec![
                    0x9100_0484, // ADD X4, X4, #1
                    0x9100_04a5, // ADD X5, X5, #1
                    0x9100_04c6, // ADD X6, X6, #1
                    0x9100_04e7, // ADD X7, X7, #1
                    0x9e62_012f, // SCVTF D15, X9: deliberately poisons XMM0's upper S lane.
                    conversion,
                    0x8b05_008a, // ADD X10, X4, X5: keeps four integer values live across conversion.
                    0x8b07_00cb, // ADD X11, X6, X7
                    0xd420_0000, // BRK #0
                ]);
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                for input in inputs() {
                    let mut context = CpuContext::default();
                    context.pc = CODE_BASE;
                    context.nzcv = 0xb;
                    context.x.fill(0xface_1234_5678_9abc);
                    context.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
                    context.x[9] = 1;
                    context
                        .v
                        .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
                    context.v[rn as usize] = input;
                    let mut expected_x = context.x;
                    expected_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
                    expected_x[10] = 32;
                    expected_x[11] = 72;
                    let mut expected_v = context.v;
                    expected_v[0] = expected(input, false, false, q, unsigned);
                    expected_v[15] = [0x3ff0_0000_0000_0000, 0];
                    assert_eq!(
                        jit.run(&mut context, &mut memory).unwrap(),
                        ExitReason::Brk(0)
                    );
                    assert_eq!(
                        context.v, expected_v,
                        "spilled lanes: opcode={conversion:#010x}, input={input:016x?}"
                    );
                    assert_eq!(context.x, expected_x);
                    assert_eq!(context.nzcv, 0xb);
                }
            }
        }
    }
}
