use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const SEED_FLAGS: u32 = 1 << 5;

struct CodeMem(Vec<u32>);
impl Memory for CodeMem {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

#[derive(Clone, Copy)]
struct Format {
    fraction: u32,
    bias: i32,
    infinity: u64,
    sign: u64,
}
fn format(double: bool) -> Format {
    if double {
        Format {
            fraction: 52,
            bias: 1023,
            infinity: 0x7ff0_0000_0000_0000,
            sign: 1 << 63,
        }
    } else {
        Format {
            fraction: 23,
            bias: 127,
            infinity: 0x7f80_0000,
            sign: 1 << 31,
        }
    }
}

// Independent integer oracle: binary-search the square-root bound instead of
// looking up the production estimate table. No host floating-point estimates.
fn estimate(bits: u64, double: bool, fpcr: u32) -> (u64, u32) {
    let f = format(double);
    let quiet = 1 << (f.fraction - 1);
    let mut bits = bits;
    let mut flags = 0;
    if fpcr & (1 << 24) != 0 && bits & !f.sign != 0 && bits & !f.sign < 1 << f.fraction {
        bits &= f.sign;
        flags |= 1 << 7;
    }
    let magnitude = bits & !f.sign;
    if magnitude > f.infinity {
        flags |= u32::from(bits & quiet == 0);
        return (
            if fpcr & (1 << 25) != 0 {
                f.infinity | quiet
            } else {
                bits | quiet
            },
            flags,
        );
    }
    if magnitude == 0 {
        return (f.infinity | (bits & f.sign), flags | 2);
    }
    if bits & f.sign != 0 {
        return (f.infinity | quiet, flags | 1);
    }
    if magnitude == f.infinity {
        return (0, flags);
    }
    let mut exponent = (bits >> f.fraction) as i32;
    let mut fraction = (bits & ((1 << f.fraction) - 1)) << (52 - f.fraction);
    if exponent == 0 {
        while fraction & (1 << 51) == 0 {
            fraction <<= 1;
            exponent -= 1;
        }
        fraction = (fraction & ((1 << 51) - 1)) << 1;
    }
    let scaled = if exponent & 1 != 0 {
        128 | (fraction >> 45)
    } else {
        256 | (fraction >> 44)
    };
    let midpoint = if scaled < 256 {
        scaled * 2 + 1
    } else {
        ((scaled >> 1) * 2 + 1) * 2
    };
    let mut lo = 512u64;
    let mut hi = 1024u64;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if midpoint * mid * mid < (1 << 28) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let answer = (lo + 1) / 2;
    let result_exponent = (3 * f.bias - 1 - exponent) / 2;
    (
        ((result_exponent as u64) << f.fraction) | ((answer & 255) << (f.fraction - 8)),
        flags,
    )
}

fn estimate_inputs(double: bool) -> Vec<u64> {
    let f = format(double);
    let mut samples = vec![
        0,
        1,
        (1 << f.fraction) - 1,
        1 << f.fraction,
        f.infinity - 1,
        f.infinity,
        f.infinity | 0x12345,
        f.infinity | (1 << (f.fraction - 1)) | 0x54321,
    ];
    // All 384 architectural input bins, across both exponent parities, with
    // lower discarded fraction bits clear and set to catch quantization drift.
    for scaled in 128..512u64 {
        let (exponent, prefix, width) = if scaled < 256 {
            (f.bias, 128, 7)
        } else {
            (f.bias + 1, 256, 8)
        };
        let value = ((exponent as u64) << f.fraction) | ((scaled - prefix) << (f.fraction - width));
        samples.extend([value, value | ((1 << (f.fraction - width)) - 1)]);
    }
    for power in 0..f.fraction {
        samples.push(1 << power);
    }
    let positive = samples.clone();
    samples.extend(positive.into_iter().map(|bits| bits | f.sign));
    samples
}

#[derive(Clone, Copy)]
struct Shape {
    scalar: bool,
    double: bool,
    q: bool,
}
const SHAPES: [Shape; 5] = [
    Shape {
        scalar: true,
        double: false,
        q: true,
    },
    Shape {
        scalar: true,
        double: true,
        q: true,
    },
    Shape {
        scalar: false,
        double: false,
        q: false,
    },
    Shape {
        scalar: false,
        double: false,
        q: true,
    },
    Shape {
        scalar: false,
        double: true,
        q: true,
    },
];
impl Shape {
    fn lanes(self) -> usize {
        if self.scalar {
            1
        } else if self.double || !self.q {
            2
        } else {
            4
        }
    }
    fn opcode(self, step: bool, rd: u32, rn: u32, rm: u32) -> u32 {
        let base = if step { 0x0ea0_fc00 } else { 0x2ea1_d800 };
        base | (u32::from(self.scalar) << 28)
            | (u32::from(self.q) << 30)
            | (u32::from(self.double) << 22)
            | (rn << 5)
            | rd
            | if step { rm << 16 } else { 0 }
    }
}
fn pack(values: &[u64], double: bool) -> [u64; 2] {
    let mut result = [0; 2];
    for (lane, value) in values.iter().copied().enumerate() {
        if double {
            result[lane] = value;
        } else {
            result[lane / 2] |= value << (32 * (lane % 2));
        }
    }
    result
}

fn pack_source(values: &[u64], shape: Shape) -> [u64; 2] {
    let mut result = pack(values, shape.double);
    if shape.scalar || !shape.q {
        result[1] = GUARD;
    }
    if shape.scalar && !shape.double {
        result[0] |= (GUARD & 0xffff_ffff) << 32;
    }
    result
}

fn run_cases(shape: Shape, step: bool, cases: &[(u64, u64, u64, u32)], fpcr: u32, pressure: bool) {
    for (rd, rn, rm) in [
        (0, 1, 2),
        (1, 1, 2),
        (2, 1, 2),
        (31, 1, 2),
        (0, 31, 2),
        (0, 1, 31),
    ] {
        let opcode = shape.opcode(step, rd, rn, rm);
        let mut words = if pressure {
            vec![
                0x9100_0484,
                0x9100_04a5,
                0x9100_04c6,
                0x9100_04e7,
                0x9e62_012f,
                opcode,
                0x8b05_008a,
                0x8b07_00cb,
            ]
        } else {
            vec![opcode]
        };
        words.push(0xd420_0000);
        let length = words.len() as u64;
        let mut memory = CodeMem(words);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        for index in 0..cases.len() {
            let lanes = shape.lanes();
            let rows: Vec<_> = (0..lanes)
                .map(|lane| cases[(index + 7 * lane) % cases.len()])
                .collect();
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.fpcr = fpcr;
            ctx.fpsr = SEED_FLAGS;
            ctx.nzcv = 0xb;
            ctx.sp = GUARD;
            ctx.x.fill(GUARD);
            ctx.v.fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
            ctx.v[rn as usize] =
                pack_source(&rows.iter().map(|row| row.0).collect::<Vec<_>>(), shape);
            ctx.v[rm as usize] =
                pack_source(&rows.iter().map(|row| row.1).collect::<Vec<_>>(), shape);
            let mut wanted_x = ctx.x;
            let mut wanted_v = ctx.v;
            wanted_v[rd as usize] = pack(
                &rows.iter().map(|row| row.2).collect::<Vec<_>>(),
                shape.double,
            );
            if pressure {
                ctx.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
                ctx.x[9] = 1;
                wanted_x = ctx.x;
                wanted_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
                wanted_x[10] = 32;
                wanted_x[11] = 72;
                wanted_v[15] = [0x3ff0_0000_0000_0000, 0];
            }
            let out = jit.run_bounded(&mut ctx, &mut memory, length).unwrap();
            assert_eq!(out.stop, StopReason::Brk(0), "opcode={opcode:#010x}");
            assert_eq!(out.retired, length);
            assert_eq!(
                ctx.v, wanted_v,
                "opcode={opcode:#010x} rows={rows:x?} pressure={pressure}"
            );
            assert_eq!(ctx.x, wanted_x, "GPR guard");
            assert_eq!(ctx.sp, GUARD);
            assert_eq!(ctx.nzcv, 0xb);
            assert_eq!(ctx.fpcr, fpcr);
            assert_eq!(
                ctx.fpsr,
                rows.iter().fold(SEED_FLAGS, |flags, row| flags | row.3)
            );
        }
    }
}

#[test]
fn estimate_all_quantization_bins_specials_subnormals_and_aliases() {
    for shape in SHAPES {
        let cases: Vec<_> = estimate_inputs(shape.double)
            .into_iter()
            .map(|bits| {
                let (answer, flags) = estimate(bits, shape.double, 0);
                (bits, 0, answer, flags)
            })
            .collect();
        run_cases(shape, false, &cases, 0, false);
    }
}

#[test]
fn estimate_dn_fz_flags_and_pressure() {
    for shape in SHAPES {
        let f = format(shape.double);
        for fpcr in [1 << 24, 1 << 25, (1 << 24) | (1 << 25)] {
            let values = [
                0,
                f.sign,
                1,
                1 | f.sign,
                (1 << f.fraction) - 1,
                f.infinity,
                f.infinity | f.sign,
                f.infinity | 0x12345,
                f.infinity | f.sign | (1 << (f.fraction - 1)) | 0x54321,
                (f.bias as u64) << f.fraction,
            ];
            let cases: Vec<_> = values
                .into_iter()
                .map(|bits| {
                    let (answer, flags) = estimate(bits, shape.double, fpcr);
                    (bits, 0, answer, flags)
                })
                .collect();
            run_cases(shape, false, &cases, fpcr, true);
        }
    }
}

fn step_cases(double: bool, fpcr: u32) -> Vec<(u64, u64, u64, u32)> {
    let f = format(double);
    let one = (f.bias as u64) << f.fraction;
    let two = ((f.bias + 1) as u64) << f.fraction;
    let three = two | (1 << (f.fraction - 1));
    let half = ((f.bias - 1) as u64) << f.fraction;
    let one_half = one | (1 << (f.fraction - 1));
    let quiet = 1 << (f.fraction - 1);
    let qnan = f.infinity | quiet | 0x12345;
    let snan = f.infinity | 0x54321;
    let nan = |bits| {
        if fpcr & (1 << 25) != 0 {
            f.infinity | quiet
        } else {
            bits | quiet
        }
    };
    let mut rows = vec![
        (one, one, one, 0),
        (two, two, half | f.sign, 0),
        (three, one, 0, 0),
        (0, f.infinity, one_half, 0),
        (f.infinity, f.sign, one_half, 0),
        (f.infinity, one, f.infinity | f.sign, 0),
        (f.infinity | f.sign, one, f.infinity, 0),
        (f.sign, one, one_half, 0),
        (1, 1, one_half, if fpcr & (1 << 24) != 0 { 128 } else { 0 }),
        (f.infinity - 1, two, (f.infinity - 1) | f.sign, 0),
        ((f.infinity - 1) | f.sign, two, f.infinity - 1, 0),
        (qnan, one, nan(qnan ^ f.sign), 0),
        (one, qnan, nan(qnan), 0),
        (snan, one, nan(snan ^ f.sign), 1),
        (one, snan, nan(snan), 1),
        (qnan, snan, nan(snan), 1),
        (snan, qnan, nan(snan ^ f.sign), 1),
        (qnan, qnan ^ 1, nan(qnan ^ f.sign), 0),
    ];
    // Halving the larger operand must preserve the smallest subnormal input.
    let (large, result) = if double {
        (0x7fe0_0000_0000_0000, 0x3ff7_ffff_ffff_ffff)
    } else {
        (0x7f00_0000, 0x3fbf_ffff)
    };
    rows.push((
        1,
        large,
        if fpcr & (1 << 24) != 0 {
            one_half
        } else {
            result
        },
        if fpcr & (1 << 24) != 0 { 128 } else { 0 },
    ));
    if fpcr == 0 {
        // Exact one-round fused cancellation: separate multiply rounds to3.
        rows.push(if double {
            (
                0x3ff0_0000_0000_0001,
                0x4007_ffff_ffff_ffff,
                0xbc9f_ffff_ffff_fffc,
                0,
            )
        } else {
            (0x3f80_0001, 0x403f_ffff, 0xb37f_fffc, 0)
        });
    }
    if fpcr & (1 << 24) != 0 {
        rows.push((1, f.infinity, one_half, 128));
    }
    rows
}

#[test]
fn refinement_fused_overflow_rescue_nan_order_specials_and_pressure() {
    for shape in SHAPES {
        for fpcr in [0, 1 << 24, 1 << 25, (1 << 24) | (1 << 25)] {
            run_cases(
                shape,
                true,
                &step_cases(shape.double, fpcr),
                fpcr,
                fpcr != 0,
            );
        }
    }
}

#[test]
fn refinement_shared_sources_and_complete_destination_alias() {
    for shape in SHAPES {
        let f = format(shape.double);
        let one = (f.bias as u64) << f.fraction;
        let two = ((f.bias + 1) as u64) << f.fraction;
        let three = two | (1 << (f.fraction - 1));
        let negative_half = (((f.bias - 1) as u64) << f.fraction) | f.sign;
        let inputs = [one, two, three, f.infinity];
        let outputs = [one, negative_half, three | f.sign, f.infinity | f.sign];
        for (rd, rn) in [(0, 1), (1, 1), (31, 31)] {
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let mut mem = CodeMem(vec![shape.opcode(true, rd, rn, rn), 0xd420_0000]);
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.fpsr = SEED_FLAGS;
            ctx.x.fill(GUARD);
            ctx.v.fill([GUARD, GUARD]);
            ctx.v[rn as usize] = pack_source(&inputs[..shape.lanes()], shape);
            let expected_x = ctx.x;
            let mut expected_v = ctx.v;
            expected_v[rd as usize] = pack(&outputs[..shape.lanes()], shape.double);
            let out = jit.run_bounded(&mut ctx, &mut mem, 2).unwrap();
            assert_eq!(out.stop, StopReason::Brk(0));
            assert_eq!(out.retired, 2);
            assert_eq!(ctx.v, expected_v);
            assert_eq!(ctx.x, expected_x);
            assert_eq!(ctx.fpsr, SEED_FLAGS);
        }
    }
}

#[test]
fn exact_wonder_encodings_and_independent_literals() {
    let s4 = SHAPES[3];
    assert_eq!(s4.opcode(false, 6, 3, 0), 0x6ea1_d866);
    assert_eq!(s4.opcode(true, 16, 6, 16), 0x4eb0_fcd0);
    assert_eq!(estimate(0x3f80_0000, false, 0), (0x3f7f_8000, 0));
    assert_eq!(estimate(0x4000_0000, false, 0), (0x3f34_8000, 0));
    assert_eq!(
        estimate(0x3ff0_0000_0000_0000, true, 0),
        (0x3fef_f000_0000_0000, 0)
    );
    assert_eq!(estimate(1, false, 0), (0x64b4_8000, 0));
    assert_eq!(estimate(1, true, 0), (0x617f_f000_0000_0000, 0));
}

#[test]
fn overwritten_estimate_destination_keeps_exception_side_effects() {
    for (input, flags) in [(0, 2), (0xbf80_0000, 1)] {
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let mut mem = CodeMem(vec![
            SHAPES[0].opcode(false, 0, 1, 0),
            0x1e27_03e0,
            0xd420_0000,
        ]);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.fpsr = SEED_FLAGS;
        ctx.v[1] = [input, GUARD];
        assert_eq!(
            jit.run_bounded(&mut ctx, &mut mem, 3).unwrap().stop,
            StopReason::Brk(0)
        );
        assert_eq!(ctx.v[0], [0, 0]);
        assert_eq!(ctx.fpsr, SEED_FLAGS | flags);
        assert_eq!(ctx.v[1], [input, GUARD]);
    }
}
