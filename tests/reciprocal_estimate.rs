use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const SEED_FLAGS: u32 = (1 << 27) | (1 << 5);

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

// Independent rational oracle: search the integer midpoint bound, instead of
// using the production reciprocal table or dividing floating-point numbers.
fn estimate(bits: u64, double: bool, fpcr: u32) -> (u64, u32) {
    let f = format(double);
    let sign = bits & f.sign;
    let mut magnitude = bits & !f.sign;
    let quiet = 1 << (f.fraction - 1);
    let mut flags = 0;
    if fpcr & (1 << 24) != 0 && magnitude != 0 && magnitude < 1 << f.fraction {
        flags |= 128;
        magnitude = 0;
    }
    if magnitude > f.infinity {
        flags |= u32::from(magnitude & quiet == 0);
        return (
            if fpcr & (1 << 25) != 0 {
                f.infinity | quiet
            } else {
                bits | quiet
            },
            flags,
        );
    }
    if magnitude == f.infinity {
        return (sign, flags);
    }
    if magnitude == 0 {
        return (sign | f.infinity, flags | 2);
    }
    // The strict tiny-input boundary lies inside the subnormal encoding.
    if magnitude < 1 << (f.fraction - 2) {
        let towards = (fpcr >> 22) & 3;
        let inf = towards == 0 || (towards == 1 && sign == 0) || (towards == 2 && sign != 0);
        return (
            sign | if inf { f.infinity } else { f.infinity - 1 },
            flags | 20,
        );
    }
    if fpcr & (1 << 24) != 0 && magnitude >= ((2 * f.bias - 1) as u64) << f.fraction {
        return (sign, flags | 8);
    }
    let mut exponent = (magnitude >> f.fraction) as i32;
    let mut significand = magnitude & ((1 << f.fraction) - 1);
    if exponent == 0 {
        let highest = 63 - significand.leading_zeros();
        let shift = f.fraction - highest;
        exponent = 1 - shift as i32;
        significand <<= shift;
    } else {
        significand |= 1 << f.fraction;
    }
    let a = significand >> (f.fraction - 8);
    let midpoint = 2 * a + 1;
    let mut lo = 256u64;
    let mut hi = 512u64;
    while lo + 1 < hi {
        let middle = (lo + hi) / 2;
        if (2 * middle - 1) * midpoint <= 1 << 19 {
            lo = middle;
        } else {
            hi = middle;
        }
    }
    let result_exponent = 2 * f.bias - 1 - exponent;
    let result = if result_exponent > 0 {
        ((result_exponent as u64) << f.fraction) | ((lo & 255) << (f.fraction - 8))
    } else {
        (lo << (f.fraction - 8)) >> (1 - result_exponent)
    };
    (sign | result, flags)
}

fn estimate_inputs(double: bool) -> Vec<u64> {
    let f = format(double);
    let threshold = 1 << (f.fraction - 2);
    let mut samples = vec![
        0,
        1,
        threshold - 1,
        threshold,
        threshold + 1,
        (1 << f.fraction) - 1,
        1 << f.fraction,
        f.infinity - 1,
        f.infinity,
        f.infinity | 0x12345,
        f.infinity | (1 << (f.fraction - 1)) | 0x54321,
    ];
    for exponent in [f.bias, f.bias + 1] {
        for prefix in 0..256u64 {
            let n = ((exponent as u64) << f.fraction) | (prefix << (f.fraction - 8));
            samples.extend([n, n | ((1 << (f.fraction - 8)) - 1)]);
        }
    }
    for exponent in [1, 2, 2 * f.bias - 2, 2 * f.bias - 1, 2 * f.bias] {
        for prefix in [0, 127, 255u64] {
            let n = ((exponent as u64) << f.fraction) | (prefix << (f.fraction - 8));
            samples.extend([n, n | ((1 << (f.fraction - 8)) - 1)]);
        }
    }
    for power in 0..f.fraction {
        let n = 1 << power;
        samples.extend([n - 1, n, n + 1]);
    }
    let positives = samples.clone();
    samples.extend(positives.into_iter().map(|n| n | f.sign));
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
        let base = if step { 0x0e20_fc00 } else { 0x0ea1_d800 };
        base | (u32::from(self.scalar) << 28)
            | (u32::from(self.q) << 30)
            | (u32::from(self.double) << 22)
            | (rn << 5)
            | rd
            | if step { rm << 16 } else { 0 }
    }
}
fn pack(values: &[u64], shape: Shape, poison: bool) -> [u64; 2] {
    let mut v = [0; 2];
    for (lane, &value) in values.iter().enumerate() {
        if shape.double {
            v[lane] = value;
        } else {
            v[lane / 2] |= value << (32 * (lane % 2));
        }
    }
    if poison && (shape.scalar || !shape.q) {
        v[1] = GUARD;
    }
    if poison && shape.scalar && !shape.double {
        v[0] |= (GUARD & 0xffff_ffff) << 32;
    }
    v
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
            let rows: Vec<_> = (0..shape.lanes())
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
            ctx.v[rn as usize] = pack(&rows.iter().map(|r| r.0).collect::<Vec<_>>(), shape, true);
            if step {
                ctx.v[rm as usize] =
                    pack(&rows.iter().map(|r| r.1).collect::<Vec<_>>(), shape, true);
            }
            let mut expected_x = ctx.x;
            let mut expected_v = ctx.v;
            expected_v[rd as usize] =
                pack(&rows.iter().map(|r| r.2).collect::<Vec<_>>(), shape, false);
            if pressure {
                ctx.x[4..8].copy_from_slice(&[10, 20, 30, 40]);
                ctx.x[9] = 1;
                expected_x = ctx.x;
                expected_x[4..8].copy_from_slice(&[11, 21, 31, 41]);
                expected_x[10] = 32;
                expected_x[11] = 72;
                expected_v[15] = [0x3ff0_0000_0000_0000, 0];
            }
            let out = jit.run_bounded(&mut ctx, &mut memory, length).unwrap();
            assert_eq!(out.stop, StopReason::Brk(0), "opcode={opcode:#010x}");
            assert_eq!(out.retired, length);
            assert_eq!(
                ctx.v, expected_v,
                "opcode={opcode:#010x} rows={rows:x?} fpcr={fpcr:x}"
            );
            assert_eq!(ctx.x, expected_x);
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
fn estimate_quantization_bins_boundaries_subnormals_and_aliases() {
    for shape in SHAPES {
        let cases: Vec<_> = estimate_inputs(shape.double)
            .into_iter()
            .map(|n| {
                let (value, flags) = estimate(n, shape.double, 0);
                (n, 0, value, flags)
            })
            .collect();
        run_cases(shape, false, &cases, 0, false);
    }
}

#[test]
fn estimate_dn_fz_overflow_modes_fpsr_and_pressure() {
    for shape in SHAPES {
        let f = format(shape.double);
        for fpcr in [
            1 << 24,
            1 << 25,
            (1 << 24) | (1 << 25),
            1 << 22,
            2 << 22,
            3 << 22,
        ] {
            let mut values = vec![
                0,
                1,
                (1 << (f.fraction - 2)) - 1,
                1 << (f.fraction - 2),
                (1 << f.fraction) - 1,
                f.infinity - 1,
                f.infinity,
                f.infinity | 0x12345,
                f.infinity | (1 << (f.fraction - 1)) | 0x54321,
                ((2 * f.bias - 1) as u64) << f.fraction,
                (((2 * f.bias - 1) as u64) << f.fraction) - 1,
            ];
            let positive = values.clone();
            values.extend(positive.into_iter().map(|value| value | f.sign));
            let cases: Vec<_> = values
                .into_iter()
                .map(|n| {
                    let (value, flags) = estimate(n, shape.double, fpcr);
                    (n, 0, value, flags)
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
    let quiet = 1 << (f.fraction - 1);
    let qn = f.infinity | quiet | 0x12345;
    let sn = f.infinity | 0x54321;
    let nan = |n| {
        if fpcr & (1 << 25) != 0 {
            f.infinity | quiet
        } else {
            n | quiet
        }
    };
    let mut rows = vec![
        (one, one, one, 0),
        (two, one, 0, 0),
        (two, two, two | f.sign, 0),
        (one | f.sign, two, ((f.bias + 2) as u64) << f.fraction, 0),
        (0, one, two, 0),
        (f.sign, one, two, 0),
        (f.infinity, one, f.infinity | f.sign, 0),
        (f.infinity | f.sign, one, f.infinity, 0),
        (one, f.infinity, f.infinity | f.sign, 0),
        (qn, one, nan(qn ^ f.sign), 0),
        (one, qn, nan(qn), 0),
        (sn, one, nan(sn ^ f.sign), 1),
        (one, sn, nan(sn), 1),
        (qn, sn, nan(sn), 1),
        (sn, qn, nan(sn ^ f.sign), 1),
        (qn, qn ^ 1, nan(qn ^ f.sign), 0),
        (1, 1, two, if fpcr & (1 << 24) != 0 { 128 } else { 0 }),
    ];
    for n in [0, f.sign] {
        for m in [f.infinity, f.infinity | f.sign] {
            rows.extend([(n, m, two, 0), (m, n, two, 0)]);
        }
    }
    if fpcr & (1 << 24) != 0 {
        rows.extend([(1, f.infinity, two, 128), (f.infinity, 1, two, 128)]);
    }
    rows.push(if double {
        (
            0x3ff0_0000_0000_0001,
            0x3fff_ffff_ffff_fffe,
            0x3980_0000_0000_0000,
            0,
        )
    } else {
        (0x3f80_0001, 0x3fff_fffe, 0x2900_0000, 0)
    });
    rows
}

#[test]
fn step_fused_literals_nan_priority_specials_and_pressure() {
    for shape in SHAPES {
        for fpcr in [0, 1 << 24, 1 << 25, (1 << 24) | (1 << 25)] {
            // Arithmetic FPSR/DIRECTED rounding are deliberately not asserted:
            // the backend's current helper contract is default RNE, DN/FZ input,
            // and NaN/denormal flags, not full FPRound exception parity.
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
fn step_all_aliases_and_sources_identical() {
    for shape in SHAPES {
        let f = format(shape.double);
        let one = (f.bias as u64) << f.fraction;
        let two = ((f.bias + 1) as u64) << f.fraction;
        let four = ((f.bias + 2) as u64) << f.fraction;
        let fourteen = (((f.bias + 3) as u64) << f.fraction) | (3 << (f.fraction - 2));
        let inputs = [one, two, four, f.infinity];
        let outputs = [one, two | f.sign, fourteen | f.sign, f.infinity | f.sign];
        for (rd, rn) in [(0, 1), (1, 1), (31, 31)] {
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let mut memory = CodeMem(vec![shape.opcode(true, rd, rn, rn), 0xd420_0000]);
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.fpsr = SEED_FLAGS;
            ctx.v.fill([GUARD, GUARD]);
            ctx.x.fill(GUARD);
            ctx.v[rn as usize] = pack(&inputs[..shape.lanes()], shape, true);
            let mut expected_v = ctx.v;
            expected_v[rd as usize] = pack(&outputs[..shape.lanes()], shape, false);
            assert_eq!(
                jit.run_bounded(&mut ctx, &mut memory, 2).unwrap().stop,
                StopReason::Brk(0)
            );
            assert_eq!(ctx.v, expected_v);
            assert!(ctx.x.iter().all(|&x| x == GUARD));
            assert_eq!(ctx.fpsr, SEED_FLAGS);
        }
    }
}

#[test]
fn exact_wonder_encodings_independent_literals_and_refinement_chain() {
    let shape = SHAPES[3];
    assert_eq!(shape.opcode(false, 17, 5, 0), 0x4ea1_d8b1);
    assert_eq!(shape.opcode(true, 18, 17, 5), 0x4e25_fe32);
    assert_eq!(estimate(0x3f80_0000, false, 0), (0x3f7f_8000, 0));
    assert_eq!(
        estimate(0x3ff0_0000_0000_0000, true, 0),
        (0x3fef_f000_0000_0000, 0)
    );
    assert_eq!(estimate(0x0020_0000, false, 0), (0x7f7f_8000, 0));
    assert_eq!(
        estimate(0x0004_0000_0000_0000, true, 0),
        (0x7fef_f000_0000_0000, 0)
    );
    assert_eq!(estimate(0x7e80_0000, false, 0), (0x007f_c000, 0));
    assert_eq!(estimate(0x7f00_0000, false, 0), (0x003f_e000, 0));
    let mut memory = CodeMem(vec![
        0x4ea1_d8b1,
        0x4e25_fe32,
        0x6e32_de31,
        0x4e25_fe32,
        0x4ea0_d8a5,
        0x6e32_de30,
        0xd420_0000,
    ]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.fpsr = SEED_FLAGS;
    ctx.v.fill([GUARD, GUARD]);
    ctx.x.fill(GUARD);
    ctx.v[5] = [0x4000_0000_3f80_0000, 0x4080_0000_4000_0000];
    let before = ctx.v;
    assert_eq!(
        jit.run_bounded(&mut ctx, &mut memory, 7).unwrap().stop,
        StopReason::Brk(0)
    );
    assert_eq!(ctx.v[16], [0x3f00_0000_3f80_0000, 0x3e80_0000_3f00_0000]);
    assert_eq!(ctx.v[5], [0, 0]);
    for n in 0..32 {
        if ![5, 16, 17, 18].contains(&n) {
            assert_eq!(ctx.v[n], before[n]);
        }
    }
    assert!(ctx.x.iter().all(|&x| x == GUARD));
    assert_eq!(ctx.fpsr, SEED_FLAGS);
}

#[test]
fn overwritten_estimate_destination_preserves_exception_side_effects() {
    for (input, fpcr, flags) in [
        (0, 0, 2),
        (1, 0, 20),
        (1, 1 << 24, 130),
        (0x7e80_0000, 1 << 24, 8),
        (0x7f80_0001, 0, 1),
    ] {
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let mut memory = CodeMem(vec![
            SHAPES[0].opcode(false, 0, 1, 0),
            0x1e27_03e0,
            0xd420_0000,
        ]);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.fpcr = fpcr;
        ctx.fpsr = SEED_FLAGS;
        ctx.v[1] = [input, GUARD];
        assert_eq!(
            jit.run_bounded(&mut ctx, &mut memory, 3).unwrap().stop,
            StopReason::Brk(0)
        );
        assert_eq!(ctx.v[0], [0, 0]);
        assert_eq!(ctx.v[1], [input, GUARD]);
        assert_eq!(ctx.fpsr, SEED_FLAGS | flags);
    }
}
