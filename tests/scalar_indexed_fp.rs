use rustarmic::{CpuContext, CpuFeatures, FlatMemory, Jit, JitConfig, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;

#[derive(Clone, Copy)]
enum Kind {
    Mla,
    Mls,
    Mul,
}
const KINDS: [Kind; 3] = [Kind::Mla, Kind::Mls, Kind::Mul];
fn encoding(
    kind: Kind,
    double: bool,
    scalar: bool,
    lane: u32,
    rd: usize,
    rn: usize,
    rm: usize,
) -> u32 {
    let operation = match kind {
        Kind::Mla => 0x1000,
        Kind::Mls => 0x5000,
        Kind::Mul => 0x9000,
    };
    let base = if scalar { 0x5f80_0000 } else { 0x4f80_0000 };
    base | operation
        | (u32::from(double) << 22)
        | (((rm as u32) & 15) << 16)
        | (((rm as u32) >> 4) << 20)
        | ((lane >> u32::from(!double)) << 11)
        | (if double { 0 } else { (lane & 1) << 21 })
        | ((rn as u32) << 5)
        | rd as u32
}
fn set_lane(vector: &mut [u64; 2], lane: usize, value: u64, double: bool) {
    if double {
        vector[lane] = value;
    } else {
        let shift = 32 * (lane & 1);
        vector[lane / 2] = (vector[lane / 2] & !(0xffff_ffff << shift)) | (value << shift);
    }
}
fn variants() -> [CpuFeatures; 2] {
    let host = Jit::new(JitConfig::default()).unwrap().host_features;
    let mut baseline = host;
    baseline.has_avx = false;
    baseline.has_fma = false;
    [host, baseline]
}
fn execute(word: u32, ctx: &mut CpuContext, expected: [[u64; 2]; 32], features: CpuFeatures) {
    ctx.pc = CODE;
    ctx.x.fill(GUARD);
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    let mut memory = FlatMemory::new(CODE, 8);
    memory.write_u32(CODE, word);
    memory.write_u32(CODE + 4, 0xd420_0000);
    let mut jit = Jit::new(JitConfig {
        host_features: Some(features),
        ..JitConfig::default()
    })
    .unwrap();
    let out = jit.run_bounded(ctx, &mut memory, 2).unwrap();
    assert_eq!(out.stop, StopReason::Brk(0), "word={word:#010x}");
    assert_eq!(out.retired, 2);
    assert_eq!(
        ctx.v, expected,
        "word={word:#010x} AVX={} FMA={}",
        features.has_avx, features.has_fma
    );
    assert!(ctx.x.iter().all(|&value| value == GUARD));
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpcr, 0);
}

struct Case {
    n: u64,
    m: u64,
    accumulator: [u64; 2],
    fused: [u64; 2],
    product: Option<u64>,
}
fn cases(double: bool) -> Vec<Case> {
    let (one, two, three, five, six, eleven, plus, minus, residue, max, inf, qnan, snan, sign) =
        if double {
            (
                0x3ff0_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x4008_0000_0000_0000,
                0x4014_0000_0000_0000,
                0x4018_0000_0000_0000,
                0x4026_0000_0000_0000,
                0x3ff0_0000_0000_0001,
                0x3fef_ffff_ffff_fffe,
                0x3970_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
                0x7ff0_0000_0000_0000,
                0x7ff8_1234_5678_9abc,
                0x7ff0_1234_5678_9abc,
                1 << 63,
            )
        } else {
            (
                0x3f80_0000,
                0x4000_0000,
                0x4040_0000,
                0x40a0_0000,
                0x40c0_0000,
                0x4130_0000,
                0x3f80_0001,
                0x3f7f_fffe,
                0x2880_0000,
                0x7f7f_ffff,
                0x7f80_0000,
                0x7fc1_2345,
                0x7f81_2345,
                1 << 31,
            )
        };
    let quiet = qnan ^ snan;
    let default_nan = inf | quiet;
    vec![
        Case {
            n: two,
            m: three,
            accumulator: [five; 2],
            fused: [eleven, one | sign],
            product: Some(six),
        },
        Case {
            n: plus,
            m: minus,
            accumulator: [one | sign, one],
            fused: [residue | sign, residue],
            product: Some(one),
        },
        Case {
            n: max,
            m: two,
            accumulator: [max | sign, max],
            fused: [max, max | sign],
            product: Some(inf),
        },
        Case {
            n: one,
            m: one,
            accumulator: [one | sign, one],
            fused: [0; 2],
            product: Some(one),
        },
        Case {
            n: 0,
            m: one,
            accumulator: [0; 2],
            fused: [0; 2],
            product: Some(0),
        },
        Case {
            n: sign,
            m: one,
            accumulator: [sign; 2],
            fused: [sign, 0],
            product: Some(sign),
        },
        Case {
            n: 1,
            m: one,
            accumulator: [0; 2],
            fused: [1, 1 | sign],
            product: Some(1),
        },
        Case {
            n: qnan,
            m: one,
            accumulator: [one; 2],
            fused: [qnan, qnan | sign],
            product: None,
        },
        Case {
            n: one,
            m: qnan,
            accumulator: [one; 2],
            fused: [qnan; 2],
            product: None,
        },
        Case {
            n: one,
            m: one,
            accumulator: [qnan; 2],
            fused: [qnan; 2],
            product: Some(one),
        },
        Case {
            n: snan,
            m: one,
            accumulator: [qnan ^ 3; 2],
            fused: [qnan, qnan | sign],
            product: None,
        },
        Case {
            n: qnan ^ 1,
            m: qnan ^ 2,
            accumulator: [qnan ^ 3; 2],
            fused: [qnan ^ 3; 2],
            product: None,
        },
        Case {
            n: snan ^ 1,
            m: snan ^ 2,
            accumulator: [snan ^ 3; 2],
            fused: [qnan ^ 3; 2],
            product: None,
        },
        Case {
            n: 0,
            m: inf,
            accumulator: [qnan; 2],
            fused: [default_nan; 2],
            product: None,
        },
        Case {
            n: 0,
            m: inf,
            accumulator: [snan; 2],
            fused: [qnan; 2],
            product: None,
        },
        Case {
            n: inf,
            m: one,
            accumulator: [inf | sign, inf],
            fused: [default_nan; 2],
            product: Some(inf),
        },
    ]
}

#[test]
fn scalar_all_indices_high_registers_fused_literals_and_source_guards() {
    for features in variants() {
        for double in [false, true] {
            for lane in 0..if double { 2 } else { 4 } {
                for rm in [2, 16, 31] {
                    for (index, kind) in KINDS.into_iter().enumerate() {
                        for case in cases(double) {
                            let result = if index == 2 {
                                let Some(product) = case.product else {
                                    continue;
                                };
                                product
                            } else {
                                case.fused[index]
                            };
                            let mut ctx = CpuContext::default();
                            ctx.v.fill([GUARD, GUARD]);
                            set_lane(&mut ctx.v[1], 0, case.n, double);
                            set_lane(&mut ctx.v[rm], lane as usize, case.m, double);
                            set_lane(&mut ctx.v[0], 0, case.accumulator[index.min(1)], double);
                            let mut expected = ctx.v;
                            expected[0] = [result, 0];
                            execute(
                                encoding(kind, double, true, lane, 0, 1, rm),
                                &mut ctx,
                                expected,
                                features,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn scalar_complete_alias_matrix_preserves_snapshots_and_clears_upper() {
    for features in variants() {
        for double in [false, true] {
            let (two, four, six, negative_two) = if double {
                (
                    0x4000_0000_0000_0000,
                    0x4010_0000_0000_0000,
                    0x4018_0000_0000_0000,
                    0xc000_0000_0000_0000,
                )
            } else {
                (0x4000_0000, 0x4080_0000, 0x40c0_0000, 0xc000_0000)
            };
            for lane in 0..if double { 2 } else { 4 } {
                for (rd, rn, rm) in [
                    (0, 1, 2),
                    (1, 1, 2),
                    (2, 1, 2),
                    (0, 1, 1),
                    (1, 1, 1),
                    (31, 31, 31),
                    (31, 1, 16),
                    (0, 16, 31),
                ] {
                    for (kind, result) in [
                        (Kind::Mul, four),
                        (Kind::Mla, six),
                        (Kind::Mls, negative_two),
                    ] {
                        let mut ctx = CpuContext::default();
                        ctx.v.fill([GUARD, GUARD]);
                        // Equal actual inputs keep all collision cases exact
                        // without a host arithmetic oracle or stale lane reads.
                        set_lane(&mut ctx.v[rd], 0, two, double);
                        set_lane(&mut ctx.v[rn], 0, two, double);
                        set_lane(&mut ctx.v[rm], lane as usize, two, double);
                        let mut expected = ctx.v;
                        expected[rd] = [result, 0];
                        execute(
                            encoding(kind, double, true, lane, rd, rn, rm),
                            &mut ctx,
                            expected,
                            features,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn vector_double_lane_high_register_bit_is_m20_not_l21() {
    for features in variants() {
        for lane in 0..2 {
            for rm in [16, 31] {
                for rd in [0, 1, rm] {
                    let mut ctx = CpuContext::default();
                    ctx.v.fill([GUARD, GUARD]);
                    ctx.v[1] = [0x4000_0000_0000_0000, 0x4008_0000_0000_0000]; // 2,3.
                    set_lane(&mut ctx.v[rm], lane as usize, 0x4014_0000_0000_0000, true); // 5.
                    let mut expected = ctx.v;
                    expected[rd] = [0x4024_0000_0000_0000, 0x402e_0000_0000_0000]; // 10,15.
                    execute(
                        encoding(Kind::Mul, true, false, lane, rd, 1, rm),
                        &mut ctx,
                        expected,
                        features,
                    );
                }
            }
        }
    }
}

#[test]
fn double_l21_reserved_in_scalar_and_vector_forms() {
    for scalar in [false, true] {
        for kind in KINDS {
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.v.fill([GUARD, GUARD]);
            let wanted_x = ctx.x;
            let wanted_v = ctx.v;
            let mut memory = FlatMemory::new(CODE, 4);
            memory.write_u32(CODE, encoding(kind, true, scalar, 0, 0, 1, 31) | (1 << 21));
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let out = jit.run_bounded(&mut ctx, &mut memory, 1).unwrap();
            assert!(matches!(out.stop, StopReason::Unsupported(_)));
            assert_eq!(out.retired, 0);
            assert_eq!(ctx.x, wanted_x);
            assert_eq!(ctx.v, wanted_v);
        }
    }
}

#[test]
fn exact_wonder_scalar_multiply_pair() {
    assert_eq!(encoding(Kind::Mul, false, true, 0, 2, 3, 0), 0x5f80_9062);
    assert_eq!(encoding(Kind::Mul, false, true, 0, 0, 1, 0), 0x5f80_9020);
    let mut ctx = CpuContext::default();
    ctx.v.fill([GUARD, GUARD]);
    set_lane(&mut ctx.v[0], 0, 0x4000_0000, false); // 2.
    set_lane(&mut ctx.v[1], 0, 0x4040_0000, false); // 3.
    set_lane(&mut ctx.v[3], 0, 0x40a0_0000, false); // 5.
    let mut expected = ctx.v;
    expected[2] = [0x4120_0000, 0]; // 10.
    execute(0x5f80_9062, &mut ctx, expected, variants()[0]);
    expected[0] = [0x40c0_0000, 0]; // 6, consumes original2 before writingV0.
    execute(0x5f80_9020, &mut ctx, expected, variants()[0]);
}
