use rustarmic::{CpuContext, FlatMemory, Jit, JitConfig};

fn pack(values: [u64; 4], double: bool) -> [u64; 2] {
    if double {
        [values[0], values[1]]
    } else {
        [values[0] | (values[1] << 32), values[2] | (values[3] << 32)]
    }
}

#[test]
fn pairwise_numbers_preserve_nan_zero_flags_aliases_and_q0() {
    for double in [false, true] {
        let (sign, infinity, quiet, one, two) = if double {
            (
                1u64 << 63,
                0x7ff0_0000_0000_0000,
                1u64 << 51,
                1.0f64.to_bits(),
                2.0f64.to_bits(),
            )
        } else {
            (
                1u64 << 31,
                0x7f80_0000,
                1u64 << 22,
                1.0f32.to_bits() as u64,
                2.0f32.to_bits() as u64,
            )
        };
        let qnan = infinity | quiet | 0x123;
        let snan = infinity | 0x456;
        for q in [false, true] {
            if double && !q {
                continue;
            }
            for minimum in [false, true] {
                for rd in [0u32, 1, 2] {
                    let word = 0x2e20_c400
                        | (u32::from(minimum) << 23)
                        | (u32::from(double) << 22)
                        | (u32::from(q) << 30)
                        | (2 << 16)
                        | (1 << 5)
                        | rd;
                    let mut jit = Jit::new(JitConfig::default()).unwrap();
                    let mut memory = FlatMemory::new(0x1000, 8);
                    memory.write_u32(0x1000, word);
                    memory.write_u32(0x1004, 0xd420_0000);
                    for fpcr in [0, 1 << 25, 1 << 24, (1 << 25) | (1 << 24)] {
                        let cases = [
                            (one, qnan, one, 0),
                            (qnan, two, two, 0),
                            (0, sign, if minimum { sign } else { 0 }, 0),
                            (one, two, if minimum { one } else { two }, 0),
                            (
                                snan,
                                one,
                                if fpcr & (1 << 25) != 0 {
                                    infinity | quiet
                                } else {
                                    snan | quiet
                                },
                                1,
                            ),
                            (
                                one,
                                snan,
                                if fpcr & (1 << 25) != 0 {
                                    infinity | quiet
                                } else {
                                    snan | quiet
                                },
                                1,
                            ),
                            (
                                qnan,
                                qnan,
                                if fpcr & (1 << 25) != 0 {
                                    infinity | quiet
                                } else {
                                    qnan
                                },
                                0,
                            ),
                            (
                                1,
                                one,
                                if minimum {
                                    if fpcr & (1 << 24) != 0 { 0 } else { 1 }
                                } else {
                                    one
                                },
                                if fpcr & (1 << 24) != 0 { 1 << 7 } else { 0 },
                            ),
                        ];
                        for (a, b, expected, flags) in cases {
                            let mut ctx = CpuContext::default();
                            ctx.pc = 0x1000;
                            ctx.fpcr = fpcr;
                            ctx.fpsr = (1 << 27) | (1 << 4);
                            ctx.x = [0xface_1234_5678_9abc; 31];
                            ctx.sp = 0x1234;
                            ctx.nzcv = 0xb;
                            ctx.v = [[0xface_1234_5678_9abc; 2]; 32];
                            ctx.v[1] = pack([a, b, a, b], double);
                            ctx.v[2] = pack([a, b, a, b], double);
                            let mut expected_v = ctx.v;
                            expected_v[rd as usize] = if double {
                                [expected; 2]
                            } else {
                                [
                                    expected | (expected << 32),
                                    if q { expected | (expected << 32) } else { 0 },
                                ]
                            };
                            let result = jit.run_bounded(&mut ctx, &mut memory, 1).unwrap();
                            assert_eq!(result.retired, 1);
                            assert_eq!(
                                ctx.v, expected_v,
                                "word={word:08x} fpcr={fpcr:08x} a={a:016x} b={b:016x}"
                            );
                            assert_eq!(ctx.fpsr, (1 << 27) | (1 << 4) | flags);
                            assert_eq!(ctx.fpcr, fpcr);
                            assert_eq!(ctx.x, [0xface_1234_5678_9abc; 31]);
                            assert_eq!((ctx.sp, ctx.nzcv), (0x1234, 0xb));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn pairwise_number_grouping_keeps_sources_and_pairs_distinct() {
    for double in [false, true] {
        for q in [false, true] {
            if double && !q {
                continue;
            }
            for minimum in [false, true] {
                let convert = |value: f64| {
                    if double {
                        value.to_bits()
                    } else {
                        (value as f32).to_bits() as u64
                    }
                };
                let mut ctx = CpuContext::default();
                ctx.pc = 0x1000;
                ctx.v[1] = pack([8.0, 3.0, 7.0, 2.0].map(convert), double);
                ctx.v[2] = pack([6.0, 1.0, 5.0, 4.0].map(convert), double);
                let expected = if double {
                    [
                        convert(if minimum { 3.0 } else { 8.0 }),
                        convert(if minimum { 1.0 } else { 6.0 }),
                    ]
                } else if q {
                    pack(
                        if minimum {
                            [3.0, 2.0, 1.0, 4.0]
                        } else {
                            [8.0, 7.0, 6.0, 5.0]
                        }
                        .map(convert),
                        false,
                    )
                } else {
                    [
                        convert(if minimum { 3.0 } else { 8.0 })
                            | (convert(if minimum { 1.0 } else { 6.0 }) << 32),
                        0,
                    ]
                };
                let word = 0x2e20_c400
                    | (u32::from(minimum) << 23)
                    | (u32::from(double) << 22)
                    | (u32::from(q) << 30)
                    | (2 << 16)
                    | (1 << 5);
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                let mut memory = FlatMemory::new(0x1000, 8);
                memory.write_u32(0x1000, word);
                memory.write_u32(0x1004, 0xd420_0000);
                assert_eq!(
                    jit.run_bounded(&mut ctx, &mut memory, 1).unwrap().retired,
                    1
                );
                assert_eq!(ctx.v[0], expected);
            }
        }
    }
}

#[test]
fn half_precision_vectors_are_not_silently_double_precision() {
    for word in [
        0x4e42_1420,
        0x6e42_1420,
        0x4e42_2420,
        0x6e42_2420,
        0x6ec2_2420,
        0x6e42_3c20,
        0x4e42_3420,
        0x6e42_0420,
        0x6e42_3420,
        0x4ec2_3420,
        0x6ec2_0420,
        0x6ec2_3420,
        0x4e42_0c20,
        0x4ec2_0c20,
        0x6e42_1c20,
        0x4ec2_1420,
    ] {
        let mut block = rustarmic::ir::Block::new(0x1000);
        let initial_len = block.code.len();
        let result = rustarmic::frontend::translator::translate_instruction(
            &mut rustarmic::ir::IrEmitter::new(&mut block, 0x1000),
            word,
        );
        assert!(matches!(result, Err(rustarmic::Error::Unsupported { .. })));
        assert_eq!(block.code.len(), initial_len);
    }
}
