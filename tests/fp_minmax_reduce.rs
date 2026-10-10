pub use rustarmic::jit;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
#[path = "../src/backend/fp_reduce.rs"]
mod fp_reduce;

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const IOC: u32 = 1;
const IDC: u32 = 1 << 7;
const DN: u32 = 1 << 25;
const FZ: u32 = 1 << 24;
const STICKY: u32 = (1 << 27) | (1 << 4);
struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get(offset as usize / 4).copied()
    }
}
fn jit() -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        ..JitConfig::default()
    })
    .unwrap()
}
fn word(mode: u32, rn: u32, rd: u32) -> u32 {
    [0x6e30_f800, 0x6eb0_f800, 0x6e30_c800, 0x6eb0_c800][mode as usize] | (rn << 5) | rd
}
fn pack(lanes: [u32; 4]) -> [u64; 2] {
    [
        lanes[0] as u64 | ((lanes[1] as u64) << 32),
        lanes[2] as u64 | ((lanes[3] as u64) << 32),
    ]
}
// Independent numeric oracle uses host comparisons, never host min/max (whose
// NaN and equal-zero behavior is not Arm's). Classification is integer IEEE.
fn expected(lanes: [u32; 4], mode: u32, fpcr: u32) -> (u32, u32) {
    let mut status = 0;
    let mut pair = |mut a: u32, mut b: u32| {
        for value in [&mut a, &mut b] {
            if fpcr & FZ != 0 && *value & 0x7f80_0000 == 0 && *value & 0x007f_ffff != 0 {
                *value &= 0x8000_0000;
                status |= IDC;
            }
        }
        let an = a & 0x7fff_ffff > 0x7f80_0000;
        let bn = b & 0x7fff_ffff > 0x7f80_0000;
        let signaling = if an && a & 0x0040_0000 == 0 {
            Some(a)
        } else if bn && b & 0x0040_0000 == 0 {
            Some(b)
        } else {
            None
        };
        if let Some(value) = signaling {
            status |= IOC;
            return if fpcr & DN != 0 {
                0x7fc0_0000
            } else {
                value | 0x0040_0000
            };
        }
        if an || bn {
            if mode >= 2 && an != bn {
                return if an { b } else { a };
            }
            return if fpcr & DN != 0 {
                0x7fc0_0000
            } else if an {
                a
            } else {
                b
            };
        }
        let af = f32::from_bits(a);
        let bf = f32::from_bits(b);
        if af == 0.0 && bf == 0.0 {
            return if mode & 1 != 0 {
                (a | b) & 0x8000_0000
            } else {
                (a & b) & 0x8000_0000
            };
        }
        if mode & 1 != 0 {
            if af < bf { a } else { b }
        } else if af > bf {
            a
        } else {
            b
        }
    };
    let left = pair(lanes[0], lanes[1]);
    let right = pair(lanes[2], lanes[3]);
    let result = pair(left, right);
    (result, status)
}
fn classes() -> Vec<u32> {
    vec![
        0,
        0x8000_0000,
        1,
        0x807f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0123,
        0xffc0_0456,
        0x7f80_0789,
        0xff80_0abc,
    ]
}

#[test]
fn exhaustive_class_tree_matches_independent_oracle() {
    let values = classes();
    let mut ctx = CpuContext::default();
    for mode in 0..4 {
        for fpcr in [0, DN, FZ, DN | FZ] {
            ctx.fpcr = fpcr;
            for &a in &values {
                for &b in &values {
                    for &c in &values {
                        for &d in &values {
                            let lanes = [a, b, c, d];
                            let source = pack(lanes);
                            let (want, flags) = expected(lanes, mode, fpcr);
                            ctx.fpsr = STICKY;
                            let result = unsafe {
                                fp_reduce::reduce(&mut ctx, source[0], source[1], mode as u64)
                            };
                            assert_eq!(
                                result, want as u64,
                                "mode{mode} fpcr{fpcr:x} lanes{lanes:x?}"
                            );
                            assert_eq!(ctx.fpsr, STICKY | flags);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn all_forms_aliases_nan_priority_zero_fpcr_and_state_guards() {
    let values = classes();
    let mut samples = Vec::new();
    for (index, &value) in values.iter().enumerate() {
        for lane in 0..4 {
            let mut input = [0x3f80_0000, 0xc000_0000, 0x4080_0000, 0xbf00_0000];
            input[lane] = value;
            samples.push(input);
        }
        samples.push([
            value,
            values[(index + 3) % values.len()],
            values[(index + 7) % values.len()],
            values[(index + 11) % values.len()],
        ]);
    }
    samples.extend([
        [0x7f80_0123, 0x3f80_0000, 0x4000_0000, 0x4040_0000],
        [0x7fc0_0123, 0x7f80_0456, 0x7fc0_0789, 0xff80_0abc],
        [0x7fc0_0123, 0xffc0_0456, 0x7fc0_0789, 0xffc0_0abc],
        [0x8000_0000, 0, 0x8000_0000, 0],
    ]);
    let mut random = 0x8d33_b280u32;
    for _ in 0..128 {
        let mut lanes = [0; 4];
        for value in &mut lanes {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            *value = random;
        }
        samples.push(lanes);
    }
    for mode in 0..4 {
        for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
            let instruction = word(mode, rn, rd);
            let mut code = Code(vec![instruction, 0xd420_0000]);
            let mut compiler = jit();
            for fpcr in [0, DN, FZ, DN | FZ, 3 << 22] {
                for &lanes in &samples {
                    let mut ctx = CpuContext::default();
                    ctx.pc = CODE;
                    ctx.x.fill(GUARD);
                    ctx.v.fill([GUARD, !GUARD]);
                    ctx.sp = GUARD;
                    ctx.nzcv = 0xb;
                    ctx.fpcr = fpcr;
                    ctx.fpsr = STICKY;
                    ctx.v[rn as usize] = pack(lanes);
                    let mut wanted = ctx.v;
                    let (answer, status) = expected(lanes, mode, fpcr);
                    wanted[rd as usize] = [answer as u64, 0];
                    let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
                    assert_eq!(result.stop, StopReason::Brk(0), "word{instruction:x}");
                    assert_eq!(result.retired, 2);
                    assert_eq!(
                        ctx.v, wanted,
                        "word{instruction:x} fpcr{fpcr:x} lanes{lanes:x?}"
                    );
                    assert_eq!(ctx.fpsr, STICKY | status);
                    assert_eq!(ctx.fpcr, fpcr);
                    assert_eq!(ctx.x, [GUARD; 31]);
                    assert_eq!(ctx.sp, GUARD);
                    assert_eq!(ctx.nzcv, 0xb);
                }
            }
        }
    }
}

#[test]
fn actual_wonder_six_reductions_keep_sources_and_clear_upper() {
    assert_eq!(word(3, 0, 1), 0x6eb0_c801);
    let instructions = [
        0x6eb0_c801,
        0x6e30_c863,
        0x6e30_c8a5,
        0x6eb0_c884,
        0x6eb0_c842,
        0x6e30_c800,
    ];
    let mut code = Code(instructions.into_iter().chain([0xd420_0000]).collect());
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v.fill([GUARD, !GUARD]);
    ctx.fpsr = STICKY;
    let lanes = [
        [0x3f80_0000, 0xc000_0000, 0x4000_0000, 0],
        [0; 4],
        [0, 0x8000_0000, 0x3f00_0000, 0xbf00_0000],
        [0x3f00_0000, 0x4000_0000, 0xbf80_0000, 0],
        [0x7fc0_0123, 0x4080_0000, 0x3f80_0000, 0x4000_0000],
        [0, 0x40c0_0000, 0x3f80_0000, 0xc000_0000],
    ];
    for (index, &input) in lanes.iter().enumerate() {
        ctx.v[index] = pack(input);
    }
    let mut wanted = ctx.v;
    for (mode, rn, rd) in [
        (3, 0, 1),
        (2, 3, 3),
        (2, 5, 5),
        (3, 4, 4),
        (3, 2, 2),
        (2, 0, 0),
    ] {
        wanted[rd] = [expected(lanes[rn], mode, 0).0 as u64, 0];
    }
    let result = jit().run_bounded(&mut ctx, &mut code, 7).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 7);
    assert_eq!(ctx.v, wanted);
    assert_eq!(ctx.fpsr, STICKY);
}

#[test]
fn overwritten_results_keep_exceptions_and_reserved_formats_reject() {
    for (lanes, fpcr, status) in [([0x7f80_0123, 0, 0, 0], 0, IOC), ([1, 0, 0, 0], FZ, IDC)] {
        let mut code = Code(vec![word(3, 1, 0), 0x1e27_03e0, 0xd420_0000]);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.fpcr = fpcr;
        ctx.fpsr = STICKY;
        ctx.v[1] = pack(lanes);
        let result = jit().run_bounded(&mut ctx, &mut code, 3).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(ctx.v[0], [0, 0]);
        assert_eq!(ctx.fpsr, STICKY | status);
    }
    for mode in 0..4 {
        for bad in [
            word(mode, 1, 0) & !(1 << 30),
            word(mode, 1, 0) | (1 << 22),
            word(mode, 1, 0) & !(1 << 29),
        ] {
            let mut code = Code(vec![bad]);
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.v.fill([GUARD, !GUARD]);
            ctx.fpsr = STICKY;
            let result = jit().run_bounded(&mut ctx, &mut code, 1).unwrap();
            assert!(
                matches!(result.stop, StopReason::Unsupported(_)),
                "word{bad:x}: {:?}",
                result.stop
            );
            assert_eq!(result.retired, 0);
            assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
            assert_eq!(ctx.fpsr, STICKY);
        }
    }
}

#[test]
fn pairwise_numeric_nan_behavior_is_not_global_signaling_selection() {
    for mode in [2, 3] {
        assert_eq!(
            expected([0x7f80_0123, 0, 0x3f80_0000, 0x4000_0000], mode, 0),
            (if mode == 2 { 0x4000_0000 } else { 0x3f80_0000 }, IOC)
        );
        assert_eq!(
            expected(
                [0x7fc0_0123, 0x7fc0_0456, 0x7fc0_0789, 0x7f80_0abc],
                mode,
                0
            ),
            (0x7fc0_0123, IOC)
        );
    }
}

#[test]
fn helper_preserves_live_gpr_simd_values_under_register_pressure() {
    for mode in 0..4 {
        for rn in [30, 31] {
            let mut words = Vec::new();
            for register in 0..16 {
                words.push(0x9100_0400 | (register << 5) | register); // ADD Xn,Xn,#1.
                words.push(0x4ea0_1c00 | (16 << 16) | (register << 5) | register); // ORR Vn.16B,Vn.16B,V16.16B.
            }
            words.push(word(mode, rn, 31));
            for pair in 0..8 {
                words.push(
                    0x8b00_0000
                        | ((pair * 2 + 1) << 16)
                        | ((pair * 2) << 5)
                        | if pair == 7 { 16 } else { 24 + pair },
                );
            }
            for register in 0..14 {
                words.push(0x4ea0_1c00 | (register << 16) | (register << 5) | (17 + register));
            }
            words.push(0xd420_0000);
            let count = words.len() as u64;
            let mut code = Code(words);
            let mut compiler = jit();
            for lanes in [
                [0x7f80_0123, 0, 0x3f80_0000, 0x4000_0000],
                [0x8000_0000, 0, 1, 0x807f_ffff],
            ] {
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.fpcr = FZ;
                ctx.fpsr = STICKY;
                ctx.nzcv = 0xb;
                ctx.sp = GUARD;
                for register in 0..31 {
                    ctx.x[register] = GUARD.wrapping_add(register as u64 * 137);
                }
                for register in 0..32 {
                    ctx.v[register] = [
                        GUARD.wrapping_add(register as u64),
                        !GUARD ^ register as u64,
                    ];
                }
                ctx.v[rn as usize] = pack(lanes);
                let mut wanted_x = ctx.x;
                let mut wanted_v = ctx.v;
                for register in 0..16 {
                    wanted_x[register] = wanted_x[register].wrapping_add(1);
                    wanted_v[register] = [
                        wanted_v[register][0] | wanted_v[16][0],
                        wanted_v[register][1] | wanted_v[16][1],
                    ];
                }
                for pair in 0..8 {
                    wanted_x[if pair == 7 { 16 } else { 24 + pair }] =
                        wanted_x[pair * 2].wrapping_add(wanted_x[pair * 2 + 1]);
                }
                for register in 0..14 {
                    wanted_v[17 + register] = wanted_v[register];
                }
                let (answer, flags) = expected(lanes, mode, FZ);
                wanted_v[31] = [answer as u64, 0];
                let result = compiler.run_bounded(&mut ctx, &mut code, count).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0));
                assert_eq!(result.retired, count);
                assert_eq!(ctx.x, wanted_x);
                assert_eq!(ctx.v, wanted_v, "mode{mode} rn{rn}");
                assert_eq!(ctx.fpsr, STICKY | flags);
                assert_eq!(ctx.fpcr, FZ);
                assert_eq!(ctx.nzcv, 0xb);
                assert_eq!(ctx.sp, GUARD);
            }
        }
    }
}
