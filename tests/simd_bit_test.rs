use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const ALIASES: [(u32, u32, u32); 10] = [
    (0, 1, 2),
    (0, 1, 0),
    (0, 1, 1),
    (0, 0, 2),
    (0, 0, 0),
    (31, 31, 31),
    (31, 1, 2),
    (0, 31, 2),
    (0, 1, 31),
    (31, 1, 1),
];
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
fn instruction(size: u32, q: bool, scalar: bool, rn: u32, rm: u32, rd: u32) -> u32 {
    (if scalar { 0x5e20_8c00 } else { 0x0e20_8c00 })
        | (u32::from(!scalar && q) << 30)
        | (size << 22)
        | (rm << 16)
        | (rn << 5)
        | rd
}
fn bytes(value: [u64; 2]) -> [u8; 16] {
    let mut result = [0; 16];
    result[..8].copy_from_slice(&value[0].to_le_bytes());
    result[8..].copy_from_slice(&value[1].to_le_bytes());
    result
}
fn vector(value: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(value[..8].try_into().unwrap()),
        u64::from_le_bytes(value[8..].try_into().unwrap()),
    ]
}
// Independent bytewise test of each lane's overlapping set bits. No native
// comparison, SIMD widening, or signed interpretation participates.
fn expected(n: [u64; 2], m: [u64; 2], size: u32, q: bool, scalar: bool) -> [u64; 2] {
    let n = bytes(n);
    let m = bytes(m);
    let count = 1usize << size;
    let active = if !scalar && q { 16 } else { 8 };
    let mut result = [0; 16];
    for start in (0..active).step_by(count) {
        let nonzero = (start..start + count).any(|byte| n[byte] & m[byte] != 0);
        result[start..start + count].fill(if nonzero { 0xff } else { 0 });
    }
    vector(result)
}
fn pattern(size: u32, start: usize) -> [u64; 2] {
    let count = 1usize << size;
    let width = count * 8;
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    let values = [
        0,
        1,
        2,
        3,
        sign - 1,
        sign,
        sign + 1,
        mask,
        0xaaaa_5555_aaaa_5555 & mask,
    ];
    let mut result = [0; 16];
    for lane in 0..16 / count {
        result[lane * count..(lane + 1) * count]
            .copy_from_slice(&values[(start + lane * 5) % values.len()].to_le_bytes()[..count]);
    }
    vector(result)
}
fn samples(size: u32) -> Vec<([u64; 2], [u64; 2])> {
    let mut output = Vec::new();
    for n in 0..9 {
        for m in 0..9 {
            output.push((pattern(size, n), pattern(size, m)));
        }
    }
    let mut random = 0x91a5_05c0_0e60_8c00u64;
    for _ in 0..64 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        output.push(([next(), next()], [next(), next()]));
    }
    output
}
fn run_cases(
    size: u32,
    q: bool,
    scalar: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    inputs: &[([u64; 2], [u64; 2])],
) {
    let word = instruction(size, q, scalar, rn, rm, rd);
    let mut code = Code(vec![word, 0xd420_0000]);
    let mut compiler = jit();
    for &(n, m) in inputs {
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.x.fill(GUARD);
        ctx.v.fill([GUARD, !GUARD]);
        ctx.v[rn as usize] = n;
        ctx.v[rm as usize] = m;
        ctx.sp = GUARD;
        ctx.nzcv = 0xb;
        ctx.fpcr = 0x03c0_0000;
        ctx.fpsr = FLAGS;
        let mut wanted = ctx.v;
        wanted[rd as usize] = expected(ctx.v[rn as usize], ctx.v[rm as usize], size, q, scalar);
        let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0), "word{word:x}");
        assert_eq!(result.retired, 2);
        assert_eq!(ctx.v, wanted, "word{word:x} n{n:x?} m{m:x?}");
        assert_eq!(ctx.x, [GUARD; 31]);
        assert_eq!(ctx.sp, GUARD);
        assert_eq!(ctx.nzcv, 0xb);
        assert_eq!(ctx.fpcr, 0x03c0_0000);
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
#[test]
fn all_seven_vector_arrangements_aliases_boundaries_and_random_values() {
    for (size, q) in [
        (0, false),
        (0, true),
        (1, false),
        (1, true),
        (2, false),
        (2, true),
        (3, true),
    ] {
        let inputs = samples(size);
        for (rn, rm, rd) in ALIASES {
            run_cases(size, q, false, rn, rm, rd, &inputs);
        }
    }
}
#[test]
fn scalar_d_aliases_high_bits_upper_poison_and_full_state_guards() {
    let mut inputs = samples(3);
    inputs.extend([
        ([0, u64::MAX], [0, u64::MAX]),
        ([1 << 63, 0], [1 << 63, 0]),
        (
            [0xaaaa_aaaa_aaaa_aaaa, u64::MAX],
            [0x5555_5555_5555_5555, u64::MAX],
        ),
    ]);
    for (rn, rm, rd) in ALIASES {
        run_cases(3, false, true, rn, rm, rd, &inputs);
    }
}
#[test]
fn every_pair_of_byte_values_matches_the_full_bitwise_truth_table() {
    let mut inputs = Vec::new();
    for n in 0u8..=255 {
        for m in 0u8..=255 {
            let mut left = [0; 16];
            let mut right = [0; 16];
            for lane in 0..16 {
                left[lane] = n.wrapping_add((lane as u8).wrapping_mul(37));
                right[lane] = m.wrapping_add((lane as u8).wrapping_mul(53));
            }
            inputs.push((vector(left), vector(right)));
        }
    }
    for q in [false, true] {
        run_cases(0, q, false, 0, 1, 2, &inputs);
    }
}
#[test]
fn exact_wonder_uxtl_bit_test_sxtl_chain_maps_nonzero_bytes_to_minus_one() {
    assert_eq!(instruction(1, false, false, 0, 0, 0), 0x0e60_8c00);
    let mut code = Code(vec![0x2f08_a400, 0x0e60_8c00, 0x0f10_a400, 0xd420_0000]);
    let mut compiler = jit();
    for mask in 0u8..16 {
        let mut source = [0xa5; 16];
        for lane in 0..4 {
            source[lane] = if mask & (1 << lane) != 0 { 0x80 } else { 0 };
        }
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.v.fill([GUARD, !GUARD]);
        ctx.v[0] = vector(source);
        ctx.fpsr = FLAGS;
        let mut wanted = ctx.v;
        let mut result = [0; 16];
        for lane in 0..4 {
            result[lane * 4..lane * 4 + 4].fill(if mask & (1 << lane) != 0 { 0xff } else { 0 });
        }
        wanted[0] = vector(result);
        let result = compiler.run_bounded(&mut ctx, &mut code, 4).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, 4);
        assert_eq!(ctx.v, wanted, "mask{mask:x}");
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
#[test]
fn reserved_vector_1d_and_scalar_nond_widths_do_not_retire_or_write() {
    for (size, q, scalar) in [
        (3, false, false),
        (0, false, true),
        (1, false, true),
        (2, false, true),
    ] {
        let word = instruction(size, q, scalar, 31, 31, 31);
        let mut code = Code(vec![word]);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.v.fill([GUARD, !GUARD]);
        ctx.x.fill(GUARD);
        ctx.sp = GUARD;
        ctx.nzcv = 0xb;
        ctx.fpcr = 0x03c0_0000;
        ctx.fpsr = FLAGS;
        let result = jit().run_bounded(&mut ctx, &mut code, 1).unwrap();
        assert!(
            matches!(result.stop, StopReason::Unsupported(_)),
            "word{word:x}"
        );
        assert_eq!(result.retired, 0);
        assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
        assert_eq!(ctx.x, [GUARD; 31]);
        assert_eq!(ctx.sp, GUARD);
        assert_eq!(ctx.nzcv, 0xb);
        assert_eq!(ctx.fpcr, 0x03c0_0000);
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
#[test]
fn scalar_and_vector_d_preserve_live_gpr_and_vector_values_under_pressure() {
    for scalar in [false, true] {
        let mut words = Vec::new();
        for r in 0..16 {
            words.push(0x9100_0400 | (r << 5) | r);
            words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
        }
        words.push(instruction(3, !scalar, scalar, 31, 30, 31));
        for p in 0..8 {
            words.push(
                0x8b00_0000
                    | ((p * 2 + 1) << 16)
                    | ((p * 2) << 5)
                    | if p == 7 { 16 } else { 24 + p },
            );
        }
        for r in 0..14 {
            words.push(0x4ea0_1c00 | (r << 16) | (r << 5) | (17 + r));
        }
        words.push(0xd420_0000);
        let count = words.len() as u64;
        let mut code = Code(words);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.nzcv = 0xb;
        ctx.fpsr = FLAGS;
        for r in 0..31 {
            ctx.x[r] = GUARD.wrapping_add(r as u64);
        }
        for r in 0..32 {
            ctx.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
        }
        ctx.v[31] = [0x8000_0000_0000_0000, 0xaaaa_aaaa_aaaa_aaaa];
        ctx.v[30] = [0x8000_0000_0000_0000, 0x5555_5555_5555_5555];
        let mut wanted_x = ctx.x;
        let mut wanted_v = ctx.v;
        for r in 0..16 {
            wanted_x[r] = wanted_x[r].wrapping_add(1);
            wanted_v[r] = [
                wanted_v[r][0] | wanted_v[16][0],
                wanted_v[r][1] | wanted_v[16][1],
            ];
        }
        wanted_v[31] = [u64::MAX, 0];
        for p in 0..8 {
            wanted_x[if p == 7 { 16 } else { 24 + p }] =
                wanted_x[p * 2].wrapping_add(wanted_x[p * 2 + 1]);
        }
        for r in 0..14 {
            wanted_v[17 + r] = wanted_v[r];
        }
        let result = jit().run_bounded(&mut ctx, &mut code, count).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, count);
        assert_eq!(ctx.x, wanted_x);
        assert_eq!(ctx.v, wanted_v);
        assert_eq!(ctx.nzcv, 0xb);
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
