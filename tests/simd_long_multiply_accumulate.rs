use rustarmic::backend::cpu_features::CpuFeatures;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const BRK: u32 = 0xd420_0000;
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
fn jit(baseline: bool) -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        host_features: baseline.then_some(CpuFeatures {
            has_sse41: true,
            has_ssse3: true,
            ..CpuFeatures::default()
        }),
        ..JitConfig::default()
    })
    .unwrap()
}
fn instruction(
    signed: bool,
    subtract: bool,
    size: u32,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
) -> u32 {
    0x0e20_8000
        | (u32::from(!signed) << 29)
        | (u32::from(subtract) << 13)
        | (size << 22)
        | (u32::from(upper) << 30)
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
// Both original narrow operands are independently sign/zero extended before
// exact i128 multiplication. The full wide accumulator wraps modulo 2^(2w).
fn expected(
    d: [u64; 2],
    n: [u64; 2],
    m: [u64; 2],
    signed: bool,
    subtract: bool,
    size: u32,
    upper: bool,
) -> [u64; 2] {
    let (d, n, m) = (bytes(d), bytes(n), bytes(m));
    let count = 1usize << size;
    let width = count * 8;
    let narrow_modulus = 1i128 << width;
    let mask = (1i128 << (width * 2)) - 1;
    let mut answer = [0; 16];
    for lane in 0..8 / count {
        let source = usize::from(upper) * 8 + lane * count;
        let destination = lane * count * 2;
        let load = |input: &[u8; 16], offset: usize, length: usize| {
            let mut bits = [0; 8];
            bits[..length].copy_from_slice(&input[offset..offset + length]);
            u64::from_le_bytes(bits) as i128
        };
        let extend = |value: i128| {
            if signed && value & (narrow_modulus / 2) != 0 {
                value - narrow_modulus
            } else {
                value
            }
        };
        let product = extend(load(&n, source, count)) * extend(load(&m, source, count));
        let accumulator = load(&d, destination, count * 2);
        let value = ((accumulator + if subtract { -product } else { product }) & mask) as u64;
        answer[destination..destination + count * 2]
            .copy_from_slice(&value.to_le_bytes()[..count * 2]);
    }
    vector(answer)
}
fn boundaries(width: usize) -> [u64; 9] {
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    [0, 1, 2, 3, sign - 1, sign, sign + 1, mask - 1, mask]
}
fn pattern(size: u32, wide: bool, rotation: usize) -> [u64; 2] {
    let count = (1usize << size) * if wide { 2 } else { 1 };
    let values = boundaries(count * 8);
    let mut answer = [0; 16];
    for lane in 0..16 / count {
        answer[lane * count..lane * count + count]
            .copy_from_slice(&values[(rotation + lane * 5) % 9].to_le_bytes()[..count]);
    }
    vector(answer)
}
fn samples(size: u32, triples: bool) -> Vec<([u64; 2], [u64; 2], [u64; 2])> {
    let mut result = Vec::new();
    for d in 0..if triples { 9 } else { 1 } {
        for n in 0..9 {
            for m in 0..9 {
                result.push((
                    pattern(size, true, d),
                    pattern(size, false, n),
                    pattern(size, false, m),
                ));
            }
        }
    }
    let mut random = 0x08c3_09b8_6ea7_a202u64;
    for _ in 0..32 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        result.push(([next(), next()], [next(), next()], [next(), next()]));
    }
    result
}
#[allow(clippy::too_many_arguments)]
fn run_case(
    compiler: &mut Jit,
    code: &mut Code,
    signed: bool,
    subtract: bool,
    size: u32,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    d: [u64; 2],
    n: [u64; 2],
    m: [u64; 2],
) {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x.fill(GUARD);
    ctx.v.fill([GUARD, !GUARD]);
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    ctx.fpcr = 0x03c0_0000;
    ctx.fpsr = FLAGS;
    ctx.v[rd as usize] = d;
    ctx.v[rn as usize] = n;
    ctx.v[rm as usize] = m;
    let mut wanted = ctx.v;
    wanted[rd as usize] = expected(
        wanted[rd as usize],
        wanted[rn as usize],
        wanted[rm as usize],
        signed,
        subtract,
        size,
        upper,
    );
    let result = compiler.run_bounded(&mut ctx, code, 2).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0), "word{:08x}", code.0[0]);
    assert_eq!(result.retired, 2);
    assert_eq!(
        ctx.v, wanted,
        "word{:08x} d{d:x?} n{n:x?} m{m:x?}",
        code.0[0]
    );
    assert_eq!(ctx.x, [GUARD; 31]);
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpcr, 0x03c0_0000);
    assert_eq!(ctx.fpsr, FLAGS);
}
fn matrix(baseline: bool, triples: bool) {
    for size in 0..3 {
        let values = samples(size, triples);
        for signed in [false, true] {
            for subtract in [false, true] {
                for upper in [false, true] {
                    for (rn, rm, rd) in ALIASES {
                        let mut code = Code(vec![
                            instruction(signed, subtract, size, upper, rn, rm, rd),
                            BRK,
                        ]);
                        let mut compiler = jit(baseline);
                        for &(d, n, m) in &values {
                            run_case(
                                &mut compiler,
                                &mut code,
                                signed,
                                subtract,
                                size,
                                upper,
                                rn,
                                rm,
                                rd,
                                d,
                                n,
                                m,
                            );
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn all_24_forms_accumulator_product_boundaries_aliases_and_random_values() {
    matrix(false, true);
}
#[test]
fn sse41_baseline_all_forms_use_original_halves_and_full_accumulator() {
    matrix(true, false);
}
#[test]
fn every_ordered_byte_pair_sign_add_sub_and_source_half_has_exact_product() {
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                let mut code = Code(vec![instruction(signed, subtract, 0, upper, 1, 2, 0), BRK]);
                let mut compiler = jit(true);
                for a in 0..256u16 {
                    for b in 0..256u16 {
                        let mut n = [0xa3; 16];
                        let mut m = [0x5c; 16];
                        let mut d = [0; 16];
                        for lane in 0..8 {
                            let start = usize::from(upper) * 8 + lane;
                            n[start] = (a as u8).wrapping_add(lane as u8 * 13);
                            m[start] = (b as u8).wrapping_add(lane as u8 * 7);
                            d[lane * 2..lane * 2 + 2].copy_from_slice(
                                &a.wrapping_mul(257)
                                    .wrapping_add(b.wrapping_mul(129))
                                    .wrapping_add(lane as u16 * 37)
                                    .to_le_bytes(),
                            );
                        }
                        run_case(
                            &mut compiler,
                            &mut code,
                            signed,
                            subtract,
                            0,
                            upper,
                            1,
                            2,
                            0,
                            vector(d),
                            vector(n),
                            vector(m),
                        );
                    }
                }
            }
        }
    }
}
#[test]
fn exact_wonder_unsigned_four_instruction_chain_has_literal_wrap_results() {
    let instructions = [0x6ea7_a202, 0x2ea7_a201, 0x6ea7_8205, 0x2ea7_8204];
    assert_eq!(instruction(false, true, 2, true, 16, 7, 2), instructions[0]);
    assert_eq!(
        instruction(false, true, 2, false, 16, 7, 1),
        instructions[1]
    );
    assert_eq!(
        instruction(false, false, 2, true, 16, 7, 5),
        instructions[2]
    );
    assert_eq!(
        instruction(false, false, 2, false, 16, 7, 4),
        instructions[3]
    );
    for length in 1..=4 {
        let mut code = Code(
            instructions[..length]
                .iter()
                .copied()
                .chain([BRK])
                .collect(),
        );
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.x.fill(GUARD);
        ctx.v.fill([GUARD, !GUARD]);
        ctx.sp = GUARD;
        ctx.nzcv = 0xb;
        ctx.fpcr = 0x03c0_0000;
        ctx.fpsr = FLAGS;
        ctx.v[16] = [0x8000_0000_ffff_ffff, 0x7fff_ffff_0000_0001];
        ctx.v[7] = [0xffff_ffff_0000_0002, 0x0000_0003_ffff_ffff];
        for rd in [1, 2, 4, 5] {
            ctx.v[rd] = [0, u64::MAX];
        }
        let mut wanted = ctx.v;
        for (rd, literal) in [
            (2, [0xffff_ffff_0000_0001, 0xffff_fffe_8000_0002]),
            (1, [0xffff_fffe_0000_0002, 0x8000_0000_7fff_ffff]),
            (5, [0x0000_0000_ffff_ffff, 0x0000_0001_7fff_fffc]),
            (4, [0x0000_0001_ffff_fffe, 0x7fff_ffff_7fff_ffff]),
        ]
        .into_iter()
        .take(length)
        {
            wanted[rd] = literal;
        }
        let result = jit(false)
            .run_bounded(&mut ctx, &mut code, (length + 1) as u64)
            .unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, (length + 1) as u64);
        assert_eq!(ctx.v, wanted, "chain prefix{length}");
        assert_eq!(ctx.x, [GUARD; 31]);
        assert_eq!(ctx.sp, GUARD);
        assert_eq!(ctx.nzcv, 0xb);
        assert_eq!(ctx.fpcr, 0x03c0_0000);
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
#[test]
fn reserved_d_to_128_source_width_never_retires_or_writes() {
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                let word = instruction(signed, subtract, 3, upper, 31, 31, 31);
                let mut code = Code(vec![word]);
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.x.fill(GUARD);
                ctx.v.fill([GUARD, !GUARD]);
                ctx.sp = GUARD;
                ctx.nzcv = 0xb;
                ctx.fpcr = 0x03c0_0000;
                ctx.fpsr = FLAGS;
                let result = jit(false).run_bounded(&mut ctx, &mut code, 1).unwrap();
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
    }
}
#[test]
fn all_forms_keep_live_gpr_and_vector_values_under_pressure() {
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                for size in 0..3 {
                    let mut words = Vec::new();
                    for r in 0..16 {
                        words.push(0x9100_0400 | (r << 5) | r);
                        words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
                    }
                    words.push(instruction(signed, subtract, size, upper, 31, 30, 31));
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
                    words.push(BRK);
                    let count = words.len() as u64;
                    let mut code = Code(words);
                    let mut ctx = CpuContext::default();
                    ctx.pc = CODE;
                    ctx.sp = GUARD;
                    ctx.nzcv = 0xb;
                    ctx.fpcr = 0x03c0_0000;
                    ctx.fpsr = FLAGS;
                    for r in 0..31 {
                        ctx.x[r] = GUARD.wrapping_add(r as u64);
                    }
                    for r in 0..32 {
                        ctx.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
                    }
                    ctx.v[31] = pattern(size, false, 5);
                    ctx.v[30] = pattern(size, false, 4);
                    let mut wanted_x = ctx.x;
                    let mut wanted_v = ctx.v;
                    for r in 0..16 {
                        wanted_x[r] = wanted_x[r].wrapping_add(1);
                        wanted_v[r] = [
                            wanted_v[r][0] | wanted_v[16][0],
                            wanted_v[r][1] | wanted_v[16][1],
                        ];
                    }
                    wanted_v[31] = expected(
                        wanted_v[31],
                        wanted_v[31],
                        wanted_v[30],
                        signed,
                        subtract,
                        size,
                        upper,
                    );
                    for p in 0..8 {
                        wanted_x[if p == 7 { 16 } else { 24 + p }] =
                            wanted_x[p * 2].wrapping_add(wanted_x[p * 2 + 1]);
                    }
                    for r in 0..14 {
                        wanted_v[17 + r] = wanted_v[r];
                    }
                    let result = jit(true).run_bounded(&mut ctx, &mut code, count).unwrap();
                    assert_eq!(result.stop, StopReason::Brk(0));
                    assert_eq!(result.retired, count);
                    assert_eq!(ctx.v, wanted_v);
                    assert_eq!(ctx.x, wanted_x);
                    assert_eq!(ctx.sp, GUARD);
                    assert_eq!(ctx.nzcv, 0xb);
                    assert_eq!(ctx.fpcr, 0x03c0_0000);
                    assert_eq!(ctx.fpsr, FLAGS);
                }
            }
        }
    }
}
