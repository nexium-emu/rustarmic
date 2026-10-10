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
// kind0: MLA; kind1: MLS; kind2: ordinary MUL.
fn instruction(kind: u32, size: u32, q: bool, rn: u32, rm: u32, rd: u32) -> u32 {
    [0x0e20_9400, 0x2e20_9400, 0x0e20_9c00][kind as usize]
        | (size << 22)
        | (u32::from(q) << 30)
        | (rm << 16)
        | (rn << 5)
        | rd
}
fn bytes(v: [u64; 2]) -> [u8; 16] {
    let mut result = [0; 16];
    result[..8].copy_from_slice(&v[0].to_le_bytes());
    result[8..].copy_from_slice(&v[1].to_le_bytes());
    result
}
fn vector(v: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(v[..8].try_into().unwrap()),
        u64::from_le_bytes(v[8..].try_into().unwrap()),
    ]
}
// Independent exact u128 multiply/accumulate, reduced modulo lane width.
// Sign interpretation cannot alter the low modular product or accumulator.
fn expected(d: [u64; 2], n: [u64; 2], m: [u64; 2], kind: u32, size: u32, q: bool) -> [u64; 2] {
    let d = bytes(d);
    let n = bytes(n);
    let m = bytes(m);
    let count = 1usize << size;
    let width = count * 8;
    let mask = (1u128 << width) - 1;
    let mut answer = [0; 16];
    for start in (0..if q { 16 } else { 8 }).step_by(count) {
        let load = |source: &[u8; 16]| {
            let mut lane = [0; 8];
            lane[..count].copy_from_slice(&source[start..start + count]);
            u64::from_le_bytes(lane) as u128
        };
        let product = load(&n) * load(&m);
        let value = (match kind {
            0 => load(&d) + product,
            1 => load(&d).wrapping_sub(product),
            _ => product,
        }) & mask;
        answer[start..start + count].copy_from_slice(&(value as u64).to_le_bytes()[..count]);
    }
    vector(answer)
}
fn pattern(size: u32, seed: usize) -> [u64; 2] {
    let count = 1usize << size;
    let width = count * 8;
    let sign = 1u64 << (width - 1);
    let mask = (1u64 << width) - 1;
    let values = [
        0,
        1,
        2,
        3,
        sign - 1,
        sign,
        sign + 1,
        mask,
        0xaaaa_5555 & mask,
    ];
    let mut result = [0; 16];
    for lane in 0..16 / count {
        result[lane * count..(lane + 1) * count]
            .copy_from_slice(&values[(seed + lane * 5) % values.len()].to_le_bytes()[..count]);
    }
    vector(result)
}
fn cases(size: u32, triples: bool) -> Vec<([u64; 2], [u64; 2], [u64; 2])> {
    let mut output = Vec::new();
    for d in 0..if triples { 9 } else { 1 } {
        for n in 0..9 {
            for m in 0..9 {
                output.push((pattern(size, d), pattern(size, n), pattern(size, m)));
            }
        }
    }
    let mut random = 0x933b_bb8c_4ea1_9402u64;
    for _ in 0..64 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        output.push(([next(), next()], [next(), next()], [next(), next()]));
    }
    output
}
fn run_case(
    compiler: &mut Jit,
    code: &mut Code,
    kind: u32,
    size: u32,
    q: bool,
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
    let mut want = ctx.v;
    want[rd as usize] = expected(
        want[rd as usize],
        want[rn as usize],
        want[rm as usize],
        kind,
        size,
        q,
    );
    let result = compiler.run_bounded(&mut ctx, code, 2).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0), "{:08x}", code.0[0]);
    assert_eq!(result.retired, 2);
    assert_eq!(ctx.v, want, "word{:08x} d{d:x?} n{n:x?} m{m:x?}", code.0[0]);
    assert_eq!(ctx.x, [GUARD; 31]);
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpcr, 0x03c0_0000);
    assert_eq!(ctx.fpsr, FLAGS);
}
fn matrix(kinds: &[u32], baseline: bool, triples: bool) {
    for size in 0..3 {
        let samples = cases(size, triples);
        for q in [false, true] {
            for &kind in kinds {
                for (rn, rm, rd) in ALIASES {
                    let mut code = Code(vec![instruction(kind, size, q, rn, rm, rd), BRK]);
                    let mut compiler = jit(baseline);
                    for &(d, n, m) in &samples {
                        run_case(&mut compiler, &mut code, kind, size, q, rn, rm, rd, d, n, m);
                    }
                }
            }
        }
    }
}
#[test]
fn all_mla_mls_formats_accumulator_products_aliases_and_overflows() {
    matrix(&[0, 1], false, true);
}
#[test]
fn ordinary_mul_all_legal_formats_including_bytes_preserves_sources() {
    matrix(&[2], false, false);
}
#[test]
fn sse41_baseline_without_optional_host_features_matches_modular_oracle() {
    matrix(&[0, 1, 2], true, false);
}
#[test]
fn every_ordered_byte_product_matches_mla_mls_mul_in_all_active_lanes() {
    for q in [false, true] {
        for kind in 0..3 {
            let mut code = Code(vec![instruction(kind, 0, q, 1, 2, 0), BRK]);
            let mut compiler = jit(true);
            for a in 0..256u16 {
                for b in 0..256u16 {
                    let mut d = [0; 16];
                    let mut n = [0; 16];
                    let mut m = [0; 16];
                    for lane in 0..16 {
                        d[lane] = (a.wrapping_mul(b).wrapping_add(lane as u16 * 37)) as u8;
                        n[lane] = (a as u8).wrapping_add(lane as u8 * 13);
                        m[lane] = (b as u8).wrapping_add(lane as u8 * 7);
                    }
                    run_case(
                        &mut compiler,
                        &mut code,
                        kind,
                        0,
                        q,
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
#[test]
fn exact_wonder_mla_word_has_literal_four_word_results() {
    assert_eq!(instruction(0, 2, true, 0, 1, 2), 0x4ea1_9402);
    let mut code = Code(vec![0x4ea1_9402, BRK]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v[0] = [0xffff_ffff_0000_0003, 0x8000_0000_ffff_fffe];
    ctx.v[1] = [0x0000_0002_0000_0007, 0x0000_0002_ffff_ffff];
    ctx.v[2] = [0x0000_0001_ffff_ffff, 0xffff_ffff_7fff_ffff];
    ctx.nzcv = 0xb;
    ctx.fpcr = 0x03c0_0000;
    ctx.fpsr = FLAGS;
    let source0 = ctx.v[0];
    let source1 = ctx.v[1];
    let result = jit(false).run_bounded(&mut ctx, &mut code, 2).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 2);
    assert_eq!(ctx.v[2], [0xffff_ffff_0000_0014, 0xffff_ffff_8000_0001]);
    assert_eq!(ctx.v[0], source0);
    assert_eq!(ctx.v[1], source1);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpcr, 0x03c0_0000);
    assert_eq!(ctx.fpsr, FLAGS);
}
#[test]
fn reserved_d_vectors_do_not_retire_or_write() {
    for kind in 0..3 {
        for q in [false, true] {
            let word = instruction(kind, 3, q, 31, 31, 31);
            let mut code = Code(vec![word]);
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.v.fill([GUARD, !GUARD]);
            ctx.x.fill(GUARD);
            ctx.sp = GUARD;
            ctx.nzcv = 0xb;
            ctx.fpcr = 0x03c0_0000;
            ctx.fpsr = FLAGS;
            let result = jit(false).run_bounded(&mut ctx, &mut code, 1).unwrap();
            assert!(
                matches!(result.stop, StopReason::Unsupported(_)),
                "word{word:x}: {:?}",
                result.stop
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
#[test]
fn multiply_accumulate_preserves_live_integer_and_vector_values_under_pressure() {
    for size in 0..3 {
        for q in [false, true] {
            for kind in 0..3 {
                let mut words = Vec::new();
                for r in 0..16 {
                    words.push(0x9100_0400 | (r << 5) | r);
                    words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
                }
                words.push(instruction(kind, size, q, 30, 31, 31));
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
                    ctx.x[r] = GUARD.wrapping_add(r as u64 * 137);
                }
                for r in 0..32 {
                    ctx.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
                }
                ctx.v[30] = pattern(size, 6);
                ctx.v[31] = pattern(size, 7);
                let mut want_x = ctx.x;
                let mut want_v = ctx.v;
                for r in 0..16 {
                    want_x[r] = want_x[r].wrapping_add(1);
                    want_v[r] = [want_v[r][0] | want_v[16][0], want_v[r][1] | want_v[16][1]];
                }
                want_v[31] = expected(want_v[31], want_v[30], want_v[31], kind, size, q);
                for p in 0..8 {
                    want_x[if p == 7 { 16 } else { 24 + p }] =
                        want_x[p * 2].wrapping_add(want_x[p * 2 + 1]);
                }
                for r in 0..14 {
                    want_v[17 + r] = want_v[r];
                }
                let result = jit(true).run_bounded(&mut ctx, &mut code, count).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0));
                assert_eq!(result.retired, count);
                assert_eq!(ctx.x, want_x);
                assert_eq!(ctx.v, want_v);
                assert_eq!(ctx.sp, GUARD);
                assert_eq!(ctx.nzcv, 0xb);
                assert_eq!(ctx.fpcr, 0x03c0_0000);
                assert_eq!(ctx.fpsr, FLAGS);
            }
        }
    }
}
