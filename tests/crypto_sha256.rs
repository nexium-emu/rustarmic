use rustarmic::backend::cpu_features::detect_features;
use rustarmic::{CpuContext, CpuFeatures, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone, Copy, Debug)]
enum Kind {
    H,
    H2,
    Su0,
    Su1,
}
const KINDS: [Kind; 4] = [Kind::H, Kind::H2, Kind::Su0, Kind::Su1];
impl Kind {
    fn word(self, rd: u32, rn: u32, rm: u32) -> u32 {
        match self {
            Self::H => 0x5e00_4000 | (rm << 16) | (rn << 5) | rd,
            Self::H2 => 0x5e00_5000 | (rm << 16) | (rn << 5) | rd,
            Self::Su0 => 0x5e28_2800 | (rn << 5) | rd,
            Self::Su1 => 0x5e00_6000 | (rm << 16) | (rn << 5) | rd,
        }
    }
}
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
fn vector(w: [u32; 4]) -> [u64; 2] {
    [
        u64::from(w[0]) | (u64::from(w[1]) << 32),
        u64::from(w[2]) | (u64::from(w[3]) << 32),
    ]
}
fn words(v: [u64; 2]) -> [u32; 4] {
    [
        v[0] as u32,
        (v[0] >> 32) as u32,
        v[1] as u32,
        (v[1] >> 32) as u32,
    ]
}
fn rotate(value: u32, shift: u32) -> u32 {
    (value >> shift) | (value << (32 - shift))
}
fn small0(v: u32) -> u32 {
    rotate(v, 7) ^ rotate(v, 18) ^ (v >> 3)
}
fn small1(v: u32) -> u32 {
    rotate(v, 17) ^ rotate(v, 19) ^ (v >> 10)
}
fn big0(v: u32) -> u32 {
    rotate(v, 2) ^ rotate(v, 13) ^ rotate(v, 22)
}
fn big1(v: u32) -> u32 {
    rotate(v, 6) ^ rotate(v, 11) ^ rotate(v, 25)
}
fn round(mut state: [u32; 8], wk: u32) -> [u32; 8] {
    let choose = (state[4] & state[5]) ^ (!state[4] & state[6]);
    let majority = (state[0] & state[1]) ^ (state[0] & state[2]) ^ (state[1] & state[2]);
    let first = state[7]
        .wrapping_add(big1(state[4]))
        .wrapping_add(choose)
        .wrapping_add(wk);
    let second = big0(state[0]).wrapping_add(majority);
    let new_e = state[3].wrapping_add(first);
    state.copy_within(0..7, 1);
    state[0] = first.wrapping_add(second);
    state[4] = new_e;
    state
}
fn expected(kind: Kind, d: [u64; 2], n: [u64; 2], m: [u64; 2]) -> [u64; 2] {
    let d = words(d);
    let n = words(n);
    let m = words(m);
    let mut result = [0; 4];
    match kind {
        Kind::H | Kind::H2 => {
            let (abcd, efgh) = if matches!(kind, Kind::H) {
                (d, n)
            } else {
                (n, d)
            };
            let mut state = [0; 8];
            state[..4].copy_from_slice(&abcd);
            state[4..].copy_from_slice(&efgh);
            for wk in m {
                state = round(state, wk);
            }
            result.copy_from_slice(if matches!(kind, Kind::H) {
                &state[..4]
            } else {
                &state[4..]
            });
        }
        Kind::Su0 => {
            let next = [d[1], d[2], d[3], n[0]];
            for lane in 0..4 {
                result[lane] = d[lane].wrapping_add(small0(next[lane]));
            }
        }
        Kind::Su1 => {
            let addend = [n[1], n[2], n[3], m[0]];
            for lane in 0..4 {
                let sigma_input = if lane < 2 {
                    m[lane + 2]
                } else {
                    result[lane - 2]
                };
                result[lane] = d[lane]
                    .wrapping_add(addend[lane])
                    .wrapping_add(small1(sigma_input));
            }
        }
    }
    vector(result)
}
fn schedule(block: [u32; 16]) -> [u32; 64] {
    let mut w = [0; 64];
    w[..16].copy_from_slice(&block);
    for t in 16..64 {
        w[t] = small1(w[t - 2])
            .wrapping_add(w[t - 7])
            .wrapping_add(small0(w[t - 15]))
            .wrapping_add(w[t - 16]);
    }
    w
}
fn features() -> Vec<CpuFeatures> {
    let detected = detect_features();
    let mut portable = detected;
    portable.has_shani = false;
    portable.has_aesni = false;
    portable.has_avx = false;
    portable.has_fma = false;
    portable.has_gfni = false;
    let mut result = vec![portable];
    if detected.has_shani {
        result.push(detected);
    }
    result
}
fn compiler(features: CpuFeatures) -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        host_features: Some(features),
        ..JitConfig::default()
    })
    .unwrap()
}
fn context() -> CpuContext {
    let mut c = CpuContext::default();
    c.pc = CODE;
    for r in 0..31 {
        c.x[r] = GUARD.wrapping_add(r as u64 * 137);
    }
    for r in 0..32 {
        c.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
    }
    c.sp = GUARD;
    c.nzcv = 0xb;
    c.fpcr = 0x07c0_0000;
    c.fpsr = FLAGS;
    c.tpidr_el0 = GUARD ^ 1;
    c.tpidrro_el0 = GUARD ^ 2;
    c.exclusive_addr = GUARD ^ 3;
    c.exclusive_value = GUARD ^ 4;
    c.exclusive_size = 8;
    c.io_value = [GUARD ^ 5, GUARD ^ 6];
    c
}
fn guards(c: &CpuContext, x: [u64; 31], v: [[u64; 2]; 32]) {
    assert_eq!(c.x, x);
    assert_eq!(c.v, v);
    assert_eq!(c.sp, GUARD);
    assert_eq!(c.nzcv, 0xb);
    assert_eq!(c.fpcr, 0x07c0_0000);
    assert_eq!(c.fpsr, FLAGS);
    assert_eq!(c.tpidr_el0, GUARD ^ 1);
    assert_eq!(c.tpidrro_el0, GUARD ^ 2);
    assert_eq!(c.exclusive_addr, GUARD ^ 3);
    assert_eq!(c.exclusive_value, GUARD ^ 4);
    assert_eq!(c.exclusive_size, 8);
    assert_eq!(c.io_value, [GUARD ^ 5, GUARD ^ 6]);
    assert_eq!(c.mem_fault, 0);
    assert_eq!(c.should_halt, 0);
}
type Sample = ([u32; 4], [u32; 4], [u32; 4]);
fn samples() -> Vec<Sample> {
    let mut result = vec![
        ([0; 4], [0; 4], [0; 4]),
        ([u32::MAX; 4], [u32::MAX; 4], [u32::MAX; 4]),
        (
            [0x55555555; 4],
            [0xaaaaaaaa; 4],
            [0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210],
        ),
    ];
    for bit in 0..128 {
        let mut basis = [0; 4];
        basis[bit / 32] = 1 << (bit % 32);
        result.extend([
            (basis, [0; 4], [0; 4]),
            ([0; 4], basis, [0; 4]),
            ([0; 4], [0; 4], basis),
        ]);
    }
    let mut seed = 0x9a5a_3a05_e282_8201u64;
    for _ in 0..128 {
        let mut next = || {
            let mut w = [0; 4];
            for lane in &mut w {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *lane = seed as u32;
            }
            w
        };
        result.push((next(), next(), next()));
    }
    result
}
fn single(
    jit: &mut Jit,
    code: &mut Code,
    kind: Kind,
    rd: usize,
    rn: usize,
    rm: usize,
    sample: Sample,
) {
    let mut c = context();
    c.v[rd] = vector(sample.0);
    c.v[rn] = vector(sample.1);
    if !matches!(kind, Kind::Su0) {
        c.v[rm] = vector(sample.2);
    }
    let x = c.x;
    let mut v = c.v;
    v[rd] = expected(kind, v[rd], v[rn], v[rm]);
    let original = code.0.clone();
    let outcome = jit.run_bounded(&mut c, code, 1000).unwrap();
    assert_eq!(
        outcome.stop,
        StopReason::Brk(0),
        "{kind:?} rd{rd} rn{rn} rm{rm}"
    );
    assert_eq!(outcome.retired, 2);
    assert_eq!(code.0, original);
    guards(&c, x, v);
}

#[test]
fn all_four_sha256_operations_bits_overflow_random_aliases_and_all_registers() {
    let samples = samples();
    let mut aliases = vec![
        (0, 1, 2),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 0, 0),
        (31, 31, 31),
        (31, 0, 1),
        (0, 31, 1),
        (0, 1, 31),
        (16, 30, 31),
    ];
    aliases.extend((0..32).map(|r| (r, (r + 7) % 32, (r + 19) % 32)));
    for features in features() {
        for kind in KINDS {
            for &(rd, rn, rm) in &aliases {
                let mut code = Code(vec![kind.word(rd, rn, rm), BRK]);
                let mut jit = compiler(features);
                for &sample in &samples {
                    single(
                        &mut jit,
                        &mut code,
                        kind,
                        rd as usize,
                        rn as usize,
                        rm as usize,
                        sample,
                    );
                }
            }
        }
    }
}

#[test]
fn nist_abc_four_round_literal_and_exact_mk8d_schedule_word() {
    assert_eq!(Kind::Su0.word(0, 1, 0), 0x5e28_2820);
    let a: [u32; 4] = INITIAL[..4].try_into().unwrap();
    let e: [u32; 4] = INITIAL[4..].try_into().unwrap();
    let wk = [K[0].wrapping_add(0x61626380), K[1], K[2], K[3]];
    let want_a = [0xd550f666, 0xc8c347a7, 0x5a6ad9ad, 0x5d6aebcd];
    let want_e = [0x24e00850, 0xf92939eb, 0x78ce7989, 0xfa2a4622];
    assert_eq!(
        expected(Kind::H, vector(a), vector(e), vector(wk)),
        vector(want_a)
    );
    assert_eq!(
        expected(Kind::H2, vector(e), vector(a), vector(wk)),
        vector(want_e)
    );
    for features in features() {
        for (kind, d, n, answer) in [(Kind::H, a, e, want_a), (Kind::H2, e, a, want_e)] {
            let mut c = context();
            c.v[0] = vector(d);
            c.v[1] = vector(n);
            c.v[2] = vector(wk);
            let x = c.x;
            let mut v = c.v;
            v[0] = vector(answer);
            let mut code = Code(vec![kind.word(0, 1, 2), BRK]);
            let outcome = compiler(features)
                .run_bounded(&mut c, &mut code, 1000)
                .unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, 2);
            guards(&c, x, v);
        }
        let mut code = Code(vec![0x5e28_2820, BRK]);
        single(
            &mut compiler(features),
            &mut code,
            Kind::Su0,
            0,
            1,
            2,
            ([0x61626380, 0, 0, 0], [0; 4], [GUARD as u32; 4]),
        );
    }
}

fn guest_schedule(features: CpuFeatures, input: [u32; 16]) -> [u32; 64] {
    let reference = schedule(input);
    let mut result = [0; 64];
    result[..16].copy_from_slice(&input);
    let mut c = context();
    for group in 0..4 {
        c.v[group + 4] = vector(input[group * 4..group * 4 + 4].try_into().unwrap());
    }
    let x = c.x;
    let mut wanted = c.v;
    let mut code = Code(Vec::new());
    for group in 0..4u32 {
        code.0.extend([
            Kind::Su0.word(group + 4, (group + 1) % 4 + 4, 0),
            Kind::Su1.word(group + 4, (group + 2) % 4 + 4, (group + 3) % 4 + 4),
            BRK,
        ]);
    }
    let original = code.0.clone();
    let mut jit = compiler(features);
    for t in (16..64).step_by(4) {
        let group = (t / 4) % 4;
        c.pc = CODE + (group * 12) as u64;
        wanted[group + 4] = vector(reference[t..t + 4].try_into().unwrap());
        let outcome = jit.run_bounded(&mut c, &mut code, 1000).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, 3);
        guards(&c, x, wanted);
        assert_eq!(code.0, original);
        result[t..t + 4].copy_from_slice(&words(c.v[group + 4]));
    }
    assert_eq!(result, reference);
    result
}

#[test]
fn complete_64_word_schedule_checks_dependency_and_ring_register_order() {
    let mut abc = [0; 16];
    abc[0] = 0x61626380;
    abc[15] = 24;
    let mut empty = [0; 16];
    empty[0] = 0x80000000;
    for features in features() {
        for input in [
            abc,
            empty,
            [u32::MAX; 16],
            std::array::from_fn(|i| 0x10203040u32.wrapping_mul(i as u32 + 1)),
        ] {
            guest_schedule(features, input);
        }
    }
    let w = schedule(abc);
    assert_eq!(
        &w[16..20],
        &[0x61626380, 0x000f0000, 0x7da86405, 0x600003c6]
    );
}
fn parse_digest(text: &str) -> [u32; 8] {
    assert_eq!(text.len(), 64);
    std::array::from_fn(|i| u32::from_str_radix(&text[i * 8..i * 8 + 8], 16).unwrap())
}
fn guest_digest(features: CpuFeatures, message: &[u8]) -> [u32; 8] {
    let mut padded = message.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&((message.len() as u64) * 8).to_be_bytes());
    let mut state = INITIAL;
    for block in padded.chunks_exact(64) {
        let input = std::array::from_fn(|i| {
            u32::from_be_bytes(block[i * 4..i * 4 + 4].try_into().unwrap())
        });
        let w = guest_schedule(features, input);
        let mut c = context();
        c.v[0] = vector(state[..4].try_into().unwrap());
        c.v[1] = vector(state[4..].try_into().unwrap());
        c.v[20] = c.v[0];
        c.v[21] = c.v[1];
        for group in 0..16 {
            c.v[group + 4] = vector(std::array::from_fn(|lane| {
                w[group * 4 + lane].wrapping_add(K[group * 4 + lane])
            }));
        }
        let x = c.x;
        let mut wanted = c.v;
        let mut scalar = state;
        let mut program = Vec::new();
        for group in 0..16u32 {
            program.extend([
                0x4ea0_1c03,
                Kind::H.word(0, 1, group + 4),
                Kind::H2.word(1, 3, group + 4),
            ]);
            wanted[3] = wanted[0];
            for lane in 0..4 {
                scalar = round(
                    scalar,
                    w[group as usize * 4 + lane].wrapping_add(K[group as usize * 4 + lane]),
                );
            }
            wanted[0] = vector(scalar[..4].try_into().unwrap());
            wanted[1] = vector(scalar[4..].try_into().unwrap());
        }
        for i in 0..8 {
            scalar[i] = scalar[i].wrapping_add(state[i]);
        }
        wanted[0] = vector(scalar[..4].try_into().unwrap());
        wanted[1] = vector(scalar[4..].try_into().unwrap());
        program.extend([0x4eb4_8400, 0x4eb5_8421, BRK]);
        let count = program.len() as u64;
        let mut code = Code(program);
        let original = code.0.clone();
        let outcome = compiler(features)
            .run_bounded(&mut c, &mut code, 1000)
            .unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, count);
        guards(&c, x, wanted);
        assert_eq!(code.0, original);
        state[..4].copy_from_slice(&words(c.v[0]));
        state[4..].copy_from_slice(&words(c.v[1]));
    }
    state
}

#[test]
fn known_sha256_empty_abc_and_two_block_message_complete_guest_hashes() {
    for features in features() {
        for (message, answer) in [
            (
                &b""[..],
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                &b"abc"[..],
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            ),
            (
                &b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"[..],
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
            ),
        ] {
            assert_eq!(guest_digest(features, message), parse_digest(answer));
        }
    }
}

#[test]
fn helper_calls_preserve_live_vector_integer_values_under_pressure() {
    for features in features() {
        for kind in KINDS {
            let mut c = context();
            let mut x = c.x;
            let mut v = c.v;
            let mut program = Vec::new();
            for r in 0..16usize {
                program.push(0x9100_0400 | ((r as u32) << 5) | r as u32);
                program.push(0x4ea0_1c00 | (16 << 16) | ((r as u32) << 5) | r as u32);
                x[r] = x[r].wrapping_add(1);
                v[r] = [v[r][0] | v[16][0], v[r][1] | v[16][1]];
            }
            program.push(kind.word(31, 30, 29));
            v[31] = expected(kind, v[31], v[30], v[29]);
            for p in 0..8usize {
                let rd = if p == 7 { 16 } else { 24 + p };
                program.push(
                    0x8b00_0000
                        | (((p * 2 + 1) as u32) << 16)
                        | (((p * 2) as u32) << 5)
                        | rd as u32,
                );
                x[rd] = x[p * 2].wrapping_add(x[p * 2 + 1]);
            }
            for r in 0..14usize {
                program
                    .push(0x4ea0_1c00 | ((r as u32) << 16) | ((r as u32) << 5) | (17 + r) as u32);
                v[17 + r] = v[r];
            }
            program.push(BRK);
            let count = program.len() as u64;
            let mut code = Code(program);
            let outcome = compiler(features)
                .run_bounded(&mut c, &mut code, 1000)
                .unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, count);
            guards(&c, x, v);
        }
    }
}

#[test]
fn one_instruction_transient_budget_preserves_partial_schedule() {
    for features in features() {
        let mut c = context();
        let x = c.x;
        let mut v = c.v;
        let mut code = Code(vec![Kind::Su0.word(0, 1, 0), Kind::Su1.word(0, 2, 3), BRK]);
        let mut jit = compiler(features);
        v[0] = expected(Kind::Su0, v[0], v[1], v[2]);
        let outcome = jit.run_bounded(&mut c, &mut code, 1).unwrap();
        assert_eq!(outcome.stop, StopReason::BudgetExhausted);
        assert_eq!(outcome.retired, 1);
        assert_eq!(c.pc, CODE + 4);
        guards(&c, x, v);
        v[0] = expected(Kind::Su1, v[0], v[2], v[3]);
        let outcome = jit.run_bounded(&mut c, &mut code, 1000).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, 2);
        guards(&c, x, v);
    }
}

#[test]
fn fixed_sha256_reserved_size_bits_reject_without_retiring() {
    for features in features() {
        for kind in KINDS {
            for changed in [1 << 22, 1 << 23] {
                let instruction = kind.word(0, 1, 2) ^ changed;
                let mut c = context();
                let x = c.x;
                let v = c.v;
                let mut code = Code(vec![instruction, BRK]);
                let outcome = compiler(features)
                    .run_bounded(&mut c, &mut code, 1)
                    .unwrap();
                assert!(
                    matches!(outcome.stop, StopReason::Unsupported(_)),
                    "word{instruction:08x}: {:?}",
                    outcome.stop
                );
                assert_eq!(outcome.retired, 0);
                assert_eq!(c.pc, CODE);
                guards(&c, x, v);
            }
        }
    }
}

#[test]
fn sha_feature_gate_selects_distinct_helper_without_inline_sha_or_avx() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    use rustarmic::backend::cpu_features::with_features;
    use rustarmic::backend::emit_block;
    use rustarmic::frontend::{TranslateOptions, translate_block_into};
    use rustarmic::ir::Block;
    use rustarmic::optimizer::optimize;
    let detected = detect_features();
    let mut masked = detected;
    masked.has_shani = false;
    masked.has_avx = false;
    let mut hardware = masked;
    hardware.has_shani = detected.has_shani;
    assert!(!features()[0].has_shani);
    for kind in KINDS {
        let code = [kind.word(0, 1, 2), BRK];
        let mut block = Block::new(CODE);
        translate_block_into(
            &mut block,
            CODE,
            &mut |address| code.get(address.checked_sub(CODE)? as usize / 4).copied(),
            TranslateOptions::default(),
        )
        .unwrap();
        optimize(&mut block);
        let software = with_features(masked, || emit_block(&block)).unwrap();
        let accelerated = with_features(hardware, || emit_block(&block)).unwrap();
        for emitted in [&software, &accelerated] {
            let instructions: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE)
                .into_iter()
                .map(|instruction| {
                    assert!(!instruction.is_invalid());
                    instruction.mnemonic()
                })
                .collect();
            assert!(instructions.contains(&Mnemonic::Call));
            assert!(!instructions.iter().any(|mnemonic| matches!(
                mnemonic,
                Mnemonic::Sha256rnds2 | Mnemonic::Sha256msg1 | Mnemonic::Sha256msg2
            )));
        }
        if detected.has_shani {
            assert_ne!(software.code, accelerated.code, "{kind:?}");
        } else {
            assert_eq!(software.code, accelerated.code, "{kind:?}");
        }
    }
}
