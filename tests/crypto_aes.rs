use rustarmic::backend::cpu_features::detect_features;
use rustarmic::{CpuContext, CpuFeatures, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];
const fn inverse_sbox() -> [u8; 256] {
    let mut result = [0; 256];
    let mut i = 0;
    while i < 256 {
        result[SBOX[i] as usize] = i as u8;
        i += 1;
    }
    result
}
const INV_SBOX: [u8; 256] = inverse_sbox();

#[derive(Clone, Copy, Debug)]
enum Kind {
    Encrypt,
    Decrypt,
    Mix,
    InverseMix,
}
const KINDS: [Kind; 4] = [Kind::Encrypt, Kind::Decrypt, Kind::Mix, Kind::InverseMix];
impl Kind {
    fn word(self, rd: u32, rn: u32) -> u32 {
        let base = match self {
            Self::Encrypt => 0x4e28_4800,
            Self::Decrypt => 0x4e28_5800,
            Self::Mix => 0x4e28_6800,
            Self::InverseMix => 0x4e28_7800,
        };
        base | (rn << 5) | rd
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
fn vector(b: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(b[..8].try_into().unwrap()),
        u64::from_le_bytes(b[8..].try_into().unwrap()),
    ]
}
fn bytes(v: [u64; 2]) -> [u8; 16] {
    let mut b = [0; 16];
    b[..8].copy_from_slice(&v[0].to_le_bytes());
    b[8..].copy_from_slice(&v[1].to_le_bytes());
    b
}
fn hex(text: &str) -> [u8; 16] {
    assert_eq!(text.len(), 32);
    let mut result = [0; 16];
    for (index, byte) in result.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).unwrap();
    }
    result
}
fn multiply(a: u8, b: u8) -> u8 {
    let mut polynomial = 0u16;
    for i in 0..8 {
        if b & (1 << i) != 0 {
            polynomial ^= u16::from(a) << i;
        }
    }
    for i in (8..15).rev() {
        if polynomial & (1 << i) != 0 {
            polynomial ^= 0x11b << (i - 8);
        }
    }
    polynomial as u8
}
fn expected(kind: Kind, d: [u64; 2], n: [u64; 2]) -> [u64; 2] {
    let mut state = bytes(n);
    if matches!(kind, Kind::Encrypt | Kind::Decrypt) {
        let old = bytes(d);
        for i in 0..16 {
            state[i] ^= old[i];
        }
    }
    let mut result = [0; 16];
    match kind {
        Kind::Encrypt | Kind::Decrypt => {
            let inverse = matches!(kind, Kind::Decrypt);
            let table = if inverse { &INV_SBOX } else { &SBOX };
            for column in 0..4 {
                for row in 0..4 {
                    let source_column = if inverse {
                        (column + 4 - row) % 4
                    } else {
                        (column + row) % 4
                    };
                    result[column * 4 + row] = table[state[source_column * 4 + row] as usize];
                }
            }
        }
        Kind::Mix | Kind::InverseMix => {
            let first_row = if matches!(kind, Kind::Mix) {
                [2, 3, 1, 1]
            } else {
                [14, 11, 13, 9]
            };
            for column in 0..4 {
                for row in 0..4 {
                    for source_row in 0..4 {
                        result[column * 4 + row] ^= multiply(
                            first_row[(source_row + 4 - row) % 4],
                            state[column * 4 + source_row],
                        );
                    }
                }
            }
        }
    }
    vector(result)
}
fn features() -> Vec<CpuFeatures> {
    let detected = detect_features();
    let mut portable = detected;
    portable.has_aesni = false;
    portable.has_avx = false;
    portable.has_fma = false;
    portable.has_gfni = false;
    let mut result = vec![portable];
    if detected.has_aesni {
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
fn check_guards(c: &CpuContext, x: [u64; 31], v: [[u64; 2]; 32]) {
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
fn check_single(
    jit: &mut Jit,
    code: &mut Code,
    kind: Kind,
    rd: usize,
    rn: usize,
    d: [u8; 16],
    n: [u8; 16],
) {
    let mut c = context();
    c.v[rd] = vector(d);
    c.v[rn] = vector(n);
    let x = c.x;
    let mut v = c.v;
    v[rd] = expected(kind, v[rd], v[rn]);
    let original_code = code.0.clone();
    let outcome = jit.run_bounded(&mut c, code, 1000).unwrap();
    assert_eq!(outcome.stop, StopReason::Brk(0));
    assert_eq!(outcome.retired, 2);
    assert_eq!(code.0, original_code);
    check_guards(&c, x, v);
}
fn samples() -> Vec<([u8; 16], [u8; 16])> {
    let mut result = vec![([0; 16], [0; 16]), ([255; 16], [255; 16])];
    for b in 0..=255u8 {
        let mut state = [0; 16];
        let mut source = [0; 16];
        for lane in 0..16 {
            state[lane] = b.wrapping_add(lane as u8 * 13);
            source[lane] = b.wrapping_mul(17).wrapping_add(lane as u8 * 7);
        }
        result.extend([(state, [0; 16]), ([0; 16], state), (state, source)]);
    }
    for bit in 0..128 {
        let mut basis = [0; 16];
        basis[bit / 8] = 1 << (bit % 8);
        result.extend([([0; 16], basis), (basis, [0; 16])]);
    }
    let mut seed = 0x9a58_7544_e287_8211u64;
    for _ in 0..64 {
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        result.push((bytes([next(), next()]), bytes([next(), next()])));
    }
    result
}

#[test]
fn independent_oracle_matches_published_aes_round_and_inverse_vectors() {
    let plain = vector(hex("00112233445566778899aabbccddeeff"));
    let key = vector(hex("000102030405060708090a0b0c0d0e0f"));
    let round = vector(hex("6353e08c0960e104cd70b751bacad0e7"));
    let mixed = vector(hex("5f72641557f5bc92f7be3b291db9f91a"));
    assert_eq!(expected(Kind::Encrypt, plain, key), round);
    assert_eq!(expected(Kind::Mix, [0; 2], round), mixed);
    assert_eq!(expected(Kind::InverseMix, [0; 2], mixed), round);
    assert_eq!(
        expected(Kind::Decrypt, round, [0; 2]),
        vector(hex("00102030405060708090a0b0c0d0e0f0"))
    );
    assert_eq!(SBOX[0], 0x63);
    assert_eq!(INV_SBOX[0x63], 0);
    assert_eq!(multiply(0x57, 0x13), 0xfe);
}

#[test]
fn all_aes_instructions_byte_basis_sboxes_aliases_and_feature_paths() {
    let samples = samples();
    for features in features() {
        for kind in KINDS {
            for (rd, rn) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31), (16, 30)] {
                let mut code = Code(vec![kind.word(rd, rn), BRK]);
                let mut jit = compiler(features);
                for &(d, n) in &samples {
                    check_single(&mut jit, &mut code, kind, rd as usize, rn as usize, d, n);
                }
            }
        }
    }
}

#[test]
fn exact_mk8d_aesimc_word_and_known_round_literals() {
    assert_eq!(Kind::InverseMix.word(1, 1), 0x4e28_7821);
    let plain = hex("00112233445566778899aabbccddeeff");
    let key = hex("000102030405060708090a0b0c0d0e0f");
    let round = hex("6353e08c0960e104cd70b751bacad0e7");
    let mixed = hex("5f72641557f5bc92f7be3b291db9f91a");
    for features in features() {
        for (kind, rd, rn, d, n, result) in [
            (Kind::Encrypt, 0, 1, plain, key, round),
            (Kind::Mix, 0, 1, [0; 16], round, mixed),
            (Kind::InverseMix, 1, 1, [0; 16], mixed, round),
            (
                Kind::Decrypt,
                0,
                1,
                round,
                [0; 16],
                hex("00102030405060708090a0b0c0d0e0f0"),
            ),
        ] {
            let mut c = context();
            c.v[rd as usize] = vector(d);
            c.v[rn as usize] = vector(n);
            let mut wanted = c.v;
            wanted[rd as usize] = vector(result);
            let x = c.x;
            let mut code = Code(vec![kind.word(rd, rn), BRK]);
            let outcome = compiler(features)
                .run_bounded(&mut c, &mut code, 1000)
                .unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, 2);
            check_guards(&c, x, wanted);
        }
    }
}

const ROUND_KEYS: [&str; 11] = [
    "000102030405060708090a0b0c0d0e0f",
    "d6aa74fdd2af72fadaa678f1d6ab76fe",
    "b692cf0b643dbdf1be9bc5006830b3fe",
    "b6ff744ed2c2c9bf6c590cbf0469bf41",
    "47f7f7bc95353e03f96c32bcfd058dfd",
    "3caaa3e8a99f9deb50f3af57adf622aa",
    "5e390f7df7a69296a7553dc10aa31f6b",
    "14f9701ae35fe28c440adf4d4ea9c026",
    "47438735a41c65b9e016baf4aebf7ad2",
    "549932d1f08557681093ed9cbe2c974e",
    "13111d7fe3944a17f307a78b4d2b30c5",
];

#[test]
fn aes128_cipher_and_equivalent_inverse_cipher_known_answer() {
    for features in features() {
        for decrypt in [false, true] {
            let mut c = context();
            c.v[0] = vector(hex(if decrypt {
                "69c4e0d86a7b0430d8cdb78070b4c55a"
            } else {
                "00112233445566778899aabbccddeeff"
            }));
            for (round, key) in ROUND_KEYS.iter().enumerate() {
                c.v[round + 1] = vector(hex(key));
            }
            let x = c.x;
            let mut wanted = c.v;
            let mut words = Vec::new();
            if decrypt {
                for key in 2..11 {
                    words.push(Kind::InverseMix.word(key, key));
                    wanted[key as usize] = expected(Kind::InverseMix, [0; 2], wanted[key as usize]);
                }
                for key in (3..=11).rev() {
                    words.push(Kind::Decrypt.word(0, key));
                    wanted[0] = expected(Kind::Decrypt, wanted[0], wanted[key as usize]);
                    words.push(Kind::InverseMix.word(0, 0));
                    wanted[0] = expected(Kind::InverseMix, [0; 2], wanted[0]);
                }
                words.push(Kind::Decrypt.word(0, 2));
                wanted[0] = expected(Kind::Decrypt, wanted[0], wanted[2]);
                words.push(0x6e21_1c00);
                wanted[0] = [wanted[0][0] ^ wanted[1][0], wanted[0][1] ^ wanted[1][1]];
            } else {
                for key in 1..=9 {
                    words.push(Kind::Encrypt.word(0, key));
                    wanted[0] = expected(Kind::Encrypt, wanted[0], wanted[key as usize]);
                    words.push(Kind::Mix.word(0, 0));
                    wanted[0] = expected(Kind::Mix, [0; 2], wanted[0]);
                }
                words.push(Kind::Encrypt.word(0, 10));
                wanted[0] = expected(Kind::Encrypt, wanted[0], wanted[10]);
                words.push(0x6e2b_1c00);
                wanted[0] = [wanted[0][0] ^ wanted[11][0], wanted[0][1] ^ wanted[11][1]];
            }
            assert_eq!(
                wanted[0],
                vector(hex(if decrypt {
                    "00112233445566778899aabbccddeeff"
                } else {
                    "69c4e0d86a7b0430d8cdb78070b4c55a"
                }))
            );
            words.push(BRK);
            let count = words.len() as u64;
            let mut code = Code(words);
            let outcome = compiler(features)
                .run_bounded(&mut c, &mut code, 1000)
                .unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, count);
            check_guards(&c, x, wanted);
        }
    }
}

#[test]
fn helpers_preserve_live_vector_integer_and_status_values_under_pressure() {
    for features in features() {
        for kind in KINDS {
            let mut words = Vec::new();
            let mut c = context();
            let mut wanted_x = c.x;
            let mut wanted_v = c.v;
            for r in 0..16usize {
                words.push(0x9100_0400 | ((r as u32) << 5) | r as u32);
                words.push(0x4ea0_1c00 | (16 << 16) | ((r as u32) << 5) | r as u32);
                wanted_x[r] = wanted_x[r].wrapping_add(1);
                wanted_v[r] = [
                    wanted_v[r][0] | wanted_v[16][0],
                    wanted_v[r][1] | wanted_v[16][1],
                ];
            }
            words.push(kind.word(31, 30));
            wanted_v[31] = expected(kind, wanted_v[31], wanted_v[30]);
            for p in 0..8usize {
                let rd = if p == 7 { 16 } else { 24 + p };
                words.push(
                    0x8b00_0000
                        | (((p * 2 + 1) as u32) << 16)
                        | (((p * 2) as u32) << 5)
                        | rd as u32,
                );
                wanted_x[rd] = wanted_x[p * 2].wrapping_add(wanted_x[p * 2 + 1]);
            }
            for r in 0..14usize {
                words.push(0x4ea0_1c00 | ((r as u32) << 16) | ((r as u32) << 5) | (17 + r) as u32);
                wanted_v[17 + r] = wanted_v[r];
            }
            words.push(BRK);
            let count = words.len() as u64;
            let mut code = Code(words);
            let outcome = compiler(features)
                .run_bounded(&mut c, &mut code, 1000)
                .unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, count);
            check_guards(&c, wanted_x, wanted_v);
        }
    }
}

#[test]
fn software_feature_mask_disables_aesni_without_changing_results() {
    let portable = features()[0];
    assert!(!portable.has_aesni);
    assert!(!portable.has_avx);
    assert!(!portable.has_fma);
    assert!(!portable.has_gfni);
    for kind in KINDS {
        let mut code = Code(vec![kind.word(0, 1), BRK]);
        check_single(
            &mut compiler(portable),
            &mut code,
            kind,
            0,
            1,
            hex("f0e1d2c3b4a5968778695a4b3c2d1e0f"),
            hex("00112233445566778899aabbccddeeff"),
        );
    }
}

#[test]
fn transient_one_instruction_budget_then_cached_chain_preserves_results() {
    for features in features() {
        let mut code = Code(vec![Kind::Encrypt.word(0, 1), Kind::Mix.word(0, 0), BRK]);
        let mut jit = compiler(features);
        let mut c = context();
        let x = c.x;
        let mut wanted = c.v;
        wanted[0] = expected(Kind::Encrypt, wanted[0], wanted[1]);
        let outcome = jit.run_bounded(&mut c, &mut code, 1).unwrap();
        assert_eq!(outcome.stop, StopReason::BudgetExhausted);
        assert_eq!(outcome.retired, 1);
        assert_eq!(c.pc, CODE + 4);
        check_guards(&c, x, wanted);
        wanted[0] = expected(Kind::Mix, [0; 2], wanted[0]);
        let outcome = jit.run_bounded(&mut c, &mut code, 1000).unwrap();
        assert_eq!(outcome.stop, StopReason::Brk(0));
        assert_eq!(outcome.retired, 2);
        check_guards(&c, x, wanted);
    }
}

#[test]
fn emitted_aesni_is_gated_independently_for_all_four_operations() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    use rustarmic::backend::cpu_features::with_features;
    use rustarmic::backend::emit_block;
    use rustarmic::frontend::{TranslateOptions, translate_block_into};
    use rustarmic::ir::Block;
    use rustarmic::optimizer::optimize;

    for features in features() {
        for kind in KINDS {
            let words = [kind.word(0, 1), BRK];
            let mut block = Block::new(CODE);
            translate_block_into(
                &mut block,
                CODE,
                &mut |address| words.get(address.checked_sub(CODE)? as usize / 4).copied(),
                TranslateOptions::default(),
            )
            .unwrap();
            optimize(&mut block);
            let emitted = with_features(features, || emit_block(&block)).unwrap();
            let instructions: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE)
                .into_iter()
                .map(|instruction| {
                    assert!(!instruction.is_invalid());
                    instruction.mnemonic()
                })
                .collect();
            let has_crypto = instructions.iter().any(|mnemonic| {
                matches!(
                    mnemonic,
                    Mnemonic::Aesenclast
                        | Mnemonic::Aesdeclast
                        | Mnemonic::Aesenc
                        | Mnemonic::Aesimc
                )
            });
            assert_eq!(has_crypto, features.has_aesni, "{kind:?}");
            if features.has_aesni {
                match kind {
                    Kind::Encrypt => assert!(instructions.contains(&Mnemonic::Aesenclast)),
                    Kind::Decrypt => assert!(instructions.contains(&Mnemonic::Aesdeclast)),
                    Kind::Mix => {
                        assert!(instructions.contains(&Mnemonic::Aesdeclast));
                        assert!(instructions.contains(&Mnemonic::Aesenc));
                    }
                    Kind::InverseMix => assert!(instructions.contains(&Mnemonic::Aesimc)),
                }
            }
        }
    }
}

#[test]
fn reserved_aes_fixed_fields_do_not_retire_or_modify_state() {
    for features in features() {
        let mut jit = compiler(features);
        for kind in KINDS {
            for changed in [1 << 22, 1 << 23, 1 << 29, 1 << 30] {
                let instruction = kind.word(0, 1) ^ changed;
                let mut code = Code(vec![instruction, BRK]);
                let mut c = context();
                let x = c.x;
                let v = c.v;
                let outcome = jit.run_bounded(&mut c, &mut code, 1).unwrap();
                assert!(
                    matches!(outcome.stop, StopReason::Unsupported(_)),
                    "word{instruction:08x}: {:?}",
                    outcome.stop
                );
                assert_eq!(outcome.retired, 0);
                assert_eq!(c.pc, CODE);
                check_guards(&c, x, v);
            }
        }
    }
}
