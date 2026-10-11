use rustarmic::backend::cpu_features::{detect_features, with_features};
use rustarmic::{CpuContext, CpuFeatures, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const FK: [u32; 4] = [0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc];
const ROUND_KEYS: [u32; 32] = [
    0xf12186f9, 0x41662b61, 0x5a6ab19a, 0x7ba92077, 0x367360f4, 0x776a0c61, 0xb6bb89b3, 0x24763151,
    0xa520307c, 0xb7584dbd, 0xc30753ed, 0x7ee55b57, 0x6988608c, 0x30d895b7, 0x44ba14af, 0x104495a1,
    0xd120b428, 0x73b55fa3, 0xcc874966, 0x92244439, 0xe89e641f, 0x98ca015a, 0xc7159060, 0x99e1fd2e,
    0xb79bd80c, 0x1d2115b0, 0x0e228aeb, 0xf1780c81, 0x428d3654, 0x62293496, 0x01cf72e5, 0x9124a012,
];
const ROUND_STATES: [u32; 32] = [
    0x27fad345, 0xa18b4cb2, 0x11c1e22a, 0xcc13e2ee, 0xf87c5bd5, 0x33220757, 0x77f4c297, 0x7a96f2eb,
    0x27dac07f, 0x42dd0f19, 0xb8a5da02, 0x907127fa, 0x8b952b83, 0xd42b7c59, 0x2ffc5831, 0xf69e6888,
    0xaf2432c4, 0xed1ec85e, 0x55a3ba22, 0x124b18aa, 0x6ae7725f, 0xf4cba1f9, 0x1dcdfa10, 0x2ff60603,
    0xeff24fdc, 0x6fe46b75, 0x893450ad, 0x7b938f4c, 0x536e4246, 0x86b3e94f, 0xd206965e, 0x681edf34,
];
const SBOX_HEX: &str = concat!(
    "d690e9fecce13db716b614c228fb2c05",
    "2b679a762abe04c3aa44132649860699",
    "9c4250f491ef987a33540b43edcfac62",
    "e4b31ca9c908e89580df94fa758f3fa6",
    "4707a7fcf37317ba83593c19e6854fa8",
    "686b81b27164da8bf8eb0f4b70569d35",
    "1e240e5e6358d1a225227c3b01217887",
    "d40046579fd327524c3602e7a0c4c89e",
    "eabf8ad240c738b5a3f7f2cef96115a1",
    "e0ae5da49b341a55ad933230f58cb1e3",
    "1df6e22e8266ca60c02923ab0d534e6f",
    "d5db3745defd8e2f03ff6a726d6c5b51",
    "8d1baf92bbddbc7f11d95c411f105ad8",
    "0ac13188a5cd7bbd2d74d012b8e5b4b0",
    "8969974a0c96777e65b9f109c56ec684",
    "18f07dec3adc4d2079ee5f3ed7cb3948",
);

#[derive(Clone, Copy, Debug)]
enum Kind { Encrypt, Key }
const KINDS: [Kind; 2] = [Kind::Encrypt, Kind::Key];
impl Kind {
    fn word(self, rd: u32, rn: u32, rm: u32) -> u32 {
        match self {
            Self::Encrypt => 0xcec0_8400 | (rn << 5) | rd,
            Self::Key => 0xce60_c800 | (rm << 16) | (rn << 5) | rd,
        }
    }
}
struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 { return None; }
        self.0.get(offset as usize / 4).copied()
    }
}
fn vector(w: [u32; 4]) -> [u64; 2] {
    [u64::from(w[0]) | (u64::from(w[1]) << 32), u64::from(w[2]) | (u64::from(w[3]) << 32)]
}
fn words(v: [u64; 2]) -> [u32; 4] {
    [v[0] as u32, (v[0] >> 32) as u32, v[1] as u32, (v[1] >> 32) as u32]
}
fn sbox() -> [u8; 256] {
    assert_eq!(SBOX_HEX.len(), 512);
    std::array::from_fn(|i| u8::from_str_radix(&SBOX_HEX[i * 2..i * 2 + 2], 16).unwrap())
}
fn permutation(table: &[u8; 256], value: u32, key_schedule: bool) -> u32 {
    let mut result = 0;
    let rotations: &[u32] = if key_schedule { &[0, 13, 23] } else { &[0, 2, 10, 18, 24] };
    for byte in 0..4 {
        let substituted = table[((value >> (byte * 8)) & 255) as usize];
        for bit in 0..8 {
            if substituted & (1 << bit) != 0 {
                for &rotation in rotations {
                    result ^= 1 << ((byte * 8 + bit + rotation) % 32);
                }
            }
        }
    }
    result
}
fn four(table: &[u8; 256], input: [u32; 4], keys: [u32; 4], key_schedule: bool) -> [u32; 4] {
    let mut chain = [0; 8];
    chain[..4].copy_from_slice(&input);
    for index in 0..4 {
        chain[index + 4] = chain[index]
            ^ permutation(table, chain[index + 1] ^ chain[index + 2] ^ chain[index + 3] ^ keys[index], key_schedule);
    }
    chain[4..8].try_into().unwrap()
}
fn expected(table: &[u8; 256], kind: Kind, d: [u64; 2], n: [u64; 2], m: [u64; 2]) -> [u64; 2] {
    let (input, keys) = match kind { Kind::Encrypt => (d, n), Kind::Key => (n, m) };
    vector(four(table, words(input), words(keys), matches!(kind, Kind::Key)))
}
fn features() -> Vec<CpuFeatures> {
    let detected = detect_features();
    let mut portable = detected;
    portable.has_sm4 = false;
    portable.has_avx = false;
    portable.has_fma = false;
    portable.has_avx512_clz16 = false;
    portable.has_gfni = false;
    let mut result = vec![portable];
    if detected.has_sm4 { result.push(detected); }
    result
}
fn compiler(features: CpuFeatures) -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        host_features: Some(features),
        ..JitConfig::default()
    }).unwrap()
}
fn context() -> CpuContext {
    let mut c = CpuContext::default();
    c.pc = CODE;
    for r in 0..31 { c.x[r] = GUARD.wrapping_add(r as u64 * 137); }
    for r in 0..32 { c.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64]; }
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
        ([0x55555555; 4], [0xaaaaaaaa; 4], [0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210]),
    ];
    for byte in 0..256u32 {
        let value = byte * 0x01010101;
        result.push(([value; 4], [0; 4], [value; 4]));
        result.push(([0; 4], [value; 4], [0; 4]));
    }
    for bit in 0..128 {
        let mut basis = [0; 4];
        basis[bit / 32] = 1 << (bit % 32);
        result.extend([(basis, [0; 4], [0; 4]), ([0; 4], basis, [0; 4]), ([0; 4], [0; 4], basis)]);
    }
    let mut seed = 0xcec0_8400_ce60_c800u64;
    for _ in 0..256 {
        let mut next = || {
            std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u32
            })
        };
        result.push((next(), next(), next()));
    }
    result
}

#[test]
fn sm4_four_rounds_all_substitutions_bits_random_all_registers_and_aliases() {
    let table = sbox();
    let samples = samples();
    let mut aliases = vec![(0, 1, 2), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 0, 0),
        (31, 31, 31), (31, 0, 1), (0, 31, 1), (0, 1, 31), (16, 30, 31)];
    aliases.extend((0..32).map(|r| (r, (r + 7) % 32, (r + 19) % 32)));
    for features in features() {
        for kind in KINDS {
            for &(rd, rn, rm) in &aliases {
                let mut code = Code(vec![kind.word(rd, rn, rm), BRK]);
                let original = code.0.clone();
                let mut jit = compiler(features);
                for &sample in &samples {
                    let mut c = context();
                    c.v[rd as usize] = vector(sample.0);
                    c.v[rn as usize] = vector(sample.1);
                    if matches!(kind, Kind::Key) { c.v[rm as usize] = vector(sample.2); }
                    let x = c.x;
                    let mut v = c.v;
                    v[rd as usize] = expected(&table, kind, v[rd as usize], v[rn as usize], v[rm as usize]);
                    let outcome = jit.run_bounded(&mut c, &mut code, 1000).unwrap();
                    assert_eq!(outcome.stop, StopReason::Brk(0), "{kind:?} rd{rd} rn{rn} rm{rm}");
                    assert_eq!(outcome.retired, 2);
                    assert_eq!(code.0, original);
                    guards(&c, x, v);
                }
            }
        }
    }
}
fn constants() -> [u32; 32] {
    std::array::from_fn(|round| u32::from_be_bytes(std::array::from_fn(|byte| ((round * 4 + byte) * 7) as u8)))
}
fn guest_keys(features: CpuFeatures, key: [u32; 4]) -> [u32; 32] {
    let table = sbox();
    let ck = constants();
    let mut c = context();
    c.v[0] = vector(std::array::from_fn(|i| key[i] ^ FK[i]));
    for group in 0..8 { c.v[group + 8] = vector(ck[group * 4..group * 4 + 4].try_into().unwrap()); }
    let x = c.x;
    let mut wanted = c.v;
    let mut program = Vec::new();
    for group in 0..8u32 {
        program.extend([Kind::Key.word(0, 0, group + 8), 0x4ea0_1c00 | (16 + group)]);
        wanted[0] = expected(&table, Kind::Key, wanted[0], wanted[0], wanted[group as usize + 8]);
        wanted[group as usize + 16] = wanted[0];
    }
    program.push(BRK);
    let mut code = Code(program);
    let original = code.0.clone();
    let result = compiler(features).run_bounded(&mut c, &mut code, 1000).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 17);
    guards(&c, x, wanted);
    assert_eq!(code.0, original);
    std::array::from_fn(|i| words(c.v[16 + i / 4])[i % 4])
}
fn guest_cipher(features: CpuFeatures, input: [u32; 4], keys: [u32; 32]) -> ([u32; 4], [u32; 32]) {
    let table = sbox();
    let mut c = context();
    c.v[0] = vector(input);
    for group in 0..8 { c.v[group + 1] = vector(keys[group * 4..group * 4 + 4].try_into().unwrap()); }
    let x = c.x;
    let mut wanted = c.v;
    let mut program = Vec::new();
    for group in 0..8u32 {
        program.extend([Kind::Encrypt.word(0, group + 1, 0), 0x4ea0_1c00 | (16 + group)]);
        wanted[0] = expected(&table, Kind::Encrypt, wanted[0], wanted[group as usize + 1], [0; 2]);
        wanted[group as usize + 16] = wanted[0];
    }
    program.push(BRK);
    let mut code = Code(program);
    let original = code.0.clone();
    let result = compiler(features).run_bounded(&mut c, &mut code, 1000).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 17);
    guards(&c, x, wanted);
    assert_eq!(code.0, original);
    let state = words(c.v[0]);
    ([state[3], state[2], state[1], state[0]], std::array::from_fn(|i| words(c.v[16 + i / 4])[i % 4]))
}

#[test]
fn published_sm4_full_key_schedule_all_round_states_encrypt_and_decrypt_kat() {
    let input = [0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210];
    let cipher = [0x681edf34, 0xd206965e, 0x86b3e94f, 0x536e4246];
    for features in features() {
        let keys = guest_keys(features, input);
        assert_eq!(keys, ROUND_KEYS);
        let (encrypted, states) = guest_cipher(features, input, keys);
        assert_eq!(states, ROUND_STATES);
        assert_eq!(encrypted, cipher);
        let inverse = std::array::from_fn(|i| keys[31 - i]);
        assert_eq!(guest_cipher(features, cipher, inverse).0, input);
        let key2 = [0xfedcba98, 0x76543210, 0x01234567, 0x89abcdef];
        let input2 = [0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f];
        let keys2 = guest_keys(features, key2);
        assert_eq!(&keys2[..4], &[0x0d8cc1b4, 0xac44f213, 0x188c0c40, 0x7537585e]);
        assert_eq!(guest_cipher(features, input2, keys2).0, [0xf766678f, 0x13f01ade, 0xac1b3ea9, 0x55adb594]);
    }
}

#[test]
fn sm4_helper_calls_preserve_live_vectors_integers_and_status_under_pressure() {
    let table = sbox();
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
            v[31] = expected(&table, kind, v[31], v[30], v[29]);
            for p in 0..8usize {
                let rd = if p == 7 { 16 } else { 24 + p };
                program.push(0x8b00_0000 | (((p * 2 + 1) as u32) << 16) | (((p * 2) as u32) << 5) | rd as u32);
                x[rd] = x[p * 2].wrapping_add(x[p * 2 + 1]);
            }
            for r in 0..14usize {
                program.push(0x4ea0_1c00 | ((r as u32) << 16) | ((r as u32) << 5) | (17 + r) as u32);
                v[17 + r] = v[r];
            }
            program.push(BRK);
            let count = program.len() as u64;
            let mut code = Code(program);
            let original = code.0.clone();
            let outcome = compiler(features).run_bounded(&mut c, &mut code, 1000).unwrap();
            assert_eq!(outcome.stop, StopReason::Brk(0));
            assert_eq!(outcome.retired, count);
            assert_eq!(code.0, original);
            guards(&c, x, v);
        }
    }
}

#[test]
fn sm4_transient_one_instruction_budget_preserves_both_dependency_and_resume() {
    let table = sbox();
    for features in features() {
        let mut c = context();
        let x = c.x;
        let mut v = c.v;
        let mut code = Code(vec![Kind::Encrypt.word(0, 1, 0), Kind::Key.word(0, 0, 2), BRK]);
        let original = code.0.clone();
        let mut jit = compiler(features);
        v[0] = expected(&table, Kind::Encrypt, v[0], v[1], v[2]);
        let first = jit.run_bounded(&mut c, &mut code, 1).unwrap();
        assert_eq!(first.stop, StopReason::BudgetExhausted);
        assert_eq!(first.retired, 1);
        assert_eq!(c.pc, CODE + 4);
        guards(&c, x, v);
        v[0] = expected(&table, Kind::Key, v[0], v[0], v[2]);
        let second = jit.run_bounded(&mut c, &mut code, 1000).unwrap();
        assert_eq!(second.stop, StopReason::Brk(0));
        assert_eq!(second.retired, 2);
        assert_eq!(code.0, original);
        guards(&c, x, v);
    }
}

#[test]
fn sm4_feature_gate_emits_exact_vex128_instruction_or_portable_call_without_execution() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};
    use rustarmic::backend::emit_block;
    use rustarmic::frontend::{TranslateOptions, translate_block_into};
    use rustarmic::ir::Block;
    use rustarmic::optimizer::optimize;
    let detected = detect_features();
    let mut portable = detected;
    portable.has_sm4 = false;
    portable.has_avx = false;
    let mut hardware = portable;
    hardware.has_sm4 = true;
    hardware.has_avx = true;
    for kind in KINDS {
        let code = [kind.word(0, 1, 2), BRK];
        let mut block = Block::new(CODE);
        translate_block_into(&mut block, CODE,
            &mut |address| code.get(address.checked_sub(CODE)? as usize / 4).copied(),
            TranslateOptions::default()).unwrap();
        optimize(&mut block);
        let software = with_features(portable, || emit_block(&block)).unwrap();
        let accelerated = with_features(hardware, || emit_block(&block)).unwrap();
        let decode = |bytes: &[u8]| Decoder::new(64, bytes, DecoderOptions::NONE).into_iter().map(|i| {
            assert!(!i.is_invalid()); i
        }).collect::<Vec<_>>();
        let fallback = decode(&software.code);
        assert!(fallback.iter().any(|i| i.mnemonic() == Mnemonic::Call));
        assert!(!fallback.iter().any(|i| matches!(i.mnemonic(), Mnemonic::Vsm4key4 | Mnemonic::Vsm4rnds4)));
        let fast = decode(&accelerated.code);
        assert!(!fast.iter().any(|i| i.mnemonic() == Mnemonic::Call));
        let mnemonic = if matches!(kind, Kind::Key) { Mnemonic::Vsm4key4 } else { Mnemonic::Vsm4rnds4 };
        let operations: Vec<_> = fast.iter().filter(|i| i.mnemonic() == mnemonic).collect();
        assert_eq!(operations.len(), 1);
        let operation = operations[0];
        assert_eq!(operation.op0_register(), Register::XMM0);
        assert_eq!(operation.op1_register(), Register::XMM0);
        assert_eq!(operation.op2_register(), Register::XMM1);
        let code = if matches!(kind, Kind::Key) {
            iced_x86::Code::VEX_Vsm4key4_xmm_xmm_xmmm128
        } else {
            iced_x86::Code::VEX_Vsm4rnds4_xmm_xmm_xmmm128
        };
        assert_eq!(operation.code(), code);
        assert_ne!(software.code, accelerated.code);
    }
}

#[test]
fn sm4_fixed_shape_and_sve_encodings_are_not_misinterpreted_as_a64_4s() {
    use disarm64::decoder::{Operation, decode};
    use rustarmic::frontend::translator::{InstStatus, translate_instruction};
    use rustarmic::ir::{Block, IrEmitter, Op, Ty};
    for kind in KINDS {
        for r in 0..32u32 {
            let word = kind.word(r, (r + 7) % 32, (r + 19) % 32);
            assert!(matches!(decode(word).map(|i| i.operation), Some(Operation::CRYPTOSM4(_))));
            let mut block = Block::new(CODE);
            assert!(matches!(translate_instruction(&mut IrEmitter::new(&mut block, CODE), word).unwrap(), InstStatus::Continue));
            let op = if matches!(kind, Kind::Key) { Op::Sm4KeySchedule } else { Op::Sm4Encrypt };
            assert_eq!(block.code.iter().filter(|a| a.op == op).count(), 1);
            let crypto = block.code.iter().find(|a| a.op == op).unwrap();
            assert_eq!(crypto.ty, Ty::U128);
            assert!(op.is_pure());
            assert!(!op.has_side_effects());
            for changed in [1 << 30, 1 << 29, 1 << 22, 1 << 23, 1 << 10] {
                assert!(!matches!(decode(word ^ changed).map(|i| i.operation), Some(Operation::CRYPTOSM4(_))), "word {:08x}", word ^ changed);
            }
        }
    }
    for word in [0x4523_e000u32, 0x4520_f000] {
        let mut block = Block::new(CODE);
        assert!(translate_instruction(&mut IrEmitter::new(&mut block, CODE), word).is_err());
        assert!(!block.code.iter().any(|a| matches!(a.op, Op::Sm4Encrypt | Op::Sm4KeySchedule)));
    }
}

#[test]
fn detected_sm4_requires_avx_os_support_and_cpuid_leaf_seven_subleaf_one() {
    let detected = detect_features();
    #[cfg(target_arch = "x86_64")]
    {
        let leaf = std::arch::x86_64::__cpuid_count(7, 0);
        let advertised = std::arch::x86_64::__cpuid(0).eax >= 7
            && leaf.eax >= 1
            && std::arch::x86_64::__cpuid_count(7, 1).eax & 4 != 0;
        assert_eq!(detected.has_sm4, advertised && std::arch::is_x86_feature_detected!("avx"));
        assert!(!detected.has_sm4 || detected.has_avx);
    }
    assert!(!features()[0].has_sm4);
}
