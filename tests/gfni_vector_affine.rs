use iced_x86::{Decoder, DecoderOptions, Mnemonic};
use rustarmic::backend::cpu_features::{CpuFeatures, detect_features, with_features};
use rustarmic::backend::emit_block;
use rustarmic::frontend::{TranslateOptions, translate_block_into};
use rustarmic::ir::{Block, IrEmitter, Op, Terminal, ValueRef};
use rustarmic::jit::code_cache::{CodeCache, ThunkFn};
use rustarmic::optimizer::optimize;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

// A BRK stop leaves ctx.pc at the BRK: the ARM ARM preferred exception return
// address for a Breakpoint Instruction exception is the BRK itself (Unicorn
// agrees). These assertions originally expected the following instruction.

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const IDENTITY: u64 = 0x0102_0408_1020_4080;
const REVERSE: u64 = 0x8040_2010_0804_0201;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Left,
    Right,
    SignedRight,
    Reverse,
}

#[derive(Clone, Copy, Debug)]
enum BitKind {
    And,
    Or,
    Xor,
    Bic,
    Orn,
    Not,
}

impl BitKind {
    fn emit(self, em: &mut IrEmitter<'_>, n: ValueRef, m: ValueRef, q: bool) -> ValueRef {
        match self {
            Self::And => em.vec_and(n, m, q),
            Self::Or => em.vec_orr(n, m, q),
            Self::Xor => em.vec_eor(n, m, q),
            Self::Bic => em.vec_bic(n, m, q),
            Self::Orn => em.vec_orn(n, m, q),
            Self::Not => em.vec_not(n, q),
        }
    }

    fn apply(self, n: [u64; 2], m: [u64; 2], q: bool) -> [u64; 2] {
        let n = bytes(n);
        let m = bytes(m);
        let mut output = [0; 16];
        for lane in 0..if q { 16 } else { 8 } {
            output[lane] = match self {
                Self::And => n[lane] & m[lane],
                Self::Or => n[lane] | m[lane],
                Self::Xor => n[lane] ^ m[lane],
                Self::Bic => n[lane] & !m[lane],
                Self::Orn => n[lane] | !m[lane],
                Self::Not => !n[lane],
            };
        }
        vector(output)
    }
}

impl Kind {
    fn word(self, q: bool, shift: u32, rn: u32, rd: u32) -> u32 {
        let base = match self {
            Self::Left => 0x0f00_5400 | ((8 + shift) << 16),
            Self::Right => 0x2f00_0400 | ((16 - shift) << 16),
            Self::SignedRight => 0x0f00_0400 | ((16 - shift) << 16),
            Self::Reverse => 0x2e60_5800,
        };
        base | (u32::from(q) << 30) | (rn << 5) | rd
    }

    fn apply(self, input: [u64; 2], q: bool, shift: u32) -> [u64; 2] {
        let input = bytes(input);
        let mut output = [0; 16];
        for lane in 0..if q { 16 } else { 8 } {
            let x = input[lane];
            output[lane] = match self {
                Self::Left => ((x as u16) << shift) as u8,
                Self::Right => ((x as u16) >> shift) as u8,
                Self::SignedRight => ((x as i8 as i16) >> shift) as u8,
                Self::Reverse => {
                    let mut y = 0;
                    for bit in 0..8 {
                        y |= ((x >> bit) & 1) << (7 - bit);
                    }
                    y
                }
            };
        }
        vector(output)
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

fn features() -> Vec<CpuFeatures> {
    let detected = detect_features();
    let mut portable = detected;
    portable.has_gfni = false;
    portable.has_avx = false;
    portable.has_fma = false;
    let mut out = vec![portable];
    if detected.has_gfni {
        out.push(detected);
    }
    out
}

fn compiler(features: CpuFeatures) -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 256 * 1024,
        host_features: Some(features),
        ..JitConfig::default()
    })
    .unwrap()
}

fn vector(bytes: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ]
}

fn bytes(value: [u64; 2]) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&value[0].to_le_bytes());
    out[8..].copy_from_slice(&value[1].to_le_bytes());
    out
}

fn sample(value: u8) -> [u64; 2] {
    vector(std::array::from_fn(|lane| {
        value.wrapping_add(lane as u8 * 17)
    }))
}

fn seeded_context() -> CpuContext {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x = std::array::from_fn(|index| GUARD ^ index as u64 * 0x1_0001);
    ctx.v = std::array::from_fn(|index| {
        [
            GUARD.rotate_left(index as u32),
            (!GUARD).rotate_right(index as u32),
        ]
    });
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    ctx.fpcr = 0x03c0_0000;
    ctx.fpsr = 0x0800_009f;
    ctx.tpidr_el0 = GUARD;
    ctx.tpidrro_el0 = !GUARD;
    ctx.exclusive_addr = GUARD;
    ctx.exclusive_value = !GUARD;
    ctx.exclusive_size = 8;
    ctx.io_value = [GUARD, !GUARD];
    ctx
}

fn check_guards(ctx: &CpuContext, expected_x: [u64; 31], expected_v: [[u64; 2]; 32]) {
    assert_eq!(ctx.x, expected_x);
    assert_eq!(ctx.v, expected_v);
    assert_eq!(
        (ctx.sp, ctx.nzcv, ctx.fpcr, ctx.fpsr),
        (GUARD, 0xb, 0x03c0_0000, 0x0800_009f)
    );
    assert_eq!((ctx.tpidr_el0, ctx.tpidrro_el0), (GUARD, !GUARD));
    assert_eq!(
        (ctx.exclusive_addr, ctx.exclusive_value, ctx.exclusive_size),
        (GUARD, !GUARD, 8)
    );
    assert_eq!(ctx.io_value, [GUARD, !GUARD]);
    assert_eq!(
        (
            ctx.should_halt,
            ctx.mem_fault,
            ctx.mem_fault_size,
            ctx.mem_fault_access,
            ctx.mem_fault_cause
        ),
        (0, 0, 0, 0, 0)
    );
    assert_eq!((ctx.mem_fault_addr, ctx.mem_fault_pc), (0, 0));
}

fn affine_byte(input: u8, matrix: u64, addend: u8) -> u8 {
    let rows = matrix.to_be_bytes();
    let mut output = addend;
    for bit in 0..8 {
        let mut parity = 0;
        for source_bit in 0..8 {
            parity ^= ((input >> source_bit) & 1) & ((rows[bit] >> source_bit) & 1);
        }
        output ^= parity << bit;
    }
    output
}

fn affine(input: [u64; 2], matrix: u64, addend: u8, q: bool) -> [u64; 2] {
    let input = bytes(input);
    let mut output = [0; 16];
    for lane in 0..if q { 16 } else { 8 } {
        output[lane] = affine_byte(input[lane], matrix, addend);
    }
    vector(output)
}

fn compile_ir(mut block: Block, features: CpuFeatures) -> (CodeCache, *const u8) {
    let cycles = block.cycles;
    optimize(&mut block);
    assert_eq!(block.cycles, cycles);
    let emitted = with_features(features, || emit_block(&block)).unwrap();
    let mut cache = CodeCache::new(256 * 1024).unwrap();
    let pointer = cache
        .install(
            CODE,
            block.end_pc,
            cycles,
            &emitted.code,
            &emitted.chains,
            emitted.body_offset,
            true,
        )
        .unwrap();
    (cache, pointer)
}

fn invoke(cache: &CodeCache, pointer: *const u8, ctx: &mut CpuContext, expected_pc: u64) {
    let thunk: ThunkFn = unsafe { core::mem::transmute(cache.thunk()) };
    assert_eq!(unsafe { thunk(pointer as u64, ctx) }, expected_pc);
}

fn finish(block: &mut Block, cycles: u32) {
    block.cycles = cycles;
    block.end_pc = CODE + u64::from(cycles) * 4;
    block.terminal = Terminal::LinkBlock {
        next_pc: block.end_pc,
    };
}

#[test]
fn all_byte_shift_values_counts_q_forms_aliases_hardware_and_fallback() {
    for features in features() {
        for kind in [Kind::Left, Kind::Right, Kind::SignedRight] {
            for q in [false, true] {
                for step in 0..8 {
                    let shift = if matches!(kind, Kind::Left) {
                        step
                    } else {
                        step + 1
                    };
                    for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
                        let word = kind.word(q, shift, rn, rd);
                        let words = vec![word, BRK];
                        let mut code = Code(words.clone());
                        let mut jit = compiler(features);
                        for input in 0..=255 {
                            let mut ctx = seeded_context();
                            ctx.v[rn as usize] = sample(input as u8);
                            let mut expected = ctx.v;
                            expected[rd as usize] = kind.apply(ctx.v[rn as usize], q, shift);
                            let expected_x = ctx.x;
                            let result = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                            assert_eq!(
                                result.stop,
                                StopReason::Brk(0),
                                "{word:08x}/{input}/{features:?}"
                            );
                            assert_eq!(result.retired, 2);
                            assert_eq!(ctx.pc, CODE + 4);
                            check_guards(&ctx, expected_x, expected);
                            assert_eq!(code.0, words);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn vector_rbit_all_bytes_registers_aliases_and_involution() {
    for features in features() {
        for q in [false, true] {
            for rn in 0..32 {
                for rd in [rn, (rn + 13) % 32] {
                    let word = Kind::Reverse.word(q, 0, rn, rd);
                    let mut code = Code(vec![word, BRK]);
                    let mut jit = compiler(features);
                    for input in 0..=255 {
                        let mut ctx = seeded_context();
                        ctx.v[rn as usize] = sample(input as u8);
                        let source = ctx.v[rn as usize];
                        let mut expected = ctx.v;
                        expected[rd as usize] = Kind::Reverse.apply(source, q, 0);
                        let expected_x = ctx.x;
                        let result = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                        assert_eq!(result.stop, StopReason::Brk(0));
                        assert_eq!(result.retired, 2);
                        check_guards(&ctx, expected_x, expected);
                        let restored = Kind::Reverse.apply(expected[rd as usize], q, 0);
                        assert_eq!(restored, [source[0], if q { source[1] } else { 0 }]);
                    }
                }
            }
        }
    }
}

#[test]
fn arbitrary_affine_matrices_addends_all_bytes_and_aliases() {
    let mut matrices = vec![
        (0, 0),
        (0, 0xa5),
        (IDENTITY, 0),
        (IDENTITY, 0xff),
        (REVERSE, 0),
        (u64::MAX, 0x5a),
        (0x0101_0101_0101_0101, 0x81),
    ];
    let mut random = 0x77c0_8040_2010_0804u64;
    for _ in 0..32 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        matrices.push((random, (random >> 37) as u8));
    }
    for features in features() {
        for &(matrix, addend) in &matrices {
            for q in [false, true] {
                for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
                    let mut block = Block::new(CODE);
                    let mut em = IrEmitter::new(&mut block, CODE);
                    let source = em.get_v_q(rn);
                    let value = em.vec_bit_affine(source, matrix, addend, q);
                    em.set_v_q(rd, value);
                    finish(&mut block, 1);
                    let (cache, pointer) = compile_ir(block, features);
                    for input in 0..=255 {
                        let mut ctx = seeded_context();
                        ctx.v[rn as usize] = sample(input as u8);
                        let mut expected = ctx.v;
                        expected[rd as usize] = affine(ctx.v[rn as usize], matrix, addend, q);
                        let expected_x = ctx.x;
                        invoke(&cache, pointer, &mut ctx, CODE + 4);
                        check_guards(&ctx, expected_x, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn composed_affines_preserve_q0_domains_addends_fanout_and_cycles() {
    let pairs = [
        (IDENTITY, 0, REVERSE, 0),
        (REVERSE, 0x53, REVERSE, 0xa6),
        (0, 0xff, IDENTITY, 0xa5),
        (0xf0e1_d2c3_b4a5_9687, 0x5a, 0x0123_4567_89ab_cdef, 0xc3),
    ];
    for features in features() {
        for &(matrix_a, add_a, matrix_b, add_b) in &pairs {
            for q_a in [false, true] {
                for q_b in [false, true] {
                    for fanout in [false, true] {
                        let mut block = Block::new(CODE);
                        let mut em = IrEmitter::new(&mut block, CODE);
                        let source = em.get_v_q(31);
                        let first = em.vec_bit_affine(source, matrix_a, add_a, q_a);
                        if fanout {
                            em.set_v_q(2, first);
                        }
                        let second = em.vec_bit_affine(first, matrix_b, add_b, q_b);
                        em.set_v_q(31, second);
                        finish(&mut block, 7);
                        let (cache, pointer) = compile_ir(block, features);
                        for input in 0..=255 {
                            let mut ctx = seeded_context();
                            ctx.v[31] = sample(input as u8);
                            let first = affine(ctx.v[31], matrix_a, add_a, q_a);
                            let mut expected = ctx.v;
                            expected[31] = affine(first, matrix_b, add_b, q_b);
                            if fanout {
                                expected[2] = first;
                            }
                            let expected_x = ctx.x;
                            invoke(&cache, pointer, &mut ctx, CODE + 28);
                            check_guards(&ctx, expected_x, expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn integer_carry_barrier_cannot_be_absorbed_into_affine_composition() {
    for features in features() {
        let mut block = Block::new(CODE);
        let mut em = IrEmitter::new(&mut block, CODE);
        let source = em.get_v_q(31);
        let first = em.vec_bit_affine(source, REVERSE, 0x7f, true);
        let other = em.get_v_q(30);
        let sum = em.vec_add(first, other, 2, true);
        let second = em.vec_bit_affine(sum, IDENTITY, 0xa5, true);
        em.set_v_q(31, second);
        finish(&mut block, 5);
        let (cache, pointer) = compile_ir(block, features);
        for input in 0..=255 {
            let mut ctx = seeded_context();
            ctx.v[31] = sample(input as u8);
            ctx.v[30] = [0x0000_0001_ffff_ffff, 0x8000_0000_7fff_ffff];
            let a = bytes(affine(ctx.v[31], REVERSE, 0x7f, true));
            let b = bytes(ctx.v[30]);
            let mut sum = [0; 16];
            for lane in 0..4 {
                let offset = lane * 4;
                let a = u32::from_le_bytes(a[offset..offset + 4].try_into().unwrap());
                let b = u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap());
                sum[offset..offset + 4].copy_from_slice(&a.wrapping_add(b).to_le_bytes());
            }
            let mut expected = ctx.v;
            expected[31] = affine(vector(sum), IDENTITY, 0xa5, true);
            let expected_x = ctx.x;
            invoke(&cache, pointer, &mut ctx, CODE + 20);
            check_guards(&ctx, expected_x, expected);
        }
    }
}

#[test]
fn affine_preserves_live_integer_and_vector_sources_under_register_pressure() {
    for features in features() {
        for q in [false, true] {
            let mut block = Block::new(CODE);
            let mut em = IrEmitter::new(&mut block, CODE);
            let vectors: Vec<_> = (0..16).map(|r| em.get_v_q(r)).collect();
            let integers: Vec<_> = (0..16).map(|r| em.get_x(r)).collect();
            let source = em.get_v_q(31);
            let first = em.vec_bit_affine(source, 0x1f3f_7f80_4020_1008, 0xc7, q);
            let second = em.vec_bit_affine(first, REVERSE, 0x31, q);
            em.set_v_q(31, second);
            for (index, value) in vectors.into_iter().enumerate() {
                em.set_v_q(if index == 15 { 0 } else { index as u8 + 16 }, value);
            }
            for (index, value) in integers.into_iter().enumerate() {
                em.set_x(index as u8 + 15, value);
            }
            finish(&mut block, 9);
            let (cache, pointer) = compile_ir(block, features);
            for input in 0..=255 {
                let mut ctx = seeded_context();
                ctx.v[31] = sample(input as u8);
                let mut expected_x = ctx.x;
                expected_x[15..31].copy_from_slice(&ctx.x[..16]);
                let mut expected = ctx.v;
                expected[16..31].copy_from_slice(&ctx.v[..15]);
                expected[0] = ctx.v[15];
                expected[31] = affine(
                    affine(ctx.v[31], 0x1f3f_7f80_4020_1008, 0xc7, q),
                    REVERSE,
                    0x31,
                    q,
                );
                invoke(&cache, pointer, &mut ctx, CODE + 36);
                check_guards(&ctx, expected_x, expected);
            }
        }
    }
}

#[test]
fn fused_guest_chains_retire_each_instruction_and_resume_at_every_budget() {
    for features in features() {
        for q_a in [false, true] {
            for q_b in [false, true] {
                let operations = [
                    (Kind::Left, q_a, 1),
                    (Kind::Reverse, q_b, 0),
                    (Kind::SignedRight, q_b, 8),
                    (Kind::Right, q_a, 3),
                ];
                let mut words: Vec<_> = operations
                    .iter()
                    .map(|&(kind, q, shift)| kind.word(q, shift, 31, 31))
                    .collect();
                words.push(BRK);
                let mut code = Code(words.clone());
                let mut jit = compiler(features);
                let mut warm = seeded_context();
                assert_eq!(jit.run_bounded(&mut warm, &mut code, 5).unwrap().retired, 5);
                for input in 0..=255 {
                    for budget in 0..=5 {
                        let mut ctx = seeded_context();
                        ctx.v[31] = sample(input as u8);
                        let mut expected = ctx.v;
                        for &(kind, q, shift) in operations.iter().take((budget as usize).min(4)) {
                            expected[31] = kind.apply(expected[31], q, shift);
                        }
                        let expected_x = ctx.x;
                        let first = jit.run_bounded(&mut ctx, &mut code, budget).unwrap();
                        assert_eq!(first.retired, budget);
                        assert_eq!(
                            first.stop,
                            if budget == 5 {
                                StopReason::Brk(0)
                            } else {
                                StopReason::BudgetExhausted
                            }
                        );
                        assert_eq!(ctx.pc, CODE + budget * 4 - if budget == 5 { 4 } else { 0 });
                        check_guards(&ctx, expected_x, expected);
                        if budget < 5 {
                            for &(kind, q, shift) in operations.iter().skip(budget as usize) {
                                expected[31] = kind.apply(expected[31], q, shift);
                            }
                            let second = jit.run_bounded(&mut ctx, &mut code, 5 - budget).unwrap();
                            assert_eq!(second.retired, 5 - budget);
                            assert_eq!(second.stop, StopReason::Brk(0));
                            assert_eq!(ctx.pc, CODE + 16);
                            check_guards(&ctx, expected_x, expected);
                        }
                        assert_eq!(code.0, words);
                    }
                }
            }
        }
    }
}

#[test]
fn emitted_gfni_is_host_gated_and_plain_affine_has_no_helper_call() {
    for features in features() {
        let mut block = Block::new(CODE);
        let mut em = IrEmitter::new(&mut block, CODE);
        let source = em.get_v_q(1);
        let result = em.vec_bit_affine(source, 0x1357_9bdf_2468_ace0, 0x5a, true);
        em.set_v_q(0, result);
        finish(&mut block, 1);
        optimize(&mut block);
        let emitted = with_features(features, || emit_block(&block)).unwrap();
        let mnemonics: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE)
            .into_iter()
            .map(|instruction| {
                assert!(!instruction.is_invalid());
                instruction.mnemonic()
            })
            .collect();
        let gfni = mnemonics
            .iter()
            .any(|m| matches!(m, Mnemonic::Gf2p8affineqb | Mnemonic::Vgf2p8affineqb));
        assert_eq!(gfni, features.has_gfni);
        assert!(!mnemonics.contains(&Mnemonic::Call));
    }
}

#[test]
fn affine_ir_contract_and_optimization_keep_instruction_cycle_metadata() {
    let mut block = Block::new(CODE);
    let mut em = IrEmitter::new(&mut block, CODE);
    let source = em.get_v_q(1);
    let value = em.vec_bit_affine(source, REVERSE, 0xa5, false);
    em.set_v_q(0, value);
    let affine = block.get(value);
    assert_eq!(affine.op, Op::VecBitAffine);
    assert_eq!(affine.imm, 0xa500);
    assert_eq!(affine.args[0], source);
    let matrix = block.get(affine.args[1]);
    assert_eq!(matrix.op, Op::ConstU64);
    assert_eq!(matrix.imm, REVERSE);
    finish(&mut block, 17);
    optimize(&mut block);
    assert_eq!(block.cycles, 17);
    assert_eq!(block.end_pc, CODE + 68);
    let mut guest = Block::new(CODE);
    let words = [
        Kind::Left.word(true, 2, 1, 0),
        Kind::Reverse.word(true, 0, 0, 0),
        Kind::Right.word(true, 3, 0, 0),
        BRK,
    ];
    translate_block_into(
        &mut guest,
        CODE,
        &mut |address| words.get(((address - CODE) / 4) as usize).copied(),
        TranslateOptions {
            max_insts: 4,
            multiblock: false,
        },
    )
    .unwrap();
    let cycles = guest.cycles;
    let end_pc = guest.end_pc;
    optimize(&mut guest);
    assert_eq!(guest.cycles, cycles);
    assert_eq!(guest.end_pc, end_pc);
    assert_eq!(cycles, 4);
}

#[test]
fn affine_fusion_shape_respects_inactive_byte_domain_with_nonzero_addend() {
    for (q_a, q_b, add_b, expected_nodes, expected_q) in [
        (true, true, 0x81, 1, true),
        (true, false, 0x81, 1, false),
        (false, false, 0x81, 1, false),
        (false, true, 0, 1, false),
        (false, true, 0x81, 2, true),
    ] {
        let mut block = Block::new(CODE);
        let mut em = IrEmitter::new(&mut block, CODE);
        let source = em.get_v_q(31);
        let first = em.vec_bit_affine(source, 0x1357_9bdf_2468_ace0, 0x53, q_a);
        let second = em.vec_bit_affine(first, 0xfedc_ba98_7654_3210, add_b, q_b);
        em.set_v_q(0, second);
        finish(&mut block, 2);
        optimize(&mut block);
        let nodes: Vec<_> = block
            .iter_live()
            .filter(|(_, a)| a.op == Op::VecBitAffine)
            .collect();
        assert_eq!(nodes.len(), expected_nodes);
        assert_eq!(nodes.last().unwrap().1.imm & 1 != 0, expected_q);
        assert_eq!(block.cycles, 2);
    }
}

#[test]
fn full_vector_forwarding_does_not_cross_scalar_overwrite_with_stale_affine_value() {
    for features in features() {
        let mut block = Block::new(CODE);
        let mut em = IrEmitter::new(&mut block, CODE);
        let source = em.get_v_q(1);
        let first = em.vec_bit_affine(source, REVERSE, 0x5a, true);
        em.set_v_q(0, first);
        let overwrite = em.const_u32(0x3f80_0000);
        em.set_v_s(0, overwrite);
        let reloaded = em.get_v_q(0);
        let second = em.vec_bit_affine(reloaded, IDENTITY, 0xa5, true);
        em.set_v_q(31, second);
        finish(&mut block, 4);
        let (cache, pointer) = compile_ir(block, features);
        let mut ctx = seeded_context();
        let mut expected = ctx.v;
        expected[0] = [0x3f80_0000, 0];
        expected[31] = affine(expected[0], IDENTITY, 0xa5, true);
        let expected_x = ctx.x;
        invoke(&cache, pointer, &mut ctx, CODE + 16);
        check_guards(&ctx, expected_x, expected);
    }
}

#[test]
fn uniform_bitlogic_chains_preserve_operand_order_and_inactive_byte_domains() {
    for features in features() {
        for kind in [
            BitKind::And,
            BitKind::Or,
            BitKind::Xor,
            BitKind::Bic,
            BitKind::Orn,
            BitKind::Not,
        ] {
            for constant_left in [false, true] {
                if constant_left && matches!(kind, BitKind::Not) {
                    continue;
                }
                for q_a in [false, true] {
                    for q_b in [false, true] {
                        for mask in [0, 1, 0x55, 0x80, 0xa5, 0xff] {
                            let low = u64::from_le_bytes([mask; 8]);
                            let high = if q_b { low } else { 0x1234_5678_9abc_def0 };
                            let mut block = Block::new(CODE);
                            let mut em = IrEmitter::new(&mut block, CODE);
                            let source = em.get_v_q(31);
                            let first = em.vec_bit_affine(source, REVERSE, 0x29, q_a);
                            let lo = em.const_u64(low);
                            let hi = em.const_u64(high);
                            let constant = em.vec_build_q(lo, hi);
                            let (n, m) = if constant_left {
                                (constant, first)
                            } else {
                                (first, constant)
                            };
                            let value = kind.emit(&mut em, n, m, q_b);
                            em.set_v_q(31, value);
                            finish(&mut block, 3);
                            let (cache, pointer) = compile_ir(block, features);
                            for input in 0..=255 {
                                let mut ctx = seeded_context();
                                ctx.v[31] = sample(input as u8);
                                let first = affine(ctx.v[31], REVERSE, 0x29, q_a);
                                let (n, m) = if constant_left {
                                    ([low, high], first)
                                } else {
                                    (first, [low, high])
                                };
                                let mut expected = ctx.v;
                                expected[31] = kind.apply(n, m, q_b);
                                let expected_x = ctx.x;
                                invoke(&cache, pointer, &mut ctx, CODE + 12);
                                check_guards(&ctx, expected_x, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn heterogeneous_active_constants_and_wide_shifts_remain_semantic_barriers() {
    for features in features() {
        for q in [false, true] {
            for kind in [
                BitKind::And,
                BitKind::Or,
                BitKind::Xor,
                BitKind::Bic,
                BitKind::Orn,
            ] {
                let mask = vector(std::array::from_fn(|lane| (lane as u8).wrapping_mul(37)));
                let mut block = Block::new(CODE);
                let mut em = IrEmitter::new(&mut block, CODE);
                let source = em.get_v_q(31);
                let first = em.vec_bit_affine(source, REVERSE, 0x29, q);
                let lo = em.const_u64(mask[0]);
                let hi = em.const_u64(mask[1]);
                let constant = em.vec_build_q(lo, hi);
                let logical = kind.emit(&mut em, first, constant, q);
                let wide = em.vec_shl_imm(logical, 1, 9, q);
                let final_value = em.vec_bit_affine(wide, IDENTITY, 0xa5, q);
                em.set_v_q(31, final_value);
                finish(&mut block, 4);
                let (cache, pointer) = compile_ir(block, features);
                for input in 0..=255 {
                    let mut ctx = seeded_context();
                    ctx.v[31] = sample(input as u8);
                    let first = affine(ctx.v[31], REVERSE, 0x29, q);
                    let logical = bytes(kind.apply(first, mask, q));
                    let mut shifted = [0; 16];
                    for lane in 0..if q { 8 } else { 4 } {
                        let offset = lane * 2;
                        let value =
                            u16::from_le_bytes(logical[offset..offset + 2].try_into().unwrap());
                        shifted[offset..offset + 2]
                            .copy_from_slice(&value.wrapping_shl(9).to_le_bytes());
                    }
                    let mut expected = ctx.v;
                    expected[31] = affine(vector(shifted), IDENTITY, 0xa5, q);
                    let expected_x = ctx.x;
                    invoke(&cache, pointer, &mut ctx, CODE + 16);
                    check_guards(&ctx, expected_x, expected);
                }
            }
        }
    }
}

fn shifted_diagonal_matrix(delta: i32, mask: u8) -> u64 {
    let mut rows = [0u8; 8];
    for output in 0..8 {
        let input = output as i32 + delta;
        if mask & (1 << output) != 0 && (0..8).contains(&input) {
            rows[output] = 1 << input;
        }
    }
    u64::from_be_bytes(rows)
}

#[test]
fn compact_shifted_diagonals_all_masks_deltas_bytes_addends_q_domains() {
    for features in features() {
        for q in [false, true] {
            for addend in [0, 0xa5] {
                for delta in -7..=7 {
                    for first_mask in (0..256).step_by(16) {
                        let matrices: Vec<_> = (first_mask..first_mask + 16)
                            .map(|mask| shifted_diagonal_matrix(delta, mask as u8))
                            .collect();
                        let mut block = Block::new(CODE);
                        let mut em = IrEmitter::new(&mut block, CODE);
                        let source = em.get_v_q(31);
                        for (index, &matrix) in matrices.iter().enumerate() {
                            let value = em.vec_bit_affine(source, matrix, addend, q);
                            em.set_v_q(index as u8, value);
                        }
                        finish(&mut block, 16);
                        let (cache, pointer) = compile_ir(block, features);
                        for input in 0..=255 {
                            let mut ctx = seeded_context();
                            ctx.v[31] = sample(input as u8);
                            let mut expected = ctx.v;
                            for (index, &matrix) in matrices.iter().enumerate() {
                                expected[index] = affine(ctx.v[31], matrix, addend, q);
                            }
                            let expected_x = ctx.x;
                            invoke(&cache, pointer, &mut ctx, CODE + 64);
                            check_guards(&ctx, expected_x, expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn compact_portable_matrices_emit_at_most_one_word_shift_mask_and_addend_xor() {
    let mut portable = detect_features();
    portable.has_gfni = false;
    portable.has_avx = false;
    portable.has_fma = false;
    for delta in -7..=7 {
        for mask in [0, 1, 0x55, 0x80, 0xa5, 0xff] {
            for addend in [0, 0xa5] {
                let mut block = Block::new(CODE);
                let mut em = IrEmitter::new(&mut block, CODE);
                let source = em.get_v_q(1);
                let result =
                    em.vec_bit_affine(source, shifted_diagonal_matrix(delta, mask), addend, true);
                em.set_v_q(0, result);
                finish(&mut block, 1);
                optimize(&mut block);
                let emitted = with_features(portable, || emit_block(&block)).unwrap();
                let mnemonics: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE)
                    .into_iter()
                    .map(|instruction| {
                        assert!(!instruction.is_invalid());
                        instruction.mnemonic()
                    })
                    .collect();
                assert!(
                    mnemonics
                        .iter()
                        .filter(|m| matches!(m, Mnemonic::Psllw | Mnemonic::Psrlw))
                        .count()
                        <= 1,
                    "{delta}/{mask:02x}/{addend:02x}: {mnemonics:?}"
                );
                assert!(
                    mnemonics
                        .iter()
                        .filter(|m| matches!(m, Mnemonic::Pand))
                        .count()
                        <= 1
                );
                assert!(
                    mnemonics
                        .iter()
                        .filter(|m| matches!(m, Mnemonic::Pxor))
                        .count()
                        <= 1
                );
                assert!(!mnemonics.contains(&Mnemonic::Call));
                assert!(
                    !mnemonics
                        .iter()
                        .any(|m| matches!(m, Mnemonic::Gf2p8affineqb | Mnemonic::Vgf2p8affineqb))
                );
            }
        }
    }
}

fn byte_variable_shift(input: [u64; 2], signed: bool, count: u8, q: bool) -> [u64; 2] {
    let input = bytes(input);
    let count = count as i8 as i16;
    let mut out = [0; 16];
    for lane in 0..if q { 16 } else { 8 } {
        out[lane] = if count >= 8 {
            0
        } else if count >= 0 {
            ((input[lane] as u16) << count as u32) as u8
        } else if -count >= 8 {
            if signed && input[lane] & 0x80 != 0 {
                0xff
            } else {
                0
            }
        } else if signed {
            ((input[lane] as i8 as i16) >> (-count) as u32) as u8
        } else {
            input[lane] >> (-count) as u32
        };
    }
    vector(out)
}

#[test]
fn proven_uniform_variable_byte_shifts_all_counts_inputs_constant_shapes_and_q() {
    for features in features() {
        for q in [false, true] {
            for signed in [false, true] {
                for duplicate in [false, true] {
                    for first_count in (0..256usize).step_by(16) {
                        let mut block = Block::new(CODE);
                        let mut em = IrEmitter::new(&mut block, CODE);
                        let source = em.get_v_q(31);
                        for index in 0..16 {
                            let count = (first_count + index) as u8;
                            let amount = if duplicate {
                                let scalar = em.const_u64(0xdead_beef_8000_0000 | u64::from(count));
                                em.vec_dup_gpr(scalar, 0, q)
                            } else {
                                let low = em.const_u64(u64::from_le_bytes([count; 8]));
                                let high = em.const_u64(if q {
                                    u64::from_le_bytes([count; 8])
                                } else {
                                    0x1234_5678_9abc_def0
                                });
                                em.vec_build_q(low, high)
                            };
                            let value = em.vec_variable_shift(source, amount, 0, q, signed);
                            em.set_v_q(if index == 15 { 31 } else { index as u8 }, value);
                        }
                        finish(&mut block, 16);
                        let (cache, pointer) = compile_ir(block, features);
                        for input in 0..=255 {
                            let mut ctx = seeded_context();
                            ctx.v[31] = sample(input as u8);
                            let source = ctx.v[31];
                            let mut expected = ctx.v;
                            for index in 0..16 {
                                expected[if index == 15 { 31 } else { index }] =
                                    byte_variable_shift(
                                        source,
                                        signed,
                                        (first_count + index) as u8,
                                        q,
                                    );
                            }
                            let expected_x = ctx.x;
                            invoke(&cache, pointer, &mut ctx, CODE + 64);
                            check_guards(&ctx, expected_x, expected);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn heterogeneous_variable_byte_counts_remain_lane_specific_not_uniform_affine() {
    let counts = vector([
        0, 1, 7, 8, 127, 128, 248, 255, 3, 253, 2, 254, 4, 252, 6, 250,
    ]);
    for features in features() {
        for q in [false, true] {
            for signed in [false, true] {
                let mut block = Block::new(CODE);
                let mut em = IrEmitter::new(&mut block, CODE);
                let source = em.get_v_q(31);
                let low = em.const_u64(counts[0]);
                let high = em.const_u64(counts[1]);
                let count = em.vec_build_q(low, high);
                let result = em.vec_variable_shift(source, count, 0, q, signed);
                em.set_v_q(31, result);
                finish(&mut block, 1);
                optimize(&mut block);
                let variable = if signed { Op::VecSshl8 } else { Op::VecUshl8 };
                assert_eq!(
                    block.iter_live().filter(|(_, a)| a.op == variable).count(),
                    1
                );
                let (cache, pointer) = compile_ir(block, features);
                for input in 0..=255 {
                    let mut ctx = seeded_context();
                    ctx.v[31] = sample(input as u8);
                    let source = bytes(ctx.v[31]);
                    let counts = bytes(counts);
                    let mut wanted = [0; 16];
                    for lane in 0..if q { 16 } else { 8 } {
                        wanted[lane] = byte_variable_shift(
                            vector([source[lane]; 16]),
                            signed,
                            counts[lane],
                            true,
                        )[0] as u8;
                    }
                    let mut expected = ctx.v;
                    expected[31] = vector(wanted);
                    let expected_x = ctx.x;
                    invoke(&cache, pointer, &mut ctx, CODE + 4);
                    check_guards(&ctx, expected_x, expected);
                }
            }
        }
    }
}

fn lane_count(input: [u64; 2], q: bool) -> [u64; 2] {
    let input = bytes(input);
    vector(std::array::from_fn(|lane| {
        if lane < if q { 16 } else { 8 } {
            input[lane].count_ones() as u8
        } else {
            0
        }
    }))
}

fn rounding_byte_shift(
    input: [u64; 2],
    old: [u64; 2],
    q: bool,
    shift: u32,
    signed: bool,
    accumulate: bool,
) -> [u64; 2] {
    let input = bytes(input);
    let old = bytes(old);
    vector(std::array::from_fn(|lane| {
        if lane >= if q { 16 } else { 8 } {
            return 0;
        }
        let value = if signed { input[lane] as i8 as i32 } else { input[lane] as i32 };
        let rounded = ((value + (1 << (shift - 1))) >> shift) as u8;
        if accumulate { old[lane].wrapping_add(rounded) } else { rounded }
    }))
}

#[test]
fn guest_cnt_and_byte_rounding_shifts_fuse_and_match_on_every_profile() {
    let mut cases = Vec::new();
    for q in [false, true] {
        for (rn, rd) in [(0u32, 1u32), (1, 1), (31, 0)] {
            cases.push((0x0e20_5800 | (u32::from(q) << 30) | (rn << 5) | rd, q, None));
            for shift in 1..=8u32 {
                for (index, base) in [0x2f00_2400u32, 0x0f00_2400, 0x2f00_3400, 0x0f00_3400]
                    .into_iter()
                    .enumerate()
                {
                    let word = base | (u32::from(q) << 30) | ((16 - shift) << 16) | (rn << 5) | rd;
                    cases.push((word, q, Some((shift, index & 1 != 0, index >= 2))));
                }
            }
        }
    }
    for &(word, _, _) in &cases {
        let mut guest = Block::new(CODE);
        let words = [word, BRK];
        translate_block_into(
            &mut guest,
            CODE,
            &mut |address| words.get(((address - CODE) / 4) as usize).copied(),
            TranslateOptions {
                max_insts: 1,
                multiblock: false,
            },
        )
        .unwrap();
        optimize(&mut guest);
        assert!(
            guest.iter_live().any(|(_, node)| node.op == Op::VecBitAffine),
            "{word:08x}"
        );
    }
    for features in features() {
        for &(word, q, shift) in &cases {
            let words = vec![word, BRK];
            let mut code = Code(words.clone());
            let mut jit = compiler(features);
            let rn = ((word >> 5) & 31) as usize;
            let rd = (word & 31) as usize;
            for input in 0..=255u8 {
                let mut ctx = seeded_context();
                ctx.v[rn] = sample(input);
                let mut expected = ctx.v;
                expected[rd] = match shift {
                    None => lane_count(ctx.v[rn], q),
                    Some((shift, signed, accumulate)) => {
                        rounding_byte_shift(ctx.v[rn], ctx.v[rd], q, shift, signed, accumulate)
                    }
                };
                let expected_x = ctx.x;
                let result = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0), "{word:08x}/{input}/{features:?}");
                assert_eq!(result.retired, 2);
                check_guards(&ctx, expected_x, expected);
            }
        }
    }
}
