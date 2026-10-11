use iced_x86::{Decoder, DecoderOptions, Mnemonic, Register};
use rustarmic::arch::RegSize;
use rustarmic::backend::cpu_features::{detect_features, with_features};
use rustarmic::backend::emit_block;
use rustarmic::frontend::translator::translate_instruction;
use rustarmic::ir::{Armlet, Block, IrEmitter, Op, Terminal, Ty};
use rustarmic::jit::code_cache::{CodeCache, ThunkFn};
use rustarmic::optimizer::optimize;
use rustarmic::{CpuContext, CpuFeatures, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const BRK: u32 = 0xd420_0000;

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
    let mut scalar = detected;
    scalar.has_avx512_clz16 = false;
    let mut bsr = scalar;
    bsr.has_lzcnt = false;
    let mut result = vec![scalar, bsr];
    if detected.has_avx512_clz16 {
        result.push(detected);
    }
    result
}

fn compiler(features: CpuFeatures) -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 256 * 1024,
        host_features: Some(features),
        ..JitConfig::default()
    })
    .unwrap()
}

fn instruction(q: bool, rn: u32, rd: u32) -> u32 {
    0x2e60_4800 | (u32::from(q) << 30) | (rn << 5) | rd
}

fn vector(lanes: [u16; 8]) -> [u64; 2] {
    let mut bytes = [0; 16];
    for (lane, value) in lanes.into_iter().enumerate() {
        bytes[lane * 2..lane * 2 + 2].copy_from_slice(&value.to_le_bytes());
    }
    [
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ]
}

fn expected(source: [u64; 2], q: bool) -> [u64; 2] {
    let bytes = [source[0].to_le_bytes(), source[1].to_le_bytes()].concat();
    let mut out = [0; 8];
    for lane in 0..if q { 8 } else { 4 } {
        let value = u16::from_le_bytes(bytes[lane * 2..lane * 2 + 2].try_into().unwrap());
        for bit in (0..16).rev() {
            if value & (1 << bit) != 0 {
                break;
            }
            out[lane] += 1;
        }
    }
    vector(out)
}

fn seeded_context() -> CpuContext {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x = std::array::from_fn(|r| GUARD.wrapping_add(r as u64 * 0x1_0001));
    ctx.v = std::array::from_fn(|r| [GUARD.rotate_left(r as u32), (!GUARD).rotate_right(r as u32)]);
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

fn check(ctx: &CpuContext, x: [u64; 31], v: [[u64; 2]; 32]) {
    assert_eq!(ctx.x, x);
    assert_eq!(ctx.v, v);
    assert_eq!((ctx.sp, ctx.nzcv, ctx.fpcr, ctx.fpsr), (GUARD, 0xb, 0x03c0_0000, 0x0800_009f));
    assert_eq!((ctx.tpidr_el0, ctx.tpidrro_el0), (GUARD, !GUARD));
    assert_eq!((ctx.exclusive_addr, ctx.exclusive_value, ctx.exclusive_size), (GUARD, !GUARD, 8));
    assert_eq!(ctx.io_value, [GUARD, !GUARD]);
    assert_eq!((ctx.mem_fault, ctx.mem_fault_size, ctx.mem_fault_access, ctx.mem_fault_cause), (0, 0, 0, 0));
    assert_eq!((ctx.mem_fault_pc, ctx.mem_fault_addr), (0, 0));
    assert_eq!(ctx.should_halt, 0);
}

fn execute(jit: &mut Jit, code: &mut Code, source: [u64; 2], q: bool, rn: u32, rd: u32) {
    let words = code.0.clone();
    let mut ctx = seeded_context();
    ctx.v[rn as usize] = source;
    let mut wanted = ctx.v;
    wanted[rd as usize] = expected(source, q);
    let x = ctx.x;
    let outcome = jit.run_bounded(&mut ctx, code, 2).unwrap();
    assert_eq!(outcome.stop, StopReason::Brk(0));
    assert_eq!(outcome.retired, 2);
    assert_eq!(ctx.pc, CODE + 8);
    check(&ctx, x, wanted);
    assert_eq!(code.0, words);
}

#[test]
fn every_halfword_in_every_active_lane_q0_q1_hardware_and_scalar_paths() {
    for features in features() {
        for q in [false, true] {
            let mut jit = compiler(features);
            let mut code = Code(vec![instruction(q, 0, 1), BRK]);
            for value in 0..=u16::MAX {
                let source = vector(std::array::from_fn(|lane| value.wrapping_add(lane as u16 * 7919)));
                execute(&mut jit, &mut code, source, q, 0, 1);
            }
        }
    }
}

#[test]
fn all_registers_aliases_zero_maximum_and_distinct_upper_four_lanes() {
    let mut sources = vec![
        vector([0, 1, 0x8000, 0xffff, 0, 0xffff, 1, 0x8000]),
        vector([0xffff, 0, 0xffff, 0, 1, 2, 4, 8]),
    ];
    for bit in 0..16 {
        let value = 1u16 << bit;
        sources.push(vector([value, value - 1, !value, value.wrapping_add(1), 0xffff, 0, 0x8000, value]));
    }
    for features in features() {
        for q in [false, true] {
            for rn in 0..32 {
                for rd in [rn, (rn + 13) % 32] {
                    let mut jit = compiler(features);
                    let mut code = Code(vec![instruction(q, rn, rd), BRK]);
                    for &source in &sources {
                        execute(&mut jit, &mut code, source, q, rn, rd);
                    }
                }
            }
        }
    }
}

#[test]
fn halfword_counts_preserve_live_vectors_gprs_status_under_pressure() {
    for features in features() {
        for q in [false, true] {
            let mut words = Vec::new();
            for r in 0..16 {
                words.push(0x9100_0400 | (r << 5) | r);
                words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
            }
            words.push(instruction(q, 31, 31));
            for p in 0..8 {
                words.push(0x8b00_0000 | ((p * 2 + 1) << 16) | ((p * 2) << 5) | if p == 7 { 16 } else { 24 + p });
            }
            for r in 0..14 {
                words.push(0x4ea0_1c00 | (r << 16) | (r << 5) | (17 + r));
            }
            words.push(BRK);
            let mut code = Code(words.clone());
            let mut ctx = seeded_context();
            ctx.v[31] = vector([0, 1, 2, 4, 8, 0x80, 0x8000, 0xffff]);
            let mut x = ctx.x;
            let mut v = ctx.v;
            for r in 0..16 {
                x[r] = x[r].wrapping_add(1);
                v[r] = [v[r][0] | v[16][0], v[r][1] | v[16][1]];
            }
            v[31] = expected(v[31], q);
            for p in 0..8 {
                x[if p == 7 { 16 } else { 24 + p }] = x[p * 2].wrapping_add(x[p * 2 + 1]);
            }
            for r in 0..14 {
                v[17 + r] = v[r];
            }
            let result = compiler(features).run_bounded(&mut ctx, &mut code, words.len() as u64).unwrap();
            assert_eq!(result.stop, StopReason::Brk(0));
            assert_eq!(result.retired, words.len() as u64);
            check(&ctx, x, v);
            assert_eq!(code.0, words);
        }
    }
}

fn ir(q: bool, legacy: bool) -> Block {
    let mut block = Block::new(CODE);
    let mut em = IrEmitter::new(&mut block, CODE);
    let source = em.get_v_q(0);
    let result = if legacy {
        let z = em.const_u64(0);
        let mut result = em.vec_build_q(z, z);
        let adjustment = em.const_u64(16);
        for lane in 0..if q { 8 } else { 4 } {
            let value = em.vec_extract_u16(source, lane);
            let leading = em.push(Armlet::new(Op::Clz32, Ty::U32).with_args(&[value]));
            let count = em.sub(leading, adjustment, RegSize::W);
            result = em.vec_ins_gpr(result, count, 1, lane, q);
        }
        result
    } else {
        em.vec_clz16(source, q)
    };
    em.set_v_q(1, result);
    block.cycles = 1;
    block.end_pc = CODE + 4;
    block.terminal = Terminal::LinkBlock { next_pc: CODE + 4 };
    optimize(&mut block);
    block
}

#[test]
fn hardware_encoding_narrows_all_eight_dwords_without_pack_lane_loss() {
    let mut features = detect_features();
    features.has_avx512_clz16 = true;
    for q in [false, true] {
        let emitted = with_features(features, || emit_block(&ir(q, false))).unwrap();
        let instructions: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE).into_iter().collect();
        assert!(instructions.iter().all(|instruction| !instruction.is_invalid()));
        let widen = instructions.iter().find(|instruction| instruction.mnemonic() == Mnemonic::Vpmovzxwd).unwrap();
        assert_eq!((widen.op0_register(), widen.op1_register()), (Register::YMM0, Register::XMM0));
        let count = instructions.iter().find(|instruction| instruction.mnemonic() == Mnemonic::Vplzcntd).unwrap();
        assert_eq!((count.op0_register(), count.op1_register()), (Register::YMM0, Register::YMM0));
        let narrow = instructions.iter().find(|instruction| instruction.mnemonic() == Mnemonic::Vpmovdw).unwrap();
        assert_eq!((narrow.op0_register(), narrow.op1_register()), (Register::XMM0, Register::YMM0));
        assert_eq!(instructions.iter().filter(|instruction| instruction.mnemonic() == Mnemonic::Vzeroupper).count(), 1);
        let narrowed = instructions.iter().position(|instruction| instruction.mnemonic() == Mnemonic::Vpmovdw).unwrap();
        assert_eq!(instructions[narrowed + 1].mnemonic(), Mnemonic::Vzeroupper);
        assert!(!instructions.iter().any(|instruction| matches!(instruction.mnemonic(), Mnemonic::Vpackusdw | Mnemonic::Packusdw | Mnemonic::Call)));
    }
}

#[test]
fn guest_halfword_and_word_clz_dispatch_to_dedicated_lowerings() {
    for (word, op) in [
        (0x2e60_4820u32, Op::VecClz16),
        (0x6e60_4820, Op::VecClz16),
        (0x2ea0_4820, Op::VecClz32),
        (0x6ea0_4820, Op::VecClz32),
    ] {
        let mut block = Block::new(CODE);
        translate_instruction(&mut IrEmitter::new(&mut block, CODE), word).unwrap();
        assert_eq!(block.iter_live().filter(|(_, node)| node.op == op).count(), 1, "{word:08x}");
        assert!(!block.iter_live().any(|(_, node)| node.op == Op::Clz32), "{word:08x}");
    }
}

#[test]
fn forced_off_emission_keeps_integer_fallback_and_correct_clobbers() {
    for mut features in features() {
        features.has_avx512_clz16 = false;
        let emitted = with_features(features, || emit_block(&ir(true, false))).unwrap();
        let instructions: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE).into_iter().collect();
        assert!(instructions.iter().all(|instruction| !instruction.is_invalid()));
        assert!(!instructions.iter().any(|instruction| matches!(instruction.mnemonic(), Mnemonic::Vplzcntd | Mnemonic::Vpmovdw | Mnemonic::Vpmovzxwd | Mnemonic::Call)));
        assert_eq!(instructions.iter().filter(|instruction| instruction.mnemonic() == Mnemonic::Pextrw).count(), 8);
        assert_eq!(instructions.iter().filter(|instruction| instruction.mnemonic() == Mnemonic::Pinsrw).count(), 8);
        let clobbers = rustarmic::backend::clobbers::clobbers_for_op(Op::VecClz16);
        assert_eq!(clobbers.xmm.bits(), 3);
        assert!(clobbers.gpr.contains(rustarmic::backend::clobbers::GprMask::RAX | rustarmic::backend::clobbers::GprMask::RCX));
    }
}

#[test]
#[ignore]
fn matched_direct_thunk_kernel_benchmark_old_new_hardware_off() {
    use std::hint::black_box;
    use std::time::Instant;
    let detected = detect_features();
    let mut scalar = detected;
    scalar.has_avx512_clz16 = false;
    for q in [false, true] {
        for (name, features, legacy) in [("old", scalar, true), ("new-off", scalar, false), ("new-host", detected, false)] {
            let block = ir(q, legacy);
            let emitted = with_features(features, || emit_block(&block)).unwrap();
            let mut cache = CodeCache::new(256 * 1024).unwrap();
            let pointer = cache.install(CODE, block.end_pc, block.cycles, &emitted.code, &emitted.chains, emitted.body_offset, true).unwrap();
            let thunk: ThunkFn = unsafe { core::mem::transmute(cache.thunk()) };
            let mut ctx = seeded_context();
            ctx.v[0] = vector([0, 1, 2, 4, 8, 0x80, 0x8000, 0xffff]);
            let mut wanted = ctx.v;
            wanted[1] = expected(ctx.v[0], q);
            let x = ctx.x;
            for _ in 0..10_000 {
                black_box(unsafe { thunk(pointer as u64, &mut ctx) });
            }
            for trial in 0..5 {
                let start = Instant::now();
                for _ in 0..2_000_000 {
                    black_box(unsafe { thunk(black_box(pointer as u64), black_box(&mut ctx)) });
                }
                println!("clz16 q={q} path={name} trial={trial} ns_per_call={:.6} code_bytes={} hardware={}", start.elapsed().as_nanos() as f64 / 2_000_000.0, emitted.code.len(), detected.has_avx512_clz16);
                check(&ctx, x, wanted);
            }
        }
    }
}
