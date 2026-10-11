#![cfg(target_arch = "x86_64")]

use rustarmic::backend::cpu_features::{CpuFeatures, detect_features};
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0x1234_5678_9abc_def0;

struct Instructions([u32; 2]);
impl Memory for Instructions {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get(offset as usize / 4).copied()
    }
}

fn vector(lanes: [u32; 4]) -> [u64; 2] {
    [
        u64::from(lanes[0]) | (u64::from(lanes[1]) << 32),
        u64::from(lanes[2]) | (u64::from(lanes[3]) << 32),
    ]
}

fn variants() -> [CpuFeatures; 4] {
    let actual = detect_features();
    let mut avx2 = actual;
    avx2.has_avx512_clz32 = false;
    avx2.has_lzcnt = false;
    avx2.prefer_avx2_clz32 = true;
    let mut lzcnt = actual;
    lzcnt.has_avx512_clz32 = false;
    lzcnt.has_avx2 = false;
    let mut portable = lzcnt;
    portable.has_lzcnt = false;
    [actual, avx2, lzcnt, portable]
}

fn seeded_context() -> CpuContext {
    let mut context = CpuContext::default();
    context.pc = CODE;
    context.x.fill(GUARD);
    context.v = std::array::from_fn(|index| [GUARD ^ index as u64, !GUARD ^ index as u64]);
    context.sp = GUARD;
    context.nzcv = 0xb;
    context.fpcr = 0x03c0_0000;
    context.fpsr = 0x0800_009f;
    context.io_value = [GUARD, !GUARD];
    context
}

fn run_case(
    jit: &mut Jit,
    memory: &mut Instructions,
    q: bool,
    rd: usize,
    rn: usize,
    input: [u32; 4],
) {
    let mut context = seeded_context();
    context.v[rn] = vector(input);
    let mut expected = context.v;
    let mut output = input.map(u32::leading_zeros);
    if !q {
        output[2..].fill(0);
    }
    expected[rd] = vector(output);
    let outcome = jit.run_bounded(&mut context, memory, 2).unwrap();
    assert_eq!(outcome.stop, StopReason::Brk(0));
    assert_eq!(outcome.retired, 2);
    assert_eq!(
        context.v, expected,
        "input={input:08x?} q={q} rd={rd} rn={rn}"
    );
    assert_eq!(context.x, [GUARD; 31]);
    assert_eq!(
        (context.sp, context.nzcv, context.fpcr, context.fpsr),
        (GUARD, 0xb, 0x03c0_0000, 0x0800_009f)
    );
    assert_eq!(context.io_value, [GUARD, !GUARD]);
    assert_eq!((context.mem_fault, context.should_halt), (0, 0));
}

fn compiler(features: CpuFeatures, q: bool, rd: u32, rn: u32) -> (Jit, Instructions) {
    let instruction = 0x2ea0_4800 | (u32::from(q) << 30) | (rn << 5) | rd;
    (
        Jit::new(JitConfig {
            code_cache_bytes: 128 * 1024,
            host_features: Some(features),
            ..JitConfig::default()
        })
        .unwrap(),
        Instructions([instruction, BRK]),
    )
}

#[test]
fn clz32_all_exponent_boundaries_upper_lane_values_aliases_and_feature_paths() {
    for features in variants() {
        for q in [false, true] {
            for (rd, rn) in [(0u32, 1u32), (1, 1), (31, 31), (0, 31), (31, 0)] {
                let (mut jit, mut memory) = compiler(features, q, rd, rn);
                for bit in 0..32 {
                    for offset in 0..=255u32 {
                        let power = 1u32 << bit;
                        let input = [
                            power.wrapping_sub(offset),
                            power.wrapping_add(offset),
                            !power ^ offset,
                            offset.rotate_left(bit),
                        ];
                        run_case(&mut jit, &mut memory, q, rd as usize, rn as usize, input);
                    }
                }
            }
        }
    }
}

#[test]
fn clz32_random_domain_matches_scalar_reference() {
    for features in variants() {
        for q in [false, true] {
            let (mut jit, mut memory) = compiler(features, q, 0, 1);
            let mut state = 0x843c_a891u32;
            for _ in 0..32768 {
                let input = std::array::from_fn(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state
                });
                run_case(&mut jit, &mut memory, q, 0, 1, input);
            }
        }
    }
}

fn read_mxcsr() -> u32 {
    let mut value = 0u32;
    unsafe {
        std::arch::asm!("stmxcsr [{pointer}]", pointer = in(reg) &mut value, options(nostack));
    }
    value
}

fn write_mxcsr(value: u32) {
    unsafe {
        std::arch::asm!("ldmxcsr [{pointer}]", pointer = in(reg) &value, options(nostack));
    }
}

struct RestoreMxcsr(u32);
impl Drop for RestoreMxcsr {
    fn drop(&mut self) {
        write_mxcsr(self.0);
    }
}

#[test]
fn clz32_avx2_trick_preserves_host_mxcsr_guest_status_and_unmasked_precision() {
    let _restore = RestoreMxcsr(read_mxcsr());
    let mut avx2 = detect_features();
    if !avx2.has_avx2 {
        return;
    }
    avx2.has_avx512_clz32 = false;
    avx2.has_lzcnt = false;
    avx2.prefer_avx2_clz32 = true;
    for q in [false, true] {
        let (mut jit, mut memory) = compiler(avx2, q, 0, 1);
        run_case(
            &mut jit,
            &mut memory,
            q,
            0,
            1,
            [0, 1, u32::MAX, 0x8000_0000],
        );
        for round in 0..4 {
            for control in [0x1f80u32, 0x9fc0, 0x0f80, 0x1fbf] {
                let csr = (control & !0x6000) | (round << 13);
                write_mxcsr(csr);
                for input in [
                    [0, 1, 0x8000_0000, u32::MAX],
                    [0x0100_0001, 0x3f00_0001, 0x7fff_ffff, 0xff00_0001],
                ] {
                    run_case(&mut jit, &mut memory, q, 0, 1, input);
                    assert_eq!(read_mxcsr(), csr, "MXCSR round={round} input={input:08x?}");
                }
            }
        }
    }
}

#[test]
#[ignore]
fn matched_direct_thunk_clz32_kernel_paths() {
    use iced_x86::{Decoder, DecoderOptions};
    use rustarmic::arch::RegSize;
    use rustarmic::backend::cpu_features::with_features;
    use rustarmic::backend::emit_block;
    use rustarmic::ir::{Armlet, Block, IrEmitter, Op, Terminal, Ty};
    use rustarmic::jit::code_cache::{CodeCache, ThunkFn};
    use rustarmic::optimizer::optimize;
    use std::hint::black_box;
    use std::time::Instant;

    fn block(legacy: bool) -> Block {
        let mut block = Block::new(CODE);
        let mut em = IrEmitter::new(&mut block, CODE);
        let source = em.get_v_q(1);
        let result = if legacy {
            let zero = em.const_u64(0);
            let mut result = em.vec_build_q(zero, zero);
            for lane in 0..4 {
                let value = em.vec_extract_u32(source, lane);
                let leading = em.push(Armlet::new(Op::Clz32, Ty::U32).with_args(&[value]));
                let count = em.sub(leading, zero, RegSize::W);
                result = em.vec_ins_gpr(result, count, 2, lane, true);
            }
            result
        } else {
            em.vec_clz32(source, true)
        };
        em.set_v_q(0, result);
        block.end_pc = CODE + 4;
        block.cycles = 1;
        block.terminal = Terminal::LinkBlock { next_pc: CODE + 4 };
        optimize(&mut block);
        block
    }

    let actual = detect_features();
    let mut avx2 = actual;
    avx2.has_avx512_clz32 = false;
    avx2.has_lzcnt = false;
    avx2.prefer_avx2_clz32 = true;
    let mut scalar = actual;
    scalar.has_avx512_clz32 = false;
    scalar.has_avx2 = false;
    let mut bsr = scalar;
    bsr.has_lzcnt = false;
    let mut paths = vec![("actual", actual, false)];
    if actual.has_avx2 {
        paths.push(("forced-avx2", avx2, false));
    }
    paths.extend([("scalar", scalar, false), ("bsr", bsr, false), ("old-scalar-ir", scalar, true)]);

    for (name, features, legacy) in paths {
        let block = block(legacy);
        let emitted = with_features(features, || emit_block(&block)).unwrap();
        let instructions: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE).into_iter().collect();
        assert!(instructions.iter().all(|instruction| !instruction.is_invalid()));
        let mut cache = CodeCache::new(256 * 1024).unwrap();
        let pointer = cache.install(CODE, block.end_pc, block.cycles, &emitted.code, &emitted.chains, emitted.body_offset, true).unwrap();
        let thunk: ThunkFn = unsafe { core::mem::transmute(cache.thunk()) };
        for (domain, input) in [("mixed", [0, 1, 0x8000_0000, u32::MAX]), ("zero", [0; 4]), ("nonzero", [1, 0x0100_0001, 0x8000_0000, u32::MAX])] {
            let mut context = seeded_context();
            context.v[1] = vector(input);
            let mut expected = context.v;
            expected[0] = vector(input.map(u32::leading_zeros));
            for _ in 0..10_000 {
                assert_eq!(unsafe { thunk(pointer as u64, &mut context) }, CODE + 4);
            }
            for trial in 0..5 {
                let start = Instant::now();
                for _ in 0..2_000_000 {
                    black_box(unsafe { thunk(black_box(pointer as u64), black_box(&mut context)) });
                }
                let elapsed = start.elapsed();
                assert_eq!(context.v, expected);
                assert_eq!(context.x, [GUARD; 31]);
                assert_eq!((context.sp, context.nzcv, context.fpcr, context.fpsr), (GUARD, 0xb, 0x03c0_0000, 0x0800_009f));
                assert_eq!(context.io_value, [GUARD, !GUARD]);
                println!("clz32 q=true path={name} domain={domain} trial={trial} ns_per_call={:.6} code_bytes={} host_instructions={} avx512={} avx2={} lzcnt={}", elapsed.as_nanos() as f64 / 2_000_000.0, emitted.code.len(), instructions.len(), features.has_avx512_clz32, features.has_avx2, features.has_lzcnt);
            }
        }
    }
}

#[test]
fn clz32_default_integer_policy_and_explicit_float_opt_in_are_distinct() {
    use iced_x86::{Decoder, DecoderOptions, Mnemonic};
    use rustarmic::backend::cpu_features::with_features;
    use rustarmic::backend::emit_block;
    use rustarmic::ir::{Block, IrEmitter, Terminal};
    use rustarmic::optimizer::optimize;

    assert!(!CpuFeatures::default().prefer_avx2_clz32);
    assert!(!detect_features().prefer_avx2_clz32);
    for avx512 in [false, true] {
        for avx2 in [false, true] {
            for lzcnt in [false, true] {
                for prefer in [false, true] {
                    let mut features = detect_features();
                    features.has_avx512_clz32 = avx512;
                    features.has_avx2 = avx2;
                    features.has_lzcnt = lzcnt;
                    features.prefer_avx2_clz32 = prefer;
                    let mut block = Block::new(CODE);
                    let mut em = IrEmitter::new(&mut block, CODE);
                    let source = em.get_v_q(1);
                    let result = em.vec_clz32(source, true);
                    em.set_v_q(0, result);
                    block.end_pc = CODE + 4;
                    block.cycles = 1;
                    block.terminal = Terminal::LinkBlock { next_pc: CODE + 4 };
                    optimize(&mut block);
                    let emitted = with_features(features, || emit_block(&block)).unwrap();
                    let instructions: Vec<_> = Decoder::new(64, &emitted.code, DecoderOptions::NONE).into_iter().collect();
                    assert!(instructions.iter().all(|instruction| !instruction.is_invalid()));
                    let count = |mnemonic| instructions.iter().filter(|instruction| instruction.mnemonic() == mnemonic).count();
                    assert_eq!(count(Mnemonic::Vplzcntd), usize::from(avx512));
                    let floating = !avx512 && avx2 && prefer;
                    assert_eq!(count(Mnemonic::Vcvtdq2ps), usize::from(floating));
                    assert_eq!(count(Mnemonic::Stmxcsr), usize::from(floating));
                    assert_eq!(count(Mnemonic::Ldmxcsr), if floating { 2 } else { 0 });
                    if !avx512 && !floating {
                        assert_eq!(count(Mnemonic::Lzcnt), if lzcnt { 4 } else { 0 });
                        assert_eq!(count(Mnemonic::Bsr), if lzcnt { 0 } else { 4 });
                    }
                }
            }
        }
    }
}
