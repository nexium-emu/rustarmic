use disarm64::decoder::{
    ASIMDSHF, ASISDSHF, SLI_Sd_Sn_IMM_VLSL, SLI_Vd_Vn_IMM_VLSL, SRI_Sd_Sn_IMM_VLSR,
    SRI_Vd_Vn_IMM_VLSR,
};
use rustarmic::backend::cpu_features::{CpuFeatures, detect_features, with_features};
use rustarmic::backend::emit_block;
use rustarmic::frontend::translate::asimd_shf;
use rustarmic::frontend::translator::translate_instruction;
use rustarmic::ir::{Block, IrEmitter, Op, Terminal};
use rustarmic::jit::code_cache::{CodeCache, ThunkFn};
use rustarmic::optimizer::optimize;
use rustarmic::{CpuContext, Error, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const ARRANGEMENTS: [(u32, bool); 7] = [
    (0, false),
    (0, true),
    (1, false),
    (1, true),
    (2, false),
    (2, true),
    (3, true),
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

fn word(left: bool, size: u32, q: bool, scalar: bool, shift: u32, rn: u32, rd: u32) -> u32 {
    let width = 8 << size;
    let base = if left { 0x2f00_5400 } else { 0x2f00_4400 };
    base | (u32::from(scalar) << 28)
        | (u32::from(scalar || q) << 30)
        | ((if left {
            width + shift
        } else {
            2 * width - shift
        }) << 16)
        | (rn << 5)
        | rd
}

fn bytes(value: [u64; 2]) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&value[0].to_le_bytes());
    out[8..].copy_from_slice(&value[1].to_le_bytes());
    out
}

fn vector(value: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(value[..8].try_into().unwrap()),
        u64::from_le_bytes(value[8..].try_into().unwrap()),
    ]
}

fn expected(
    source: [u64; 2],
    old: [u64; 2],
    left: bool,
    size: u32,
    q: bool,
    scalar: bool,
    shift: u32,
) -> [u64; 2] {
    let source = bytes(source);
    let old = bytes(old);
    let lane_bytes = 1usize << size;
    let width = lane_bytes * 8;
    let count = if scalar {
        1
    } else {
        (if q { 16 } else { 8 }) / lane_bytes
    };
    let mut out = [0; 16];
    for lane in 0..count {
        for bit in 0..width {
            let source_bit = if left {
                bit.checked_sub(shift as usize)
            } else {
                (bit + (shift as usize) < width).then_some(bit + shift as usize)
            };
            let (input, input_bit) = if let Some(source_bit) = source_bit {
                (&source, source_bit)
            } else {
                (&old, bit)
            };
            let value = (input[lane * lane_bytes + input_bit / 8] >> (input_bit % 8)) & 1;
            out[lane * lane_bytes + bit / 8] |= value << (bit % 8);
        }
    }
    vector(out)
}

fn context() -> CpuContext {
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

fn check(ctx: &CpuContext, expected_x: [u64; 31], expected_v: [[u64; 2]; 32]) {
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

fn samples(size: u32) -> Vec<([u64; 2], [u64; 2])> {
    let width = 8 << size;
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    let values = [0, 1, 2, 3, sign - 1, sign, sign + 1, mask - 1, mask];
    let lane_bytes = 1usize << size;
    let mut out = Vec::new();
    for a in 0..values.len() {
        for b in 0..values.len() {
            let mut source = [0; 16];
            let mut old = [0; 16];
            for lane in 0..16 / lane_bytes {
                source[lane * lane_bytes..(lane + 1) * lane_bytes].copy_from_slice(
                    &values[(a + lane * 3) % values.len()].to_le_bytes()[..lane_bytes],
                );
                old[lane * lane_bytes..(lane + 1) * lane_bytes].copy_from_slice(
                    &values[(b + lane * 5) % values.len()].to_le_bytes()[..lane_bytes],
                );
            }
            out.push((vector(source), vector(old)));
        }
    }
    for bit in 0..width {
        out.push(([1 << bit, !(1 << bit)], [!(1 << bit), 1 << bit]));
        out.push(([!(1 << bit), 1 << bit], [1 << bit, !(1 << bit)]));
    }
    let mut random = 0x534c_4953_5249_0080u64;
    for _ in 0..32 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        out.push(([next(), next()], [next(), next()]));
    }
    out
}

fn matrix(left: bool, scalar: bool) {
    for features in features() {
        for &(size, q) in if scalar {
            &[(3, false)][..]
        } else {
            &ARRANGEMENTS[..]
        } {
            let patterns = samples(size);
            let width = 8 << size;
            for shift in if left { 0..width } else { 1..width + 1 } {
                for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
                    let instruction = word(left, size, q, scalar, shift, rn, rd);
                    let words = vec![instruction, BRK];
                    let mut code = Code(words.clone());
                    let mut jit = compiler(features);
                    for &(source, old) in &patterns {
                        let mut ctx = context();
                        ctx.v[rd as usize] = old;
                        ctx.v[rn as usize] = source;
                        let mut wanted = ctx.v;
                        wanted[rd as usize] = expected(
                            ctx.v[rn as usize],
                            ctx.v[rd as usize],
                            left,
                            size,
                            q,
                            scalar,
                            shift,
                        );
                        let expected_x = ctx.x;
                        let result = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                        assert_eq!(
                            result.stop,
                            StopReason::Brk(0),
                            "{instruction:08x}/{features:?}"
                        );
                        assert_eq!(result.retired, 2);
                        assert_eq!(ctx.pc, CODE + 8);
                        check(&ctx, expected_x, wanted);
                        assert_eq!(code.0, words);
                    }
                }
            }
        }
    }
}

#[test]
fn vector_sli_all_legal_arrangements_counts_aliases_and_bit_positions() {
    matrix(true, false);
}

#[test]
fn vector_sri_all_legal_arrangements_counts_aliases_and_bit_positions() {
    matrix(false, false);
}

#[test]
fn scalar_d_sli_sri_all_counts_aliases_and_inactive_upper_zero() {
    matrix(true, true);
    matrix(false, true);
}

#[test]
fn every_byte_source_destination_pair_all_counts_host_and_portable() {
    for features in features() {
        for left in [false, true] {
            for q in [false, true] {
                let lanes = if q { 16 } else { 8 };
                for shift in if left { 0..8 } else { 1..9 } {
                    let instruction = word(left, 0, q, false, shift, 31, 0);
                    let mut code = Code(vec![instruction, BRK]);
                    let mut jit = compiler(features);
                    for input in 0..=255u8 {
                        for old_base in (0..256usize).step_by(lanes) {
                            let mut ctx = context();
                            ctx.v[31] = vector([input; 16]);
                            ctx.v[0] = vector(std::array::from_fn(|lane| (old_base + lane) as u8));
                            let mut wanted = ctx.v;
                            wanted[0] = expected(ctx.v[31], ctx.v[0], left, 0, q, false, shift);
                            let expected_x = ctx.x;
                            let result = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                            assert_eq!((result.stop, result.retired), (StopReason::Brk(0), 2));
                            check(&ctx, expected_x, wanted);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn mixed_insert_chain_preserves_each_budget_cutpoint_and_full_width_endpoint() {
    for features in features() {
        let operations = [
            (true, 0, false, false, 3, 0, 31),
            (false, 0, false, false, 8, 31, 31),
            (true, 3, false, true, 0, 30, 31),
            (false, 3, false, true, 64, 31, 31),
        ];
        let mut words: Vec<_> = operations
            .iter()
            .map(|&(left, size, q, scalar, shift, rn, rd)| {
                word(left, size, q, scalar, shift, rn, rd)
            })
            .collect();
        words.push(BRK);
        let mut code = Code(words.clone());
        let mut jit = compiler(features);
        let mut warm = context();
        assert_eq!(jit.run_bounded(&mut warm, &mut code, 5).unwrap().retired, 5);
        for budget in 0..=5 {
            let mut ctx = context();
            let mut wanted = ctx.v;
            for &(left, size, q, scalar, shift, rn, rd) in
                operations.iter().take((budget as usize).min(4))
            {
                wanted[rd as usize] = expected(
                    wanted[rn as usize],
                    wanted[rd as usize],
                    left,
                    size,
                    q,
                    scalar,
                    shift,
                );
            }
            let expected_x = ctx.x;
            let result = jit.run_bounded(&mut ctx, &mut code, budget).unwrap();
            assert_eq!(result.retired, budget);
            assert_eq!(
                result.stop,
                if budget == 5 {
                    StopReason::Brk(0)
                } else {
                    StopReason::BudgetExhausted
                }
            );
            assert_eq!(ctx.pc, CODE + budget * 4);
            check(&ctx, expected_x, wanted);
            if budget < 5 {
                for &(left, size, q, scalar, shift, rn, rd) in
                    operations.iter().skip(budget as usize)
                {
                    wanted[rd as usize] = expected(
                        wanted[rn as usize],
                        wanted[rd as usize],
                        left,
                        size,
                        q,
                        scalar,
                        shift,
                    );
                }
                let result = jit.run_bounded(&mut ctx, &mut code, 5 - budget).unwrap();
                assert_eq!(
                    (result.stop, result.retired),
                    (StopReason::Brk(0), 5 - budget)
                );
                assert_eq!(ctx.pc, CODE + 20);
                check(&ctx, expected_x, wanted);
            }
            assert_eq!(code.0, words);
        }
    }
}

#[test]
fn invalid_insert_sizes_are_rejected_before_sources_are_captured() {
    for left in [false, true] {
        for scalar in [false, true] {
            for q in [false, true] {
                for immh in 0..16 {
                    if immh != 0 && !(scalar && immh & 8 == 0) && !(!scalar && immh & 8 != 0 && !q)
                    {
                        continue;
                    }
                    let raw = (if left { 0x2f00_5400 } else { 0x2f00_4400 })
                        | (u32::from(scalar) << 28)
                        | (u32::from(scalar || q) << 30)
                        | (immh << 19)
                        | (31 << 5)
                        | 31;
                    let mut block = Block::new(CODE);
                    let mut em = IrEmitter::new(&mut block, CODE);
                    let error = if scalar {
                        let instruction = if left {
                            ASISDSHF::SLI_Sd_Sn_IMM_VLSL(SLI_Sd_Sn_IMM_VLSL(raw))
                        } else {
                            ASISDSHF::SRI_Sd_Sn_IMM_VLSR(SRI_Sd_Sn_IMM_VLSR(raw))
                        };
                        asimd_shf::translate_scalar(&mut em, instruction).unwrap_err()
                    } else {
                        let instruction = if left {
                            ASIMDSHF::SLI_Vd_Vn_IMM_VLSL(SLI_Vd_Vn_IMM_VLSL(raw))
                        } else {
                            ASIMDSHF::SRI_Vd_Vn_IMM_VLSR(SRI_Vd_Vn_IMM_VLSR(raw))
                        };
                        asimd_shf::translate(&mut em, instruction).unwrap_err()
                    };
                    assert!(
                        matches!(error, Error::Unsupported { pc: CODE, opcode } if opcode == raw)
                    );
                    assert_eq!(block.len(), 1);
                }
            }
        }
    }
}

#[test]
fn distinct_insert_operands_retain_two_sources_not_a_false_unary_affine_map() {
    for left in [false, true] {
        let mut block = Block::new(CODE);
        translate_instruction(
            &mut IrEmitter::new(&mut block, CODE),
            word(left, 0, true, false, 3, 1, 2),
        )
        .unwrap();
        block.cycles = 1;
        optimize(&mut block);
        assert_eq!(
            block.iter_live().filter(|(_, a)| a.op == Op::GetV).count(),
            2
        );
        assert_eq!(
            block
                .iter_live()
                .filter(|(_, a)| a.op == Op::VecOrr)
                .count(),
            1
        );
        assert_eq!(block.cycles, 1);
    }
}

#[test]
fn insert_preserves_live_integer_and_full_vector_snapshots_under_pressure() {
    for features in features() {
        for left in [false, true] {
            for &(size, q, scalar) in &[
                (0, false, false),
                (0, true, false),
                (1, true, false),
                (2, true, false),
                (3, true, false),
                (3, false, true),
            ] {
                let width = 8 << size;
                for shift in [
                    if left { 0 } else { 1 },
                    width - 1,
                    if left { width - 1 } else { width },
                ] {
                    let mut block = Block::new(CODE);
                    let mut em = IrEmitter::new(&mut block, CODE);
                    let vectors: Vec<_> = (0..16).map(|reg| em.get_v_q(reg)).collect();
                    let integers: Vec<_> = (0..16).map(|reg| em.get_x(reg)).collect();
                    translate_instruction(&mut em, word(left, size, q, scalar, shift, 30, 31))
                        .unwrap();
                    for (index, value) in vectors.into_iter().enumerate() {
                        em.set_v_q(if index == 15 { 0 } else { index as u8 + 16 }, value);
                    }
                    for (index, value) in integers.into_iter().enumerate() {
                        em.set_x(index as u8 + 15, value);
                    }
                    block.cycles = 3;
                    block.end_pc = CODE + 12;
                    block.terminal = Terminal::LinkBlock { next_pc: CODE + 12 };
                    optimize(&mut block);
                    let emitted = with_features(features, || emit_block(&block)).unwrap();
                    let mut cache = CodeCache::new(256 * 1024).unwrap();
                    let pointer = cache
                        .install(
                            CODE,
                            CODE + 12,
                            3,
                            &emitted.code,
                            &emitted.chains,
                            emitted.body_offset,
                            true,
                        )
                        .unwrap();
                    let thunk: ThunkFn = unsafe { core::mem::transmute(cache.thunk()) };
                    let mut ctx = context();
                    let mut wanted = ctx.v;
                    wanted[31] = expected(ctx.v[30], ctx.v[31], left, size, q, scalar, shift);
                    wanted[16..31].copy_from_slice(&ctx.v[..15]);
                    wanted[0] = ctx.v[15];
                    let mut expected_x = ctx.x;
                    expected_x[15..31].copy_from_slice(&ctx.x[..16]);
                    assert_eq!(unsafe { thunk(pointer as u64, &mut ctx) }, CODE + 12);
                    check(&ctx, expected_x, wanted);
                }
            }
        }
    }
}
