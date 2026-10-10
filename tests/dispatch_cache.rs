//! Exact-tag dispatch metadata must remain an optimization of the normal
//! CodeCache table, never an alternative authority or an execution shortcut.

use rustarmic::jit::code_cache::CodeCache;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x10_0000;
const GUARD: u64 = 0x1234_5678_9abc_def0;

fn install(cache: &mut CodeCache, pc: u64, count: u32, normal: bool) -> *const u8 {
    cache
        .install(
            pc,
            pc + u64::from(count) * 4,
            count,
            &[0xc3],
            &[],
            0,
            normal,
        )
        .unwrap()
}

fn cache() -> CodeCache {
    CodeCache::new(64 * 1024).unwrap()
}

#[test]
fn exact_tags_resolve_collisions_through_authoritative_table() {
    let mut cache = cache();
    let mut entries = Vec::new();
    // All these PCs deliberately share the folded dispatch-cache index.
    for index in 0..32u64 {
        let pc = CODE + index * 0x4_0000;
        let count = index as u32 % 7 + 1;
        entries.push((pc, install(&mut cache, pc, count, true), count));
    }
    for _ in 0..8 {
        for &(pc, pointer, count) in &entries {
            assert_eq!(cache.lookup_meta(pc), Some((pointer, count)));
            assert_eq!(cache.lookup(pc), Some(pointer));
            assert_eq!(cache.lookup_meta(pc + 0x80_0000), None);
            assert_eq!(cache.lookup_meta(pc), Some((pointer, count)));
        }
    }
}

#[test]
fn candidate_fold12_full_collisions_preserve_high_and_nearby_pc_tags() {
    let mut cache = cache();
    let mut entries = Vec::new();
    // 4 MiB strides collide for both the original 256/fold10 and candidate
    // 1024/fold12 cache. Widely separated high PCs also share the low index.
    for base in [CODE, CODE | (1u64 << 56), CODE | (0xfu64 << 60)] {
        for index in 0..32u64 {
            let pc = base + index * 0x40_0000;
            let count = index as u32 % 7 + 1;
            entries.push((pc, install(&mut cache, pc, count, true), count));
        }
    }
    for _ in 0..8 {
        for &(pc, pointer, count) in &entries {
            assert_eq!(cache.lookup_meta(pc), Some((pointer, count)));
            // These missing full tags cannot be mistaken for a resident slot,
            // even when their high-only difference preserves its folded index.
            assert_eq!(cache.lookup_meta(pc ^ (1u64 << 47)), None);
            assert_eq!(cache.lookup_meta(pc + 4), None);
            assert_eq!(cache.lookup_meta(pc), Some((pointer, count)));
        }
    }
    let (_, first_pointer, first_count) = entries[0];
    let nearby = install(&mut cache, CODE + 4, 11, true);
    for _ in 0..8 {
        assert_eq!(cache.lookup_meta(CODE), Some((first_pointer, first_count)));
        assert_eq!(cache.lookup_meta(CODE + 4), Some((nearby, 11)));
    }
}

#[test]
fn normal_replacement_refreshes_pointer_and_instruction_count() {
    let mut cache = cache();
    let first = install(&mut cache, CODE, 7, true);
    let outside = install(&mut cache, CODE + 0x100, 5, true);
    assert_eq!(cache.lookup_meta(CODE), Some((first, 7)));
    let replacement = install(&mut cache, CODE, 2, true);
    assert_ne!(first, replacement);
    for _ in 0..8 {
        assert_eq!(cache.lookup_meta(CODE), Some((replacement, 2)));
        assert_eq!(cache.lookup_meta(CODE + 0x100), Some((outside, 5)));
    }
}

#[test]
fn overlap_overflow_and_reset_discard_stale_metadata() {
    let mut cache = cache();
    let inside = install(&mut cache, CODE, 8, true);
    let outside = install(&mut cache, CODE + 0x1000, 3, true);
    assert_eq!(cache.lookup_meta(CODE), Some((inside, 8)));
    assert_eq!(cache.lookup_meta(CODE + 0x1000), Some((outside, 3)));
    // The invalidation starts within the block rather than at its entry PC.
    cache.invalidate_range(CODE + 16, 4);
    assert_eq!(cache.lookup_meta(CODE), None);
    assert_eq!(cache.lookup_meta(CODE + 0x1000), Some((outside, 3)));
    cache.invalidate_range(u64::MAX - 1, 8);
    assert_eq!(cache.lookup_meta(CODE + 0x1000), None);
    let second = install(&mut cache, CODE + 0x2000, 9, true);
    assert_eq!(cache.lookup_meta(CODE + 0x2000), Some((second, 9)));
    cache.reset().unwrap();
    assert_eq!(cache.lookup_meta(CODE + 0x2000), None);
    let replacement = install(&mut cache, CODE, 1, true);
    assert_eq!(replacement, inside, "reset reuses the original code cursor");
    assert_eq!(cache.lookup_meta(CODE), Some((replacement, 1)));
    assert_eq!(cache.lookup_meta(CODE + 0x1000), None);
}

#[test]
fn transient_same_pc_never_replaces_normal_metadata() {
    let mut cache = cache();
    let normal = install(&mut cache, CODE, 16, true);
    assert_eq!(cache.lookup_meta(CODE), Some((normal, 16)));
    let transient = install(&mut cache, CODE, 1, false);
    assert_ne!(normal, transient);
    let other_transient = install(&mut cache, CODE + 0x100, 2, false);
    assert_ne!(normal, other_transient);
    for _ in 0..16 {
        assert_eq!(cache.lookup_meta(CODE), Some((normal, 16)));
        assert_eq!(cache.lookup_meta(CODE + 0x100), None);
    }
}

struct Program {
    words: Vec<u32>,
    fetches: u64,
}

impl Memory for Program {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.fetches += 1;
        let offset = addr.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.words.get((offset / 4) as usize).copied()
    }
}

fn context() -> CpuContext {
    let mut ctx = CpuContext::default();
    ctx.x = [GUARD; 31];
    ctx.x[0] = 0;
    ctx.v = [[GUARD, !GUARD]; 32];
    ctx.pc = CODE;
    ctx.sp = GUARD;
    ctx.nzcv = 0xf;
    ctx.fpcr = 0x0300_0000;
    ctx.fpsr = 0x0800_009f;
    ctx
}

fn check(ctx: &CpuContext, increments: u64, pc: u64) {
    let mut expected = [GUARD; 31];
    expected[0] = increments;
    assert_eq!(ctx.x, expected);
    assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
    assert_eq!(ctx.pc, pc);
    assert_eq!(
        (ctx.sp, ctx.nzcv, ctx.fpcr, ctx.fpsr),
        (GUARD, 0xf, 0x0300_0000, 0x0800_009f)
    );
    assert_eq!((ctx.mem_fault, ctx.should_halt), (0, 0));
}

fn jit() -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 64 * 1024,
        ..JitConfig::default()
    })
    .unwrap()
}

#[test]
fn warmed_normal_block_keeps_exact_zero_and_one_instruction_budgets() {
    // A cached four-instruction block must not bypass a one-instruction limit.
    let mut program = Program {
        words: vec![0x9100_0400, 0x9100_0400, 0x9100_0400, 0x17ff_fffd],
        fetches: 0,
    };
    let before = program.words.clone();
    let mut jit = jit();
    let mut warm = context();
    let result = jit.run_bounded(&mut warm, &mut program, 1024).unwrap();
    assert_eq!(
        (result.stop, result.retired),
        (StopReason::BudgetExhausted, 1024)
    );
    check(&warm, 768, CODE);
    let normal = jit.cache.lookup_meta(CODE).unwrap();
    assert_eq!(normal.1, 4);
    let mut ctx = context();
    let fetches = program.fetches;
    let result = jit.run_bounded(&mut ctx, &mut program, 0).unwrap();
    assert_eq!(
        (result.stop, result.retired),
        (StopReason::BudgetExhausted, 0)
    );
    assert_eq!(program.fetches, fetches);
    check(&ctx, 0, CODE);
    for _ in 0..8 {
        ctx = context();
        let result = jit.run_bounded(&mut ctx, &mut program, 1).unwrap();
        assert_eq!(
            (result.stop, result.retired),
            (StopReason::BudgetExhausted, 1)
        );
        check(&ctx, 1, CODE + 4);
        assert_eq!(jit.cache.lookup_meta(CODE), Some(normal));
    }
    ctx = context();
    let fetches = program.fetches;
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!(
        (result.stop, result.retired),
        (StopReason::BudgetExhausted, 1024)
    );
    check(&ctx, 768, CODE);
    assert_eq!(
        program.fetches, fetches,
        "normal metadata must remain cached"
    );
    assert_eq!(program.words, before);
}

#[test]
fn warm_dispatch_respects_halts_and_recompiled_svc_after_invalidation() {
    let mut program = Program {
        words: vec![0x9100_0400, 0x17ff_ffff],
        fetches: 0,
    };
    let mut jit = jit();
    let mut ctx = context();
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!(
        (result.stop, result.retired),
        (StopReason::BudgetExhausted, 1024)
    );
    check(&ctx, 512, CODE);
    let fetches = program.fetches;
    ctx = context();
    ctx.should_halt = 1;
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!((result.stop, result.retired), (StopReason::Halted, 0));
    assert_eq!(program.fetches, fetches);
    check(&ctx, 0, CODE);
    // Invalidating only the second instruction evicts the whole old block.
    program.words[1] = 0xd400_00e1; // SVC #7.
    jit.invalidate_range(CODE + 4, 4);
    ctx = context();
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!((result.stop, result.retired), (StopReason::Svc(7), 2));
    check(&ctx, 1, CODE + 8);
    let fetches = program.fetches;
    ctx = context();
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!((result.stop, result.retired), (StopReason::Svc(7), 2));
    assert_eq!(program.fetches, fetches);
    check(&ctx, 1, CODE + 8);
}

#[test]
fn invalidation_prevents_warm_pointer_from_hiding_execute_fault() {
    let mut program = Program {
        words: vec![0x9100_0400, 0x17ff_ffff],
        fetches: 0,
    };
    let mut jit = jit();
    let mut ctx = context();
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!(result.retired, 1024);
    assert!(jit.cache.lookup_meta(CODE).is_some());
    program.words.clear();
    jit.invalidate_range(CODE, 8);
    ctx = context();
    let result = jit.run_bounded(&mut ctx, &mut program, 1024).unwrap();
    assert_eq!(result.retired, 0);
    let StopReason::MemoryFault(fault) = result.stop else {
        panic!("invalidated unmapped code must fault");
    };
    assert_eq!((fault.pc, fault.address, fault.size), (CODE, CODE, 4));
    assert_eq!(fault.access, rustarmic::MemoryAccess::Execute);
    check(&ctx, 0, CODE);
}
