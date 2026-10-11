//! Block chaining, the dispatch table and the in-code budget / halt checks.

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::cell::Cell;

const CODE: u64 = 0x1000;
const BRK_0: u32 = 0xD420_0000;
const NOP: u32 = 0xD503_201F;
const RET: u32 = 0xD65F_03C0;

fn movz(rd: u32, imm: u32) -> u32 {
    0xD280_0000 | (imm << 5) | rd
}
fn add_imm(rd: u32, rn: u32, imm: u32) -> u32 {
    0x9100_0000 | (imm << 10) | (rn << 5) | rd
}
fn cmp_imm(rn: u32, imm: u32) -> u32 {
    0xF100_001F | (imm << 10) | (rn << 5)
}
fn off(from: usize, to: usize) -> i64 {
    (to as i64 - from as i64) * 4
}
fn b(from: usize, to: usize) -> u32 {
    0x1400_0000 | ((off(from, to) >> 2) as u32 & 0x03FF_FFFF)
}
fn bl(from: usize, to: usize) -> u32 {
    0x9400_0000 | ((off(from, to) >> 2) as u32 & 0x03FF_FFFF)
}
fn b_ne(from: usize, to: usize) -> u32 {
    0x5400_0001 | ((((off(from, to) >> 2) as u32) & 0x7FFFF) << 5)
}
fn at(i: usize) -> u64 {
    CODE + 4 * i as u64
}

struct Code(Vec<u32>);

impl Memory for Code {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let i = addr.checked_sub(CODE)? / 4;
        self.0.get(i as usize).copied()
    }
}

thread_local! {
    static STORES: Cell<u64> = const { Cell::new(0) };
    static HALT_AFTER: Cell<u64> = const { Cell::new(u64::MAX) };
    static FAULT_AFTER: Cell<u64> = const { Cell::new(u64::MAX) };
}

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, _addr: u64, _size: u8) {
    unsafe { (*ctx).io_value = [0, 0] };
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    let n = STORES.with(|c| {
        c.set(c.get() + 1);
        c.get()
    });
    unsafe {
        if n == HALT_AFTER.with(|c| c.get()) {
            (*ctx).should_halt = 1;
        }
        if n == FAULT_AFTER.with(|c| c.get()) {
            let ctx = &mut *ctx;
            ctx.mem_fault = 1;
            ctx.mem_fault_access = 1;
            ctx.mem_fault_size = size;
            ctx.mem_fault_addr = addr;
            ctx.mem_fault_pc = ctx.pc;
        }
    }
}

fn ctx() -> CpuContext {
    STORES.with(|c| c.set(0));
    HALT_AFTER.with(|c| c.set(u64::MAX));
    FAULT_AFTER.with(|c| c.set(u64::MAX));
    let mut ctx = CpuContext::default();
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
    ctx.pc = CODE;
    ctx
}

fn jit() -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 4 << 20,
        ..JitConfig::default()
    })
    .unwrap()
}

/// Without multiblock translation every `B` ends a block, so these tests
/// see one block per straight-line run.
fn jit_single() -> Jit {
    let mut config = JitConfig {
        code_cache_bytes: 4 << 20,
        ..JitConfig::default()
    };
    config.translate.multiblock = false;
    Jit::new(config).unwrap()
}

/// movz x0,#0 ; loop: add x0,x0,#1 ; cmp x0,#N ; b.ne loop ; brk
fn counting_loop(n: u32) -> Code {
    Code(vec![
        movz(0, 0),
        add_imm(0, 0, 1),
        cmp_imm(0, n),
        b_ne(3, 1),
        BRK_0,
    ])
}

#[test]
fn chained_loop_runs_to_completion_with_exact_retired_count() {
    let mut mem = counting_loop(1000);
    let mut ctx = ctx();
    let mut jit = jit();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::Brk(0));
    assert_eq!(ctx.x[0], 1000);
    // Entry block: 4, loop block: 3 per iteration (999 more), brk: 1.
    assert_eq!(out.retired, 4 + 3 * 999 + 1);
    // The loop block branches to itself and is linked to itself.
    assert!(jit.cache.inbound_links(at(1)) >= 2, "loop not chained");
}

#[test]
fn chained_loop_honours_every_budget() {
    for budget in [1u64, 2, 3, 4, 5, 7, 100, 1001] {
        let mut mem = counting_loop(300);
        let mut ctx = ctx();
        let mut jit = jit();
        let mut total = 0u64;
        let mut slices = 0;
        loop {
            let out = jit.run_bounded(&mut ctx, &mut mem, budget).unwrap();
            assert!(out.retired <= budget, "budget {budget}: {}", out.retired);
            total += out.retired;
            slices += 1;
            assert!(slices < 100_000, "no progress at budget {budget}");
            match out.reason {
                ExitReason::BudgetExhausted => {
                    assert!(out.retired > 0, "slice made no progress");
                }
                ExitReason::Brk(0) => break,
                other => panic!("budget {budget}: {other:?}"),
            }
        }
        assert_eq!(ctx.x[0], 300, "budget {budget}");
        assert_eq!(total, 4 + 3 * 299 + 1, "budget {budget}");
    }
}

#[test]
fn slice_ends_early_instead_of_compiling_a_capped_block() {
    // 4-instruction block followed by a 3-instruction block.
    let mut mem = Code(vec![NOP, NOP, NOP, b(3, 4), NOP, NOP, BRK_0]);
    let mut ctx = ctx();
    let mut jit = jit_single();
    let out = jit.run_bounded(&mut ctx, &mut mem, 5).unwrap();
    assert_eq!(out.reason, ExitReason::BudgetExhausted);
    assert_eq!(out.retired, 4);
    assert_eq!(ctx.pc, at(4));
    let out = jit.run_bounded(&mut ctx, &mut mem, 5).unwrap();
    assert_eq!(out.reason, ExitReason::Brk(0));
    assert_eq!(out.retired, 3);
}

#[test]
fn repeated_single_steps_reuse_capped_blocks() {
    let mut mem = Code(vec![NOP, NOP, NOP, BRK_0]);
    let mut jit = jit();
    let mut ctx = ctx();
    let out = jit.run_bounded(&mut ctx, &mut mem, 1).unwrap();
    assert_eq!((out.reason, out.retired, ctx.pc), (ExitReason::BudgetExhausted, 1, at(1)));
    let used = jit.cache.used_bytes();
    for _ in 0..10 {
        ctx.pc = CODE;
        let out = jit.run_bounded(&mut ctx, &mut mem, 1).unwrap();
        assert_eq!((out.retired, ctx.pc), (1, at(1)));
    }
    assert_eq!(jit.cache.used_bytes(), used, "single steps leak cache bytes");
}

#[test]
fn halt_request_stops_a_chained_infinite_loop() {
    // loop: str x1,[x0] ; b loop
    let mut mem = Code(vec![0xF900_0001, b(1, 0)]);
    let mut ctx = ctx();
    HALT_AFTER.with(|c| c.set(5));
    let mut jit = jit();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::Stopped);
    assert_eq!(STORES.with(|c| c.get()), 5);
    assert_eq!(out.retired, 10);
    assert_eq!(ctx.pc, CODE, "resume PC must be the next block");
    assert_eq!(ctx.should_halt, 0);
}

#[test]
fn memory_fault_stops_a_chained_infinite_loop() {
    let mut mem = Code(vec![0xF900_0001, b(1, 0)]);
    let mut ctx = ctx();
    ctx.x[0] = 0xDEAD_0000;
    FAULT_AFTER.with(|c| c.set(3));
    let mut jit = jit();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::MemoryFault(0xDEAD_0000));
    assert_eq!(STORES.with(|c| c.get()), 3);
    assert_eq!(out.retired, 6);
}

#[test]
fn invalidation_unlinks_inbound_chains() {
    // A: movz x0,#1 ; b B        B: movz x1,#7 ; brk
    let mut mem = Code(vec![movz(0, 1), b(1, 2), movz(1, 7), BRK_0]);
    let mut jit = jit_single();
    let mut ctx = ctx();
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    assert_eq!(ctx.x[1], 7);
    // A was compiled before B existed; B's install linked A's exit.
    assert_eq!(jit.cache.inbound_links(at(2)), 1);

    mem.0[2] = movz(1, 9);
    jit.invalidate_range(at(2), 4);
    assert!(!jit.cache.contains(at(2)));
    assert_eq!(jit.cache.inbound_links(at(2)), 0);
    ctx.pc = CODE;
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    assert_eq!(ctx.x[1], 9, "stale chained block executed");
    assert_eq!(jit.cache.inbound_links(at(2)), 1, "relinked after recompile");

    // Invalidate the source instead; the target keeps running correctly
    // and the dead source's site is no longer registered.
    mem.0[0] = movz(0, 5);
    jit.invalidate_range(at(0), 4);
    assert_eq!(jit.cache.inbound_links(at(2)), 0);
    ctx.pc = CODE;
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    assert_eq!((ctx.x[0], ctx.x[1]), (5, 9));
    assert_eq!(jit.cache.inbound_links(at(2)), 1);
}

#[test]
fn invalidating_a_self_loop_stops_it_from_chaining_to_stale_code() {
    let mut mem = counting_loop(50);
    let mut jit = jit();
    let mut ctx = ctx();
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    assert_eq!(ctx.x[0], 50);
    // Loop body now adds 2: 25 iterations instead of 50.
    mem.0[1] = add_imm(0, 0, 2);
    jit.invalidate_range(at(1), 4);
    ctx.pc = CODE;
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::Brk(0));
    assert_eq!(ctx.x[0], 50);
    assert_eq!(out.retired, 4 + 3 * 24 + 1, "stale loop body executed");
}

#[test]
fn calls_and_returns_through_the_dispatch_table() {
    // 0: movz x2,#0
    // 1: loop: bl f
    // 2: add x2,x2,#1
    // 3: cmp x2,#200
    // 4: b.ne loop
    // 5: brk
    // 6: f: add x0,x0,#3
    // 7: ret
    let mut mem = Code(vec![
        movz(2, 0),
        bl(1, 6),
        add_imm(2, 2, 1),
        cmp_imm(2, 200),
        b_ne(4, 1),
        BRK_0,
        add_imm(0, 0, 3),
        RET,
    ]);
    let mut jit = jit();
    let mut ctx = ctx();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::Brk(0));
    assert_eq!((ctx.x[0], ctx.x[2]), (600, 200));
    assert_eq!(ctx.x[30], at(2));
    // movz+bl, then per iteration: f(2) + add/cmp/b.ne(3) + bl(1), minus
    // the final bl, plus brk.
    assert_eq!(out.retired, 2 + 200 * 5 + 199 + 1);

    // Patch f and invalidate it: the return path and the dispatch table
    // must not reach the old translation.
    mem.0[6] = add_imm(0, 0, 1);
    jit.invalidate_range(at(6), 4);
    ctx.pc = CODE;
    ctx.x[0] = 0;
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    assert_eq!(ctx.x[0], 200);
}

#[test]
fn budget_holds_across_indirect_branches() {
    let mut mem = Code(vec![
        movz(2, 0),
        bl(1, 6),
        add_imm(2, 2, 1),
        cmp_imm(2, 50),
        b_ne(4, 1),
        BRK_0,
        add_imm(0, 0, 3),
        RET,
    ]);
    let mut jit = jit();
    let mut ctx = ctx();
    let mut total = 0;
    loop {
        let out = jit.run_bounded(&mut ctx, &mut mem, 7).unwrap();
        assert!(out.retired <= 7 && out.retired > 0);
        total += out.retired;
        if out.reason == ExitReason::Brk(0) {
            break;
        }
        assert_eq!(out.reason, ExitReason::BudgetExhausted);
    }
    assert_eq!((ctx.x[0], ctx.x[2]), (150, 50));
    assert_eq!(total, 2 + 50 * 5 + 49 + 1);
}

#[test]
fn cache_reset_while_linked_keeps_running() {
    // A tiny cache forces repeated resets while blocks are linked.
    let mut words = Vec::new();
    for i in 0..200usize {
        words.push(add_imm(0, 0, 1));
        words.push(b(2 * i + 1, 2 * i + 2));
    }
    words.push(BRK_0);
    let mut mem = Code(words);
    let mut jit = Jit::new(JitConfig {
        code_cache_bytes: 64 * 1024,
        ..JitConfig::default()
    })
    .unwrap();
    for round in 0..3 {
        let mut ctx = ctx();
        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0), "round {round}");
        assert_eq!(ctx.x[0], 200);
    }
}

#[test]
fn multiblock_follows_branches_and_invalidates_every_range() {
    // 0: movz x0,#1 ; 1: b 4 ; 2,3: (never executed) ; 4: bl 8 ; 5: brk
    // 8: movz x1,#7 ; 9: ret
    let mut mem = Code(vec![
        movz(0, 1),
        b(1, 4),
        BRK_0,
        BRK_0,
        bl(4, 8),
        BRK_0,
        NOP,
        NOP,
        movz(1, 7),
        RET,
    ]);
    let mut jit = jit();
    let mut ctx = ctx();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::Brk(0));
    assert_eq!((ctx.x[0], ctx.x[1], ctx.x[30]), (1, 7, at(5)));
    assert_eq!(out.retired, 6);
    // One block covers 0..2, 4..5 and 8..10; nothing was compiled at 4 or 8.
    assert!(jit.cache.contains(CODE));
    assert!(!jit.cache.contains(at(4)) && !jit.cache.contains(at(8)));

    // A write to the third segment only must still drop the block.
    mem.0[8] = movz(1, 9);
    jit.invalidate_range(at(8), 4);
    assert!(!jit.cache.contains(CODE));
    ctx.pc = CODE;
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    assert_eq!(ctx.x[1], 9);
    // ... and the gap between segments is not covered.
    jit.invalidate_range(at(2), 8);
    assert!(jit.cache.contains(CODE));
}

#[test]
fn multiblock_matches_single_block_retirement_under_budgets() {
    let words = vec![
        movz(2, 0),
        bl(1, 6),
        add_imm(2, 2, 1),
        cmp_imm(2, 20),
        b_ne(4, 1),
        b(5, 9),
        add_imm(0, 0, 3),
        b(7, 8),
        RET,
        BRK_0,
    ];
    for budget in [1u64, 3, 6, 10, u64::MAX] {
        let mut results = Vec::new();
        for multi in [false, true] {
            let mut jit = if multi { jit() } else { jit_single() };
            let mut mem = Code(words.clone());
            let mut ctx = ctx();
            let mut total = 0;
            loop {
                let out = jit.run_bounded(&mut ctx, &mut mem, budget).unwrap();
                assert!(out.retired <= budget);
                total += out.retired;
                if out.reason == ExitReason::Brk(0) {
                    break;
                }
                assert_eq!(out.reason, ExitReason::BudgetExhausted);
            }
            results.push((total, ctx.x[0], ctx.x[2], ctx.pc));
        }
        assert_eq!(results[0], results[1], "budget {budget}");
        assert_eq!(results[0].1, 60);
    }
}

#[test]
fn multiblock_rolls_back_when_the_followed_target_cannot_be_translated() {
    // Unmapped branch target: the fault is reported at the target after the
    // instructions before the branch retired, as without multiblock.
    let mut mem = Code(vec![movz(0, 1), b(1, 4096)]);
    let mut jit = jit();
    let mut ctx = ctx();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.retired, 2);
    assert_eq!(ctx.x[0], 1);
    assert_eq!(out.reason, ExitReason::MemoryFault(at(4096)));

    // Undecodable target two instructions after the branch.
    let mut mem = Code(vec![movz(0, 1), b(1, 3), BRK_0, movz(1, 2), 0x0000_0000]);
    let mut jit2 = crate::jit();
    let mut ctx2 = crate::ctx();
    let out = jit2.run_bounded(&mut ctx2, &mut mem, u64::MAX).unwrap();
    // Same as single-block translation: the block at 3 fails as a whole,
    // so only the two instructions before the branch retire.
    assert_eq!(out.retired, 2);
    assert_eq!((ctx2.x[0], ctx2.x[1]), (1, 0));
    assert!(
        matches!(out.stop, rustarmic::StopReason::Unsupported(ref info) if info.pc == at(4)),
        "{:?}",
        out.stop
    );
}

#[test]
fn link_blocks_off_returns_to_the_dispatcher_with_same_results() {
    let mut config = JitConfig {
        code_cache_bytes: 4 << 20,
        ..JitConfig::default()
    };
    config.link_blocks = false;
    let mut jit = Jit::new(config).unwrap();
    let mut mem = counting_loop(100);
    let mut ctx = ctx();
    let out = jit.run_bounded(&mut ctx, &mut mem, u64::MAX).unwrap();
    assert_eq!(out.reason, ExitReason::Brk(0));
    assert_eq!(ctx.x[0], 100);
    assert_eq!(out.retired, 4 + 3 * 99 + 1);
    assert_eq!(jit.cache.inbound_links(at(1)), 0);
}
