//! Page-table fastmem: direct host access is only taken for pages whose
//! permission byte allows it; everything else must reach the callbacks.

use rustarmic::jit::context::{PAGE_DIRECT_READ, PAGE_DIRECT_WRITE};
use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::cell::Cell;

const CODE_BASE: u64 = 0x10_0000;
const BRK_0: u32 = 0xD420_0000;
const ADDR_BITS: u8 = 16; // 64 KiB guest window, 16 pages
const PAGES: usize = 1 << (ADDR_BITS - 12);

thread_local! {
    static READS: Cell<u64> = const { Cell::new(0) };
    static WRITES: Cell<u64> = const { Cell::new(0) };
    static LAST_ADDR: Cell<u64> = const { Cell::new(0) };
    static LAST_SIZE: Cell<u8> = const { Cell::new(0) };
    static LAST_WRITE: Cell<[u64; 2]> = const { Cell::new([0; 2]) };
}

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, addr: u64, size: u8) {
    READS.with(|c| c.set(c.get() + 1));
    LAST_ADDR.with(|c| c.set(addr));
    LAST_SIZE.with(|c| c.set(size));
    unsafe {
        (*ctx).io_value = [0x5105_5105_5105_5105, 0x6206_6206_6206_6206];
    }
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    WRITES.with(|c| c.set(c.get() + 1));
    LAST_ADDR.with(|c| c.set(addr));
    LAST_SIZE.with(|c| c.set(size));
    LAST_WRITE.with(|c| c.set(unsafe { (*ctx).io_value }));
}

struct CodeMem(Vec<u8>);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(CODE_BASE)? as usize;
        let word = self.0.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(word.try_into().ok()?))
    }
}

struct Env {
    arena: Vec<u8>,
    table: Vec<u8>,
    ctx: CpuContext,
}

fn env() -> Env {
    READS.with(|c| c.set(0));
    WRITES.with(|c| c.set(0));
    let mut arena = vec![0u8; 1 << ADDR_BITS];
    for (i, b) in arena.iter_mut().enumerate() {
        *b = (i as u8) ^ ((i >> 8) as u8).wrapping_mul(7);
    }
    let table = vec![0u8; PAGES];
    let mut ctx = CpuContext::default();
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
    ctx.pc = CODE_BASE;
    Env { arena, table, ctx }
}

fn run(env: &mut Env, words: &[u32]) -> ExitReason {
    env.ctx.mem_base = env.arena.as_mut_ptr();
    env.ctx.page_table = env.table.as_ptr();
    env.ctx.pc = CODE_BASE;
    let mut mem = CodeMem(words.iter().flat_map(|w| w.to_le_bytes()).collect());
    let mut jit = Jit::new(JitConfig {
        fastmem_page_bits: Some(ADDR_BITS),
        code_cache_bytes: 1 << 20,
        ..JitConfig::default()
    })
    .expect("jit init");
    jit.run(&mut env.ctx, &mut mem).expect("run")
}

fn reads() -> u64 {
    READS.with(|c| c.get())
}
fn writes() -> u64 {
    WRITES.with(|c| c.get())
}

const LDR_X1_X0: u32 = 0xF940_0001; // ldr x1, [x0]
const LDRB_W1_X0: u32 = 0x3940_0001; // ldrb w1, [x0]
const LDRH_W1_X0: u32 = 0x7940_0001; // ldrh w1, [x0]
const LDR_W1_X0: u32 = 0xB940_0001; // ldr w1, [x0]
const STR_X1_X0: u32 = 0xF900_0001; // str x1, [x0]
const STRB_W1_X0: u32 = 0x3900_0001; // strb w1, [x0]
const STR_W1_X0: u32 = 0xB900_0001; // str w1, [x0]
const LDR_Q1_X0: u32 = 0x3DC0_0001; // ldr q1, [x0]
const STR_Q1_X0: u32 = 0x3D80_0001; // str q1, [x0]
const LDUR_X1_X0: u32 = 0xF840_0001; // ldur x1, [x0]

fn expect_u64(arena: &[u8], addr: usize) -> u64 {
    u64::from_le_bytes(arena[addr..addr + 8].try_into().unwrap())
}

#[test]
fn readable_page_loads_inline_for_every_width() {
    let mut e = env();
    e.table[3] = PAGE_DIRECT_READ;
    for (insn, bytes) in [
        (LDRB_W1_X0, 1usize),
        (LDRH_W1_X0, 2),
        (LDR_W1_X0, 4),
        (LDR_X1_X0, 8),
    ] {
        e.ctx.x[0] = 0x3008;
        e.ctx.x[1] = !0;
        assert!(matches!(run(&mut e, &[insn, BRK_0]), ExitReason::Brk(_)));
        let mut want = [0u8; 8];
        want[..bytes].copy_from_slice(&e.arena[0x3008..0x3008 + bytes]);
        assert_eq!(e.ctx.x[1], u64::from_le_bytes(want), "width {bytes}");
    }
    assert_eq!(reads(), 0, "permitted loads must not call the hook");
}

#[test]
fn page_without_read_bit_calls_hook() {
    let mut e = env();
    e.table[3] = PAGE_DIRECT_WRITE; // write-only direct access
    e.ctx.x[0] = 0x3008;
    run(&mut e, &[LDR_X1_X0, BRK_0]);
    assert_eq!(reads(), 1);
    assert_eq!(e.ctx.x[1], 0x5105_5105_5105_5105);
    assert_eq!(LAST_ADDR.with(|c| c.get()), 0x3008);
    assert_eq!(LAST_SIZE.with(|c| c.get()), 8);
    // Ran through to the BRK, whose own address is the reported PC.
    assert_eq!(e.ctx.pc, CODE_BASE + 4);
}

#[test]
fn unmapped_page_inside_window_calls_hook_not_host_memory() {
    let mut e = env();
    e.table[2] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
    e.table[4] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
    e.ctx.x[0] = 0x3000; // page 3 has no bits
    e.ctx.x[1] = 0x77;
    let before = e.arena.clone();
    run(&mut e, &[STR_X1_X0, LDR_X1_X0, BRK_0]);
    assert_eq!((reads(), writes()), (1, 1));
    assert_eq!(e.arena, before, "store to an unpermitted page must not land");
}

#[test]
fn address_outside_window_calls_hook() {
    for addr in [1u64 << ADDR_BITS, (1u64 << ADDR_BITS) + 0x3008, 1 << 40, u64::MAX - 7, u64::MAX] {
        let mut e = env();
        e.table.fill(PAGE_DIRECT_READ | PAGE_DIRECT_WRITE);
        e.ctx.x[0] = addr;
        run(&mut e, &[LDRB_W1_X0, BRK_0]);
        assert_eq!(reads(), 1, "addr {addr:#x}");
        assert_eq!(LAST_ADDR.with(|c| c.get()), addr);
        let mut e = env();
        e.table.fill(PAGE_DIRECT_READ | PAGE_DIRECT_WRITE);
        e.ctx.x[0] = addr;
        let before = e.arena.clone();
        run(&mut e, &[STR_X1_X0, BRK_0]);
        assert_eq!(writes(), 1, "addr {addr:#x}");
        assert_eq!(e.arena, before);
    }
}

#[test]
fn page_straddling_access_calls_hook_even_when_both_pages_allow() {
    let mut e = env();
    e.table.fill(PAGE_DIRECT_READ | PAGE_DIRECT_WRITE);
    e.ctx.x[0] = 0x3FF9; // 8 bytes: 0x3ff9..0x4001
    run(&mut e, &[LDUR_X1_X0, BRK_0]);
    assert_eq!(reads(), 1);
    // Last address that still fits in the page is inline.
    let mut e = env();
    e.table.fill(PAGE_DIRECT_READ | PAGE_DIRECT_WRITE);
    e.ctx.x[0] = 0x3FF8;
    run(&mut e, &[LDUR_X1_X0, BRK_0]);
    assert_eq!(reads(), 0);
    assert_eq!(e.ctx.x[1], expect_u64(&e.arena, 0x3FF8));
    // Last page of the window, last byte.
    let mut e = env();
    e.table.fill(PAGE_DIRECT_READ | PAGE_DIRECT_WRITE);
    e.ctx.x[0] = 0xFFFF;
    run(&mut e, &[LDRB_W1_X0, BRK_0]);
    assert_eq!(reads(), 0);
    assert_eq!(e.ctx.x[1], e.arena[0xFFFF] as u64);
}

#[test]
fn writable_page_stores_inline_and_write_protected_page_calls_hook() {
    let mut e = env();
    e.table[5] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
    e.table[6] = PAGE_DIRECT_READ; // e.g. executable or watched page
    e.ctx.x[0] = 0x5010;
    e.ctx.x[1] = 0x1122_3344_5566_7788;
    run(&mut e, &[STR_X1_X0, BRK_0]);
    assert_eq!(writes(), 0);
    assert_eq!(expect_u64(&e.arena, 0x5010), 0x1122_3344_5566_7788);

    e.ctx.x[0] = 0x5021;
    run(&mut e, &[STRB_W1_X0, BRK_0]);
    assert_eq!(e.arena[0x5021], 0x88);
    assert_eq!(e.arena[0x5022], (0x22u8) ^ (0x50u8).wrapping_mul(7));
    let before_w = e.arena.clone();
    e.ctx.x[0] = 0x5030;
    run(&mut e, &[STR_W1_X0, BRK_0]);
    assert_eq!(&e.arena[0x5030..0x5034], &[0x88, 0x77, 0x66, 0x55]);
    assert_eq!(&e.arena[0x5034..0x5038], &before_w[0x5034..0x5038]);
    assert_eq!(writes(), 0);

    let before = e.arena.clone();
    e.ctx.x[0] = 0x6010;
    run(&mut e, &[STR_X1_X0, BRK_0]);
    assert_eq!(writes(), 1, "store to read-only-direct page must call hook");
    assert_eq!(LAST_ADDR.with(|c| c.get()), 0x6010);
    assert_eq!(LAST_WRITE.with(|c| c.get())[0], 0x1122_3344_5566_7788);
    assert_eq!(e.arena, before);
    // ... while loads from it stay inline.
    run(&mut e, &[LDR_X1_X0, BRK_0]);
    assert_eq!(reads(), 0);
}

#[test]
fn vector_128_load_and_store_are_single_accesses() {
    let mut e = env();
    e.table[7] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
    e.ctx.x[0] = 0x7100;
    run(&mut e, &[LDR_Q1_X0, BRK_0]);
    assert_eq!(e.ctx.v[1][0], expect_u64(&e.arena, 0x7100));
    assert_eq!(e.ctx.v[1][1], expect_u64(&e.arena, 0x7108));
    e.ctx.v[1] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
    e.ctx.x[0] = 0x7200;
    run(&mut e, &[STR_Q1_X0, BRK_0]);
    assert_eq!(expect_u64(&e.arena, 0x7200), 0xAAAA_BBBB_CCCC_DDDD);
    assert_eq!(expect_u64(&e.arena, 0x7208), 0x1111_2222_3333_4444);
    assert_eq!((reads(), writes()), (0, 0));

    // Unpermitted: exactly one callback each, carrying all 16 bytes.
    e.ctx.x[0] = 0x8100;
    run(&mut e, &[STR_Q1_X0, LDR_Q1_X0, BRK_0]);
    assert_eq!((reads(), writes()), (1, 1));
    assert_eq!(LAST_SIZE.with(|c| c.get()), 16);
    assert_eq!(
        LAST_WRITE.with(|c| c.get()),
        [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444]
    );
    assert_eq!(e.ctx.v[1], [0x5105_5105_5105_5105, 0x6206_6206_6206_6206]);
    // 16-byte access ending exactly at the page end is inline; one byte
    // further straddles.
    e.ctx.x[0] = 0x7FF0;
    run(&mut e, &[LDR_Q1_X0, BRK_0]);
    assert_eq!((reads(), writes()), (1, 1));
    e.table[8] = PAGE_DIRECT_READ;
    e.ctx.x[0] = 0x7FF1;
    run(&mut e, &[0x3CC0_0001 /* ldur q1,[x0] */, BRK_0]);
    assert_eq!(reads(), 2);
}

/// A direct-path access is the last op of a block that ends in a chain site
/// (unconditional branch, block-length cut, conditional branch).  The
/// access's fast/slow join label and the chain-site label used to land on
/// the same instruction, which the assembler rejects.
#[test]
fn access_as_last_op_before_chained_terminal() {
    const B_NEXT: u32 = 0x1400_0001; // b .+4
    const BEQ_NEXT: u32 = 0x5400_0020; // b.eq .+4
    for insn in [STR_X1_X0, STR_Q1_X0, STRB_W1_X0, LDR_X1_X0] {
        for term in [B_NEXT, BEQ_NEXT] {
            let mut e = env();
            e.table[3] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
            e.ctx.x[0] = 0x3008;
            e.ctx.x[1] = 0x1122_3344_5566_7788;
            assert!(
                matches!(run(&mut e, &[insn, term, BRK_0]), ExitReason::Brk(_)),
                "{insn:#x} then {term:#x}"
            );
        }
    }
    // Block cut by the instruction cap right after the access.
    let mut e = env();
    e.table[3] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
    e.ctx.x[0] = 0x3008;
    e.ctx.mem_base = e.arena.as_mut_ptr();
    e.ctx.page_table = e.table.as_ptr();
    e.ctx.pc = CODE_BASE;
    let words = [STR_X1_X0, STR_X1_X0, BRK_0];
    let mut mem = CodeMem(words.iter().flat_map(|w| w.to_le_bytes()).collect());
    let mut jit = Jit::new(JitConfig {
        fastmem_page_bits: Some(ADDR_BITS),
        code_cache_bytes: 1 << 20,
        ..JitConfig::default()
    })
    .unwrap();
    let out = jit.run_bounded(&mut e.ctx, &mut mem, 1).expect("capped block compiles");
    assert_eq!(out.retired, 1);
}
