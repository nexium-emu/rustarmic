//! Exclusives and LSE atomics must be atomic across host threads: several
//! threads, each running its own JIT instance over one shared guest memory,
//! hammer the same counters and the final values must be exact.  Exercised
//! with the inline fastmem path (page table and flat window), with
//! callbacks that perform the RMW on the host pointer, and with the default
//! lock-based `mem_atomic` fallback.

use rustarmic::jit::atomic::{context_host_atomic, default_mem_atomic};
use rustarmic::jit::context::{PAGE_DIRECT_READ, PAGE_DIRECT_WRITE};
use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

const CODE_BASE: u64 = 0x10_0000;
const DATA: u64 = 0x4000;
const ADDR_BITS: u8 = 16;
const THREADS: usize = 4;
const ITERS: u64 = 20_000;
const BRK_0: u32 = 0xD420_0000;

#[repr(C, align(4096))]
struct Arena([u8; 1 << ADDR_BITS]);

/// Base of the arena currently under test (tests in this file serialise on
/// `SERIAL`).
static BASE: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
static CALLBACK_ATOMICS: AtomicU64 = AtomicU64::new(0);

fn host(addr: u64) -> *mut u8 {
    unsafe { BASE.load(Ordering::Relaxed).add(addr as usize) }
}

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, addr: u64, size: u8) {
    let mut buf = [0u8; 16];
    unsafe {
        core::ptr::copy_nonoverlapping(host(addr), buf.as_mut_ptr(), size as usize);
        (*ctx).io_value = [
            u64::from_le_bytes(buf[..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..].try_into().unwrap()),
        ];
    }
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    unsafe {
        let io = (*ctx).io_value;
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&io[0].to_le_bytes());
        buf[8..].copy_from_slice(&io[1].to_le_bytes());
        core::ptr::copy_nonoverlapping(buf.as_ptr(), host(addr), size as usize);
    }
}

unsafe extern "C" fn hk_atomic(ctx: *mut CpuContext, addr: u64, size: u8, op: u8) {
    CALLBACK_ATOMICS.fetch_add(1, Ordering::Relaxed);
    unsafe { context_host_atomic(ctx, host(addr), size, op) };
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// Page-table fastmem with the data page directly writable.
    PageTable,
    /// Legacy flat fastmem window.
    Window,
    /// Callbacks only; `mem_atomic` does host atomics.
    HostCallback,
    /// Callbacks only; default `mem_atomic` (global lock over read/write).
    DefaultCallback,
}

struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.0.get((addr.checked_sub(CODE_BASE)? / 4) as usize).copied()
    }
}

/// Runs `code` on `THREADS` threads, each with x0 = DATA, x1 = ITERS and
/// x5 = 1, until BRK, over one shared arena.  Returns the arena.
fn run_threads(mode: Mode, code: &[u32]) -> Box<Arena> {
    let mut arena = Box::new(Arena([0; 1 << ADDR_BITS]));
    BASE.store(arena.0.as_mut_ptr(), Ordering::SeqCst);
    let mut table = vec![0u8; 1 << (ADDR_BITS - 12)];
    table.fill(PAGE_DIRECT_READ | PAGE_DIRECT_WRITE);
    let base = arena.0.as_mut_ptr() as usize;
    let table_ptr = table.as_ptr() as usize;
    std::thread::scope(|s| {
        for t in 0..THREADS {
            let code = code.to_vec();
            s.spawn(move || {
                let mut config = JitConfig {
                    code_cache_bytes: 1 << 20,
                    ..JitConfig::default()
                };
                match mode {
                    Mode::PageTable => config.fastmem_page_bits = Some(ADDR_BITS),
                    Mode::Window => config.use_fastmem = true,
                    _ => {}
                }
                // Vary block linking across threads too.
                config.link_blocks = t % 2 == 0;
                let mut jit = Jit::new(config).unwrap();
                let mut ctx = CpuContext::default();
                ctx.mem_read = hk_read;
                ctx.mem_write = hk_write;
                ctx.mem_atomic = match mode {
                    Mode::DefaultCallback => default_mem_atomic,
                    _ => hk_atomic,
                };
                ctx.mem_base = base as *mut u8;
                ctx.mem_base_va = 0;
                ctx.mem_size = 1 << ADDR_BITS;
                ctx.page_table = table_ptr as *const u8;
                ctx.pc = CODE_BASE;
                ctx.x[0] = DATA;
                ctx.x[1] = ITERS;
                ctx.x[5] = 1;
                let mut mem = Code(code);
                // Small slices so threads interleave at block boundaries too.
                loop {
                    let out = jit.run_bounded(&mut ctx, &mut mem, 1000 + t as u64).unwrap();
                    match out.reason {
                        ExitReason::BudgetExhausted => continue,
                        ExitReason::Brk(0) => break,
                        other => panic!("{mode:?}: unexpected exit {other:?}"),
                    }
                }
            });
        }
    });
    drop(table);
    arena
}

fn read(arena: &Arena, off: u64, bytes: usize) -> u128 {
    let mut buf = [0u8; 16];
    buf[..bytes].copy_from_slice(&arena.0[(DATA + off) as usize..][..bytes]);
    u128::from_le_bytes(buf)
}

const MODES: [Mode; 4] = [
    Mode::PageTable,
    Mode::Window,
    Mode::HostCallback,
    Mode::DefaultCallback,
];

const TOTAL: u64 = THREADS as u64 * ITERS;

/// ldaxr / add / stlxr / cbnz loop, `ld` and `st` being the LDAXR and STLXR
/// base encodings and `add` the add-immediate base (sf selects width).
fn exclusive_inc_loop(ld: u32, st: u32, add: u32) -> Vec<u32> {
    vec![
        ld | 2,                    // ldaxr{b,h} w2/x2, [x0]
        add | (1 << 10) | (2 << 5) | 2, // add w2/x2, w2/x2, #1
        st | (3 << 16) | 2,        // stlxr{b,h} w3, w2/x2, [x0]
        0x35FF_FFA3,               // cbnz w3, -12
        0xF100_0421,               // subs x1, x1, #1
        0x54FF_FF61,               // b.ne -20
        BRK_0,
    ]
}

#[test]
fn exclusive_increment_is_exact_all_widths() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // (ldaxr, stlxr, add base, bytes)
    let forms: [(u32, u32, u32, usize); 4] = [
        (0x085F_FC00, 0x0800_FC00, 0x1100_0000, 1),
        (0x485F_FC00, 0x4800_FC00, 0x1100_0000, 2),
        (0x885F_FC00, 0x8800_FC00, 0x1100_0000, 4),
        (0xC85F_FC00, 0xC800_FC00, 0x9100_0000, 8),
    ];
    for mode in MODES {
        for (ld, st, add, bytes) in forms {
            let arena = run_threads(mode, &exclusive_inc_loop(ld, st, add));
            let mask = if bytes == 16 { u128::MAX } else { (1u128 << (bytes * 8)) - 1 };
            assert_eq!(
                read(&arena, 0, bytes),
                u128::from(TOTAL) & mask,
                "{mode:?} {bytes}-byte exclusive increment lost updates"
            );
        }
    }
}

#[test]
fn exclusive_pair_increment_is_exact() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for mode in MODES {
        for wide in [false, true] {
            let (ldaxp, stlxp, add) = if wide {
                (0xC87F_8000u32, 0xC820_8000u32, 0x9100_0000u32)
            } else {
                (0x887F_8000, 0x8820_8000, 0x1100_0000)
            };
            let code = vec![
                ldaxp | (4 << 10) | 2,                 // ldaxp r2, r4, [x0]
                add | (1 << 10) | (2 << 5) | 2,        // add r2, r2, #1
                add | (2 << 10) | (4 << 5) | 4,        // add r4, r4, #2
                stlxp | (3 << 16) | (4 << 10) | 2,     // stlxp w3, r2, r4, [x0]
                0x35FF_FF83,                           // cbnz w3, -16
                0xF100_0421,                           // subs x1, x1, #1
                0x54FF_FF41,                           // b.ne -24
                BRK_0,
            ];
            let arena = run_threads(mode, &code);
            let half = if wide { 8 } else { 4 };
            assert_eq!(read(&arena, 0, half), u128::from(TOTAL), "{mode:?} wide={wide} lo");
            assert_eq!(
                read(&arena, half as u64, half),
                u128::from(2 * TOTAL),
                "{mode:?} wide={wide} hi"
            );
        }
    }
}

#[test]
fn lse_ldadd_and_cas_are_exact() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for mode in MODES {
        // ldaddal{b,h} w5/x5, w6/x6, [x0]; subs; b.ne; brk
        for (base, bytes) in [
            (0x38E0_0000u32, 1usize),
            (0x78E0_0000, 2),
            (0xB8E0_0000, 4),
            (0xF8E0_0000, 8),
        ] {
            let code = vec![base | (5 << 16) | 6, 0xF100_0421, 0x54FF_FFC1, BRK_0];
            let arena = run_threads(mode, &code);
            let mask = (1u128 << (bytes * 8)) - 1;
            assert_eq!(
                read(&arena, 0, bytes),
                u128::from(TOTAL) & mask,
                "{mode:?} ldadd {bytes} bytes"
            );
        }

        // CAS increment loop, 32 and 64 bit.
        for wide in [false, true] {
            let sf = if wide { 1u32 << 31 } else { 0 };
            let (ldr, cas) = if wide {
                (0xF940_0002u32, 0xC8E0_FC00u32)
            } else {
                (0xB940_0002, 0x88E0_FC00)
            };
            let code = vec![
                ldr,                                     // ldr r2, [x0]
                sf | 0x1100_0444,                        // add r4, r2, #1
                sf | 0x2A02_03E7,                        // mov r7, r2
                cas | (7 << 16) | 4,                     // casal r7, r4, [x0]
                sf | 0x6B02_00FF,                        // cmp r7, r2
                sf | 0x2A07_03E2,                        // mov r2, r7
                0x54FF_FF61,                             // b.ne retry (-20)
                0xF100_0421,                             // subs x1, x1, #1
                0x54FF_FF01,                             // b.ne loop (-32)
                BRK_0,
            ];
            let arena = run_threads(mode, &code);
            let bytes = if wide { 8 } else { 4 };
            assert_eq!(read(&arena, 0, bytes), u128::from(TOTAL), "{mode:?} cas wide={wide}");
        }

        // CASP x-pair increment of a 128-bit counter (lo += 1, hi += 3).
        let code = vec![
            0xA940_1002, // ldp x2, x4, [x0]
            0x9100_0448, // add x8, x2, #1
            0x9100_0C89, // add x9, x4, #3
            0xAA02_03E6, // mov x6, x2
            0xAA04_03E7, // mov x7, x4
            0x4866_FC08, // caspal x6, x7, x8, x9, [x0]
            0xEB0200DF, // cmp x6, x2
            0xFA44_00E0, // ccmp x7, x4, #0, eq
            0xAA06_03E2, // mov x2, x6
            0xAA07_03E4, // mov x4, x7
            0x54FF_FEE1, // b.ne retry (-36)
            0xF100_0421, // subs x1, x1, #1
            0x54FF_FE81, // b.ne loop (-48)
            BRK_0,
        ];
        let arena = run_threads(mode, &code);
        assert_eq!(read(&arena, 0, 8), u128::from(TOTAL), "{mode:?} casp lo");
        assert_eq!(read(&arena, 8, 8), u128::from(3 * TOTAL), "{mode:?} casp hi");
    }
}

/// LSE ops do not touch NZCV and produce the architectural old value and
/// memory result for every operation and width (single thread).
#[test]
fn lse_ops_values_and_flags() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // (opc field, model)
    let ops: [(u32, &str); 8] = [
        (0b000, "add"),
        (0b001, "clr"),
        (0b010, "eor"),
        (0b011, "set"),
        (0b100, "smax"),
        (0b101, "smin"),
        (0b110, "umax"),
        (0b111, "umin"),
    ];
    let vals: [u64; 6] = [0, 1, 0x7F, 0x80, 0xFFFF_FFFF_FFFF_FFFF, 0x8000_0000_8000_8080];
    for mode in [Mode::PageTable, Mode::HostCallback, Mode::DefaultCallback] {
        for size in 0..4u32 {
            let bytes = 1usize << size;
            let mask = if bytes == 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 };
            let sext = |v: u64| -> i64 {
                let sh = 64 - bytes * 8;
                ((v << sh) as i64) >> sh
            };
            for (opc, name) in ops.iter().chain([(0x100u32, "swp")].iter()) {
                for &m in &vals {
                    for &r in &vals {
                        let insn = if *name == "swp" {
                            (size << 30) | 0x38E0_8000 | (5 << 16) | 6
                        } else {
                            (size << 30) | 0x38E0_0000 | (opc << 12) | (5 << 16) | 6
                        };
                        let (old, new) = {
                            let (m, r) = (m & mask, r & mask);
                            let n = match *name {
                                "add" => m.wrapping_add(r),
                                "clr" => m & !r,
                                "eor" => m ^ r,
                                "set" => m | r,
                                "smax" => if sext(m) >= sext(r) { m } else { r },
                                "smin" => if sext(m) <= sext(r) { m } else { r },
                                "umax" => m.max(r),
                                "umin" => m.min(r),
                                _ => r,
                            };
                            (m, n & mask)
                        };
                        let mut arena = Box::new(Arena([0; 1 << ADDR_BITS]));
                        arena.0[DATA as usize..DATA as usize + 8].copy_from_slice(&m.to_le_bytes());
                        arena.0[DATA as usize + 8] = 0xEE;
                        BASE.store(arena.0.as_mut_ptr(), Ordering::SeqCst);
                        let table = vec![PAGE_DIRECT_READ | PAGE_DIRECT_WRITE; 16];
                        let mut config = JitConfig {
                            code_cache_bytes: 1 << 20,
                            ..JitConfig::default()
                        };
                        if let Mode::PageTable = mode {
                            config.fastmem_page_bits = Some(ADDR_BITS);
                        }
                        let mut jit = Jit::new(config).unwrap();
                        let mut ctx = CpuContext::default();
                        ctx.mem_read = hk_read;
                        ctx.mem_write = hk_write;
                        ctx.mem_atomic = if let Mode::DefaultCallback = mode {
                            default_mem_atomic
                        } else {
                            hk_atomic
                        };
                        ctx.mem_base = arena.0.as_mut_ptr();
                        ctx.page_table = table.as_ptr();
                        ctx.pc = CODE_BASE;
                        ctx.x[0] = DATA;
                        ctx.x[5] = r;
                        ctx.x[6] = 0xDEAD_BEEF_DEAD_BEEF;
                        ctx.nzcv = 0b1010;
                        let mut mem = Code(vec![insn, BRK_0]);
                        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
                        let got_mem = u64::from_le_bytes(
                            arena.0[DATA as usize..DATA as usize + 8].try_into().unwrap(),
                        );
                        let want_mem = (m & !mask) | new;
                        let ctxs = format!("{mode:?} {name} bytes={bytes} m={m:#x} r={r:#x}");
                        assert_eq!(got_mem, want_mem, "{ctxs}: memory");
                        assert_eq!(arena.0[DATA as usize + 8], 0xEE, "{ctxs}: overrun");
                        assert_eq!(ctx.x[6], old, "{ctxs}: old value");
                        assert_eq!(ctx.nzcv, 0b1010, "{ctxs}: NZCV must be preserved");
                    }
                }
            }
        }
    }
}

/// A failing store-exclusive must not write, and a successful one after an
/// intervening (non-exclusive) write of a *different* value must fail.
#[test]
fn store_exclusive_fails_when_memory_changed() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for mode in MODES {
        let mut arena = Box::new(Arena([0; 1 << ADDR_BITS]));
        arena.0[DATA as usize] = 5;
        BASE.store(arena.0.as_mut_ptr(), Ordering::SeqCst);
        let table = vec![PAGE_DIRECT_READ | PAGE_DIRECT_WRITE; 16];
        let mut config = JitConfig {
            code_cache_bytes: 1 << 20,
            ..JitConfig::default()
        };
        match mode {
            Mode::PageTable => config.fastmem_page_bits = Some(ADDR_BITS),
            Mode::Window => config.use_fastmem = true,
            _ => {}
        }
        let mut jit = Jit::new(config).unwrap();
        let mut ctx = CpuContext::default();
        ctx.mem_read = hk_read;
        ctx.mem_write = hk_write;
        ctx.mem_atomic = if let Mode::DefaultCallback = mode {
            default_mem_atomic
        } else {
            hk_atomic
        };
        ctx.mem_base = arena.0.as_mut_ptr();
        ctx.mem_size = 1 << ADDR_BITS;
        ctx.page_table = table.as_ptr();
        ctx.pc = CODE_BASE;
        ctx.x[0] = DATA;
        // ldaxr w2, [x0]; brk  -- then another "core" writes 9 --
        // stlxr w3, w4, [x0]; brk
        let mut mem = Code(vec![0x885F_FC02, BRK_0, 0x8803_FC04, BRK_0]);
        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
        assert_eq!(ctx.x[2], 5);
        arena.0[DATA as usize] = 9;
        ctx.x[4] = 77;
        ctx.pc = CODE_BASE + 8;
        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
        assert_eq!(ctx.x[3], 1, "{mode:?}: stlxr must fail");
        assert_eq!(arena.0[DATA as usize], 9, "{mode:?}: failed stlxr wrote");
        // Monitor is cleared: a second stlxr without ldaxr fails too.
        ctx.pc = CODE_BASE + 8;
        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
        assert_eq!(ctx.x[3], 1);
        // A fresh pair succeeds.
        ctx.pc = CODE_BASE;
        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
        ctx.pc = CODE_BASE + 8;
        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
        assert_eq!(ctx.x[3], 0, "{mode:?}");
        assert_eq!(arena.0[DATA as usize], 77, "{mode:?}");
    }
}

/// Callback-mode atomics really go through `mem_atomic`, and page-table
/// mode with the data page direct does not.
#[test]
fn atomic_routing() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let code = vec![0xB8E0_0000 | (5 << 16) | 6, 0xF100_0421, 0x54FF_FFC1, BRK_0];
    CALLBACK_ATOMICS.store(0, Ordering::SeqCst);
    let _ = run_threads(Mode::PageTable, &code);
    assert_eq!(CALLBACK_ATOMICS.load(Ordering::SeqCst), 0);
    let _ = run_threads(Mode::HostCallback, &code);
    assert_eq!(CALLBACK_ATOMICS.load(Ordering::SeqCst), TOTAL);
}
