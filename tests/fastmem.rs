//! Soft-fastmem regression — exercises both the in-range fast path (direct
//! `[mem_base + offset]` access, no fn-ptr) and the out-of-range slow path
//! (fall through to the existing fn-ptr handlers).

#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// Tests share the FALLBACK_READS/WRITES counters and the slow-path hooks,
// so cargo's default parallel execution would bleed counts across tests.
// Every test holds this mutex for its body to force serial execution.
static SERIALIZE: Mutex<()> = Mutex::new(());

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x10_0000;

fn build_code(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for w in words { v.extend_from_slice(&w.to_le_bytes()); }
    v
}

struct CodeMem { bytes: Vec<u8>, base: u64 }

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(self.base)? as usize;
        if off + 4 > self.bytes.len() { return None; }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[off..off + 4]);
        Some(u32::from_le_bytes(buf))
    }
}

// Counters to prove which path actually executed.
static FALLBACK_READS:  AtomicU64 = AtomicU64::new(0);
static FALLBACK_WRITES: AtomicU64 = AtomicU64::new(0);

// Slow-path handlers — they should be called ONLY when fastmem misses
// (mem_size = 0, out-of-range VA, or fastmem disabled).
unsafe extern "C" fn hk_read32(_: *mut CpuContext, _addr: u64) -> u32 {
    FALLBACK_READS.fetch_add(1, Ordering::Relaxed);
    0xFEEDFACE
}
unsafe extern "C" fn hk_read64(_: *mut CpuContext, _addr: u64) -> u64 {
    FALLBACK_READS.fetch_add(1, Ordering::Relaxed);
    0xCAFE_BABE_DEAD_BEEF
}
unsafe extern "C" fn hk_write32(_: *mut CpuContext, _addr: u64, _v: u32) {
    FALLBACK_WRITES.fetch_add(1, Ordering::Relaxed);
}
unsafe extern "C" fn hk_write64(_: *mut CpuContext, _addr: u64, _v: u64) {
    FALLBACK_WRITES.fetch_add(1, Ordering::Relaxed);
}

fn install_counting_hooks(ctx: &mut CpuContext) {
    ctx.mem_read32  = hk_read32;
    ctx.mem_read64  = hk_read64;
    ctx.mem_write32 = hk_write32;
    ctx.mem_write64 = hk_write64;
}

const BRK_0: u32 = 0xD420_0000;

fn run_with_cfg(code: Vec<u8>, ctx: &mut CpuContext, cfg: JitConfig) -> ExitReason {
    let mut mem = CodeMem { bytes: code, base: CODE_BASE };
    let mut jit = Jit::new(cfg).expect("jit init");
    jit.run(ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

#[test]
fn fastmem_disabled_routes_through_fn_ptrs() {
    // Sanity: with use_fastmem=false (default), every load goes through the
    // fn-ptr handler regardless of mem_base/mem_size setup.
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_READS.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];
    backing[0..8].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE;

    let code = build_code(&[
        0xF9400001, // ldr x1, [x0]
        BRK_0,
    ]);
    run_with_cfg(code, &mut ctx, JitConfig::default());
    assert_eq!(FALLBACK_READS.load(Ordering::Relaxed), 1,
               "fastmem disabled → slow path must run");
    assert_eq!(ctx.x[1], 0xCAFE_BABE_DEAD_BEEF, "X1 must hold fn-ptr result");
}

#[test]
fn fastmem_in_range_load_bypasses_fn_ptr() {
    // Backing memory holds a known qword; fastmem on; load from in-range VA.
    // Expected: zero fn-ptr calls; X1 = byte pattern from backing memory.
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_READS.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];
    backing[0x100..0x108].copy_from_slice(&0xAAAA_BBBB_CCCC_DDDDu64.to_le_bytes());

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x100;

    let code = build_code(&[
        0xF9400001, // ldr x1, [x0]
        BRK_0,
    ]);
    let cfg = JitConfig { use_fastmem: true, ..JitConfig::default() };
    run_with_cfg(code, &mut ctx, cfg);
    assert_eq!(ctx.x[1], 0xAAAA_BBBB_CCCC_DDDD, "fast path must load from backing");
    assert_eq!(FALLBACK_READS.load(Ordering::Relaxed), 0,
               "in-range fastmem must NOT invoke fn-ptr");
}

#[test]
fn fastmem_in_range_store_bypasses_fn_ptr() {
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_WRITES.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x200;
    ctx.x[1] = 0x1234_5678_9ABC_DEF0;

    let code = build_code(&[
        0xF9000001, // str x1, [x0]
        BRK_0,
    ]);
    let cfg = JitConfig { use_fastmem: true, ..JitConfig::default() };
    run_with_cfg(code, &mut ctx, cfg);
    assert_eq!(FALLBACK_WRITES.load(Ordering::Relaxed), 0,
               "in-range fastmem store must NOT invoke fn-ptr");
    // Re-read the backing buffer directly.
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&backing[0x200..0x208]);
    assert_eq!(u64::from_le_bytes(buf), 0x1234_5678_9ABC_DEF0,
               "fast path must have written directly to backing");
}

#[test]
fn fastmem_out_of_range_falls_through_to_fn_ptr() {
    // VA above mem_base_va + mem_size → bounds check fails → slow path.
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_READS.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;   // covers [DATA_BASE, DATA_BASE+0x1000)
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x2000;          // way past the range

    let code = build_code(&[
        0xF9400001, // ldr x1, [x0]
        BRK_0,
    ]);
    let cfg = JitConfig { use_fastmem: true, ..JitConfig::default() };
    run_with_cfg(code, &mut ctx, cfg);
    assert_eq!(FALLBACK_READS.load(Ordering::Relaxed), 1,
               "out-of-range access must fall to fn-ptr");
    assert_eq!(ctx.x[1], 0xCAFE_BABE_DEAD_BEEF, "X1 must hold slow-path result");
}

#[test]
fn fastmem_before_range_falls_through_to_fn_ptr() {
    // VA < mem_base_va → subtract wraps to huge unsigned → bounds check fails.
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_READS.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE - 0x10;            // before the region

    let code = build_code(&[
        0xF9400001, // ldr x1, [x0]
        BRK_0,
    ]);
    let cfg = JitConfig { use_fastmem: true, ..JitConfig::default() };
    run_with_cfg(code, &mut ctx, cfg);
    assert_eq!(FALLBACK_READS.load(Ordering::Relaxed), 1,
               "wrap-around access must fall to fn-ptr");
}

#[test]
fn fastmem_zero_size_disables_path() {
    // mem_size = 0 means every access falls to fn-ptr even with use_fastmem
    // on. Models the "fastmem enabled but no region declared yet" state.
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_READS.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = 0;                       // no region
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE;

    let code = build_code(&[
        0xF9400001, // ldr x1, [x0]
        BRK_0,
    ]);
    let cfg = JitConfig { use_fastmem: true, ..JitConfig::default() };
    run_with_cfg(code, &mut ctx, cfg);
    assert_eq!(FALLBACK_READS.load(Ordering::Relaxed), 1,
               "mem_size=0 must disable the fast path");
}

#[test]
fn fastmem_word_load_in_range() {
    // 32-bit load through the fast path.
    let _g = SERIALIZE.lock().unwrap();
    FALLBACK_READS.store(0, Ordering::Relaxed);
    let mut backing = vec![0u8; 0x1000];
    backing[0x40..0x44].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());

    let mut ctx = CpuContext::default();
    install_counting_hooks(&mut ctx);
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x40;

    let code = build_code(&[
        0xB9400001, // ldr w1, [x0]
        BRK_0,
    ]);
    let cfg = JitConfig { use_fastmem: true, ..JitConfig::default() };
    run_with_cfg(code, &mut ctx, cfg);
    assert_eq!(ctx.x[1], 0xDEAD_BEEF, "W1 must equal the 32-bit value");
    assert_eq!(FALLBACK_READS.load(Ordering::Relaxed), 0);
}
