use iced_x86::{Decoder, DecoderOptions, Instruction, Mnemonic, OpKind, Register};
use rustarmic::backend::emit_block;
use rustarmic::frontend::translator::translate_instruction;
use rustarmic::ir::{Block, IrEmitter, Terminal};
use rustarmic::jit::context::{PAGE_DIRECT_READ, PAGE_DIRECT_WRITE};
use rustarmic::jit::fastmem_direct::handled_faults;
use rustarmic::optimizer::optimize;
use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::cell::Cell;
use std::sync::{Mutex, MutexGuard};

const CODE_BASE: u64 = 0x10_0000;
const BRK_0: u32 = 0xD420_0000;
const BITS: u8 = 20;
const WINDOW: usize = 1 << BITS;
const PAGE: usize = 4096;

const RW_START: usize = 2 * PAGE;
const RW_LEN: usize = 2 * PAGE;
const RW_END: usize = RW_START + RW_LEN;
const RO_PAGE: usize = 5 * PAGE;
const HOLE: usize = 7 * PAGE;

const LDR_X1_X0: u32 = 0xF940_0001;
const LDRB_W1_X0: u32 = 0x3940_0001;
const LDRH_W1_X0: u32 = 0x7940_0001;
const LDR_W1_X0: u32 = 0xB940_0001;
const STR_X1_X0: u32 = 0xF900_0001;
const STRB_W1_X0: u32 = 0x3900_0001;
const STRH_W1_X0: u32 = 0x7900_0001;
const STR_W1_X0: u32 = 0xB900_0001;
const LDR_Q1_X0: u32 = 0x3DC0_0001;
const STR_Q1_X0: u32 = 0x3D80_0001;
const LDUR_X1_X0: u32 = 0xF840_0001;
const STUR_X1_X0: u32 = 0xF800_0001;
const LDUR_Q1_X0: u32 = 0x3CC0_0001;
const LDR_X2_X0: u32 = 0xF940_0002;
const LDR_X3_X4: u32 = 0xF940_0083;
const LDR_X2_X3: u32 = 0xF940_0062;
const SUB_X5_X5_1: u32 = 0xD100_04A5;
const ADD_X2_X2_1: u32 = 0x9100_0442;
const LDXR_X2_X0: u32 = 0xC85F_7C02;
const STXR_W3_X2_X0: u32 = 0xC803_7C02;
const LDADDAL_X1_X2_X0: u32 = 0xF8E1_0002;
const LDCLRAL_X1_X2_X0: u32 = 0xF8E1_1002;
const CASAL_X1_X2_X0: u32 = 0xC8E1_FC02;

const READ_PATTERN: [u64; 2] = [0x5105_5105_5105_5105, 0x6206_6206_6206_6206];

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

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
        (*ctx).io_value = READ_PATTERN;
    }
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    WRITES.with(|c| c.set(c.get() + 1));
    LAST_ADDR.with(|c| c.set(addr));
    LAST_SIZE.with(|c| c.set(size));
    LAST_WRITE.with(|c| c.set(unsafe { (*ctx).io_value }));
}

fn reads() -> u64 {
    READS.with(|c| c.get())
}
fn writes() -> u64 {
    WRITES.with(|c| c.get())
}
fn last_addr() -> u64 {
    LAST_ADDR.with(|c| c.get())
}
fn last_size() -> u8 {
    LAST_SIZE.with(|c| c.get())
}
fn last_write() -> [u64; 2] {
    LAST_WRITE.with(|c| c.get())
}
fn reset_hooks() {
    READS.with(|c| c.set(0));
    WRITES.with(|c| c.set(0));
    LAST_ADDR.with(|c| c.set(0));
    LAST_SIZE.with(|c| c.set(0));
    LAST_WRITE.with(|c| c.set([0; 2]));
}

fn pattern(i: usize) -> u8 {
    (i as u8) ^ ((i >> 8) as u8).wrapping_mul(7)
}

#[cfg(windows)]
mod os {
    use core::ffi::c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READONLY: u32 = 0x02;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, kind: u32, protect: u32) -> *mut c_void;
        fn VirtualProtect(addr: *mut c_void, size: usize, protect: u32, old: *mut u32) -> i32;
        fn VirtualFree(addr: *mut c_void, size: usize, kind: u32) -> i32;
    }

    pub fn reserve(len: usize) -> *mut u8 {
        let p = unsafe { VirtualAlloc(core::ptr::null_mut(), len, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!p.is_null(), "VirtualAlloc reserve failed");
        p.cast()
    }

    pub fn commit(ptr: *mut u8, len: usize) {
        let p = unsafe { VirtualAlloc(ptr.cast(), len, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!p.is_null(), "VirtualAlloc commit failed");
    }

    pub fn protect_read_only(ptr: *mut u8, len: usize) {
        let mut old = 0u32;
        let ok = unsafe { VirtualProtect(ptr.cast(), len, PAGE_READONLY, &mut old) };
        assert_ne!(ok, 0, "VirtualProtect failed");
    }

    pub fn release(ptr: *mut u8, _len: usize) {
        unsafe { VirtualFree(ptr.cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(unix)]
mod os {
    use core::ffi::c_void;

    const PROT_NONE: i32 = 0;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    #[cfg(target_os = "linux")]
    const MAP_ANONYMOUS: i32 = 0x20;
    #[cfg(target_os = "linux")]
    const MAP_NORESERVE: i32 = 0x4000;
    #[cfg(not(target_os = "linux"))]
    const MAP_ANONYMOUS: i32 = 0x1000;
    #[cfg(not(target_os = "linux"))]
    const MAP_NORESERVE: i32 = 0;

    unsafe extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }

    pub fn reserve(len: usize) -> *mut u8 {
        let p = unsafe {
            mmap(
                core::ptr::null_mut(),
                len,
                PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(p as isize != -1 && !p.is_null(), "mmap failed");
        p.cast()
    }

    pub fn commit(ptr: *mut u8, len: usize) {
        assert_eq!(unsafe { mprotect(ptr.cast(), len, PROT_READ | PROT_WRITE) }, 0);
    }

    pub fn protect_read_only(ptr: *mut u8, len: usize) {
        assert_eq!(unsafe { mprotect(ptr.cast(), len, PROT_READ) }, 0);
    }

    pub fn release(ptr: *mut u8, len: usize) {
        unsafe { munmap(ptr.cast(), len) };
    }
}

struct Arena {
    base: *mut u8,
}

impl Arena {
    fn new() -> Self {
        let base = os::reserve(WINDOW);
        os::commit(unsafe { base.add(RW_START) }, RW_LEN);
        os::commit(unsafe { base.add(RO_PAGE) }, PAGE);
        for i in (RW_START..RW_END).chain(RO_PAGE..RO_PAGE + PAGE) {
            unsafe { base.add(i).write(pattern(i)) };
        }
        os::protect_read_only(unsafe { base.add(RO_PAGE) }, PAGE);
        Self { base }
    }

    fn bytes(&self, offset: usize, len: usize) -> Vec<u8> {
        (0..len).map(|i| unsafe { self.base.add(offset + i).read() }).collect()
    }

    fn u64_at(&self, offset: usize) -> u64 {
        u64::from_le_bytes(self.bytes(offset, 8).try_into().unwrap())
    }

    fn ro_page_is_pristine(&self) -> bool {
        (RO_PAGE..RO_PAGE + PAGE).all(|i| unsafe { self.base.add(i).read() } == pattern(i))
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        os::release(self.base, WINDOW);
    }
}

struct CodeMem(Vec<u8>);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = (addr.checked_sub(CODE_BASE)? as usize) % 0x100;
        let word = self.0.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(word.try_into().ok()?))
    }
}

fn code(words: &[u32]) -> CodeMem {
    CodeMem(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

struct Env {
    arena: Arena,
    ctx: CpuContext,
}

fn env() -> Env {
    reset_hooks();
    let arena = Arena::new();
    let mut ctx = CpuContext::default();
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
    ctx.mem_base = arena.base;
    ctx.pc = CODE_BASE;
    Env { arena, ctx }
}

fn config(rwx: bool) -> JitConfig {
    JitConfig {
        fastmem_direct_bits: Some(BITS),
        rwx_code_cache: rwx,
        code_cache_bytes: 1 << 20,
        ..JitConfig::default()
    }
}

fn jit(rwx: bool) -> Jit {
    Jit::new(config(rwx)).expect("jit init")
}

fn run_at(jit: &mut Jit, env: &mut Env, pc: u64, words: &[u32]) -> ExitReason {
    env.ctx.pc = pc;
    let mut mem = code(words);
    jit.run(&mut env.ctx, &mut mem).expect("run")
}

fn run(jit: &mut Jit, env: &mut Env, words: &[u32]) -> ExitReason {
    run_at(jit, env, CODE_BASE, words)
}

fn run_fresh(jit: &mut Jit, env: &mut Env, words: &[u32]) -> ExitReason {
    jit.invalidate_range(CODE_BASE, 0x100);
    run_at(jit, env, CODE_BASE, words)
}

#[derive(Debug, PartialEq, Eq)]
struct Observed {
    exit: ExitReason,
    reads: u64,
    writes: u64,
    last_addr: u64,
    last_size: u8,
    last_write: [u64; 2],
    x: [u64; 6],
    pc: u64,
}

fn observe(exit: ExitReason, ctx: &CpuContext) -> Observed {
    Observed {
        exit,
        reads: reads(),
        writes: writes(),
        last_addr: last_addr(),
        last_size: last_size(),
        last_write: last_write(),
        x: ctx.x[..6].try_into().unwrap(),
        pc: ctx.pc,
    }
}

fn run_page_table(words: &[u32], regs: &[(usize, u64)]) -> Observed {
    reset_hooks();
    let mut arena = vec![0u8; WINDOW];
    for (i, b) in arena.iter_mut().enumerate() {
        *b = pattern(i);
    }
    let mut table = vec![0u8; WINDOW / PAGE];
    for page in RW_START / PAGE..RW_END / PAGE {
        table[page] = PAGE_DIRECT_READ | PAGE_DIRECT_WRITE;
    }
    table[RO_PAGE / PAGE] = PAGE_DIRECT_READ;
    let mut ctx = CpuContext::default();
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
    ctx.mem_base = arena.as_mut_ptr();
    ctx.page_table = table.as_ptr();
    ctx.pc = CODE_BASE;
    for &(reg, value) in regs {
        ctx.x[reg] = value;
    }
    let mut jit = Jit::new(JitConfig {
        fastmem_page_bits: Some(BITS),
        code_cache_bytes: 1 << 20,
        ..JitConfig::default()
    })
    .expect("jit init");
    let exit = jit.run(&mut ctx, &mut code(words)).expect("run");
    observe(exit, &ctx)
}

#[test]
fn rejects_out_of_range_direct_bits() {
    for bits in [0u8, 64, 200] {
        assert!(
            Jit::new(JitConfig {
                fastmem_direct_bits: Some(bits),
                code_cache_bytes: 1 << 20,
                ..JitConfig::default()
            })
            .is_err(),
            "bits {bits}"
        );
    }
}

#[test]
fn direct_loads_and_stores_of_every_width_hit_the_arena() {
    let _serial = serial();
    for rwx in [false, true] {
        let mut e = env();
        let mut j = jit(rwx);
        let before = handled_faults();
        for (insn, bytes) in [
            (LDRB_W1_X0, 1usize),
            (LDRH_W1_X0, 2),
            (LDR_W1_X0, 4),
            (LDR_X1_X0, 8),
        ] {
            e.ctx.x[0] = (RW_START + 0x108) as u64;
            e.ctx.x[1] = !0;
            assert!(matches!(run_fresh(&mut j, &mut e, &[insn, BRK_0]), ExitReason::Brk(_)));
            let mut want = [0u8; 8];
            want[..bytes].copy_from_slice(&e.arena.bytes(RW_START + 0x108, bytes));
            assert_eq!(e.ctx.x[1], u64::from_le_bytes(want), "load width {bytes}");
        }
        e.ctx.x[0] = (RW_START + 0x200) as u64;
        run_fresh(&mut j, &mut e, &[LDR_Q1_X0, BRK_0]);
        assert_eq!(e.ctx.v[1][0], e.arena.u64_at(RW_START + 0x200));
        assert_eq!(e.ctx.v[1][1], e.arena.u64_at(RW_START + 0x208));

        e.ctx.x[1] = 0x1122_3344_5566_7788;
        for (insn, bytes) in [
            (STRB_W1_X0, 1usize),
            (STRH_W1_X0, 2),
            (STR_W1_X0, 4),
            (STR_X1_X0, 8),
        ] {
            let at = RW_START + 0x1010 + bytes * 16;
            e.ctx.x[0] = at as u64;
            let untouched = e.arena.bytes(at + bytes, 8);
            assert!(matches!(run_fresh(&mut j, &mut e, &[insn, BRK_0]), ExitReason::Brk(_)));
            assert_eq!(
                &e.arena.bytes(at, bytes)[..],
                &0x1122_3344_5566_7788u64.to_le_bytes()[..bytes],
                "store width {bytes}"
            );
            assert_eq!(e.arena.bytes(at + bytes, 8), untouched, "store width {bytes} spilled");
        }
        e.ctx.v[1] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
        e.ctx.x[0] = (RW_START + 0x1800) as u64;
        run_fresh(&mut j, &mut e, &[STR_Q1_X0, BRK_0]);
        assert_eq!(e.arena.u64_at(RW_START + 0x1800), 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(e.arena.u64_at(RW_START + 0x1808), 0x1111_2222_3333_4444);

        assert_eq!((reads(), writes()), (0, 0), "rwx {rwx}: permitted accesses must stay inline");
        assert_eq!(handled_faults(), before, "rwx {rwx}: no host fault expected");
        assert_eq!(j.direct_sites_patched(), 0);
        assert!(j.cache.direct_site_count() >= 10);
    }
}

#[test]
fn hole_access_reaches_callbacks_once_and_matches_page_table_mode() {
    let _serial = serial();
    let words = [STR_X1_X0, LDR_X2_X0, LDR_X3_X4, BRK_0];
    let regs = [
        (0usize, (HOLE + 0x18) as u64),
        (1, 0x0F0E_0D0C_0B0A_0908),
        (4, (RW_START + 0x40) as u64),
    ];
    for rwx in [false, true] {
        let mut e = env();
        for &(reg, value) in &regs {
            e.ctx.x[reg] = value;
        }
        let mut j = jit(rwx);
        let before = handled_faults();
        let exit = run(&mut j, &mut e, &words);
        let direct = observe(exit, &e.ctx);
        assert_eq!(direct.exit, ExitReason::Brk(0));
        assert_eq!((direct.reads, direct.writes), (1, 1), "rwx {rwx}");
        assert_eq!(direct.x[2], READ_PATTERN[0]);
        assert_eq!(direct.x[3], e.arena.u64_at(RW_START + 0x40));
        assert_eq!(direct.pc, CODE_BASE + 12, "block must run on to the BRK");
        assert_eq!(handled_faults() - before, 2, "rwx {rwx}: one fault per access site");
        assert_eq!(j.direct_sites_patched(), 2);

        let table = run_page_table(&words, &regs);
        assert_eq!(direct, table, "rwx {rwx}: direct and page-table modes must agree");

        reset_hooks();
        for _ in 0..3 {
            run(&mut j, &mut e, &words);
        }
        assert_eq!((reads(), writes()), (3, 3));
        assert_eq!(handled_faults() - before, 2, "patched sites must not fault again");
    }
}

#[test]
fn store_to_read_only_page_calls_hook_and_faults_once() {
    let _serial = serial();
    for rwx in [false, true] {
        let mut e = env();
        let mut j = jit(rwx);
        let before = handled_faults();
        e.ctx.x[0] = (RO_PAGE + 0x10) as u64;
        e.ctx.x[1] = 0x1122_3344_5566_7788;
        assert!(matches!(run(&mut j, &mut e, &[STR_X1_X0, BRK_0]), ExitReason::Brk(_)));
        assert_eq!(writes(), 1, "rwx {rwx}");
        assert_eq!(last_addr(), (RO_PAGE + 0x10) as u64);
        assert_eq!(last_size(), 8);
        assert_eq!(last_write()[0], 0x1122_3344_5566_7788);
        assert!(e.arena.ro_page_is_pristine());
        assert_eq!(handled_faults() - before, 1, "rwx {rwx}");
        assert_eq!(j.direct_sites_patched(), 1);
        for _ in 0..8 {
            run(&mut j, &mut e, &[STR_X1_X0, BRK_0]);
        }
        assert_eq!(writes(), 9);
        assert_eq!(handled_faults() - before, 1, "rwx {rwx}: patched site faulted again");
        assert_eq!(j.direct_sites_patched(), 1);
        assert!(e.arena.ro_page_is_pristine());

        e.ctx.x[2] = 0;
        run_fresh(&mut j, &mut e, &[LDR_X2_X0, BRK_0]);
        assert_eq!(reads(), 0, "loads from the read-only page stay inline");
        assert_eq!(e.ctx.x[2], e.arena.u64_at(RO_PAGE + 0x10));
        assert_eq!(handled_faults() - before, 1);
    }
}

#[test]
fn address_at_or_above_window_calls_hook_without_faulting() {
    let _serial = serial();
    let mut e = env();
    let mut j = jit(true);
    let before = handled_faults();
    let mut expected = 0;
    for addr in [
        1u64 << BITS,
        (1u64 << BITS) + RW_START as u64 + 8,
        1 << 40,
        u64::MAX - 7,
        u64::MAX,
    ] {
        e.ctx.x[0] = addr;
        e.ctx.x[1] = 0x55;
        run_fresh(&mut j, &mut e, &[LDRB_W1_X0, BRK_0]);
        expected += 1;
        assert_eq!(reads(), expected, "addr {addr:#x}");
        assert_eq!(last_addr(), addr);
        assert_eq!(e.ctx.x[1], READ_PATTERN[0] & 0xFF);
        run_fresh(&mut j, &mut e, &[STR_X1_X0, BRK_0]);
        assert_eq!(writes(), expected, "addr {addr:#x}");
        assert_eq!(last_addr(), addr);
    }
    assert_eq!(handled_faults(), before, "guarded addresses must not reach the host");
    assert_eq!(j.direct_sites_patched(), 0);
}

#[test]
fn access_straddling_committed_and_uncommitted_page_calls_hook() {
    let _serial = serial();
    for rwx in [false, true] {
        let mut e = env();
        let mut j = jit(rwx);
        let before = handled_faults();
        e.ctx.x[0] = (RW_END - 4) as u64;
        e.ctx.x[1] = 0x0102_0304_0506_0708;
        run(&mut j, &mut e, &[LDUR_X1_X0, BRK_0]);
        assert_eq!((reads(), last_addr(), last_size()), (1, (RW_END - 4) as u64, 8), "rwx {rwx}");
        assert_eq!(e.ctx.x[1], READ_PATTERN[0]);
        e.ctx.x[1] = 0x0102_0304_0506_0708;
        run_fresh(&mut j, &mut e, &[STUR_X1_X0, BRK_0]);
        assert_eq!((writes(), last_addr(), last_size()), (1, (RW_END - 4) as u64, 8));
        assert_eq!(last_write()[0], 0x0102_0304_0506_0708);
        e.ctx.x[0] = (RW_END - 8) as u64;
        run_fresh(&mut j, &mut e, &[LDUR_Q1_X0, BRK_0]);
        assert_eq!((reads(), last_size()), (2, 16));
        assert_eq!(e.ctx.v[1], READ_PATTERN);
        assert_eq!(handled_faults() - before, 3, "rwx {rwx}");
        assert_eq!(j.direct_sites_patched(), 3);

        e.ctx.x[0] = (RW_END - 8) as u64;
        run_fresh(&mut j, &mut e, &[LDUR_X1_X0, BRK_0]);
        assert_eq!(reads(), 2, "an access ending exactly at the committed end is inline");
        assert_eq!(e.ctx.x[1], e.arena.u64_at(RW_END - 8));
    }
}

#[test]
fn vector_128_store_and_load_to_unmapped_page_are_single_callbacks() {
    let _serial = serial();
    let mut e = env();
    let mut j = jit(true);
    let before = handled_faults();
    e.ctx.v[1] = [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444];
    for target in [HOLE + 0x100, RO_PAGE + 0x100] {
        e.ctx.x[0] = target as u64;
        run(&mut j, &mut e, &[STR_Q1_X0, BRK_0]);
        assert_eq!(last_size(), 16);
        assert_eq!(last_addr(), target as u64);
        assert_eq!(last_write(), [0xAAAA_BBBB_CCCC_DDDD, 0x1111_2222_3333_4444]);
    }
    assert_eq!(writes(), 2);
    assert!(e.arena.ro_page_is_pristine());
    e.ctx.x[0] = (HOLE + 0x100) as u64;
    run_fresh(&mut j, &mut e, &[LDR_Q1_X0, BRK_0]);
    assert_eq!((reads(), last_size()), (1, 16));
    assert_eq!(e.ctx.v[1], READ_PATTERN);
    assert_eq!(handled_faults() - before, 2, "second str q1 must use the patched slow path");
    assert_eq!(j.direct_sites_patched(), 2);
}

#[test]
fn exclusive_and_lse_atomics_run_inline_or_fall_back_to_the_atomic_callback() {
    let _serial = serial();
    for rwx in [false, true] {
        let mut e = env();
        let mut j = jit(rwx);
        let before = handled_faults();

        let counter = RW_START + 0x400;
        let initial = e.arena.u64_at(counter);
        e.ctx.x[0] = counter as u64;
        run(&mut j, &mut e, &[LDXR_X2_X0, ADD_X2_X2_1, STXR_W3_X2_X0, BRK_0]);
        assert_eq!(e.ctx.x[3], 0, "store-exclusive must succeed");
        assert_eq!(e.arena.u64_at(counter), initial.wrapping_add(1));
        e.ctx.x[1] = 7;
        run_fresh(&mut j, &mut e, &[LDADDAL_X1_X2_X0, BRK_0]);
        assert_eq!(e.ctx.x[2], initial.wrapping_add(1));
        assert_eq!(e.arena.u64_at(counter), initial.wrapping_add(8));
        e.ctx.x[1] = 0xFF;
        run_fresh(&mut j, &mut e, &[LDCLRAL_X1_X2_X0, BRK_0]);
        assert_eq!(e.ctx.x[2], initial.wrapping_add(8));
        assert_eq!(e.arena.u64_at(counter), initial.wrapping_add(8) & !0xFF);
        e.ctx.x[1] = initial.wrapping_add(8) & !0xFF;
        e.ctx.x[2] = 0x4242;
        run_fresh(&mut j, &mut e, &[CASAL_X1_X2_X0, BRK_0]);
        assert_eq!(e.ctx.x[1], initial.wrapping_add(8) & !0xFF);
        assert_eq!(e.arena.u64_at(counter), 0x4242);
        assert_eq!((reads(), writes()), (0, 0), "rwx {rwx}: writable page atomics stay inline");
        assert_eq!(handled_faults(), before);

        e.ctx.x[0] = (RO_PAGE + 0x20) as u64;
        run_fresh(&mut j, &mut e, &[LDXR_X2_X0, STXR_W3_X2_X0, BRK_0]);
        assert_eq!(e.ctx.x[2], e.arena.u64_at(RO_PAGE + 0x20));
        assert_eq!((reads(), writes()), (1, 0), "rwx {rwx}: stxr on read-only memory");
        assert_eq!(e.ctx.x[3], 1, "the callback saw a different value, so the exclusive fails");
        assert_eq!(handled_faults() - before, 1);

        e.ctx.x[1] = 0x10;
        run_fresh(&mut j, &mut e, &[LDADDAL_X1_X2_X0, BRK_0]);
        assert_eq!((reads(), writes()), (2, 1));
        assert_eq!(last_addr(), (RO_PAGE + 0x20) as u64);
        assert_eq!(last_size(), 8);
        assert_eq!(e.ctx.x[2], READ_PATTERN[0]);
        assert_eq!(last_write()[0], READ_PATTERN[0] + 0x10);
        assert_eq!(handled_faults() - before, 2);

        e.ctx.x[1] = 0x05;
        run_fresh(&mut j, &mut e, &[LDCLRAL_X1_X2_X0, BRK_0]);
        assert_eq!((reads(), writes()), (3, 2));
        assert_eq!(e.ctx.x[2], READ_PATTERN[0]);
        assert_eq!(last_write()[0], READ_PATTERN[0] & !0x05);
        assert_eq!(handled_faults() - before, 3);

        e.ctx.x[1] = READ_PATTERN[0];
        e.ctx.x[2] = 0xDEAD;
        run_fresh(&mut j, &mut e, &[CASAL_X1_X2_X0, BRK_0]);
        assert_eq!((reads(), writes()), (4, 3));
        assert_eq!(e.ctx.x[1], READ_PATTERN[0]);
        assert_eq!(last_write()[0], 0xDEAD);
        assert_eq!(handled_faults() - before, 4);

        e.ctx.x[0] = (HOLE + 0x20) as u64;
        e.ctx.x[1] = 1;
        run_fresh(&mut j, &mut e, &[LDADDAL_X1_X2_X0, BRK_0]);
        assert_eq!((reads(), writes()), (5, 4));
        assert_eq!(last_addr(), (HOLE + 0x20) as u64);
        assert_eq!(handled_faults() - before, 5);
        assert_eq!(j.direct_sites_patched(), 5);
        assert!(e.arena.ro_page_is_pristine());

        reset_hooks();
        e.ctx.x[0] = (RO_PAGE + 0x20) as u64;
        for _ in 0..4 {
            run(&mut j, &mut e, &[LDADDAL_X1_X2_X0, BRK_0]);
        }
        assert_eq!((reads(), writes()), (4, 4));
        assert_eq!(handled_faults() - before, 5, "rwx {rwx}: patched atomic site faulted again");
    }
}

#[test]
fn linked_blocks_patch_their_own_sites() {
    let _serial = serial();
    const B_TO_3: u32 = 0x1400_0002;
    const CBNZ_X5_BACK_5: u32 = 0xB5FF_FF65;
    let words = [
        STR_X1_X0,
        B_TO_3,
        BRK_0,
        LDR_X2_X3,
        SUB_X5_X5_1,
        CBNZ_X5_BACK_5,
        BRK_0,
    ];
    for rwx in [false, true] {
        let mut e = env();
        let mut cfg = config(rwx);
        cfg.translate.multiblock = false;
        cfg.link_blocks = true;
        let mut j = Jit::new(cfg).unwrap();
        let before = handled_faults();
        e.ctx.x[0] = (RO_PAGE + 0x30) as u64;
        e.ctx.x[1] = 0x7777_6666_5555_4444;
        e.ctx.x[3] = (HOLE + 0x30) as u64;
        e.ctx.x[5] = 6;
        assert_eq!(run(&mut j, &mut e, &words), ExitReason::Brk(0));
        assert_eq!(e.ctx.pc, CODE_BASE + 24);
        assert_eq!((writes(), reads()), (6, 6), "rwx {rwx}");
        assert_eq!(e.ctx.x[2], READ_PATTERN[0]);
        assert_eq!(e.ctx.x[5], 0);
        assert_eq!(handled_faults() - before, 2, "rwx {rwx}: one fault per site across linked blocks");
        assert_eq!(j.direct_sites_patched(), 2);
        assert!(j.cache.inbound_links(CODE_BASE) >= 1, "loop back edge not linked");
        assert!(j.cache.inbound_links(CODE_BASE + 12) >= 1, "forward edge not linked");
        assert!(e.arena.ro_page_is_pristine());

        e.ctx.x[5] = 3;
        assert_eq!(run(&mut j, &mut e, &words), ExitReason::Brk(0));
        assert_eq!((writes(), reads()), (9, 9));
        assert_eq!(handled_faults() - before, 2);
    }
}

#[test]
fn cache_reset_forgets_sites_and_recompiled_block_patches_again() {
    let _serial = serial();
    let mut e = env();
    let mut j = jit(true);
    let before = handled_faults();
    e.ctx.x[0] = (RO_PAGE + 0x40) as u64;
    e.ctx.x[1] = 1;
    run(&mut j, &mut e, &[STR_X1_X0, BRK_0]);
    assert_eq!(writes(), 1);
    assert_eq!(handled_faults() - before, 1);
    assert_eq!(j.direct_sites_patched(), 1);
    assert!(j.cache.direct_site_count() >= 1);
    j.cache.reset().unwrap();
    assert_eq!(j.cache.direct_site_count(), 0);
    run(&mut j, &mut e, &[STR_X1_X0, BRK_0]);
    assert_eq!(writes(), 2);
    assert_eq!(handled_faults() - before, 2, "the recompiled block must patch again");
    assert_eq!(j.direct_sites_patched(), 2);
    assert!(j.cache.direct_site_count() >= 1);
    assert!(e.arena.ro_page_is_pristine());
}

#[test]
fn capped_block_sites_are_registered_and_patched() {
    let _serial = serial();
    for rwx in [false, true] {
        let mut e = env();
        let mut j = jit(rwx);
        let before = handled_faults();
        e.ctx.x[0] = (RO_PAGE + 0x60) as u64;
        e.ctx.x[1] = 0x0BAD_F00D;
        let words = [STR_X1_X0, STR_X1_X0, BRK_0];
        let mut mem = code(&words);
        for step in 0..2 {
            e.ctx.pc = CODE_BASE + step * 4;
            let out = j.run_bounded(&mut e.ctx, &mut mem, 1).expect("capped block runs");
            assert_eq!(out.retired, 1, "rwx {rwx}: step {step}");
            assert_eq!(e.ctx.pc, CODE_BASE + (step + 1) * 4);
        }
        assert_eq!(writes(), 2, "rwx {rwx}");
        assert_eq!(last_addr(), (RO_PAGE + 0x60) as u64);
        assert_eq!(last_write()[0], 0x0BAD_F00D);
        assert_eq!(handled_faults() - before, 2, "rwx {rwx}: each capped block has its own site");
        assert_eq!(j.direct_sites_patched(), 2);
        assert!(e.arena.ro_page_is_pristine());

        e.ctx.pc = CODE_BASE;
        let out = j.run_bounded(&mut e.ctx, &mut mem, 1).expect("cached capped block runs");
        assert_eq!(out.retired, 1);
        assert_eq!(writes(), 3);
        assert_eq!(handled_faults() - before, 2, "rwx {rwx}: patched capped site faulted again");
    }
}

#[test]
fn cache_rollover_keeps_every_new_site_patchable() {
    let _serial = serial();
    const BLOCKS: u64 = 1000;
    let mut e = env();
    let mut j = Jit::new(JitConfig {
        code_cache_bytes: 1 << 16,
        ..config(true)
    })
    .unwrap();
    let before = handled_faults();
    e.ctx.x[0] = (RO_PAGE + 0x50) as u64;
    e.ctx.x[1] = 9;
    for k in 0..BLOCKS {
        let pc = CODE_BASE + k * 0x100;
        assert!(matches!(run_at(&mut j, &mut e, pc, &[STR_X1_X0, BRK_0]), ExitReason::Brk(_)));
    }
    assert_eq!(writes(), BLOCKS);
    assert_eq!(handled_faults() - before, BLOCKS);
    assert_eq!(j.direct_sites_patched(), BLOCKS);
    assert!(
        (j.cache.direct_site_count() as u64) < BLOCKS,
        "a 64 KiB cache must have rolled over and forgotten earlier sites"
    );
    assert!(e.arena.ro_page_is_pristine());
}

fn emitted(word: u32) -> rustarmic::backend::EmittedBlock {
    let mut block = Block::new(CODE_BASE);
    translate_instruction(&mut IrEmitter::new(&mut block, CODE_BASE), word).unwrap();
    block.cycles = 1;
    block.end_pc = CODE_BASE + 4;
    block.terminal = Terminal::LinkBlock { next_pc: CODE_BASE + 4 };
    block.fastmem_direct_bits = BITS;
    optimize(&mut block);
    emit_block(&block).unwrap()
}

fn decode(code: &[u8]) -> Vec<Instruction> {
    let mut decoder = Decoder::with_ip(64, code, 0, DecoderOptions::NONE);
    let mut out = Vec::new();
    while decoder.can_decode() {
        out.push(decoder.decode());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ")
}

#[test]
fn direct_fast_path_is_shift_test_and_one_host_access() {
    let page_table_offset = {
        let ctx = CpuContext::default();
        let base = &ctx as *const CpuContext as usize;
        (&ctx.page_table as *const *const u8 as usize - base) as i64
    };
    let mem_base_offset = {
        let ctx = CpuContext::default();
        let base = &ctx as *const CpuContext as usize;
        (&ctx.mem_base as *const *mut u8 as usize - base) as i64
    };
    for (word, name, store) in [(LDR_W1_X0, "ldr w1, [x0]", false), (STR_W1_X0, "str w1, [x0]", true)] {
        let e = emitted(word);
        assert_eq!(e.direct_sites.len(), 1, "{name}");
        let site = e.direct_sites[0];
        assert!(site.len >= 3, "{name}: site len {}", site.len);
        let rel = site.slow_offset as i64 - (site.offset as i64 + 2);
        assert!((-128..=127).contains(&rel), "{name}: slow path out of short-jump range");
        let instructions = decode(&e.code);
        let at = instructions
            .iter()
            .position(|i| i.ip() == u64::from(site.offset))
            .expect("site offset is an instruction boundary");
        let access = instructions[at];
        assert_eq!(access.len(), site.len as usize, "{name}");
        assert_eq!(access.mnemonic(), Mnemonic::Mov, "{name}");
        assert!((0..access.op_count()).any(|i| access.op_kind(i) == OpKind::Memory), "{name}");
        assert_eq!(access.memory_base(), Register::RAX, "{name}");
        assert!(matches!(access.memory_index(), Register::RDX | Register::RSI), "{name}");
        assert_eq!(access.memory_displacement64(), 0, "{name}");
        let guard = &instructions[at - 4..at];
        assert_eq!(guard[0].mnemonic(), Mnemonic::Mov, "{name}");
        assert_eq!(guard[0].op0_kind(), OpKind::Register, "{name}");
        assert_eq!(guard[1].mnemonic(), Mnemonic::Shr, "{name}");
        assert_eq!(guard[1].immediate8(), BITS, "{name}");
        assert_eq!(guard[2].mnemonic(), Mnemonic::Jne, "{name}");
        assert_eq!(guard[2].near_branch64(), u64::from(site.slow_offset), "{name}");
        assert_eq!(guard[3].mnemonic(), Mnemonic::Mov, "{name}");
        assert_eq!(guard[3].memory_base(), Register::R15, "{name}");
        assert_eq!(guard[3].memory_displacement64() as i64, mem_base_offset, "{name}");
        let join = instructions[at + 1];
        assert_eq!(join.mnemonic(), Mnemonic::Jmp, "{name}: fast path must skip the slow path");
        for i in &instructions {
            assert!(
                !(i.memory_base() == Register::R15 && i.memory_displacement64() as i64 == page_table_offset),
                "{name}: direct mode must not read the page table"
            );
            assert!(
                !(i.mnemonic() == Mnemonic::And && i.immediate32() == 0xFFF),
                "{name}: direct mode has no page-boundary check"
            );
        }
        let start = guard[0].ip() as usize;
        let end = join.ip() as usize + join.len();
        println!(
            "{name} ({}): {}",
            if store { "store" } else { "load" },
            hex(&e.code[start..end])
        );
    }
}
