use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::sync::Mutex;

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x10_0000;

static MEM: Mutex<Vec<u8>> = Mutex::new(Vec::new());

fn mem_init(size: usize) {
    let mut m = MEM.lock().unwrap();
    if m.len() < size { m.resize(size, 0); }
}

fn mem_read(addr: u64, bytes: usize) -> u64 {
    let m = MEM.lock().unwrap();
    let off = (addr - DATA_BASE) as usize;
    let mut buf = [0u8; 8];
    buf[..bytes].copy_from_slice(&m[off..off + bytes]);
    u64::from_le_bytes(buf)
}

fn mem_write(addr: u64, value: u64, bytes: usize) {
    let mut m = MEM.lock().unwrap();
    let off = (addr - DATA_BASE) as usize;
    m[off..off + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read8(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u8 {
    mem_read(addr, 1) as u8
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read16(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u16 {
    mem_read(addr, 2) as u16
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read32(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u32 {
    mem_read(addr, 4) as u32
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read64(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u64 {
    mem_read(addr, 8)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write8(_: u64, addr: u64, value: u8, _: *mut CpuContext) {
    mem_write(addr, value as u64, 1)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write16(_: u64, addr: u64, value: u16, _: *mut CpuContext) {
    mem_write(addr, value as u64, 2)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write32(_: u64, addr: u64, value: u32, _: *mut CpuContext) {
    mem_write(addr, value as u64, 4)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write64(_: u64, addr: u64, value: u64, _: *mut CpuContext) {
    mem_write(addr, value, 8)
}

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

fn run(code: Vec<u8>, ctx: &mut CpuContext) -> ExitReason {
    let mut mem = CodeMem { bytes: code, base: CODE_BASE };
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    jit.run(ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

#[test]
fn ldxr_stxr_success_then_self_fail() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x200, 0x12345678_9ABCDEF0, 8);

    // movz x1, #0x200 ; movz x0, #(DATA_BASE>>... build address)
    // Easier: put DATA_BASE+0x200 in X0 via ctx, then:
    //   ldxr  x2, [x0]            ; x2 = mem; reservation set on x0
    //   movz  x3, #0xAAAA
    //   stxr  w4, x3, [x0]        ; should succeed -> w4 = 0
    //   movz  x5, #0xBBBB
    //   stxr  w6, x5, [x0]        ; reservation cleared by prev stxr -> w6 = 1
    //   brk #0
    let code = build_code(&[
        0xC85F7C02, // ldxr x2, [x0]
        0xD2955543, // movz x3, #0xAAAA
        0xC8047C03, // stxr w4, x3, [x0]
        0xD2977765, // movz x5, #0xBBBB
        0xC8067C05, // stxr w6, x5, [x0]
        0xD4200000, // brk #0
    ]);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x200;
    let mut mem = CodeMem { bytes: code, base: CODE_BASE };
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let exit = jit.run(&mut ctx, &mut mem).unwrap_or(ExitReason::Stopped);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);

    assert_eq!(ctx.x[2], 0x12345678_9ABCDEF0, "ldxr loaded original value");
    assert_eq!(ctx.x[4], 0, "first stxr must succeed (reservation held)");
    assert_eq!(ctx.x[6], 1, "second stxr must fail (reservation cleared)");
    assert_eq!(mem_read(DATA_BASE + 0x200, 8), 0xAAAA,
        "memory should hold first stxr's value, not second's");
}

#[test]
fn clrex_clears_reservation_so_stxr_fails() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x300, 0xDEAD_BEEF_CAFE_BABE, 8);

    // ldxr x2, [x0]   ; load + set reservation
    // clrex           ; explicitly clear reservation
    // movz x3, #0x11
    // stxr w4, x3, [x0]  ; w4 must be 1 (no reservation)
    let code = build_code(&[
        0xC85F7C02, // ldxr x2, [x0]
        0xD5033F5F, // clrex
        0xD2800223, // movz x3, #0x11
        0xC8047C03, // stxr w4, x3, [x0]
        0xD4200000, // brk #0
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x300;
    let mut mem = CodeMem { bytes: code, base: CODE_BASE };
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let exit = jit.run(&mut ctx, &mut mem).unwrap_or(ExitReason::Stopped);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK");
    assert_eq!(ctx.x[2], 0xDEAD_BEEF_CAFE_BABE, "ldxr loaded original");
    assert_eq!(ctx.x[4], 1, "stxr must fail after clrex");
    assert_eq!(mem_read(DATA_BASE + 0x300, 8), 0xDEAD_BEEF_CAFE_BABE,
        "memory unchanged");
}

#[test]
fn str_then_ldr_round_trip() {
    mem_init(0x10000);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE;
    let code = build_code(&[
        0xD2802001, // movz x1, #0x100
        0x8B000020, // add  x0, x1, x0       (x0 = DATA_BASE + 0x100)
        0xD2824682, // movz x2, #0x1234
        0xF9000002, // str  x2, [x0]
        0xF9400003, // ldr  x3, [x0]
        0xD4200000, // brk
    ]);
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
    assert_eq!(ctx.x[2], 0x1234, "X2 should still hold the stored value");
    assert_eq!(ctx.x[3], 0x1234, "X3 should be loaded value");

    let stored = mem_read(DATA_BASE + 0x100, 8);
    assert_eq!(stored, 0x1234, "memory at DATA_BASE+0x100 should be 0x1234");
}

#[test]
fn ldadd_atomic_add_returns_old_and_writes_sum() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x400, 100, 8);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x400;
    ctx.x[1] = 7;
    let code = build_code(&[
        0xF821_0002, // ldadd x1, x2, [x0]
        0xD4200000,
    ]);
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
    assert_eq!(ctx.x[2], 100, "X2 must receive old memory value");
    assert_eq!(mem_read(DATA_BASE + 0x400, 8), 107, "memory must equal old + Rs");
}

#[test]
fn swp_atomic_exchanges_value() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x500, 0xAAAA, 8);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x500;
    ctx.x[1] = 0xBBBB;
    let code = build_code(&[
        0xF821_8002, // swp x1, x2, [x0]
        0xD4200000,
    ]);
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
    assert_eq!(ctx.x[2], 0xAAAA, "X2 must receive old memory");
    assert_eq!(mem_read(DATA_BASE + 0x500, 8), 0xBBBB, "memory must hold Rs");
}

#[test]
fn cas_success_writes_rt_and_returns_old_in_rs() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x600, 0x100, 8);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x600;
    ctx.x[1] = 0x100;   // compare value (matches memory)
    ctx.x[2] = 0x999;   // new value
    let code = build_code(&[
        0xC8A1_7C02, // cas x1, x2, [x0]
        0xD4200000,
    ]);
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
    assert_eq!(ctx.x[1], 0x100, "Rs must always receive old memory");
    assert_eq!(mem_read(DATA_BASE + 0x600, 8), 0x999, "matching CAS must write Rt");
}

#[test]
fn cas_failure_leaves_memory_unchanged() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x700, 0x100, 8);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE + 0x700;
    ctx.x[1] = 0x200;   // wrong compare value
    ctx.x[2] = 0x999;
    let code = build_code(&[
        0xC8A1_7C02, // cas x1, x2, [x0]
        0xD4200000,
    ]);
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
    assert_eq!(ctx.x[1], 0x100, "Rs always receives old memory");
    assert_eq!(mem_read(DATA_BASE + 0x700, 8), 0x100, "non-matching CAS must NOT store");
}

#[test]
fn ldr_dt_loads_double_from_memory() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x800, (3.14_f64).to_bits(), 8);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = DATA_BASE + 0x800;
    let code = build_code(&[
        0xFD40_0020, // ldr d0, [x1]
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 3.14, "LDR D0 must load 3.14 from memory");
    assert_eq!(ctx.v[0][1], 0, "high lane zeroed");
}

#[test]
fn str_dt_stores_double_to_memory() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0x900, 0xAAAA_AAAA_AAAA_AAAA, 8);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = DATA_BASE + 0x900;
    ctx.v[0] = [(2.5_f64).to_bits(), 0];
    let code = build_code(&[
        0xFD00_0020, // str d0, [x1]
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(mem_read(DATA_BASE + 0x900, 8)), 2.5, "STR D0 must write 2.5");
}

#[test]
fn ldr_st_loads_float_from_memory() {
    mem_init(0x10000);
    mem_write(DATA_BASE + 0xA00, (1.5_f32).to_bits() as u64, 4);

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = DATA_BASE + 0xA00;
    let code = build_code(&[
        0xBD40_0020, // ldr s0, [x1]
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(f32::from_bits(ctx.v[0][0] as u32), 1.5);
    assert_eq!(ctx.v[0][0] >> 32, 0, "upper 32 of lane 0 zeroed");
}
