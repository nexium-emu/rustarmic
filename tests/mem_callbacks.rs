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
