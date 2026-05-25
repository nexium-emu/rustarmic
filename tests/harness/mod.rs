//! Shared harness for differential tests.

use rustarmic::{CpuContext, Jit, JitConfig, Memory};
use std::sync::Mutex;
use unicorn_engine::{Arch, Mode, Prot, RegisterARM64, Unicorn};

// Flat backing store shared between rustarmic memory callbacks and the
// harness setup helpers. Each test should call `mem_init` to size it before
// running anything that touches memory. Lives at offset `DATA_BASE` in the
// guest address space (matches the Unicorn mapping below).
static MEM: Mutex<Vec<u8>> = Mutex::new(Vec::new());

#[allow(dead_code)]
pub fn mem_init(size: usize) {
    let mut m = MEM.lock().unwrap();
    if m.len() < size { m.resize(size, 0); }
}

fn mem_offset(addr: u64) -> usize { (addr - DATA_BASE) as usize }

fn read_bytes(addr: u64, bytes: usize) -> u64 {
    let m = MEM.lock().unwrap();
    let off = mem_offset(addr);
    if off + bytes > m.len() { return 0; }
    let mut buf = [0u8; 8];
    buf[..bytes].copy_from_slice(&m[off..off + bytes]);
    u64::from_le_bytes(buf)
}

fn write_bytes(addr: u64, value: u64, bytes: usize) {
    let mut m = MEM.lock().unwrap();
    let off = mem_offset(addr);
    if off + bytes > m.len() { return; }
    m[off..off + bytes].copy_from_slice(&value.to_le_bytes()[..bytes]);
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read8(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u8 {
    read_bytes(addr, 1) as u8
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read16(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u16 {
    read_bytes(addr, 2) as u16
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read32(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u32 {
    read_bytes(addr, 4) as u32
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_read64(_: u64, addr: u64, _: u64, _: *mut CpuContext) -> u64 {
    read_bytes(addr, 8)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write8(_: u64, addr: u64, v: u8, _: *mut CpuContext) {
    write_bytes(addr, v as u64, 1)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write16(_: u64, addr: u64, v: u16, _: *mut CpuContext) {
    write_bytes(addr, v as u64, 2)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write32(_: u64, addr: u64, v: u32, _: *mut CpuContext) {
    write_bytes(addr, v as u64, 4)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rustarmic_mem_write64(_: u64, addr: u64, v: u64, _: *mut CpuContext) {
    write_bytes(addr, v, 8)
}

pub const CODE_BASE: u64 = 0x1000;
pub const CODE_SIZE: u64 = 0x1000;
pub const DATA_BASE: u64 = 0x10_0000;
pub const DATA_SIZE: u64 = 0x10_0000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegState {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub nzcv: u8,
    /// V[0..31], each represented as `[low_u64, high_u64]` (little-endian
    /// lane order — V[i][0] holds bytes 0..7, V[i][1] holds bytes 8..15).
    pub v: [[u64; 2]; 32],
}

/// Execute `code` on Unicorn and rustarmic with the same initial state and
/// return the final register snapshot from each.
pub fn run_pair(code: &[u8], init: RegState) -> (RegState, RegState) {
    let uni_state = run_unicorn(code, init);
    let jit_state = run_rustarmic(code, init);
    (uni_state, jit_state)
}

fn run_unicorn(code: &[u8], init: RegState) -> RegState {
    let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN)
        .expect("unicorn init failed");

    // Map code region.
    emu.mem_map(CODE_BASE, CODE_SIZE, Prot::ALL).unwrap();
    emu.mem_write(CODE_BASE, code).unwrap();

    // Map data region (acts as stack + heap).
    emu.mem_map(DATA_BASE, DATA_SIZE, Prot::ALL).unwrap();

    // Enable FP/SIMD via CPACR_EL1.FPEN = 0b11. Without this Unicorn traps
    // every NEON instruction silently (emu_start returns an error we ignore)
    // leaving the V regs untouched — making the diff look like our JIT is
    // wrong when it's actually executing correctly.
    let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
    let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));

    // Seed registers.
    for i in 0..31 {
        emu.reg_write(arm_reg(i), init.x[i]).unwrap();
    }
    emu.reg_write(RegisterARM64::SP, init.sp).unwrap();
    emu.reg_write(RegisterARM64::NZCV, (init.nzcv as u64) << 28).unwrap();
    for i in 0..32 {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&init.v[i][0].to_le_bytes());
        buf[8..].copy_from_slice(&init.v[i][1].to_le_bytes());
        emu.reg_write_long(arm_qreg(i), &buf).unwrap();
    }

    // Run until either BRK or end of code mapping. Unicorn stops on BRK
    // automatically via the exception handler; we set a sane instruction
    // limit just in case.
    let end = CODE_BASE + code.len() as u64;
    let _ = emu.emu_start(init.pc, end, 0, 1024);

    let mut out = RegState::default();
    for i in 0..31 {
        out.x[i] = emu.reg_read(arm_reg(i)).unwrap();
    }
    out.sp = emu.reg_read(RegisterARM64::SP).unwrap();
    out.pc = emu.reg_read(RegisterARM64::PC).unwrap();
    let nzcv_full = emu.reg_read(RegisterARM64::NZCV).unwrap();
    out.nzcv = ((nzcv_full >> 28) & 0xF) as u8;
    for i in 0..32 {
        let bytes = emu.reg_read_long(arm_qreg(i)).unwrap();
        let mut lo = [0u8; 8]; lo.copy_from_slice(&bytes[..8]);
        let mut hi = [0u8; 8]; hi.copy_from_slice(&bytes[8..]);
        out.v[i] = [u64::from_le_bytes(lo), u64::from_le_bytes(hi)];
    }
    out
}

fn arm_reg(i: usize) -> RegisterARM64 {
    use RegisterARM64::*;
    match i {
        0 => X0, 1 => X1, 2 => X2, 3 => X3, 4 => X4, 5 => X5, 6 => X6, 7 => X7,
        8 => X8, 9 => X9, 10 => X10, 11 => X11, 12 => X12, 13 => X13, 14 => X14, 15 => X15,
        16 => X16, 17 => X17, 18 => X18, 19 => X19, 20 => X20, 21 => X21, 22 => X22, 23 => X23,
        24 => X24, 25 => X25, 26 => X26, 27 => X27, 28 => X28, 29 => X29, 30 => X30,
        _ => unreachable!(),
    }
}

fn arm_qreg(i: usize) -> RegisterARM64 {
    use RegisterARM64::*;
    match i {
        0 => Q0,  1 => Q1,  2 => Q2,  3 => Q3,  4 => Q4,  5 => Q5,  6 => Q6,  7 => Q7,
        8 => Q8,  9 => Q9,  10 => Q10, 11 => Q11, 12 => Q12, 13 => Q13, 14 => Q14, 15 => Q15,
        16 => Q16, 17 => Q17, 18 => Q18, 19 => Q19, 20 => Q20, 21 => Q21, 22 => Q22, 23 => Q23,
        24 => Q24, 25 => Q25, 26 => Q26, 27 => Q27, 28 => Q28, 29 => Q29, 30 => Q30, 31 => Q31,
        _ => unreachable!(),
    }
}

struct HostMem {
    code: Vec<u8>,
    code_base: u64,
}

impl Memory for HostMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(self.code_base)? as usize;
        if off + 4 > self.code.len() { return None; }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.code[off..off + 4]);
        Some(u32::from_le_bytes(buf))
    }
}

fn run_rustarmic(code: &[u8], init: RegState) -> RegState {
    let mut mem = HostMem { code: code.to_vec(), code_base: CODE_BASE };
    // We don't currently exercise data accesses against a fastmem pointer;
    // those tests will live alongside data-bearing snippets and use a
    // separate flat allocation later. Pass null for now.

    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let mut ctx = CpuContext::default();
    ctx.pc = init.pc;
    ctx.sp = init.sp;
    ctx.nzcv = init.nzcv;
    for i in 0..31 {
        ctx.x[i] = init.x[i];
    }
    for i in 0..32 {
        ctx.v[i] = init.v[i];
    }

    // Each `run` chains through linked blocks internally and only returns on
    // exception / unchainable exit; one call is enough for the harness.
    let _ = jit.run(&mut ctx, &mut mem);

    let mut out = RegState::default();
    for i in 0..31 { out.x[i] = ctx.x[i]; }
    out.sp = ctx.sp;
    out.pc = ctx.pc;
    out.nzcv = ctx.nzcv;
    for i in 0..32 { out.v[i] = ctx.v[i]; }
    out
}
