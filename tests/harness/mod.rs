//! Shared harness for differential tests.

use rustarmic::jit::memory::FlatMemory;
use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use unicorn_engine::unicorn_const::{Arch, Mode, Permission};
use unicorn_engine::{RegisterARM64, Unicorn};

pub const CODE_BASE: u64 = 0x1000;
pub const CODE_SIZE: usize = 0x1000;
pub const DATA_BASE: u64 = 0x10_0000;
pub const DATA_SIZE: usize = 0x10_0000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegState {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    pub nzcv: u8,
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
    emu.mem_map(CODE_BASE, CODE_SIZE, Permission::ALL).unwrap();
    emu.mem_write(CODE_BASE, code).unwrap();

    // Map data region (acts as stack + heap).
    emu.mem_map(DATA_BASE, DATA_SIZE, Permission::ALL).unwrap();

    // Seed registers.
    for i in 0..31 {
        emu.reg_write(arm_reg(i), init.x[i]).unwrap();
    }
    emu.reg_write(RegisterARM64::SP, init.sp).unwrap();
    emu.reg_write(RegisterARM64::NZCV, (init.nzcv as u64) << 28).unwrap();

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

    // Loop until the JIT signals an exception or returns Stopped.
    let mut budget = 32;
    while budget > 0 {
        match jit.run(&mut ctx, &mut mem).unwrap_or(ExitReason::Stopped) {
            ExitReason::Brk(_) | ExitReason::Svc(_) | ExitReason::Hvc(_) |
            ExitReason::Stopped => break,
            ExitReason::MemoryFault(_) => break,
        }
        budget -= 1;
    }

    let mut out = RegState::default();
    for i in 0..31 { out.x[i] = ctx.x[i]; }
    out.sp = ctx.sp;
    out.pc = ctx.pc;
    out.nzcv = ctx.nzcv;
    out
}
