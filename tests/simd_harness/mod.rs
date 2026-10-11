//! Single-snippet differential runner: one Unicorn instance (max CPU
//! model) and one Rustarmic `Jit` reused across cases, comparing every
//! general, vector and flag register plus the data region.
#![allow(dead_code)]

use rustarmic::jit::context::{PAGE_DIRECT_READ, PAGE_DIRECT_WRITE};
use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::cell::RefCell;
use unicorn_engine::{Arch, Mode, Prot, RegisterARM64, Unicorn};

pub const MEM_SIZE: usize = 0x40000;
pub const CODE: u64 = 0x10000;
pub const CODE_LIMIT: u64 = 0x11000;
pub const DATA: u64 = 0x20000;
pub const DATA_SIZE: usize = 0x10000;
pub const BRK_0: u32 = 0xD420_0000;
const PAGE_BITS: u8 = 18;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct State {
    pub x: [u64; 31],
    pub sp: u64,
    pub nzcv: u8,
    pub fpcr: u32,
    pub v: [[u64; 2]; 32],
    pub data: Vec<u8>,
}

impl State {
    pub fn new() -> Self {
        Self {
            x: [0; 31],
            sp: DATA + 0x8000,
            nzcv: 0,
            fpcr: 0,
            v: [[0; 2]; 32],
            data: vec![0; DATA_SIZE],
        }
    }

    /// Readable description of the differences against `other`.
    pub fn diff(&self, other: &State) -> String {
        let mut s = String::new();
        for i in 0..31 {
            if self.x[i] != other.x[i] {
                s += &format!(" x{i}: {:#x} vs {:#x};", self.x[i], other.x[i]);
            }
        }
        if self.sp != other.sp {
            s += &format!(" sp: {:#x} vs {:#x};", self.sp, other.sp);
        }
        if self.nzcv != other.nzcv {
            s += &format!(" nzcv: {:04b} vs {:04b};", self.nzcv, other.nzcv);
        }
        for i in 0..32 {
            if self.v[i] != other.v[i] {
                s += &format!(
                    " v{i}: {:016x}_{:016x} vs {:016x}_{:016x};",
                    self.v[i][1], self.v[i][0], other.v[i][1], other.v[i][0]
                );
            }
        }
        if self.data != other.data {
            let first = self.data.iter().zip(&other.data).position(|(a, b)| a != b).unwrap();
            s += &format!(
                " data differs from {:#x}: {:02x?} vs {:02x?};",
                DATA + first as u64,
                &self.data[first..(first + 16).min(DATA_SIZE)],
                &other.data[first..(first + 16).min(DATA_SIZE)]
            );
        }
        s
    }
}

fn xreg(i: usize) -> RegisterARM64 {
    use RegisterARM64::*;
    [
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19,
        X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30,
    ][i]
}

fn qreg(i: usize) -> RegisterARM64 {
    use RegisterARM64::*;
    [
        Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19,
        Q20, Q21, Q22, Q23, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31,
    ][i]
}

pub struct Uni {
    emu: Unicorn<'static, ()>,
}

/// Outcome of running a snippet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exit {
    /// Reached the final BRK.
    Done,
    /// Stopped early (exception, unsupported instruction, fault).
    Other(String),
}

impl Uni {
    pub fn new() -> Self {
        let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).unwrap();
        // UC_CPU_ARM64_MAX: every optional extension (dot product, SHA-3, ...)
        emu.ctl_set_cpu_model(3).unwrap();
        emu.mem_map(0x1000, MEM_SIZE as u64 - 0x1000, Prot::ALL).unwrap();
        let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
        let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));
        Self { emu }
    }

    pub fn run(&mut self, words: &[u32], init: &State) -> (State, Exit) {
        let emu = &mut self.emu;
        let mut code = Vec::new();
        for w in words {
            code.extend_from_slice(&w.to_le_bytes());
        }
        emu.mem_write(CODE, &code).unwrap();
        // Unicorn may keep translations of the previous snippet.
        emu.ctl_flush_tb().unwrap();
        emu.mem_write(DATA, &init.data).unwrap();
        for i in 0..31 {
            emu.reg_write(xreg(i), init.x[i]).unwrap();
        }
        emu.reg_write(RegisterARM64::SP, init.sp).unwrap();
        emu.reg_write(RegisterARM64::NZCV, (init.nzcv as u64) << 28).unwrap();
        emu.reg_write(RegisterARM64::FPCR, init.fpcr as u64).unwrap();
        emu.reg_write(RegisterARM64::FPSR, 0).unwrap();
        for i in 0..32 {
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&init.v[i][0].to_le_bytes());
            b[8..].copy_from_slice(&init.v[i][1].to_le_bytes());
            emu.reg_write_long(qreg(i), &b).unwrap();
        }
        let end = CODE + 4 * (words.len() as u64 - 1);
        let res = emu.emu_start(CODE, end, 0, 10_000);
        let pc = emu.reg_read(RegisterARM64::PC).unwrap();
        let exit = if res.is_ok() && pc == end {
            Exit::Done
        } else {
            Exit::Other(format!("{res:?} pc={pc:#x}"))
        };
        let mut out = State::new();
        for i in 0..31 {
            out.x[i] = emu.reg_read(xreg(i)).unwrap();
        }
        out.sp = emu.reg_read(RegisterARM64::SP).unwrap();
        out.nzcv = ((emu.reg_read(RegisterARM64::NZCV).unwrap() >> 28) & 0xF) as u8;
        out.fpcr = init.fpcr;
        for i in 0..32 {
            let b = emu.reg_read_long(qreg(i)).unwrap();
            out.v[i] = [
                u64::from_le_bytes(b[..8].try_into().unwrap()),
                u64::from_le_bytes(b[8..16].try_into().unwrap()),
            ];
        }
        emu.mem_read(DATA, &mut out.data).unwrap();
        (out, exit)
    }
}

thread_local! {
    static GUEST: RefCell<Vec<u8>> = RefCell::new(vec![0; MEM_SIZE]);
}

fn in_range(addr: u64, size: u8) -> bool {
    addr >= 0x1000 && addr.checked_add(size as u64).is_some_and(|e| e <= MEM_SIZE as u64)
}

fn fault(ctx: *mut CpuContext, addr: u64, size: u8, write: bool) {
    unsafe {
        let ctx = &mut *ctx;
        ctx.mem_fault = 1;
        ctx.mem_fault_access = write as u8;
        ctx.mem_fault_size = size;
        ctx.mem_fault_addr = addr;
        ctx.mem_fault_pc = ctx.pc;
    }
}

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, addr: u64, size: u8) {
    if !in_range(addr, size) {
        fault(ctx, addr, size, false);
        unsafe { (*ctx).io_value = [0; 2] };
        return;
    }
    let mut buf = [0u8; 16];
    GUEST.with(|g| buf[..size as usize].copy_from_slice(&g.borrow()[addr as usize..][..size as usize]));
    unsafe {
        (*ctx).io_value = [
            u64::from_le_bytes(buf[..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..].try_into().unwrap()),
        ];
    }
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    if !in_range(addr, size) {
        fault(ctx, addr, size, true);
        return;
    }
    let io = unsafe { (*ctx).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    GUEST.with(|g| g.borrow_mut()[addr as usize..][..size as usize].copy_from_slice(&buf[..size as usize]));
}

struct Fetch;
impl Memory for Fetch {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        if !(CODE..CODE_LIMIT).contains(&addr) {
            return None;
        }
        GUEST.with(|g| Some(u32::from_le_bytes(g.borrow()[addr as usize..][..4].try_into().unwrap())))
    }
}

pub struct Rj {
    jit: Jit,
    pagetable: bool,
    table: Vec<u8>,
}

impl Rj {
    /// `pagetable`: inline page-table fastmem for the data pages;
    /// otherwise every access goes through the callbacks.
    pub fn new(pagetable: bool) -> Self {
        let mut config = JitConfig {
            code_cache_bytes: 16 << 20,
            ..JitConfig::default()
        };
        if pagetable {
            config.fastmem_page_bits = Some(PAGE_BITS);
        }
        let mut table = vec![PAGE_DIRECT_READ | PAGE_DIRECT_WRITE; MEM_SIZE >> 12];
        table[0] = 0;
        for page in (CODE >> 12)..(CODE_LIMIT >> 12) {
            table[page as usize] = PAGE_DIRECT_READ;
        }
        Self {
            jit: Jit::new(config).unwrap(),
            pagetable,
            table,
        }
    }

    pub fn run(&mut self, words: &[u32], init: &State) -> (State, Exit, CpuContext) {
        GUEST.with(|g| {
            let mut g = g.borrow_mut();
            for (i, w) in words.iter().enumerate() {
                g[CODE as usize + 4 * i..][..4].copy_from_slice(&w.to_le_bytes());
            }
            g[DATA as usize..][..DATA_SIZE].copy_from_slice(&init.data);
        });
        self.jit.invalidate_range(CODE, CODE_LIMIT - CODE);
        let mut ctx = CpuContext::default();
        ctx.mem_read = hk_read;
        ctx.mem_write = hk_write;
        if self.pagetable {
            ctx.mem_base = GUEST.with(|g| g.borrow_mut().as_mut_ptr());
            ctx.page_table = self.table.as_ptr();
        }
        ctx.pc = CODE;
        ctx.sp = init.sp;
        ctx.x = init.x;
        ctx.nzcv = init.nzcv;
        ctx.fpcr = init.fpcr;
        ctx.v = init.v;
        let exit = match self.jit.run_bounded(&mut ctx, &mut Fetch, 10_000) {
            Ok(o) if o.reason == ExitReason::Brk(0) => Exit::Done,
            Ok(o) => Exit::Other(format!("{:?} {:?}", o.reason, o.stop)),
            Err(e) => Exit::Other(format!("error {e:?}")),
        };
        let mut out = State::new();
        out.x = ctx.x;
        out.sp = ctx.sp;
        out.nzcv = ctx.nzcv & 0xF;
        out.fpcr = init.fpcr;
        out.v = ctx.v;
        GUEST.with(|g| out.data.copy_from_slice(&g.borrow()[DATA as usize..][..DATA_SIZE]));
        (out, exit, ctx)
    }
}
