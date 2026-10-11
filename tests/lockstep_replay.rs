//! Replays a NeXium lockstep repro (`~/.config/NeXium/lockstep/*.repro`) on
//! Unicorn and Rustarmic, and compares both with the Dynarmic end state the
//! repro recorded:
//!
//! ```text
//! RUSTARMIC_LOCKSTEP_REPRO=/path/to/file.repro \
//!     cargo test --features unicorn --test lockstep_replay -- --nocapture
//! ```
//!
//! Without the variable only the built-in self-test runs.

#![cfg(feature = "unicorn")]

use std::cell::RefCell;
use std::collections::BTreeMap;

use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
use unicorn_engine::{Arch, Mode, Prot, RegisterARM64, Unicorn};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct State {
    x: [u64; 31],
    sp: u64,
    pc: u64,
    v: [[u64; 2]; 32],
    nzcv: u8,
    fpcr: u32,
    fpsr: u32,
    tpidr: u64,
    tpidrro: u64,
}

impl State {
    fn set(&mut self, field: &str, values: &[u64]) -> Result<(), String> {
        let one = || values.first().copied().ok_or_else(|| format!("{field}: no value"));
        match field {
            "pc" => self.pc = one()?,
            "sp" => self.sp = one()?,
            "nzcv" => self.nzcv = one()? as u8,
            "fpcr" => self.fpcr = one()? as u32,
            "fpsr" => self.fpsr = one()? as u32,
            "tpidr" => self.tpidr = one()?,
            "tpidrro" => self.tpidrro = one()?,
            _ => {
                if let Some(i) = field.strip_prefix('x').and_then(|i| i.parse::<usize>().ok()) {
                    self.x[i] = one()?;
                } else if let Some(i) = field.strip_prefix('v').and_then(|i| i.parse::<usize>().ok()) {
                    self.v[i] = [values[0], values[1]];
                } else {
                    return Err(format!("unknown field {field}"));
                }
            }
        }
        Ok(())
    }

    fn diff(&self, other: &State) -> Vec<String> {
        let mut out = Vec::new();
        for i in 0..31 {
            if self.x[i] != other.x[i] {
                out.push(format!("x{i} {:#x} vs {:#x}", self.x[i], other.x[i]));
            }
        }
        for (name, a, b) in [("sp", self.sp, other.sp), ("pc", self.pc, other.pc)] {
            if a != b {
                out.push(format!("{name} {a:#x} vs {b:#x}"));
            }
        }
        if self.nzcv != other.nzcv {
            out.push(format!("nzcv {:#x} vs {:#x}", self.nzcv, other.nzcv));
        }
        if self.fpsr != other.fpsr {
            out.push(format!("fpsr {:#x} vs {:#x}", self.fpsr, other.fpsr));
        }
        for i in 0..32 {
            if self.v[i] != other.v[i] {
                out.push(format!(
                    "v{i} {:#x}_{:016x} vs {:#x}_{:016x}",
                    self.v[i][1], self.v[i][0], other.v[i][1], other.v[i][0]
                ));
            }
        }
        out
    }
}

#[derive(Debug, Default)]
struct Repro {
    insns: u64,
    start: State,
    dynarmic: State,
    rustarmic: State,
    code: Vec<(u64, Vec<u8>)>,
    data: Vec<(u64, Vec<u8>)>,
}

fn num(s: &str) -> Result<u64, String> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16),
        None => s.parse(),
    }
    .map_err(|e| format!("{s}: {e}"))
}

fn parse(text: &str) -> Result<Repro, String> {
    let mut r = Repro::default();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let key = it.next().unwrap();
        let rest: Vec<&str> = it.collect();
        match key {
            "class" | "pc" | "word" | "dyn.mem" | "rust.mem" => {}
            "insns" => r.insns = num(rest[0])?,
            "code" | "data" => {
                let bytes = (0..rest[1].len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&rest[1][i..i + 2], 16).map_err(|e| e.to_string()))
                    .collect::<Result<Vec<u8>, String>>()?;
                let entry = (num(rest[0])?, bytes);
                if key == "code" {
                    r.code.push(entry);
                } else {
                    r.data.push(entry);
                }
            }
            _ => {
                let (prefix, field) = key.split_once('.').ok_or(format!("bad key {key}"))?;
                let values = rest.iter().map(|v| num(v)).collect::<Result<Vec<_>, _>>()?;
                match prefix {
                    "start" => r.start.set(field, &values)?,
                    "dyn" => r.dynarmic.set(field, &values)?,
                    "rust" => r.rustarmic.set(field, &values)?,
                    _ => return Err(format!("bad key {key}")),
                }
            }
        }
    }
    Ok(r)
}

/// Sparse 4 KiB pages holding the repro's memory.
fn pages(r: &Repro) -> BTreeMap<u64, Vec<u8>> {
    let mut pages: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    for (addr, bytes) in r.code.iter().chain(&r.data) {
        for (i, &b) in bytes.iter().enumerate() {
            let a = addr + i as u64;
            pages.entry(a & !0xFFF).or_insert_with(|| vec![0; 0x1000])[(a & 0xFFF) as usize] = b;
        }
    }
    pages
}

fn xreg(i: usize) -> RegisterARM64 {
    use RegisterARM64::*;
    [
        X0, X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21,
        X22, X23, X24, X25, X26, X27, X28, X29, X30,
    ][i]
}

fn qreg(i: usize) -> RegisterARM64 {
    use RegisterARM64::*;
    [
        Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16, Q17, Q18, Q19, Q20, Q21,
        Q22, Q23, Q24, Q25, Q26, Q27, Q28, Q29, Q30, Q31,
    ][i]
}

fn run_unicorn(r: &Repro) -> State {
    let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN).expect("unicorn");
    for (page, bytes) in pages(r) {
        emu.mem_map(page, 0x1000, Prot::ALL).unwrap();
        emu.mem_write(page, &bytes).unwrap();
    }
    let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
    let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));
    let s = &r.start;
    for i in 0..31 {
        emu.reg_write(xreg(i), s.x[i]).unwrap();
    }
    emu.reg_write(RegisterARM64::SP, s.sp).unwrap();
    emu.reg_write(RegisterARM64::NZCV, u64::from(s.nzcv) << 28).unwrap();
    let _ = emu.reg_write(RegisterARM64::FPCR, u64::from(s.fpcr));
    let _ = emu.reg_write(RegisterARM64::FPSR, u64::from(s.fpsr));
    let _ = emu.reg_write(RegisterARM64::TPIDR_EL0, s.tpidr);
    let _ = emu.reg_write(RegisterARM64::TPIDRRO_EL0, s.tpidrro);
    for i in 0..32 {
        let mut buf = [0u8; 16];
        buf[..8].copy_from_slice(&s.v[i][0].to_le_bytes());
        buf[8..].copy_from_slice(&s.v[i][1].to_le_bytes());
        emu.reg_write_long(qreg(i), &buf).unwrap();
    }
    let _ = emu.emu_start(s.pc, u64::MAX, 0, r.insns as usize);
    let mut out = State::default();
    for i in 0..31 {
        out.x[i] = emu.reg_read(xreg(i)).unwrap();
    }
    out.sp = emu.reg_read(RegisterARM64::SP).unwrap();
    out.pc = emu.reg_read(RegisterARM64::PC).unwrap();
    out.nzcv = ((emu.reg_read(RegisterARM64::NZCV).unwrap() >> 28) & 0xF) as u8;
    out.fpcr = emu.reg_read(RegisterARM64::FPCR).unwrap_or(0) as u32;
    out.fpsr = emu.reg_read(RegisterARM64::FPSR).unwrap_or(0) as u32;
    out.tpidr = emu.reg_read(RegisterARM64::TPIDR_EL0).unwrap_or(0);
    out.tpidrro = emu.reg_read(RegisterARM64::TPIDRRO_EL0).unwrap_or(0);
    for i in 0..32 {
        let b = emu.reg_read_long(qreg(i)).unwrap();
        out.v[i] = [
            u64::from_le_bytes(b[..8].try_into().unwrap()),
            u64::from_le_bytes(b[8..16].try_into().unwrap()),
        ];
    }
    out
}

thread_local! {
    static MEM: RefCell<BTreeMap<u64, Vec<u8>>> = const { RefCell::new(BTreeMap::new()) };
}

fn mem_get(addr: u64) -> Option<u8> {
    MEM.with(|m| m.borrow().get(&(addr & !0xFFF)).map(|p| p[(addr & 0xFFF) as usize]))
}

fn mem_set(addr: u64, b: u8) -> bool {
    MEM.with(|m| {
        m.borrow_mut()
            .get_mut(&(addr & !0xFFF))
            .map(|p| p[(addr & 0xFFF) as usize] = b)
            .is_some()
    })
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
    let mut buf = [0u8; 16];
    for i in 0..size as usize {
        match mem_get(addr + i as u64) {
            Some(b) => buf[i] = b,
            None => return fault(ctx, addr, size, false),
        }
    }
    unsafe {
        (*ctx).io_value = [
            u64::from_le_bytes(buf[..8].try_into().unwrap()),
            u64::from_le_bytes(buf[8..].try_into().unwrap()),
        ];
    }
}

unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    let io = unsafe { (*ctx).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    for i in 0..size as usize {
        if !mem_set(addr + i as u64, buf[i]) {
            return fault(ctx, addr, size, true);
        }
    }
}

unsafe extern "C" fn hk_atomic(ctx: *mut CpuContext, addr: u64, size: u8, op: u8) {
    use rustarmic::jit::atomic::{AtomicOp, apply};
    let n = size as usize;
    let mut buf = [0u8; 16];
    for i in 0..n {
        match mem_get(addr + i as u64) {
            Some(b) => buf[i] = b,
            None => return fault(ctx, addr, size, true),
        }
    }
    let to128 = |v: [u64; 2]| u128::from(v[0]) | (u128::from(v[1]) << 64);
    let old = u128::from_le_bytes(buf);
    let (operand, expected) = unsafe { (to128((*ctx).io_value), to128((*ctx).atomic_expected)) };
    if let Some(new) = apply(AtomicOp::from_u8(op).unwrap_or(AtomicOp::Cas), size, old, operand, expected) {
        for (i, b) in new.to_le_bytes()[..n].iter().enumerate() {
            mem_set(addr + i as u64, *b);
        }
    }
    unsafe { (*ctx).io_value = [old as u64, (old >> 64) as u64] };
}

struct Fetch;

impl Memory for Fetch {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = mem_get(addr + i as u64)?;
        }
        Some(u32::from_le_bytes(b))
    }
}

fn run_rustarmic(r: &Repro) -> State {
    MEM.with(|m| *m.borrow_mut() = pages(r));
    let mut jit = Jit::new(JitConfig::default()).expect("jit");
    let mut ctx = CpuContext::default();
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
    ctx.mem_atomic = hk_atomic;
    let s = &r.start;
    ctx.x = s.x;
    ctx.sp = s.sp;
    ctx.pc = s.pc;
    ctx.v = s.v;
    ctx.nzcv = s.nzcv;
    ctx.fpcr = s.fpcr;
    ctx.fpsr = s.fpsr;
    ctx.tpidr_el0 = s.tpidr;
    ctx.tpidrro_el0 = s.tpidrro;
    let mut retired = 0;
    while retired < r.insns {
        let out = jit.run_bounded(&mut ctx, &mut Fetch, r.insns - retired).expect("run");
        retired += out.retired;
        match out.stop {
            StopReason::BudgetExhausted => {}
            StopReason::Halted if out.retired > 0 => {}
            _ => break,
        }
    }
    State {
        x: ctx.x,
        sp: ctx.sp,
        pc: ctx.pc,
        v: ctx.v,
        nzcv: ctx.nzcv & 0xF,
        fpcr: ctx.fpcr,
        fpsr: ctx.fpsr,
        tpidr: ctx.tpidr_el0,
        tpidrro: ctx.tpidrro_el0,
    }
}

/// Runs a repro and prints which engines agree; returns
/// (unicorn vs dynarmic, rustarmic vs dynarmic, unicorn vs rustarmic) diffs.
fn replay(r: &Repro) -> (Vec<String>, Vec<String>, Vec<String>) {
    let unicorn = run_unicorn(r);
    let rust = run_rustarmic(r);
    let ud = unicorn.diff(&r.dynarmic);
    let rd = rust.diff(&r.dynarmic);
    let ur = unicorn.diff(&rust);
    eprintln!(
        "{} instructions from pc={:#x}\n  unicorn vs recorded dynarmic: {}\n  rustarmic vs recorded dynarmic: {}\n  unicorn vs rustarmic: {}\n  rustarmic now vs rustarmic recorded in the game: {}",
        r.insns,
        r.start.pc,
        summary(&ud),
        summary(&rd),
        summary(&ur),
        summary(&rust.diff(&r.rustarmic)),
    );
    (ud, rd, ur)
}

fn summary(d: &[String]) -> String {
    if d.is_empty() { "agree".into() } else { d.join("; ") }
}

#[test]
fn replay_file_from_env() {
    let Some(path) = std::env::var_os("RUSTARMIC_LOCKSTEP_REPRO") else {
        return;
    };
    let text = std::fs::read_to_string(&path).expect("read repro");
    let repro = parse(&text).expect("parse repro");
    replay(&repro);
}

/// A repro as NeXium writes it, for a planted miscompile of `add x7, x7,
/// #5`: Unicorn and an unmodified Rustarmic must both match Dynarmic.
#[test]
fn self_test_replays_a_repro() {
    let text = "\
# NeXium lockstep repro v1
class divergence
pc 0x100000
word 0x910014e7
insns 3
start.pc 0x100000
start.sp 0x208000
start.x7 0x64
start.x20 0x200000
start.nzcv 0x0
start.v1 0x3f800000 0x0
code 0x100000 e7140091870600f91f2003d500000014
data 0x200000 00000000000000000000000000000000
dyn.pc 0x10000c
dyn.sp 0x208000
dyn.x7 0x69
dyn.x20 0x200000
dyn.v1 0x3f800000 0x0
rust.pc 0x10000c
rust.sp 0x208000
rust.x7 0x169
rust.x20 0x200000
rust.v1 0x3f800000 0x0
";
    let repro = parse(text).unwrap();
    assert_eq!(repro.insns, 3);
    let (ud, rd, ur) = replay(&repro);
    assert!(ud.is_empty(), "unicorn disagrees with the recorded dynarmic state: {ud:?}");
    assert!(rd.is_empty(), "rustarmic disagrees with the recorded dynarmic state: {rd:?}");
    assert!(ur.is_empty());
    let planted = run_rustarmic(&repro).diff(&repro.rustarmic);
    assert!(planted.iter().any(|d| d.starts_with("x7")), "{planted:?}");
}
