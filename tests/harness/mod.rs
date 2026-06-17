use rustarmic::{CpuContext, Jit, JitConfig, Memory};
use std::sync::Mutex;
use unicorn_engine::{Arch, Mode, Prot, RegisterARM64, Unicorn};

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

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, addr: u64, size: u8) {
    let n = size as usize;
    let m = MEM.lock().unwrap();
    let off = mem_offset(addr);
    let mut buf = [0u8; 16];
    if off + n <= m.len() {
        buf[..n].copy_from_slice(&m[off..off + n]);
    }
    let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
    unsafe { (*ctx).io_value = [lo, hi]; }
}
unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    let n = size as usize;
    let io = unsafe { (*ctx).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    let mut m = MEM.lock().unwrap();
    let off = mem_offset(addr);
    if off + n <= m.len() {
        m[off..off + n].copy_from_slice(&buf[..n]);
    }
}

#[allow(dead_code)]
pub fn install_hooks(ctx: &mut CpuContext) {
    ctx.mem_read  = hk_read;
    ctx.mem_write = hk_write;
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
    pub v: [[u64; 2]; 32],
}

pub fn run_pair(code: &[u8], init: RegState) -> (RegState, RegState) {
    let uni_state = run_unicorn(code, init);
    let jit_state = run_rustarmic(code, init);
    (uni_state, jit_state)
}

fn run_unicorn(code: &[u8], init: RegState) -> RegState {
    let mut emu = Unicorn::new(Arch::ARM64, Mode::LITTLE_ENDIAN)
        .expect("unicorn init failed");

    emu.mem_map(CODE_BASE, CODE_SIZE, Prot::ALL).unwrap();
    emu.mem_write(CODE_BASE, code).unwrap();

    emu.mem_map(DATA_BASE, DATA_SIZE, Prot::ALL).unwrap();

    let cpacr = emu.reg_read(RegisterARM64::CPACR_EL1).unwrap_or(0);
    let _ = emu.reg_write(RegisterARM64::CPACR_EL1, cpacr | (0b11 << 20));

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

    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let mut ctx = CpuContext::default();
    install_hooks(&mut ctx);
    ctx.pc = init.pc;
    ctx.sp = init.sp;
    ctx.nzcv = init.nzcv;
    for i in 0..31 {
        ctx.x[i] = init.x[i];
    }
    for i in 0..32 {
        ctx.v[i] = init.v[i];
    }

    let _ = jit.run(&mut ctx, &mut mem);

    let mut out = RegState::default();
    for i in 0..31 { out.x[i] = ctx.x[i]; }
    out.sp = ctx.sp;
    out.pc = ctx.pc;
    out.nzcv = ctx.nzcv;
    for i in 0..32 { out.v[i] = ctx.v[i]; }
    out
}
