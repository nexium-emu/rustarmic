#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};
use std::sync::Mutex;

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x10_0000;

static MEM: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static SERIAL: Mutex<()> = Mutex::new(());

fn mem_init(size: usize) {
    let mut m = MEM.lock().unwrap();
    if m.len() < size {
        m.resize(size, 0);
    }
}

unsafe extern "C" fn hk_read(ctx: *mut CpuContext, addr: u64, size: u8) {
    let n = size as usize;
    let m = MEM.lock().unwrap();
    let off = (addr - DATA_BASE) as usize;
    let mut buf = [0u8; 16];
    if off + n <= m.len() {
        buf[..n].copy_from_slice(&m[off..off + n]);
    }
    let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
    let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
    unsafe {
        (*ctx).io_value = [lo, hi];
    }
}
unsafe extern "C" fn hk_write(ctx: *mut CpuContext, addr: u64, size: u8) {
    let n = size as usize;
    let io = unsafe { (*ctx).io_value };
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&io[0].to_le_bytes());
    buf[8..].copy_from_slice(&io[1].to_le_bytes());
    let mut m = MEM.lock().unwrap();
    let off = (addr - DATA_BASE) as usize;
    m[off..off + n].copy_from_slice(&buf[..n]);
}

fn install_hooks(ctx: &mut CpuContext) {
    ctx.mem_read = hk_read;
    ctx.mem_write = hk_write;
}

fn build_code(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

struct CodeMem {
    bytes: Vec<u8>,
    base: u64,
}
impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(self.base)? as usize;
        if off + 4 > self.bytes.len() {
            return None;
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[off..off + 4]);
        Some(u32::from_le_bytes(buf))
    }
}

fn run(code: &[u32], ctx: &mut CpuContext) -> ExitReason {
    install_hooks(ctx);
    let mut mem = CodeMem {
        bytes: build_code(code),
        base: CODE_BASE,
    };
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    jit.run(ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

const FIB_ITER: &[u32] = &[
    0x2A0003E2, 0x52800000, 0x52800021, 0x340000C2, 0x0B010003, 0x2A0103E0, 0x2A0303E1, 0x51000442,
    0x17FFFFFB, 0xD4200000,
];

fn run_fib(n: u32) -> u64 {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = n as u64;
    let exit = run(FIB_ITER, &mut ctx);
    assert!(
        matches!(exit, ExitReason::Brk(_)),
        "fib({n}): expected BRK, got {exit:?}"
    );
    ctx.x[0]
}

#[test]
fn fibonacci_iterative_small() {
    assert_eq!(run_fib(0), 0);
    assert_eq!(run_fib(1), 1);
    assert_eq!(run_fib(2), 1);
    assert_eq!(run_fib(3), 2);
    assert_eq!(run_fib(5), 5);
    assert_eq!(run_fib(10), 55);
}

#[test]
fn fibonacci_iterative_larger() {
    assert_eq!(run_fib(20), 6765);
    assert_eq!(run_fib(30), 832040);
    assert_eq!(run_fib(47), 2_971_215_073);
}

const SUM_ARRAY: &[u32] = &[
    0xD2800002, 0xB40000C1, 0xB9400003, 0x8B030042, 0x91001000, 0xD1000421, 0x17FFFFFB, 0xAA0203E0,
    0xD4200000,
];

#[test]
fn sum_small_array() {
    let _g = SERIAL.lock().unwrap();
    mem_init(0x1000);
    {
        let mut m = MEM.lock().unwrap();
        for (i, v) in [1u32, 2, 3, 4, 5].iter().enumerate() {
            m[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE;
    ctx.x[1] = 5;
    let exit = run(SUM_ARRAY, &mut ctx);
    assert!(
        matches!(exit, ExitReason::Brk(_)),
        "sum: expected BRK, got {exit:?}"
    );
    assert_eq!(ctx.x[0], 1 + 2 + 3 + 4 + 5);
}

#[test]
fn sum_thousand_array() {
    let _g = SERIAL.lock().unwrap();
    let n: usize = 1000;
    mem_init(n * 4);
    let expected: u64 = {
        let mut m = MEM.lock().unwrap();
        let mut s = 0u64;
        for i in 0..n {
            let v = (i as u32).wrapping_mul(7).wrapping_add(13);
            m[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            s += v as u64;
        }
        s
    };

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE;
    ctx.x[1] = n as u64;
    let exit = run(SUM_ARRAY, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)));
    assert_eq!(ctx.x[0], expected, "sum mismatch");
}

const FIB_TO_MEM: &[u32] = &[
    0x2A0003E2, 0x52800000, 0x52800021, 0xAA0603E3, 0x34000102, 0xB9000060, 0x91001063, 0x0B010004,
    0x2A0103E0, 0x2A0403E1, 0x51000442, 0x17FFFFF9, 0xD4200000,
];

#[test]
fn fib_sequence_to_memory() {
    let _g = SERIAL.lock().unwrap();
    mem_init(0x1000);
    {
        let mut m = MEM.lock().unwrap();
        for b in m.iter_mut() {
            *b = 0xAA;
        }
    }

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = 10;
    ctx.x[6] = DATA_BASE;
    let exit = run(FIB_TO_MEM, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "fib_mem: got {exit:?}");

    let expected: [u32; 10] = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34];
    let m = MEM.lock().unwrap();
    for (i, &want) in expected.iter().enumerate() {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&m[i * 4..i * 4 + 4]);
        let got = u32::from_le_bytes(buf);
        assert_eq!(got, want, "fib[{i}]: expected {want}, got {got}");
    }
}
