#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;

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

fn run_with(code: Vec<u8>, ctx: &mut CpuContext) -> ExitReason {
    let mut mem = CodeMem { bytes: code, base: CODE_BASE };
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    jit.run(ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

const MRS_X0_CNTPCT_EL0: u32 = 0xD53B_E020;
const MRS_X1_CNTPCT_EL0: u32 = 0xD53B_E021;
const MRS_X0_CNTVCT_EL0: u32 = 0xD53B_E040;
const MRS_X0_CNTFRQ_EL0: u32 = 0xD53B_E000;
const BRK_0:             u32 = 0xD420_0000;

#[test]
fn cntpct_default_is_nonzero_and_monotonic() {
    let code = build_code(&[MRS_X0_CNTPCT_EL0, MRS_X1_CNTPCT_EL0, BRK_0]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run_with(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "should hit BRK, got {:?}", exit);
    assert!(ctx.x[0] != 0, "CNTPCT_EL0 should not be stuck at zero");
    assert!(ctx.x[1] >= ctx.x[0], "CNTPCT_EL0 must be monotonic ({} -> {})", ctx.x[0], ctx.x[1]);
}

#[test]
fn cntvct_default_is_nonzero() {
    let code = build_code(&[MRS_X0_CNTVCT_EL0, BRK_0]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run_with(code, &mut ctx);
    assert!(ctx.x[0] != 0, "CNTVCT_EL0 should not be stuck at zero");
}

#[test]
fn cntfrq_reads_context_field() {
    let code = build_code(&[MRS_X0_CNTFRQ_EL0, BRK_0]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run_with(code, &mut ctx);
    assert_eq!(ctx.x[0], 19_200_000, "CNTFRQ_EL0 default should be the Switch rate (19.2 MHz)");
}

#[test]
fn cntfrq_honors_embedder_override() {
    let code = build_code(&[MRS_X0_CNTFRQ_EL0, BRK_0]);
    let mut ctx = CpuContext::default();
    ctx.cntfrq_el0 = 1_000_000_000;
    ctx.pc = CODE_BASE;
    run_with(code, &mut ctx);
    assert_eq!(ctx.x[0], 1_000_000_000, "embedder-set CNTFRQ_EL0 should be visible to MRS");
}

static mut TEST_COUNTER: u64 = 0xCAFE_F00D_0000_0000;

unsafe extern "C" fn test_read_cntpct(_ctx: *mut CpuContext) -> u64 {
    let v = unsafe { TEST_COUNTER };
    unsafe { TEST_COUNTER = TEST_COUNTER.wrapping_add(1); }
    v
}

#[test]
fn cntpct_honors_embedder_callback() {
    let code = build_code(&[MRS_X0_CNTPCT_EL0, MRS_X1_CNTPCT_EL0, BRK_0]);
    let mut ctx = CpuContext::default();
    ctx.read_cntpct = test_read_cntpct;
    ctx.pc = CODE_BASE;
    unsafe { TEST_COUNTER = 0xCAFE_F00D_0000_0000; }
    run_with(code, &mut ctx);
    assert_eq!(ctx.x[0], 0xCAFE_F00D_0000_0000, "first MRS should see initial counter");
    assert_eq!(ctx.x[1], 0xCAFE_F00D_0000_0001, "second MRS should see incremented counter");
}
