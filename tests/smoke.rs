//! Self-contained end-to-end smoke tests. No external engine required —
//! these compare JIT output against hand-computed expected register state,
//! which keeps the default `cargo test` runnable without libclang/LLVM.

#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;

fn build_code(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for w in words { v.extend_from_slice(&w.to_le_bytes()); }
    v
}

struct CodeMem {
    bytes: Vec<u8>,
    base:  u64,
}

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
fn movz_into_x0() {
    // movz x0, #0x1234
    // brk  #0
    let code = build_code(&[0xD282_4680, 0xD420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x1234, "X0 should be 0x1234 after MOVZ");
}

#[test]
fn add_imm_pipeline() {
    // movz x0, #100
    // add  x0, x0, #50
    // brk  #0
    let code = build_code(&[
        0xD280_0C80, // movz x0, #100
        0x9100_C800, // add  x0, x0, #50
        0xD420_0000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 150, "X0 should be 150 (100 + 50)");
}

#[test]
fn sub_imm_negative() {
    // movz x0, #10
    // sub  x0, x0, #15  -> wraps to -5 = 0xFFFFFFFFFFFFFFFB
    // brk  #0
    let code = build_code(&[
        0xD280_0140, // movz x0, #10
        0xD100_3C00, // sub  x0, x0, #15
        0xD420_0000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0] as i64, -5, "X0 should wrap to -5 after SUB");
}

#[test]
fn movz_into_x5_and_orr_reg() {
    // movz x0, #0xFF
    // movz x1, #0x0F
    // orr  x2, x0, x1
    // brk  #0
    let code = build_code(&[
        0xD2801FE0, // movz x0, #0xFF
        0xD28001E1, // movz x1, #0x0F
        0xAA010002, // orr  x2, x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "should hit BRK, got {:?}", exit);
    assert_eq!(ctx.x[0], 0xFF);
    assert_eq!(ctx.x[1], 0x0F);
    assert_eq!(ctx.x[2], 0xFF);
}

#[test]
fn ubfm_zero_extend_byte() {
    // movz x0, #0x1234ABCD low half; we just need a value with high bits set
    // movz x0, #0xCD; movz x1, #0xFF; and x2, x0, x1  is enough to test masking
    // Instead, exercise UBFM directly with a known constant.
    //
    // movz x0, #0xFFFF
    // ubfm x1, x0, #0, #7    ; extract bits [7:0] of x0 -> 0xFF
    // brk #0
    let code = build_code(&[
        0xD29FFFE0, // movz x0, #0xFFFF
        0xD3401C01, // ubfm x1, x0, #0, #7
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0xFFFF);
    assert_eq!(ctx.x[1], 0xFF, "UBFM should mask to low 8 bits");
}

#[test]
fn csel_picks_based_on_nzcv() {
    // movz x0, #100
    // movz x1, #200
    // subs xzr, x0, x1     ; sets flags: 100 - 200 negative, N=1
    // csel x2, x0, x1, mi  ; cond MI true -> x2 = x0 = 100
    // brk #0
    let code = build_code(&[
        0xD2800C80, // movz x0, #100
        0xD2801901, // movz x1, #200
        0xEB01001F, // subs xzr, x0, x1
        0x9A814002, // csel x2, x0, x1, mi
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 100, "CSEL with MI should pick X0");
}

#[test]
fn madd_three_operand() {
    // movz x0, #5
    // movz x1, #7
    // movz x2, #3
    // madd x3, x0, x1, x2   ; x3 = x2 + x0 * x1 = 3 + 5*7 = 38
    // brk #0
    let code = build_code(&[
        0xD28000A0, // movz x0, #5
        0xD28000E1, // movz x1, #7
        0xD2800062, // movz x2, #3
        0x9B010803, // madd x3, x0, x1, x2
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[3], 38, "MADD should compute Ra + Rn*Rm");
}

#[test]
fn add_sub_chain_uses_constant_folding() {
    // movz x0, #100
    // add  x1, x0, #1
    // add  x2, x1, #2
    // sub  x3, x2, #50
    // brk  #0
    let code = build_code(&[
        0xD2800C80, // movz x0, #100
        0x91000401, // add  x1, x0, #1
        0x91000822, // add  x2, x1, #2
        0xD100C843, // sub  x3, x2, #50
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 100);
    assert_eq!(ctx.x[1], 101);
    assert_eq!(ctx.x[2], 103);
    assert_eq!(ctx.x[3], 53);
}
