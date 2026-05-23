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
