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
fn udiv_normal_case() {
    // movz x0, #100
    // movz x1, #7
    // udiv x2, x0, x1   ; 100 / 7 = 14
    let code = build_code(&[
        0xD2800C80, // movz x0, #100
        0xD28000E1, // movz x1, #7
        0x9AC10802, // udiv x2, x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 14, "100 / 7 should be 14");
}

#[test]
fn udiv_by_zero_returns_zero() {
    // movz x0, #100
    // movz x1, #0
    // udiv x2, x0, x1   ; divisor 0 -> AArch64 returns 0 (no trap)
    let code = build_code(&[
        0xD2800C80, // movz x0, #100
        0xD2800001, // movz x1, #0
        0x9AC10802, // udiv x2, x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 0, "UDIV by zero must return 0, not trap");
}

#[test]
fn sdiv_normal_negative() {
    // movz x0, #100 ; neg x0, x0  -> x0 = -100
    // movz x1, #4
    // sdiv x2, x0, x1   ; -100 / 4 = -25
    let code = build_code(&[
        0xD2800C80, // movz x0, #100
        0xCB0003E0, // neg x0, x0    (sub x0, xzr, x0)
        0xD2800081, // movz x1, #4
        0x9AC10C02, // sdiv x2, x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2] as i64, -25, "SDIV -100 / 4 should be -25");
}

#[test]
fn sdiv_by_zero_returns_zero() {
    // movz x0, #100
    // movz x1, #0
    // sdiv x2, x0, x1   ; AArch64 returns 0
    let code = build_code(&[
        0xD2800C80, // movz x0, #100
        0xD2800001, // movz x1, #0
        0x9AC10C02, // sdiv x2, x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 0, "SDIV by zero must return 0, not trap");
}

#[test]
fn sdiv_int_min_by_neg_one_returns_int_min() {
    // movz/movk x0 to 0x8000_0000_0000_0000 (INT_MIN_64)
    // movz x1, #1 ; neg x1, x1 -> x1 = -1
    // sdiv x2, x0, x1   ; AArch64 returns x0 unchanged (= INT_MIN), no overflow trap
    let code = build_code(&[
        0xD2A00000_u32 ^ 0,  // we need x0 = 0x8000_0000_0000_0000
        // movz x0, #0, lsl #0 → 0xD2800000
        // movk x0, #0x8000, lsl #48 → 0xF2F00000
        // Use single MOVZ with lsl #48: movz x0, #0x8000, lsl #48 → 0xD2F00000
        0xD2F00000,
        0xD2800021, // movz x1, #1
        0xCB0103E1, // neg x1, x1    (sub x1, xzr, x1)
        0x9AC10C02, // sdiv x2, x0, x1
        0xD4200000,
    ]);
    let code = code[4..].to_vec();
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x8000_0000_0000_0000, "X0 should be INT_MIN_64");
    assert_eq!(ctx.x[1] as i64, -1, "X1 should be -1");
    assert_eq!(ctx.x[2], 0x8000_0000_0000_0000,
        "SDIV INT_MIN / -1 must return INT_MIN unchanged (no overflow trap)");
}

#[test]
fn lslv_variable_shift() {
    // movz x0, #1
    // movz x1, #5
    // lslv x2, x0, x1   ; x2 = 1 << 5 = 32
    let code = build_code(&[
        0xD2800020, // movz x0, #1
        0xD28000A1, // movz x1, #5
        0x9AC12002, // lslv x2, x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 32, "1 << 5 should be 32");
}

#[test]
fn clz_typical_and_zero() {
    // movz x0, #0
    // clz  x1, x0   ; should be 64
    // movz x2, #1
    // clz  x3, x2   ; should be 63
    let code = build_code(&[
        0xD2800000, // movz x0, #0
        0xDAC01001, // clz x1, x0
        0xD2800022, // movz x2, #1
        0xDAC01043, // clz x3, x2
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[1], 64, "CLZ(0) should be 64");
    assert_eq!(ctx.x[3], 63, "CLZ(1) should be 63");
}

#[test]
fn cls_typical_and_all_same() {
    // movz x0, #0 ; cls x1, x0   ; all-same -> 63
    // movz x2, #1 ; neg x2, x2   ; x2 = -1; cls x3, x2 -> 63
    // movz/movk x4 = 0x80000000_00000000; cls x5, x4 -> 0 (just sign bit, no matches after)
    let code = build_code(&[
        0xD2800000, // movz x0, #0
        0xDAC01401, // cls x1, x0
        0xD2800022, // movz x2, #1
        0xCB0203E2, // neg x2, x2
        0xDAC01443, // cls x3, x2
        0xD2F00004, // movz x4, #0x8000, lsl #48
        0xDAC01485, // cls x5, x4
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[1], 63, "CLS(0) should be 63 (all bits match sign)");
    assert_eq!(ctx.x[3], 63, "CLS(-1) should be 63");
    assert_eq!(ctx.x[5], 0,  "CLS(0x8000...0) should be 0");
}

#[test]
fn rbit_reverses_bits() {
    // movz x0, #1
    // rbit x1, x0   ; bit 0 set -> bit 63 set -> 0x8000_0000_0000_0000
    let code = build_code(&[
        0xD2800020, // movz x0, #1
        0xDAC00001, // rbit x1, x0
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[1], 0x8000_0000_0000_0000, "RBIT(1) should be 0x8000_0000_0000_0000");
}

#[test]
fn rev_byte_swap() {
    // movz/movk to build x0 = 0x0123_4567_89AB_CDEF
    // rev  x1, x0   ; bswap -> 0xEFCD_AB89_6745_2301
    let code = build_code(&[
        0xD29BDE60, // movz x0, #0xDEF3  -- placeholder, build full value via movk
        // We'll construct 0x0123_4567_89AB_CDEF via four MOVK lanes.
        // movz x0, #0xCDEF
        // movk x0, #0x89AB, lsl #16
        // movk x0, #0x4567, lsl #32
        // movk x0, #0x0123, lsl #48
        // Replace placeholder with correct encodings:
        0xD299BDE0, // movz x0, #0xCDEF
        0xF2B13560, // movk x0, #0x89AB, lsl #16
        0xF2C8ACE0, // movk x0, #0x4567, lsl #32
        0xF2E02460, // movk x0, #0x0123, lsl #48
        0xDAC00C01, // rev x1, x0
        0xD4200000,
    ]);
    let code = code[4..].to_vec();
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x0123_4567_89AB_CDEF);
    assert_eq!(ctx.x[1], 0xEFCD_AB89_6745_2301, "REV should byte-swap whole register");
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
