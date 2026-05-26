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
fn ccmp_imm_failed_cond_uses_nzcv_imm() {
    // movz x0, #1
    // movz x1, #2
    // subs xzr, x0, x1     ; 1-2=-1 -> N=1,Z=0,C=0,V=0 (EQ fails)
    // ccmp x0, #5, #0xA, eq ; cond EQ fails -> NZCV <- imm = 0xA = 0b1010 (N=1,Z=0,C=1,V=0)
    // csinc x2, xzr, xzr, mi; cond MI (N=1) holds -> x2 = xzr = 0
    let code = build_code(&[
        0xD2800020, // movz x0, #1
        0xD2800041, // movz x1, #2
        0xEB01001F, // subs xzr, x0, x1
        0xFA45080A, // ccmp x0, #5, #0xA, eq
        0x9A9F47E2, // csinc x2, xzr, xzr, mi
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv & 0xF, 0xA, "After CCMP-fail, NZCV should equal imm nibble 0xA");
    assert_eq!(ctx.x[2], 0, "csinc on MI(N=1) should pick xzr (0)");
}

#[test]
fn ccmp_imm_passed_cond_does_compare() {
    // movz x0, #5 ; movz x1, #5
    // subs xzr, x0, x1     ; Z=1 (EQ true)
    // ccmp x0, #5, #0xF, eq ; EQ holds -> do compare 5-5=0 -> N=0,Z=1,C=1,V=0 = nibble 6
    // csinc x2, xzr, xzr, ne; NE fails -> x2 = xzr+1 = 1
    let code = build_code(&[
        0xD28000A0, // movz x0, #5
        0xD28000A1, // movz x1, #5
        0xEB01001F, // subs xzr, x0, x1
        0xFA45080F, // ccmp x0, #5, #0xF, eq
        0x9A9F17E2, // csinc x2, xzr, xzr, ne
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv & 0xF, 0b0110, "After CCMP-pass, NZCV should be compare-result (Z=1,C=1)");
    assert_eq!(ctx.x[2], 1, "csinc on NE-fail should pick xzr+1 = 1");
}

#[test]
fn adc_carries_from_subs() {
    // movz x0, #10 ; movz x1, #5
    // subs xzr, x0, x1   ; 10 - 5 = 5, no borrow -> C=1
    // movz x2, #100 ; movz x3, #1
    // adc  x4, x2, x3    ; x4 = 100 + 1 + C(=1) = 102
    let code = build_code(&[
        0xD2800140, // movz x0, #10
        0xD28000A1, // movz x1, #5
        0xEB01001F, // subs xzr, x0, x1
        0xD2800C82, // movz x2, #100
        0xD2800023, // movz x3, #1
        0x9A030044, // adc x4, x2, x3
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[4], 102, "ADC should add 100 + 1 + carry(=1) = 102");
}

#[test]
fn adc_no_carry_from_subs() {
    // movz x0, #5 ; movz x1, #10
    // subs xzr, x0, x1   ; 5 - 10 underflows -> C=0
    // adc x4, x2, x3 with x2=100, x3=1 -> 100 + 1 + 0 = 101
    let code = build_code(&[
        0xD28000A0, // movz x0, #5
        0xD2800141, // movz x1, #10
        0xEB01001F, // subs xzr, x0, x1
        0xD2800C82, // movz x2, #100
        0xD2800023, // movz x3, #1
        0x9A030044, // adc x4, x2, x3
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[4], 101, "ADC without carry should be 100 + 1 = 101");
}

#[test]
fn two_block_direct_branch_chains() {
    // Block A at 0x1000:
    //   movz x0, #1         (PC 0x1000)
    //   b 0x1100            (PC 0x1004, offset +0xFC bytes = +63 words = imm26 0x3F)
    //
    // Block B at 0x1100:
    //   movz x1, #5
    //   add  x1, x1, x0     ; x1 = 5 + 1 = 6
    //   brk  #0
    //
    // First run translates A (patch site registered targeting 0x1100, not compiled).
    // Falls back through chainable terminator: mov rax, 0x1100; ret.
    // Dispatcher loops, translates B at 0x1100, install applies pending patch on A's
    // terminator (rewrites the 5-byte jmp placeholder to point at B's host code).
    // B executes, hits BRK.
    //
    // The visible result is the same with or without chaining — this test confirms
    // the chainable-terminator format doesn't break correctness.

    let mut code = vec![0u8; 0x10C];
    code[0..4].copy_from_slice(&0xD2800020u32.to_le_bytes());   // movz x0, #1
    code[4..8].copy_from_slice(&0x1400003Fu32.to_le_bytes());   // b 0x1100
    code[0x100..0x104].copy_from_slice(&0xD28000A1u32.to_le_bytes()); // movz x1, #5
    code[0x104..0x108].copy_from_slice(&0x8B000021u32.to_le_bytes()); // add x1, x1, x0
    code[0x108..0x10C].copy_from_slice(&0xD4200000u32.to_le_bytes()); // brk #0

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK exit, got {:?}", exit);
    assert_eq!(ctx.x[0], 1, "X0 from block A");
    assert_eq!(ctx.x[1], 6, "X1 = 5 + 1 from block B (after branch from A)");
}

#[test]
fn cbnz_not_taken_falls_through() {
    // movz x0, #0
    // cbnz x0, +0xFC   ; x0==0 so NOT taken; fall through to brk
    // brk #0
    let code = build_code(&[
        0xD2800000, // movz x0, #0
        0xB50007E0, // cbnz x0, +0xFC (target 0x1100, not taken)
        0xD4200000, // brk #0
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
}

#[test]
fn cbnz_loop_chains() {
    // Two-block loop using CBNZ — exercises the taken side of the conditional
    // chain (backward branch to .loop).
    //
    //   movz x0, #5           ; counter
    //   movz x1, #0           ; accumulator
    //   .loop:
    //   add  x1, x1, x0       ; accum += counter
    //   sub  x0, x0, #1       ; counter--
    //   cbnz x0, .loop        ; if counter != 0, loop
    //   brk  #0
    let code = build_code(&[
        0xD28000A0, // movz x0, #5             (PC 0x1000)
        0xD2800001, // movz x1, #0             (PC 0x1004)
        0x8B000021, // add x1, x1, x0          (PC 0x1008)  ; .loop
        0xD1000400, // sub x0, x0, #1          (PC 0x100C)
        0xB5FFFFC0, // cbnz x0, .loop          (PC 0x1010)  ; imm19 = -2 words
        0xD4200000, // brk #0                  (PC 0x1014)
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(matches!(exit, ExitReason::Brk(_)), "expected BRK, got {:?}", exit);
    assert_eq!(ctx.x[0], 0,  "counter ends at 0");
    assert_eq!(ctx.x[1], 15, "accumulator = 5+4+3+2+1 = 15");
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

#[test]
fn msr_then_mrs_tpidr_el0_round_trip() {
    // movz x0, #0xCAFE
    // msr tpidr_el0, x0
    // mrs x1, tpidr_el0
    // brk #0
    let code = build_code(&[
        0xD299_5FC0, // movz x0, #0xCAFE
        0xD51B_D040, // msr tpidr_el0, x0
        0xD53B_D041, // mrs x1, tpidr_el0
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.tpidr_el0, 0xCAFE, "MSR should write tpidr_el0 slot");
    assert_eq!(ctx.x[1], 0xCAFE, "MRS should round-trip the value");
}

#[test]
fn pacia_then_autia_round_trips_pointer() {
    // movz x0, #0xABCD
    // pacia x0, x1     ; identity in our model
    // autia x0, x1     ; identity in our model
    // brk #0
    let code = build_code(&[
        0xD295_79A0, // movz x0, #0xABCD
        0xDAC1_0020, // pacia x0, x1
        0xDAC1_1020, // autia x0, x1
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0xABCD, "PACIA+AUTIA should be identity in our model");
}

#[test]
fn fmov_v_to_v_single_precision() {
    // fmov s1, s2  ; copy low 32 bits of v2 to v1, zero rest
    let code = build_code(&[
        0x1E20_4041,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[2] = [0x1122_3344_5566_7788, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    run(code, &mut ctx);
    assert_eq!(ctx.v[1][0], 0x5566_7788, "S-precision FMOV copies low 32 bits");
    assert_eq!(ctx.v[1][1], 0, "S-precision FMOV zeros upper 96 bits");
}

#[test]
fn fmov_v_to_v_double_precision() {
    // fmov d1, d2  ; copy low 64 bits of v2 to v1, zero upper 64 bits
    let code = build_code(&[
        0x1E60_4041,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[2] = [0x1122_3344_5566_7788, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    run(code, &mut ctx);
    assert_eq!(ctx.v[1][0], 0x1122_3344_5566_7788, "D FMOV copies low 64 bits");
    assert_eq!(ctx.v[1][1], 0, "D FMOV zeros upper 64 bits");
}

#[test]
fn fadd_d_two_doubles() {
    // fadd d2, d0, d1   ; 1.5 + 2.25 = 3.75
    let code = build_code(&[
        0x1E61_2802,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(1.5_f64).to_bits(), 0];
    ctx.v[1] = [(2.25_f64).to_bits(), 0];
    run(code, &mut ctx);
    let result = f64::from_bits(ctx.v[2][0]);
    assert_eq!(result, 3.75, "FADD D should add doubles");
    assert_eq!(ctx.v[2][1], 0, "FADD D zeros upper 64 bits");
}

#[test]
fn fmul_s_two_floats() {
    // fmul s2, s0, s1   ; 1.5 * 2.0 = 3.0
    let code = build_code(&[
        0x1E21_0802,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(1.5_f32).to_bits() as u64, 0];
    ctx.v[1] = [(2.0_f32).to_bits() as u64, 0];
    run(code, &mut ctx);
    let bits = ctx.v[2][0] as u32;
    let result = f32::from_bits(bits);
    assert_eq!(result, 3.0, "FMUL S should multiply floats");
    assert_eq!(ctx.v[2][1], 0, "FMUL S zeros upper 96 bits");
    assert_eq!(ctx.v[2][0] >> 32, 0, "and the upper 32 bits of lane 0");
}

#[test]
fn fcmp_d_less_sets_n_flag() {
    // fcmp d0, d1     ; 1.0 vs 2.0 → less → NZCV = 1000 (N=1)
    // brk
    let code = build_code(&[
        0x1E61_2000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(1.0_f64).to_bits(), 0];
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b1000, "1.0 < 2.0 should set N only");
}

#[test]
fn fcmp_d_equal_sets_z_and_c() {
    // fcmp d0, d1   ; 1.5 == 1.5 → NZCV = 0110 (Z=1, C=1)
    let code = build_code(&[
        0x1E61_2000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(1.5_f64).to_bits(), 0];
    ctx.v[1] = [(1.5_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0110, "equal should set Z and C");
}

#[test]
fn fcmp_d_greater_sets_c_only() {
    // fcmp d0, d1   ; 3.0 > 1.0 → NZCV = 0010 (C=1)
    let code = build_code(&[
        0x1E61_2000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(3.0_f64).to_bits(), 0];
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0010, "greater should set C only");
}

#[test]
fn fcmp_d_nan_sets_c_and_v() {
    // fcmp d0, d1   ; NaN vs 1.0 → unordered → NZCV = 0011 (C=1, V=1)
    let code = build_code(&[
        0x1E61_2000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [f64::NAN.to_bits(), 0];
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0011, "unordered (NaN) should set C and V");
}

#[test]
fn fcmp_d_against_zero_immediate() {
    // fcmp d0, #0.0   ; 5.0 > 0 → NZCV = 0010 (C=1)
    let code = build_code(&[
        0x1E60_2008,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(5.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0010, "5.0 > 0 should set C");
}

#[test]
fn fcsel_d_picks_taken_when_eq() {
    // movz x0, ... ; load 7.5 into d0, 3.5 into d1
    // fcmp d2, d3   ; if eq (we make them eq) → fcsel picks d0
    // fcsel d4, d0, d1, eq
    // brk
    let code = build_code(&[
        0x1E63_2040, // fcmp d2, d3
        0x1E61_0C04, // fcsel d4, d0, d1, eq
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(7.5_f64).to_bits(), 0];
    ctx.v[1] = [(3.5_f64).to_bits(), 0];
    ctx.v[2] = [(1.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[4][0]), 7.5, "EQ → fcsel picks Fn (d0)");
}

#[test]
fn fcsel_d_picks_not_taken_when_ne() {
    let code = build_code(&[
        0x1E63_2040, // fcmp d2, d3
        0x1E61_1C04, // fcsel d4, d0, d1, ne  (cond=0001)
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(7.5_f64).to_bits(), 0];
    ctx.v[1] = [(3.5_f64).to_bits(), 0];
    ctx.v[2] = [(1.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];  // 1.0 == 1.0 → NE fails
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[4][0]), 3.5, "NE fails → fcsel picks Fm (d1)");
}

#[test]
fn fmov_d_immediate_loads_1_0() {
    // fmov d0, #1.0   (imm8 = 0x70 in VFPExpandImm)
    let code = build_code(&[
        0x1E6E_1000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xDEAD_BEEF_DEAD_BEEF, 0xDEAD_BEEF_DEAD_BEEF];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 1.0);
    assert_eq!(ctx.v[0][1], 0, "high lane zeroed");
}

#[test]
fn fmov_d_immediate_loads_2_0() {
    // fmov d0, #2.0   (imm8 = 0x00)
    let code = build_code(&[
        0x1E60_1000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 2.0);
}

#[test]
fn fmov_s_immediate_loads_1_0() {
    // fmov s0, #1.0    (ptype = 00, imm8 = 0x70)
    // 0001 1110 0010 1110 0001 0000 0000 0000
    let code = build_code(&[
        0x1E2E_1000,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    let bits = ctx.v[0][0] as u32;
    assert_eq!(f32::from_bits(bits), 1.0);
    assert_eq!(ctx.v[0][0] >> 32, 0, "upper 32 of lane 0 zeroed");
    assert_eq!(ctx.v[0][1], 0);
}

#[test]
fn fneg_d_flips_sign_bit() {
    // fneg d0, d1   ; v[0] = -v[1]
    let code = build_code(&[
        0x1E61_4020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.5_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -3.5);
}

#[test]
fn fabs_d_clears_sign_bit() {
    // fabs d0, d1   ; v[0] = |v[1]|
    let code = build_code(&[
        0x1E60_C020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(-7.25_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 7.25);
}

#[test]
fn fsqrt_d_computes_square_root() {
    // fsqrt d0, d1
    let code = build_code(&[
        0x1E61_C020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(4.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 2.0);
}

#[test]
fn fmadd_d_computes_a_plus_n_times_m() {
    // fmadd d0, d1, d2, d3   ; D0 = D3 + D1*D2 = 1 + 2*3 = 7
    let code = build_code(&[
        0x1F42_0C20,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    ctx.v[2] = [(3.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 7.0);
}

#[test]
fn fmsub_d_computes_a_minus_n_times_m() {
    // fmsub d0, d1, d2, d3   ; D0 = D3 - D1*D2 = 1 - 6 = -5
    let code = build_code(&[
        0x1F42_8C20,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    ctx.v[2] = [(3.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -5.0);
}

#[test]
fn fnmadd_d_computes_neg_a_minus_n_times_m() {
    // fnmadd d0, d1, d2, d3  ; D0 = -D3 - D1*D2 = -1 - 6 = -7
    let code = build_code(&[
        0x1F62_0C20,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    ctx.v[2] = [(3.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -7.0);
}

#[test]
fn fnmsub_d_computes_neg_a_plus_n_times_m() {
    // fnmsub d0, d1, d2, d3  ; D0 = -D3 + D1*D2 = -1 + 6 = 5
    let code = build_code(&[
        0x1F62_8C20,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    ctx.v[2] = [(3.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 5.0);
}

#[test]
fn fcvtzs_w_from_double_truncates() {
    // fcvtzs w0, d1   ; floor(3.75) = 3
    let code = build_code(&[
        0x1E78_0020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.75_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.x[0] as i32, 3, "FCVTZS truncates toward zero");
}

#[test]
fn fcvtzs_x_from_double_negative() {
    // fcvtzs x0, d1   ; truncate -3.75 → -3
    let code = build_code(&[
        0x9E78_0020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(-3.75_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.x[0] as i64, -3, "FCVTZS truncates toward zero, not floor");
}

#[test]
fn scvtf_d_from_x_signed_int() {
    // scvtf d0, x1   ; -42 → -42.0
    let code = build_code(&[
        0x9E62_0020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = (-42_i64) as u64;
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -42.0);
}

#[test]
fn fmov_d_from_x_copies_bits() {
    // fmov d0, x1   ; raw bit copy
    let code = build_code(&[
        0x9E67_0020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = (1.5_f64).to_bits();
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 1.5);
    assert_eq!(ctx.v[0][1], 0, "high lane zeroed");
}

#[test]
fn fmov_x_from_d_copies_bits() {
    // fmov x0, d1   ; raw bit copy back
    let code = build_code(&[
        0x9E66_0020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.5_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.x[0]), 2.5);
}

#[test]
fn fcvt_s_to_d_promotes_float() {
    // fcvt d0, s1
    let code = build_code(&[
        0x1E22_C020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(1.5_f32).to_bits() as u64, 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 1.5_f64);
}

#[test]
fn fcvt_d_to_s_demotes_double() {
    // fcvt s0, d1
    let code = build_code(&[
        0x1E62_4020,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.25_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f32::from_bits(ctx.v[0][0] as u32), 2.25_f32);
    assert_eq!(ctx.v[0][0] >> 32, 0);
}

#[test]
fn fmax_d_picks_larger() {
    // fmax d0, d1, d2   ; max(3.0, 5.0) = 5.0
    let code = build_code(&[
        0x1E62_4820,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.0_f64).to_bits(), 0];
    ctx.v[2] = [(5.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 5.0);
}

#[test]
fn fmin_d_picks_smaller() {
    // fmin d0, d1, d2   ; min(3.0, 5.0) = 3.0
    let code = build_code(&[
        0x1E62_5820,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.0_f64).to_bits(), 0];
    ctx.v[2] = [(5.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 3.0);
}

#[test]
fn fnmul_d_negates_product() {
    // fnmul d0, d1, d2   ; -(2.0 * 3.0) = -6.0
    let code = build_code(&[
        0x1E62_8820,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    ctx.v[2] = [(3.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -6.0);
}

#[test]
fn fccmp_d_cond_holds_runs_fcmp() {
    // fccmp d1, d2, #0, eq
    //   pre-NZCV = 0100 (Z=1) → cond EQ holds → FCMP runs: 1.0 < 2.0 → NZCV=1000
    let code = build_code(&[
        0x1E62_0420,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.nzcv = 0b0100; // Z=1
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    ctx.v[2] = [(2.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b1000, "cond held → FCMP set N");
}

#[test]
fn fccmp_d_cond_fails_uses_immediate_nzcv() {
    // fccmp d1, d2, #0xA, eq  (imm4 = 0xA = 1010)
    //   pre-NZCV = 0000 → EQ fails → final NZCV = imm4 = 1010
    let code = build_code(&[
        0x1E62_042A,  // imm4=0xA in low nibble
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.nzcv = 0;
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    ctx.v[2] = [(2.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b1010, "cond failed → NZCV = imm4");
}

#[test]
fn mul_fold_collapses_chain_to_single_mul() {
    // movz x0, #5         ; a = 5
    // movz x4, #3         ; c = 3
    // mul  x1, x0, x4     ; x1 = 5 * 3 = 15
    // lsl  x2, x1, #2     ; x2 = 60     (= (c*a) << b)
    // add  x3, x2, x0     ; x3 = 65     (= a * 13)
    let code = build_code(&[
        0xD28000A0, // movz x0, #5
        0xD2800064, // movz x4, #3
        0x9B04_7C01, // mul x1, x0, x4
        0xD37E_F422, // lsl x2, x1, #2
        0x8B00_0043, // add x3, x2, x0
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[3], 65, "((c*a)<<b)+a must compute correctly");
}

#[test]
fn vec_add_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1 = 16 bytes 0x01..0x10; V2 = 16 bytes all 0x10
    ctx.v[1] = [0x0807_0605_0403_0201, 0x100F_0E0D_0C0B_0A09];
    ctx.v[2] = [0x1010_1010_1010_1010, 0x1010_1010_1010_1010];
    let code = build_code(&[
        0x4E22_8420, // add v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x1817_1615_1413_1211);
    assert_eq!(ctx.v[0][1], 0x201F_1E1D_1C1B_1A19);
}

#[test]
fn vec_add_8h_wraps_per_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Each 16-bit lane: 0xFFFF + 1 = 0x0000 (wraps within lane).
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.v[2] = [0x0001_0001_0001_0001, 0x0001_0001_0001_0001];
    let code = build_code(&[
        0x4E62_8420, // add v0.8h, v1.8h, v2.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0], [0, 0], "every 16-bit lane wraps independently");
}

#[test]
fn vec_add_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0002_0000_0001, 0x0000_0004_0000_0003];
    ctx.v[2] = [0x0000_000A_0000_000A, 0x0000_000A_0000_000A];
    let code = build_code(&[
        0x4EA2_8420, // add v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_000C_0000_000B);
    assert_eq!(ctx.v[0][1], 0x0000_000E_0000_000D);
}

#[test]
fn vec_add_2d() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    ctx.v[2] = [0x1000_0000_0000_0001, 0x0000_0000_0000_0003];
    let code = build_code(&[
        0x4EE2_8420, // add v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x2111_1111_1111_1112);
    assert_eq!(ctx.v[0][1], 0x2222_2222_2222_2225);
}

#[test]
fn vec_add_8b_zeros_upper_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0807_0605_0403_0201, 0xDEAD_BEEF_CAFE_BABE];
    ctx.v[2] = [0x0101_0101_0101_0101, 0x9999_9999_9999_9999];
    let code = build_code(&[
        0x0E22_8420, // add v0.8b, v1.8b, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0908_0706_0504_0302, "low half = lanewise add");
    assert_eq!(ctx.v[0][1], 0, "upper 64 bits must be zeroed for 8B form");
}

#[test]
fn vec_sub_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0014_0000_0014, 0x0000_0014_0000_0014];
    ctx.v[2] = [0x0000_0004_0000_0001, 0x0000_0006_0000_0005];
    let code = build_code(&[
        0x6EA2_8420, // sub v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0010_0000_0013);
    assert_eq!(ctx.v[0][1], 0x0000_000E_0000_000F);
}

#[test]
fn vec_and_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xFF00_FF00_FF00_FF00, 0xAAAA_AAAA_AAAA_AAAA];
    ctx.v[2] = [0x0FF0_0FF0_0FF0_0FF0, 0x5555_5555_5555_5555];
    let code = build_code(&[
        0x4E22_1C20, // and v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0F00_0F00_0F00_0F00);
    assert_eq!(ctx.v[0][1], 0x0000_0000_0000_0000);
}

#[test]
fn vec_eor_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xAAAA_AAAA_AAAA_AAAA, 0xFFFF_0000_FFFF_0000];
    ctx.v[2] = [0x5555_5555_5555_5555, 0xFFFF_FFFF_0000_0000];
    let code = build_code(&[
        0x6E22_1C20, // eor v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0x0000_FFFF_FFFF_0000);
}

#[test]
fn vec_bic_clears_bits_per_mask() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // BIC Vd, Vn, Vm  →  Vd = Vn AND NOT Vm
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xAAAA_AAAA_AAAA_AAAA];
    ctx.v[2] = [0x00FF_00FF_00FF_00FF, 0x000F_000F_000F_000F];
    let code = build_code(&[
        0x4E62_1C20, // bic v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFF00_FF00_FF00_FF00);
    assert_eq!(ctx.v[0][1], 0xAAA0_AAA0_AAA0_AAA0);
}

#[test]
fn vec_orn_or_inverted() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // ORN Vd, Vn, Vm  →  Vd = Vn OR NOT Vm
    ctx.v[1] = [0x0000_0000_0000_0000, 0x0F0F_0F0F_0F0F_0F0F];
    ctx.v[2] = [0x00FF_00FF_00FF_00FF, 0xFFFF_FFFF_FFFF_FFFF];
    let code = build_code(&[
        0x4EE2_1C20, // orn v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // ~v2[0] = 0xFF00_FF00_FF00_FF00; OR v1[0]=0  → 0xFF00FF00FF00FF00
    assert_eq!(ctx.v[0][0], 0xFF00_FF00_FF00_FF00);
    // ~v2[1] = 0; OR v1[1] = 0x0F0F... → 0x0F0F0F0F0F0F0F0F
    assert_eq!(ctx.v[0][1], 0x0F0F_0F0F_0F0F_0F0F);
}

#[test]
fn vec_neg_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0001_FFFF_FFFE, 0x8000_0000_7FFF_FFFF];
    let code = build_code(&[
        0x6EA0_B820, // neg v0.4s, v1.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Per-lane two's complement:
    //   0x00000001 -> 0xFFFFFFFF
    //   0xFFFFFFFE -> 0x00000002
    //   0x7FFFFFFF -> 0x80000001
    //   0x80000000 -> 0x80000000 (wraps on its own representable bound)
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_0000_0002);
    assert_eq!(ctx.v[0][1], 0x8000_0000_8000_0001);
}

#[test]
fn vec_abs_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0001_FFFF_FFFE, 0x8000_0001_7FFF_FFFF];
    let code = build_code(&[
        0x4EA0_B820, // abs v0.4s, v1.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // |0x00000001|=1, |0xFFFFFFFE| = |-2| = 2, |0x7FFFFFFF|=0x7FFFFFFF,
    // |0x80000001|= 0x7FFFFFFF
    assert_eq!(ctx.v[0][0], 0x0000_0001_0000_0002);
    assert_eq!(ctx.v[0][1], 0x7FFF_FFFF_7FFF_FFFF);
}

#[test]
fn vec_not_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xF0F0_F0F0_F0F0_F0F0, 0x0123_4567_89AB_CDEF];
    let code = build_code(&[
        0x6E20_5820, // not v0.16b, v1.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], !0xF0F0_F0F0_F0F0_F0F0);
    assert_eq!(ctx.v[0][1], !0x0123_4567_89AB_CDEF);
}

#[test]
fn vec_mul_8h() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Lane-by-lane H multiply: low 16 of (a*b)
    ctx.v[1] = [0x0003_0002_0001_0000, 0x0007_0006_0005_0004];
    ctx.v[2] = [0x0010_0010_0010_0010, 0x0010_0010_0010_0010];
    let code = build_code(&[
        0x4E62_9C20, // mul v0.8h, v1.8h, v2.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0030_0020_0010_0000);
    assert_eq!(ctx.v[0][1], 0x0070_0060_0050_0040);
}

#[test]
fn vec_mul_4s_wraps_within_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0002_FFFF_FFFF, 0x0000_0003_0000_0004];
    ctx.v[2] = [0x0000_0003_0000_0002, 0x0000_0005_0000_0010];
    let code = build_code(&[
        0x4EA2_9C20, // mul v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 0xFFFFFFFF * 2 = 0x1_FFFFFFFE -> low 32 = 0xFFFFFFFE
    // 0x2 * 0x3 = 6
    // 0x4 * 0x10 = 0x40
    // 0x3 * 0x5 = 0xF
    assert_eq!(ctx.v[0][0], 0x0000_0006_FFFF_FFFE);
    assert_eq!(ctx.v[0][1], 0x0000_000F_0000_0040);
}

#[test]
fn vec_shl_imm_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0001_0000_00FF, 0x0000_0010_0000_0100];
    let code = build_code(&[
        0x4F22_5420, // shl v0.4s, v1.4s, #2
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0004_0000_03FC);
    assert_eq!(ctx.v[0][1], 0x0000_0040_0000_0400);
}

#[test]
fn vec_ushr_imm_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_FFFF_8000_0000, 0xF000_0000_0000_0010];
    let code = build_code(&[
        0x6F3E_0420, // ushr v0.4s, v1.4s, #2
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Each 32-bit lane >> 2 unsigned:
    //   0x80000000 >> 2 = 0x20000000
    //   0x0000FFFF >> 2 = 0x00003FFF
    //   0x00000010 >> 2 = 0x00000004
    //   0xF0000000 >> 2 = 0x3C000000
    assert_eq!(ctx.v[0][0], 0x0000_3FFF_2000_0000);
    assert_eq!(ctx.v[0][1], 0x3C00_0000_0000_0004);
}

#[test]
fn vec_sshr_imm_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_FFFF_8000_0000, 0xF000_0000_0000_0010];
    let code = build_code(&[
        0x4F3E_0420, // sshr v0.4s, v1.4s, #2
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Arithmetic >> 2:
    //   0x80000000 -> 0xE0000000 (sign-extended)
    //   0x0000FFFF -> 0x00003FFF
    //   0x00000010 -> 0x00000004
    //   0xF0000000 -> 0xFC000000
    assert_eq!(ctx.v[0][0], 0x0000_3FFF_E000_0000);
    assert_eq!(ctx.v[0][1], 0xFC00_0000_0000_0004);
}

#[test]
fn vec_cmeq_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0002_0000_0001, 0x0000_0004_0000_0003];
    ctx.v[2] = [0x0000_0099_0000_0001, 0x0000_0004_0000_0099];
    let code = build_code(&[
        0x6EA2_8C20, // cmeq v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Lane equality: lane0=eq (1==1), lane1=ne, lane2=ne, lane3=eq (4==4)
    assert_eq!(ctx.v[0][0], 0x0000_0000_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn vec_cmgt_signed_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFD, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[
        0x4EA2_3420, // cmgt v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Signed compare:
    //   lane0: -2  >s -3  → true
    //   lane1:  5  >s  3  → true
    //   lane2: 0x80000000 (= -2147483648) >s -1 → false
    //   lane3: 0x7FFFFFFF (max int) >s 0 → true
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn vec_cmge_signed_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0003, 0xFFFF_FFFE_FFFF_FFFE];
    ctx.v[2] = [0x0000_0005_0000_0004, 0xFFFF_FFFE_FFFF_FFFD];
    let code = build_code(&[
        0x4EA2_3C20, // cmge v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // lane0: 3 >= 4 → false
    // lane1: 5 >= 5 → true
    // lane2: -2 >= -3 → true
    // lane3: -2 >= -2 → true
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_0000_0000);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn vec_cmhi_unsigned_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFD, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[
        0x6EA2_3420, // cmhi v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Unsigned compare:
    //   lane0: 0xFFFFFFFE  >u 0xFFFFFFFD → true
    //   lane1: 5 >u 3 → true
    //   lane2: 0x80000000 >u 0xFFFFFFFF → false (0x80000000 < 0xFFFFFFFF unsigned)
    //   lane3: 0x7FFFFFFF >u 0 → true
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn vec_cmhs_unsigned_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0003, 0x8000_0000_FFFF_FFFF];
    ctx.v[2] = [0x0000_0005_0000_0004, 0xFFFF_FFFF_FFFF_FFFF];
    let code = build_code(&[
        0x6EA2_3C20, // cmhs v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Unsigned >=:
    //   lane0: 3 >=u 4 → false
    //   lane1: 5 >=u 5 → true
    //   lane2: 0xFFFFFFFF >=u 0xFFFFFFFF → true
    //   lane3: 0x80000000 >=u 0xFFFFFFFF → false
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_0000_0000);
    assert_eq!(ctx.v[0][1], 0x0000_0000_FFFF_FFFF);
}

#[test]
fn vec_bit_inserts_when_mask_set() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // BIT Vd, Vn, Vm  →  Vd = (Vd & ~Vm) | (Vn & Vm)
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB]; // initial Vd
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_DEAD_BEEF]; // Vn (insertion source)
    ctx.v[2] = [0xFF00_FF00_FF00_FF00, 0x0000_FFFF_FFFF_0000]; // Vm (mask)
    let code = build_code(&[
        0x6EA2_1C20, // bit v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let exp0 = (0xAAAA_AAAA_AAAA_AAAAu64 & !0xFF00_FF00_FF00_FF00) | (0x1234_5678_9ABC_DEF0 & 0xFF00_FF00_FF00_FF00);
    let exp1 = (0xBBBB_BBBB_BBBB_BBBBu64 & !0x0000_FFFF_FFFF_0000) | (0xCAFE_BABE_DEAD_BEEF & 0x0000_FFFF_FFFF_0000);
    assert_eq!(ctx.v[0][0], exp0);
    assert_eq!(ctx.v[0][1], exp1);
}

#[test]
fn vec_bif_inserts_when_mask_clear() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // BIF Vd, Vn, Vm  →  Vd = (Vd & Vm) | (Vn & ~Vm)
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[2] = [0xFF00_FF00_FF00_FF00, 0x0000_FFFF_FFFF_0000];
    let code = build_code(&[
        0x6EE2_1C20, // bif v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let exp0 = (0xAAAA_AAAA_AAAA_AAAAu64 & 0xFF00_FF00_FF00_FF00) | (0x1234_5678_9ABC_DEF0 & !0xFF00_FF00_FF00_FF00);
    let exp1 = (0xBBBB_BBBB_BBBB_BBBBu64 & 0x0000_FFFF_FFFF_0000) | (0xCAFE_BABE_DEAD_BEEF & !0x0000_FFFF_FFFF_0000);
    assert_eq!(ctx.v[0][0], exp0);
    assert_eq!(ctx.v[0][1], exp1);
}

#[test]
fn vec_bsl_selects_per_bit() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // BSL Vd, Vn, Vm  →  Vd = (Vn & Vd) | (Vm & ~Vd)  (Vd is the mask)
    ctx.v[0] = [0xFF00_FF00_FF00_FF00, 0x0000_FFFF_FFFF_0000]; // Vd is the mask
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222]; // Vn (take where Vd=1)
    ctx.v[2] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB]; // Vm (take where Vd=0)
    let code = build_code(&[
        0x6E62_1C20, // bsl v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let mask0 = 0xFF00_FF00_FF00_FF00u64;
    let mask1 = 0x0000_FFFF_FFFF_0000u64;
    let exp0 = (0x1111_1111_1111_1111u64 & mask0) | (0xAAAA_AAAA_AAAA_AAAA & !mask0);
    let exp1 = (0x2222_2222_2222_2222u64 & mask1) | (0xBBBB_BBBB_BBBB_BBBB & !mask1);
    assert_eq!(ctx.v[0][0], exp0);
    assert_eq!(ctx.v[0][1], exp1);
}

#[test]
fn vec_dup_4s_from_gpr() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = 0xDEAD_BEEF_CAFE_BABE;
    let code = build_code(&[
        0x4E04_0C20, // dup v0.4s, w1
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let want = 0xCAFE_BABE_CAFE_BABEu64;
    assert_eq!(ctx.v[0][0], want);
    assert_eq!(ctx.v[0][1], want);
}

#[test]
fn vec_dup_16b_from_gpr() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = 0xCAFE_BABE_DEAD_BE5A;
    let code = build_code(&[
        0x4E01_0C20, // dup v0.16b, w1
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let want = 0x5A5A_5A5A_5A5A_5A5Au64;
    assert_eq!(ctx.v[0][0], want);
    assert_eq!(ctx.v[0][1], want);
}

#[test]
fn vec_dup_2d_from_gpr() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = 0x1122_3344_5566_7788;
    let code = build_code(&[
        0x4E08_0C20, // dup v0.2d, x1
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x1122_3344_5566_7788);
    assert_eq!(ctx.v[0][1], 0x1122_3344_5566_7788);
}

#[test]
fn vec_umov_w_from_s_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Lane layout (4S): lane0=lo[0..32], lane1=lo[32..64], lane2=hi[0..32], lane3=hi[32..64]
    ctx.v[1] = [0x9999_9999_1111_1111, 0xDEAD_BEEF_2222_2222];
    let code = build_code(&[
        0x0E14_3C20, // umov w0, v1.s[2]
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // S[2] = low 32 of v[1][1] = 0x2222_2222 (little-endian within u64).
    assert_eq!(ctx.x[0], 0x2222_2222);
}

#[test]
fn vec_smov_x_from_b_lane_sign_extends() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Put 0x80 at byte lane 2 of V1 (third byte of v[1][0] little-endian).
    ctx.v[1] = [0x0000_0000_0080_0000, 0];
    let code = build_code(&[
        0x4E05_2C20, // smov x0, v1.b[2]  (imm5=00101 → byte lane 2)
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // lane 2 byte = 0x80 → sign-extend to 0xFFFFFFFFFFFFFF80
    assert_eq!(ctx.x[0], 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
fn vec_ins_b_lane_from_gpr_preserves_rest() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.x[1] = 0x12_34_56_78_9A_BC_DE_F0;
    let code = build_code(&[
        0x4E07_1C20, // ins v0.b[3], w1
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Low byte of W1 is 0xF0; replaces byte lane 3 of V0.
    // V0 was [0xAAAA_AAAA_AAAA_AAAA, ...] (16 bytes 0xAA).
    // After: byte[3] = 0xF0, others 0xAA.
    assert_eq!(ctx.v[0][0], 0xAAAA_AAAA_F0AA_AAAA);
    assert_eq!(ctx.v[0][1], 0xBBBB_BBBB_BBBB_BBBB);
}

#[test]
fn vec_dup_4s_from_element() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xAAAA_AAAA_1111_1111, 0xDEAD_BEEF_2222_2222];
    let code = build_code(&[
        0x4E14_0420, // dup v0.4s, v1.s[2]
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // V1.S[2] = low 32 of v[1][1] = 0x2222_2222. Broadcast to all 4 S lanes.
    assert_eq!(ctx.v[0][0], 0x2222_2222_2222_2222);
    assert_eq!(ctx.v[0][1], 0x2222_2222_2222_2222);
}

#[test]
fn vec_ext_byte_offset_4() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1 bytes 0x00..0x0F, V2 bytes 0x10..0x1F (little-endian within u64).
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    ctx.v[2] = [0x1716_1514_1312_1110, 0x1F1E_1D1C_1B1A_1918];
    let code = build_code(&[
        0x6E02_2020, // ext v0.16b, v1.16b, v2.16b, #4
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // EXT byte-offset 4 of {V2:V1}: bytes [4..16) of V1 then [0..4) of V2.
    assert_eq!(ctx.v[0][0], 0x0B0A_0908_0706_0504);
    assert_eq!(ctx.v[0][1], 0x1312_1110_0F0E_0D0C);
}

#[test]
fn vec_zip1_4s_interleaves_low_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // 4S lanes: lane0..lane3
    // V1 = [A0, A1, A2, A3], V2 = [B0, B1, B2, B3]
    // ZIP1 = [A0, B0, A1, B1]
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[
        0x4E82_3820, // zip1 v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Result lanes [A0, B0, A1, B1] → v[0][0]=B0:A0, v[0][1]=B1:A1
    assert_eq!(ctx.v[0][0], 0x0000_00B0_0000_00A0);
    assert_eq!(ctx.v[0][1], 0x0000_00B1_0000_00A1);
}

#[test]
fn vec_zip2_4s_interleaves_high_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[
        0x4E82_7820, // zip2 v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // ZIP2 4S = [A2, B2, A3, B3] → v[0][0]=B2:A2, v[0][1]=B3:A3
    assert_eq!(ctx.v[0][0], 0x0000_00B2_0000_00A2);
    assert_eq!(ctx.v[0][1], 0x0000_00B3_0000_00A3);
}

#[test]
fn vec_zip1_8h_interleaves_per_halfword() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA3A3_A2A2_A1A1_A0A0, 0xA7A7_A6A6_A5A5_A4A4];
    ctx.v[2] = [0xB3B3_B2B2_B1B1_B0B0, 0xB7B7_B6B6_B5B5_B4B4];
    let code = build_code(&[
        0x4E42_3820, // zip1 v0.8h, v1.8h, v2.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // ZIP1 8H lanes = [A0,B0,A1,B1,A2,B2,A3,B3]
    assert_eq!(ctx.v[0][0], 0xB1B1_A1A1_B0B0_A0A0);
    assert_eq!(ctx.v[0][1], 0xB3B3_A3A3_B2B2_A2A2);
}

#[test]
fn vec_smax_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_0000_0001, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[
        0x4EA2_6420, // smax v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Signed max per lane:
    //   lane0: max(-2, 1) = 1
    //   lane1: max(5, 3)  = 5
    //   lane2: max(0x80000000=-2^31, -1) = -1 = 0xFFFFFFFF
    //   lane3: max(0x7FFFFFFF, 0) = 0x7FFFFFFF
    assert_eq!(ctx.v[0][0], 0x0000_0005_0000_0001);
    assert_eq!(ctx.v[0][1], 0x7FFF_FFFF_FFFF_FFFF);
}

#[test]
fn vec_smin_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_0000_0001, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[
        0x4EA2_6C20, // smin v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    //   lane0: min(-2, 1)   = -2 = 0xFFFFFFFE
    //   lane1: min(5, 3)    = 3
    //   lane2: min(INT_MIN, -1) = INT_MIN = 0x80000000
    //   lane3: min(INT_MAX, 0)  = 0
    assert_eq!(ctx.v[0][0], 0x0000_0003_FFFF_FFFE);
    assert_eq!(ctx.v[0][1], 0x0000_0000_8000_0000);
}

#[test]
fn vec_umax_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0001, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFE, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[
        0x6EA2_6420, // umax v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    //   lane0: umax(1, 0xFFFFFFFE) = 0xFFFFFFFE
    //   lane1: umax(5, 3) = 5
    //   lane2: umax(0x80000000, 0xFFFFFFFF) = 0xFFFFFFFF
    //   lane3: umax(0x7FFFFFFF, 0) = 0x7FFFFFFF
    assert_eq!(ctx.v[0][0], 0x0000_0005_FFFF_FFFE);
    assert_eq!(ctx.v[0][1], 0x7FFF_FFFF_FFFF_FFFF);
}

#[test]
fn vec_umin_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0001, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFE, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[
        0x6EA2_6C20, // umin v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    //   lane0: umin(1, 0xFFFFFFFE) = 1
    //   lane1: umin(5, 3) = 3
    //   lane2: umin(0x80000000, 0xFFFFFFFF) = 0x80000000
    //   lane3: umin(0x7FFFFFFF, 0) = 0
    assert_eq!(ctx.v[0][0], 0x0000_0003_0000_0001);
    assert_eq!(ctx.v[0][1], 0x0000_0000_8000_0000);
}

#[test]
fn vec_addv_4s_sums_all_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // 4 32-bit lanes: 0x10, 0x20, 0x30, 0x40 → sum = 0xA0
    ctx.v[1] = [0x0000_0020_0000_0010, 0x0000_0040_0000_0030];
    let code = build_code(&[
        0x4EB1_B820, // addv s0, v1.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0] as u32, 0xA0);
    assert_eq!(ctx.v[0][0] >> 32, 0, "upper 32 of lane 0 zeroed");
    assert_eq!(ctx.v[0][1], 0, "upper 64 zeroed");
}

#[test]
fn vec_fadd_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Lanes (low-to-high): 1.0, 2.0, 3.0, 4.0
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (1.0_f32).to_bits() as u64;
    let v1_hi = ((4.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_lo = ((20.0_f32).to_bits() as u64) << 32 | (10.0_f32).to_bits() as u64;
    let v2_hi = ((40.0_f32).to_bits() as u64) << 32 | (30.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    ctx.v[2] = [v2_lo, v2_hi];
    let code = build_code(&[
        0x4E22_D420, // fadd v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let exp_lo = ((22.0_f32).to_bits() as u64) << 32 | (11.0_f32).to_bits() as u64;
    let exp_hi = ((44.0_f32).to_bits() as u64) << 32 | (33.0_f32).to_bits() as u64;
    assert_eq!(ctx.v[0][0], exp_lo);
    assert_eq!(ctx.v[0][1], exp_hi);
}

#[test]
fn vec_fmul_2d() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.5_f64).to_bits(), (3.0_f64).to_bits()];
    ctx.v[2] = [(4.0_f64).to_bits(), (1.5_f64).to_bits()];
    let code = build_code(&[
        0x6E62_DC20, // fmul v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 10.0);
    assert_eq!(f64::from_bits(ctx.v[0][1]), 4.5);
}

#[test]
fn vec_fadd_2s_zeros_upper_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (1.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, 0xDEAD_BEEF_CAFE_BABE];
    let v2_lo = ((20.0_f32).to_bits() as u64) << 32 | (10.0_f32).to_bits() as u64;
    ctx.v[2] = [v2_lo, 0x1234_5678_9ABC_DEF0];
    let code = build_code(&[
        0x0E22_D420, // fadd v0.2s, v1.2s, v2.2s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let exp_lo = ((22.0_f32).to_bits() as u64) << 32 | (11.0_f32).to_bits() as u64;
    assert_eq!(ctx.v[0][0], exp_lo);
    assert_eq!(ctx.v[0][1], 0, "2S form must zero upper 64");
}

#[test]
fn vec_fneg_4s_flips_signs() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let v1_lo = ((-2.0_f32).to_bits() as u64) << 32 | (1.5_f32).to_bits() as u64;
    let v1_hi = ((0.0_f32).to_bits() as u64) << 32 | (3.14_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    let code = build_code(&[
        0x6EA0_F820, // fneg v0.4s, v1.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let exp_lo = ((2.0_f32).to_bits() as u64) << 32 | ((-1.5_f32).to_bits() as u64);
    // -0.0 has sign bit set in IEEE 754.
    let exp_hi = ((-0.0_f32).to_bits() as u64) << 32 | ((-3.14_f32).to_bits() as u64);
    assert_eq!(ctx.v[0][0], exp_lo);
    assert_eq!(ctx.v[0][1], exp_hi);
}

#[test]
fn vec_fabs_2d_strips_sign() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(-7.5_f64).to_bits(), (3.14_f64).to_bits()];
    let code = build_code(&[
        0x4EE0_F820, // fabs v0.2d, v1.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 7.5);
    assert_eq!(f64::from_bits(ctx.v[0][1]), 3.14);
}

#[test]
fn vec_fsqrt_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let v1_lo = ((9.0_f32).to_bits() as u64) << 32 | (4.0_f32).to_bits() as u64;
    let v1_hi = ((25.0_f32).to_bits() as u64) << 32 | (16.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    let code = build_code(&[
        0x6EA1_F820, // fsqrt v0.4s, v1.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let exp_lo = ((3.0_f32).to_bits() as u64) << 32 | (2.0_f32).to_bits() as u64;
    let exp_hi = ((5.0_f32).to_bits() as u64) << 32 | (4.0_f32).to_bits() as u64;
    assert_eq!(ctx.v[0][0], exp_lo);
    assert_eq!(ctx.v[0][1], exp_hi);
}

#[test]
fn vec_saddl_8h_signed_widening_add() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Low 64 of V1 = 8 bytes: 0x80, 0x10, 0x7F, 0xFF, 0x01, 0x02, 0x03, 0x04
    // Low 64 of V2 = 8 bytes: 0x01, 0xF0, 0x01, 0x01, 0xFF, 0xFE, 0xFD, 0xFC
    ctx.v[1] = [0x0403_0201_FF7F_1080, 0xDEAD_BEEF_CAFE_BABE]; // hi ignored
    ctx.v[2] = [0xFCFD_FEFF_0101_F001, 0xDEAD_BEEF_CAFE_BABE];
    let code = build_code(&[
        0x0E22_0020, // saddl v0.8h, v1.8b, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Lane by lane (signed extend B to H, then add):
    //   B0: -128 + 1   = -127 = 0xFF81
    //   B1: 0x10 + -16 = 0
    //   B2: 127 + 1    = 128 = 0x0080
    //   B3: -1  + 1    = 0
    //   B4: 1   + -1   = 0
    //   B5: 2   + -2   = 0
    //   B6: 3   + -3   = 0
    //   B7: 4   + -4   = 0
    assert_eq!(ctx.v[0][0], 0x0000_0080_0000_FF81);
    assert_eq!(ctx.v[0][1], 0x0000_0000_0000_0000);
}

#[test]
fn vec_uaddl_8h_unsigned_widening_add() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0403_0201_FF7F_1080, 0];
    ctx.v[2] = [0x0000_0000_0101_F001, 0];
    let code = build_code(&[
        0x2E22_0020, // uaddl v0.8h, v1.8b, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Unsigned widen + add:
    //   B0: 0x80 + 0x01 = 0x0081
    //   B1: 0x10 + 0xF0 = 0x0100
    //   B2: 0x7F + 0x01 = 0x0080
    //   B3: 0xFF + 0x01 = 0x0100
    //   B4..B7: 1+0,2+0,3+0,4+0
    assert_eq!(ctx.v[0][0], 0x0100_0080_0100_0081);
    assert_eq!(ctx.v[0][1], 0x0004_0003_0002_0001);
}

#[test]
fn vec_saddl2_8h_reads_high_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Bytes 8..15 of each source are what's used.
    ctx.v[1] = [0xDEAD_BEEF_CAFE_BABE, 0x0403_0201_FF7F_1080];
    ctx.v[2] = [0xDEAD_BEEF_CAFE_BABE, 0xFCFD_FEFF_0101_F001];
    let code = build_code(&[
        0x4E22_0020, // saddl2 v0.8h, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Same expected values as the SADDL test (since we set the high halves
    // of v1/v2 to the same bytes the SADDL low-half test used).
    assert_eq!(ctx.v[0][0], 0x0000_0080_0000_FF81);
    assert_eq!(ctx.v[0][1], 0x0000_0000_0000_0000);
}

#[test]
fn vec_xtn_8b_truncates_each_h_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // 8 H lanes; XTN takes the low byte of each.
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_FACE_FEED];
    let code = build_code(&[
        0x0E21_2820, // xtn v0.8b, v1.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Each H lane's low byte (little-endian within u64): 0xF0, 0xBC, 0x78, 0x34, 0xED, 0xCE, 0xBE, 0xFE
    assert_eq!(ctx.v[0][0], 0xFEBE_CEED_3478_BCF0);
    assert_eq!(ctx.v[0][1], 0, "upper 64 zeroed for XTN.8B");
}

#[test]
fn vec_tbl_16b_single_table_with_out_of_range() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Table V1 = bytes 0xA0..0xAF
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    // Indices V2 = [0, 5, 10, 15, 16, 100, 200, 7,  1, 2, 3, 4, 6, 8, 9, 11]
    //   First four read 0xA0, 0xA5, 0xAA, 0xAF.
    //   Indices 16/100/200 are out-of-range → zero.
    //   Then 0xA7, 0xA1, 0xA2, 0xA3, 0xA4, 0xA6, 0xA8, 0xA9, 0xAB.
    ctx.v[2] = [0x07_C8_64_10_0F_0A_05_00, 0x0B_09_08_06_04_03_02_01];
    let code = build_code(&[
        0x4E02_0020, // tbl v0.16b, {v1.16b}, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xA7_00_00_00_AF_AA_A5_A0);
    assert_eq!(ctx.v[0][1], 0xAB_A9_A8_A6_A4_A3_A2_A1);
}

#[test]
fn vec_tbl_8b_zeros_upper_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    // Indices: low 8 = [0,2,4,6,8,10,12,14], high 8 ignored
    ctx.v[2] = [0x0E_0C_0A_08_06_04_02_00, 0xDEAD_BEEF_CAFE_BABE];
    let code = build_code(&[
        0x0E02_0020, // tbl v0.8b, {v1.16b}, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Picks bytes 0,2,4,6,8,10,12,14 of V1 → 0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E
    assert_eq!(ctx.v[0][0], 0x0E_0C_0A_08_06_04_02_00);
    assert_eq!(ctx.v[0][1], 0, "8B form zeros upper 64");
}

#[test]
fn vec_rev16_16b_swaps_byte_pairs() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // 16 bytes 0x00..0x0F in little-endian byte order within u64.
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[
        0x4E20_1820, // rev16 v0.16b, v1.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Swap each pair: [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14]
    assert_eq!(ctx.v[0][0], 0x0607_0405_0203_0001);
    assert_eq!(ctx.v[0][1], 0x0E0F_0C0D_0A0B_0809);
}

#[test]
fn vec_rev32_4s_byte_reverse_within_word() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[
        0x6E20_0820, // rev32 v0.16b, v1.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Reverse 4 bytes inside each S lane: [3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12]
    assert_eq!(ctx.v[0][0], 0x0405_0607_0001_0203);
    assert_eq!(ctx.v[0][1], 0x0C0D_0E0F_0809_0A0B);
}

#[test]
fn vec_rev32_8h_swap_halfwords_within_word() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[
        0x6E60_0820, // rev32 v0.8h, v1.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Swap H pairs inside each S: [2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13]
    assert_eq!(ctx.v[0][0], 0x0504_0706_0100_0302);
    assert_eq!(ctx.v[0][1], 0x0D0C_0F0E_0908_0B0A);
}

#[test]
fn vec_rev64_16b_full_qword_byte_reverse() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[
        0x4E20_0820, // rev64 v0.16b, v1.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Reverse 8 bytes inside each D lane.
    assert_eq!(ctx.v[0][0], 0x0001_0203_0405_0607);
    assert_eq!(ctx.v[0][1], 0x0809_0A0B_0C0D_0E0F);
}

#[test]
fn vec_rev64_4s_swap_words_within_qword() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[
        0x4EA0_0820, // rev64 v0.4s, v1.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Swap S pairs inside each D: [4,5,6,7, 0,1,2,3, 12,13,14,15, 8,9,10,11]
    assert_eq!(ctx.v[0][0], 0x0302_0100_0706_0504);
    assert_eq!(ctx.v[0][1], 0x0B0A_0908_0F0E_0D0C);
}

#[test]
fn vec_rev16_8b_zeros_upper_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0xDEAD_BEEF_CAFE_BABE];
    let code = build_code(&[
        0x0E20_1820, // rev16 v0.8b, v1.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0607_0405_0203_0001);
    assert_eq!(ctx.v[0][1], 0, "8B form zeros upper 64");
}

#[test]
fn vec_uzp1_4s_picks_even_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1.4S lanes [A0, A1, A2, A3]; V2.4S lanes [B0, B1, B2, B3]
    // UZP1 result = [A0, A2, B0, B2]
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[
        0x4E82_1820, // uzp1 v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_00A2_0000_00A0);
    assert_eq!(ctx.v[0][1], 0x0000_00B2_0000_00B0);
}

#[test]
fn vec_uzp2_4s_picks_odd_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[
        0x4E82_5820, // uzp2 v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // UZP2 = [A1, A3, B1, B3]
    assert_eq!(ctx.v[0][0], 0x0000_00A3_0000_00A1);
    assert_eq!(ctx.v[0][1], 0x0000_00B3_0000_00B1);
}

#[test]
fn vec_trn1_4s_transposes_even_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[
        0x4E82_2820, // trn1 v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // TRN1 = [A0, B0, A2, B2]
    assert_eq!(ctx.v[0][0], 0x0000_00B0_0000_00A0);
    assert_eq!(ctx.v[0][1], 0x0000_00B2_0000_00A2);
}

#[test]
fn vec_trn2_4s_transposes_odd_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[
        0x4E82_6820, // trn2 v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // TRN2 = [A1, B1, A3, B3]
    assert_eq!(ctx.v[0][0], 0x0000_00B1_0000_00A1);
    assert_eq!(ctx.v[0][1], 0x0000_00B3_0000_00A3);
}

#[test]
fn vec_uzp1_16b_picks_even_bytes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1 bytes 0xA0..0xAF; V2 bytes 0xB0..0xBF
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0xB7B6_B5B4_B3B2_B1B0, 0xBFBE_BDBC_BBBA_B9B8];
    let code = build_code(&[
        0x4E02_1820, // uzp1 v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Even bytes of Vn then even bytes of Vm:
    // [A0, A2, A4, A6, A8, AA, AC, AE,  B0, B2, B4, B6, B8, BA, BC, BE]
    assert_eq!(ctx.v[0][0], 0xAEAC_AAA8_A6A4_A2A0);
    assert_eq!(ctx.v[0][1], 0xBEBC_BAB8_B6B4_B2B0);
}

#[test]
fn vec_uzp1_2d_picks_low_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    ctx.v[2] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
    let code = build_code(&[
        0x4EC2_1820, // uzp1 v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // UZP1.2D = [Vn[0], Vm[0]]
    assert_eq!(ctx.v[0][0], 0x1111_1111_1111_1111);
    assert_eq!(ctx.v[0][1], 0x3333_3333_3333_3333);
}

#[test]
fn vec_uzp2_2d_picks_high_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    ctx.v[2] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
    let code = build_code(&[
        0x4EC2_5820, // uzp2 v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // UZP2.2D = [Vn[1], Vm[1]]
    assert_eq!(ctx.v[0][0], 0x2222_2222_2222_2222);
    assert_eq!(ctx.v[0][1], 0x4444_4444_4444_4444);
}

#[test]
fn vec_trn1_8h_transposes_even_h_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA3A3_A2A2_A1A1_A0A0, 0xA7A7_A6A6_A5A5_A4A4];
    ctx.v[2] = [0xB3B3_B2B2_B1B1_B0B0, 0xB7B7_B6B6_B5B5_B4B4];
    let code = build_code(&[
        0x4E42_2820, // trn1 v0.8h, v1.8h, v2.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // TRN1.8H = [A0, B0, A2, B2, A4, B4, A6, B6]
    assert_eq!(ctx.v[0][0], 0xB2B2_A2A2_B0B0_A0A0);
    assert_eq!(ctx.v[0][1], 0xB6B6_A6A6_B4B4_A4A4);
}

#[test]
fn vec_ssubl_8h_signed_widening_sub() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // 8 bytes per source — V1 - V2 lane by lane, sign-extended.
    ctx.v[1] = [0x0403_0201_FF7F_1080, 0];
    ctx.v[2] = [0xFCFD_FEFF_0101_F001, 0];
    let code = build_code(&[
        0x0E22_2020, // ssubl v0.8h, v1.8b, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    //   B0: 0x80 (-128) - 0x01 (1)   = -129 = 0xFF7F
    //   B1: 0x10 (16)   - 0xF0 (-16) =   32 = 0x0020
    //   B2: 0x7F (127)  - 0x01 (1)   =  126 = 0x007E
    //   B3: 0xFF (-1)   - 0x01 (1)   =   -2 = 0xFFFE
    //   B4..B7: 1- -1, 2- -2, 3- -3, 4- -4 → 2, 4, 6, 8
    assert_eq!(ctx.v[0][0], 0xFFFE_007E_0020_FF7F);
    assert_eq!(ctx.v[0][1], 0x0008_0006_0004_0002);
}

#[test]
fn vec_smull_4s_widening_mul() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1.4H = lanes [3, -2, 100, 0x7FFF]; V2.4H = [4, 5, -10, 2]
    let pack_h = |a: i16, b: i16, c: i16, d: i16| -> u64 {
        ((a as u16 as u64))
            | ((b as u16 as u64) << 16)
            | ((c as u16 as u64) << 32)
            | ((d as u16 as u64) << 48)
    };
    ctx.v[1] = [pack_h(3, -2, 100, 0x7FFF), 0];
    ctx.v[2] = [pack_h(4, 5, -10, 2), 0];
    let code = build_code(&[
        0x0E62_C020, // smull v0.4s, v1.4h, v2.4h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 4 S lanes: 3*4=12, -2*5=-10, 100*-10=-1000, 32767*2=65534
    let pack_s = |a: i32, b: i32| -> u64 {
        (a as u32 as u64) | ((b as u32 as u64) << 32)
    };
    assert_eq!(ctx.v[0][0], pack_s(12, -10));
    assert_eq!(ctx.v[0][1], pack_s(-1000, 65534));
}

#[test]
fn vec_umull_8h_widening_unsigned_mul() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0807_0605_0403_0201, 0];
    ctx.v[2] = [0x1010_1010_1010_1010, 0];
    let code = build_code(&[
        0x2E22_C020, // umull v0.8h, v1.8b, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 8 H lanes: 1*0x10=0x10, 2*0x10=0x20, ..., 8*0x10=0x80
    assert_eq!(ctx.v[0][0], 0x0040_0030_0020_0010);
    assert_eq!(ctx.v[0][1], 0x0080_0070_0060_0050);
}

#[test]
fn vec_cmhi_8b_unsigned() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_01_FF_7F_05_00_FE_03, 0];
    ctx.v[2] = [0x01_02_03_7F_05_FF_FD_04, 0];
    let code = build_code(&[
        0x2E22_3420, // cmhi v0.8b, v1.8b, v2.8b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Per byte unsigned compare:
    //   B0: 0x03 > 0x04 → false  = 0x00
    //   B1: 0xFE > 0xFD → true   = 0xFF
    //   B2: 0x00 > 0xFF → false  = 0x00
    //   B3: 0x05 > 0x05 → false  = 0x00
    //   B4: 0x7F > 0x7F → false  = 0x00
    //   B5: 0xFF > 0x03 → true   = 0xFF
    //   B6: 0x01 > 0x02 → false  = 0x00
    //   B7: 0x80 > 0x01 → true   = 0xFF
    assert_eq!(ctx.v[0][0], 0xFF_00_FF_00_00_00_FF_00);
    assert_eq!(ctx.v[0][1], 0);
}

#[test]
fn vec_cmhs_16b_unsigned() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_01_FF_7F_05_00_FE_03, 0x10_20_30_40_50_60_70_80];
    ctx.v[2] = [0x01_02_03_7F_05_FF_FD_04, 0x11_20_2F_40_50_61_70_FF];
    let code = build_code(&[
        0x6E22_3C20, // cmhs v0.16b, v1.16b, v2.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Per byte unsigned >= (equal = true, true = 0xFF).
    // Low half:
    //   B0: 3 >= 4 → false
    //   B1: 0xFE >= 0xFD → true
    //   B2: 0 >= 0xFF → false
    //   B3: 5 >= 5 → true
    //   B4: 0x7F >= 0x7F → true
    //   B5: 0xFF >= 3 → true
    //   B6: 1 >= 2 → false
    //   B7: 0x80 >= 1 → true
    assert_eq!(ctx.v[0][0], 0xFF_00_FF_FF_FF_00_FF_00);
    // High half:
    //   B8 : 0x80 >= 0xFF → false
    //   B9 : 0x70 >= 0x70 → true
    //   B10: 0x60 >= 0x61 → false
    //   B11: 0x50 >= 0x50 → true
    //   B12: 0x40 >= 0x40 → true
    //   B13: 0x30 >= 0x2F → true
    //   B14: 0x20 >= 0x20 → true
    //   B15: 0x10 >= 0x11 → false
    assert_eq!(ctx.v[0][1], 0x00_FF_FF_FF_FF_00_FF_00);
}

#[test]
fn vec_xtn2_16b_preserves_low_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Pre-set V0's low half to a sentinel that XTN2 must preserve.
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    // V1 = 8 H lanes; XTN2 takes low byte of each into V0's UPPER 64.
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_FACE_FEED];
    let code = build_code(&[
        0x4E21_2820, // xtn2 v0.16b, v1.8h
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Low half preserved exactly.
    assert_eq!(ctx.v[0][0], 0xAAAA_AAAA_AAAA_AAAA);
    // Upper 64 = packed low bytes of each H lane (same pattern as XTN smoke).
    assert_eq!(ctx.v[0][1], 0xFEBE_CEED_3478_BCF0);
}

#[test]
fn vec_fcmeq_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Lanes: 1.0, 2.0, 3.0, NaN
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (1.0_f32).to_bits() as u64;
    let v1_hi = ((f32::NAN).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_lo = ((2.0_f32).to_bits() as u64) << 32 | (5.0_f32).to_bits() as u64;
    let v2_hi = ((1.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    ctx.v[2] = [v2_lo, v2_hi];
    let code = build_code(&[
        0x4E22_E420, // fcmeq v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 1==5? false, 2==2? true, 3==3? true, NaN==1? false (NaN comparison)
    assert_eq!(ctx.v[0][0], 0xFFFFFFFF_00000000);
    assert_eq!(ctx.v[0][1], 0x00000000_FFFFFFFF);
}

#[test]
fn vec_fcmgt_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1: 5.0, 2.0, 3.0, NaN  vs  V2: 3.0, 7.0, 3.0, 1.0
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (5.0_f32).to_bits() as u64;
    let v1_hi = ((f32::NAN).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_lo = ((7.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_hi = ((1.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    ctx.v[2] = [v2_lo, v2_hi];
    let code = build_code(&[
        0x6EA2_E420, // fcmgt v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 5>3? T, 2>7? F, 3>3? F, NaN>1? F
    assert_eq!(ctx.v[0][0], 0x00000000_FFFFFFFF);
    assert_eq!(ctx.v[0][1], 0x00000000_00000000);
}

#[test]
fn vec_fcmge_2d() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Two D lanes per source.
    ctx.v[1] = [(2.5_f64).to_bits(), (f64::NAN).to_bits()];
    ctx.v[2] = [(2.5_f64).to_bits(), (1.0_f64).to_bits()];
    let code = build_code(&[
        0x6E62_E420, // fcmge v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 2.5 >= 2.5 → true; NaN >= 1.0 → false
    assert_eq!(ctx.v[0][0], 0xFFFFFFFF_FFFFFFFF);
    assert_eq!(ctx.v[0][1], 0);
}

#[test]
fn vec_fmla_4s_accumulates_product() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Vd starts at [1.0, 2.0, 3.0, 4.0]; Vn = [10, 10, 10, 10]; Vm = [2, 3, 4, 5]
    // After FMLA: Vd = Vd + Vn*Vm = [1+20, 2+30, 3+40, 4+50] = [21, 32, 43, 54]
    let pack_s = |a: f32, b: f32| -> u64 {
        (a.to_bits() as u64) | ((b.to_bits() as u64) << 32)
    };
    ctx.v[0] = [pack_s(1.0, 2.0), pack_s(3.0, 4.0)];
    ctx.v[1] = [pack_s(10.0, 10.0), pack_s(10.0, 10.0)];
    ctx.v[2] = [pack_s(2.0, 3.0), pack_s(4.0, 5.0)];
    let code = build_code(&[
        0x4E22_CC20, // fmla v0.4s, v1.4s, v2.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], pack_s(21.0, 32.0));
    assert_eq!(ctx.v[0][1], pack_s(43.0, 54.0));
}

#[test]
fn vec_fmls_2d_subtracts_product() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Vd = [100, 50]; Vn = [10, 5]; Vm = [3, 4]
    // FMLS: Vd = Vd - Vn*Vm = [100 - 30, 50 - 20] = [70, 30]
    ctx.v[0] = [(100.0_f64).to_bits(), (50.0_f64).to_bits()];
    ctx.v[1] = [(10.0_f64).to_bits(), (5.0_f64).to_bits()];
    ctx.v[2] = [(3.0_f64).to_bits(), (4.0_f64).to_bits()];
    let code = build_code(&[
        0x4EE2_CC20, // fmls v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 70.0);
    assert_eq!(f64::from_bits(ctx.v[0][1]), 30.0);
}

#[test]
fn vec_shl_imm_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Each byte shifted left by 2. 0x80<<2 = 0x00 (overflow), 0x40<<2 = 0x00.
    ctx.v[1] = [0x80_40_20_10_08_04_02_01, 0xFF_7F_3F_1F_0F_07_03_01];
    let code = build_code(&[
        0x4F0A_5420, // shl v0.16b, v1.16b, #2
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Per byte << 2 (8-bit only):
    //  Low: 1<<2=4, 2<<2=8, 4<<2=0x10, 8<<2=0x20, 0x10<<2=0x40, 0x20<<2=0x80, 0x40<<2=0, 0x80<<2=0
    //  High: 1<<2=4, 3<<2=0xC, 7<<2=0x1C, 0xF<<2=0x3C, 0x1F<<2=0x7C, 0x3F<<2=0xFC, 0x7F<<2=0xFC, 0xFF<<2=0xFC
    assert_eq!(ctx.v[0][0], 0x00_00_80_40_20_10_08_04);
    assert_eq!(ctx.v[0][1], 0xFC_FC_FC_7C_3C_1C_0C_04);
}

#[test]
fn vec_ushr_imm_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_40_20_10_08_04_02_01, 0xFF_7F_3F_1F_0F_07_03_01];
    let code = build_code(&[
        0x6F0E_0420, // ushr v0.16b, v1.16b, #2
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Per byte >> 2 unsigned.
    //  Low: 1>>2=0, 2>>2=0, 4>>2=1, 8>>2=2, 0x10>>2=4, 0x20>>2=8, 0x40>>2=0x10, 0x80>>2=0x20
    //  High: 1>>2=0, 3>>2=0, 7>>2=1, 0xF>>2=3, 0x1F>>2=7, 0x3F>>2=0xF, 0x7F>>2=0x1F, 0xFF>>2=0x3F
    assert_eq!(ctx.v[0][0], 0x20_10_08_04_02_01_00_00);
    assert_eq!(ctx.v[0][1], 0x3F_1F_0F_07_03_01_00_00);
}

#[test]
fn vec_sshr_imm_16b_sign_extends() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_40_20_10_08_04_02_01, 0xFF_7F_3F_1F_0F_07_03_01];
    let code = build_code(&[
        0x4F0E_0420, // sshr v0.16b, v1.16b, #2
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Per byte arithmetic >> 2.
    //  Low: 1>>2=0, 2>>2=0, 4>>2=1, 8>>2=2, 0x10>>2=4, 0x20>>2=8, 0x40>>2=0x10, 0x80=-128>>2=-32=0xE0
    //  High: 1>>2=0, 3>>2=0, 7>>2=1, 0xF>>2=3, 0x1F>>2=7, 0x3F>>2=0xF, 0x7F>>2=0x1F, 0xFF=-1>>2=-1=0xFF
    assert_eq!(ctx.v[0][0], 0xE0_10_08_04_02_01_00_00);
    assert_eq!(ctx.v[0][1], 0xFF_1F_0F_07_03_01_00_00);
}

#[test]
fn vec_mul_2d_via_decomposition() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Mix of edge cases:
    //   lane 0:  3 * 7              = 21
    //   lane 1:  0x1_0000_0001 * 5  = 0x5_0000_0005 (cross dword carry)
    //   plus another pair via second instruction (not used here)
    ctx.v[1] = [3, 0x0000_0001_0000_0001];
    ctx.v[2] = [7, 5];
    let code = build_code(&[
        0x4EE2_9C20, // mul v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 21);
    assert_eq!(ctx.v[0][1], 0x0000_0005_0000_0005);
}

#[test]
fn vec_mul_2d_wraps_at_64() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // 2^63 * 2 = 2^64 → wraps to 0 (low 64 of product).
    ctx.v[1] = [0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.v[2] = [2, 0xFFFF_FFFF_FFFF_FFFF];
    let code = build_code(&[
        0x4EE2_9C20, // mul v0.2d, v1.2d, v2.2d
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // (2^63)*2 mod 2^64 = 0
    assert_eq!(ctx.v[0][0], 0);
    // (-1)*(-1) = 1 in 64-bit arithmetic
    assert_eq!(ctx.v[0][1], 1);
}

#[test]
fn vec_smull_2d_via_pmovsxdq() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Two S lanes per source: low 64 of each vector.
    //   S0: -3 * 7      = -21          = 0xFFFFFFFFFFFFFFEB
    //   S1: 1000000 * 5 = 5_000_000
    let pack_s = |a: i32, b: i32| -> u64 {
        (a as u32 as u64) | ((b as u32 as u64) << 32)
    };
    ctx.v[1] = [pack_s(-3, 1_000_000), 0];
    ctx.v[2] = [pack_s(7, 5), 0];
    let code = build_code(&[
        0x0EA2_C020, // smull v0.2d, v1.2s, v2.2s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0] as i64, -21);
    assert_eq!(ctx.v[0][1], 5_000_000);
}

#[test]
fn vec_umull_2d_via_pmovzxdq() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let pack_s = |a: u32, b: u32| -> u64 {
        (a as u64) | ((b as u64) << 32)
    };
    ctx.v[1] = [pack_s(0xFFFF_FFFF, 0x1234_5678), 0];
    ctx.v[2] = [pack_s(2, 0xCAFE_BABE), 0];
    let code = build_code(&[
        0x2EA2_C020, // umull v0.2d, v1.2s, v2.2s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // S0: 0xFFFFFFFF * 2 = 0x1_FFFFFFFE
    // S1: 0x12345678 * 0xCAFEBABE = ?
    assert_eq!(ctx.v[0][0], 0xFFFFFFFFu64 * 2);
    assert_eq!(ctx.v[0][1], 0x1234_5678u64 * 0xCAFE_BABEu64);
}

#[test]
fn vec_sshr_imm_2d_arithmetic_shift() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // lane 0 = positive; lane 1 = negative.
    ctx.v[1] = [0x0000_0000_0000_0080, 0x8000_0000_0000_0000];
    let code = build_code(&[
        0x4F7C_0420, // sshr v0.2d, v1.2d, #4
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // 0x80 >> 4 = 0x8 (positive); 0x8000... >> 4 (arithmetic) = 0xF800_0000_0000_0000
    assert_eq!(ctx.v[0][0], 0x0000_0000_0000_0008);
    assert_eq!(ctx.v[0][1], 0xF800_0000_0000_0000);
}

#[test]
fn vec_sshr_imm_2d_by_one() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0x4000_0000_0000_0000];
    let code = build_code(&[
        0x4F7F_0420, // sshr v0.2d, v1.2d, #1
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // -1 arith >> 1 = -1; positive 0x4000... >> 1 = 0x2000...
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0x2000_0000_0000_0000);
}

#[test]
fn vec_tbl2_16b_two_register_table() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Table = V1||V2; V1 bytes 0xA0..0xAF, V2 bytes 0xB0..0xBF.
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0xB7B6_B5B4_B3B2_B1B0, 0xBFBE_BDBC_BBBA_B9B8];
    // Indices: mix of low (V1), high (V2), and out-of-range.
    let mut idx = [0u8; 16];
    let want = [0u8, 5, 16, 31, 32, 200, 17, 15,  3, 18, 50, 7,  29, 1, 100, 14];
    for (i, &v) in want.iter().enumerate() { idx[i] = v; }
    let mut lo = [0u8; 8]; lo.copy_from_slice(&idx[..8]);
    let mut hi = [0u8; 8]; hi.copy_from_slice(&idx[8..]);
    ctx.v[3] = [u64::from_le_bytes(lo), u64::from_le_bytes(hi)];

    let code = build_code(&[
        0x4E03_2020, // tbl v0.16b, {v1.16b, v2.16b}, v3.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Expected per index:
    //   0   → V1[0]  = 0xA0
    //   5   → V1[5]  = 0xA5
    //   16  → V2[0]  = 0xB0
    //   31  → V2[15] = 0xBF
    //   32  → 0
    //   200 → 0
    //   17  → V2[1]  = 0xB1
    //   15  → V1[15] = 0xAF
    //   3   → V1[3]  = 0xA3
    //   18  → V2[2]  = 0xB2
    //   50  → 0
    //   7   → V1[7]  = 0xA7
    //   29  → V2[13] = 0xBD
    //   1   → V1[1]  = 0xA1
    //   100 → 0
    //   14  → V1[14] = 0xAE
    let want = [0xA0u8, 0xA5, 0xB0, 0xBF, 0x00, 0x00, 0xB1, 0xAF,
                0xA3,    0xB2, 0x00, 0xA7, 0xBD, 0xA1, 0x00, 0xAE];
    let mut lo = [0u8; 8]; lo.copy_from_slice(&want[..8]);
    let mut hi = [0u8; 8]; hi.copy_from_slice(&want[8..]);
    assert_eq!(ctx.v[0][0], u64::from_le_bytes(lo));
    assert_eq!(ctx.v[0][1], u64::from_le_bytes(hi));
}

#[test]
fn vec_tbl3_16b_three_register_table() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Table = V1||V2||V3 (48 bytes, 0xA0..0xCF).
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0xB7B6_B5B4_B3B2_B1B0, 0xBFBE_BDBC_BBBA_B9B8];
    ctx.v[3] = [0xC7C6_C5C4_C3C2_C1C0, 0xCFCE_CDCC_CBCA_C9C8];
    // Indices spanning all three chunks plus out-of-range.
    let want_idx = [0u8, 16, 32, 47, 48, 200, 33, 15,  3, 18, 35, 7,  31, 1, 100, 46];
    let mut idx = [0u8; 16];
    for (i, &v) in want_idx.iter().enumerate() { idx[i] = v; }
    let mut lo = [0u8; 8]; lo.copy_from_slice(&idx[..8]);
    let mut hi = [0u8; 8]; hi.copy_from_slice(&idx[8..]);
    ctx.v[4] = [u64::from_le_bytes(lo), u64::from_le_bytes(hi)];

    let code = build_code(&[
        0x4E04_4020, // tbl v0.16b, {v1.16b, v2.16b, v3.16b}, v4.16b
        0xD4200000,
    ]);
    run(code, &mut ctx);
    let want = [0xA0u8, 0xB0, 0xC0, 0xCF, 0x00, 0x00, 0xC1, 0xAF,
                0xA3,    0xB2, 0xC3, 0xA7, 0xBF, 0xA1, 0x00, 0xCE];
    let mut lo = [0u8; 8]; lo.copy_from_slice(&want[..8]);
    let mut hi = [0u8; 8]; hi.copy_from_slice(&want[8..]);
    assert_eq!(ctx.v[0][0], u64::from_le_bytes(lo));
    assert_eq!(ctx.v[0][1], u64::from_le_bytes(hi));
}

#[test]
fn vec_fmls_nan_input_clears_sign() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // V1 (Vd) = arbitrary; V2 (Vn) = NaN with sign bit SET in lane 0;
    // V3 (Vm) = arbitrary.
    // ARM FMLS = Vd - Vn*Vm; with Vn = sign-set NaN, the FPNeg inside ARM's
    // FMLS pseudocode flips the NaN's sign before propagation, so the output
    // NaN should have sign 0.
    ctx.v[1] = [(1.5_f32).to_bits() as u64, 0];
    ctx.v[2] = [0xFFFFFFFFu64, 0];  // sign-set NaN in lane 0
    ctx.v[3] = [(2.0_f32).to_bits() as u64, 0];
    let code = build_code(&[
        0x0EA3_CC41, // fmls v1.2s, v2.2s, v3.2s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // Result lane 0 should be NaN with sign bit clear (0x7FFFFFFF), not
    // 0xFFFFFFFF.
    let lane0 = ctx.v[1][0] as u32;
    assert!(f32::from_bits(lane0).is_nan(), "result lane 0 should be NaN, got {:#010x}", lane0);
    assert_eq!(lane0 >> 31, 0, "NaN sign bit must be clear after FMLS, got {:#010x}", lane0);
}

/// Dynarmic-style FRINT test: input 0x4001e17c ≈ 2.0294 in each of 4 single-
/// precision lanes. All round-modes converge on 2.0 except FRINTP which goes
/// to 3.0. Verifies each rounding mode picks the right x86 ROUNDPS predicate.
#[test]
fn vec_frint_family_dynarmic_test() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0x4001e17c4001e17c, 0x4001e17c4001e17c];
    let code = build_code(&[
        0x4E218801, // frintn v1.4s, v0.4s
        0x4E219802, // frintm v2.4s, v0.4s
        0x4EA18803, // frintp v3.4s, v0.4s
        0x4EA19804, // frintz v4.4s, v0.4s
        0x6E218805, // frinta v5.4s, v0.4s
        0x6E219806, // frintx v6.4s, v0.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0], [0x4001e17c4001e17c, 0x4001e17c4001e17c], "input preserved");
    assert_eq!(ctx.v[1], [0x4000000040000000, 0x4000000040000000], "FRINTN");
    assert_eq!(ctx.v[2], [0x4000000040000000, 0x4000000040000000], "FRINTM");
    assert_eq!(ctx.v[3], [0x4040000040400000, 0x4040000040400000], "FRINTP");
    assert_eq!(ctx.v[4], [0x4000000040000000, 0x4000000040000000], "FRINTZ");
    assert_eq!(ctx.v[5], [0x4000000040000000, 0x4000000040000000], "FRINTA");
    assert_eq!(ctx.v[6], [0x4000000040000000, 0x4000000040000000], "FRINTX");
}

/// FRINTA on half-integer values where ties-away-from-zero differs from
/// ties-to-even — the case ROUNDPS imm=0 would get wrong for us.
#[test]
fn vec_frinta_ties_away_from_zero() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    // Lanes: 0.5, 1.5, 2.5, -0.5
    let pack = |a: f32, b: f32| ((a.to_bits() as u64) | ((b.to_bits() as u64) << 32));
    ctx.v[0] = [pack(0.5, 1.5), pack(2.5, -0.5)];
    let code = build_code(&[
        0x6E218801, // frinta v1.4s, v0.4s
        0xD4200000,
    ]);
    run(code, &mut ctx);
    // FRINTA ties-away: 0.5→1.0, 1.5→2.0, 2.5→3.0, -0.5→-1.0
    let want = [pack(1.0, 2.0), pack(3.0, -1.0)];
    assert_eq!(ctx.v[1], want);
}
