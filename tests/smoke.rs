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
