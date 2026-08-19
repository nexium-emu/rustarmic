#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;

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

fn run(code: Vec<u8>, ctx: &mut CpuContext) -> ExitReason {
    let mut mem = CodeMem {
        bytes: code,
        base: CODE_BASE,
    };
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    jit.run(ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

#[test]
fn movz_into_x0() {
    let code = build_code(&[0xD282_4680, 0xD420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x1234, "X0 should be 0x1234 after MOVZ");
}

#[test]
fn add_imm_pipeline() {
    let code = build_code(&[0xD280_0C80, 0x9100_C800, 0xD420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 150, "X0 should be 150 (100 + 50)");
}

#[test]
fn sub_imm_negative() {
    let code = build_code(&[0xD280_0140, 0xD100_3C00, 0xD420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0] as i64, -5, "X0 should wrap to -5 after SUB");
}

#[test]
fn movz_into_x5_and_orr_reg() {
    let code = build_code(&[0xD2801FE0, 0xD28001E1, 0xAA010002, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(
        matches!(exit, ExitReason::Brk(_)),
        "should hit BRK, got {:?}",
        exit
    );
    assert_eq!(ctx.x[0], 0xFF);
    assert_eq!(ctx.x[1], 0x0F);
    assert_eq!(ctx.x[2], 0xFF);
}

#[test]
fn ubfm_zero_extend_byte() {
    let code = build_code(&[0xD29FFFE0, 0xD3401C01, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0xFFFF);
    assert_eq!(ctx.x[1], 0xFF, "UBFM should mask to low 8 bits");
}

#[test]
fn csel_picks_based_on_nzcv() {
    let code = build_code(&[0xD2800C80, 0xD2801901, 0xEB01001F, 0x9A814002, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 100, "CSEL with MI should pick X0");
}

#[test]
fn madd_three_operand() {
    let code = build_code(&[0xD28000A0, 0xD28000E1, 0xD2800062, 0x9B010803, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[3], 38, "MADD should compute Ra + Rn*Rm");
}

#[test]
fn udiv_normal_case() {
    let code = build_code(&[0xD2800C80, 0xD28000E1, 0x9AC10802, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 14, "100 / 7 should be 14");
}

#[test]
fn udiv_by_zero_returns_zero() {
    let code = build_code(&[0xD2800C80, 0xD2800001, 0x9AC10802, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 0, "UDIV by zero must return 0, not trap");
}

#[test]
fn sdiv_normal_negative() {
    let code = build_code(&[0xD2800C80, 0xCB0003E0, 0xD2800081, 0x9AC10C02, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2] as i64, -25, "SDIV -100 / 4 should be -25");
}

#[test]
fn sdiv_by_zero_returns_zero() {
    let code = build_code(&[0xD2800C80, 0xD2800001, 0x9AC10C02, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 0, "SDIV by zero must return 0, not trap");
}

#[test]
fn sdiv_int_min_by_neg_one_returns_int_min() {
    let code = build_code(&[
        0xD2A00000_u32 ^ 0,
        0xD2F00000,
        0xD2800021,
        0xCB0103E1,
        0x9AC10C02,
        0xD4200000,
    ]);
    let code = code[4..].to_vec();
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x8000_0000_0000_0000, "X0 should be INT_MIN_64");
    assert_eq!(ctx.x[1] as i64, -1, "X1 should be -1");
    assert_eq!(
        ctx.x[2], 0x8000_0000_0000_0000,
        "SDIV INT_MIN / -1 must return INT_MIN unchanged (no overflow trap)"
    );
}

#[test]
fn lslv_variable_shift() {
    let code = build_code(&[0xD2800020, 0xD28000A1, 0x9AC12002, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[2], 32, "1 << 5 should be 32");
}

#[test]
fn clz_typical_and_zero() {
    let code = build_code(&[0xD2800000, 0xDAC01001, 0xD2800022, 0xDAC01043, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[1], 64, "CLZ(0) should be 64");
    assert_eq!(ctx.x[3], 63, "CLZ(1) should be 63");
}

#[test]
fn cls_typical_and_all_same() {
    let code = build_code(&[
        0xD2800000, 0xDAC01401, 0xD2800022, 0xCB0203E2, 0xDAC01443, 0xD2F00004, 0xDAC01485,
        0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[1], 63);
    assert_eq!(ctx.x[3], 63);
    assert_eq!(ctx.x[5], 0);
}

fn clz64(input: u64) -> u64 {
    let code = build_code(&[0xDAC01001, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = input;
    run(code, &mut ctx);
    ctx.x[1]
}
fn clz32(input: u32) -> u32 {
    let code = build_code(&[0x5AC01001, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = input as u64;
    run(code, &mut ctx);
    ctx.x[1] as u32
}
fn cls64(input: u64) -> u64 {
    let code = build_code(&[0xDAC01401, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = input;
    run(code, &mut ctx);
    ctx.x[1]
}
fn cls32(input: u32) -> u32 {
    let code = build_code(&[0x5AC01401, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = input as u64;
    run(code, &mut ctx);
    ctx.x[1] as u32
}

fn cls64_oracle(x: u64) -> u64 {
    (x ^ ((x as i64 >> 1) as u64)).leading_zeros() as u64 - 1
}
fn cls32_oracle(x: u32) -> u32 {
    (x ^ ((x as i32 >> 1) as u32)).leading_zeros() - 1
}

#[test]
fn clz64_full_coverage() {
    for &v in &[
        0u64,
        1,
        0xFFFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x0000_0000_0000_0080,
        0x0123_4567_89AB_CDEF,
        0xDEAD_BEEF_CAFE_BABE,
    ] {
        assert_eq!(clz64(v), v.leading_zeros() as u64, "CLZ64({v:#018x})");
    }
}
#[test]
fn clz32_full_coverage() {
    for &v in &[
        0u32,
        1,
        0xFFFF_FFFF,
        0x8000_0000,
        0x4000_0000,
        0x0123_4567,
        0xDEAD_BEEF,
    ] {
        assert_eq!(clz32(v), v.leading_zeros(), "CLZ32({v:#010x})");
    }
}
#[test]
fn cls64_full_coverage() {
    for &v in &[
        0u64,
        1,
        !0u64,
        0x4000_0000_0000_0000,
        0x3FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
        0x8000_0000_0000_0001,
        0xC000_0000_0000_0000,
        0xE000_0000_0000_0000,
        0x0123_4567_89AB_CDEF,
        0xFEDC_BA98_7654_3210,
    ] {
        assert_eq!(
            cls64(v),
            cls64_oracle(v),
            "CLS64({v:#018x}) expected {}",
            cls64_oracle(v)
        );
    }
}
#[test]
fn cls32_full_coverage() {
    for &v in &[
        0u32,
        1,
        !0u32,
        0x4000_0000,
        0x3FFF_FFFF,
        0x8000_0000,
        0x8000_0001,
        0xC000_0000,
        0xE000_0000,
        0x0123_4567,
        0xFEDC_BA98,
    ] {
        assert_eq!(
            cls32(v),
            cls32_oracle(v),
            "CLS32({v:#010x}) expected {}",
            cls32_oracle(v)
        );
    }
}

#[test]
fn rbit_reverses_bits() {
    let code = build_code(&[0xD2800020, 0xDAC00001, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(
        ctx.x[1], 0x8000_0000_0000_0000,
        "RBIT(1) should be 0x8000_0000_0000_0000"
    );
}

fn rbit64(input: u64) -> u64 {
    let code = build_code(&[0xDAC00001, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = input;
    run(code, &mut ctx);
    ctx.x[1]
}

fn rbit32(input: u32) -> u32 {
    let code = build_code(&[0x5AC00001, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = input as u64;
    run(code, &mut ctx);
    ctx.x[1] as u32
}

#[test]
fn rbit64_full_coverage() {
    for &v in &[
        0u64,
        1,
        0xFFFF_FFFF_FFFF_FFFF,
        0x5555_5555_5555_5555,
        0xAAAA_AAAA_AAAA_AAAA,
        0x0123_4567_89AB_CDEF,
        0xF0F0_F0F0_F0F0_F0F0,
        0x0F0F_0F0F_0F0F_0F0F,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0080,
        0xDEAD_BEEF_CAFE_BABE,
    ] {
        assert_eq!(
            rbit64(v),
            v.reverse_bits(),
            "RBIT64({v:#018x}) expected {:#018x}",
            v.reverse_bits()
        );
    }
}

#[test]
fn rbit32_full_coverage() {
    for &v in &[
        0u32,
        1,
        0xFFFF_FFFF,
        0x5555_5555,
        0xAAAA_AAAA,
        0x0123_4567,
        0xDEAD_BEEF,
        0x8000_0000,
        0x0000_0080,
    ] {
        assert_eq!(
            rbit32(v),
            v.reverse_bits(),
            "RBIT32({v:#010x}) expected {:#010x}",
            v.reverse_bits()
        );
    }
}

#[test]
fn rev_byte_swap() {
    let code = build_code(&[
        0xD29BDE60, 0xD299BDE0, 0xF2B13560, 0xF2C8ACE0, 0xF2E02460, 0xDAC00C01, 0xD4200000,
    ]);
    let code = code[4..].to_vec();
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x0123_4567_89AB_CDEF);
    assert_eq!(
        ctx.x[1], 0xEFCD_AB89_6745_2301,
        "REV should byte-swap whole register"
    );
}

#[test]
fn ccmp_imm_failed_cond_uses_nzcv_imm() {
    let code = build_code(&[
        0xD2800020, 0xD2800041, 0xEB01001F, 0xFA45080A, 0x9A9F47E2, 0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(
        ctx.nzcv & 0xF,
        0xA,
        "After CCMP-fail, NZCV should equal imm nibble 0xA"
    );
    assert_eq!(ctx.x[2], 0, "csinc on MI(N=1) should pick xzr (0)");
}

#[test]
fn ccmp_imm_passed_cond_does_compare() {
    let code = build_code(&[
        0xD28000A0, 0xD28000A1, 0xEB01001F, 0xFA45080F, 0x9A9F17E2, 0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(
        ctx.nzcv & 0xF,
        0b0110,
        "After CCMP-pass, NZCV should be compare-result (Z=1,C=1)"
    );
    assert_eq!(ctx.x[2], 1, "csinc on NE-fail should pick xzr+1 = 1");
}

#[test]
fn adc_carries_from_subs() {
    let code = build_code(&[
        0xD2800140, 0xD28000A1, 0xEB01001F, 0xD2800C82, 0xD2800023, 0x9A030044, 0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[4], 102, "ADC should add 100 + 1 + carry(=1) = 102");
}

#[test]
fn adc_no_carry_from_subs() {
    let code = build_code(&[
        0xD28000A0, 0xD2800141, 0xEB01001F, 0xD2800C82, 0xD2800023, 0x9A030044, 0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.x[4], 101, "ADC without carry should be 100 + 1 = 101");
}

#[test]
fn two_block_direct_branch_chains() {
    let mut code = vec![0u8; 0x10C];
    code[0..4].copy_from_slice(&0xD2800020u32.to_le_bytes());
    code[4..8].copy_from_slice(&0x1400003Fu32.to_le_bytes());
    code[0x100..0x104].copy_from_slice(&0xD28000A1u32.to_le_bytes());
    code[0x104..0x108].copy_from_slice(&0x8B000021u32.to_le_bytes());
    code[0x108..0x10C].copy_from_slice(&0xD4200000u32.to_le_bytes());

    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(
        matches!(exit, ExitReason::Brk(_)),
        "expected BRK exit, got {:?}",
        exit
    );
    assert_eq!(ctx.x[0], 1, "X0 from block A");
    assert_eq!(ctx.x[1], 6, "X1 = 5 + 1 from block B (after branch from A)");
}

#[test]
fn cbnz_not_taken_falls_through() {
    let code = build_code(&[0xD2800000, 0xB50007E0, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(
        matches!(exit, ExitReason::Brk(_)),
        "expected BRK, got {:?}",
        exit
    );
}

#[test]
fn cbnz_loop_chains() {
    let code = build_code(&[
        0xD28000A0, 0xD2800001, 0x8B000021, 0xD1000400, 0xB5FFFFC0, 0xD4200000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let exit = run(code, &mut ctx);
    assert!(
        matches!(exit, ExitReason::Brk(_)),
        "expected BRK, got {:?}",
        exit
    );
    assert_eq!(ctx.x[0], 0, "counter ends at 0");
    assert_eq!(ctx.x[1], 15, "accumulator = 5+4+3+2+1 = 15");
}

#[test]
fn add_sub_chain_uses_constant_folding() {
    let code = build_code(&[0xD2800C80, 0x91000401, 0x91000822, 0xD100C843, 0xD4200000]);
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
    let code = build_code(&[0xD299_5FC0, 0xD51B_D040, 0xD53B_D041, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(ctx.tpidr_el0, 0xCAFE, "MSR should write tpidr_el0 slot");
    assert_eq!(ctx.x[1], 0xCAFE, "MRS should round-trip the value");
}

#[test]
fn pacia_then_autia_round_trips_pointer() {
    let code = build_code(&[0xD295_79A0, 0xDAC1_0020, 0xDAC1_1020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(
        ctx.x[0], 0xABCD,
        "PACIA+AUTIA should be identity in our model"
    );
}

#[test]
fn fmov_v_to_v_single_precision() {
    let code = build_code(&[0x1E20_4041, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[2] = [0x1122_3344_5566_7788, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    run(code, &mut ctx);
    assert_eq!(
        ctx.v[1][0], 0x5566_7788,
        "S-precision FMOV copies low 32 bits"
    );
    assert_eq!(ctx.v[1][1], 0, "S-precision FMOV zeros upper 96 bits");
}

#[test]
fn fmov_v_to_v_double_precision() {
    let code = build_code(&[0x1E60_4041, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[2] = [0x1122_3344_5566_7788, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    run(code, &mut ctx);
    assert_eq!(
        ctx.v[1][0], 0x1122_3344_5566_7788,
        "D FMOV copies low 64 bits"
    );
    assert_eq!(ctx.v[1][1], 0, "D FMOV zeros upper 64 bits");
}

#[test]
fn fadd_d_two_doubles() {
    let code = build_code(&[0x1E61_2802, 0xD4200000]);
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
    let code = build_code(&[0x1E21_0802, 0xD4200000]);
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
    let code = build_code(&[0x1E61_2000, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(1.0_f64).to_bits(), 0];
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b1000, "1.0 < 2.0 should set N only");
}

#[test]
fn fcmp_d_equal_sets_z_and_c() {
    let code = build_code(&[0x1E61_2000, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(1.5_f64).to_bits(), 0];
    ctx.v[1] = [(1.5_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0110, "equal should set Z and C");
}

#[test]
fn fcmp_d_greater_sets_c_only() {
    let code = build_code(&[0x1E61_2000, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(3.0_f64).to_bits(), 0];
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0010, "greater should set C only");
}

#[test]
fn fcmp_d_nan_sets_c_and_v() {
    let code = build_code(&[0x1E61_2000, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [f64::NAN.to_bits(), 0];
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0011, "unordered (NaN) should set C and V");
}

#[test]
fn fcmp_d_against_zero_immediate() {
    let code = build_code(&[0x1E60_2008, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(5.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b0010, "5.0 > 0 should set C");
}

#[test]
fn fcsel_d_picks_taken_when_eq() {
    let code = build_code(&[0x1E63_2040, 0x1E61_0C04, 0xD4200000]);
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
    let code = build_code(&[0x1E63_2040, 0x1E61_1C04, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(7.5_f64).to_bits(), 0];
    ctx.v[1] = [(3.5_f64).to_bits(), 0];
    ctx.v[2] = [(1.0_f64).to_bits(), 0];
    ctx.v[3] = [(1.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(
        f64::from_bits(ctx.v[4][0]),
        3.5,
        "NE fails → fcsel picks Fm (d1)"
    );
}

#[test]
fn fmov_d_immediate_loads_1_0() {
    let code = build_code(&[0x1E6E_1000, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xDEAD_BEEF_DEAD_BEEF, 0xDEAD_BEEF_DEAD_BEEF];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 1.0);
    assert_eq!(ctx.v[0][1], 0, "high lane zeroed");
}

#[test]
fn fmov_d_immediate_loads_2_0() {
    let code = build_code(&[0x1E60_1000, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 2.0);
}

#[test]
fn fmov_s_immediate_loads_1_0() {
    let code = build_code(&[0x1E2E_1000, 0xD4200000]);
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
    let code = build_code(&[0x1E61_4020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.5_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -3.5);
}

#[test]
fn fabs_d_clears_sign_bit() {
    let code = build_code(&[0x1E60_C020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(-7.25_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 7.25);
}

#[test]
fn fsqrt_d_computes_square_root() {
    let code = build_code(&[0x1E61_C020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(4.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 2.0);
}

#[test]
fn fmadd_d_computes_a_plus_n_times_m() {
    let code = build_code(&[0x1F42_0C20, 0xD4200000]);
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
    let code = build_code(&[0x1F42_8C20, 0xD4200000]);
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
    let code = build_code(&[0x1F62_0C20, 0xD4200000]);
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
    let code = build_code(&[0x1F62_8C20, 0xD4200000]);
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
    let code = build_code(&[0x1E78_0020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.75_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.x[0] as i32, 3, "FCVTZS truncates toward zero");
}

#[test]
fn fcvtzs_x_from_double_negative() {
    let code = build_code(&[0x9E78_0020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(-3.75_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(
        ctx.x[0] as i64, -3,
        "FCVTZS truncates toward zero, not floor"
    );
}

#[test]
fn scvtf_d_from_x_signed_int() {
    let code = build_code(&[0x9E62_0020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = (-42_i64) as u64;
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -42.0);
}

#[test]
fn fmov_d_from_x_copies_bits() {
    let code = build_code(&[0x9E67_0020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[1] = (1.5_f64).to_bits();
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 1.5);
    assert_eq!(ctx.v[0][1], 0, "high lane zeroed");
}

#[test]
fn fmov_x_from_d_copies_bits() {
    let code = build_code(&[0x9E66_0020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.5_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.x[0]), 2.5);
}

#[test]
fn fcvt_s_to_d_promotes_float() {
    let code = build_code(&[0x1E22_C020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(1.5_f32).to_bits() as u64, 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 1.5_f64);
}

#[test]
fn fcvt_d_to_s_demotes_double() {
    let code = build_code(&[0x1E62_4020, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.25_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f32::from_bits(ctx.v[0][0] as u32), 2.25_f32);
    assert_eq!(ctx.v[0][0] >> 32, 0);
}

#[test]
fn fmax_d_picks_larger() {
    let code = build_code(&[0x1E62_4820, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.0_f64).to_bits(), 0];
    ctx.v[2] = [(5.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 5.0);
}

#[test]
fn fmin_d_picks_smaller() {
    let code = build_code(&[0x1E62_5820, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(3.0_f64).to_bits(), 0];
    ctx.v[2] = [(5.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 3.0);
}

#[test]
fn fnmul_d_negates_product() {
    let code = build_code(&[0x1E62_8820, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.0_f64).to_bits(), 0];
    ctx.v[2] = [(3.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), -6.0);
}

#[test]
fn fccmp_d_cond_holds_runs_fcmp() {
    let code = build_code(&[0x1E62_0420, 0xD4200000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.nzcv = 0b0100;
    ctx.v[1] = [(1.0_f64).to_bits(), 0];
    ctx.v[2] = [(2.0_f64).to_bits(), 0];
    run(code, &mut ctx);
    assert_eq!(ctx.nzcv, 0b1000, "cond held → FCMP set N");
}

#[test]
fn fccmp_d_cond_fails_uses_immediate_nzcv() {
    let code = build_code(&[0x1E62_042A, 0xD4200000]);
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
    let code = build_code(&[
        0xD28000A0,
        0xD2800064,
        0x9B04_7C01,
        0xD37E_F422,
        0x8B00_0043,
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
    ctx.v[1] = [0x0807_0605_0403_0201, 0x100F_0E0D_0C0B_0A09];
    ctx.v[2] = [0x1010_1010_1010_1010, 0x1010_1010_1010_1010];
    let code = build_code(&[0x4E22_8420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x1817_1615_1413_1211);
    assert_eq!(ctx.v[0][1], 0x201F_1E1D_1C1B_1A19);
}

#[test]
fn vec_add_8h_wraps_per_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.v[2] = [0x0001_0001_0001_0001, 0x0001_0001_0001_0001];
    let code = build_code(&[0x4E62_8420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0], [0, 0], "every 16-bit lane wraps independently");
}

#[test]
fn vec_add_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0002_0000_0001, 0x0000_0004_0000_0003];
    ctx.v[2] = [0x0000_000A_0000_000A, 0x0000_000A_0000_000A];
    let code = build_code(&[0x4EA2_8420, 0xD4200000]);
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
    let code = build_code(&[0x4EE2_8420, 0xD4200000]);
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
    let code = build_code(&[0x0E22_8420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(
        ctx.v[0][0], 0x0908_0706_0504_0302,
        "low half = lanewise add"
    );
    assert_eq!(ctx.v[0][1], 0, "upper 64 bits must be zeroed for 8B form");
}

#[test]
fn vec_sub_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0014_0000_0014, 0x0000_0014_0000_0014];
    ctx.v[2] = [0x0000_0004_0000_0001, 0x0000_0006_0000_0005];
    let code = build_code(&[0x6EA2_8420, 0xD4200000]);
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
    let code = build_code(&[0x4E22_1C20, 0xD4200000]);
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
    let code = build_code(&[0x6E22_1C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0x0000_FFFF_FFFF_0000);
}

#[test]
fn vec_bic_clears_bits_per_mask() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0xAAAA_AAAA_AAAA_AAAA];
    ctx.v[2] = [0x00FF_00FF_00FF_00FF, 0x000F_000F_000F_000F];
    let code = build_code(&[0x4E62_1C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFF00_FF00_FF00_FF00);
    assert_eq!(ctx.v[0][1], 0xAAA0_AAA0_AAA0_AAA0);
}

#[test]
fn vec_orn_or_inverted() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0000_0000_0000, 0x0F0F_0F0F_0F0F_0F0F];
    ctx.v[2] = [0x00FF_00FF_00FF_00FF, 0xFFFF_FFFF_FFFF_FFFF];
    let code = build_code(&[0x4EE2_1C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFF00_FF00_FF00_FF00);
    assert_eq!(ctx.v[0][1], 0x0F0F_0F0F_0F0F_0F0F);
}

#[test]
fn vec_neg_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0001_FFFF_FFFE, 0x8000_0000_7FFF_FFFF];
    let code = build_code(&[0x6EA0_B820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_0000_0002);
    assert_eq!(ctx.v[0][1], 0x8000_0000_8000_0001);
}

#[test]
fn vec_abs_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0001_FFFF_FFFE, 0x8000_0001_7FFF_FFFF];
    let code = build_code(&[0x4EA0_B820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0001_0000_0002);
    assert_eq!(ctx.v[0][1], 0x7FFF_FFFF_7FFF_FFFF);
}

#[test]
fn vec_not_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xF0F0_F0F0_F0F0_F0F0, 0x0123_4567_89AB_CDEF];
    let code = build_code(&[0x6E20_5820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], !0xF0F0_F0F0_F0F0_F0F0);
    assert_eq!(ctx.v[0][1], !0x0123_4567_89AB_CDEF);
}

#[test]
fn vec_mul_8h() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0003_0002_0001_0000, 0x0007_0006_0005_0004];
    ctx.v[2] = [0x0010_0010_0010_0010, 0x0010_0010_0010_0010];
    let code = build_code(&[0x4E62_9C20, 0xD4200000]);
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
    let code = build_code(&[0x4EA2_9C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0006_FFFF_FFFE);
    assert_eq!(ctx.v[0][1], 0x0000_000F_0000_0040);
}

#[test]
fn vec_shl_imm_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0001_0000_00FF, 0x0000_0010_0000_0100];
    let code = build_code(&[0x4F22_5420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0004_0000_03FC);
    assert_eq!(ctx.v[0][1], 0x0000_0040_0000_0400);
}

#[test]
fn vec_ushr_imm_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_FFFF_8000_0000, 0xF000_0000_0000_0010];
    let code = build_code(&[0x6F3E_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_3FFF_2000_0000);
    assert_eq!(ctx.v[0][1], 0x3C00_0000_0000_0004);
}

#[test]
fn vec_sshr_imm_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_FFFF_8000_0000, 0xF000_0000_0000_0010];
    let code = build_code(&[0x4F3E_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_3FFF_E000_0000);
    assert_eq!(ctx.v[0][1], 0xFC00_0000_0000_0004);
}

#[test]
fn vec_cmeq_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0002_0000_0001, 0x0000_0004_0000_0003];
    ctx.v[2] = [0x0000_0099_0000_0001, 0x0000_0004_0000_0099];
    let code = build_code(&[0x6EA2_8C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0000_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn vec_cmgt_signed_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFD, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[0x4EA2_3420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn vec_cmge_signed_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0003, 0xFFFF_FFFE_FFFF_FFFE];
    ctx.v[2] = [0x0000_0005_0000_0004, 0xFFFF_FFFE_FFFF_FFFD];
    let code = build_code(&[0x4EA2_3C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_0000_0000);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn vec_cmhi_unsigned_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFD, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[0x6EA2_3420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0xFFFF_FFFF_0000_0000);
}

#[test]
fn vec_cmhs_unsigned_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0003, 0x8000_0000_FFFF_FFFF];
    ctx.v[2] = [0x0000_0005_0000_0004, 0xFFFF_FFFF_FFFF_FFFF];
    let code = build_code(&[0x6EA2_3C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_0000_0000);
    assert_eq!(ctx.v[0][1], 0x0000_0000_FFFF_FFFF);
}

#[test]
fn vec_bit_inserts_when_mask_set() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[2] = [0xFF00_FF00_FF00_FF00, 0x0000_FFFF_FFFF_0000];
    let code = build_code(&[0x6EA2_1C20, 0xD4200000]);
    run(code, &mut ctx);
    let exp0 = (0xAAAA_AAAA_AAAA_AAAAu64 & !0xFF00_FF00_FF00_FF00)
        | (0x1234_5678_9ABC_DEF0 & 0xFF00_FF00_FF00_FF00);
    let exp1 = (0xBBBB_BBBB_BBBB_BBBBu64 & !0x0000_FFFF_FFFF_0000)
        | (0xCAFE_BABE_DEAD_BEEF & 0x0000_FFFF_FFFF_0000);
    assert_eq!(ctx.v[0][0], exp0);
    assert_eq!(ctx.v[0][1], exp1);
}

#[test]
fn vec_bif_inserts_when_mask_clear() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_DEAD_BEEF];
    ctx.v[2] = [0xFF00_FF00_FF00_FF00, 0x0000_FFFF_FFFF_0000];
    let code = build_code(&[0x6EE2_1C20, 0xD4200000]);
    run(code, &mut ctx);
    let exp0 = (0xAAAA_AAAA_AAAA_AAAAu64 & 0xFF00_FF00_FF00_FF00)
        | (0x1234_5678_9ABC_DEF0 & !0xFF00_FF00_FF00_FF00);
    let exp1 = (0xBBBB_BBBB_BBBB_BBBBu64 & 0x0000_FFFF_FFFF_0000)
        | (0xCAFE_BABE_DEAD_BEEF & !0x0000_FFFF_FFFF_0000);
    assert_eq!(ctx.v[0][0], exp0);
    assert_eq!(ctx.v[0][1], exp1);
}

#[test]
fn vec_bsl_selects_per_bit() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xFF00_FF00_FF00_FF00, 0x0000_FFFF_FFFF_0000];
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    ctx.v[2] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    let code = build_code(&[0x6E62_1C20, 0xD4200000]);
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
    let code = build_code(&[0x4E04_0C20, 0xD4200000]);
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
    let code = build_code(&[0x4E01_0C20, 0xD4200000]);
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
    let code = build_code(&[0x4E08_0C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x1122_3344_5566_7788);
    assert_eq!(ctx.v[0][1], 0x1122_3344_5566_7788);
}

#[test]
fn vec_umov_w_from_s_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x9999_9999_1111_1111, 0xDEAD_BEEF_2222_2222];
    let code = build_code(&[0x0E14_3C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0x2222_2222);
}

#[test]
fn vec_smov_x_from_b_lane_sign_extends() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0000_0080_0000, 0];
    let code = build_code(&[0x4E05_2C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.x[0], 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
fn vec_ins_b_lane_from_gpr_preserves_rest() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.x[1] = 0x12_34_56_78_9A_BC_DE_F0;
    let code = build_code(&[0x4E07_1C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xAAAA_AAAA_F0AA_AAAA);
    assert_eq!(ctx.v[0][1], 0xBBBB_BBBB_BBBB_BBBB);
}

#[test]
fn vec_dup_4s_from_element() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xAAAA_AAAA_1111_1111, 0xDEAD_BEEF_2222_2222];
    let code = build_code(&[0x4E14_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x2222_2222_2222_2222);
    assert_eq!(ctx.v[0][1], 0x2222_2222_2222_2222);
}

#[test]
fn vec_ext_byte_offset_4() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    ctx.v[2] = [0x1716_1514_1312_1110, 0x1F1E_1D1C_1B1A_1918];
    let code = build_code(&[0x6E02_2020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0B0A_0908_0706_0504);
    assert_eq!(ctx.v[0][1], 0x1312_1110_0F0E_0D0C);
}

#[test]
fn vec_zip1_4s_interleaves_low_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[0x4E82_3820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_00B0_0000_00A0);
    assert_eq!(ctx.v[0][1], 0x0000_00B1_0000_00A1);
}

#[test]
fn vec_zip2_4s_interleaves_high_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[0x4E82_7820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_00B2_0000_00A2);
    assert_eq!(ctx.v[0][1], 0x0000_00B3_0000_00A3);
}

#[test]
fn vec_zip1_8h_interleaves_per_halfword() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA3A3_A2A2_A1A1_A0A0, 0xA7A7_A6A6_A5A5_A4A4];
    ctx.v[2] = [0xB3B3_B2B2_B1B1_B0B0, 0xB7B7_B6B6_B5B5_B4B4];
    let code = build_code(&[0x4E42_3820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xB1B1_A1A1_B0B0_A0A0);
    assert_eq!(ctx.v[0][1], 0xB3B3_A3A3_B2B2_A2A2);
}

#[test]
fn vec_smax_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_0000_0001, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[0x4EA2_6420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0005_0000_0001);
    assert_eq!(ctx.v[0][1], 0x7FFF_FFFF_FFFF_FFFF);
}

#[test]
fn vec_smin_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_FFFF_FFFE, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_0000_0001, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[0x4EA2_6C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0003_FFFF_FFFE);
    assert_eq!(ctx.v[0][1], 0x0000_0000_8000_0000);
}

#[test]
fn vec_umax_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0001, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFE, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[0x6EA2_6420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0005_FFFF_FFFE);
    assert_eq!(ctx.v[0][1], 0x7FFF_FFFF_FFFF_FFFF);
}

#[test]
fn vec_umin_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0005_0000_0001, 0x7FFF_FFFF_8000_0000];
    ctx.v[2] = [0x0000_0003_FFFF_FFFE, 0x0000_0000_FFFF_FFFF];
    let code = build_code(&[0x6EA2_6C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0003_0000_0001);
    assert_eq!(ctx.v[0][1], 0x0000_0000_8000_0000);
}

#[test]
fn vec_addv_4s_sums_all_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0020_0000_0010, 0x0000_0040_0000_0030];
    let code = build_code(&[0x4EB1_B820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0] as u32, 0xA0);
    assert_eq!(ctx.v[0][0] >> 32, 0, "upper 32 of lane 0 zeroed");
    assert_eq!(ctx.v[0][1], 0, "upper 64 zeroed");
}

#[test]
fn vec_fadd_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (1.0_f32).to_bits() as u64;
    let v1_hi = ((4.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_lo = ((20.0_f32).to_bits() as u64) << 32 | (10.0_f32).to_bits() as u64;
    let v2_hi = ((40.0_f32).to_bits() as u64) << 32 | (30.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    ctx.v[2] = [v2_lo, v2_hi];
    let code = build_code(&[0x4E22_D420, 0xD4200000]);
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
    let code = build_code(&[0x6E62_DC20, 0xD4200000]);
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
    let code = build_code(&[0x0E22_D420, 0xD4200000]);
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
    let code = build_code(&[0x6EA0_F820, 0xD4200000]);
    run(code, &mut ctx);
    let exp_lo = ((2.0_f32).to_bits() as u64) << 32 | ((-1.5_f32).to_bits() as u64);
    let exp_hi = ((-0.0_f32).to_bits() as u64) << 32 | ((-3.14_f32).to_bits() as u64);
    assert_eq!(ctx.v[0][0], exp_lo);
    assert_eq!(ctx.v[0][1], exp_hi);
}

#[test]
fn vec_fabs_2d_strips_sign() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(-7.5_f64).to_bits(), (3.14_f64).to_bits()];
    let code = build_code(&[0x4EE0_F820, 0xD4200000]);
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
    let code = build_code(&[0x6EA1_F820, 0xD4200000]);
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
    ctx.v[1] = [0x0403_0201_FF7F_1080, 0xDEAD_BEEF_CAFE_BABE];
    ctx.v[2] = [0xFCFD_FEFF_0101_F001, 0xDEAD_BEEF_CAFE_BABE];
    let code = build_code(&[0x0E22_0020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0080_0000_FF81);
    assert_eq!(ctx.v[0][1], 0x0000_0000_0000_0000);
}

#[test]
fn vec_uaddl_8h_unsigned_widening_add() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0403_0201_FF7F_1080, 0];
    ctx.v[2] = [0x0000_0000_0101_F001, 0];
    let code = build_code(&[0x2E22_0020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0100_0080_0100_0081);
    assert_eq!(ctx.v[0][1], 0x0004_0003_0002_0001);
}

#[test]
fn vec_saddl2_8h_reads_high_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xDEAD_BEEF_CAFE_BABE, 0x0403_0201_FF7F_1080];
    ctx.v[2] = [0xDEAD_BEEF_CAFE_BABE, 0xFCFD_FEFF_0101_F001];
    let code = build_code(&[0x4E22_0020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0080_0000_FF81);
    assert_eq!(ctx.v[0][1], 0x0000_0000_0000_0000);
}

#[test]
fn vec_xtn_8b_truncates_each_h_lane() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_FACE_FEED];
    let code = build_code(&[0x0E21_2820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFEBE_CEED_3478_BCF0);
    assert_eq!(ctx.v[0][1], 0, "upper 64 zeroed for XTN.8B");
}

#[test]
fn vec_tbl_16b_single_table_with_out_of_range() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0x07_C8_64_10_0F_0A_05_00, 0x0B_09_08_06_04_03_02_01];
    let code = build_code(&[0x4E02_0020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xA7_00_00_00_AF_AA_A5_A0);
    assert_eq!(ctx.v[0][1], 0xAB_A9_A8_A6_A4_A3_A2_A1);
}

#[test]
fn vec_tbl_8b_zeros_upper_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    ctx.v[2] = [0x0E_0C_0A_08_06_04_02_00, 0xDEAD_BEEF_CAFE_BABE];
    let code = build_code(&[0x0E02_0020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0E_0C_0A_08_06_04_02_00);
    assert_eq!(ctx.v[0][1], 0, "8B form zeros upper 64");
}

#[test]
fn vec_rev16_16b_swaps_byte_pairs() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[0x4E20_1820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0607_0405_0203_0001);
    assert_eq!(ctx.v[0][1], 0x0E0F_0C0D_0A0B_0809);
}

#[test]
fn vec_rev32_4s_byte_reverse_within_word() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[0x6E20_0820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0405_0607_0001_0203);
    assert_eq!(ctx.v[0][1], 0x0C0D_0E0F_0809_0A0B);
}

#[test]
fn vec_rev32_8h_swap_halfwords_within_word() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[0x6E60_0820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0504_0706_0100_0302);
    assert_eq!(ctx.v[0][1], 0x0D0C_0F0E_0908_0B0A);
}

#[test]
fn vec_rev64_16b_full_qword_byte_reverse() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[0x4E20_0820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0001_0203_0405_0607);
    assert_eq!(ctx.v[0][1], 0x0809_0A0B_0C0D_0E0F);
}

#[test]
fn vec_rev64_4s_swap_words_within_qword() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0x0F0E_0D0C_0B0A_0908];
    let code = build_code(&[0x4EA0_0820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0302_0100_0706_0504);
    assert_eq!(ctx.v[0][1], 0x0B0A_0908_0F0E_0D0C);
}

#[test]
fn vec_rev16_8b_zeros_upper_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0706_0504_0302_0100, 0xDEAD_BEEF_CAFE_BABE];
    let code = build_code(&[0x0E20_1820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0607_0405_0203_0001);
    assert_eq!(ctx.v[0][1], 0, "8B form zeros upper 64");
}

#[test]
fn vec_uzp1_4s_picks_even_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[0x4E82_1820, 0xD4200000]);
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
    let code = build_code(&[0x4E82_5820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_00A3_0000_00A1);
    assert_eq!(ctx.v[0][1], 0x0000_00B3_0000_00B1);
}

#[test]
fn vec_trn1_4s_transposes_even_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[0x4E82_2820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_00B0_0000_00A0);
    assert_eq!(ctx.v[0][1], 0x0000_00B2_0000_00A2);
}

#[test]
fn vec_trn2_4s_transposes_odd_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_00A1_0000_00A0, 0x0000_00A3_0000_00A2];
    ctx.v[2] = [0x0000_00B1_0000_00B0, 0x0000_00B3_0000_00B2];
    let code = build_code(&[0x4E82_6820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_00B1_0000_00A1);
    assert_eq!(ctx.v[0][1], 0x0000_00B3_0000_00A3);
}

#[test]
fn vec_uzp1_16b_picks_even_bytes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0xB7B6_B5B4_B3B2_B1B0, 0xBFBE_BDBC_BBBA_B9B8];
    let code = build_code(&[0x4E02_1820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xAEAC_AAA8_A6A4_A2A0);
    assert_eq!(ctx.v[0][1], 0xBEBC_BAB8_B6B4_B2B0);
}

#[test]
fn vec_uzp1_2d_picks_low_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    ctx.v[2] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
    let code = build_code(&[0x4EC2_1820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x1111_1111_1111_1111);
    assert_eq!(ctx.v[0][1], 0x3333_3333_3333_3333);
}

#[test]
fn vec_uzp2_2d_picks_high_halves() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x1111_1111_1111_1111, 0x2222_2222_2222_2222];
    ctx.v[2] = [0x3333_3333_3333_3333, 0x4444_4444_4444_4444];
    let code = build_code(&[0x4EC2_5820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x2222_2222_2222_2222);
    assert_eq!(ctx.v[0][1], 0x4444_4444_4444_4444);
}

#[test]
fn vec_trn1_8h_transposes_even_h_lanes() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA3A3_A2A2_A1A1_A0A0, 0xA7A7_A6A6_A5A5_A4A4];
    ctx.v[2] = [0xB3B3_B2B2_B1B1_B0B0, 0xB7B7_B6B6_B5B5_B4B4];
    let code = build_code(&[0x4E42_2820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xB2B2_A2A2_B0B0_A0A0);
    assert_eq!(ctx.v[0][1], 0xB6B6_A6A6_B4B4_A4A4);
}

#[test]
fn vec_ssubl_8h_signed_widening_sub() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0403_0201_FF7F_1080, 0];
    ctx.v[2] = [0xFCFD_FEFF_0101_F001, 0];
    let code = build_code(&[0x0E22_2020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFE_007E_0020_FF7F);
    assert_eq!(ctx.v[0][1], 0x0008_0006_0004_0002);
}

#[test]
fn vec_smull_4s_widening_mul() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let pack_h = |a: i16, b: i16, c: i16, d: i16| -> u64 {
        (a as u16 as u64)
            | ((b as u16 as u64) << 16)
            | ((c as u16 as u64) << 32)
            | ((d as u16 as u64) << 48)
    };
    ctx.v[1] = [pack_h(3, -2, 100, 0x7FFF), 0];
    ctx.v[2] = [pack_h(4, 5, -10, 2), 0];
    let code = build_code(&[0x0E62_C020, 0xD4200000]);
    run(code, &mut ctx);
    let pack_s = |a: i32, b: i32| -> u64 { (a as u32 as u64) | ((b as u32 as u64) << 32) };
    assert_eq!(ctx.v[0][0], pack_s(12, -10));
    assert_eq!(ctx.v[0][1], pack_s(-1000, 65534));
}

#[test]
fn vec_umull_8h_widening_unsigned_mul() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0807_0605_0403_0201, 0];
    ctx.v[2] = [0x1010_1010_1010_1010, 0];
    let code = build_code(&[0x2E22_C020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0040_0030_0020_0010);
    assert_eq!(ctx.v[0][1], 0x0080_0070_0060_0050);
}

#[test]
fn vec_cmhi_8b_unsigned() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_01_FF_7F_05_00_FE_03, 0];
    ctx.v[2] = [0x01_02_03_7F_05_FF_FD_04, 0];
    let code = build_code(&[0x2E22_3420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFF_00_FF_00_00_00_FF_00);
    assert_eq!(ctx.v[0][1], 0);
}

#[test]
fn vec_cmhs_16b_unsigned() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_01_FF_7F_05_00_FE_03, 0x10_20_30_40_50_60_70_80];
    ctx.v[2] = [0x01_02_03_7F_05_FF_FD_04, 0x11_20_2F_40_50_61_70_FF];
    let code = build_code(&[0x6E22_3C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFF_00_FF_FF_FF_00_FF_00);
    assert_eq!(ctx.v[0][1], 0x00_FF_FF_FF_FF_00_FF_00);
}

#[test]
fn vec_xtn2_16b_preserves_low_half() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0xAAAA_AAAA_AAAA_AAAA, 0xBBBB_BBBB_BBBB_BBBB];
    ctx.v[1] = [0x1234_5678_9ABC_DEF0, 0xCAFE_BABE_FACE_FEED];
    let code = build_code(&[0x4E21_2820, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xAAAA_AAAA_AAAA_AAAA);
    assert_eq!(ctx.v[0][1], 0xFEBE_CEED_3478_BCF0);
}

#[test]
fn vec_fcmeq_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (1.0_f32).to_bits() as u64;
    let v1_hi = ((f32::NAN).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_lo = ((2.0_f32).to_bits() as u64) << 32 | (5.0_f32).to_bits() as u64;
    let v2_hi = ((1.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    ctx.v[2] = [v2_lo, v2_hi];
    let code = build_code(&[0x4E22_E420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFFFFFF_00000000);
    assert_eq!(ctx.v[0][1], 0x00000000_FFFFFFFF);
}

#[test]
fn vec_fcmgt_4s() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let v1_lo = ((2.0_f32).to_bits() as u64) << 32 | (5.0_f32).to_bits() as u64;
    let v1_hi = ((f32::NAN).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_lo = ((7.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    let v2_hi = ((1.0_f32).to_bits() as u64) << 32 | (3.0_f32).to_bits() as u64;
    ctx.v[1] = [v1_lo, v1_hi];
    ctx.v[2] = [v2_lo, v2_hi];
    let code = build_code(&[0x6EA2_E420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x00000000_FFFFFFFF);
    assert_eq!(ctx.v[0][1], 0x00000000_00000000);
}

#[test]
fn vec_fcmge_2d() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(2.5_f64).to_bits(), (f64::NAN).to_bits()];
    ctx.v[2] = [(2.5_f64).to_bits(), (1.0_f64).to_bits()];
    let code = build_code(&[0x6E62_E420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFFFFFF_FFFFFFFF);
    assert_eq!(ctx.v[0][1], 0);
}

#[test]
fn vec_fmla_4s_accumulates_product() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let pack_s = |a: f32, b: f32| -> u64 { (a.to_bits() as u64) | ((b.to_bits() as u64) << 32) };
    ctx.v[0] = [pack_s(1.0, 2.0), pack_s(3.0, 4.0)];
    ctx.v[1] = [pack_s(10.0, 10.0), pack_s(10.0, 10.0)];
    ctx.v[2] = [pack_s(2.0, 3.0), pack_s(4.0, 5.0)];
    let code = build_code(&[0x4E22_CC20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], pack_s(21.0, 32.0));
    assert_eq!(ctx.v[0][1], pack_s(43.0, 54.0));
}

#[test]
fn vec_fmls_2d_subtracts_product() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [(100.0_f64).to_bits(), (50.0_f64).to_bits()];
    ctx.v[1] = [(10.0_f64).to_bits(), (5.0_f64).to_bits()];
    ctx.v[2] = [(3.0_f64).to_bits(), (4.0_f64).to_bits()];
    let code = build_code(&[0x4EE2_CC20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(f64::from_bits(ctx.v[0][0]), 70.0);
    assert_eq!(f64::from_bits(ctx.v[0][1]), 30.0);
}

#[test]
fn vec_shl_imm_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_40_20_10_08_04_02_01, 0xFF_7F_3F_1F_0F_07_03_01];
    let code = build_code(&[0x4F0A_5420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x00_00_80_40_20_10_08_04);
    assert_eq!(ctx.v[0][1], 0xFC_FC_FC_7C_3C_1C_0C_04);
}

#[test]
fn vec_ushr_imm_16b() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_40_20_10_08_04_02_01, 0xFF_7F_3F_1F_0F_07_03_01];
    let code = build_code(&[0x6F0E_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x20_10_08_04_02_01_00_00);
    assert_eq!(ctx.v[0][1], 0x3F_1F_0F_07_03_01_00_00);
}

#[test]
fn vec_sshr_imm_16b_sign_extends() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x80_40_20_10_08_04_02_01, 0xFF_7F_3F_1F_0F_07_03_01];
    let code = build_code(&[0x4F0E_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xE0_10_08_04_02_01_00_00);
    assert_eq!(ctx.v[0][1], 0xFF_1F_0F_07_03_01_00_00);
}

#[test]
fn vec_mul_2d_via_decomposition() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [3, 0x0000_0001_0000_0001];
    ctx.v[2] = [7, 5];
    let code = build_code(&[0x4EE2_9C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 21);
    assert_eq!(ctx.v[0][1], 0x0000_0005_0000_0005);
}

#[test]
fn vec_mul_2d_wraps_at_64() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF];
    ctx.v[2] = [2, 0xFFFF_FFFF_FFFF_FFFF];
    let code = build_code(&[0x4EE2_9C20, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0);
    assert_eq!(ctx.v[0][1], 1);
}

#[test]
fn vec_smull_2d_via_pmovsxdq() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let pack_s = |a: i32, b: i32| -> u64 { (a as u32 as u64) | ((b as u32 as u64) << 32) };
    ctx.v[1] = [pack_s(-3, 1_000_000), 0];
    ctx.v[2] = [pack_s(7, 5), 0];
    let code = build_code(&[0x0EA2_C020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0] as i64, -21);
    assert_eq!(ctx.v[0][1], 5_000_000);
}

#[test]
fn vec_umull_2d_via_pmovzxdq() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let pack_s = |a: u32, b: u32| -> u64 { (a as u64) | ((b as u64) << 32) };
    ctx.v[1] = [pack_s(0xFFFF_FFFF, 0x1234_5678), 0];
    ctx.v[2] = [pack_s(2, 0xCAFE_BABE), 0];
    let code = build_code(&[0x2EA2_C020, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFFFFFFu64 * 2);
    assert_eq!(ctx.v[0][1], 0x1234_5678u64 * 0xCAFE_BABEu64);
}

#[test]
fn vec_sshr_imm_2d_arithmetic_shift() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0x0000_0000_0000_0080, 0x8000_0000_0000_0000];
    let code = build_code(&[0x4F7C_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0x0000_0000_0000_0008);
    assert_eq!(ctx.v[0][1], 0xF800_0000_0000_0000);
}

#[test]
fn vec_sshr_imm_2d_by_one() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xFFFF_FFFF_FFFF_FFFF, 0x4000_0000_0000_0000];
    let code = build_code(&[0x4F7F_0420, 0xD4200000]);
    run(code, &mut ctx);
    assert_eq!(ctx.v[0][0], 0xFFFF_FFFF_FFFF_FFFF);
    assert_eq!(ctx.v[0][1], 0x2000_0000_0000_0000);
}

#[test]
fn vec_tbl2_16b_two_register_table() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0xB7B6_B5B4_B3B2_B1B0, 0xBFBE_BDBC_BBBA_B9B8];
    let mut idx = [0u8; 16];
    let want = [
        0u8, 5, 16, 31, 32, 200, 17, 15, 3, 18, 50, 7, 29, 1, 100, 14,
    ];
    for (i, &v) in want.iter().enumerate() {
        idx[i] = v;
    }
    let mut lo = [0u8; 8];
    lo.copy_from_slice(&idx[..8]);
    let mut hi = [0u8; 8];
    hi.copy_from_slice(&idx[8..]);
    ctx.v[3] = [u64::from_le_bytes(lo), u64::from_le_bytes(hi)];

    let code = build_code(&[0x4E03_2020, 0xD4200000]);
    run(code, &mut ctx);
    let want = [
        0xA0u8, 0xA5, 0xB0, 0xBF, 0x00, 0x00, 0xB1, 0xAF, 0xA3, 0xB2, 0x00, 0xA7, 0xBD, 0xA1, 0x00,
        0xAE,
    ];
    let mut lo = [0u8; 8];
    lo.copy_from_slice(&want[..8]);
    let mut hi = [0u8; 8];
    hi.copy_from_slice(&want[8..]);
    assert_eq!(ctx.v[0][0], u64::from_le_bytes(lo));
    assert_eq!(ctx.v[0][1], u64::from_le_bytes(hi));
}

#[test]
fn vec_tbl3_16b_three_register_table() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [0xA7A6_A5A4_A3A2_A1A0, 0xAFAE_ADAC_ABAA_A9A8];
    ctx.v[2] = [0xB7B6_B5B4_B3B2_B1B0, 0xBFBE_BDBC_BBBA_B9B8];
    ctx.v[3] = [0xC7C6_C5C4_C3C2_C1C0, 0xCFCE_CDCC_CBCA_C9C8];
    let want_idx = [
        0u8, 16, 32, 47, 48, 200, 33, 15, 3, 18, 35, 7, 31, 1, 100, 46,
    ];
    let mut idx = [0u8; 16];
    for (i, &v) in want_idx.iter().enumerate() {
        idx[i] = v;
    }
    let mut lo = [0u8; 8];
    lo.copy_from_slice(&idx[..8]);
    let mut hi = [0u8; 8];
    hi.copy_from_slice(&idx[8..]);
    ctx.v[4] = [u64::from_le_bytes(lo), u64::from_le_bytes(hi)];

    let code = build_code(&[0x4E04_4020, 0xD4200000]);
    run(code, &mut ctx);
    let want = [
        0xA0u8, 0xB0, 0xC0, 0xCF, 0x00, 0x00, 0xC1, 0xAF, 0xA3, 0xB2, 0xC3, 0xA7, 0xBF, 0xA1, 0x00,
        0xCE,
    ];
    let mut lo = [0u8; 8];
    lo.copy_from_slice(&want[..8]);
    let mut hi = [0u8; 8];
    hi.copy_from_slice(&want[8..]);
    assert_eq!(ctx.v[0][0], u64::from_le_bytes(lo));
    assert_eq!(ctx.v[0][1], u64::from_le_bytes(hi));
}

#[test]
fn vec_fmls_nan_input_clears_sign() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[1] = [(1.5_f32).to_bits() as u64, 0];
    ctx.v[2] = [0xFFFFFFFFu64, 0];
    ctx.v[3] = [(2.0_f32).to_bits() as u64, 0];
    let code = build_code(&[0x0EA3_CC41, 0xD4200000]);
    run(code, &mut ctx);
    let lane0 = ctx.v[1][0] as u32;
    assert!(
        f32::from_bits(lane0).is_nan(),
        "result lane 0 should be NaN, got {:#010x}",
        lane0
    );
    assert_eq!(
        lane0 >> 31,
        0,
        "NaN sign bit must be clear after FMLS, got {:#010x}",
        lane0
    );
}

#[test]
fn vec_frint_family_dynarmic_test() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.v[0] = [0x4001e17c4001e17c, 0x4001e17c4001e17c];
    let code = build_code(&[
        0x4E218801, 0x4E219802, 0x4EA18803, 0x4EA19804, 0x6E218805, 0x6E219806, 0xD4200000,
    ]);
    run(code, &mut ctx);
    assert_eq!(
        ctx.v[0],
        [0x4001e17c4001e17c, 0x4001e17c4001e17c],
        "input preserved"
    );
    assert_eq!(ctx.v[1], [0x4000000040000000, 0x4000000040000000], "FRINTN");
    assert_eq!(ctx.v[2], [0x4000000040000000, 0x4000000040000000], "FRINTM");
    assert_eq!(ctx.v[3], [0x4040000040400000, 0x4040000040400000], "FRINTP");
    assert_eq!(ctx.v[4], [0x4000000040000000, 0x4000000040000000], "FRINTZ");
    assert_eq!(ctx.v[5], [0x4000000040000000, 0x4000000040000000], "FRINTA");
    assert_eq!(ctx.v[6], [0x4000000040000000, 0x4000000040000000], "FRINTX");
}

#[test]
fn vec_frinta_ties_away_from_zero() {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let pack = |a: f32, b: f32| (a.to_bits() as u64) | ((b.to_bits() as u64) << 32);
    ctx.v[0] = [pack(0.5, 1.5), pack(2.5, -0.5)];
    let code = build_code(&[0x6E218801, 0xD4200000]);
    run(code, &mut ctx);
    let want = [pack(1.0, 2.0), pack(3.0, -1.0)];
    assert_eq!(ctx.v[1], want);
}
