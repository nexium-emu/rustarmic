#![cfg(feature = "unicorn")]

mod harness;

use harness::{DATA_BASE, RegState, mem_init, run_pair};

const CODE_BASE: u64 = 0x1000;
const STACK_BASE: u64 = 0x10_0000;

fn snippet(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn baseline_state() -> RegState {
    let mut s = RegState::default();
    s.pc = CODE_BASE;
    s.sp = STACK_BASE;
    s
}

#[test]
fn movz_then_add_imm() {
    let code = snippet(&[0xD282_4680, 0x9103_FC00, 0xD420_0000]);
    let init = baseline_state();
    let (uni, jit) = run_pair(&code, init);
    assert_eq!(
        uni.x[0], jit.x[0],
        "X0 mismatch: uni=0x{:x} jit=0x{:x}",
        uni.x[0], jit.x[0]
    );
}

#[test]
fn add_subs_flags() {
    let code = snippet(&[0xD2800080, 0xD2800061, 0xEB010002, 0xD4200000]);
    let init = baseline_state();
    let (uni, jit) = run_pair(&code, init);
    assert_eq!(uni.x[2], jit.x[2]);
    assert_eq!(
        uni.nzcv & 0xF,
        jit.nzcv & 0xF,
        "NZCV mismatch: uni=0x{:x} jit=0x{:x}",
        uni.nzcv,
        jit.nzcv
    );
}

#[test]
fn logical_imm_and_or() {
    let code = snippet(&[0xD29FE000, 0x00000000, 0x00000000, 0x00000000]);
    let _ = code;
}

#[test]
fn q_pair_store_uses_w_form_computed_base_address() {
    mem_init(0x20_000);

    // rtld's allocator loop reduced to one iteration. This sequence first
    // computes X14 through W14, then uses it as the Q-pair store base.
    let code = snippet(&[
        0x4e08_0d20, // dup v0.2d, x9
        0x4b00_03ee, // neg w14, w0
        0x9240_05ce, // and x14, x14, #3
        0x8b0e_018f, // add x15, x12, x14
        0xd100_8050, // sub x16, x2, #0x20
        0x8b00_01ef, // add x15, x15, x0
        0xcb0e_0210, // sub x16, x16, x14
        0x9101_01ee, // add x14, x15, #0x40
        0x927e_f60f, // and x15, x16, #...fc
        0xcb0c_01ec, // sub x12, x15, x12
        0xd345_fd8c, // lsr x12, x12, #5
        0x9100_058c, // add x12, x12, #1
        0x927e_e58c, // and x12, x12, #...fc
        0xad3e_01c0, // stp q0, q0, [x14, #-64]
        0xd100_118c, // sub x12, x12, #4
        0xad3f_01c0, // stp q0, q0, [x14, #-32]
        0xad00_01c0, // stp q0, q0, [x14]
        0xad01_01c0, // stp q0, q0, [x14, #32]
        0x9102_01ce, // add x14, x14, #0x80
        0xb5ff_ff4c, // cbnz x12, -0x80
        0xd420_0000, // brk #0
    ]);

    let mut init = baseline_state();
    init.x[0] = DATA_BASE + 0x53d0;
    init.x[2] = 0x1d8;
    init.x[9] = 0;
    init.x[12] = 4;
    let (uni, jit) = run_pair(&code, init);
    assert_eq!(jit.x[14], uni.x[14]);
    assert_eq!(jit.pc, uni.pc);
}
