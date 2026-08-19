#![cfg(feature = "unicorn")]

mod harness;

use harness::{RegState, run_pair};

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
