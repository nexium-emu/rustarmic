//! Random differential fuzzing against Unicorn.
//!
//! We generate a sequence of well-formed AArch64 instructions drawn from the
//! subset rustarmic implements, terminate with BRK, then compare register
//! state against Unicorn. We keep the test counts modest by default so this
//! runs in CI; bump `--features fuzz_long` to crank them up.
//!
//! Enable with `cargo test --features unicorn`.

#![cfg(feature = "unicorn")]

mod harness;

use harness::{run_pair, RegState};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const CODE_BASE: u64 = 0x1000;
const STACK_BASE: u64 = 0x10_0000;

fn baseline_state(seed: u64) -> RegState {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut s = RegState::default();
    s.pc = CODE_BASE;
    s.sp = STACK_BASE;
    for i in 0..31 {
        // Keep values modest so signed/unsigned overflows happen but not on every op.
        s.x[i] = rng.r#gen::<u32>() as u64;
    }
    s
}

fn gen_block(rng: &mut ChaCha8Rng, n: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity((n + 1) * 4);
    for _ in 0..n {
        let word = gen_inst(rng);
        out.extend_from_slice(&word.to_le_bytes());
    }
    // Terminator: BRK #0.
    out.extend_from_slice(&0xD420_0000u32.to_le_bytes());
    out
}

/// Generate a single instruction drawn from a curated, definitely-decodable
/// subset. We avoid the SP encoding (rd=31) when ZR semantics matter.
fn gen_inst(rng: &mut ChaCha8Rng) -> u32 {
    let pick: u32 = rng.r#gen_range(0..14);
    let rd: u32 = rng.r#gen_range(0..28);
    let rn: u32 = rng.r#gen_range(0..28);
    let rm: u32 = rng.r#gen_range(0..28);
    let imm12: u32 = rng.r#gen_range(0..0x1000);
    let shift_amt: u32 = rng.r#gen_range(0..63);
    match pick {
        0  => 0x9100_0000 | (imm12 << 10) | (rn << 5) | rd, // add  xd, xn, #imm12
        1  => 0xD100_0000 | (imm12 << 10) | (rn << 5) | rd, // sub  xd, xn, #imm12
        2  => 0x8B00_0000 | (rm << 16) | (rn << 5) | rd,    // add  xd, xn, xm
        3  => 0xCB00_0000 | (rm << 16) | (rn << 5) | rd,    // sub  xd, xn, xm
        4  => 0xAA00_0000 | (rm << 16) | (rn << 5) | rd,    // orr  xd, xn, xm
        5  => 0x4A00_0000 | (rm << 16) | (rn << 5) | rd,    // eor  wd, wn, wm (32-bit)
        6  => 0x8A00_0000 | (rm << 16) | (rn << 5) | rd,    // and  xd, xn, xm
        7  => 0xCA00_0000 | (rm << 16) | (rn << 5) | rd,    // eor  xd, xn, xm
        // mul = madd Xd, Xn, Xm, XZR
        8  => 0x9B00_7C00 | (rm << 16) | (rn << 5) | rd,
        // lsl xd, xn, #imm6 (LSL imm = UBFM Xd, Xn, #(-imm % 64), #(63 - imm))
        9  => {
            let s = shift_amt & 63;
            let immr = (64u32 - s) & 63;
            let imms = 63 - s;
            0xD340_0000 | (immr << 16) | (imms << 10) | (rn << 5) | rd
        }
        // lsr xd, xn, #imm6 (LSR imm = UBFM Xd, Xn, #imm, #63)
        10 => {
            let s = shift_amt & 63;
            0xD340_0000 | (s << 16) | (63 << 10) | (rn << 5) | rd
        }
        // asr xd, xn, #imm6 (ASR imm = SBFM Xd, Xn, #imm, #63)
        11 => {
            let s = shift_amt & 63;
            0x9340_0000 | (s << 16) | (63 << 10) | (rn << 5) | rd
        }
        // adds xd, xn, #imm12  (writes flags)
        12 => 0xB100_0000 | (imm12 << 10) | (rn << 5) | rd,
        // subs xd, xn, #imm12  (writes flags)
        13 => 0xF100_0000 | (imm12 << 10) | (rn << 5) | rd,
        _  => unreachable!(),
    }
}

fn fuzz_with_seed(seed: u64, cases: u32, min_len: usize, max_len: usize, label: &str) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut fail_count = 0;
    for case in 0..cases {
        let len = rng.r#gen_range(min_len..max_len);
        let code = gen_block(&mut rng, len);
        let init = baseline_state(seed ^ (case as u64));
        let (uni, jit) = run_pair(&code, init);

        // Compare X0..X27 (gen_inst restricts rd/rn/rm to that range).
        for i in 0..28 {
            if uni.x[i] != jit.x[i] {
                eprintln!(
                    "{}: case {} X{} mismatch uni=0x{:016x} jit=0x{:016x}",
                    label, case, i, uni.x[i], jit.x[i],
                );
                fail_count += 1;
            }
        }
        // Compare NZCV (low nibble).
        if uni.nzcv != jit.nzcv {
            eprintln!(
                "{}: case {} NZCV mismatch uni={:04b} jit={:04b}",
                label, case, uni.nzcv, jit.nzcv,
            );
            fail_count += 1;
        }
    }
    assert_eq!(fail_count, 0, "{}: {} mismatches", label, fail_count);
}

#[test]
fn fuzz_small_blocks() {
    fuzz_with_seed(0xC0FFEE, 64, 2, 8, "small");
}

#[test]
fn fuzz_medium_blocks() {
    fuzz_with_seed(0xDEAD_BEEF, 32, 8, 24, "medium");
}

#[test]
fn fuzz_large_blocks() {
    fuzz_with_seed(0xFEED_FACE, 16, 24, 64, "large");
}
