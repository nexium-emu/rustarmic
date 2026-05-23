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
    let pick: u32 = rng.r#gen_range(0..6);
    let rd: u32 = rng.r#gen_range(0..28);
    let rn: u32 = rng.r#gen_range(0..28);
    let rm: u32 = rng.r#gen_range(0..28);
    let imm12: u32 = rng.r#gen_range(0..0x1000);
    match pick {
        0 => 0x9100_0000 | (imm12 << 10) | (rn << 5) | rd, // add  xd, xn, #imm12
        1 => 0xD100_0000 | (imm12 << 10) | (rn << 5) | rd, // sub  xd, xn, #imm12
        2 => 0x8B00_0000 | (rm << 16) | (rn << 5) | rd,    // add  xd, xn, xm
        3 => 0xCB00_0000 | (rm << 16) | (rn << 5) | rd,    // sub  xd, xn, xm
        4 => 0xAA00_0000 | (rm << 16) | (rn << 5) | rd,    // orr  xd, xn, xm
        5 => 0x4A00_0000 | (rm << 16) | (rn << 5) | rd,    // eor  wd, wn, wm (32-bit)
        _ => unreachable!(),
    }
}

#[test]
fn fuzz_small_blocks() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    let mut fail_count = 0;
    let mut tested = 0;
    for case in 0..32 {
        let code = gen_block(&mut rng, rng.r#gen_range(2..8));
        let init = baseline_state(0x12345 + case);
        let (uni, jit) = run_pair(&code, init);
        tested += 1;

        // Compare X0..X27 (we restricted rd to that range above).
        for i in 0..28 {
            if uni.x[i] != jit.x[i] {
                eprintln!("case {} X{} mismatch uni=0x{:x} jit=0x{:x}", case, i, uni.x[i], jit.x[i]);
                fail_count += 1;
            }
        }
    }
    eprintln!("fuzz: tested {} cases, {} register mismatches", tested, fail_count);
    assert_eq!(fail_count, 0, "register mismatches detected");
}
