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
    // V regs get full 128 bits of random — bit-exact lane operations need to
    // see varied byte values, not just zeros from default-init.
    for i in 0..32 {
        s.v[i] = [rng.r#gen(), rng.r#gen()];
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

// ─── NEON fuzzing ────────────────────────────────────────────────────────

fn gen_neon_block(rng: &mut ChaCha8Rng, n: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity((n + 1) * 4);
    for _ in 0..n {
        let word = gen_neon_inst(rng);
        out.extend_from_slice(&word.to_le_bytes());
    }
    out.extend_from_slice(&0xD420_0000u32.to_le_bytes());
    out
}

/// Generate a NEON instruction drawn from the set we have backend coverage
/// for. We restrict Vd/Vn/Vm to V0..V15 so the harness compare can ignore
/// the upper range if it ever needs to, and so register reuse is frequent
/// enough to exercise read-after-write paths.
fn gen_neon_inst(rng: &mut ChaCha8Rng) -> u32 {
    // Default to the full op set; tests narrow this via env var when
    // bisecting a mismatch.
    let max_pick: u32 = std::env::var("FUZZ_NEON_MAX")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(28);
    let pick: u32 = rng.r#gen_range(0..max_pick);
    let vd: u32 = rng.r#gen_range(0..16);
    let vn: u32 = rng.r#gen_range(0..16);
    let vm: u32 = rng.r#gen_range(0..16);

    // size: 00=B, 01=H, 10=S, 11=D — caller picks a valid one per op.
    fn enc_same(q: u32, u: u32, size: u32, rm: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (0 << 31) | (q << 30) | (u << 29) | (0b01110 << 24)
            | (size << 22) | (1 << 21) | (rm << 16)
            | (opcode << 11) | (1 << 10) | (rn << 5) | rd
    }
    fn enc_misc(q: u32, u: u32, size: u32, opcode: u32, rn: u32, rd: u32) -> u32 {
        (0 << 31) | (q << 30) | (u << 29) | (0b01110 << 24)
            | (size << 22) | (1 << 21) | (0 << 17) // bits 20:17 zero
            | (opcode << 12) | (0b10 << 10) | (rn << 5) | rd
    }
    /// Pick (Q, size) where size in 0..max_size, but exclude (Q=0, size=3)
    /// which is reserved for most per-lane ops (the 1D form).
    fn pick_q_size(rng: &mut ChaCha8Rng, max_size: u32) -> (u32, u32) {
        let q = rng.r#gen_range(0..2);
        let size = if q == 0 {
            rng.r#gen_range(0..max_size.min(3))
        } else {
            rng.r#gen_range(0..max_size)
        };
        (q, size)
    }

    match pick {
        // ADD/SUB all lane sizes (Q=0 excludes size=3 since 1D is reserved).
        0 => { let (q, s) = pick_q_size(rng, 4); enc_same(q, 0, s, vm, 0b10000, vn, vd) }
        1 => { let (q, s) = pick_q_size(rng, 4); enc_same(q, 1, s, vm, 0b10000, vn, vd) }
        // Logical AND/ORR/EOR/BIC/ORN .16B and .8B
        2 => enc_same(rng.r#gen_range(0..2), 0, 0b00, vm, 0b00011, vn, vd), // AND
        3 => enc_same(rng.r#gen_range(0..2), 0, 0b10, vm, 0b00011, vn, vd), // ORR
        4 => enc_same(rng.r#gen_range(0..2), 1, 0b00, vm, 0b00011, vn, vd), // EOR
        5 => enc_same(rng.r#gen_range(0..2), 0, 0b01, vm, 0b00011, vn, vd), // BIC
        6 => enc_same(rng.r#gen_range(0..2), 0, 0b11, vm, 0b00011, vn, vd), // ORN
        // MUL — only 16-bit (size=01) and 32-bit (size=10) lanes.
        7 => enc_same(rng.r#gen_range(0..2), 0, [1u32, 2].choose_rng(rng), vm, 0b10011, vn, vd),
        // SMAX/SMIN/UMAX/UMIN — 8/16/32-bit lanes only.
        8  => enc_same(rng.r#gen_range(0..2), 0, rng.r#gen_range(0..3), vm, 0b01100, vn, vd), // SMAX
        9  => enc_same(rng.r#gen_range(0..2), 0, rng.r#gen_range(0..3), vm, 0b01101, vn, vd), // SMIN
        10 => enc_same(rng.r#gen_range(0..2), 1, rng.r#gen_range(0..3), vm, 0b01100, vn, vd), // UMAX
        11 => enc_same(rng.r#gen_range(0..2), 1, rng.r#gen_range(0..3), vm, 0b01101, vn, vd), // UMIN
        // CMEQ all sizes (Q=0 excludes size=3); CMGT/CMGE same; CMHI/CMHS 16/32-bit.
        12 => { let (q, s) = pick_q_size(rng, 4); enc_same(q, 1, s, vm, 0b10001, vn, vd) } // CMEQ
        13 => { let (q, s) = pick_q_size(rng, 4); enc_same(q, 0, s, vm, 0b00110, vn, vd) } // CMGT
        14 => { let (q, s) = pick_q_size(rng, 4); enc_same(q, 0, s, vm, 0b00111, vn, vd) } // CMGE
        15 => { let q = rng.r#gen_range(0..2); let s = rng.r#gen_range(1..3); enc_same(q, 1, s, vm, 0b00110, vn, vd) } // CMHI
        16 => { let q = rng.r#gen_range(0..2); let s = rng.r#gen_range(1..3); enc_same(q, 1, s, vm, 0b00111, vn, vd) } // CMHS
        // BIT / BIF / BSL.16B
        17 => enc_same(1, 1, 0b10, vm, 0b00011, vn, vd), // BIT
        18 => enc_same(1, 1, 0b11, vm, 0b00011, vn, vd), // BIF
        19 => enc_same(1, 1, 0b01, vm, 0b00011, vn, vd), // BSL
        // NEG / ABS / NOT (ASIMDMISC). Avoid size=11 for ABS, and exclude
        // (Q=0, size=3) for NEG (reserved 1D form).
        20 => { let (q, s) = pick_q_size(rng, 4); enc_misc(q, 1, s, 0b01011, vn, vd) }    // NEG
        21 => { let (q, s) = pick_q_size(rng, 3); enc_misc(q, 0, s, 0b01011, vn, vd) }    // ABS
        22 => enc_misc(rng.r#gen_range(0..2), 1, 0b00, 0b00101, vn, vd),                  // NOT
        // ZIP1 / ZIP2 (ASIMDPERM). Q=0 size=3 (1D form) is meaningless for
        // ZIP2 and unsupported in our emit; restrict to (Q=1, any size) or
        // (Q=0, size in 0..3).
        23 => {
            let q = rng.r#gen_range(0..2);
            let size = if q == 1 { rng.r#gen_range(0..4) } else { rng.r#gen_range(0..3) };
            let zip2 = rng.r#gen_range(0..2) == 1;
            let opcode_high = if zip2 { 0b111 } else { 0b011 };
            (0 << 31) | (q << 30) | (0 << 29) | (0b01110 << 24)
                | (size << 22) | (0 << 21) | (vm << 16)
                | (0 << 15) | (opcode_high << 12) | (1 << 11) | (0 << 10)
                | (vn << 5) | vd
        }
        // REV16 / REV32 / REV64 (ASIMDMISC). Each container has its own valid
        // size range. opcode (bits 16:12): REV16=0b00001, REV32=0b00000 (U=1),
        // REV64=0b00000 (U=0).
        24 => enc_misc(rng.r#gen_range(0..2), 0, 0b00, 0b00001, vn, vd),                  // REV16: only size=00
        25 => { let s = rng.r#gen_range(0..2); enc_misc(rng.r#gen_range(0..2), 1, s, 0b00000, vn, vd) } // REV32: size 00/01
        26 => { let s = rng.r#gen_range(0..3); enc_misc(rng.r#gen_range(0..2), 0, s, 0b00000, vn, vd) } // REV64: size 00/01/10
        // UZP1 / UZP2 / TRN1 / TRN2 (ASIMDPERM). opcode bits 14:12:
        //   UZP1=001, TRN1=010, UZP2=101, TRN2=110
        // (Q=0, size=11) is reserved (1D), so we restrict size like ZIP.
        27 => {
            let q = rng.r#gen_range(0..2);
            let size = if q == 1 { rng.r#gen_range(0..4) } else { rng.r#gen_range(0..3) };
            let op_pick = rng.r#gen_range(0..4);
            let opcode_high = match op_pick {
                0 => 0b001, // UZP1
                1 => 0b101, // UZP2
                2 => 0b010, // TRN1
                _ => 0b110, // TRN2
            };
            (0 << 31) | (q << 30) | (0 << 29) | (0b01110 << 24)
                | (size << 22) | (0 << 21) | (vm << 16)
                | (0 << 15) | (opcode_high << 12) | (1 << 11) | (0 << 10)
                | (vn << 5) | vd
        }
        _ => unreachable!(),
    }
}

trait ChooseRng<T> { fn choose_rng(&self, rng: &mut ChaCha8Rng) -> T; }
impl<T: Copy> ChooseRng<T> for [T] {
    fn choose_rng(&self, rng: &mut ChaCha8Rng) -> T {
        self[rng.r#gen_range(0..self.len())]
    }
}

fn fuzz_neon_with_seed(seed: u64, cases: u32, min_len: usize, max_len: usize, label: &str) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut fail_count = 0;
    for case in 0..cases {
        let len = rng.r#gen_range(min_len..max_len);
        let code = gen_neon_block(&mut rng, len);
        let init = baseline_state(seed ^ (case as u64));
        let (uni, jit) = run_pair(&code, init);

        let mut case_failed = false;
        for i in 0..16 {
            if uni.v[i] != jit.v[i] {
                if !case_failed {
                    eprint!("{}: case {} code:", label, case);
                    for chunk in code.chunks(4) {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(chunk);
                        eprint!(" {:08x}", u32::from_le_bytes(b));
                    }
                    eprintln!();
                    case_failed = true;
                }
                eprintln!(
                    "  V{} uni=[{:016x},{:016x}] jit=[{:016x},{:016x}]",
                    i, uni.v[i][1], uni.v[i][0], jit.v[i][1], jit.v[i][0],
                );
                fail_count += 1;
            }
        }
        for i in 0..28 {
            if uni.x[i] != jit.x[i] {
                eprintln!(
                    "{}: case {} X{} mismatch uni=0x{:016x} jit=0x{:016x}",
                    label, case, i, uni.x[i], jit.x[i],
                );
                fail_count += 1;
            }
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

#[test]
fn fuzz_neon_small() {
    fuzz_neon_with_seed(0x1234_5678, 64, 2, 8, "neon-small");
}

#[test]
fn fuzz_neon_medium() {
    fuzz_neon_with_seed(0xABCD_EF01, 32, 8, 24, "neon-medium");
}

#[test]
fn fuzz_neon_large() {
    fuzz_neon_with_seed(0xBADC_AFE0, 16, 24, 64, "neon-large");
}
