//! AdvSIMD structure loads / stores (LD1-LD4, ST1-ST4, LD1R-LD4R in every
//! register count, arrangement, lane and writeback form) against Unicorn,
//! with page-table fastmem and with every access through the callbacks.
#![cfg(feature = "unicorn")]

mod simd_harness;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use simd_harness::{BRK_0, DATA, DATA_SIZE, Exit, Rj, State, Uni};

fn cases() -> u64 {
    std::env::var("RUSTARMIC_LDST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4000)
}

fn random_state(rng: &mut ChaCha8Rng) -> State {
    let mut s = State::new();
    for x in s.x.iter_mut() {
        *x = rng.r#gen();
    }
    for v in s.v.iter_mut() {
        *v = [rng.r#gen(), rng.r#gen()];
    }
    rng.fill(&mut s.data[..]);
    s.nzcv = rng.gen_range(0..16);
    s
}

/// One random structure load / store plus the base / offset register
/// values that keep the whole transfer inside the data region.
fn gen_case(rng: &mut ChaCha8Rng, s: &mut State) -> u32 {
    let q: u32 = rng.gen_range(0..2);
    let l: u32 = rng.gen_range(0..2);
    let rt: u32 = if rng.gen_bool(0.2) { rng.gen_range(28..32) } else { rng.gen_range(0..32) };
    let rn: u32 = if rng.gen_bool(0.1) { 31 } else { rng.gen_range(0..31) };
    let post = rng.gen_bool(0.5);
    let rm: u32 = if rng.gen_bool(0.5) {
        31
    } else {
        loop {
            let r = rng.gen_range(0..31);
            if r != rn || rng.gen_bool(0.1) {
                break r;
            }
        }
    };
    let insn = match rng.gen_range(0..3) {
        0 => {
            // multiple structures
            let opcode = [0b0000u32, 0b0010, 0b0100, 0b0110, 0b0111, 0b1000, 0b1010][rng.gen_range(0..7)];
            let size = loop {
                let sz: u32 = rng.gen_range(0..4);
                let selem1 = matches!(opcode, 0b0010 | 0b0110 | 0b0111 | 0b1010);
                if !(sz == 3 && q == 0 && !selem1) {
                    break sz;
                }
            };
            let base = if post { 0x0C80_0000 | (rm << 16) } else { 0x0C00_0000 };
            (q << 30) | base | (l << 22) | (opcode << 12) | (size << 10) | (rn << 5) | rt
        }
        _ => {
            // single structure / replicate
            let r: u32 = rng.gen_range(0..2);
            let replicate = l == 1 && rng.gen_bool(0.3);
            let (opcode, s_bit, size) = if replicate {
                (0b110 | (rng.gen_range(0..2u32)), 0, rng.gen_range(0..4u32))
            } else {
                match rng.gen_range(0..4) {
                    0 => (rng.gen_range(0..2u32), rng.gen_range(0..2u32), rng.gen_range(0..4u32)),
                    1 => (0b010 | rng.gen_range(0..2u32), rng.gen_range(0..2u32), rng.gen_range(0..2u32) << 1),
                    2 => (0b100 | rng.gen_range(0..2u32), rng.gen_range(0..2u32), 0),
                    _ => (0b100 | rng.gen_range(0..2u32), 0, 1),
                }
            };
            let base = if post { 0x0D80_0000 | (rm << 16) } else { 0x0D00_0000 };
            (q << 30) | base | (l << 22) | (r << 21) | (opcode << 13) | (s_bit << 12) | (size << 10) | (rn << 5) | rt
        }
    };
    // Base anywhere in the middle of the data region, often unaligned and
    // sometimes straddling a page boundary.
    let addr = if rng.gen_bool(0.3) {
        DATA + 0x1000 * rng.gen_range(1..14u64) - rng.gen_range(0..64u64)
    } else {
        DATA + 0x100 + rng.gen_range(0..(DATA_SIZE as u64 - 0x200))
    };
    if rn == 31 {
        s.sp = addr;
    } else {
        s.x[rn as usize] = addr;
    }
    if post && rm != 31 && rm != rn {
        s.x[rm as usize] = rng.gen_range(-512i64..512) as u64;
    }
    insn
}

#[test]
fn structure_loads_stores_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5151_1D57);
    let mut uni = Uni::new();
    let mut fast = Rj::new(true);
    let mut slow = Rj::new(false);
    let mut checked = 0u64;
    for case in 0..cases() {
        let mut s = random_state(&mut rng);
        let insn = gen_case(&mut rng, &mut s);
        let words = [insn, BRK_0];
        let (u, ue) = uni.run(&words, &s);
        if ue != Exit::Done {
            // Unallocated in the architecture (Unicorn raises UNDEFINED);
            // Rustarmic must not silently accept it either.
            let (_, je, _) = slow.run(&words, &s);
            assert_ne!(je, Exit::Done, "case {case}: {insn:08x} undefined in unicorn ({ue:?}) but ran in rustarmic");
            continue;
        }
        for (name, rj) in [("pagetable", &mut fast), ("callbacks", &mut slow)] {
            let (j, je, _) = rj.run(&words, &s);
            assert_eq!(je, Exit::Done, "case {case} {name}: {insn:08x} stopped: {je:?}");
            assert!(u == j, "case {case} {name}: {insn:08x} unicorn vs rustarmic:{}", u.diff(&j));
        }
        checked += 1;
    }
    assert!(checked > cases() * 3 / 4, "too few valid cases: {checked}");
}

/// Sequences of structure accesses mixed with ordinary loads / stores and
/// vector arithmetic, so the new IR shares blocks with other code.
#[test]
fn structure_access_sequences_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xD00D_5EED);
    let mut uni = Uni::new();
    let mut fast = Rj::new(true);
    let mut slow = Rj::new(false);
    for case in 0..cases() / 8 {
        let mut s = random_state(&mut rng);
        let mut words = Vec::new();
        let n = rng.gen_range(2..10);
        while words.len() < n {
            let mut probe = s.clone();
            let insn = gen_case(&mut rng, &mut probe);
            // Keep each base register pointing into the data region: only
            // take instructions whose base / offset registers were already
            // set up (rebased below).
            let rn = (insn >> 5) & 31;
            let rm = (insn >> 16) & 31;
            if rn == 31 {
                continue;
            }
            // Re-anchor the base before every access.
            let addr = DATA + 0x400 + rng.gen_range(0..(DATA_SIZE as u64 - 0x800));
            words.push(0xD280_0000 | (((addr & 0xFFFF) as u32) << 5) | rn); // movz xN, #lo
            words.push(0xF2A0_0000 | ((((addr >> 16) & 0xFFFF) as u32) << 5) | rn); // movk xN, #hi, lsl 16
            let post = insn & (1 << 23) != 0;
            if post && rm != 31 && rm != rn {
                words.push(0xD280_0000 | (rng.gen_range(0..256u32) << 5) | rm);
            }
            words.push(insn);
            if rng.gen_bool(0.3) {
                // add v<a>.4s, v<b>.4s, v<c>.4s
                let (a, b, c) = (rng.gen_range(0..32u32), rng.gen_range(0..32u32), rng.gen_range(0..32u32));
                words.push(0x4EA0_8400 | (c << 16) | (b << 5) | a);
            }
        }
        words.push(BRK_0);
        let (u, ue) = uni.run(&words, &s);
        if ue != Exit::Done {
            continue;
        }
        for (name, rj) in [("pagetable", &mut fast), ("callbacks", &mut slow)] {
            let (j, je, _) = rj.run(&words, &s);
            assert_eq!(je, Exit::Done, "case {case} {name}: {words:08x?} stopped: {je:?}");
            assert!(u == j, "case {case} {name}: {words:08x?} unicorn vs rustarmic:{}", u.diff(&j));
        }
        s.nzcv = 0;
    }
}

/// The instruction that stopped a game: `st1 {v0.s}[2], [x8]`.
#[test]
fn st1_lane_s_regression_0x4d008100() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x4D00_8100);
    let mut uni = Uni::new();
    let mut fast = Rj::new(true);
    let mut slow = Rj::new(false);
    for _ in 0..32 {
        let mut s = random_state(&mut rng);
        s.x[8] = DATA + rng.gen_range(0..(DATA_SIZE as u64 - 4));
        let words = [0x4D00_8100, BRK_0];
        let (u, ue) = uni.run(&words, &s);
        assert_eq!(ue, Exit::Done);
        for rj in [&mut fast, &mut slow] {
            let (j, je, _) = rj.run(&words, &s);
            assert_eq!(je, Exit::Done);
            assert!(u == j, "{}", u.diff(&j));
        }
        let off = (s.x[8] - DATA) as usize;
        assert_eq!(&u.data[off..off + 4], &((s.v[0][1] as u32).to_le_bytes()));
    }
}

/// A multi-register access that runs off the end of guest memory reports
/// the first faulting element, and accesses before it still happen.  The
/// ARM ARM LD1-LD4 / ST1-ST4 pseudocode performs one `Mem[]` access per
/// element in ascending address order, so with 40 mapped bytes the first
/// fault is the byte element at `base + 40` (this test previously expected
/// `base + 32`, the start of the 16-byte chunk the old lowering used).
#[test]
fn multi_register_fault_reports_first_failing_address() {
    let end = simd_harness::MEM_SIZE as u64;
    for pagetable in [false, true] {
        for (insn, what) in [
            (0x4C00_2100u32, "st1 {v0.16b-v3.16b}, [x8]"),
            (0x4C40_0100, "ld4 {v0.16b-v3.16b}, [x8]"),
            (0x4C40_2100, "ld1 {v0.16b-v3.16b}, [x8]"),
        ] {
            let mut rj = Rj::new(pagetable);
            let mut s = State::new();
            for (i, v) in s.v.iter_mut().enumerate() {
                *v = [0x0101_0101_0101_0101 * i as u64, !(i as u64)];
            }
            let base = end - 40;
            s.x[8] = base;
            let (_, exit, ctx) = rj.run(&[insn, BRK_0], &s);
            assert!(
                matches!(exit, Exit::Other(ref m) if m.contains("MemoryFault")),
                "{what}: {exit:?}"
            );
            assert_eq!(ctx.mem_fault_addr, base + 40, "{what} pagetable={pagetable}");
            assert_eq!(ctx.mem_fault_pc, simd_harness::CODE, "{what}");
        }
    }
}

/// LDNP / STNP (all register kinds) and LDTR* / STTR* (all sizes and
/// sign-extending forms).
#[test]
fn ldnp_stnp_and_unprivileged_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x1D7_9A1E);
    let mut uni = Uni::new();
    let mut fast = Rj::new(true);
    let mut slow = Rj::new(false);
    for case in 0..cases() / 2 {
        let mut s = random_state(&mut rng);
        let rn: u32 = if rng.gen_bool(0.1) { 31 } else { rng.gen_range(0..31) };
        let rt: u32 = rng.gen_range(0..32);
        let insn = if rng.gen_bool(0.5) {
            let (opc, v) = [(0u32, 0u32), (2, 0), (0, 1), (1, 1), (2, 1)][rng.gen_range(0..5)];
            let l: u32 = rng.gen_range(0..2);
            let rt2 = loop {
                let r = rng.gen_range(0..32);
                if r != rt || l == 0 {
                    break r;
                }
            };
            let imm7: u32 = rng.gen_range(0..128);
            (opc << 30) | 0x2800_0000 | (v << 26) | (l << 22) | (imm7 << 15) | (rt2 << 10) | (rn << 5) | rt
        } else {
            let (size, opc) = [(0u32, 0u32), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3)]
                [rng.gen_range(0..13)];
            let imm9: u32 = rng.gen_range(0..512);
            (size << 30) | 0x3800_0800 | (opc << 22) | (imm9 << 12) | (rn << 5) | rt
        };
        let addr = DATA + 0x800 + rng.gen_range(0..(DATA_SIZE as u64 - 0x1000));
        if rn == 31 {
            s.sp = addr;
        } else {
            s.x[rn as usize] = addr;
        }
        let words = [insn, BRK_0];
        let (u, ue) = uni.run(&words, &s);
        assert_eq!(ue, Exit::Done, "case {case}: {insn:08x}");
        for (name, rj) in [("pagetable", &mut fast), ("callbacks", &mut slow)] {
            let (j, je, _) = rj.run(&words, &s);
            assert_eq!(je, Exit::Done, "case {case} {name}: {insn:08x} stopped: {je:?}");
            assert!(u == j, "case {case} {name}: {insn:08x} unicorn vs rustarmic:{}", u.diff(&j));
        }
    }
}

/// LDR / STR of B and H vector registers in every addressing form
/// (unsigned offset, unscaled, pre / post index, register offset).
#[test]
fn byte_and_half_vector_loads_stores_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xB_4A1F);
    let mut uni = Uni::new();
    let mut fast = Rj::new(true);
    let mut slow = Rj::new(false);
    for case in 0..cases() / 4 {
        let mut s = random_state(&mut rng);
        let size: u32 = rng.gen_range(0..2);
        let opc: u32 = rng.gen_range(0..2); // store / load
        let rn: u32 = rng.gen_range(0..31);
        let rt: u32 = rng.gen_range(0..32);
        let rm: u32 = loop {
            let r = rng.gen_range(0..31);
            if r != rn {
                break r;
            }
        };
        let insn = (size << 30) | (opc << 22) | (rn << 5) | rt | match rng.gen_range(0..4) {
            0 => 0x3D00_0000 | (rng.gen_range(0..1024u32) << 10),
            1 => 0x3C00_0000 | (rng.gen_range(0..512u32) << 12),
            2 => 0x3C00_0400 | ((rng.gen_range(0..2u32) * 2) << 10) | (rng.gen_range(0..512u32) << 12),
            _ => {
                s.x[rm as usize] = rng.gen_range(0..1024u64);
                0x3C20_0800 | (rm << 16) | ([0b011u32, 0b111][rng.gen_range(0..2)] << 13) | (rng.gen_range(0..2u32) << 12)
            }
        };
        s.x[rn as usize] = DATA + 0x1000 + rng.gen_range(0..0x8000u64);
        let words = [insn, BRK_0];
        let (u, ue) = uni.run(&words, &s);
        assert_eq!(ue, Exit::Done, "case {case}: {insn:08x}");
        for (name, rj) in [("pagetable", &mut fast), ("callbacks", &mut slow)] {
            let (j, je, _) = rj.run(&words, &s);
            assert_eq!(je, Exit::Done, "case {case} {name}: {insn:08x} stopped: {je:?}");
            assert!(u == j, "case {case} {name}: {insn:08x} unicorn vs rustarmic:{}", u.diff(&j));
        }
    }
}

/// A long straight-line run of 4-register structure accesses: many
/// 128-bit temporaries live across memory callbacks, so blocks spill
/// heavily (and are retried shorter if the spill frame overflows).
#[test]
fn long_runs_of_four_register_accesses_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x4444);
    let mut uni = Uni::new();
    let mut fast = Rj::new(true);
    let mut slow = Rj::new(false);
    for _ in 0..4 {
        let mut s = random_state(&mut rng);
        s.x[8] = DATA + 0x1000;
        let mut words = Vec::new();
        for i in 0..120u32 {
            let rt = rng.gen_range(0..32u32);
            let size = rng.gen_range(0..4u32);
            let l = (i % 3 != 0) as u32;
            let opcode = [0b0000u32, 0b0010][rng.gen_range(0..2)];
            // ld4/st4/ld1/st1 {4 regs}, [x8], #64 (post-index immediate)
            words.push(0x4C9F_0000 | (l << 22) | (opcode << 12) | (size << 10) | (8 << 5) | rt);
        }
        words.push(BRK_0);
        let (u, ue) = uni.run(&words, &s);
        assert_eq!(ue, Exit::Done);
        for (name, rj) in [("pagetable", &mut fast), ("callbacks", &mut slow)] {
            let (j, je, _) = rj.run(&words, &s);
            assert_eq!(je, Exit::Done, "{name}");
            assert!(u == j, "{name}: unicorn vs rustarmic:{}", u.diff(&j));
        }
    }
}
