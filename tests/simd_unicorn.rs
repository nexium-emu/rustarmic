//! Scalar and vector AdvSIMD, float <-> integer / fixed-point conversions,
//! CRC32, AES / SHA and dot product against Unicorn (max CPU model).
//!
//! Every template randomises all of an encoding group's fields; encodings
//! Unicorn treats as UNDEFINED are skipped (and must not run in
//! Rustarmic).  Instructions lowered to the out-of-line helpers are fed
//! edge-case operands (NaNs, infinities, denormals, saturation limits)
//! under random FPCR.{FZ, DN} (and RMode for conversions); instructions
//! with an inline lowering get NaN-free operands and the default FPCR,
//! which is all the inline paths model.
#![cfg(feature = "unicorn")]

mod simd_harness;

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustarmic::frontend::{TranslateOptions, translate_block_into};
use rustarmic::ir::{Block, Op};
use simd_harness::{BRK_0, CODE, DATA, Exit, Rj, State, Uni};
use std::collections::BTreeMap;

fn cases() -> u64 {
    std::env::var("RUSTARMIC_SIMD_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000)
}

/// (name, fixed bits, randomised bits, conversion-like (random RMode))
struct Template {
    name: &'static str,
    fixed: u32,
    random: u32,
    rmode: bool,
    /// Must the JIT implement every encoding Unicorn accepts?
    complete: bool,
}

const RD_RN: u32 = 0x3FF;
const RM: u32 = 0x1F << 16;

fn templates() -> Vec<Template> {
    let t = |name, fixed, random, rmode, complete| Template { name, fixed, random, rmode, complete };
    let u = 1 << 29;
    let q = 1 << 30;
    let size = 3 << 22;
    vec![
        t("asisd_same", 0x5E20_0400, u | size | RM | (0x1F << 11) | RD_RN, false, false),
        t("asisd_diff", 0x5E20_0000, u | size | RM | (0xF << 12) | RD_RN, false, true),
        t("asisd_misc", 0x5E20_0800, u | size | (0x1F << 12) | RD_RN, true, false),
        t("asisd_pair", 0x5E30_0800, u | size | (0x1F << 12) | RD_RN, false, false),
        t("asisd_dup", 0x5E00_0400, (0x1F << 16) | RD_RN, false, true),
        t("asisd_shf", 0x5F00_0400, u | (0x7F << 16) | (0x1F << 11) | RD_RN, true, false),
        t("asisd_elem", 0x5F00_0000, u | size | (0x3 << 20) | (0xF << 16) | (0xF << 12) | (1 << 11) | RD_RN, false, false),
        t("asisd_same2", 0x5E00_8400, u | size | RM | (0xF << 11) | RD_RN, false, true),
        t("asimd_same", 0x0E20_0400, q | u | size | RM | (0x1F << 11) | RD_RN, false, false),
        t("asimd_diff", 0x0E20_0000, q | u | size | RM | (0xF << 12) | RD_RN, false, true),
        t("asimd_misc", 0x0E20_0800, q | u | size | (0x1F << 12) | RD_RN, true, false),
        t("asimd_all", 0x0E30_0800, q | u | size | (0x1F << 12) | RD_RN, false, false),
        t("asimd_shf", 0x0F00_0400, q | u | (0x7F << 16) | (0x1F << 11) | RD_RN, true, false),
        t("asimd_elem", 0x0F00_0000, q | u | size | (0x3 << 20) | (0xF << 16) | (0xF << 12) | (1 << 11) | RD_RN, false, false),
        t("asimd_same2", 0x0E00_8400, q | u | size | RM | (0xF << 11) | RD_RN, false, false),
        t("aes", 0x4E28_0800, (0x1F << 12) | RD_RN, false, true),
        t("sha3reg", 0x5E00_0000, RM | (0x7 << 12) | RD_RN, false, true),
        t("sha2reg", 0x5E28_0800, (0x1F << 12) | RD_RN, false, true),
        t("sha512", 0xCE60_8000, RM | (0x3 << 10) | RD_RN, false, true),
        t("sha512su0", 0xCEC0_8000, RD_RN, false, true),
        t("eor3_bcax", 0xCE00_0000, (1 << 21) | RM | (0x1F << 10) | RD_RN, false, true),
        t("xar", 0xCE80_0000, RM | (0x3F << 10) | RD_RN, false, true),
        t("float2int", 0x1E20_0000, (1 << 31) | (3 << 22) | (3 << 19) | (7 << 16) | RD_RN, true, false),
        t("float2fix", 0x1E00_0000, (1 << 31) | (3 << 22) | (3 << 19) | (7 << 16) | (0x3F << 10) | RD_RN, true, false),
        t("crc32", 0x1AC0_4000, (1 << 31) | RM | (1 << 12) | (3 << 10) | RD_RN, false, true),
        t("floatdp1", 0x1E20_4000, (3 << 22) | (0x3F << 15) | RD_RN, true, false),
        t("asimd_tbl", 0x0E00_0000, q | RM | (3 << 13) | (1 << 12) | RD_RN, false, true),
    ]
}

/// Does Rustarmic lower `insn` to an out-of-line helper call?
fn uses_helper(insn: u32) -> bool {
    let mut b = Block::new(CODE);
    let opts = TranslateOptions { max_insts: 1, multiblock: false };
    let mut fetch = |pc: u64| Some(if pc == CODE { insn } else { BRK_0 });
    translate_block_into(&mut b, CODE, &mut fetch, opts).is_ok()
        && b.code.iter().any(|a| a.op == Op::CallHelper)
}

const F32_EDGE: &[u32] = &[
    0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0001, 0x7F80_0001,
    0xFFA0_0000, 0x0000_0001, 0x807F_FFFF, 0x0080_0000, 0x7F7F_FFFF, 0x3F80_0000, 0xBF80_0000,
    0x4F00_0000, 0xCF00_0000, 0x4F80_0000, 0x5F00_0000, 0xDF00_0000, 0x3F00_0000, 0x3FC0_0000,
    0x4040_0000, 0x7E80_0000, 0x0020_0000, 0x001F_FFFF, 0x4EFF_FFFF, 0xBF00_0000,
];
const F64_EDGE: &[u64] = &[
    0, 0x8000_0000_0000_0000, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000,
    0x7FF0_0000_0000_0001, 0xFFF4_0000_0000_0000, 1, 0x800F_FFFF_FFFF_FFFF, 0x0010_0000_0000_0000,
    0x7FEF_FFFF_FFFF_FFFF, 0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000, 0x41E0_0000_0000_0000,
    0xC1E0_0000_0000_0000, 0x43E0_0000_0000_0000, 0xC3E0_0000_0000_0000, 0x43F0_0000_0000_0000,
    0x41EF_FFFF_FFE0_0000, 0x3FE0_0000_0000_0000, 0x3FF8_0000_0000_0000, 0x7FD0_0000_0000_0000,
    0x0004_0000_0000_0000, 0x3800_0000_0000_0000, 0x47F0_0000_0000_0000, 0x36A0_0000_0000_0000,
];
const INT_EDGE: &[u64] = &[
    0, 1, 2, 0x7F, 0x80, 0xFF, 0x7FFF, 0x8000, 0xFFFF, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF,
    0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, u64::MAX, 0x8000_0000_0000_0001,
];

fn lane_value(rng: &mut ChaCha8Rng, nan_free: bool) -> u128 {
    let mut v = 0u128;
    let mut bit = 0u32;
    while bit < 128 {
        let pick = rng.random_range(0..10);
        let (val, width): (u64, u32) = match pick {
            0..=2 => (rng.random::<u64>(), 64),
            3 => (F32_EDGE[rng.random_range(0..F32_EDGE.len())] as u64, 32),
            4 => (F64_EDGE[rng.random_range(0..F64_EDGE.len())], 64),
            5 => (INT_EDGE[rng.random_range(0..INT_EDGE.len())], [8, 16, 32, 64][rng.random_range(0..4)]),
            6 => (((rng.random_range(-300i32..300) as f32) * if rng.random_bool(0.5) { 0.5 } else { 1.0 }).to_bits() as u64, 32),
            7 => ((rng.random_range(-1e6f64..1e6)).to_bits(), 64),
            8 => (rng.random_range(0..16u64), [8, 16, 32, 64][rng.random_range(0..4)]),
            _ => ((rng.random::<f32>() * 4.0 - 2.0).to_bits() as u64, 32),
        };
        let width = width.min(128 - bit);
        let m = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
        v |= ((val & m) as u128) << bit;
        bit += width;
    }
    if nan_free {
        // Clear the top exponent bit of every 32-bit lane: no NaN or
        // infinity in either single or double interpretation.
        v &= 0xBFFF_FFFF_BFFF_FFFF_BFFF_FFFF_BFFF_FFFFu128;
    }
    v
}

fn random_state(rng: &mut ChaCha8Rng, nan_free: bool) -> State {
    let mut s = State::new();
    for x in s.x.iter_mut() {
        *x = if rng.random_bool(0.5) {
            INT_EDGE[rng.random_range(0..INT_EDGE.len())]
        } else {
            rng.random()
        };
    }
    for v in s.v.iter_mut() {
        let x = lane_value(rng, nan_free);
        *v = [x as u64, (x >> 64) as u64];
    }
    s.nzcv = rng.random_range(0..16);
    s
}

#[derive(Default)]
struct Stats {
    checked: u64,
    helper: u64,
    undefined: u64,
    gaps: BTreeMap<u32, String>,
}

fn check(tpl: &Template, rng: &mut ChaCha8Rng, uni: &mut Uni, rj: &mut Rj, n: u64, stats: &mut Stats) {
    for case in 0..n {
        let insn = tpl.fixed | (rng.random::<u32>() & tpl.random);
        if tpl.name.ends_with("shf") && (insn >> 19) & 0xF == 0 {
            continue; // modified immediate, not a shift
        }
        let helper = uses_helper(insn);
        // Inline scalar conversions handle NaN / overflow like Arm, so they
        // get the full edge-case inputs too.
        let nan_free = !helper && !tpl.name.starts_with("float");
        let mut s = random_state(rng, nan_free);
        if helper {
            let mut fpcr = 0u32;
            if rng.random_bool(0.3) {
                fpcr |= 1 << 24; // FZ
            }
            if rng.random_bool(0.3) {
                fpcr |= 1 << 25; // DN
            }
            if tpl.rmode && rng.random_bool(0.5) {
                fpcr |= rng.random_range(0..4u32) << 22;
            }
            s.fpcr = fpcr;
        }
        let words = [insn, BRK_0];
        let (u, ue) = uni.run(&words, &s);
        let (j, je, _) = rj.run(&words, &s);
        if ue != Exit::Done {
            stats.undefined += 1;
            // MUL .2D is reserved but kept as an extension (tests/smoke.rs).
            if insn & 0xFFE0_FC00 == 0x4EE0_9C00 {
                continue;
            }
            assert_ne!(
                je,
                Exit::Done,
                "{} case {case}: {insn:08x} (helper={helper}) is UNDEFINED in unicorn ({ue:?}) but ran in rustarmic",
                tpl.name
            );
            continue;
        }
        if je != Exit::Done {
            stats.gaps.entry(insn).or_insert_with(|| format!("{}: {je:?}", tpl.name));
            continue;
        }
        stats.checked += 1;
        stats.helper += helper as u64;
        // Inline (non-helper) FP paths use host NaN encodings (x86 produces
        // 0xFFC00000 where Arm produces 0x7FC00000); only the helpers model
        // Arm NaNs bit-exactly, so inline results compare NaN-tolerantly.
        let same = if helper { u == j } else { same_modulo_nan(&u, &j) };
        if !same {
            panic!(
                "{} case {case}: {insn:08x} (helper={helper}, fpcr={:#x}) unicorn vs rustarmic:{}\ninputs: x={:x?}\nv={:x?}",
                tpl.name,
                s.fpcr,
                u.diff(&j),
                s.x,
                s.v
            );
        }
    }
}

fn run_templates(filter: impl Fn(&Template) -> bool, seed: u64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut uni = Uni::new();
    let mut rj = Rj::new(true);
    let mut gaps_in_complete = Vec::new();
    for tpl in templates().iter().filter(|t| filter(t)) {
        let mut stats = Stats::default();
        check(tpl, &mut rng, &mut uni, &mut rj, cases(), &mut stats);
        eprintln!(
            "{}: {} checked ({} via helper), {} undefined, {} unsupported encodings",
            tpl.name,
            stats.checked,
            stats.helper,
            stats.undefined,
            stats.gaps.len()
        );
        for (insn, why) in stats.gaps.iter().take(8) {
            eprintln!("    gap {insn:08x}: {why}");
        }
        if tpl.complete && !stats.gaps.is_empty() {
            gaps_in_complete.push(format!("{}: {:08x?}", tpl.name, stats.gaps.keys().take(8).collect::<Vec<_>>()));
        }
        assert!(stats.checked > 0, "{}: nothing checked", tpl.name);
    }
    assert!(gaps_in_complete.is_empty(), "unsupported encodings: {gaps_in_complete:#?}");
}

#[test]
fn scalar_simd_matches_unicorn() {
    run_templates(|t| t.name.starts_with("asisd"), 0xA515_D001);
}

#[test]
fn vector_simd_matches_unicorn() {
    run_templates(|t| t.name.starts_with("asimd"), 0xA515_D002);
}

#[test]
fn crypto_matches_unicorn() {
    run_templates(
        |t| matches!(t.name, "aes" | "sha3reg" | "sha2reg" | "sha512" | "sha512su0" | "eor3_bcax" | "xar"),
        0xA515_D003,
    );
}

#[test]
fn conversions_and_crc_match_unicorn() {
    run_templates(|t| matches!(t.name, "float2int" | "float2fix" | "crc32" | "floatdp1"), 0xA515_D004);
}

/// Helper-lowered instructions in sequences with other code: values must
/// survive the call (caller-saved host registers) and results must be
/// visible to the following instructions.
#[test]
fn helper_sequences_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xA515_D005);
    let mut uni = Uni::new();
    let mut rj = Rj::new(true);
    let tpls = templates();
    for case in 0..cases() / 4 {
        let mut words = Vec::new();
        let n = rng.random_range(2..12);
        while words.len() < n {
            let tpl = &tpls[rng.random_range(0..tpls.len())];
            let insn = tpl.fixed | (rng.random::<u32>() & tpl.random);
            if !uses_helper(insn) {
                continue;
            }
            words.push(insn);
            match rng.random_range(0..4) {
                // add x<a>, x<b>, x<c>
                0 => words.push(0x8B00_0000 | (rng.random_range(0..31u32) << 16) | (rng.random_range(0..31u32) << 5) | rng.random_range(0..29u32)),
                // eor v<a>.16b, v<b>.16b, v<c>.16b
                1 => words.push(0x6E20_1C00 | (rng.random_range(0..32u32) << 16) | (rng.random_range(0..32u32) << 5) | rng.random_range(0..32u32)),
                // str q<a>, [x29, #off]
                2 => words.push(0x3D80_0000 | (rng.random_range(0..64u32) << 10) | (29 << 5) | rng.random_range(0..32u32)),
                _ => {}
            }
        }
        words.push(BRK_0);
        let mut s = random_state(&mut rng, false);
        s.x[29] = DATA + 0x100;
        let (u, ue) = uni.run(&words, &s);
        if ue != Exit::Done {
            continue;
        }
        let (j, je, _) = rj.run(&words, &s);
        assert_eq!(je, Exit::Done, "case {case}: {words:08x?}");
        assert!(u == j, "case {case}: {words:08x?} unicorn vs rustarmic:{}", u.diff(&j));
    }
}

/// Equal except that a NaN lane (single or double view) matches any NaN.
fn same_modulo_nan(a: &State, b: &State) -> bool {
    if a.x != b.x || a.sp != b.sp || a.nzcv != b.nzcv || a.data != b.data {
        return false;
    }
    a.v.iter().zip(&b.v).all(|(x, y)| {
        if x == y {
            return true;
        }
        let singles = (0..4).all(|i| {
            let p = (x[i / 2] >> (32 * (i % 2))) as u32;
            let q = (y[i / 2] >> (32 * (i % 2))) as u32;
            p == q || (f32::from_bits(p).is_nan() && f32::from_bits(q).is_nan())
        });
        let doubles = (0..2).all(|i| x[i] == y[i] || (f64::from_bits(x[i]).is_nan() && f64::from_bits(y[i]).is_nan()));
        singles || doubles
    })
}

/// Inline scalar FMAX / FMIN / FMAXNM / FMINNM: signed zeros, quiet NaN
/// propagation and the NM number preference, bit-exact.  (Signalling
/// NaNs are only approximated inline and are left out here.)
#[test]
fn scalar_fmax_fmin_inline_match_unicorn() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xA515_D006);
    let mut uni = Uni::new();
    let mut rj = Rj::new(true);
    let s_vals: &[u64] = &[0, 0x8000_0000, 0x3F80_0000, 0xBF80_0000, 0x7FC0_0000, 0xFFC0_1234, 0x7F80_0000, 0xFF80_0000, 1, 0x8000_0001];
    let d_vals: &[u64] = &[
        0, 1 << 63, 0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000, 0x7FF8_0000_0000_0000,
        0xFFF8_0000_0000_1234, 0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000, 1, (1 << 63) | 1,
    ];
    for case in 0..2000 {
        let double = rng.random_bool(0.5);
        let op = rng.random_range(0b0100..0b1000u32);
        let insn = 0x1E20_0800 | ((double as u32) << 22) | (1 << 16) | (op << 12) | (2 << 5) | 3;
        let vals = if double { d_vals } else { s_vals };
        let mut s = State::new();
        for v in s.v.iter_mut() {
            *v = [rng.random(), rng.random()];
        }
        let pick = |rng: &mut ChaCha8Rng| if rng.random_bool(0.8) { vals[rng.random_range(0..vals.len())] } else { rng.random::<u32>() as u64 & 0x3FFF_FFFF };
        s.v[2] = [pick(&mut rng), 0];
        s.v[1] = [pick(&mut rng), 0];
        assert!(!uses_helper(insn));
        let words = [insn, BRK_0];
        let (u, ue) = uni.run(&words, &s);
        assert_eq!(ue, Exit::Done);
        let (j, je, _) = rj.run(&words, &s);
        assert_eq!(je, Exit::Done);
        assert!(u == j, "case {case}: {insn:08x} n={:#x} m={:#x}:{}", s.v[2][0], s.v[1][0], u.diff(&j));
    }
}
