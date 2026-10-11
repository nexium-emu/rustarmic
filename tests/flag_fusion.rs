//! `cmp`/`cmn` + `b.cond` fused onto host flags must agree with the ARM
//! condition semantics for every condition, width and edge value, and the
//! materialised NZCV byte must be unchanged.

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE: u64 = 0x1000;
const BRK_0: u32 = 0xD420_0000;

struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.0.get(((addr.checked_sub(CODE)?) / 4) as usize).copied()
    }
}

fn arm_flags(sub: bool, wide: bool, a: u64, b: u64) -> u8 {
    let (n, z, c, v) = if wide {
        if sub {
            let r = a.wrapping_sub(b);
            (r >> 63, r == 0, a >= b, ((a ^ b) & (a ^ r)) >> 63)
        } else {
            let (r, carry) = a.overflowing_add(b);
            (r >> 63, r == 0, carry, (!(a ^ b) & (a ^ r)) >> 63)
        }
    } else {
        let (a, b) = (a as u32, b as u32);
        if sub {
            let r = a.wrapping_sub(b);
            ((r >> 31) as u64, r == 0, a >= b, (((a ^ b) & (a ^ r)) >> 31) as u64)
        } else {
            let (r, carry) = a.overflowing_add(b);
            ((r >> 31) as u64, r == 0, carry, ((!(a ^ b) & (a ^ r)) >> 31) as u64)
        }
    };
    ((n as u8) << 3) | ((z as u8) << 2) | ((c as u8) << 1) | v as u8
}

fn holds(cond: u32, nzcv: u8) -> bool {
    let (n, z, c, v) = (nzcv >> 3 & 1, nzcv >> 2 & 1, nzcv >> 1 & 1, nzcv & 1);
    let base = match cond >> 1 {
        0 => z == 1,
        1 => c == 1,
        2 => n == 1,
        3 => v == 1,
        4 => c == 1 && z == 0,
        5 => n == v,
        6 => n == v && z == 0,
        _ => true,
    };
    if cond & 1 == 1 && cond != 15 { !base } else { base }
}

const VALUES: [u64; 12] = [
    0,
    1,
    2,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    u64::MAX,
    u64::MAX - 1,
    0x1234_5678_9ABC_DEF0,
];

#[test]
fn fused_compare_branch_matches_arm_semantics() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    for (base, sub, wide) in [
        (0xEB00_0000u32, true, true),
        (0x6B00_0000, true, false),
        (0xAB00_0000, false, true),
        (0x2B00_0000, false, false),
    ] {
        for rd in [31u32, 3] {
            for cond in 0..16u32 {
                // 0: op rd, x0, x1 ; 1: b.cond 4 ; 2: movz x2,#1 ; 3: brk
                // 4: movz x2,#2 ; 5: brk
                let mut mem = Code(vec![
                    base | (1 << 16) | rd,
                    0x5400_0000 | (3 << 5) | cond,
                    0xD280_0022,
                    BRK_0,
                    0xD280_0042,
                    BRK_0,
                ]);
                let mut jit = Jit::new(JitConfig {
                    code_cache_bytes: 1 << 20,
                    ..JitConfig::default()
                })
                .unwrap();
                for &a in &VALUES {
                    for &b in &VALUES {
                        let mut ctx = CpuContext::default();
                        ctx.mem_read = no_mem;
                        ctx.mem_write = no_mem;
                        ctx.pc = CODE;
                        ctx.x[0] = a;
                        ctx.x[1] = b;
                        ctx.nzcv = 0xF ^ arm_flags(sub, wide, a, b);
                        let out = jit.run(&mut ctx, &mut mem).unwrap();
                        assert_eq!(out, ExitReason::Brk(0));
                        let flags = arm_flags(sub, wide, a, b);
                        let want = if holds(cond, flags) { 2 } else { 1 };
                        assert_eq!(
                            ctx.x[2], want,
                            "op={base:#x} cond={cond} a={a:#x} b={b:#x} flags={flags:04b}"
                        );
                        assert_eq!(ctx.nzcv & 0xF, flags, "materialised NZCV");
                    }
                }
            }
        }
    }
}

#[test]
fn fused_compare_immediate_branch_matches_arm_semantics() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    // (base, sub, wide): CMP/CMN immediate forms, Rd = XZR, Rn = X0.
    for (base, sub, wide) in [
        (0xF100_001Fu32, true, true),
        (0x7100_001F, true, false),
        (0xB100_001F, false, true),
        (0x3100_001F, false, false),
    ] {
        for (imm12, shift) in [(0u32, 0u32), (1, 0), (0xFFF, 0), (0x800, 1), (0xFFF, 1)] {
            let b = u64::from(imm12) << (12 * shift);
            for cond in 0..16u32 {
                let mut mem = Code(vec![
                    base | (shift << 22) | (imm12 << 10),
                    0x5400_0000 | (3 << 5) | cond,
                    0xD280_0022,
                    BRK_0,
                    0xD280_0042,
                    BRK_0,
                ]);
                let mut jit = Jit::new(JitConfig {
                    code_cache_bytes: 1 << 20,
                    ..JitConfig::default()
                })
                .unwrap();
                for &a in VALUES.iter().chain(&[b, b.wrapping_sub(1), b.wrapping_neg()]) {
                    let mut ctx = CpuContext::default();
                    ctx.mem_read = no_mem;
                    ctx.mem_write = no_mem;
                    ctx.pc = CODE;
                    ctx.x[0] = a;
                    let out = jit.run(&mut ctx, &mut mem).unwrap();
                    assert_eq!(out, ExitReason::Brk(0));
                    let flags = arm_flags(sub, wide, a, b);
                    let want = if holds(cond, flags) { 2 } else { 1 };
                    assert_eq!(
                        ctx.x[2], want,
                        "op={base:#x} imm={b:#x} cond={cond} a={a:#x} flags={flags:04b}"
                    );
                    assert_eq!(ctx.nzcv & 0xF, flags);
                }
            }
        }
    }
}

/// ANDS / BICS / TST set N and Z from the result and clear C and V.  Checked
/// both with the branch fused onto the producer (same block) and unfused
/// (branch in a separate block reading the stored NZCV byte).
#[test]
fn logical_flag_setters_clear_carry_and_overflow() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    // (encoding with Rn = X0, Rm = X1 where applicable, op on a/b, wide)
    let ops: [(u32, fn(u64, u64) -> u64, bool, &str); 10] = [
        (0xEA00_0000 | (1 << 16), |a, b| a & b, true, "ands x"),
        (0x6A00_0000 | (1 << 16), |a, b| a & b, false, "ands w"),
        (0xEA20_0000 | (1 << 16), |a, b| a & !b, true, "bics x"),
        (0x6A20_0000 | (1 << 16), |a, b| a & !b, false, "bics w"),
        // ands x, x0, #0xff
        (0xF240_0000 | (7 << 10), |a, _| a & 0xFF, true, "ands x imm 0xff"),
        // ands x, x0, #0x8000000000000000
        (0xF240_0000 | (1 << 16), |a, _| a & (1 << 63), true, "ands x imm msb"),
        // ands w, w0, #0xff
        (0x7200_0000 | (7 << 10), |a, _| a & 0xFF, false, "ands w imm 0xff"),
        // ands w, w0, #0x80000000
        (0x7200_0000 | (1 << 16), |a, _| a & 0x8000_0000, false, "ands w imm msb"),
        // ands x, x0, x1, lsl #1
        (0xEA00_0000 | (1 << 16) | (1 << 10), |a, b| a & (b << 1), true, "ands x lsl"),
        // bics w, w0, w1, asr #4
        (
            0x6AA0_0000 | (1 << 16) | (4 << 10),
            |a, b| a & !(((b as u32 as i32) >> 4) as u32 as u64),
            false,
            "bics w asr",
        ),
    ];
    for (enc, f, wide, name) in ops {
        for rd in [31u32, 3] {
            for fused in [true, false] {
                for cond in 0..16u32 {
                    let insn = enc | rd;
                    let bcond = 0x5400_0000 | (3 << 5) | cond;
                    // Fused: op ; b.cond +3*4 ...
                    // Unfused: op ; b +4 ; b.cond (separate block).
                    let mut words = vec![insn];
                    if !fused {
                        words.push(0x1400_0001);
                    }
                    words.extend([bcond, 0xD280_0022, BRK_0, 0xD280_0042, BRK_0]);
                    let mut mem = Code(words);
                    let mut config = JitConfig {
                        code_cache_bytes: 1 << 20,
                        ..JitConfig::default()
                    };
                    if !fused {
                        config.translate.multiblock = false;
                    }
                    let mut jit = Jit::new(config).unwrap();
                    for &a in &VALUES {
                        for &b in &VALUES {
                            let mut ctx = CpuContext::default();
                            ctx.mem_read = no_mem;
                            ctx.mem_write = no_mem;
                            ctx.pc = CODE;
                            ctx.x[0] = a;
                            ctx.x[1] = b;
                            ctx.nzcv = 0xF;
                            let out = jit.run(&mut ctx, &mut mem).unwrap();
                            assert_eq!(out, ExitReason::Brk(0));
                            let r = f(a, b);
                            let (n, z) = if wide {
                                (r >> 63 == 1, r == 0)
                            } else {
                                ((r as u32) >> 31 == 1, r as u32 == 0)
                            };
                            let flags = ((n as u8) << 3) | ((z as u8) << 2);
                            assert_eq!(
                                ctx.nzcv & 0xF,
                                flags,
                                "{name} fused={fused} a={a:#x} b={b:#x}: materialised NZCV"
                            );
                            let want = if holds(cond, flags) { 2 } else { 1 };
                            assert_eq!(
                                ctx.x[2], want,
                                "{name} fused={fused} cond={cond} a={a:#x} b={b:#x} flags={flags:04b}"
                            );
                            if rd == 3 {
                                let r = if wide { r } else { r as u32 as u64 };
                                assert_eq!(ctx.x[3], r, "{name} result");
                            }
                        }
                    }
                }
            }
        }
    }
}

/// ADC / SBC / ADCS / SBCS take their carry from the flags value, even when
/// the NZCV write that produced it is later overwritten in the same block
/// (the optimizer drops such dead SetNzcv stores).
#[test]
fn carry_consumers_see_flags_from_eliminated_nzcv_store() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    // ccmp x0, #0, #nzcv, eq  (Z clear on entry, so NZCV := #nzcv)
    // <carry op> x3, x4, x5
    // cmp x6, x7
    // brk
    for (op, name) in [
        (0x9A05_0083u32, "adc"),
        (0xDA05_0083, "sbc"),
        (0xBA05_0083, "adcs"),
        (0xFA05_0083, "sbcs"),
    ] {
        for c in [0u32, 1] {
            let mut mem = Code(vec![0xFA40_0800 | (c << 1), op, 0xEB07_00DF, BRK_0]);
            let mut jit = Jit::new(JitConfig {
                code_cache_bytes: 1 << 20,
                ..JitConfig::default()
            })
            .unwrap();
            let mut ctx = CpuContext::default();
            ctx.mem_read = no_mem;
            ctx.mem_write = no_mem;
            ctx.pc = CODE;
            ctx.nzcv = 0;
            ctx.x[0] = 5;
            ctx.x[4] = 10;
            ctx.x[5] = 3;
            assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
            let want = if name.starts_with("adc") {
                13 + u64::from(c)
            } else {
                // n + !m + C = n - m - 1 + C
                10 - 3 - 1 + u64::from(c)
            };
            assert_eq!(ctx.x[3], want, "{name} with C={c}");
        }
    }
}

/// ADCS / SBCS flags, fused branch afterwards included.
#[test]
fn adcs_sbcs_flags_match_arm() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    for (base, sub, wide) in [
        (0xBA00_0000u32, false, true),
        (0x3A00_0000, false, false),
        (0xFA00_0000, true, true),
        (0x7A00_0000, true, false),
    ] {
        let mut jit = Jit::new(JitConfig {
            code_cache_bytes: 1 << 20,
            ..JitConfig::default()
        })
        .unwrap();
        // adcs/sbcs x3, x0, x1 ; b.cs +3 ; movz x2,#1 ; brk ; movz x2,#2 ; brk
        let mut mem = Code(vec![
            base | (1 << 16) | 3,
            0x5400_0000 | (3 << 5) | 2,
            0xD280_0022,
            BRK_0,
            0xD280_0042,
            BRK_0,
        ]);
        for &a in &VALUES {
            for &b in &VALUES {
                for cin in [0u8, 1] {
                    let mut ctx = CpuContext::default();
                    ctx.mem_read = no_mem;
                    ctx.mem_write = no_mem;
                    ctx.pc = CODE;
                    ctx.x[0] = a;
                    ctx.x[1] = b;
                    ctx.nzcv = cin << 1;
                    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
                    let bits = if wide { 64 } else { 32 };
                    let m = if wide { u128::from(u64::MAX) } else { 0xFFFF_FFFF };
                    let (x, y) = (u128::from(a) & m, u128::from(b) & m);
                    let y = if sub { !y & m } else { y };
                    let full = x + y + u128::from(cin);
                    let r = full & m;
                    let n = (r >> (bits - 1)) & 1;
                    let z = (r == 0) as u128;
                    let c = (full >> bits) & 1;
                    let sx = (x >> (bits - 1)) & 1;
                    let sy = (y >> (bits - 1)) & 1;
                    let v = ((sx == sy) && (n != sx)) as u128;
                    let flags = ((n << 3) | (z << 2) | (c << 1) | v) as u8;
                    let ctxs = format!("{base:#x} a={a:#x} b={b:#x} cin={cin}");
                    assert_eq!(ctx.x[3] as u128, r, "{ctxs}");
                    assert_eq!(ctx.nzcv & 0xF, flags, "{ctxs}");
                    assert_eq!(ctx.x[2], if c == 1 { 2 } else { 1 }, "{ctxs}");
                }
            }
        }
    }
}
