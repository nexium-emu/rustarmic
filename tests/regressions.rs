//! Regressions found by the control-flow differential fuzzer
//! (`tests/control_flow_fuzz.rs`).

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE: u64 = 0x1000;
const BRK_0: u32 = 0xD420_0000;

struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.0.get(((addr.checked_sub(CODE)?) / 4) as usize).copied()
    }
}

/// Architectural SBFM / UBFM (DecodeBitMasks-based model).
fn bfm_model(signed: bool, width: u32, immr: u32, imms: u32, src: u64) -> u64 {
    let mask = |n: u32| if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
    let src = src & mask(width);
    let rot = if immr == 0 {
        src
    } else {
        ((src >> immr) | (src << (width - immr))) & mask(width)
    };
    // wmask = ROR(Ones(imms + 1), immr) within `width`; tmask = Ones(diff+1)
    let ones = mask(imms + 1);
    let wmask = if immr == 0 {
        ones
    } else {
        ((ones >> immr) | (ones << (width - immr))) & mask(width)
    };
    let diff = imms.wrapping_sub(immr) & 63;
    let tmask = mask((diff & (width - 1)) + 1);
    let bot = rot & wmask;
    let top = if signed && (src >> imms) & 1 == 1 {
        mask(width)
    } else {
        0
    };
    ((top & !tmask) | (bot & tmask)) & mask(width)
}

#[test]
fn sbfm_ubfm_all_fields_match_model() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    let srcs = [
        0u64,
        u64::MAX,
        0x8000_0000_0000_0000,
        0x0000_0000_8000_0000,
        0x1234_5678_9ABC_DEF0,
        0xF6E6_8000_0002_3B37,
        0x0002_0000,
    ];
    let mut jit = Jit::new(JitConfig {
        code_cache_bytes: 16 << 20,
        ..JitConfig::default()
    })
    .unwrap();
    for sf in [0u32, 1] {
        let width = if sf == 1 { 64 } else { 32 };
        for signed in [true, false] {
            for immr in 0..width {
                for imms in 0..width {
                    let opc = if signed { 0b00 } else { 0b10 };
                    let insn = (sf << 31) | (opc << 29) | 0x1300_0000 | (sf << 22)
                        | (immr << 16) | (imms << 10) | 3;
                    let mut mem = Code(vec![insn, BRK_0]);
                    for &src in &srcs {
                        let mut ctx = CpuContext::default();
                        ctx.mem_read = no_mem;
                        ctx.mem_write = no_mem;
                        ctx.pc = CODE;
                        ctx.x[0] = src;
                        ctx.x[3] = 0xAAAA_AAAA_AAAA_AAAA;
                        assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
                        let want = bfm_model(signed, width, immr, imms, src);
                        assert_eq!(
                            ctx.x[3], want,
                            "{} sf={sf} immr={immr} imms={imms} src={src:#x}",
                            if signed { "sbfm" } else { "ubfm" }
                        );
                    }
                    jit.invalidate_range(CODE, 8);
                }
            }
        }
    }
}

#[test]
fn extr_all_lsb_match_model() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    let mut jit = Jit::new(JitConfig {
        code_cache_bytes: 4 << 20,
        ..JitConfig::default()
    })
    .unwrap();
    let (n, m) = (0x0123_4567_89AB_CDEFu64, 0xFEDC_BA98_7654_3210u64);
    for sf in [0u32, 1] {
        let width = if sf == 1 { 64 } else { 32 };
        for rm in [1u32, 0] {
            for lsb in 0..width {
                // extr r3, r0, r<rm>, #lsb
                let insn = (sf << 31) | 0x1380_0000 | (sf << 22) | (rm << 16) | (lsb << 10) | 3;
                let mut mem = Code(vec![insn, BRK_0]);
                let mut ctx = CpuContext::default();
                ctx.mem_read = no_mem;
                ctx.mem_write = no_mem;
                ctx.pc = CODE;
                ctx.x[0] = n;
                ctx.x[1] = m;
                assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
                let lo = if rm == 1 { m } else { n };
                let want = if width == 64 {
                    let concat = (u128::from(n) << 64) | u128::from(lo);
                    (concat >> lsb) as u64
                } else {
                    let concat = (u64::from(n as u32) << 32) | u64::from(lo as u32);
                    (concat >> lsb) as u32 as u64
                };
                assert_eq!(ctx.x[3], want, "extr sf={sf} rm={rm} lsb={lsb}");
                jit.invalidate_range(CODE, 8);
            }
        }
    }
}

/// CBZ/CBNZ on a W register whose X value (forwarded within the block, or a
/// folded constant) has only upper bits set must test the low 32 bits.
#[test]
fn cbz_w_ignores_upper_half_of_forwarded_value() {
    unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected memory access");
    }
    // 0: movz x12, #0x29c6, lsl 32  (or: add x12, x0, #0 with x0 runtime)
    // 1: cb{n}z w12, +3 ; 2: movz x2,#1 ; 3: brk ; 4: movz x2,#2 ; 5: brk
    for first in [0xD2C5_38CCu32, 0x9100_000C] {
        for (op, taken_when_zero) in [(0x3400_0000u32, true), (0x3500_0000, false)] {
            let mut mem = Code(vec![
                first,
                op | (3 << 5) | 12,
                0xD280_0022,
                BRK_0,
                0xD280_0042,
                BRK_0,
            ]);
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let mut ctx = CpuContext::default();
            ctx.mem_read = no_mem;
            ctx.mem_write = no_mem;
            ctx.pc = CODE;
            ctx.x[0] = 0x29C6_0000_0000;
            assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
            let want = if taken_when_zero { 2 } else { 1 };
            assert_eq!(ctx.x[2], want, "first={first:#x} op={op:#x}");
        }
    }
}

/// `st1 {v0.s}[2], [x8]` (0x4d008100) used to stop a game as an
/// unsupported ASISDLSO instruction.  It must store exactly lane 2.
#[test]
fn st1_single_lane_s_0x4d008100_stores_lane_2() {
    use std::cell::RefCell;
    thread_local! {
        static STORES: RefCell<Vec<(u64, u8, [u64; 2])>> = const { RefCell::new(Vec::new()) };
    }
    unsafe extern "C" fn no_read(_: *mut CpuContext, _: u64, _: u8) {
        panic!("unexpected read");
    }
    unsafe extern "C" fn record_write(ctx: *mut CpuContext, addr: u64, size: u8) {
        let v = unsafe { (*ctx).io_value };
        STORES.with(|s| s.borrow_mut().push((addr, size, v)));
    }
    let mut mem = Code(vec![0x4D00_8100, BRK_0]);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut ctx = CpuContext::default();
    ctx.mem_read = no_read;
    ctx.mem_write = record_write;
    ctx.pc = CODE;
    ctx.x[8] = 0x8000_1234;
    ctx.v[0] = [0x1111_1111_2222_2222, 0x3333_3333_4444_4444];
    assert_eq!(jit.run(&mut ctx, &mut mem).unwrap(), ExitReason::Brk(0));
    let stores = STORES.with(|s| s.borrow().clone());
    assert_eq!(stores.len(), 1);
    assert_eq!(stores[0].0, 0x8000_1234);
    assert_eq!(stores[0].1, 4);
    assert_eq!(stores[0].2[0] as u32, 0x4444_4444);
    assert_eq!(ctx.x[8], 0x8000_1234, "no writeback without post-index");
}
