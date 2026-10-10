use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
// Exhaustively call the same integer helper source without JIT setup per
// sample; separate instruction tests below validate the full native plumbing.
pub use rustarmic::jit;
#[path = "../src/backend/saturating.rs"]
mod saturating;

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const QC: u32 = 1 << 27;
const FLAGS: u32 = 0x9f;
struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get(offset as usize / 4).copied()
    }
}
fn opcode(kind: u32, size: u32, scalar: bool, upper: bool, rn: u32, rd: u32) -> u32 {
    let base = match kind {
        0 => 0x0e21_4800,
        1 => 0x2e21_4800,
        2 => 0x2e21_2800,
        _ => unreachable!(),
    };
    base | (size << 22)
        | (u32::from(scalar) * 0x5000_0000)
        | (u32::from(!scalar && upper) << 30)
        | (rn << 5)
        | rd
}
fn oracle(bits: u64, size: u32, kind: u32) -> (u64, bool) {
    let width = 8 << size;
    let value = match (kind == 1, size) {
        (true, 0) => bits as u16 as i128,
        (true, 1) => bits as u32 as i128,
        (true, 2) => bits as i128,
        (false, 0) => bits as i16 as i128,
        (false, 1) => bits as i32 as i128,
        (false, 2) => bits as i64 as i128,
        _ => unreachable!(),
    };
    let (minimum, maximum) = if kind == 0 {
        (-(1i128 << (width - 1)), (1i128 << (width - 1)) - 1)
    } else {
        (0, (1i128 << width) - 1)
    };
    let answer = if value < minimum {
        minimum
    } else if value > maximum {
        maximum
    } else {
        value
    };
    (
        answer as u64 & ((1 << width) - 1),
        value < minimum || value > maximum,
    )
}
fn expected(
    source: [u64; 2],
    old: [u64; 2],
    size: u32,
    kind: u32,
    scalar: bool,
    upper: bool,
) -> ([u64; 2], bool) {
    let width = 8 << size;
    let input = source[0] as u128 | ((source[1] as u128) << 64);
    let mut result = 0;
    let mut saturated = false;
    for lane in 0..if scalar { 1 } else { 64 / width } {
        let (answer, sat) = oracle((input >> (lane * width * 2)) as u64, size, kind);
        result |= answer << (lane * width);
        saturated |= sat;
    }
    (
        if !scalar && upper {
            [old[0], result]
        } else {
            [result, 0]
        },
        saturated,
    )
}
fn values(size: u32) -> Vec<u64> {
    let width = 8 << size;
    let source = width * 2;
    let mut values = vec![
        0,
        1,
        2,
        u64::MAX,
        1 << (source - 1),
        (1 << (source - 1)) - 1,
    ];
    for boundary in [1u64 << (width - 1), 1 << width] {
        values.extend([
            boundary - 1,
            boundary,
            boundary + 1,
            0u64.wrapping_sub(boundary - 1),
            0u64.wrapping_sub(boundary),
            0u64.wrapping_sub(boundary + 1),
        ]);
    }
    values
}
fn input(values: &[u64], index: usize, size: u32) -> [u64; 2] {
    let bits = 16 << size;
    let mut input = 0u128;
    let mask = (1u128 << bits) - 1;
    for lane in 0..128 / bits {
        input |=
            ((values[(index + lane as usize * 7) % values.len()] as u128) & mask) << (lane * bits);
    }
    [input as u64, (input >> 64) as u64]
}

#[test]
fn exhaustive_all_16bit_inputs_saturation_and_sticky_qc() {
    for kind in 0..3 {
        let mut ctx = CpuContext::default();
        for bits in 0..=u16::MAX as u64 {
            let (answer, saturated) = oracle(bits, 0, kind);
            for initial in [FLAGS, FLAGS | QC] {
                ctx.fpsr = initial;
                let result = unsafe {
                    saturating::narrow(
                        &mut ctx,
                        bits | (GUARD << 16),
                        GUARD,
                        u64::from((kind << 2) | 16),
                    )
                };
                assert_eq!(result, answer, "kind={kind} source={bits:#x}");
                assert_eq!(ctx.fpsr, initial | if saturated { QC } else { 0 });
            }
        }
    }
}

#[test]
fn every_kind_width_scalar_first_second_alias_and_edge_lane() {
    for kind in 0..3 {
        for size in 0..3 {
            let values = values(size);
            for (scalar, upper) in [(true, false), (false, false), (false, true)] {
                for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
                    let instruction = opcode(kind, size, scalar, upper, rn, rd);
                    let mut code = Code(vec![instruction, 0xd420_0000]);
                    let mut jit = Jit::new(JitConfig::default()).unwrap();
                    for index in 0..values.len() {
                        for flags in [FLAGS, FLAGS | QC] {
                            let mut ctx = CpuContext::default();
                            ctx.pc = CODE;
                            ctx.sp = GUARD;
                            ctx.nzcv = 0xb;
                            ctx.fpcr = 0x100_0000;
                            ctx.fpsr = flags;
                            ctx.x.fill(GUARD);
                            ctx.v.fill([GUARD, !GUARD]);
                            ctx.v[rn as usize] = input(&values, index, size);
                            let mut wanted = ctx.v;
                            let (answer, saturated) = expected(
                                ctx.v[rn as usize],
                                ctx.v[rd as usize],
                                size,
                                kind,
                                scalar,
                                upper,
                            );
                            wanted[rd as usize] = answer;
                            let result = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                            assert_eq!(result.stop, StopReason::Brk(0), "opcode={instruction:#x}");
                            assert_eq!(result.retired, 2);
                            assert_eq!(ctx.v, wanted, "opcode={instruction:#x} sample={index}");
                            assert_eq!(ctx.fpsr, flags | if saturated { QC } else { 0 });
                            assert_eq!(ctx.x, [GUARD; 31]);
                            assert_eq!(ctx.sp, GUARD);
                            assert_eq!(ctx.nzcv, 0xb);
                            assert_eq!(ctx.fpcr, 0x100_0000);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn exact_wonder_chain_and_literal_result() {
    assert_eq!(opcode(0, 1, false, false, 1, 1), 0x0e61_4821);
    assert_eq!(opcode(2, 0, false, false, 1, 1), 0x2e21_2821);
    let mut code = Code(vec![0x0e61_4821, 0x2e21_2821, 0xd420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v[1] = [0x0000_0000_ffff_ffff, 0x0000_0100_0000_00ff]; // −1,0,255,256.
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let result = jit.run_bounded(&mut ctx, &mut code, 3).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(ctx.v[1], [0x0000_0000_ffff_0000, 0]);
    assert_eq!(ctx.fpsr, QC);
}

#[test]
fn overwritten_destination_retains_qc_and_reserved_size_rejects() {
    let mut code = Code(vec![
        opcode(0, 0, false, false, 1, 0),
        0x1e27_03e0,
        0xd420_0000,
    ]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v[1] = [128, 0];
    ctx.fpsr = FLAGS;
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    assert_eq!(
        jit.run_bounded(&mut ctx, &mut code, 3).unwrap().stop,
        StopReason::Brk(0)
    );
    assert_eq!(ctx.v[0], [0, 0]);
    assert_eq!(ctx.fpsr, FLAGS | QC);
    for kind in 0..3 {
        for (scalar, upper) in [(true, false), (false, false), (false, true)] {
            let mut code = Code(vec![opcode(kind, 3, scalar, upper, 31, 31)]);
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.v.fill([GUARD, !GUARD]);
            ctx.fpsr = FLAGS;
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            let result = jit.run_bounded(&mut ctx, &mut code, 1).unwrap();
            assert!(matches!(result.stop, StopReason::Unsupported(_)));
            assert_eq!(result.retired, 0);
            assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
            assert_eq!(ctx.x, [GUARD; 31]);
            assert_eq!(ctx.fpsr, FLAGS);
        }
    }
}
