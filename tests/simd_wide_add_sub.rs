use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const ALIASES: [(u32, u32, u32); 10] = [
    (0, 1, 2),
    (0, 1, 0),
    (0, 1, 1),
    (0, 0, 2),
    (0, 0, 0),
    (31, 31, 31),
    (31, 1, 2),
    (0, 31, 2),
    (0, 1, 31),
    (31, 1, 1),
];
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
fn jit() -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        ..JitConfig::default()
    })
    .unwrap()
}
fn instruction(
    signed: bool,
    subtract: bool,
    size: u32,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
) -> u32 {
    0x0e20_1000
        | (u32::from(!signed) << 29)
        | (u32::from(upper) << 30)
        | (size << 22)
        | (rm << 16)
        | (u32::from(subtract) << 13)
        | (rn << 5)
        | rd
}
fn bytes(value: [u64; 2]) -> [u8; 16] {
    let mut result = [0; 16];
    result[..8].copy_from_slice(&value[0].to_le_bytes());
    result[8..].copy_from_slice(&value[1].to_le_bytes());
    result
}
fn vector(value: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(value[..8].try_into().unwrap()),
        u64::from_le_bytes(value[8..].try_into().unwrap()),
    ]
}
// Independent integer oracle: only Vm is widened; Vn keeps all its original
// double-width lanes. Exact addition/subtraction is reduced modulo 2^(2w).
fn expected(
    wide: [u64; 2],
    narrow: [u64; 2],
    signed: bool,
    subtract: bool,
    size: u32,
    upper: bool,
) -> [u64; 2] {
    let wide = bytes(wide);
    let narrow = bytes(narrow);
    let count = 1usize << size;
    let width = count * 8;
    let narrow_modulus = 1i128 << width;
    let mask = (1i128 << (width * 2)) - 1;
    let mut answer = [0; 16];
    for lane in 0..8 / count {
        let wide_offset = lane * count * 2;
        let narrow_offset = usize::from(upper) * 8 + lane * count;
        let mut a = [0; 8];
        let mut b = [0; 8];
        a[..count * 2].copy_from_slice(&wide[wide_offset..wide_offset + count * 2]);
        b[..count].copy_from_slice(&narrow[narrow_offset..narrow_offset + count]);
        let value = u64::from_le_bytes(b) as i128;
        let value = if signed && value & (narrow_modulus >> 1) != 0 {
            value - narrow_modulus
        } else {
            value
        };
        let result =
            ((u64::from_le_bytes(a) as i128 + if subtract { -value } else { value }) & mask) as u64;
        answer[wide_offset..wide_offset + count * 2]
            .copy_from_slice(&result.to_le_bytes()[..count * 2]);
    }
    vector(answer)
}
fn boundaries(width: usize) -> [u64; 9] {
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    [0, 1, 2, 3, sign - 1, sign, sign + 1, mask - 1, mask]
}
fn pattern(size: u32, wide: bool, start: usize) -> [u64; 2] {
    let count = (1usize << size) * if wide { 2 } else { 1 };
    let values = boundaries(count * 8);
    let mut result = [0; 16];
    for lane in 0..16 / count {
        result[lane * count..(lane + 1) * count]
            .copy_from_slice(&values[(start + lane * 5) % values.len()].to_le_bytes()[..count]);
    }
    vector(result)
}
fn samples(size: u32) -> Vec<([u64; 2], [u64; 2])> {
    let mut output = Vec::new();
    for wide in 0..9 {
        for narrow in 0..9 {
            output.push((pattern(size, true, wide), pattern(size, false, narrow)));
        }
    }
    let mut random = 0x91a3_bac0_2e61_1000u64;
    for _ in 0..64 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        output.push(([next(), next()], [next(), next()]));
    }
    output
}
fn check(signed: bool, subtract: bool, size: u32, upper: bool, samples: &[([u64; 2], [u64; 2])]) {
    for (rn, rm, rd) in ALIASES {
        let word = instruction(signed, subtract, size, upper, rn, rm, rd);
        let mut code = Code(vec![word, 0xd420_0000]);
        let mut compiler = jit();
        for &(wide, narrow) in samples {
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.v.fill([GUARD, !GUARD]);
            ctx.v[rn as usize] = wide;
            ctx.v[rm as usize] = narrow;
            ctx.sp = GUARD;
            ctx.nzcv = 0xb;
            ctx.fpcr = 0x03c0_0000;
            ctx.fpsr = FLAGS;
            let mut wanted = ctx.v;
            wanted[rd as usize] = expected(
                ctx.v[rn as usize],
                ctx.v[rm as usize],
                signed,
                subtract,
                size,
                upper,
            );
            let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
            assert_eq!(result.stop, StopReason::Brk(0), "word{word:x}");
            assert_eq!(result.retired, 2);
            assert_eq!(
                ctx.v, wanted,
                "word{word:x} wide{wide:x?} narrow{narrow:x?}"
            );
            assert_eq!(ctx.x, [GUARD; 31]);
            assert_eq!(ctx.sp, GUARD);
            assert_eq!(ctx.nzcv, 0xb);
            assert_eq!(ctx.fpcr, 0x03c0_0000);
            assert_eq!(ctx.fpsr, FLAGS);
        }
    }
}
#[test]
fn signed_add_sub_all_widths_halves_aliases_boundaries_and_random_values() {
    for size in 0..3 {
        let samples = samples(size);
        for upper in [false, true] {
            for subtract in [false, true] {
                check(true, subtract, size, upper, &samples);
            }
        }
    }
}
#[test]
fn unsigned_add_sub_all_widths_halves_aliases_boundaries_and_random_values() {
    for size in 0..3 {
        let samples = samples(size);
        for upper in [false, true] {
            for subtract in [false, true] {
                check(false, subtract, size, upper, &samples);
            }
        }
    }
}
#[test]
fn every_byte_value_against_every_wide_boundary_with_aliases_and_half_selection() {
    let mut sources = Vec::new();
    for value in 0u8..=255 {
        let mut narrow = [0; 16];
        for (lane, byte) in narrow.iter_mut().enumerate() {
            *byte = value.wrapping_add((lane as u8).wrapping_mul(37));
        }
        for wide in 0..9 {
            sources.push((pattern(0, true, wide), vector(narrow)));
        }
    }
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                check(signed, subtract, 0, upper, &sources);
            }
        }
    }
}
#[test]
fn exact_wonder_uaddw_words_preserve_both_aliased_original_operands() {
    assert_eq!(instruction(false, false, 1, false, 0, 1, 0), 0x2e61_1000);
    assert_eq!(instruction(false, false, 1, false, 1, 0, 0), 0x2e60_1020);
    for count in 1..=3 {
        let mut words = vec![0x2e61_1000];
        words.extend(std::iter::repeat_n(0x2e60_1020, count - 1));
        words.push(0xd420_0000);
        let mut code = Code(words);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.v.fill([GUARD, !GUARD]);
        ctx.v[0] = [0x7fff_ffff_ffff_fffe, 0x0000_0001_8000_0000];
        ctx.v[1] = [0xffff_8000_0001_0002, 0xffff_0001_8000_7fff];
        ctx.fpsr = FLAGS;
        let mut wanted = ctx.v;
        wanted[0] = match count {
            1 => [0x8000_0000_0000_0000, 0x0001_0000_8000_8000],
            2 => [0xffff_8000_0001_0002, 0xffff_8001_8000_7fff],
            3 => [0xffff_8001_0001_0004, 0x0000_0000_8000_ffff],
            _ => unreachable!(),
        };
        let result = jit()
            .run_bounded(&mut ctx, &mut code, (count + 1) as u64)
            .unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, (count + 1) as u64);
        assert_eq!(ctx.v, wanted, "prefix{count}");
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
#[test]
fn reserved_narrow_d_widths_do_not_retire_or_write() {
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                let word = instruction(signed, subtract, 3, upper, 31, 31, 31);
                let mut code = Code(vec![word]);
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.v.fill([GUARD, !GUARD]);
                ctx.x.fill(GUARD);
                ctx.sp = GUARD;
                ctx.nzcv = 0xb;
                ctx.fpcr = 0x03c0_0000;
                ctx.fpsr = FLAGS;
                let result = jit().run_bounded(&mut ctx, &mut code, 1).unwrap();
                assert!(
                    matches!(result.stop, StopReason::Unsupported(_)),
                    "word{word:x}"
                );
                assert_eq!(result.retired, 0);
                assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
                assert_eq!(ctx.x, [GUARD; 31]);
                assert_eq!(ctx.sp, GUARD);
                assert_eq!(ctx.nzcv, 0xb);
                assert_eq!(ctx.fpcr, 0x03c0_0000);
                assert_eq!(ctx.fpsr, FLAGS);
            }
        }
    }
}
#[test]
fn all_forms_preserve_live_gpr_and_vector_values_under_pressure() {
    for signed in [false, true] {
        for subtract in [false, true] {
            for upper in [false, true] {
                for size in 0..3 {
                    let mut words = Vec::new();
                    for r in 0..16 {
                        words.push(0x9100_0400 | (r << 5) | r);
                        words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
                    }
                    words.push(instruction(signed, subtract, size, upper, 31, 30, 31));
                    for p in 0..8 {
                        words.push(
                            0x8b00_0000
                                | ((p * 2 + 1) << 16)
                                | ((p * 2) << 5)
                                | if p == 7 { 16 } else { 24 + p },
                        );
                    }
                    for r in 0..14 {
                        words.push(0x4ea0_1c00 | (r << 16) | (r << 5) | (17 + r));
                    }
                    words.push(0xd420_0000);
                    let count = words.len() as u64;
                    let mut code = Code(words);
                    let mut ctx = CpuContext::default();
                    ctx.pc = CODE;
                    ctx.nzcv = 0xb;
                    ctx.fpsr = FLAGS;
                    for r in 0..31 {
                        ctx.x[r] = GUARD.wrapping_add(r as u64);
                    }
                    for r in 0..32 {
                        ctx.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
                    }
                    ctx.v[31] = pattern(size, true, 5);
                    ctx.v[30] = pattern(size, false, 4);
                    let mut wanted_x = ctx.x;
                    let mut wanted_v = ctx.v;
                    for r in 0..16 {
                        wanted_x[r] = wanted_x[r].wrapping_add(1);
                        wanted_v[r] = [
                            wanted_v[r][0] | wanted_v[16][0],
                            wanted_v[r][1] | wanted_v[16][1],
                        ];
                    }
                    wanted_v[31] =
                        expected(wanted_v[31], wanted_v[30], signed, subtract, size, upper);
                    for p in 0..8 {
                        wanted_x[if p == 7 { 16 } else { 24 + p }] =
                            wanted_x[p * 2].wrapping_add(wanted_x[p * 2 + 1]);
                    }
                    for r in 0..14 {
                        wanted_v[17 + r] = wanted_v[r];
                    }
                    let result = jit().run_bounded(&mut ctx, &mut code, count).unwrap();
                    assert_eq!(result.stop, StopReason::Brk(0));
                    assert_eq!(result.retired, count);
                    assert_eq!(ctx.x, wanted_x);
                    assert_eq!(ctx.v, wanted_v);
                    assert_eq!(ctx.nzcv, 0xb);
                    assert_eq!(ctx.fpsr, FLAGS);
                }
            }
        }
    }
}
