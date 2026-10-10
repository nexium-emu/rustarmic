use rustarmic::{CpuContext, CpuFeatures, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const DATA: u64 = 0x4000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
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
fn jit(bsr: bool) -> Jit {
    Jit::new(JitConfig {
        use_fastmem: true,
        code_cache_bytes: 1024 * 1024,
        host_features: bsr.then_some(CpuFeatures {
            has_sse41: true,
            has_ssse3: true,
            ..CpuFeatures::default()
        }),
        ..JitConfig::default()
    })
    .unwrap()
}
fn instruction(signed: bool, size: u32, q: bool, rn: u32, rd: u32) -> u32 {
    0x0e20_4800 | (u32::from(!signed) << 29) | (u32::from(q) << 30) | (size << 22) | (rn << 5) | rd
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
// Bit-walk oracle does not use host leading_zeros, CLZ32 prefix correction,
// sign-mask normalization, or the backend's BSR/LZCNT paths.
fn count(value: u64, width: usize, signed: bool) -> u64 {
    let wanted = if signed {
        (value >> (width - 1)) & 1
    } else {
        0
    };
    let first = if signed { width - 1 } else { width };
    let mut result = 0;
    for bit in (0..first).rev() {
        if (value >> bit) & 1 != wanted {
            break;
        }
        result += 1;
    }
    result
}
fn expected(source: [u64; 2], signed: bool, size: u32, q: bool) -> [u64; 2] {
    let source = bytes(source);
    let count_bytes = 1usize << size;
    let mut output = [0; 16];
    for start in (0..if q { 16 } else { 8 }).step_by(count_bytes) {
        let mut raw = [0; 8];
        raw[..count_bytes].copy_from_slice(&source[start..start + count_bytes]);
        let answer = count(u64::from_le_bytes(raw), count_bytes * 8, signed);
        output[start..start + count_bytes].copy_from_slice(&answer.to_le_bytes()[..count_bytes]);
    }
    vector(output)
}
fn samples(size: u32) -> Vec<[u64; 2]> {
    let count_bytes = 1usize << size;
    let width = count_bytes * 8;
    let mask = (1u64 << width) - 1;
    let mut values = vec![0, mask];
    for bit in 0..width {
        let value = 1u64 << bit;
        values.extend([
            value,
            value - 1,
            (value + 1) & mask,
            !value & mask,
            !(value - 1) & mask,
            !(value + 1) & mask,
        ]);
    }
    let mut output = Vec::new();
    for start in 0..values.len() {
        let mut raw = [0; 16];
        for lane in 0..16 / count_bytes {
            raw[lane * count_bytes..(lane + 1) * count_bytes].copy_from_slice(
                &values[(start + lane * 5) % values.len()].to_le_bytes()[..count_bytes],
            );
        }
        output.push(vector(raw));
    }
    let mut random = 0x804f_0380_2ea0_4800u64;
    for _ in 0..64 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        output.push([next(), next()]);
    }
    output
}
fn run_cases(signed: bool, size: u32, q: bool, bsr: bool, inputs: &[[u64; 2]]) {
    for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
        let word = instruction(signed, size, q, rn, rd);
        let mut code = Code(vec![word, 0xd420_0000]);
        let mut compiler = jit(bsr);
        for &source in inputs {
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.v.fill([GUARD, !GUARD]);
            ctx.v[rn as usize] = source;
            ctx.sp = GUARD;
            ctx.nzcv = 0xb;
            ctx.fpcr = 0x03c0_0000;
            ctx.fpsr = FLAGS;
            let mut wanted = ctx.v;
            wanted[rd as usize] = expected(source, signed, size, q);
            let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
            assert_eq!(result.stop, StopReason::Brk(0), "word{word:x} bsr{bsr}");
            assert_eq!(result.retired, 2);
            assert_eq!(ctx.v, wanted, "word{word:x} bsr{bsr} source{source:x?}");
            assert_eq!(ctx.x, [GUARD; 31]);
            assert_eq!(ctx.sp, GUARD);
            assert_eq!(ctx.nzcv, 0xb);
            assert_eq!(ctx.fpcr, 0x03c0_0000);
            assert_eq!(ctx.fpsr, FLAGS);
        }
    }
}
#[test]
fn clz_all_widths_q_aliases_leading_positions_and_random_values() {
    for size in 0..3 {
        for q in [false, true] {
            run_cases(false, size, q, false, &samples(size));
        }
    }
}
#[test]
fn cls_all_widths_q_aliases_leading_sign_positions_and_random_values() {
    for size in 0..3 {
        for q in [false, true] {
            run_cases(true, size, q, false, &samples(size));
        }
    }
}
#[test]
fn baseline_without_lzcnt_matches_all_leading_positions_and_zero_edges() {
    for signed in [false, true] {
        for size in 0..3 {
            for q in [false, true] {
                run_cases(signed, size, q, true, &samples(size));
            }
        }
    }
}
#[test]
fn every_byte_and_halfword_value_matches_both_kinds_with_and_without_lzcnt() {
    for size in 0..2 {
        let count_bytes = 1usize << size;
        let maximum = (1u32 << (count_bytes * 8)) - 1;
        let inputs: Vec<_> = (0..=maximum)
            .map(|value| {
                let mut raw = [0; 16];
                for lane in 0..16 / count_bytes {
                    let value = value.wrapping_add((lane as u32).wrapping_mul(37)) & maximum;
                    raw[lane * count_bytes..(lane + 1) * count_bytes]
                        .copy_from_slice(&value.to_le_bytes()[..count_bytes]);
                }
                vector(raw)
            })
            .collect();
        // All65,536 halfword inputs, and all256 byte inputs, occupy every lane
        // over the cyclic pattern. Alias tests above cover all other layouts.
        for signed in [false, true] {
            for bsr in [false, true] {
                let word = instruction(signed, size, true, 0, 1);
                let mut code = Code(vec![word, 0xd420_0000]);
                let mut compiler = jit(bsr);
                for &source in &inputs {
                    let mut ctx = CpuContext::default();
                    ctx.pc = CODE;
                    ctx.v.fill([GUARD, !GUARD]);
                    ctx.v[0] = source;
                    ctx.nzcv = 0xb;
                    ctx.fpsr = FLAGS;
                    let mut wanted = ctx.v;
                    wanted[1] = expected(source, signed, size, true);
                    let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
                    assert_eq!(result.stop, StopReason::Brk(0));
                    assert_eq!(result.retired, 2);
                    assert_eq!(
                        ctx.v, wanted,
                        "size{size} signed{signed} bsr{bsr} source{source:x?}"
                    );
                    assert_eq!(ctx.nzcv, 0xb);
                    assert_eq!(ctx.fpsr, FLAGS);
                }
            }
        }
    }
}
#[test]
fn exact_wonder_width_prefix_retains_cmp_flags_and_byte_result() {
    assert_eq!(instruction(false, 2, false, 0, 0), 0x2ea0_4800);
    let words = vec![
        0xb940_96c8,
        0x5280_03aa,
        0x5100_0509,
        0x7100_011f,
        0x0e04_0d20,
        0x2ea0_4800,
        0x0e01_3c09,
        0x4b09_0149,
        0x1a9f_c128,
        0x3900_72c8,
        0xd420_0000,
    ];
    let mut values = vec![
        0,
        1,
        2,
        3,
        5,
        7,
        15,
        16,
        17,
        0x7fff_ffff,
        0x8000_0000,
        0xffff_ffff,
    ];
    for bit in 0..31 {
        values.extend([1u32 << bit, (1u32 << bit) + 1]);
    }
    for bsr in [false, true] {
        let mut compiler = jit(bsr);
        let mut code = Code(words.clone());
        for &value in &values {
            let mut data = [0xa5; 256];
            data[0x94..0x98].copy_from_slice(&value.to_le_bytes());
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.x.fill(GUARD);
            ctx.x[22] = DATA;
            ctx.v.fill([GUARD, !GUARD]);
            ctx.sp = GUARD;
            ctx.fpsr = FLAGS;
            ctx.mem_base = data.as_mut_ptr();
            ctx.mem_base_va = DATA;
            ctx.mem_size = data.len() as u64;
            let leading = count(u64::from(value.wrapping_sub(1)), 32, false) as u32;
            let index = 29u32.wrapping_sub(leading);
            let wanted = if value as i32 > 0 { index } else { 0 };
            let mut expected_data = data;
            expected_data[0x1c] = wanted as u8;
            let mut wanted_x = ctx.x;
            wanted_x[8] = u64::from(wanted);
            wanted_x[9] = u64::from(index);
            wanted_x[10] = 29;
            let mut wanted_v = ctx.v;
            wanted_v[0] = [u64::from(leading) | (u64::from(leading) << 32), 0];
            let result = compiler
                .run_bounded(&mut ctx, &mut code, words.len() as u64)
                .unwrap();
            assert_eq!(result.stop, StopReason::Brk(0), "value{value:x} bsr{bsr}");
            assert_eq!(result.retired, words.len() as u64);
            assert_eq!(ctx.x, wanted_x, "value{value:x} bsr{bsr}");
            assert_eq!(ctx.v, wanted_v);
            assert_eq!(data, expected_data);
            assert_eq!(
                ctx.nzcv,
                if value == 0 {
                    6
                } else if value >> 31 != 0 {
                    0xa
                } else {
                    2
                }
            );
            assert_eq!(ctx.sp, GUARD);
            assert_eq!(ctx.fpsr, FLAGS);
        }
    }
}
#[test]
fn reserved_d_vectors_do_not_retire_or_write() {
    for signed in [false, true] {
        for q in [false, true] {
            let word = instruction(signed, 3, q, 31, 31);
            let mut code = Code(vec![word]);
            let mut ctx = CpuContext::default();
            ctx.pc = CODE;
            ctx.v.fill([GUARD, !GUARD]);
            ctx.x.fill(GUARD);
            ctx.sp = GUARD;
            ctx.nzcv = 0xb;
            ctx.fpcr = 0x03c0_0000;
            ctx.fpsr = FLAGS;
            let result = jit(false).run_bounded(&mut ctx, &mut code, 1).unwrap();
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
#[test]
fn narrow_counts_preserve_live_gpr_and_simd_values_under_pressure() {
    for signed in [false, true] {
        for bsr in [false, true] {
            let mut words = Vec::new();
            for r in 0..16 {
                words.push(0x9100_0400 | (r << 5) | r);
                words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
            }
            words.push(instruction(signed, 0, true, 31, 31));
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
            ctx.v[31] = [0x8000_ff01_7ffe_4000, 0x1020_4080_0001_ffff];
            let mut wanted_x = ctx.x;
            let mut wanted_v = ctx.v;
            for r in 0..16 {
                wanted_x[r] = wanted_x[r].wrapping_add(1);
                wanted_v[r] = [
                    wanted_v[r][0] | wanted_v[16][0],
                    wanted_v[r][1] | wanted_v[16][1],
                ];
            }
            wanted_v[31] = expected(wanted_v[31], signed, 0, true);
            for p in 0..8 {
                wanted_x[if p == 7 { 16 } else { 24 + p }] =
                    wanted_x[p * 2].wrapping_add(wanted_x[p * 2 + 1]);
            }
            for r in 0..14 {
                wanted_v[17 + r] = wanted_v[r];
            }
            let result = jit(bsr).run_bounded(&mut ctx, &mut code, count).unwrap();
            assert_eq!(result.stop, StopReason::Brk(0));
            assert_eq!(result.retired, count);
            assert_eq!(ctx.x, wanted_x);
            assert_eq!(ctx.v, wanted_v);
            assert_eq!(ctx.nzcv, 0xb);
            assert_eq!(ctx.fpsr, FLAGS);
        }
    }
}
