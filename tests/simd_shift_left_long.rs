use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
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
fn jit() -> Jit {
    Jit::new(JitConfig {
        code_cache_bytes: 1024 * 1024,
        ..JitConfig::default()
    })
    .unwrap()
}
fn instruction(signed: bool, size: u32, upper: bool, shift: u32, rn: u32, rd: u32) -> u32 {
    0x0f00_a400
        | (u32::from(!signed) << 29)
        | (u32::from(upper) << 30)
        | (((8 << size) + shift) << 16)
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
// Decode each source lane as an independent mathematical integer, shift it,
// and reduce modulo the doubled destination width. No SIMD implementation or
// native casting/shift path is shared with this oracle.
fn expected(source: [u64; 2], signed: bool, size: u32, upper: bool, shift: u32) -> [u64; 2] {
    let source = bytes(source);
    let count = 1usize << size;
    let width = count * 8;
    let modulus = 1i128 << width;
    let destination_mask = (1i128 << (width * 2)) - 1;
    let mut answer = [0; 16];
    for lane in 0..8 / count {
        let offset = usize::from(upper) * 8 + lane * count;
        let mut raw = [0; 8];
        raw[..count].copy_from_slice(&source[offset..offset + count]);
        let value = u64::from_le_bytes(raw) as i128;
        let value = if signed && value & (modulus >> 1) != 0 {
            value - modulus
        } else {
            value
        };
        let result = ((value << shift) & destination_mask) as u64;
        answer[lane * count * 2..(lane + 1) * count * 2]
            .copy_from_slice(&result.to_le_bytes()[..count * 2]);
    }
    vector(answer)
}
fn samples(size: u32) -> Vec<[u64; 2]> {
    let count = 1usize << size;
    let width = count * 8;
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    let values = [0, 1, 2, 3, sign - 1, sign, sign + 1, mask - 1, mask];
    let mut output = Vec::new();
    for start in 0..values.len() {
        let mut source = [0; 16];
        for lane in 0..16 / count {
            source[lane * count..(lane + 1) * count]
                .copy_from_slice(&values[(start + lane * 5) % values.len()].to_le_bytes()[..count]);
        }
        output.push(vector(source));
    }
    let mut random = 0x91a5_2340_2f08_a400u64;
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
fn check_samples(signed: bool, size: u32, upper: bool, samples: &[[u64; 2]]) {
    for shift in 0..8 << size {
        for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
            let word = instruction(signed, size, upper, shift, rn, rd);
            let mut code = Code(vec![word, 0xd420_0000]);
            let mut compiler = jit();
            for &source in samples {
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
                wanted[rd as usize] = expected(source, signed, size, upper, shift);
                let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0), "word{word:x}");
                assert_eq!(result.retired, 2);
                assert_eq!(ctx.v, wanted, "word{word:x} src{source:x?}");
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
fn signed_all_widths_shifts_source_halves_aliases_and_state_guards() {
    for size in 0..3 {
        let samples = samples(size);
        for upper in [false, true] {
            check_samples(true, size, upper, &samples);
        }
    }
}
#[test]
fn unsigned_all_widths_shifts_source_halves_aliases_and_state_guards() {
    for size in 0..3 {
        let samples = samples(size);
        for upper in [false, true] {
            check_samples(false, size, upper, &samples);
        }
    }
}
#[test]
fn every_byte_value_at_every_shift_in_each_half_signed_and_unsigned() {
    let sources: Vec<_> = (0u8..=255)
        .map(|value| {
            let mut source = [0; 16];
            for (lane, byte) in source.iter_mut().enumerate() {
                *byte = value.wrapping_add((lane as u8).wrapping_mul(37));
            }
            vector(source)
        })
        .collect();
    for signed in [false, true] {
        for upper in [false, true] {
            check_samples(signed, 0, upper, &sources);
        }
    }
}
#[test]
fn exact_wonder_unsigned_compare_signed_alias_chain_has_literal_results() {
    assert_eq!(instruction(false, 0, false, 0, 0, 0), 0x2f08_a400);
    assert_eq!(instruction(true, 1, false, 0, 0, 0), 0x0f10_a400);
    let mut code = Code(vec![0x2f08_a400, 0x0e60_9800, 0x0f10_a400, 0xd420_0000]);
    // USHLL V0.8H,V0.8B,#0; CMEQ V0.4H,V0.4H,#0;
    // SSHLL V0.4S,V0.4H,#0. Zero byte comparisons become signed minus one.
    let mut compiler = jit();
    for mask in 0u8..16 {
        let mut source = [0xa5; 16];
        for lane in 0..4 {
            source[lane] = if mask & (1 << lane) != 0 { 0 } else { 0xff };
        }
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.v.fill([GUARD, !GUARD]);
        ctx.v[0] = vector(source);
        ctx.fpsr = FLAGS;
        let mut wanted = ctx.v;
        let mut output = [0; 16];
        for lane in 0..4 {
            output[lane * 4..lane * 4 + 4].fill(if mask & (1 << lane) != 0 { 0xff } else { 0 });
        }
        wanted[0] = vector(output);
        let result = compiler.run_bounded(&mut ctx, &mut code, 4).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, 4);
        assert_eq!(ctx.v, wanted, "mask{mask:x}");
        assert_eq!(ctx.fpsr, FLAGS);
    }
}
#[test]
fn reserved_d_to_128_widths_do_not_retire_or_write() {
    for signed in [false, true] {
        for upper in [false, true] {
            for shift in [0, 1, 31, 63] {
                let word = instruction(signed, 3, upper, shift, 31, 31);
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
fn maximal_signed_upper_shift_preserves_live_values_under_pressure() {
    let mut words = Vec::new();
    for r in 0..16 {
        words.push(0x9100_0400 | (r << 5) | r);
        words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
    }
    words.push(instruction(true, 2, true, 31, 31, 31));
    for p in 0..8 {
        words.push(
            0x8b00_0000 | ((p * 2 + 1) << 16) | ((p * 2) << 5) | if p == 7 { 16 } else { 24 + p },
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
    ctx.v[31] = [GUARD, 0x7fff_ffff_8000_0001];
    let mut wanted_x = ctx.x;
    let mut wanted_v = ctx.v;
    for r in 0..16 {
        wanted_x[r] = wanted_x[r].wrapping_add(1);
        wanted_v[r] = [
            wanted_v[r][0] | wanted_v[16][0],
            wanted_v[r][1] | wanted_v[16][1],
        ];
    }
    wanted_v[31] = [0xc000_0000_8000_0000, 0x3fff_ffff_8000_0000];
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
