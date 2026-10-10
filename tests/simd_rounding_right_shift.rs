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
fn instruction(
    accumulate: bool,
    signed: bool,
    size: u32,
    q: bool,
    scalar: bool,
    shift: u32,
    rn: u32,
    rd: u32,
) -> u32 {
    let width = 8 << size;
    (if scalar { 0x5f00_2400 } else { 0x0f00_2400 })
        | (u32::from(accumulate) << 12)
        | (u32::from(!signed) << 29)
        | (u32::from(!scalar && q) << 30)
        | ((2 * width - shift) << 16)
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
// Independent integer oracle: convert each lane to an i128 mathematical
// integer, add the rounding bias in i128, shift, then reduce modulo 2^w.
fn expected(
    source: [u64; 2],
    old: [u64; 2],
    accumulate: bool,
    signed: bool,
    size: u32,
    q: bool,
    scalar: bool,
    shift: u32,
) -> [u64; 2] {
    let source = bytes(source);
    let old = bytes(old);
    let count = 1usize << size;
    let width = count * 8;
    let modulus = 1i128 << width;
    let mut answer = [0; 16];
    for lane in 0..if scalar {
        1
    } else {
        (if q { 16 } else { 8 }) / count
    } {
        let offset = lane * count;
        let mut s = [0; 8];
        let mut a = [0; 8];
        s[..count].copy_from_slice(&source[offset..offset + count]);
        a[..count].copy_from_slice(&old[offset..offset + count]);
        let source = u64::from_le_bytes(s) as i128;
        let value = if signed && source & (modulus >> 1) != 0 {
            source - modulus
        } else {
            source
        };
        let rounded = (value + (1i128 << (shift - 1))) >> shift;
        let old = if accumulate { u64::from_le_bytes(a) as i128 } else { 0 };
        let result = (old + rounded) & (modulus - 1);
        answer[offset..offset + count].copy_from_slice(&(result as u64).to_le_bytes()[..count]);
    }
    vector(answer)
}
fn samples(size: u32) -> Vec<([u64; 2], [u64; 2])> {
    let width = 8 << size;
    let mask = u64::MAX >> (64 - width);
    let sign = 1u64 << (width - 1);
    let values = [0, 1, 2, 3, sign - 1, sign, sign + 1, mask - 1, mask];
    let count = 1usize << size;
    let mut output = Vec::new();
    for source_start in 0..values.len() {
        for accum_start in 0..values.len() {
            let mut source = [0; 16];
            let mut accum = [0; 16];
            for lane in 0..16 / count {
                source[lane * count..(lane + 1) * count].copy_from_slice(
                    &values[(source_start + lane * 3) % values.len()].to_le_bytes()[..count],
                );
                accum[lane * count..(lane + 1) * count].copy_from_slice(
                    &values[(accum_start + lane * 5) % values.len()].to_le_bytes()[..count],
                );
            }
            output.push((vector(source), vector(accum)));
        }
    }
    let mut random = 0x920b_f600_0f3f_1401u64;
    for _ in 0..32 {
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
fn check(signed: bool, size: u32, q: bool, scalar: bool) {
    for accumulate in [false, true] {
    let samples = samples(size);
    for shift in 1..=8 << size {
        for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
            let word = instruction(accumulate, signed, size, q, scalar, shift, rn, rd);
            let mut code = Code(vec![word, 0xd420_0000]);
            let mut compiler = jit();
            for &(source, old) in &samples {
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.x.fill(GUARD);
                ctx.v.fill([GUARD, !GUARD]);
                ctx.sp = GUARD;
                ctx.nzcv = 0xb;
                ctx.fpcr = 0x03c0_0000;
                ctx.fpsr = FLAGS;
                ctx.v[rd as usize] = old;
                ctx.v[rn as usize] = source;
                let mut wanted = ctx.v;
                wanted[rd as usize] =
                    expected(source, ctx.v[rd as usize], accumulate, signed, size, q, scalar, shift);
                let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0), "word{word:x}");
                assert_eq!(result.retired, 2);
                assert_eq!(ctx.v, wanted, "word{word:x} src{source:x?} old{old:x?}");
                assert_eq!(ctx.x, [GUARD; 31]);
                assert_eq!(ctx.sp, GUARD);
                assert_eq!(ctx.nzcv, 0xb);
                assert_eq!(ctx.fpcr, 0x03c0_0000);
                assert_eq!(ctx.fpsr, FLAGS);
            }
        }
    }
    }
}

#[test]
fn vector_signed_all_widths_shifts_aliases_and_modular_accumulation() {
    for (size, q) in [
        (0, false),
        (0, true),
        (1, false),
        (1, true),
        (2, false),
        (2, true),
        (3, true),
    ] {
        check(true, size, q, false);
    }
}
#[test]
fn vector_unsigned_all_widths_shifts_aliases_and_modular_accumulation() {
    for (size, q) in [
        (0, false),
        (0, true),
        (1, false),
        (1, true),
        (2, false),
        (2, true),
        (3, true),
    ] {
        check(false, size, q, false);
    }
}
#[test]
fn scalar_d_all_shifts_signed_unsigned_and_full_width() {
    check(true, 3, false, true);
    check(false, 3, false, true);
}
#[test]
fn exact_wonder_rounding_words_have_literal_results() {
    assert_eq!(instruction(true, true, 2, true, false, 15, 21, 6), 0x4f31_36a6);
    assert_eq!(instruction(false, true, 2, true, false, 15, 23, 4), 0x4f31_26e4);
    let mut code = Code(vec![0x4f31_36a6, 0x4f31_26e4, 0xd420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v.fill([GUARD, !GUARD]);
    ctx.v[21] = [0xffff_c000_0000_4000, 0x8000_0000_7fff_ffff];
    ctx.v[23] = ctx.v[21];
    ctx.v[6] = [0xffff_ffff_0000_0005, 0xffff_ffff_0000_0001];
    let source = ctx.v[21];
    ctx.fpsr = FLAGS;
    let result = jit().run_bounded(&mut ctx, &mut code, 3).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 3);
    assert_eq!(ctx.v[6], [0xffff_ffff_0000_0006, 0xfffe_ffff_0001_0001]);
    assert_eq!(ctx.v[4], [0x0000_0000_0000_0001, 0xffff_0000_0001_0000]);
    assert_eq!(ctx.v[21], source);
    assert_eq!(ctx.v[23], source);
    assert_eq!(ctx.fpsr, FLAGS);
}
#[test]
fn reserved_scalar_widths_and_vector_1d_do_not_retire_or_write() {
    for signed in [false, true] {
        for (size, q, scalar) in [
            (0, false, true),
            (1, false, true),
            (2, false, true),
            (3, false, false),
        ] {
            let word = instruction(true, signed, size, q, scalar, 1, 31, 31);
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
            assert_eq!(ctx.fpsr, FLAGS);
            assert_eq!(ctx.fpcr, 0x03c0_0000);
            assert_eq!(ctx.sp, GUARD);
            assert_eq!(ctx.nzcv, 0xb);
        }
    }
}

#[test]
fn signed_d_full_width_preserves_live_values_under_pressure() {
    for scalar in [false, true] {
        let mut words = Vec::new();
        for r in 0..16 {
            words.push(0x9100_0400 | (r << 5) | r);
            words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
        }
        words.push(instruction(true, true, 3, !scalar, scalar, 64, 31, 31));
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
        ctx.v[31] = [0x8000_0000_0000_0000, 0x7fff_ffff_ffff_ffff];
        let original = ctx.v[31];
        let mut wanted_x = ctx.x;
        let mut wanted_v = ctx.v;
        for r in 0..16 {
            wanted_x[r] = wanted_x[r].wrapping_add(1);
            wanted_v[r] = [
                wanted_v[r][0] | wanted_v[16][0],
                wanted_v[r][1] | wanted_v[16][1],
            ];
        }
        for p in 0..8 {
            wanted_x[if p == 7 { 16 } else { 24 + p }] =
                wanted_x[p * 2].wrapping_add(wanted_x[p * 2 + 1]);
        }
        for r in 0..14 {
            wanted_v[17 + r] = wanted_v[r];
        }
        wanted_v[31] = if scalar {
            [original[0], 0]
        } else {
            original
        };
        let result = jit().run_bounded(&mut ctx, &mut code, count).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, count);
        assert_eq!(ctx.x, wanted_x);
        assert_eq!(ctx.v, wanted_v);
        assert_eq!(ctx.nzcv, 0xb);
        assert_eq!(ctx.fpsr, FLAGS);
    }
}

#[test]
fn scalar_plain_shifts_handle_every_amount_and_clear_upper() {
    for (base, signed, left) in [(0x5f00_0400, true, false), (0x7f00_0400, false, false), (0x5f00_5400, false, true)] {
        for shift in if left { 0..64 } else { 1..65 } {
            let encoded = if left { 64 + shift } else { 128 - shift };
            let word = base | (encoded << 16) | (31 << 5) | 31;
            let mut compiler = jit();
            for source in [0u64, 1, u64::MAX, 1 << 63, (1 << 63) - 1, 0xaaaa_5555_0123_4567] {
                let mut code = Code(vec![word, 0xd420_0000]);
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.v.fill([GUARD, !GUARD]);
                ctx.v[31] = [source, !source];
                ctx.fpsr = FLAGS;
                ctx.nzcv = 0xb;
                let result = compiler.run_bounded(&mut ctx, &mut code, 2).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0), "word {word:x}");
                let expected = if left { source << shift } else if signed {
                    ((source as i64 as i128) >> shift) as u64
                } else { ((source as u128) >> shift) as u64 };
                assert_eq!(ctx.v[31], [expected, 0], "word {word:x}");
                assert_eq!(&ctx.v[..31], &[[GUARD, !GUARD]; 31]);
                assert_eq!(ctx.fpsr, FLAGS);
                assert_eq!(ctx.nzcv, 0xb);
            }
        }
    }
}

#[test]
fn prefetch_hints_do_not_access_unmapped_memory_or_modify_state() {
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    unsafe extern "C" fn forbidden(context: *mut CpuContext, _: u64, _: u8) {
        CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { (*context).mem_fault = 1; }
    }
    // Actual Wonder and SDK hint words, plus an SP-based hint with a signed offset.
    let mut code = Code(vec![0xf880_c354, 0xf89c_0100, 0xf89f_83e0, 0xd420_0000]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x.fill(GUARD);
    ctx.v.fill([GUARD, !GUARD]);
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    ctx.fpcr = 0x03c0_0000;
    ctx.fpsr = FLAGS;
    ctx.mem_read = forbidden;
    ctx.mem_write = forbidden;
    let result = jit().run_bounded(&mut ctx, &mut code, 4).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 4);
    assert_eq!(ctx.x, [GUARD; 31]);
    assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpcr, 0x03c0_0000);
    assert_eq!(ctx.fpsr, FLAGS);
    assert_eq!(CALLS.load(std::sync::atomic::Ordering::Relaxed), 0);
}
