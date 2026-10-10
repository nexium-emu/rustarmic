pub use rustarmic::jit;
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
#[path = "../src/backend/fp_minmax_number.rs"]
mod fp_minmax_number;
#[allow(dead_code)]
#[path = "../src/backend/fp_reduce.rs"]
mod fp_reduce;

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const BRK: u32 = 0xd420_0000;
const DN: u32 = 1 << 25;
const FZ: u32 = 1 << 24;
const STICKY: u32 = (1 << 27) | (1 << 4);
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
fn word(minimum: bool, double: bool, q: bool, rn: u32, rm: u32, rd: u32) -> u32 {
    0x0e20_c400
        | (u32::from(minimum) << 23)
        | (u32::from(double) << 22)
        | (u32::from(q) << 30)
        | (rm << 16)
        | (rn << 5)
        | rd
}
fn classes(double: bool) -> Vec<u64> {
    if double {
        vec![
            0,
            0x8000_0000_0000_0000,
            1,
            0x000f_ffff_ffff_ffff,
            0x8000_0000_0000_0001,
            0x800f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x8010_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x4000_0000_0000_0000,
            0xc000_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0123,
            0xfff8_0000_0000_0456,
            0x7ff0_0000_0000_0789,
            0xfff0_0000_0000_0abc,
        ]
    } else {
        vec![
            0,
            0x8000_0000,
            1,
            0x007f_ffff,
            0x8000_0001,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x4000_0000,
            0xc000_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0123,
            0xffc0_0456,
            0x7f80_0789,
            0xff80_0abc,
        ]
    }
}
// Independent oracle: integer classification, but ordinary host comparisons
// for numeric ordering (not the implementation's transformed-integer keys).
// Never use host min/max: their NaN and signed-zero selection is different.
fn pair(mut n: u64, mut m: u64, double: bool, minimum: bool, fpcr: u32) -> (u64, u32) {
    let (sign, infinity, quiet) = if double {
        (1 << 63, 0x7ff0_0000_0000_0000, 1 << 51)
    } else {
        (1 << 31, 0x7f80_0000, 1 << 22)
    };
    let mut flags = 0;
    for value in [&mut n, &mut m] {
        if fpcr & FZ != 0 && *value & infinity == 0 && *value & (infinity - 1) != 0 {
            *value &= sign;
            flags |= 1 << 7;
        }
    }
    let nan_n = n & !sign > infinity;
    let nan_m = m & !sign > infinity;
    let signaling = if nan_n && n & quiet == 0 {
        Some(n)
    } else if nan_m && m & quiet == 0 {
        Some(m)
    } else {
        None
    };
    if let Some(selected) = signaling {
        return (
            if fpcr & DN != 0 {
                infinity | quiet
            } else {
                selected | quiet
            },
            flags | 1,
        );
    }
    if nan_n || nan_m {
        if nan_n != nan_m {
            return (if nan_n { m } else { n }, flags);
        }
        return (if fpcr & DN != 0 { infinity | quiet } else { n }, flags);
    }
    if n & !sign == 0 && m & !sign == 0 {
        return (
            if minimum {
                (n | m) & sign
            } else {
                (n & m) & sign
            },
            flags,
        );
    }
    let n_less = if double {
        f64::from_bits(n) < f64::from_bits(m)
    } else {
        f32::from_bits(n as u32) < f32::from_bits(m as u32)
    };
    (if n_less == minimum { n } else { m }, flags)
}
fn extract(v: [u64; 2], lane: usize, double: bool) -> u64 {
    if double {
        v[lane]
    } else {
        (v[lane / 2] >> (32 * (lane % 2))) & 0xffff_ffff
    }
}
fn put(v: &mut [u64; 2], lane: usize, double: bool, value: u64) {
    if double {
        v[lane] = value;
    } else {
        v[lane / 2] |= (value & 0xffff_ffff) << (32 * (lane % 2));
    }
}
fn vector(values: &[u64], index: usize, double: bool) -> [u64; 2] {
    let mut result = [0; 2];
    for lane in 0..if double { 2 } else { 4 } {
        put(
            &mut result,
            lane,
            double,
            values[(index + lane * 7) % values.len()],
        );
    }
    result
}
fn expected(
    n: [u64; 2],
    m: [u64; 2],
    double: bool,
    q: bool,
    minimum: bool,
    fpcr: u32,
) -> ([u64; 2], u32) {
    let mut result = [0; 2];
    let mut flags = 0;
    for lane in 0..if double || !q { 2 } else { 4 } {
        let (value, status) = pair(
            extract(n, lane, double),
            extract(m, lane, double),
            double,
            minimum,
            fpcr,
        );
        put(&mut result, lane, double, value);
        flags |= status;
    }
    (result, flags)
}
fn context(fpcr: u32) -> CpuContext {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x.fill(GUARD);
    ctx.v.fill([GUARD, !GUARD]);
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    ctx.fpcr = fpcr;
    ctx.fpsr = STICKY;
    ctx
}
fn check(
    compiler: &mut Jit,
    code: &mut Code,
    ctx: &mut CpuContext,
    double: bool,
    q: bool,
    minimum: bool,
    rn: usize,
    rm: usize,
    rd: usize,
) {
    let mut want = ctx.v;
    let (answer, status) = expected(want[rn], want[rm], double, q, minimum, ctx.fpcr);
    want[rd] = answer;
    let fpcr = ctx.fpcr;
    let result = compiler.run_bounded(ctx, code, 2).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 2);
    assert_eq!(ctx.v, want, "word{:08x} fpcr{fpcr:x}", code.0[0]);
    assert_eq!(ctx.fpsr, STICKY | status);
    assert_eq!(ctx.fpcr, fpcr);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.x, [GUARD; 31]);
}

#[test]
fn independent_oracle_has_architectural_nan_zero_and_numeric_literals() {
    assert_eq!(
        pair(0x7fc0_0123, 0x3f80_0000, false, true, DN),
        (0x3f80_0000, 0)
    );
    assert_eq!(
        pair(0x7fc0_0123, 0x7f80_0456, false, true, 0),
        (0x7fc0_0456, 1)
    );
    assert_eq!(
        pair(0xff80_0123, 0x7f80_0456, false, false, 0),
        (0xffc0_0123, 1)
    );
    assert_eq!(pair(0x8000_0000, 0, false, true, 0), (0x8000_0000, 0));
    assert_eq!(pair(0x8000_0000, 0, false, false, 0), (0, 0));
    assert_eq!(
        pair(0x7ff8_0000_0000_0123, 0x7ff0_0000_0000_0456, true, true, 0),
        (0x7ff8_0000_0000_0456, 1)
    );
    assert_eq!(
        pair(1, 0x8000_0000_0000_0000, true, true, FZ),
        (0x8000_0000_0000_0000, 1 << 7)
    );
    assert_eq!(
        pair(0xc000_0000_0000_0000, 0xbff0_0000_0000_0000, true, false, 0),
        (0xbff0_0000_0000_0000, 0)
    );
}

#[test]
fn helper_every_ordered_class_pair_matches_independent_oracle() {
    for double in [false, true] {
        let values = classes(double);
        for minimum in [false, true] {
            for fpcr in [0, DN, FZ, DN | FZ, 3 << 22] {
                for &n in &values {
                    for &m in &values {
                        let mut ctx = context(fpcr);
                        let (want, status) = pair(n, m, double, minimum, fpcr);
                        let result = unsafe {
                            if double {
                                fp_minmax_number::double(&mut ctx, n, m, u64::from(minimum))
                            } else {
                                fp_minmax_number::single(
                                    &mut ctx,
                                    n | (GUARD << 32),
                                    m | (GUARD << 32),
                                    u64::from(minimum),
                                )
                            }
                        };
                        assert_eq!(
                            result, want,
                            "double{double} min{minimum} fpcr{fpcr:x} n{n:x} m{m:x}"
                        );
                        assert_eq!(ctx.fpsr, STICKY | status);
                        assert_eq!(ctx.fpcr, fpcr);
                        assert_eq!(ctx.nzcv, 0xb);
                    }
                }
            }
        }
    }
}

#[test]
fn all_vector_forms_pair_classes_random_aliases_and_full_state_guards() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        let values = classes(double);
        let mut samples = Vec::new();
        for a in 0..values.len() {
            for b in 0..values.len() {
                samples.push((vector(&values, a, double), vector(&values, b, double)));
            }
        }
        let mut random = 0x933e_bc8c_4ea5_c400u64;
        for _ in 0..128 {
            let mut pair = [[0; 2]; 2];
            for v in &mut pair {
                for value in v {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    *value = random;
                }
            }
            samples.push((pair[0], pair[1]));
        }
        for minimum in [false, true] {
            for (rn, rm, rd) in [
                (1, 2, 0),
                (0, 1, 0),
                (1, 0, 0),
                (0, 0, 0),
                (1, 1, 0),
                (31, 0, 31),
                (0, 31, 31),
                (31, 31, 31),
            ] {
                let mut code = Code(vec![word(minimum, double, q, rn, rm, rd), BRK]);
                let mut compiler = jit();
                for fpcr in [0, DN, FZ, DN | FZ, 3 << 22] {
                    for &(n, m) in &samples {
                        let mut ctx = context(fpcr);
                        ctx.v[rn as usize] = n;
                        ctx.v[rm as usize] = m;
                        check(
                            &mut compiler,
                            &mut code,
                            &mut ctx,
                            double,
                            q,
                            minimum,
                            rn as usize,
                            rm as usize,
                            rd as usize,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn exact_wonder_aliased_numeric_clamp_and_rounding_prefix() {
    assert_eq!(word(true, false, true, 0, 5, 0), 0x4ea5_c400);
    let mut code = Code(vec![
        0x4ea5_c400,
        0x4e26_f400,
        0x4e21_d400,
        0x4ea1_b800,
        BRK,
    ]);
    let mut ctx = context(0);
    let inputs = [8192.0f32, -8193.0, 1.75, -1.75];
    let mut n = [0; 2];
    let mut bias = [0; 2];
    for (lane, &value) in inputs.iter().enumerate() {
        put(&mut n, lane, false, value.to_bits() as u64);
        put(
            &mut bias,
            lane,
            false,
            (if value < 0.0 { -0.5f32 } else { 0.5f32 }).to_bits() as u64,
        );
    }
    ctx.v[0] = n;
    ctx.v[1] = bias;
    ctx.v[5] = [0x45ff_f800_45ff_f800; 2];
    ctx.v[6] = [0xc600_0000_c600_0000; 2];
    let mut want = ctx.v;
    want[0] = [0xffff_e000_0000_1fff, 0xffff_fffe_0000_0002];
    let result = jit().run_bounded(&mut ctx, &mut code, 5).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 5);
    assert_eq!(ctx.v, want);
    assert_eq!(ctx.x, [GUARD; 31]);
    assert_eq!(ctx.sp, GUARD);
    assert_eq!(ctx.nzcv, 0xb);
    assert_eq!(ctx.fpsr, STICKY);
}

#[test]
fn overwritten_results_retain_ioc_idc_and_reserved_formats_reject() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        for minimum in [false, true] {
            for (n, fpcr, flag) in [
                (
                    if double {
                        0x7ff0_0000_0000_0123
                    } else {
                        0x7f80_0123
                    },
                    0,
                    1,
                ),
                (1, FZ, 1 << 7),
            ] {
                let mut code = Code(vec![word(minimum, double, q, 1, 2, 0), 0x1e27_03e0, BRK]);
                let mut ctx = context(fpcr);
                ctx.v[1] = [n, 0];
                ctx.v[2] = [0, 0];
                let result = jit().run_bounded(&mut ctx, &mut code, 3).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0));
                assert_eq!(ctx.v[0], [0, 0]);
                assert_eq!(ctx.fpsr, STICKY | flag);
            }
            for instruction in [
                word(minimum, true, false, 1, 2, 0),
                word(minimum, false, true, 1, 2, 0) & !0x0020_0000 | 0x0040_0000,
            ] {
                let mut code = Code(vec![instruction]);
                let mut ctx = context(DN | FZ);
                let result = jit().run_bounded(&mut ctx, &mut code, 1).unwrap();
                assert!(
                    matches!(result.stop, StopReason::Unsupported(_)),
                    "{instruction:08x}: {:?}",
                    result.stop
                );
                assert_eq!(result.retired, 0);
                assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
                assert_eq!(ctx.x, [GUARD; 31]);
                assert_eq!(ctx.fpsr, STICKY);
                assert_eq!(ctx.fpcr, DN | FZ);
            }
        }
    }
}

#[test]
fn helper_calls_preserve_live_integer_and_vector_values_under_pressure() {
    for (double, q) in [(false, false), (false, true), (true, true)] {
        for minimum in [false, true] {
            let mut words = Vec::new();
            for r in 0..16 {
                words.push(0x9100_0400 | (r << 5) | r);
                words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
            }
            words.push(word(minimum, double, q, 30, 31, 31));
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
            words.push(BRK);
            let count = words.len() as u64;
            let mut code = Code(words);
            let mut compiler = jit();
            for fpcr in [0, DN, FZ, DN | FZ] {
                let mut ctx = context(fpcr);
                for r in 0..31 {
                    ctx.x[r] = GUARD.wrapping_add(r as u64 * 137);
                }
                for r in 0..32 {
                    ctx.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
                }
                ctx.v[30] = vector(&classes(double), 16, double);
                ctx.v[31] = vector(&classes(double), 18, double);
                let mut want_x = ctx.x;
                let mut want_v = ctx.v;
                for r in 0..16 {
                    want_x[r] = want_x[r].wrapping_add(1);
                    want_v[r] = [want_v[r][0] | want_v[16][0], want_v[r][1] | want_v[16][1]];
                }
                let (answer, status) = expected(want_v[30], want_v[31], double, q, minimum, fpcr);
                want_v[31] = answer;
                for p in 0..8 {
                    want_x[if p == 7 { 16 } else { 24 + p }] =
                        want_x[p * 2].wrapping_add(want_x[p * 2 + 1]);
                }
                for r in 0..14 {
                    want_v[17 + r] = want_v[r];
                }
                let result = compiler.run_bounded(&mut ctx, &mut code, count).unwrap();
                assert_eq!(result.stop, StopReason::Brk(0));
                assert_eq!(result.retired, count);
                assert_eq!(ctx.v, want_v);
                assert_eq!(ctx.x, want_x);
                assert_eq!(ctx.fpcr, fpcr);
                assert_eq!(ctx.fpsr, STICKY | status);
                assert_eq!(ctx.nzcv, 0xb);
                assert_eq!(ctx.sp, GUARD);
            }
        }
    }
}
