use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};
const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
const FLAGS: u32 = 0x0800_009f;
const BRK: u32 = 0xd420_0000;
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
// kind0: SHLL; kind1: scalar ADDP; kind2/3: scalar ADD/SUB.
fn word(kind: u32, size: u32, upper: bool, rn: u32, rm: u32, rd: u32) -> u32 {
    match kind {
        0 => 0x2e21_3800 | (size << 22) | (u32::from(upper) << 30) | (rn << 5) | rd,
        1 => 0x5ef1_b800 | (rn << 5) | rd,
        _ => (if kind == 2 { 0x5ee0_8400 } else { 0x7ee0_8400 }) | (rm << 16) | (rn << 5) | rd,
    }
}
fn bytes(v: [u64; 2]) -> [u8; 16] {
    let mut a = [0; 16];
    a[..8].copy_from_slice(&v[0].to_le_bytes());
    a[8..].copy_from_slice(&v[1].to_le_bytes());
    a
}
fn vector(v: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(v[..8].try_into().unwrap()),
        u64::from_le_bytes(v[8..].try_into().unwrap()),
    ]
}
fn expected(kind: u32, size: u32, upper: bool, n: [u64; 2], m: [u64; 2]) -> [u64; 2] {
    if kind != 0 {
        return [
            match kind {
                1 => n[0].wrapping_add(n[1]),
                2 => n[0].wrapping_add(m[0]),
                _ => n[0].wrapping_sub(m[0]),
            },
            0,
        ];
    }
    let source = bytes(n);
    let count = 1usize << size;
    let shift = count * 8;
    let mut output = [0; 16];
    for lane in 0..8 / count {
        let offset = usize::from(upper) * 8 + lane * count;
        let mut raw = [0; 8];
        raw[..count].copy_from_slice(&source[offset..offset + count]);
        let value = (u64::from_le_bytes(raw) as u128) * (1u128 << shift);
        output[lane * count * 2..(lane + 1) * count * 2]
            .copy_from_slice(&(value as u64).to_le_bytes()[..count * 2]);
    }
    vector(output)
}
fn context() -> CpuContext {
    let mut c = CpuContext::default();
    c.pc = CODE;
    c.x.fill(GUARD);
    c.v.fill([GUARD, !GUARD]);
    c.sp = GUARD;
    c.nzcv = 0xb;
    c.fpcr = 0x03c0_0000;
    c.fpsr = FLAGS;
    c
}
fn run(
    compiler: &mut Jit,
    code: &mut Code,
    kind: u32,
    size: u32,
    upper: bool,
    rn: u32,
    rm: u32,
    rd: u32,
    n: [u64; 2],
    m: [u64; 2],
) {
    let mut c = context();
    c.v[rn as usize] = n;
    if kind >= 2 {
        c.v[rm as usize] = m;
    }
    let mut want = c.v;
    want[rd as usize] = expected(kind, size, upper, want[rn as usize], want[rm as usize]);
    let result = compiler.run_bounded(&mut c, code, 2).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 2);
    assert_eq!(c.v, want, "word{:08x} n{n:x?} m{m:x?}", code.0[0]);
    assert_eq!(c.x, [GUARD; 31]);
    assert_eq!(c.sp, GUARD);
    assert_eq!(c.nzcv, 0xb);
    assert_eq!(c.fpcr, 0x03c0_0000);
    assert_eq!(c.fpsr, FLAGS);
}
fn pattern(size: u32, rotation: usize) -> [u64; 2] {
    let count = 1usize << size;
    let shift = count * 8;
    let sign = 1u64 << (shift - 1);
    let mask = (1u64 << shift) - 1;
    let values = [
        0,
        1,
        mask,
        sign,
        sign - 1,
        sign + 1,
        0xaaaa_5555 & mask,
        0x0123_4567 & mask,
    ];
    let mut v = [0; 16];
    for lane in 0..16 / count {
        v[lane * count..(lane + 1) * count].copy_from_slice(
            &values[(lane + rotation + if lane >= 8 / count { 3 } else { 0 }) % 8].to_le_bytes()
                [..count],
        );
    }
    vector(v)
}
#[test]
fn widening_every_width_half_alias_and_boundary_random_input() {
    for size in 0..3 {
        let mut inputs = (0..8).map(|r| pattern(size, r)).collect::<Vec<_>>();
        let mut random = 0x91fa_a482_e613_820u64;
        for _ in 0..128 {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let n = random;
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            inputs.push([n, random]);
        }
        for upper in [false, true] {
            for (rn, rd) in [(0, 1), (0, 0), (31, 31), (31, 2), (2, 31)] {
                let mut code = Code(vec![word(0, size, upper, rn, 0, rd), BRK]);
                let mut compiler = jit();
                for &n in &inputs {
                    run(
                        &mut compiler,
                        &mut code,
                        0,
                        size,
                        upper,
                        rn,
                        0,
                        rd,
                        n,
                        [0; 2],
                    );
                }
            }
        }
    }
}
#[test]
fn every_byte_and_halfword_value_widens_in_every_physical_lane_and_half() {
    for size in 0..2 {
        let count = 1usize << size;
        let limit = 1u32 << (count * 8);
        for upper in [false, true] {
            let mut code = Code(vec![word(0, size, upper, 31, 0, 31), BRK]);
            let mut compiler = jit();
            for value in 0..limit {
                let mut n = [0; 16];
                for lane in 0..16 / count {
                    let v = value.wrapping_add(lane as u32 * 37);
                    n[lane * count..(lane + 1) * count].copy_from_slice(&v.to_le_bytes()[..count]);
                }
                run(
                    &mut compiler,
                    &mut code,
                    0,
                    size,
                    upper,
                    31,
                    0,
                    31,
                    vector(n),
                    [0; 2],
                );
            }
        }
    }
}
#[test]
fn scalar_pair_and_binary_all_aliases_overflow_signed_boundaries_and_poisoned_upper() {
    let values = [
        0,
        1,
        2,
        3,
        1 << 31,
        (1 << 32) - 1,
        1 << 32,
        (1u64 << 63) - 1,
        1u64 << 63,
        (1u64 << 63) + 1,
        u64::MAX - 1,
        u64::MAX,
        GUARD,
    ];
    for kind in 1..4 {
        for (rn, rm, rd) in [
            (0, 1, 2),
            (0, 1, 0),
            (0, 1, 1),
            (0, 0, 2),
            (0, 0, 0),
            (31, 31, 31),
            (31, 0, 31),
            (0, 31, 31),
        ] {
            let mut code = Code(vec![word(kind, 3, false, rn, rm, rd), BRK]);
            let mut compiler = jit();
            for &a in &values {
                for &b in &values {
                    run(
                        &mut compiler,
                        &mut code,
                        kind,
                        3,
                        false,
                        rn,
                        rm,
                        rd,
                        [a, b],
                        [b, !a],
                    );
                }
            }
        }
    }
}
#[test]
fn exact_observed_wonder_words_have_independent_hand_literals() {
    assert_eq!(word(0, 1, false, 1, 0, 0), 0x2e61_3820);
    assert_eq!(word(1, 3, false, 0, 0, 4), 0x5ef1_b804);
    assert_eq!(word(3, 3, false, 0, 1, 3), 0x7ee1_8403);
    assert_eq!(
        expected(0, 1, false, [0xffff_8000_0001_0000, GUARD], [0; 2]),
        [0x0001_0000_0000_0000, 0xffff_0000_8000_0000]
    );
    assert_eq!(expected(1, 3, false, [u64::MAX, 2], [0; 2]), [1, 0]);
    assert_eq!(
        expected(3, 3, false, [0, GUARD], [1, !GUARD]),
        [u64::MAX, 0]
    );
    let mut code = Code(vec![0x2e61_3820, 0x5ef1_b804, 0x7ee1_8403, BRK]);
    let mut c = context();
    c.v[1] = [0xffff_8000_0001_0000, GUARD];
    let mut want = c.v;
    want[0] = [0x0001_0000_0000_0000, 0xffff_0000_8000_0000];
    want[4] = [0x0000_0000_8000_0000, 0];
    want[3] = [0x0001_0000_0000_0000u64.wrapping_sub(want[1][0]), 0];
    let result = jit().run_bounded(&mut c, &mut code, 4).unwrap();
    assert_eq!(result.stop, StopReason::Brk(0));
    assert_eq!(result.retired, 4);
    assert_eq!(c.v, want);
    assert_eq!(c.nzcv, 0xb);
    assert_eq!(c.fpsr, FLAGS);
}
#[test]
fn reserved_widen_d_and_scalar_nond_widths_do_not_retire_or_mutate() {
    let mut words = vec![word(0, 3, false, 31, 0, 31), word(0, 3, true, 31, 0, 31)];
    for kind in 1..4 {
        for size in 0..3 {
            words.push((word(kind, 3, false, 31, 31, 31) & !(3 << 22)) | (size << 22));
        }
    }
    for word in words {
        let mut code = Code(vec![word]);
        let mut c = context();
        let result = jit().run_bounded(&mut c, &mut code, 1).unwrap();
        assert!(
            matches!(result.stop, StopReason::Unsupported(_)),
            "{word:x}: {:?}",
            result.stop
        );
        assert_eq!(result.retired, 0);
        assert_eq!(c.v, [[GUARD, !GUARD]; 32]);
        assert_eq!(c.x, [GUARD; 31]);
        assert_eq!(c.sp, GUARD);
        assert_eq!(c.nzcv, 0xb);
        assert_eq!(c.fpcr, 0x03c0_0000);
        assert_eq!(c.fpsr, FLAGS);
    }
}
#[test]
fn widening_and_scalar_operations_preserve_live_integer_and_simd_pressure_values() {
    let operations = [
        word(0, 0, false, 30, 0, 31),
        word(0, 1, true, 30, 0, 31),
        word(0, 2, true, 30, 0, 31),
        word(1, 3, false, 31, 0, 31),
        word(2, 3, false, 30, 31, 31),
        word(3, 3, false, 30, 31, 31),
    ];
    for (index, operation) in operations.into_iter().enumerate() {
        let mut words = Vec::new();
        for r in 0..16 {
            words.push(0x9100_0400 | (r << 5) | r);
            words.push(0x4ea0_1c00 | (16 << 16) | (r << 5) | r);
        }
        words.push(operation);
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
        let mut c = context();
        for r in 0..31 {
            c.x[r] = GUARD.wrapping_add(r as u64 * 137);
        }
        for r in 0..32 {
            c.v[r] = [GUARD.wrapping_add(r as u64), !GUARD ^ r as u64];
        }
        let mut want_x = c.x;
        let mut want_v = c.v;
        for r in 0..16 {
            want_x[r] = want_x[r].wrapping_add(1);
            want_v[r] = [want_v[r][0] | want_v[16][0], want_v[r][1] | want_v[16][1]];
        }
        want_v[31] = match index {
            0 => expected(0, 0, false, want_v[30], want_v[31]),
            1 => expected(0, 1, true, want_v[30], want_v[31]),
            2 => expected(0, 2, true, want_v[30], want_v[31]),
            3 => expected(1, 3, false, want_v[31], want_v[31]),
            4 => expected(2, 3, false, want_v[30], want_v[31]),
            _ => expected(3, 3, false, want_v[30], want_v[31]),
        };
        for p in 0..8 {
            want_x[if p == 7 { 16 } else { 24 + p }] =
                want_x[p * 2].wrapping_add(want_x[p * 2 + 1]);
        }
        for r in 0..14 {
            want_v[17 + r] = want_v[r];
        }
        let result = jit().run_bounded(&mut c, &mut code, count).unwrap();
        assert_eq!(result.stop, StopReason::Brk(0));
        assert_eq!(result.retired, count);
        assert_eq!(c.v, want_v);
        assert_eq!(c.x, want_x);
        assert_eq!(c.sp, GUARD);
        assert_eq!(c.nzcv, 0xb);
        assert_eq!(c.fpcr, 0x03c0_0000);
        assert_eq!(c.fpsr, FLAGS);
    }
}
