use rustarmic::{CpuContext, CpuFeatures, ExitReason, Jit, JitConfig, Memory};

const CODE: u64 = 0x1000;
const CANARY: u64 = 0xface_1234_5678_9abc;

struct CodeMem(Vec<u32>);
impl Memory for CodeMem {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

struct Case {
    name: &'static str,
    n: u64,
    m: u64,
    a: [u64; 4],
    result: [u64; 4],
}

fn cases(double: bool) -> Vec<Case> {
    let (one, two, three, five, eleven, plus, minus, residue, sign, max, qnan, snan, inf) =
        if double {
            (
                0x3ff0_0000_0000_0000,
                0x4000_0000_0000_0000,
                0x4008_0000_0000_0000,
                0x4014_0000_0000_0000,
                0x4026_0000_0000_0000,
                0x3ff0_0000_0000_0001,
                0x3fef_ffff_ffff_fffe,
                0x3970_0000_0000_0000,
                0x8000_0000_0000_0000,
                0x7fef_ffff_ffff_ffff,
                0x7ff8_1234_5678_9abc,
                0x7ff0_1234_5678_9abc,
                0x7ff0_0000_0000_0000,
            )
        } else {
            (
                0x3f80_0000,
                0x4000_0000,
                0x4040_0000,
                0x40a0_0000,
                0x4130_0000,
                0x3f80_0001,
                0x3f7f_fffe,
                0x2880_0000,
                0x8000_0000,
                0x7f7f_ffff,
                0x7fc1_2345,
                0x7f81_2345,
                0x7f80_0000,
            )
        };
    let quiet = qnan ^ snan;
    let default_nan = inf | quiet;
    vec![
        Case {
            name: "fused cancellation",
            n: plus,
            m: minus,
            a: [one | sign, one, one | sign, one],
            result: [residue | sign, residue, residue, residue | sign],
        },
        Case {
            name: "product overflow rescued",
            n: max,
            m: two,
            a: [max | sign, max, max | sign, max],
            result: [max, max | sign, max | sign, max],
        },
        Case {
            name: "ordinary",
            n: two,
            m: three,
            a: [five; 4],
            result: [eleven, one | sign, eleven | sign, one],
        },
        Case {
            name: "exact cancellation",
            n: one,
            m: one,
            a: [one | sign, one, one | sign, one],
            result: [0; 4],
        },
        Case {
            name: "positive zero",
            n: 0,
            m: one,
            a: [0; 4],
            result: [0, 0, sign, 0],
        },
        Case {
            name: "negative zero",
            n: sign,
            m: one,
            a: [sign; 4],
            result: [sign, 0, 0, 0],
        },
        Case {
            name: "quiet NaN N",
            n: qnan,
            m: one,
            a: [one; 4],
            result: [qnan, qnan | sign, qnan | sign, qnan],
        },
        Case {
            name: "quiet NaN M",
            n: one,
            m: qnan,
            a: [one; 4],
            result: [qnan; 4],
        },
        Case {
            name: "quiet NaN A",
            n: one,
            m: one,
            a: [qnan; 4],
            result: [qnan, qnan, qnan | sign, qnan | sign],
        },
        Case {
            name: "signaling NaN N",
            n: snan,
            m: one,
            a: [one; 4],
            result: [qnan, qnan | sign, qnan | sign, qnan],
        },
        Case {
            name: "signaling NaN M",
            n: one,
            m: snan,
            a: [one; 4],
            result: [qnan; 4],
        },
        Case {
            name: "signaling NaN A",
            n: one,
            m: one,
            a: [snan; 4],
            result: [qnan, qnan, qnan | sign, qnan | sign],
        },
        Case {
            name: "quiet priority A N M",
            n: qnan ^ 1,
            m: qnan ^ 2,
            a: [qnan ^ 3; 4],
            result: [qnan ^ 3, qnan ^ 3, (qnan ^ 3) | sign, (qnan ^ 3) | sign],
        },
        Case {
            name: "signaling N precedes quiet A",
            n: snan ^ 1,
            m: snan ^ 2,
            a: [qnan ^ 3; 4],
            result: [qnan ^ 1, (qnan ^ 1) | sign, (qnan ^ 1) | sign, qnan ^ 1],
        },
        Case {
            name: "signaling A precedes signaling N",
            n: snan ^ 1,
            m: snan ^ 2,
            a: [snan ^ 3; 4],
            result: [qnan ^ 3, qnan ^ 3, (qnan ^ 3) | sign, (qnan ^ 3) | sign],
        },
        Case {
            name: "invalid product overrides quiet A",
            n: 0,
            m: inf,
            a: [qnan; 4],
            result: [default_nan; 4],
        },
        Case {
            name: "signaling A precedes invalid product",
            n: 0,
            m: inf,
            a: [snan; 4],
            result: [qnan, qnan, qnan | sign, qnan | sign],
        },
        Case {
            name: "invalid product default NaN",
            n: inf,
            m: 0,
            a: [one; 4],
            result: [default_nan; 4],
        },
        Case {
            name: "invalid sum default NaN",
            n: inf,
            m: one,
            a: [inf | sign, inf, inf | sign, inf],
            result: [default_nan; 4],
        },
        Case {
            name: "infinite product",
            n: inf,
            m: one,
            a: [one; 4],
            result: [inf, inf | sign, inf | sign, inf],
        },
        Case {
            name: "subnormal result",
            n: 1,
            m: one,
            a: [0, 0, sign, sign],
            result: [1, 1 | sign, 1 | sign, 1],
        },
    ]
}

fn encoding(kind: u32, double: bool, rd: usize) -> u32 {
    let variant = match kind {
        0 => 0,
        1 => 1 << 15,
        2 => 1 << 21,
        _ => (1 << 21) | (1 << 15),
    };
    0x1f02_0c20 | (u32::from(double) << 22) | variant | rd as u32
}

fn check(double: bool, features: CpuFeatures) {
    for kind in 0..4 {
        for rd in [0, 1, 2, 3, 31] {
            let mut jit = Jit::new(JitConfig {
                host_features: Some(features),
                ..JitConfig::default()
            })
            .unwrap();
            let opcode = encoding(kind, double, rd);
            let mut memory = CodeMem(vec![opcode, 0xd420_0000]);
            for case in cases(double) {
                let mut context = CpuContext::default();
                context.pc = CODE;
                context.x.fill(CANARY);
                context.sp = CANARY;
                context.nzcv = 0xa;
                context.v.fill([CANARY, CANARY]);
                for (reg, input) in [case.n, case.m, case.a[kind as usize]]
                    .into_iter()
                    .enumerate()
                {
                    context.v[reg + 1] = [
                        input | if double { 0 } else { 0xdead_beef_0000_0000 },
                        CANARY,
                    ];
                }
                let mut expected_vectors = context.v;
                expected_vectors[rd] = [case.result[kind as usize], 0];
                assert_eq!(
                    jit.run(&mut context, &mut memory).unwrap(),
                    ExitReason::Brk(0)
                );
                assert_eq!(
                    context.v, expected_vectors,
                    "{} opcode={opcode:#010x} AVX={} FMA={}",
                    case.name, features.has_avx, features.has_fma
                );
                assert!(context.x.iter().all(|&value| value == CANARY));
                assert_eq!(context.sp, CANARY);
                assert_eq!(context.nzcv, 0xa);
            }
        }
    }
}

fn host_features() -> CpuFeatures {
    Jit::new(JitConfig::default()).unwrap().host_features
}

#[test]
fn single_fused_hardware() {
    let features = host_features();
    if features.has_avx && features.has_fma {
        check(false, features);
    }
}
#[test]
fn double_fused_hardware() {
    let features = host_features();
    if features.has_avx && features.has_fma {
        check(true, features);
    }
}
#[test]
fn single_fused_baseline_without_avx_or_fma() {
    let mut features = host_features();
    features.has_avx = false;
    features.has_fma = false;
    check(false, features);
}
#[test]
fn double_fused_baseline_without_avx_or_fma() {
    let mut features = host_features();
    features.has_avx = false;
    features.has_fma = false;
    check(true, features);
}

#[test]
fn avx_without_fma_uses_fused_helper() {
    let mut features = host_features();
    features.has_fma = false;
    check(false, features);
    check(true, features);
}

#[test]
fn helper_call_preserves_values_live_across_fma() {
    for hardware in [false, true] {
        let mut features = host_features();
        if hardware && !(features.has_avx && features.has_fma) {
            continue;
        }
        if !hardware {
            features.has_avx = false;
            features.has_fma = false;
        }
        // Keep twelve integer values live in GPR/XMM/stack locations across
        // the FMA call. The NaN forces the helper even on an FMA host.
        let mut words = Vec::new();
        for reg in 4..16 {
            words.push(0x9100_0000 | (reg << 5) | (reg + 12));
        } // ADD Xd,Xn,#0
        words.push(encoding(0, true, 0));
        for reg in 4..16 {
            words.push(0x9100_0000 | ((reg + 12) << 5) | reg);
        }
        words.push(0xd420_0000);
        let mut memory = CodeMem(words);
        let mut context = CpuContext::default();
        context.pc = CODE;
        for reg in 0..31 {
            context.x[reg] = CANARY.wrapping_add(reg as u64);
        }
        let original = context.x;
        context.v[1] = [0x7ff8_1234_5678_9abc, CANARY];
        context.v[2] = [0x3ff0_0000_0000_0000, CANARY];
        context.v[3] = [0x3ff0_0000_0000_0000, CANARY];
        let mut jit = Jit::new(JitConfig {
            host_features: Some(features),
            ..JitConfig::default()
        })
        .unwrap();
        assert_eq!(
            jit.run(&mut context, &mut memory).unwrap(),
            ExitReason::Brk(0)
        );
        for reg in 4..16 {
            assert_eq!(context.x[reg], original[reg]);
            assert_eq!(context.x[reg + 12], original[reg]);
        }
        assert_eq!(context.v[0], [0x7ff8_1234_5678_9abc, 0]);
    }
}
