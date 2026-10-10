use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xface_1234_5678_9abc;
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

#[derive(Clone, Copy)]
struct Form {
    scalar: bool,
    double: bool,
    q: bool,
}
impl Form {
    fn lanes(self) -> usize {
        if self.scalar {
            1
        } else if self.double || !self.q {
            2
        } else {
            4
        }
    }
    fn opcode(self, rn: u32, rm: u32, rd: u32) -> u32 {
        (if self.scalar {
            0x7ea0_d400
        } else {
            0x2ea0_d400
        }) | (u32::from(self.double) << 22)
            | (u32::from(!self.scalar && self.q) << 30)
            | (rm << 16)
            | (rn << 5)
            | rd
    }
}
const FORMS: [Form; 5] = [
    Form {
        scalar: true,
        double: false,
        q: false,
    },
    Form {
        scalar: true,
        double: true,
        q: false,
    },
    Form {
        scalar: false,
        double: false,
        q: false,
    },
    Form {
        scalar: false,
        double: false,
        q: true,
    },
    Form {
        scalar: false,
        double: true,
        q: true,
    },
];

fn oracle(n: u64, m: u64, double: bool) -> u64 {
    let (sign, infinity, quiet) = if double {
        (1 << 63, 0x7ff0_0000_0000_0000, 1 << 51)
    } else {
        (1 << 31, 0x7f80_0000, 1 << 22)
    };
    let nan = |v: u64| v & !sign > infinity;
    // Arm FPSub uses original N/M: signaling first, N before M. Only the
    // result's sign is cleared by FPAbs; M is not negated before NaN selection.
    for value in [n, m] {
        if nan(value) && value & quiet == 0 {
            return (value | quiet) & !sign;
        }
    }
    for value in [n, m] {
        if nan(value) {
            return value & !sign;
        }
    }
    if n & !sign == infinity && m == n {
        return infinity | quiet;
    }
    if double {
        (f64::from_bits(n) - f64::from_bits(m)).to_bits() & !sign
    } else {
        (f32::from_bits(n as u32) - f32::from_bits(m as u32)).to_bits() as u64 & !sign
    }
}

fn pairs(double: bool) -> Vec<(u64, u64)> {
    let (sign, infinity, one, half, one_ulp, max) = if double {
        (
            1 << 63,
            0x7ff0_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0x3fe0_0000_0000_0000,
            0x3cb0_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
        )
    } else {
        (
            1 << 31,
            0x7f80_0000,
            0x3f80_0000,
            0x3f00_0000,
            0x3400_0000,
            0x7f7f_ffff,
        )
    };
    vec![
        (0, 0),
        (sign, 0),
        (0, sign),
        (sign, sign),
        (one, half),
        (half, one),
        (one, one),
        (one, one | sign),
        (one + 1, one),
        (one, one + 1),
        (one, one_ulp | sign),
        (one, one_ulp - 1),
        (1, 2),
        (2, 1),
        (1, sign | 1),
        (max, max),
        (max, max | sign),
        (infinity, one),
        (one, infinity),
        (infinity, infinity),
        (infinity | sign, infinity | sign),
        (infinity, infinity | sign),
        (infinity | sign, infinity),
    ]
}

fn vector(values: &[(u64, u64)], start: usize, form: Form, second: bool) -> [u64; 2] {
    let mut bytes = [0xc7; 16];
    let width = if form.double { 8 } else { 4 };
    for lane in 0..form.lanes() {
        let pair = values[(start + lane) % values.len()];
        let value = if second { pair.1 } else { pair.0 };
        bytes[lane * width..(lane + 1) * width].copy_from_slice(&value.to_le_bytes()[..width]);
    }
    [
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ]
}

fn get(source: [u64; 2], lane: usize, double: bool) -> u64 {
    if double {
        source[lane]
    } else {
        (source[lane / 2] >> ((lane % 2) * 32)) as u32 as u64
    }
}

fn run_matrix(specials: bool) {
    for form in FORMS {
        let mut values = pairs(form.double);
        if specials {
            let (sign, infinity, quiet) = if form.double {
                (1 << 63, 0x7ff0_0000_0000_0000, 1 << 51)
            } else {
                (1 << 31, 0x7f80_0000, 1 << 22)
            };
            values = vec![
                (infinity | quiet | 0x123, 0),
                (0, infinity | quiet | 0x456),
                (sign | infinity | 0x123, 0),
                (0, sign | infinity | 0x456),
                (infinity | quiet | 0x123, infinity | quiet | 0x456),
                (infinity | 0x123, infinity | 0x456),
                (infinity | 0x123, infinity | quiet | 0x456),
                (infinity | quiet | 0x123, infinity | 0x456),
            ];
        }
        for (rn, rm, rd) in [
            (1, 2, 0),
            (1, 2, 1),
            (1, 2, 2),
            (1, 1, 1),
            (31, 0, 31),
            (0, 31, 0),
            (31, 31, 2),
        ] {
            let instruction = form.opcode(rn, rm, rd);
            let mut code = Code(vec![instruction, BRK]);
            let mut jit = Jit::new(JitConfig::default()).unwrap();
            for sample in 0..values.len() {
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.sp = GUARD;
                ctx.x.fill(GUARD);
                ctx.nzcv = 0xb;
                ctx.fpsr = 1 << 27;
                ctx.v.fill([GUARD, !GUARD]);
                ctx.v[rn as usize] = vector(&values, sample, form, false);
                if rm != rn {
                    ctx.v[rm as usize] = vector(&values, sample, form, true);
                }
                let mut wanted = ctx.v;
                let mut result = [0; 2];
                for lane in 0..form.lanes() {
                    let answer = oracle(
                        get(ctx.v[rn as usize], lane, form.double),
                        get(ctx.v[rm as usize], lane, form.double),
                        form.double,
                    );
                    if form.double {
                        result[lane] = answer;
                    } else {
                        result[lane / 2] |= answer << ((lane % 2) * 32);
                    }
                }
                wanted[rd as usize] = result;
                let execution = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                assert_eq!(execution.stop, StopReason::Brk(0));
                assert_eq!(
                    ctx.v, wanted,
                    "opcode={instruction:#x} sample={sample} specials={specials}"
                );
                assert_eq!(ctx.x, [GUARD; 31]);
                assert_eq!((ctx.sp, ctx.nzcv, ctx.fpcr), (GUARD, 0xb, 0));
                // Generic FSUB exception flags are not this frontend's contract.
                assert_eq!(ctx.fpsr & (1 << 27), 1 << 27, "unrelated sticky QC");
            }
        }
    }
}

#[test]
fn target_precision_finite_zero_infinity_all_formats_and_aliases() {
    run_matrix(false);
}

#[test]
fn architectural_nan_selection_original_n_m_then_abs() {
    run_matrix(true);
}

#[test]
fn exact_wonder_word_and_target_rounding_literals() {
    assert_eq!(FORMS[2].opcode(1, 2, 1), 0x2ea2_d421);
    assert_eq!(oracle(0x3f80_0001, 0x3f80_0000, false), 0x3400_0000);
    assert_eq!(
        oracle(0x3ff0_0000_0000_0001, 0x3ff0_0000_0000_0000, true),
        0x3cb0_0000_0000_0000
    );
    let mut code = Code(vec![0x2ea2_d421, BRK]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v[1] = [0x3f80_0000_c000_0000, GUARD]; // -2,1.
    ctx.v[2] = [0xbf80_0000_3f00_0000, !GUARD]; // .5,-1.
    let source = ctx.v[2];
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    assert_eq!(
        jit.run_bounded(&mut ctx, &mut code, 2).unwrap().stop,
        StopReason::Brk(0)
    );
    assert_eq!(ctx.v[1], [0x4000_0000_4020_0000, 0]); // 2.5,2.
    assert_eq!(ctx.v[2], source);
}

#[test]
fn reserved_vector_1d_and_out_of_scope_half_reject() {
    for instruction in [0x2ee2_d421, 0x2ec2_1421, 0x7ec2_1421] {
        let mut code = Code(vec![instruction]);
        let mut ctx = CpuContext::default();
        ctx.pc = CODE;
        ctx.v.fill([GUARD, !GUARD]);
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let result = jit.run_bounded(&mut ctx, &mut code, 1).unwrap();
        assert!(matches!(result.stop,StopReason::Unsupported(info) if info.opcode == instruction));
        assert_eq!(result.retired, 0);
        assert_eq!(ctx.v, [[GUARD, !GUARD]; 32]);
    }
}

#[test]
fn scoped_nan_invalid_ioc_dn_and_overwritten_result_keep_sticky_flags() {
    for form in FORMS {
        let (infinity, quiet) = if form.double {
            (0x7ff0_0000_0000_0000, 1 << 51)
        } else {
            (0x7f80_0000, 1 << 22)
        };
        for (n, m, ioc) in [
            (infinity | quiet | 0x123, infinity | 0x456, true),
            (infinity | quiet | 0x123, infinity | quiet | 0x456, false),
            (infinity, infinity, true),
        ] {
            for dn in [false, true] {
                let mut code = Code(vec![form.opcode(1, 2, 0), BRK]);
                let mut ctx = CpuContext::default();
                ctx.pc = CODE;
                ctx.fpcr = u32::from(dn) << 25;
                ctx.fpsr = (1 << 27) | 2;
                ctx.v[1] = vector(&[(n, m)], 0, form, false);
                ctx.v[2] = vector(&[(n, m)], 0, form, true);
                let mut expected = [0; 2];
                let answer = if dn {
                    infinity | quiet
                } else {
                    oracle(n, m, form.double)
                };
                for lane in 0..form.lanes() {
                    if form.double {
                        expected[lane] = answer;
                    } else {
                        expected[lane / 2] |= answer << ((lane % 2) * 32);
                    }
                }
                let mut jit = Jit::new(JitConfig::default()).unwrap();
                assert_eq!(
                    jit.run_bounded(&mut ctx, &mut code, 2).unwrap().stop,
                    StopReason::Brk(0)
                );
                assert_eq!(ctx.v[0], expected);
                assert_eq!(ctx.fpsr, (1 << 27) | 2 | u32::from(ioc));
                assert_eq!(ctx.fpcr, u32::from(dn) << 25);
            }
        }
    }
    // The result is dead, but invalid FPSub must still raise sticky IOC.
    let mut code = Code(vec![FORMS[0].opcode(1, 2, 0), 0x1e27_03e0, BRK]);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.v[1] = [0x7f80_0000, GUARD];
    ctx.v[2] = [0x7f80_0000, !GUARD];
    ctx.fpsr = 1 << 27;
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    assert_eq!(
        jit.run_bounded(&mut ctx, &mut code, 3).unwrap().stop,
        StopReason::Brk(0)
    );
    assert_eq!(ctx.v[0], [0, 0]);
    assert_eq!(ctx.fpsr, (1 << 27) | 1);
}
