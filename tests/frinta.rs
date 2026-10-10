use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
struct CodeMem([u32; 2]);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let offset = addr.checked_sub(CODE_BASE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get((offset / 4) as usize).copied()
    }
}

fn check_cases(opcode: u32, cases: impl IntoIterator<Item = (u64, u64)>) {
    let cases = cases.into_iter().collect::<Vec<_>>();
    for rn in [0usize, 1] {
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let mut memory = CodeMem([opcode | ((rn as u32) << 5), 0xd420_0000]);
        for &(input, expected) in &cases {
            let mut context = CpuContext::default();
            context.pc = CODE_BASE;
            context.nzcv = 0xa;
            context.x.fill(0xface_1234_5678_9abc);
            context
                .v
                .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
            context.v[rn] = [input, 0xdead_beef_feed_face];
            let original_v = context.v;
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v[0],
                [expected, 0],
                "opcode={opcode:#010x} input={input:#018x} rn={rn}"
            );
            assert_eq!(
                &context.v[1..],
                &original_v[1..],
                "FRINTA must preserve other vector registers"
            );
            assert_eq!(context.nzcv, 0xa);
            assert!(context.x.iter().all(|&x| x == 0xface_1234_5678_9abc));
        }
    }
}

#[test]
fn scalar_single_rounds_nearest_ties_away() {
    let inputs = [
        0.0,
        -0.0,
        0.25,
        -0.25,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.75,
        -3.75,
        f32::from_bits(0x3eff_ffff),
        f32::from_bits(0x3f00_0001),
        f32::from_bits(0xbeff_ffff),
        f32::from_bits(0xbf00_0001),
        f32::from_bits(0x3fbf_ffff),
        f32::from_bits(0x3fc0_0001),
        f32::from_bits(0xbfbf_ffff),
        f32::from_bits(0xbfc0_0001),
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        8_388_607.5,
        8_388_608.0,
        8_388_609.0,
        -8_388_609.0,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc1_2345),
        f32::from_bits(0x7f81_2345),
        f32::from_bits(0xff81_2345),
    ];
    check_cases(
        0x1e26_4000,
        inputs.into_iter().map(|v| {
            let expected = if v.is_nan() {
                v.to_bits() | 0x0040_0000
            } else {
                v.round().to_bits()
            };
            (v.to_bits() as u64, expected as u64)
        }),
    );
}

#[test]
fn scalar_double_rounds_nearest_ties_away() {
    let inputs = [
        0.0,
        -0.0,
        0.25,
        -0.25,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.75,
        -3.75,
        f64::from_bits(0x3fdf_ffff_ffff_ffff),
        f64::from_bits(0x3fe0_0000_0000_0001),
        f64::from_bits(0xbfdf_ffff_ffff_ffff),
        f64::from_bits(0xbfe0_0000_0000_0001),
        f64::from_bits(0x3ff7_ffff_ffff_ffff),
        f64::from_bits(0x3ff8_0000_0000_0001),
        f64::from_bits(0xbff7_ffff_ffff_ffff),
        f64::from_bits(0xbff8_0000_0000_0001),
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        4_503_599_627_370_495.5,
        4_503_599_627_370_496.0,
        4_503_599_627_370_497.0,
        -4_503_599_627_370_497.0,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_1234_5678_9abc),
        f64::from_bits(0x7ff0_1234_5678_9abc),
        f64::from_bits(0xfff0_1234_5678_9abc),
    ];
    check_cases(
        0x1e66_4000,
        inputs.into_iter().map(|v| {
            let expected = if v.is_nan() {
                v.to_bits() | 0x0008_0000_0000_0000
            } else {
                v.round().to_bits()
            };
            (v.to_bits(), expected)
        }),
    );
}

fn check_vectors(opcode: u32, cases: &[([u64; 2], [u64; 2])]) {
    for rn in [0usize, 1] {
        let mut jit = Jit::new(JitConfig::default()).unwrap();
        let mut memory = CodeMem([opcode | ((rn as u32) << 5), 0xd420_0000]);
        for &(input, expected) in cases {
            let mut context = CpuContext::default();
            context.pc = CODE_BASE;
            context.nzcv = 0xa;
            context
                .v
                .fill([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]);
            context.v[rn] = input;
            let original = context.v;
            assert_eq!(
                jit.run(&mut context, &mut memory).unwrap(),
                ExitReason::Brk(0)
            );
            assert_eq!(
                context.v[0], expected,
                "opcode={opcode:#010x} input={input:016x?}"
            );
            assert_eq!(&context.v[1..], &original[1..]);
            assert_eq!(context.nzcv, 0xa);
        }
    }
}

#[test]
fn vector_single_rounding_preserves_tie_neighbors_and_integral_values() {
    let pack = |values: [u32; 4]| {
        [
            values[0] as u64 | ((values[1] as u64) << 32),
            values[2] as u64 | ((values[3] as u64) << 32),
        ]
    };
    let cases = [
        (
            pack([0x3eff_ffff, 0xbeff_ffff, 0x3f00_0000, 0xbf00_0000]),
            pack([0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000]),
        ),
        (
            pack([0x3fbf_ffff, 0xbfbf_ffff, 0x3fc0_0000, 0xbfc0_0000]),
            pack([0x3f80_0000, 0xbf80_0000, 0x4000_0000, 0xc000_0000]),
        ),
        (
            pack([0x4aff_ffff, 0xcaff_ffff, 0x4b00_0001, 0xcb00_0001]),
            pack([0x4b00_0000, 0xcb00_0000, 0x4b00_0001, 0xcb00_0001]),
        ),
        (
            pack([1, 0x8000_0001, 0, 0x8000_0000]),
            pack([0, 0x8000_0000, 0, 0x8000_0000]),
        ),
        (
            pack([0x7f80_0000, 0xff80_0000, 0x7f81_2345, 0xffc1_2345]),
            pack([0x7f80_0000, 0xff80_0000, 0x7fc1_2345, 0xffc1_2345]),
        ),
    ];
    check_vectors(0x6e21_8800, &cases);
    let half = cases.map(|(input, result)| (input, [result[0], 0]));
    check_vectors(0x2e21_8800, &half);
}

#[test]
fn vector_double_rounding_preserves_tie_neighbors_and_integral_values() {
    let cases = [
        (
            [0x3fdf_ffff_ffff_ffff, 0xbfdf_ffff_ffff_ffff],
            [0, 0x8000_0000_0000_0000],
        ),
        (
            [0x3ff7_ffff_ffff_ffff, 0xbff7_ffff_ffff_ffff],
            [0x3ff0_0000_0000_0000, 0xbff0_0000_0000_0000],
        ),
        (
            [0x3ff8_0000_0000_0000, 0xbff8_0000_0000_0000],
            [0x4000_0000_0000_0000, 0xc000_0000_0000_0000],
        ),
        (
            [0x432f_ffff_ffff_ffff, 0xc32f_ffff_ffff_ffff],
            [0x4330_0000_0000_0000, 0xc330_0000_0000_0000],
        ),
        (
            [0x4330_0000_0000_0001, 0xc330_0000_0000_0001],
            [0x4330_0000_0000_0001, 0xc330_0000_0000_0001],
        ),
        ([1, 0x8000_0000_0000_0001], [0, 0x8000_0000_0000_0000]),
        (
            [0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000],
            [0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000],
        ),
        (
            [0x7ff0_1234_5678_9abc, 0xfff8_1234_5678_9abc],
            [0x7ff8_1234_5678_9abc, 0xfff8_1234_5678_9abc],
        ),
    ];
    check_vectors(0x6e61_8800, &cases);
}
