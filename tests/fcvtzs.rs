use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
const BRK_0: u32 = 0xd420_0000;

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
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut memory = CodeMem([opcode, BRK_0]);
    for (input, expected) in cases {
        let mut context = CpuContext::default();
        context.pc = CODE_BASE;
        context.x.fill(0xface_1234_5678_9abc);
        context.v[1] = [input, 0xfedc_ba98_7654_3210];
        context.nzcv = 0b1010;
        let source = context.v[1];
        let exit = jit.run(&mut context, &mut memory).unwrap();
        assert_eq!(exit, ExitReason::Brk(0));
        assert_eq!(
            context.x[0], expected,
            "opcode={opcode:#010x}, input={input:#018x}"
        );
        assert_eq!(context.v[1], source, "FCVTZS must preserve its source");
        assert_eq!(context.nzcv, 0b1010, "FCVTZS must preserve NZCV");
        assert!(context.x[1..].iter().all(|&x| x == 0xface_1234_5678_9abc));
    }
}

fn single_inputs() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        0.999,
        -0.999,
        3.75,
        -3.75,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x4eff_ffff), // Last f32 below the signed W upper limit.
        f32::from_bits(0x4f00_0000), // 2^31, first positive W overflow.
        f32::from_bits(0x4f00_0001),
        f32::from_bits(0xcf00_0000), // -2^31, also x86's invalid-conversion sentinel.
        f32::from_bits(0xcf00_0001),
        f32::from_bits(0x5eff_ffff), // Last f32 below the signed X upper limit.
        f32::from_bits(0x5f00_0000), // 2^63.
        f32::from_bits(0x5f00_0001),
        f32::from_bits(0xdf00_0000), // -2^63.
        f32::from_bits(0xdf00_0001),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345), // Positive quiet NaN.
        f32::from_bits(0xffc1_2345), // Negative quiet NaN.
        f32::from_bits(0x7f81_2345), // Signaling NaN.
        f32::from_bits(0xff81_2345),
    ]
}

fn double_inputs() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        0.999,
        -0.999,
        3.75,
        -3.75,
        f64::from_bits(1),
        f64::from_bits(0x8000_0000_0000_0001),
        2_147_483_647.0,
        2_147_483_647.75,
        2_147_483_648.0,
        2_147_483_649.0,
        -2_147_483_648.0,
        -2_147_483_648.75,
        -2_147_483_649.0,
        f64::from_bits(0x43df_ffff_ffff_ffff), // Last f64 below 2^63.
        f64::from_bits(0x43e0_0000_0000_0000), // 2^63.
        f64::from_bits(0x43e0_0000_0000_0001),
        f64::from_bits(0xc3e0_0000_0000_0000), // -2^63.
        f64::from_bits(0xc3e0_0000_0000_0001),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_1234_5678_9abc),
        f64::from_bits(0x7ff0_1234_5678_9abc),
        f64::from_bits(0xfff0_1234_5678_9abc),
    ]
}

// Rust's specified float-to-integer casts independently supply the Arm result:
// truncation toward zero, saturation at the destination bounds, NaN to zero.
#[test]
fn single_to_signed_w_saturates_and_zero_extends() {
    check_cases(
        0x1e38_0020,
        single_inputs()
            .into_iter()
            .map(|v| (v.to_bits() as u64, v as i32 as u32 as u64)),
    );
}

#[test]
fn single_to_signed_x_saturates() {
    check_cases(
        0x9e38_0020,
        single_inputs()
            .into_iter()
            .map(|v| (v.to_bits() as u64, v as i64 as u64)),
    );
}

#[test]
fn double_to_signed_w_saturates_and_zero_extends() {
    check_cases(
        0x1e78_0020,
        double_inputs()
            .into_iter()
            .map(|v| (v.to_bits(), v as i32 as u32 as u64)),
    );
}

#[test]
fn double_to_signed_x_saturates() {
    check_cases(
        0x9e78_0020,
        double_inputs()
            .into_iter()
            .map(|v| (v.to_bits(), v as i64 as u64)),
    );
}
