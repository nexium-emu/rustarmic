use rustarmic::backend::cpu_features::{CpuFeatures, detect_features};
use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const BRK: u32 = 0xd420_0000;
const GUARD: u64 = 0xdead_beef_0123_4567;

struct Instructions([u32; 2]);

impl Memory for Instructions {
    fn fetch_inst(&mut self, address: u64) -> Option<u32> {
        let offset = address.checked_sub(CODE)?;
        if offset & 3 != 0 {
            return None;
        }
        self.0.get(offset as usize / 4).copied()
    }
}

fn vector(bytes: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ]
}

fn variants() -> [CpuFeatures; 3] {
    let actual = detect_features();
    let mut legacy = actual;
    legacy.has_avx = false;
    let mut portable = legacy;
    portable.has_ssse3 = false;
    [actual, legacy, portable]
}

fn seeded_context() -> CpuContext {
    let mut context = CpuContext::default();
    context.pc = CODE;
    context.x.fill(GUARD);
    context.v = std::array::from_fn(|index| [GUARD ^ index as u64, !GUARD ^ index as u64]);
    context.sp = GUARD;
    context.nzcv = 0xb;
    context.fpcr = 0x03c0_0000;
    context.fpsr = 0x0800_009f;
    context.io_value = [GUARD, !GUARD];
    context
}

#[test]
fn reverse_byte_and_halfword_groups_all_features_lengths_values_and_aliases() {
    for features in variants() {
        for group_log2 in [1u32, 2, 3] {
            for element_log2 in 0..group_log2 {
                for q in [false, true] {
                    for (rd, rn) in [(0u32, 1u32), (1, 1), (31, 31), (0, 31), (31, 0)] {
                        let base = if group_log2 == 2 {
                            0x2e20_0800
                        } else if group_log2 == 1 {
                            0x0e20_1800
                        } else {
                            0x0e20_0800
                        };
                        let instruction =
                            base | (element_log2 << 22) | (u32::from(q) << 30) | (rn << 5) | rd;
                        let mut memory = Instructions([instruction, BRK]);
                        let mut jit = Jit::new(JitConfig {
                            code_cache_bytes: 128 * 1024,
                            host_features: Some(features),
                            ..JitConfig::default()
                        })
                        .unwrap();
                        for value in 0..=255u8 {
                            let input = std::array::from_fn(|lane| {
                                value.wrapping_add((lane as u8).wrapping_mul(17))
                            });
                            let mut context = seeded_context();
                            context.v[rn as usize] = vector(input);
                            let mut expected = context.v;
                            let group = 1usize << group_log2;
                            let element = 1usize << element_log2;
                            let mut output = [0u8; 16];
                            for byte in 0..if q { 16 } else { 8 } {
                                let base = byte / group * group;
                                let lane = byte % group / element;
                                output[byte] =
                                    input[base + group - element * (lane + 1) + byte % element];
                            }
                            expected[rd as usize] = vector(output);
                            let outcome = jit.run_bounded(&mut context, &mut memory, 2).unwrap();
                            assert_eq!(outcome.stop, StopReason::Brk(0));
                            assert_eq!(outcome.retired, 2);
                            assert_eq!(
                                context.v, expected,
                                "word={instruction:08x} features={features:?} value={value}"
                            );
                            assert_eq!(context.x, [GUARD; 31]);
                            assert_eq!(
                                (context.sp, context.nzcv, context.fpcr, context.fpsr),
                                (GUARD, 0xb, 0x03c0_0000, 0x0800_009f)
                            );
                            assert_eq!(context.io_value, [GUARD, !GUARD]);
                            assert_eq!((context.mem_fault, context.should_halt), (0, 0));
                            assert_eq!(memory.0, [instruction, BRK]);
                        }
                    }
                }
            }
        }
    }
}
