use rustarmic::{CpuContext, Jit, JitConfig, Memory, StopReason};

const CODE: u64 = 0x1000;
const GUARD: u64 = 0xfedc_ba98_7654_3210;

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

fn vector(bytes: [u8; 16]) -> [u64; 2] {
    [
        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..].try_into().unwrap()),
    ]
}

fn bytes(value: [u64; 2]) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&value[0].to_le_bytes());
    out[8..].copy_from_slice(&value[1].to_le_bytes());
    out
}

fn context() -> CpuContext {
    let mut ctx = CpuContext::default();
    ctx.pc = CODE;
    ctx.x.fill(GUARD);
    ctx.v = std::array::from_fn(|index| [GUARD ^ index as u64, !GUARD ^ index as u64]);
    ctx.sp = GUARD;
    ctx.nzcv = 0xb;
    ctx.fpcr = 0x03c0_0000;
    ctx.fpsr = 0x0800_009f;
    ctx.tpidr_el0 = GUARD;
    ctx.tpidrro_el0 = !GUARD;
    ctx.exclusive_addr = GUARD;
    ctx.exclusive_value = !GUARD;
    ctx.exclusive_size = 8;
    ctx.io_value = [GUARD, !GUARD];
    ctx
}

fn check_guards(ctx: &CpuContext, expected: [[u64; 2]; 32]) {
    assert_eq!(ctx.v, expected);
    assert_eq!(ctx.x, [GUARD; 31]);
    assert_eq!(
        (ctx.sp, ctx.nzcv, ctx.fpcr, ctx.fpsr),
        (GUARD, 0xb, 0x03c0_0000, 0x0800_009f)
    );
    assert_eq!((ctx.tpidr_el0, ctx.tpidrro_el0), (GUARD, !GUARD));
    assert_eq!(
        (ctx.exclusive_addr, ctx.exclusive_value, ctx.exclusive_size),
        (GUARD, !GUARD, 8)
    );
    assert_eq!(ctx.io_value, [GUARD, !GUARD]);
    assert_eq!(
        (
            ctx.should_halt,
            ctx.mem_fault,
            ctx.mem_fault_size,
            ctx.mem_fault_access,
            ctx.mem_fault_cause
        ),
        (0, 0, 0, 0, 0)
    );
    assert_eq!((ctx.mem_fault_addr, ctx.mem_fault_pc), (0, 0));
}

#[test]
fn exact_q0_ext_uses_m_low_not_poisoned_n_high() {
    let mut ctx = context();
    ctx.v[1] = vector([
        0, 1, 2, 3, 4, 5, 6, 7, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa,
    ]);
    ctx.v[2] = vector([
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb, 0xbb,
        0xbb,
    ]);
    let mut expected = ctx.v;
    expected[0] = vector([1, 2, 3, 4, 5, 6, 7, 0x10, 0, 0, 0, 0, 0, 0, 0, 0]);
    let words = vec![0x2e02_0820, 0xd420_0000];
    let mut code = Code(words.clone());
    let mut jit = Jit::new(JitConfig {
        code_cache_bytes: 128 * 1024,
        ..JitConfig::default()
    })
    .unwrap();
    let outcome = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
    assert_eq!(outcome.stop, StopReason::Brk(0));
    assert_eq!(outcome.retired, 2);
    check_guards(&ctx, expected);
    assert_eq!(code.0, words);
}

#[test]
fn all_ext_offsets_q_forms_and_source_destination_aliases() {
    for q in [false, true] {
        for offset in 0..if q { 16 } else { 8 } {
            for (rd, rn, rm) in [
                (0, 1, 2),
                (1, 1, 2),
                (2, 1, 2),
                (31, 31, 0),
                (0, 31, 31),
                (31, 31, 31),
            ] {
                let word = 0x2e00_0000
                    | (u32::from(q) << 30)
                    | (rm << 16)
                    | (offset << 11)
                    | (rn << 5)
                    | rd;
                let mut code = Code(vec![word, 0xd420_0000]);
                let mut jit = Jit::new(JitConfig {
                    code_cache_bytes: 128 * 1024,
                    ..JitConfig::default()
                })
                .unwrap();
                for seed in 0..32u8 {
                    let mut ctx = context();
                    ctx.v[rn as usize] =
                        vector(std::array::from_fn(|i| seed.wrapping_add(i as u8 * 3)));
                    ctx.v[rm as usize] = vector(std::array::from_fn(|i| {
                        seed.wrapping_add(0x80).wrapping_add(i as u8 * 5)
                    }));
                    let n = bytes(ctx.v[rn as usize]);
                    let m = bytes(ctx.v[rm as usize]);
                    let len = if q { 16 } else { 8 };
                    let joined: Vec<_> = n[..len].iter().chain(&m[..len]).copied().collect();
                    let mut wanted = [0; 16];
                    wanted[..len].copy_from_slice(&joined[offset as usize..offset as usize + len]);
                    let mut expected = ctx.v;
                    expected[rd as usize] = vector(wanted);
                    let outcome = jit.run_bounded(&mut ctx, &mut code, 2).unwrap();
                    assert_eq!(outcome.stop, StopReason::Brk(0), "{word:08x}");
                    assert_eq!(outcome.retired, 2);
                    check_guards(&ctx, expected);
                }
            }
        }
    }
}
