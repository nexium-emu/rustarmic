use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE: u64 = 0x1000;
const TARGET: u64 = CODE + 0x100;
const RETAA: u32 = 0xD65F_0BFF;
const RETAB: u32 = 0xD65F_0FFF;

struct Code(Vec<u32>);
impl Memory for Code {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        self.0.get(((addr.checked_sub(CODE)?) / 4) as usize).copied()
    }
}

unsafe extern "C" fn no_mem(_: *mut CpuContext, _: u64, _: u8) {
    panic!("unexpected memory access");
}

fn brk(imm: u32) -> u32 {
    0xD420_0000 | (imm << 5)
}

fn run(insn: u32, setup: impl Fn(&mut CpuContext)) -> CpuContext {
    let mut words = vec![brk(3); 0x41];
    words[0] = insn;
    words[1] = brk(1);
    words[0x40] = brk(2);
    let mut jit = Jit::new(JitConfig::default()).unwrap();
    let mut ctx = CpuContext::default();
    ctx.mem_read = no_mem;
    ctx.mem_write = no_mem;
    ctx.pc = CODE;
    setup(&mut ctx);
    assert_eq!(jit.run(&mut ctx, &mut Code(words)).unwrap(), ExitReason::Brk(2), "{insn:#010x}");
    ctx
}

#[test]
fn authenticated_returns_branch_to_the_link_register() {
    for insn in [RETAA, RETAB] {
        let ctx = run(insn, |ctx| {
            ctx.x[30] = TARGET;
            ctx.sp = 0x7000;
        });
        assert_eq!(ctx.x[30], TARGET);
    }
}

#[test]
fn authenticated_register_branches_use_rn_and_ignore_the_modifier() {
    for key in [0u32, 1 << 10] {
        for insn in [0xD61F_081F | key | (5 << 5), 0xD71F_0800 | key | (5 << 5) | 6] {
            let ctx = run(insn, |ctx| {
                ctx.x[5] = TARGET;
                ctx.x[6] = 0xDEAD_BEEF;
                ctx.x[30] = 0x1234;
            });
            assert_eq!((ctx.x[5], ctx.x[30]), (TARGET, 0x1234), "{insn:#010x}");
        }
    }
}

#[test]
fn authenticated_calls_link_past_the_branch() {
    for key in [0u32, 1 << 10] {
        for insn in [0xD63F_081F | key | (7 << 5), 0xD73F_0800 | key | (7 << 5) | 31] {
            let ctx = run(insn, |ctx| ctx.x[7] = TARGET);
            assert_eq!(ctx.x[30], CODE + 4, "{insn:#010x}");
        }
    }
}
