use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;
const DATA_BASE: u64 = 0x20_0000;
const BRK_0: u32 = 0xD420_0000;

struct CodeMem(Vec<u8>);

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(CODE_BASE)? as usize;
        let bytes = self.0.get(off..off.checked_add(4)?)?;
        Some(u32::from_le_bytes(bytes.try_into().ok()?))
    }
}

fn run(words: &[u32], data: &[u8]) -> CpuContext {
    let mut code = Vec::with_capacity(words.len() * 4);
    for word in words {
        code.extend_from_slice(&word.to_le_bytes());
    }
    let mut backing = vec![0u8; 0x1000];
    backing[..data.len()].copy_from_slice(data);
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    ctx.x[0] = DATA_BASE;
    ctx.mem_base = backing.as_mut_ptr();
    ctx.mem_base_va = DATA_BASE;
    ctx.mem_size = backing.len() as u64;
    let mut jit = Jit::new(JitConfig {
        use_fastmem: true,
        ..JitConfig::default()
    })
    .unwrap();
    let exit = jit.run(&mut ctx, &mut CodeMem(code)).unwrap();
    assert_eq!(exit, ExitReason::Brk(0));
    ctx
}

#[test]
fn ldrsb_w_zero_extends_signed_result_to_32_bits() {
    let ctx = run(&[0x3980_0001, BRK_0], &[0x80]);
    assert_eq!(ctx.x[1], 0x0000_0000_FFFF_FF80);
}

#[test]
fn ldrsb_x_sign_extends_to_64_bits() {
    let ctx = run(&[0x39C0_0001, BRK_0], &[0x80]);
    assert_eq!(ctx.x[1], 0xFFFF_FFFF_FFFF_FF80);
}

#[test]
fn ldrsh_w_zero_extends_signed_result_to_32_bits() {
    let ctx = run(&[0x7980_0001, BRK_0], &[0x00, 0x80]);
    assert_eq!(ctx.x[1], 0x0000_0000_FFFF_8000);
}

#[test]
fn ldrsh_x_sign_extends_to_64_bits() {
    let ctx = run(&[0x79C0_0001, BRK_0], &[0x00, 0x80]);
    assert_eq!(ctx.x[1], 0xFFFF_FFFF_FFFF_8000);
}
