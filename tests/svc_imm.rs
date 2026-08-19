#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;

fn build_code(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for w in words {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v
}

struct CodeMem {
    bytes: Vec<u8>,
    base: u64,
}

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(self.base)? as usize;
        if off + 4 > self.bytes.len() {
            return None;
        }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[off..off + 4]);
        Some(u32::from_le_bytes(buf))
    }
}

fn run_one_insn(insn: u32) -> ExitReason {
    let code = build_code(&[insn]);
    let mut mem = CodeMem {
        bytes: code,
        base: CODE_BASE,
    };
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    jit.run(&mut ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

const fn svc(imm: u16) -> u32 {
    0xD400_0001 | ((imm as u32) << 5)
}
const fn brk(imm: u16) -> u32 {
    0xD420_0000 | ((imm as u32) << 5)
}
const fn hvc(imm: u16) -> u32 {
    0xD400_0002 | ((imm as u32) << 5)
}

#[test]
fn svc_carries_immediate_0x1c() {
    assert_eq!(run_one_insn(svc(0x1c)), ExitReason::Svc(0x1c));
}

#[test]
fn svc_carries_immediate_zero() {
    assert_eq!(run_one_insn(svc(0)), ExitReason::Svc(0));
}

#[test]
fn svc_carries_large_immediate() {
    assert_eq!(run_one_insn(svc(0xABCD)), ExitReason::Svc(0xABCD));
}

#[test]
fn brk_carries_immediate() {
    assert_eq!(run_one_insn(brk(0x42)), ExitReason::Brk(0x42));
}

#[test]
fn hvc_carries_immediate() {
    assert_eq!(run_one_insn(hvc(0x55)), ExitReason::Hvc(0x55));
}
