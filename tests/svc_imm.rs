//! Regression for the exit-token SVC/BRK/HVC immediate.
//!
//! Before this fix, `ExitReason::Svc(N)` always reported N=0 because the
//! emitter at `src/backend/emit.rs` discarded the imm field of
//! `Terminal::Exception` and the dispatcher at `src/jit/mod.rs` only
//! looked at the low byte (the kind). NeXium's SVC dispatcher keys every
//! kernel handler off this immediate, so dropping it would misroute every
//! libnx syscall.

#[allow(dead_code)]
mod common;

use rustarmic::{CpuContext, ExitReason, Jit, JitConfig, Memory};

const CODE_BASE: u64 = 0x1000;

fn build_code(words: &[u32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(words.len() * 4);
    for w in words { v.extend_from_slice(&w.to_le_bytes()); }
    v
}

struct CodeMem { bytes: Vec<u8>, base: u64 }

impl Memory for CodeMem {
    fn fetch_inst(&mut self, addr: u64) -> Option<u32> {
        let off = addr.checked_sub(self.base)? as usize;
        if off + 4 > self.bytes.len() { return None; }
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.bytes[off..off + 4]);
        Some(u32::from_le_bytes(buf))
    }
}

fn run_one_insn(insn: u32) -> ExitReason {
    let code = build_code(&[insn]);
    let mut mem = CodeMem { bytes: code, base: CODE_BASE };
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    jit.run(&mut ctx, &mut mem).unwrap_or(ExitReason::Stopped)
}

// AArch64 "Exception generation" encoding:
//   bits[31:21] = 1101_0100_opc(3)   (opc: 000 SVC, 001 BRK, 000 HVC)
//   bits[20:5]  = imm16
//   bits[4:0]   = op2(3)|LL(2)        (LL: 01 SVC, 00 BRK, 10 HVC)
const fn svc(imm: u16) -> u32 { 0xD400_0001 | ((imm as u32) << 5) }
const fn brk(imm: u16) -> u32 { 0xD420_0000 | ((imm as u32) << 5) }
const fn hvc(imm: u16) -> u32 { 0xD400_0002 | ((imm as u32) << 5) }

#[test]
fn svc_carries_immediate_0x1c() {
    // hbmenu's WaitProcessWideKeyAtomic case — this is the regression that
    // was misrouting every libnx SVC as Svc(0).
    assert_eq!(run_one_insn(svc(0x1c)), ExitReason::Svc(0x1c));
}

#[test]
fn svc_carries_immediate_zero() {
    // The historical bug returned Svc(0) by accident; verify that an actual
    // svc #0 still reports 0 (i.e. the encoding can represent it).
    assert_eq!(run_one_insn(svc(0)), ExitReason::Svc(0));
}

#[test]
fn svc_carries_large_immediate() {
    // libnx uses SVC numbers up through 0x7F today; pick something near the
    // top of the u16 range to catch any sign-extension or truncation bug in
    // the bits[8..24] packing.
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
