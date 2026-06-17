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

const fn movz_x0(imm: u16) -> u32 {
    0xD280_0000 | ((imm as u32) << 5)
}
const BRK_0: u32 = 0xD420_0000;

#[test]
fn invalidate_range_clears_cached_block() {
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let mut mem = CodeMem { bytes: build_code(&[movz_x0(0x111), BRK_0]), base: CODE_BASE };

    jit.run(&mut ctx, &mut mem).unwrap();
    assert!(jit.cache.lookup(CODE_BASE).is_some(),
            "block should be cached after first run");

    jit.invalidate_range(CODE_BASE, 4);
    assert!(jit.cache.lookup(CODE_BASE).is_none(),
            "block should be evicted by invalidate_range");
}

#[test]
fn invalidate_range_preserves_blocks_outside() {
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let mut ctx = CpuContext::default();
    ctx.pc = CODE_BASE;
    let mut mem = CodeMem { bytes: build_code(&[movz_x0(0x111), BRK_0]), base: CODE_BASE };

    jit.run(&mut ctx, &mut mem).unwrap();
    assert!(jit.cache.lookup(CODE_BASE).is_some());

    jit.invalidate_range(0xF000_0000, 0x1000);
    assert!(jit.cache.lookup(CODE_BASE).is_some(),
            "out-of-range invalidate must leave block cached");
}

#[test]
fn invalidate_range_forces_recompile_after_rewrite() {
    let mut jit = Jit::new(JitConfig::default()).expect("jit init");
    let mut ctx = CpuContext::default();
    let mut mem = CodeMem { bytes: build_code(&[movz_x0(0x111), BRK_0]), base: CODE_BASE };

    ctx.pc = CODE_BASE;
    let exit = jit.run(&mut ctx, &mut mem).unwrap();
    assert!(matches!(exit, ExitReason::Brk(_)));
    assert_eq!(ctx.x[0], 0x111, "first run sees the original code");

    mem.bytes = build_code(&[movz_x0(0x222), BRK_0]);

    ctx.pc = CODE_BASE;
    ctx.x[0] = 0;
    jit.run(&mut ctx, &mut mem).unwrap();
    assert_eq!(ctx.x[0], 0x111, "stale cache still returns the old constant");

    jit.invalidate_range(CODE_BASE, 8);
    ctx.pc = CODE_BASE;
    ctx.x[0] = 0;
    jit.run(&mut ctx, &mut mem).unwrap();
    assert_eq!(ctx.x[0], 0x222, "after invalidate, recompile picks up new bytes");
}
