//! Test-only deliberate miscompilation (cargo feature `fault-injection`).
//!
//! Differential harnesses (NeXium's lockstep mode) need a known-wrong JIT to
//! prove they report a divergence at the exact guest instruction.  A `Jit`
//! with `fault_injection` set XORs a constant into the destination register
//! (`Rd`, bits 4:0) of every translated instruction whose encoding matches
//! `mask`/`value`.  The setting is scoped to the `Jit` that owns it: `compile`
//! publishes it in a thread-local only while it translates, so other JITs on
//! other threads are unaffected.

use std::cell::Cell;

use crate::arch::RegSize;
use crate::ir::IrEmitter;

/// Miscompile instructions with `insn & mask == value` by XORing `xor` into
/// their destination register after the instruction's own effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaultInjection {
    pub mask: u32,
    pub value: u32,
    pub xor: u64,
}

thread_local! {
    static ACTIVE: Cell<Option<FaultInjection>> = const { Cell::new(None) };
}

/// Restores the previous thread-local setting on drop.
pub struct Scope(Option<FaultInjection>);

impl Drop for Scope {
    fn drop(&mut self) {
        ACTIVE.with(|a| a.set(self.0));
    }
}

/// Publishes `injection` for translations on this thread until the returned
/// guard drops.
pub fn scope(injection: Option<FaultInjection>) -> Scope {
    Scope(ACTIVE.with(|a| a.replace(injection)))
}

/// Applies the active injection to the instruction just translated.
pub(crate) fn apply(em: &mut IrEmitter<'_>, insn: u32) {
    let Some(f) = ACTIVE.with(Cell::get) else {
        return;
    };
    if insn & f.mask != f.value {
        return;
    }
    let rd = (insn & 31) as u8;
    if rd == 31 {
        return;
    }
    let value = em.get_x(rd);
    let xor = em.const_u64(f.xor);
    let corrupted = em.eor(value, xor, RegSize::X);
    em.set_x(rd, corrupted);
}
