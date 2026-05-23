//! Basic block: a flat `Vec<Armlet>` plus a terminator descriptor.

use crate::ir::{Armlet, Op, ValueRef};

/// What happens at the end of this block — driven by the final terminator
/// armlet but flattened here for convenience for the backend and chaining.
#[derive(Clone, Copy, Debug)]
pub enum Terminal {
    /// We haven't translated a terminator yet.
    Invalid,
    /// Falls through to `next_pc` because we hit our budget.
    LinkBlock { next_pc: u64 },
    /// Direct branch to a known PC (unconditional, B / BL).
    DirectBranch { target_pc: u64, link: bool },
    /// Conditional direct branch; one side known, fall-through PC also known.
    ConditionalBranch { cond_nzcv: ValueRef, cond_code: u8, taken_pc: u64, not_taken_pc: u64 },
    /// CBZ/CBNZ.
    CompareBranchZero { value: ValueRef, inverse: bool, taken_pc: u64, not_taken_pc: u64 },
    /// TBZ/TBNZ.
    TestBranchBit { value: ValueRef, bit: u8, inverse: bool, taken_pc: u64, not_taken_pc: u64 },
    /// Indirect branch (BR/RET): target only known at runtime.
    IndirectBranch { target: ValueRef, link: bool, is_ret: bool },
    /// SVC/BRK/HVC: deliver an exception then return to host dispatch.
    Exception { kind: ExceptionKind, imm: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExceptionKind {
    Svc, Brk, Hvc, UnknownInst,
}

/// A translation unit.
///
/// Holds the flat SSA stream plus enough context to feed the backend.
pub struct Block {
    /// SSA instructions. Indices into this vector are stable `ValueRef`s.
    pub code: Vec<Armlet>,
    /// Guest PC of the first instruction translated into this block.
    pub start_pc: u64,
    /// Guest PC one past the last instruction (i.e. fall-through PC).
    pub end_pc: u64,
    /// Block terminator descriptor (also reflected by the final armlet).
    pub terminal: Terminal,
    /// Cycle count estimate (1 per guest insn for now).
    pub cycles: u32,
}

impl Block {
    /// Initial capacity sized for a typical hot block (~32 guest insns → ~96 armlets).
    pub const INITIAL_CAPACITY: usize = 128;

    pub fn new(start_pc: u64) -> Self {
        Self {
            code: Vec::with_capacity(Self::INITIAL_CAPACITY),
            start_pc,
            end_pc: start_pc,
            terminal: Terminal::Invalid,
            cycles: 0,
        }
    }

    /// Push a new armlet and return its SSA name.
    #[inline]
    pub fn push(&mut self, armlet: Armlet) -> ValueRef {
        let idx = self.code.len() as u32;
        debug_assert!(idx < u32::MAX, "block too large");
        self.code.push(armlet);
        ValueRef::new(idx)
    }

    #[inline]
    pub fn len(&self) -> usize { self.code.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.code.is_empty() }

    #[inline]
    pub fn get(&self, v: ValueRef) -> &Armlet {
        &self.code[v.as_usize()]
    }

    #[inline]
    pub fn get_mut(&mut self, v: ValueRef) -> &mut Armlet {
        &mut self.code[v.as_usize()]
    }

    /// Walk armlets in program order, skipping eliminated slots.
    #[inline]
    pub fn iter_live(&self) -> impl Iterator<Item = (ValueRef, &Armlet)> {
        self.code.iter().enumerate().filter_map(|(i, a)| {
            if a.is_eliminated() || a.op == Op::Void { None }
            else { Some((ValueRef::new(i as u32), a)) }
        })
    }
}
