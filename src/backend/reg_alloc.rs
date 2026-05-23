//! Per-block "register" allocator.
//!
//! V1 strategy: every SSA value gets a stack slot. The slot index is just the
//! ValueRef. Slot widths are bucketed by `Ty` so we can lay them out densely
//! in two arenas (8-byte slots for U1..U64, 16-byte slots for U128).
//!
//! Why such a simple scheme? It costs us a few extra mov/mov pairs per armlet
//! but lets us hit feature completeness on the IR/backend boundary quickly,
//! and the resulting code is *correct* — which is the prerequisite for any
//! sane perf work. The IR layout is friendly to a future linear-scan: every
//! SSA def lives at exactly one index, so a live-range pass is O(n) over the
//! `Vec<Armlet>`.

use crate::ir::{Block, Ty, ValueRef};

#[derive(Clone, Copy, Debug)]
pub struct ValueLoc {
    /// Negative offset from `rbp` (so positive number).
    pub stack_offset: i32,
    pub width: u8, // bytes
}

pub struct Allocation {
    pub slots: Vec<ValueLoc>,
    pub frame_bytes: i32,
}

impl Allocation {
    /// Compute slot offsets. The frame layout is:
    ///
    /// ```text
    /// rbp + 0     : saved rbp
    /// rbp + 8     : return addr
    /// rbp - 8     : saved rbx
    /// rbp - 16    : saved r12
    /// rbp - 24    : saved r13
    /// rbp - 32    : saved r14
    /// rbp - 40    : saved r15  (= CTX_REG)
    /// rbp - 48..  : value slots
    /// ```
    pub fn build(block: &Block) -> Self {
        const SAVED_SIZE: i32 = 40; // five callee-saved 64-bit slots (rbx, r12..r15)
        let n = block.code.len();
        let mut slots = Vec::with_capacity(n);
        let mut next_offset: i32 = SAVED_SIZE; // bytes consumed below rbp

        for armlet in &block.code {
            let width = match armlet.ty {
                Ty::Void | Ty::U1 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::Nzcv => 4,
                Ty::U64  => 8,
                Ty::U128 => 16,
            };
            // 8-byte align the next slot for >=8-byte values; 4-byte align otherwise.
            let align = if width >= 8 { 8 } else { 4 };
            next_offset = (next_offset + align - 1) & -align;
            next_offset += width;
            slots.push(ValueLoc {
                stack_offset: next_offset,
                width: width as u8,
            });
        }

        // Round up to 16 bytes so the stack stays 16-byte aligned after our
        // sub rsp inside the prologue.
        let frame_bytes = (next_offset + 15) & -16;
        Self { slots, frame_bytes }
    }

    #[inline]
    pub fn loc(&self, v: ValueRef) -> ValueLoc {
        self.slots[v.as_usize()]
    }
}
