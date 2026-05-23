//! The Armlet IR.
//!
//! ## Why a Vec, not a linked list
//!
//! Dynarmic stores its IR as `intrusive_list<Inst>` — pointer-chained nodes
//! living on the heap. That made every traversal pay an L1/L2 miss per node,
//! and every insertion/removal touched four pointers under a global allocator.
//!
//! We flatten the IR into a single `Vec<Armlet>`. Each instruction is 32 bytes
//! (one half of a 64-byte cache line), and a `ValueRef` is just a `u32` index
//! into that vector. Walking the block is a tight pointer-bump loop; the
//! optimizer is a forward pass that streams from L1.
//!
//! ## SSA invariants
//!
//! - A value's "name" is its index in `Block::code`. There is no separate
//!   def table — `code[i]` *is* `%i`.
//! - `ValueRef::NONE` (= `u32::MAX`) is the sentinel for unused arg slots.
//! - Pure ops can be killed in place (`Op::Identity` or `Op::Void`) without
//!   shifting the vector — index stability is critical.

pub mod opcode;
pub mod value;
pub mod armlet;
pub mod block;
pub mod emitter;
pub mod cfg;

pub use opcode::Op;
pub use value::{ValueRef, Ty};
pub use armlet::{Armlet, ArmletFlags};
pub use block::{Block, Terminal};
pub use emitter::IrEmitter;
pub use cfg::Cfg;
