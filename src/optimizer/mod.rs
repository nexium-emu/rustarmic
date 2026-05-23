//! Single-pass optimizer.
//!
//! The pass walks the block in a single forward sweep:
//!
//! 1. **Resolve operands.** For each armlet, look up its arg slots; if an arg
//!    points at a constant or an `Identity`, rewrite the arg in place. This is
//!    copy-propagation+constant-propagation done as a unification pass, with
//!    no work list.
//!
//! 2. **Const-fold pure ops** whose operands are now constants. The folded
//!    armlet becomes `ConstU32`/`ConstU64`.
//!
//! 3. **Track GPR/NZCV reaching definitions** in a small fixed-size array so
//!    `SetX(reg, v); GetX(reg)` becomes `Identity(v)` immediately. This is the
//!    AArch64 equivalent of "context-store elimination" in dynarmic and is
//!    the single highest-impact optimization for translated code.
//!
//! After the forward pass a backward sweep computes use counts and runs DCE.
//!
//! The whole thing stays in cache: ~32 bytes per armlet, all index math, no
//! allocations except the optional `uses` buffer (reused across blocks).

mod pass;

pub use pass::{optimize, optimize_with_scratch, Scratch};
