//! The single forward-then-DCE-backward optimization pass.

use crate::arch::NUM_GPRS;
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};

/// Reusable scratch buffers, kept alive across blocks to avoid reallocations.
#[derive(Default)]
pub struct Scratch {
    /// Per-armlet use count, saturated at 255.
    uses: Vec<u8>,
    /// Per-armlet constant value (if any) — `None` means non-constant or eliminated.
    consts: Vec<Option<u64>>,
}

impl Scratch {
    pub fn new() -> Self { Self::default() }

    fn resize(&mut self, n: usize) {
        self.uses.clear();
        self.uses.resize(n, 0);
        self.consts.clear();
        self.consts.resize(n, None);
    }
}

/// Run the optimizer over `block`, allocating fresh scratch buffers.
pub fn optimize(block: &mut Block) {
    let mut scratch = Scratch::new();
    optimize_with_scratch(block, &mut scratch);
}

/// Run the optimizer over `block`, reusing the provided scratch buffers.
pub fn optimize_with_scratch(block: &mut Block, scratch: &mut Scratch) {
    let n = block.code.len();
    if n == 0 { return; }
    scratch.resize(n);

    // ── Forward pass ────────────────────────────────────────────────────────
    //
    // We track:
    //   - reaching def for each guest GPR (NUM_GPRS slots) + SP + NZCV.
    //   - latest constant per ValueRef in `scratch.consts`.
    //
    // Every armlet:
    //   1. Resolve its args (copy/const prop).
    //   2. If the op is pure and all operands are constants, fold to a Const*.
    //   3. If it's a GetX/GetW/GetSp/GetNzcv with a known reaching def, rewrite
    //      to an `Identity` of that def.
    //   4. If it's a SetX/SetW/SetSp/SetNzcv, record the reaching def.
    //
    // We never touch indices, so SSA references remain valid throughout.

    let mut reach_x:    [ValueRef; NUM_GPRS] = [ValueRef::NONE; NUM_GPRS];
    let mut reach_sp:   ValueRef = ValueRef::NONE;
    let mut reach_nzcv: ValueRef = ValueRef::NONE;

    for i in 0..n {
        // First, resolve operands (read-only borrow needed).
        let mut a = block.code[i];

        for slot in a.args.iter_mut() {
            if slot.is_none() { continue; }
            // Chase identity chains. SSA guarantees args[0].idx < idx so the
            // loop terminates in at most `i` steps; in practice 1.
            while slot.is_some() {
                let pointed = &block.code[slot.as_usize()];
                if pointed.op != Op::Identity { break; }
                let next = pointed.args[0];
                if next.is_none() || next.as_usize() >= slot.as_usize() { break; }
                *slot = next;
            }
        }

        // After operand resolution, attempt optimizations on the op itself.
        match a.op {
            // ── GPR / state reads → identity to last writer ──────────────────
            Op::GetX => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    let def = reach_x[reg];
                    if def.is_some() {
                        a.become_identity(def);
                    }
                }
            }
            Op::GetW => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    let def = reach_x[reg];
                    if def.is_some() {
                        // Reading W view of a 64-bit def: mask low 32 bits.
                        // The optimizer can't always insert a new armlet *before*
                        // i without shifting indices, so we just lean on the
                        // backend to do the mask. We still rewrite as identity
                        // for value-numbering purposes.
                        a.become_identity(def);
                        a.ty = Ty::U32;
                    }
                }
            }
            Op::GetSp => {
                if reach_sp.is_some() {
                    a.become_identity(reach_sp);
                }
            }
            Op::GetNzcv => {
                if reach_nzcv.is_some() {
                    a.become_identity(reach_nzcv);
                }
            }

            // ── GPR / state writes — record reaching def, keep armlet ────────
            Op::SetX => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    reach_x[reg] = a.args[0];
                }
            }
            Op::SetW => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    // Top half zeros, so the reaching def of the 64-bit reg
                    // becomes the masked 32-bit value. We model this as the
                    // raw ValueRef — the backend already knows W writes zero.
                    reach_x[reg] = a.args[0];
                }
            }
            Op::SetSp   => { reach_sp   = a.args[0]; }
            Op::SetNzcv => { reach_nzcv = a.args[0]; }

            // ── Constant materializations record into scratch ────────────────
            Op::ConstU32 => { scratch.consts[i] = Some(a.imm & 0xFFFF_FFFF); }
            Op::ConstU64 => { scratch.consts[i] = Some(a.imm); }

            // ── Pure arithmetic — try to const-fold ──────────────────────────
            op if op.is_pure() => {
                if let Some(folded) = try_fold(op, &a, &scratch.consts) {
                    match a.ty {
                        Ty::U32 => a.become_const_u32(folded as u32),
                        Ty::U64 => a.become_const_u64(folded),
                        _ => {}
                    }
                }
            }

            _ => {}
        }

        // After folding, propagate any constant the rewritten armlet now is.
        match a.op {
            Op::ConstU32 => { scratch.consts[i] = Some(a.imm & 0xFFFF_FFFF); }
            Op::ConstU64 => { scratch.consts[i] = Some(a.imm); }
            Op::Identity => {
                let src = a.args[0];
                if src.is_some() {
                    scratch.consts[i] = scratch.consts[src.as_usize()];
                }
            }
            _ => {}
        }

        block.code[i] = a;
    }

    // ── Backward DCE sweep ──────────────────────────────────────────────────
    //
    // Count uses (already populated by walking forward would have required
    // another pass; we accept one final backward sweep to keep the use-set
    // exact). Then mark dead armlets.

    // Use counts
    for i in 0..n {
        let a = block.code[i];
        if a.is_eliminated() { continue; }
        for arg in a.args.iter() {
            if arg.is_some() {
                let u = &mut scratch.uses[arg.as_usize()];
                *u = u.saturating_add(1);
            }
        }
    }

    for i in (0..n).rev() {
        let a = &mut block.code[i];
        if a.is_eliminated() { continue; }
        if a.op.has_side_effects() { continue; }
        if scratch.uses[i] == 0 {
            // Decrement use counts of operands before killing.
            for arg in a.args {
                if arg.is_some() {
                    let u = &mut scratch.uses[arg.as_usize()];
                    *u = u.saturating_sub(1);
                }
            }
            a.mark_eliminated();
        }
    }
}

/// Constant-fold a pure armlet when all needed operands are constants.
fn try_fold(op: Op, a: &Armlet, consts: &[Option<u64>]) -> Option<u64> {
    let get = |v: ValueRef| -> Option<u64> {
        if v.is_none() { None } else { consts[v.as_usize()] }
    };
    let x = get(a.args[0])?;
    let y_opt = get(a.args[1]);

    use Op::*;
    let r = match op {
        Add32 => (x as u32).wrapping_add(y_opt? as u32) as u64,
        Add64 => x.wrapping_add(y_opt?),
        Sub32 => (x as u32).wrapping_sub(y_opt? as u32) as u64,
        Sub64 => x.wrapping_sub(y_opt?),
        And32 => (x as u32 & y_opt? as u32) as u64,
        And64 => x & y_opt?,
        Or32  => (x as u32 | y_opt? as u32) as u64,
        Or64  => x | y_opt?,
        Eor32 => (x as u32 ^ y_opt? as u32) as u64,
        Eor64 => x ^ y_opt?,
        Lsl32 => ((x as u32).wrapping_shl(y_opt? as u32 & 31)) as u64,
        Lsl64 => x.wrapping_shl(y_opt? as u32 & 63),
        Lsr32 => ((x as u32).wrapping_shr(y_opt? as u32 & 31)) as u64,
        Lsr64 => x.wrapping_shr(y_opt? as u32 & 63),
        Asr32 => ((x as i32).wrapping_shr(y_opt? as u32 & 31)) as u32 as u64,
        Asr64 => ((x as i64).wrapping_shr(y_opt? as u32 & 63)) as u64,
        Ror32 => ((x as u32).rotate_right(y_opt? as u32 & 31)) as u64,
        Ror64 => x.rotate_right(y_opt? as u32 & 63),
        Mul32 => ((x as u32).wrapping_mul(y_opt? as u32)) as u64,
        Mul64 => x.wrapping_mul(y_opt?),
        Not32 => (!x) & 0xFFFF_FFFF,
        Not64 => !x,
        Neg32 => ((!(x as u32)).wrapping_add(1)) as u64,
        Neg64 => (!x).wrapping_add(1),
        Identity => x,
        _ => return None,
    };
    Some(r)
}
