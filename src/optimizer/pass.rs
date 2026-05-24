use crate::arch::NUM_GPRS;
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};

#[derive(Default)]
pub struct Scratch {
    uses: Vec<u8>,
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

pub fn optimize(block: &mut Block) {
    let mut scratch = Scratch::new();
    optimize_with_scratch(block, &mut scratch);
}

pub fn optimize_with_scratch(block: &mut Block, scratch: &mut Scratch) {
    let mut n = block.code.len();
    if n == 0 { return; }
    scratch.resize(n);

    assume::assume!(unsafe: scratch.consts.len() == n);
    assume::assume!(unsafe: scratch.uses.len()   == n);
    assume::assume!(unsafe: block.code.len()     == n);

    let mut reach_x:    [ValueRef; NUM_GPRS] = [ValueRef::NONE; NUM_GPRS];
    let mut reach_sp:   ValueRef = ValueRef::NONE;
    let mut reach_nzcv: ValueRef = ValueRef::NONE;

    let mut cursor = block.head_vr();
    while let Some(vr) = cursor {
        let i = vr.as_usize();
        assume::assume!(unsafe: i < n);
        let next_cursor = block.next_of(vr);

        let mut a = block.code[i];

        for slot in a.args.iter_mut() {
            if slot.is_none() { continue; }
            while slot.is_some() {
                let sidx = slot.as_usize();
                assume::assume!(unsafe: sidx < n);
                let pointed = &block.code[sidx];
                if pointed.op != Op::Identity { break; }
                let nxt = pointed.args[0];
                if nxt.is_none() || nxt.as_usize() >= sidx { break; }
                *slot = nxt;
            }
        }

        match a.op {
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

            Op::SetX => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    reach_x[reg] = a.args[0];
                }
            }
            Op::SetW => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    reach_x[reg] = a.args[0];
                }
            }
            Op::SetSp   => { reach_sp   = a.args[0]; }
            Op::SetNzcv => { reach_nzcv = a.args[0]; }

            Op::AddsFlags32 | Op::AddsFlags64
            | Op::SubsFlags32 | Op::SubsFlags64
            | Op::Fcmp32 | Op::Fcmp64 => {
                reach_nzcv = ValueRef::NONE;
            }

            Op::ConstU32 => { scratch.consts[i] = Some(a.imm & 0xFFFF_FFFF); }
            Op::ConstU64 => { scratch.consts[i] = Some(a.imm); }

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

        // Mul-fold peephole: collapse `a * K1 ± a * K2` (constructed from
        // chains of Mul/Lsl over a common base) into a single `a * K_total`
        // when the chain already contains a Mul. This shortens the critical
        // path on the common `((c*a)<<b)+a` idiom.
        if matches!(a.op, Op::Add32 | Op::Add64 | Op::Sub32 | Op::Sub64) {
            if let Some((base, coeff)) = try_mul_fold(&a, block, &scratch.consts) {
                let (const_op, mul_op) = if a.op.size_bits() == 32 {
                    (Op::ConstU32, Op::Mul32)
                } else {
                    (Op::ConstU64, Op::Mul64)
                };
                let const_vr = block.insert_before(
                    vr,
                    Armlet::new(const_op, a.ty).with_imm(coeff),
                );
                let mul_vr = block.insert_before(
                    vr,
                    Armlet::new(mul_op, a.ty).with_args(&[base, const_vr]),
                );
                a.become_identity(mul_vr);
                n = block.code.len();
                scratch.consts.resize(n, None);
                scratch.uses.resize(n, 0);
                scratch.consts[const_vr.as_usize()] = Some(coeff);
            }
        }

        match a.op {
            Op::ConstU32 => { scratch.consts[i] = Some(a.imm & 0xFFFF_FFFF); }
            Op::ConstU64 => { scratch.consts[i] = Some(a.imm); }
            Op::Identity => {
                let src = a.args[0];
                if src.is_some() {
                    let sidx = src.as_usize();
                    assume::assume!(unsafe: sidx < n);
                    scratch.consts[i] = scratch.consts[sidx];
                }
            }
            _ => {}
        }

        block.code[i] = a;
        cursor = next_cursor;
    }

    let mut cursor = block.head_vr();
    while let Some(vr) = cursor {
        let i = vr.as_usize();
        assume::assume!(unsafe: i < n);
        let a = &block.code[i];
        for arg in a.args.iter() {
            if arg.is_some() {
                let aidx = arg.as_usize();
                assume::assume!(unsafe: aidx < n);
                let u = &mut scratch.uses[aidx];
                *u = u.saturating_add(1);
            }
        }
        cursor = block.next_of(vr);
    }

    use crate::ir::Terminal;
    let term_vrs: [Option<ValueRef>; 2] = match block.terminal {
        Terminal::ConditionalBranch { cond_nzcv, .. } => [Some(cond_nzcv), None],
        Terminal::CompareBranchZero { value, .. }
        | Terminal::TestBranchBit { value, .. } => [Some(value), None],
        Terminal::IndirectBranch { target, .. } => [Some(target), None],
        _ => [None, None],
    };
    for v in term_vrs {
        if let Some(v) = v {
            if v.is_some() {
                let aidx = v.as_usize();
                let u = &mut scratch.uses[aidx];
                *u = u.saturating_add(1);
            }
        }
    }

    let mut cursor = block.tail_vr();
    while let Some(vr) = cursor {
        let i = vr.as_usize();
        assume::assume!(unsafe: i < n);
        let prev_cursor = block.prev_of(vr);

        let a = block.code[i];
        if a.op.has_side_effects() {
            cursor = prev_cursor;
            continue;
        }
        if scratch.uses[i] == 0 {
            for arg in a.args {
                if arg.is_some() {
                    let aidx = arg.as_usize();
                    assume::assume!(unsafe: aidx < n);
                    let u = &mut scratch.uses[aidx];
                    *u = u.saturating_sub(1);
                }
            }
            block.unlink(vr);
        }
        cursor = prev_cursor;
    }
}

struct Term {
    base: ValueRef,
    coeff: u64,
    has_mul: bool,
}

/// Walk `vr` back through chains of `Mul/Lsl/Identity` to extract a `(base,
/// coefficient)` pair such that the node represents `base * coefficient`
/// (modulo the bit width). `has_mul` records whether a Mul was traversed —
/// the peephole only fires when at least one side already has a Mul, since
/// otherwise the existing shift+add sequence is faster than a 3-cycle imul.
fn extract_term(block: &Block, vr: ValueRef, consts: &[Option<u64>], bits: u32) -> Term {
    let i = vr.as_usize();
    if i >= block.code.len() || vr.is_none() {
        return Term { base: vr, coeff: 1, has_mul: false };
    }
    let a = &block.code[i];
    let get_c = |v: ValueRef| -> Option<u64> {
        if v.is_none() { None } else { consts.get(v.as_usize()).copied().flatten() }
    };
    match a.op {
        Op::Mul32 | Op::Mul64 if a.op.size_bits() == bits => {
            if let Some(c) = get_c(a.args[0]) {
                let inner = extract_term(block, a.args[1], consts, bits);
                Term { base: inner.base, coeff: inner.coeff.wrapping_mul(c), has_mul: true }
            } else if let Some(c) = get_c(a.args[1]) {
                let inner = extract_term(block, a.args[0], consts, bits);
                Term { base: inner.base, coeff: inner.coeff.wrapping_mul(c), has_mul: true }
            } else {
                Term { base: vr, coeff: 1, has_mul: false }
            }
        }
        Op::Lsl32 | Op::Lsl64 if a.op.size_bits() == bits => {
            if let Some(k) = get_c(a.args[1]) {
                let inner = extract_term(block, a.args[0], consts, bits);
                let shift = (k as u32) & (bits - 1);
                Term {
                    base: inner.base,
                    coeff: inner.coeff.wrapping_shl(shift),
                    has_mul: inner.has_mul,
                }
            } else {
                Term { base: vr, coeff: 1, has_mul: false }
            }
        }
        Op::Identity => extract_term(block, a.args[0], consts, bits),
        _ => Term { base: vr, coeff: 1, has_mul: false },
    }
}

fn try_mul_fold(
    add: &Armlet,
    block: &Block,
    consts: &[Option<u64>],
) -> Option<(ValueRef, u64)> {
    let bits = add.op.size_bits();
    let mask: u64 = if bits >= 64 { !0 } else { (1u64 << bits) - 1 };
    let is_sub = matches!(add.op, Op::Sub32 | Op::Sub64);

    let lhs = extract_term(block, add.args[0], consts, bits);
    let rhs = extract_term(block, add.args[1], consts, bits);
    if lhs.base != rhs.base || lhs.base.is_none() { return None; }
    if !(lhs.has_mul || rhs.has_mul) { return None; }

    let combined = if is_sub {
        lhs.coeff.wrapping_sub(rhs.coeff)
    } else {
        lhs.coeff.wrapping_add(rhs.coeff)
    } & mask;

    // Skip degenerate cases that the rest of the optimizer (or DCE) handles
    // better than emitting a Mul: 0 → const-fold to 0, 1 → identity.
    if combined == 0 || combined == 1 { return None; }
    Some((lhs.base, combined))
}

fn try_fold(op: Op, a: &crate::ir::Armlet, consts: &[Option<u64>]) -> Option<u64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::RegSize;
    use crate::ir::IrEmitter;

    #[test]
    fn mul_fold_collapses_mul_shift_add_chain() {
        // Build: r = ((c * a) << b) + a, with c = 3, b = 2 → coefficient 13.
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c = em.const_u64(3);
        let mul = em.push(Armlet::new(Op::Mul64, Ty::U64).with_args(&[a, c]));
        let b = em.const_u64(2);
        let shifted = em.push(Armlet::new(Op::Lsl64, Ty::U64).with_args(&[mul, b]));
        let added = em.push(Armlet::new(Op::Add64, Ty::U64).with_args(&[shifted, a]));
        em.set_x(1, added);

        optimize(&mut block);

        // The original Add should now be Identity, pointing at a freshly
        // inserted Mul whose other operand is a const equal to 13.
        let add_node = &block.code[added.as_usize()];
        assert_eq!(add_node.op, Op::Identity, "Add should be rewritten to Identity");

        let target = add_node.args[0];
        let mul_node = &block.code[target.as_usize()];
        assert_eq!(mul_node.op, Op::Mul64);
        assert_eq!(mul_node.args[0], a, "Mul operand[0] should be the base");

        let coeff_node = &block.code[mul_node.args[1].as_usize()];
        assert_eq!(coeff_node.op, Op::ConstU64);
        assert_eq!(coeff_node.imm, 13, "coefficient should be (3 << 2) + 1 = 13");
    }

    #[test]
    fn mul_fold_skips_chain_without_mul() {
        // (a << 2) + a — pure shift+add, no mul. Should NOT fold (folding into
        // an imul would be slower than the 2-instruction shift+add).
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let two = em.const_u64(2);
        let shifted = em.push(Armlet::new(Op::Lsl64, Ty::U64).with_args(&[a, two]));
        let added = em.push(Armlet::new(Op::Add64, Ty::U64).with_args(&[shifted, a]));
        em.set_x(1, added);

        optimize(&mut block);

        let add_node = &block.code[added.as_usize()];
        assert_eq!(add_node.op, Op::Add64, "no Mul in chain → leave as Add");
    }

    #[test]
    fn mul_fold_handles_commutative_add() {
        // a + (c * a) — same pattern, operands swapped.
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c = em.const_u64(7);
        let mul = em.push(Armlet::new(Op::Mul64, Ty::U64).with_args(&[a, c]));
        let added = em.push(Armlet::new(Op::Add64, Ty::U64).with_args(&[a, mul]));
        em.set_x(1, added);

        optimize(&mut block);

        let add_node = &block.code[added.as_usize()];
        assert_eq!(add_node.op, Op::Identity);
        let mul_node = &block.code[add_node.args[0].as_usize()];
        let coeff_node = &block.code[mul_node.args[1].as_usize()];
        assert_eq!(coeff_node.imm, 8, "7 + 1 = 8");
    }

    // Suppress unused-warning for RegSize import (kept for symmetry with
    // other tests if extended later).
    #[allow(dead_code)]
    fn _silence_unused(_: RegSize) {}
}
