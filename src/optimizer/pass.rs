#![allow(clippy::manual_flatten, clippy::collapsible_match)]

use crate::arch::{Cond, NUM_GPRS, Nzcv};
use crate::ir::{Armlet, Block, Op, Terminal, Ty, ValueRef};

#[derive(Default)]
pub struct Scratch {
    uses: Vec<u8>,
    consts: Vec<Option<u64>>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

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
    if n == 0 {
        return;
    }
    scratch.resize(n);

    let mut reach_x: [ValueRef; NUM_GPRS] = [ValueRef::NONE; NUM_GPRS];
    let mut reach_sp: ValueRef = ValueRef::NONE;
    let mut reach_nzcv: ValueRef = ValueRef::NONE;

    let mut last_setx: [ValueRef; NUM_GPRS] = [ValueRef::NONE; NUM_GPRS];
    let mut last_set_sp: ValueRef = ValueRef::NONE;
    let mut last_set_nzcv: ValueRef = ValueRef::NONE;

    let mut cursor = block.head_vr();
    while let Some(vr) = cursor {
        let i = vr.as_usize();
        unsafe { core::hint::assert_unchecked(i < n) };
        let next_cursor = block.next_of(vr);

        let mut a = block.code[i];

        for slot in a.args.iter_mut() {
            if slot.is_none() {
                continue;
            }
            while slot.is_some() {
                let sidx = slot.as_usize();
                unsafe { core::hint::assert_unchecked(sidx < n) };
                let pointed = &block.code[sidx];
                if pointed.op != Op::Identity {
                    break;
                }
                let nxt = pointed.args[0];
                if nxt.is_none() || nxt.as_usize() >= sidx {
                    break;
                }
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
                    } else {
                        last_setx[reg] = ValueRef::NONE;
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
                    } else {
                        last_setx[reg] = ValueRef::NONE;
                    }
                }
            }
            Op::GetSp => {
                if reach_sp.is_some() {
                    a.become_identity(reach_sp);
                } else {
                    last_set_sp = ValueRef::NONE;
                }
            }
            Op::GetNzcv => {
                if reach_nzcv.is_some() {
                    a.become_identity(reach_nzcv);
                } else {
                    last_set_nzcv = ValueRef::NONE;
                }
            }

            Op::SetX | Op::SetW => {
                let reg = a.imm as usize;
                if reg < NUM_GPRS {
                    let prev = last_setx[reg];
                    if prev.is_some() {
                        block.unlink(prev);
                    }
                    reach_x[reg] = a.args[0];
                    last_setx[reg] = vr;
                }
            }
            Op::SetSp => {
                let prev = last_set_sp;
                if prev.is_some() {
                    block.unlink(prev);
                }
                reach_sp = a.args[0];
                last_set_sp = vr;
            }
            Op::SetNzcv => {
                let prev = last_set_nzcv;
                if prev.is_some() {
                    block.unlink(prev);
                }
                reach_nzcv = a.args[0];
                last_set_nzcv = vr;
            }

            Op::AddsFlags32
            | Op::AddsFlags64
            | Op::SubsFlags32
            | Op::SubsFlags64
            | Op::Fcmp32
            | Op::Fcmp64 => {
                reach_nzcv = ValueRef::NONE;
                let prev = last_set_nzcv;
                if prev.is_some() {
                    block.unlink(prev);
                }
                last_set_nzcv = ValueRef::NONE;
            }

            Op::ConstU32 => {
                scratch.consts[i] = Some(a.imm & 0xFFFF_FFFF);
            }
            Op::ConstU64 => {
                scratch.consts[i] = Some(a.imm);
            }

            op if op.is_pure() => {
                if let Some(folded) = try_fold(op, &a, &scratch.consts) {
                    match a.ty {
                        Ty::U32 => a.become_const_u32(folded as u32),
                        Ty::U64 => a.become_const_u64(folded),
                        _ => {}
                    }
                } else if let Some(simp) = try_strength_reduce(op, &a, &scratch.consts) {
                    match simp {
                        Simplify::ToConst(v) => match a.ty {
                            Ty::U32 => a.become_const_u32(v as u32),
                            Ty::U64 => a.become_const_u64(v),
                            _ => {}
                        },
                        Simplify::ToIdentity(vr) => a.become_identity(vr),
                    }
                } else if let Some((base, new_const, new_op)) =
                    try_combine_const(&a, block, &scratch.consts)
                {
                    let (const_op, const_ty) = if new_op.size_bits() == 32 {
                        (Op::ConstU32, Ty::U32)
                    } else {
                        (Op::ConstU64, Ty::U64)
                    };
                    let const_vr = block
                        .insert_before(vr, Armlet::new(const_op, const_ty).with_imm(new_const));
                    a.op = new_op;
                    a.args = [base, const_vr, ValueRef::NONE, ValueRef::NONE];
                    n = block.code.len();
                    scratch.consts.resize(n, None);
                    scratch.uses.resize(n, 0);
                    scratch.consts[const_vr.as_usize()] = Some(new_const);
                }
            }

            _ => {
                if a.op.has_side_effects() {
                    for s in last_setx.iter_mut() {
                        *s = ValueRef::NONE;
                    }
                    last_set_sp = ValueRef::NONE;
                    last_set_nzcv = ValueRef::NONE;
                }
            }
        }

        if matches!(a.op, Op::Add32 | Op::Add64 | Op::Sub32 | Op::Sub64) {
            if let Some((base, coeff)) = try_mul_fold(&a, block, &scratch.consts) {
                let (const_op, mul_op) = if a.op.size_bits() == 32 {
                    (Op::ConstU32, Op::Mul32)
                } else {
                    (Op::ConstU64, Op::Mul64)
                };
                let const_vr = block.insert_before(vr, Armlet::new(const_op, a.ty).with_imm(coeff));
                let mul_vr =
                    block.insert_before(vr, Armlet::new(mul_op, a.ty).with_args(&[base, const_vr]));
                a.become_identity(mul_vr);
                n = block.code.len();
                scratch.consts.resize(n, None);
                scratch.uses.resize(n, 0);
                scratch.consts[const_vr.as_usize()] = Some(coeff);
            }
        }

        match a.op {
            Op::ConstU32 => {
                scratch.consts[i] = Some(a.imm & 0xFFFF_FFFF);
            }
            Op::ConstU64 => {
                scratch.consts[i] = Some(a.imm);
            }
            Op::Identity => {
                let src = a.args[0];
                if src.is_some() {
                    let sidx = src.as_usize();
                    unsafe { core::hint::assert_unchecked(sidx < n) };
                    scratch.consts[i] = scratch.consts[sidx];
                }
            }
            _ => {}
        }

        block.code[i] = a;
        cursor = next_cursor;
    }

    simplify_terminal(block, &scratch.consts);
    n = block.code.len();

    let mut cursor = block.head_vr();
    while let Some(vr) = cursor {
        let i = vr.as_usize();
        unsafe { core::hint::assert_unchecked(i < n) };
        let a = &block.code[i];
        for arg in a.args.iter() {
            if arg.is_some() {
                let aidx = arg.as_usize();
                unsafe { core::hint::assert_unchecked(aidx < n) };
                let u = &mut scratch.uses[aidx];
                *u = u.saturating_add(1);
            }
        }
        cursor = block.next_of(vr);
    }

    let term_vrs: [Option<ValueRef>; 2] = match block.terminal {
        Terminal::ConditionalBranch { cond_nzcv, .. } => [Some(cond_nzcv), None],
        Terminal::CompareBranchZero { value, .. } | Terminal::TestBranchBit { value, .. } => {
            [Some(value), None]
        }
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
        unsafe { core::hint::assert_unchecked(i < n) };
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
                    unsafe { core::hint::assert_unchecked(aidx < n) };
                    let u = &mut scratch.uses[aidx];
                    *u = u.saturating_sub(1);
                }
            }
            block.unlink(vr);
        }
        cursor = prev_cursor;
    }
}

fn simplify_terminal(block: &mut Block, consts: &[Option<u64>]) {
    let const_of = |v: ValueRef| -> Option<u64> {
        if v.is_none() {
            None
        } else {
            consts.get(v.as_usize()).copied().flatten()
        }
    };

    let new = match block.terminal {
        Terminal::CompareBranchZero {
            value,
            inverse,
            taken_pc,
            not_taken_pc,
        } => const_of(value).map(|v| {
            let take = (v == 0) ^ inverse;
            Terminal::DirectBranch {
                target_pc: if take { taken_pc } else { not_taken_pc },
                link: false,
            }
        }),
        Terminal::TestBranchBit {
            value,
            bit,
            inverse,
            taken_pc,
            not_taken_pc,
        } => const_of(value).map(|v| {
            let bit_set = ((v >> bit) & 1) != 0;
            let take = if inverse { bit_set } else { !bit_set };
            Terminal::DirectBranch {
                target_pc: if take { taken_pc } else { not_taken_pc },
                link: false,
            }
        }),
        Terminal::ConditionalBranch {
            cond_nzcv,
            cond_code,
            taken_pc,
            not_taken_pc,
        } => {
            let cond = Cond::from_bits(cond_code);
            if matches!(cond, Cond::AL | Cond::NV) {
                Some(Terminal::DirectBranch {
                    target_pc: taken_pc,
                    link: false,
                })
            } else {
                const_of(cond_nzcv).map(|nz| {
                    let take = Nzcv(nz as u8).check(cond);
                    Terminal::DirectBranch {
                        target_pc: if take { taken_pc } else { not_taken_pc },
                        link: false,
                    }
                })
            }
        }
        _ => None,
    };

    if let Some(new_term) = new {
        block.terminal = new_term;
        if let Some(tail) = block.tail_vr() {
            let op = block.code[tail.as_usize()].op;
            if matches!(op, Op::CbZ | Op::CbNz | Op::TbZ | Op::TbNz | Op::BranchCond) {
                block.unlink(tail);
            }
        }
    }
}

fn try_combine_const(
    a: &Armlet,
    block: &Block,
    consts: &[Option<u64>],
) -> Option<(ValueRef, u64, Op)> {
    let outer_op = a.op;
    let outer_const = consts.get(a.args[1].as_usize()).copied().flatten()?;
    let inner_vr = a.args[0];
    if !inner_vr.is_some() {
        return None;
    }
    let inner = block.code.get(inner_vr.as_usize())?;
    let inner_const = consts.get(inner.args[1].as_usize()).copied().flatten()?;
    let base = inner.args[0];
    if !base.is_some() {
        return None;
    }

    let bits = outer_op.size_bits();
    let mask: u64 = if bits >= 64 { !0 } else { (1u64 << bits) - 1 };
    let bits_minus_1 = (bits - 1) as u64;

    use Op::*;
    let (new_const, new_op): (u64, Op) = match (outer_op, inner.op) {
        (Add32, Add32) => (inner_const.wrapping_add(outer_const) & mask, Add32),
        (Add64, Add64) => (inner_const.wrapping_add(outer_const), Add64),
        (Sub32, Sub32) => (
            0u64.wrapping_sub(inner_const.wrapping_add(outer_const)) & mask,
            Add32,
        ),
        (Sub64, Sub64) => (
            0u64.wrapping_sub(inner_const.wrapping_add(outer_const)),
            Add64,
        ),
        (And32, And32) => (inner_const & outer_const & mask, And32),
        (And64, And64) => (inner_const & outer_const, And64),
        (Or32, Or32) => ((inner_const | outer_const) & mask, Or32),
        (Or64, Or64) => (inner_const | outer_const, Or64),
        (Eor32, Eor32) => ((inner_const ^ outer_const) & mask, Eor32),
        (Eor64, Eor64) => (inner_const ^ outer_const, Eor64),
        (Mul32, Mul32) => (inner_const.wrapping_mul(outer_const) & mask, Mul32),
        (Mul64, Mul64) => (inner_const.wrapping_mul(outer_const), Mul64),

        (Add32, Sub32) => (outer_const.wrapping_sub(inner_const) & mask, Add32),
        (Add64, Sub64) => (outer_const.wrapping_sub(inner_const), Add64),
        (Sub32, Add32) => (inner_const.wrapping_sub(outer_const) & mask, Add32),
        (Sub64, Add64) => (inner_const.wrapping_sub(outer_const), Add64),

        (Lsl32, Lsl32)
        | (Lsl64, Lsl64)
        | (Lsr32, Lsr32)
        | (Lsr64, Lsr64)
        | (Asr32, Asr32)
        | (Asr64, Asr64) => {
            let sum = inner_const.wrapping_add(outer_const);
            if sum > bits_minus_1 {
                return None;
            }
            (sum, outer_op)
        }
        (Ror32, Ror32) | (Ror64, Ror64) => (
            inner_const.wrapping_add(outer_const) & bits_minus_1,
            outer_op,
        ),

        _ => return None,
    };

    Some((base, new_const, new_op))
}

enum Simplify {
    ToConst(u64),
    ToIdentity(ValueRef),
}

fn try_strength_reduce(op: Op, a: &Armlet, consts: &[Option<u64>]) -> Option<Simplify> {
    let arg0 = a.args[0];
    let arg1 = a.args[1];
    let c0 = if arg0.is_some() {
        consts.get(arg0.as_usize()).copied().flatten()
    } else {
        None
    };
    let c1 = if arg1.is_some() {
        consts.get(arg1.as_usize()).copied().flatten()
    } else {
        None
    };

    let all_ones_32: u64 = 0xFFFF_FFFF;
    let all_ones_64: u64 = !0;

    use Op::*;
    match op {
        Add32 | Add64 => {
            if c0 == Some(0) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(0) {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        Sub32 | Sub64 => {
            if c1 == Some(0) {
                return Some(Simplify::ToIdentity(arg0));
            }
            if arg0 == arg1 && arg0.is_some() {
                return Some(Simplify::ToConst(0));
            }
        }
        And32 => {
            if c0 == Some(0) || c1 == Some(0) {
                return Some(Simplify::ToConst(0));
            }
            if c0 == Some(all_ones_32) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(all_ones_32) {
                return Some(Simplify::ToIdentity(arg0));
            }
            if arg0 == arg1 && arg0.is_some() {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        And64 => {
            if c0 == Some(0) || c1 == Some(0) {
                return Some(Simplify::ToConst(0));
            }
            if c0 == Some(all_ones_64) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(all_ones_64) {
                return Some(Simplify::ToIdentity(arg0));
            }
            if arg0 == arg1 && arg0.is_some() {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        Or32 => {
            if c0 == Some(0) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(0) {
                return Some(Simplify::ToIdentity(arg0));
            }
            if c0 == Some(all_ones_32) || c1 == Some(all_ones_32) {
                return Some(Simplify::ToConst(all_ones_32));
            }
            if arg0 == arg1 && arg0.is_some() {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        Or64 => {
            if c0 == Some(0) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(0) {
                return Some(Simplify::ToIdentity(arg0));
            }
            if c0 == Some(all_ones_64) || c1 == Some(all_ones_64) {
                return Some(Simplify::ToConst(all_ones_64));
            }
            if arg0 == arg1 && arg0.is_some() {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        Eor32 | Eor64 => {
            if c0 == Some(0) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(0) {
                return Some(Simplify::ToIdentity(arg0));
            }
            if arg0 == arg1 && arg0.is_some() {
                return Some(Simplify::ToConst(0));
            }
        }
        Mul32 | Mul64 => {
            if c0 == Some(0) || c1 == Some(0) {
                return Some(Simplify::ToConst(0));
            }
            if c0 == Some(1) {
                return Some(Simplify::ToIdentity(arg1));
            }
            if c1 == Some(1) {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        Lsl32 | Lsl64 | Lsr32 | Lsr64 | Asr32 | Asr64 | Ror32 | Ror64 => {
            if c1 == Some(0) {
                return Some(Simplify::ToIdentity(arg0));
            }
        }
        _ => {}
    }
    None
}

struct Term {
    base: ValueRef,
    coeff: u64,
    has_mul: bool,
}

fn extract_term(block: &Block, vr: ValueRef, consts: &[Option<u64>], bits: u32) -> Term {
    let i = vr.as_usize();
    if i >= block.code.len() || vr.is_none() {
        return Term {
            base: vr,
            coeff: 1,
            has_mul: false,
        };
    }
    let a = &block.code[i];
    let get_c = |v: ValueRef| -> Option<u64> {
        if v.is_none() {
            None
        } else {
            consts.get(v.as_usize()).copied().flatten()
        }
    };
    match a.op {
        Op::Mul32 | Op::Mul64 if a.op.size_bits() == bits => {
            if let Some(c) = get_c(a.args[0]) {
                let inner = extract_term(block, a.args[1], consts, bits);
                Term {
                    base: inner.base,
                    coeff: inner.coeff.wrapping_mul(c),
                    has_mul: true,
                }
            } else if let Some(c) = get_c(a.args[1]) {
                let inner = extract_term(block, a.args[0], consts, bits);
                Term {
                    base: inner.base,
                    coeff: inner.coeff.wrapping_mul(c),
                    has_mul: true,
                }
            } else {
                Term {
                    base: vr,
                    coeff: 1,
                    has_mul: false,
                }
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
                Term {
                    base: vr,
                    coeff: 1,
                    has_mul: false,
                }
            }
        }
        Op::Identity => extract_term(block, a.args[0], consts, bits),
        _ => Term {
            base: vr,
            coeff: 1,
            has_mul: false,
        },
    }
}

fn try_mul_fold(add: &Armlet, block: &Block, consts: &[Option<u64>]) -> Option<(ValueRef, u64)> {
    let bits = add.op.size_bits();
    let mask: u64 = if bits >= 64 { !0 } else { (1u64 << bits) - 1 };
    let is_sub = matches!(add.op, Op::Sub32 | Op::Sub64);

    let lhs = extract_term(block, add.args[0], consts, bits);
    let rhs = extract_term(block, add.args[1], consts, bits);
    if lhs.base != rhs.base || lhs.base.is_none() {
        return None;
    }
    if !(lhs.has_mul || rhs.has_mul) {
        return None;
    }

    let combined = if is_sub {
        lhs.coeff.wrapping_sub(rhs.coeff)
    } else {
        lhs.coeff.wrapping_add(rhs.coeff)
    } & mask;

    if combined == 0 || combined == 1 {
        return None;
    }
    Some((lhs.base, combined))
}

fn try_fold(op: Op, a: &crate::ir::Armlet, consts: &[Option<u64>]) -> Option<u64> {
    let get = |v: ValueRef| -> Option<u64> {
        if v.is_none() {
            None
        } else {
            consts[v.as_usize()]
        }
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
        Or32 => (x as u32 | y_opt? as u32) as u64,
        Or64 => x | y_opt?,
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

        let add_node = &block.code[added.as_usize()];
        assert_eq!(
            add_node.op,
            Op::Identity,
            "Add should be rewritten to Identity"
        );

        let target = add_node.args[0];
        let mul_node = &block.code[target.as_usize()];
        assert_eq!(mul_node.op, Op::Mul64);
        assert_eq!(mul_node.args[0], a, "Mul operand[0] should be the base");

        let coeff_node = &block.code[mul_node.args[1].as_usize()];
        assert_eq!(coeff_node.op, Op::ConstU64);
        assert_eq!(
            coeff_node.imm, 13,
            "coefficient should be (3 << 2) + 1 = 13"
        );
    }

    #[test]
    fn mul_fold_skips_chain_without_mul() {
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

    #[allow(dead_code)]
    fn _silence_unused(_: RegSize) {}

    fn final_setx_source(block: &Block, set_idx: ValueRef) -> (Op, u64) {
        let set = &block.code[set_idx.as_usize()];
        assert_eq!(set.op, Op::SetX);
        let src = &block.code[set.args[0].as_usize()];
        (src.op, src.imm)
    }

    fn run_binop(op: Op, b_const: Option<u64>, ty: Ty) -> (Op, u64) {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let b = match b_const {
            Some(v) if ty == Ty::U64 => em.const_u64(v),
            Some(v) => em.const_u32(v as u32),
            None => em.get_x(1),
        };
        let r = em.push(Armlet::new(op, ty).with_args(&[a, b]));
        let set_vr = em.push(Armlet::new(Op::SetX, Ty::Void).with_args(&[r]).with_imm(2));
        optimize(&mut block);
        final_setx_source(&block, set_vr)
    }

    #[test]
    fn strength_add_zero_propagates_to_setx() {
        let (op, _) = run_binop(Op::Add64, Some(0), Ty::U64);
        assert_eq!(op, Op::GetX, "x + 0 should reduce to x");
    }

    #[test]
    fn strength_sub_zero_propagates_to_setx() {
        let (op, _) = run_binop(Op::Sub64, Some(0), Ty::U64);
        assert_eq!(op, Op::GetX, "x - 0 should reduce to x");
    }

    #[test]
    fn strength_mul_zero_becomes_const_zero() {
        let (op, imm) = run_binop(Op::Mul64, Some(0), Ty::U64);
        assert_eq!(op, Op::ConstU64);
        assert_eq!(imm, 0, "x * 0 should reduce to 0");
    }

    #[test]
    fn strength_mul_one_propagates_to_setx() {
        let (op, _) = run_binop(Op::Mul64, Some(1), Ty::U64);
        assert_eq!(op, Op::GetX, "x * 1 should reduce to x");
    }

    #[test]
    fn strength_and_zero_becomes_const_zero() {
        let (op, imm) = run_binop(Op::And64, Some(0), Ty::U64);
        assert_eq!(op, Op::ConstU64);
        assert_eq!(imm, 0);
    }

    #[test]
    fn strength_and_all_ones_propagates_to_setx() {
        let (op, _) = run_binop(Op::And64, Some(!0), Ty::U64);
        assert_eq!(op, Op::GetX, "x & ~0 should reduce to x");
    }

    #[test]
    fn strength_or_zero_propagates_to_setx() {
        let (op, _) = run_binop(Op::Or64, Some(0), Ty::U64);
        assert_eq!(op, Op::GetX, "x | 0 should reduce to x");
    }

    #[test]
    fn strength_or_all_ones_becomes_const() {
        let (op, imm) = run_binop(Op::Or64, Some(!0), Ty::U64);
        assert_eq!(op, Op::ConstU64);
        assert_eq!(imm, !0);
    }

    #[test]
    fn strength_eor_zero_propagates_to_setx() {
        let (op, _) = run_binop(Op::Eor64, Some(0), Ty::U64);
        assert_eq!(op, Op::GetX, "x ^ 0 should reduce to x");
    }

    #[test]
    fn strength_lsl_zero_propagates_to_setx() {
        let (op, _) = run_binop(Op::Lsl64, Some(0), Ty::U64);
        assert_eq!(op, Op::GetX, "x << 0 should reduce to x");
    }

    #[test]
    fn strength_xor_self_becomes_zero() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let r = em.push(Armlet::new(Op::Eor64, Ty::U64).with_args(&[a, a]));
        let set_vr = em.push(Armlet::new(Op::SetX, Ty::Void).with_args(&[r]).with_imm(1));
        optimize(&mut block);
        let (op, imm) = final_setx_source(&block, set_vr);
        assert_eq!(op, Op::ConstU64);
        assert_eq!(imm, 0);
    }

    #[test]
    fn strength_sub_self_becomes_zero() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let r = em.push(Armlet::new(Op::Sub64, Ty::U64).with_args(&[a, a]));
        let set_vr = em.push(Armlet::new(Op::SetX, Ty::Void).with_args(&[r]).with_imm(1));
        optimize(&mut block);
        let (op, imm) = final_setx_source(&block, set_vr);
        assert_eq!(op, Op::ConstU64);
        assert_eq!(imm, 0);
    }

    #[test]
    fn strength_and_self_propagates_to_setx() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let r = em.push(Armlet::new(Op::And64, Ty::U64).with_args(&[a, a]));
        let set_vr = em.push(Armlet::new(Op::SetX, Ty::Void).with_args(&[r]).with_imm(1));
        optimize(&mut block);
        let (op, _) = final_setx_source(&block, set_vr);
        assert_eq!(op, Op::GetX, "x & x should reduce to x");
    }

    fn outer_binop_const(block: &Block, set_vr: ValueRef) -> (Op, u64) {
        let set = &block.code[set_vr.as_usize()];
        let outer = &block.code[set.args[0].as_usize()];
        let c = &block.code[outer.args[1].as_usize()];
        (outer.op, c.imm)
    }

    #[test]
    fn combine_add_add_collapses_constants() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c1 = em.const_u64(5);
        let inner = em.push(Armlet::new(Op::Add64, Ty::U64).with_args(&[a, c1]));
        let c2 = em.const_u64(3);
        let outer = em.push(Armlet::new(Op::Add64, Ty::U64).with_args(&[inner, c2]));
        let set_vr = em.push(
            Armlet::new(Op::SetX, Ty::Void)
                .with_args(&[outer])
                .with_imm(1),
        );
        optimize(&mut block);
        let (op, imm) = outer_binop_const(&block, set_vr);
        assert_eq!(op, Op::Add64);
        assert_eq!(imm, 8, "(x + 5) + 3 → x + 8");
    }

    #[test]
    fn combine_sub_sub_collapses_to_add_with_negated_const() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c1 = em.const_u64(5);
        let inner = em.push(Armlet::new(Op::Sub64, Ty::U64).with_args(&[a, c1]));
        let c2 = em.const_u64(3);
        let outer = em.push(Armlet::new(Op::Sub64, Ty::U64).with_args(&[inner, c2]));
        let set_vr = em.push(
            Armlet::new(Op::SetX, Ty::Void)
                .with_args(&[outer])
                .with_imm(1),
        );
        optimize(&mut block);
        let (op, imm) = outer_binop_const(&block, set_vr);
        assert_eq!(op, Op::Add64, "Sub-Sub canonicalises to Add");
        assert_eq!(imm as i64, -8, "(x - 5) - 3 → x + (-8)");
    }

    #[test]
    fn combine_add_sub_crossover() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c1 = em.const_u64(10);
        let inner = em.push(Armlet::new(Op::Sub64, Ty::U64).with_args(&[a, c1]));
        let c2 = em.const_u64(3);
        let outer = em.push(Armlet::new(Op::Add64, Ty::U64).with_args(&[inner, c2]));
        let set_vr = em.push(
            Armlet::new(Op::SetX, Ty::Void)
                .with_args(&[outer])
                .with_imm(1),
        );
        optimize(&mut block);
        let (op, imm) = outer_binop_const(&block, set_vr);
        assert_eq!(op, Op::Add64);
        assert_eq!(imm as i64, -7, "(x - 10) + 3 → x + (-7)");
    }

    #[test]
    fn combine_shl_shl_sums_amounts() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c1 = em.const_u64(3);
        let inner = em.push(Armlet::new(Op::Lsl64, Ty::U64).with_args(&[a, c1]));
        let c2 = em.const_u64(4);
        let outer = em.push(Armlet::new(Op::Lsl64, Ty::U64).with_args(&[inner, c2]));
        let set_vr = em.push(
            Armlet::new(Op::SetX, Ty::Void)
                .with_args(&[outer])
                .with_imm(1),
        );
        optimize(&mut block);
        let (op, imm) = outer_binop_const(&block, set_vr);
        assert_eq!(op, Op::Lsl64);
        assert_eq!(imm, 7, "(x << 3) << 4 → x << 7");
    }

    #[test]
    fn terminal_cbz_with_zero_value_becomes_direct_branch_taken() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let v = em.const_u64(0);
        em.push(
            Armlet::new(Op::CbZ, Ty::Void)
                .with_args(&[v])
                .with_imm(0x2000),
        );
        block.terminal = Terminal::CompareBranchZero {
            value: v,
            inverse: false,
            taken_pc: 0x2000,
            not_taken_pc: 0x1008,
        };
        optimize(&mut block);
        match block.terminal {
            Terminal::DirectBranch { target_pc, .. } => assert_eq!(target_pc, 0x2000),
            other => panic!("expected DirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn terminal_cbnz_with_nonzero_value_becomes_direct_branch_taken() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let v = em.const_u64(42);
        em.push(
            Armlet::new(Op::CbNz, Ty::Void)
                .with_args(&[v])
                .with_imm(0x2000),
        );
        block.terminal = Terminal::CompareBranchZero {
            value: v,
            inverse: true,
            taken_pc: 0x2000,
            not_taken_pc: 0x1008,
        };
        optimize(&mut block);
        match block.terminal {
            Terminal::DirectBranch { target_pc, .. } => assert_eq!(target_pc, 0x2000),
            other => panic!("expected DirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn terminal_tbz_with_clear_bit_becomes_direct_branch_taken() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let v = em.const_u64(0x10);
        em.push(
            Armlet::new(Op::TbZ, Ty::Void)
                .with_args(&[v])
                .with_imm(0x2000),
        );
        block.terminal = Terminal::TestBranchBit {
            value: v,
            bit: 0,
            inverse: false,
            taken_pc: 0x2000,
            not_taken_pc: 0x1008,
        };
        optimize(&mut block);
        match block.terminal {
            Terminal::DirectBranch { target_pc, .. } => assert_eq!(target_pc, 0x2000),
            other => panic!("expected DirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn terminal_branchcond_always_collapses() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let nz = em.get_nzcv();
        em.push(
            Armlet::new(Op::BranchCond, Ty::Void)
                .with_args(&[nz])
                .with_imm((0x2000u64 << 8) | (Cond::AL as u64)),
        );
        block.terminal = Terminal::ConditionalBranch {
            cond_nzcv: nz,
            cond_code: Cond::AL as u8,
            taken_pc: 0x2000,
            not_taken_pc: 0x1008,
        };
        optimize(&mut block);
        match block.terminal {
            Terminal::DirectBranch { target_pc, .. } => assert_eq!(target_pc, 0x2000),
            other => panic!("AL cond should collapse to DirectBranch, got {:?}", other),
        }
    }

    #[test]
    fn dse_drops_overwritten_setx() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let v1 = em.const_u64(1);
        em.set_x(0, v1);
        let v2 = em.const_u64(2);
        em.set_x(0, v2);
        optimize(&mut block);
        let setx_count = block
            .iter_live()
            .filter(|(_, a)| matches!(a.op, Op::SetX))
            .count();
        assert_eq!(setx_count, 1, "DSE should drop the first SetX");
    }

    #[test]
    fn dse_preserves_setx_when_observed_by_store() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let v1 = em.const_u64(1);
        em.set_x(0, v1);
        let addr = em.const_u64(0x4000);
        let val = em.const_u64(0x99);
        em.store(addr, val, 8);
        let v2 = em.const_u64(2);
        em.set_x(0, v2);
        optimize(&mut block);
        let setx_count = block
            .iter_live()
            .filter(|(_, a)| matches!(a.op, Op::SetX))
            .count();
        assert_eq!(
            setx_count, 2,
            "store callback may observe ctx.x; keep first SetX"
        );
    }

    #[test]
    fn dse_preserves_setx_when_consumed_by_getx() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let v1 = em.const_u64(1);
        em.set_x(0, v1);
        let read_back = em.get_x(0);
        em.set_x(1, read_back);
        let v2 = em.const_u64(2);
        em.set_x(0, v2);
        optimize(&mut block);
        let setx1 = block
            .iter_live()
            .find(|(_, a)| matches!(a.op, Op::SetX) && a.imm == 1)
            .expect("SetX(1) should remain");
        let src = &block.code[setx1.1.args[0].as_usize()];
        assert_eq!(src.op, Op::ConstU64);
        assert_eq!(src.imm, 1);
    }

    #[test]
    fn combine_and_and_intersects_masks() {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let a = em.get_x(0);
        let c1 = em.const_u64(0xFF00);
        let inner = em.push(Armlet::new(Op::And64, Ty::U64).with_args(&[a, c1]));
        let c2 = em.const_u64(0x0FF0);
        let outer = em.push(Armlet::new(Op::And64, Ty::U64).with_args(&[inner, c2]));
        let set_vr = em.push(
            Armlet::new(Op::SetX, Ty::Void)
                .with_args(&[outer])
                .with_imm(1),
        );
        optimize(&mut block);
        let (op, imm) = outer_binop_const(&block, set_vr);
        assert_eq!(op, Op::And64);
        assert_eq!(imm, 0x0F00, "(x & 0xFF00) & 0x0FF0 → x & 0x0F00");
    }
}
