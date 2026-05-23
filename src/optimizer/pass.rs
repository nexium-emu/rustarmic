use crate::arch::NUM_GPRS;
use crate::ir::{Block, Op, Ty, ValueRef};

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
    let n = block.code.len();
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
            | Op::SubsFlags32 | Op::SubsFlags64 => {
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
