use super::ByteMap;
use crate::backend::gfni::affine_cost;
use crate::ir::{Armlet, Block, Op, Terminal, Ty, ValueRef};

fn count_reference(block: &Block, uses: &mut [u8], mut value: ValueRef) {
    while value.is_some() {
        let Some(count) = uses.get_mut(value.as_usize()) else { break };
        *count = count.saturating_add(1);
        let node = &block.code[value.as_usize()];
        if node.op != Op::Identity {
            break;
        }
        let next = node.args[0];
        if next.is_none() || next.as_usize() >= value.as_usize() {
            break;
        }
        value = next;
    }
}

pub(in crate::optimizer) fn seed_uses(block: &Block, uses: &mut [u8]) {
    for (_, node) in block.iter_live() {
        for value in node.args {
            count_reference(block, uses, value);
        }
    }
    let values = match block.terminal {
        Terminal::ConditionalBranch { cond_nzcv, .. } => [cond_nzcv, ValueRef::NONE],
        Terminal::CompareBranchZero { value, .. } | Terminal::TestBranchBit { value, .. } => [value, ValueRef::NONE],
        Terminal::IndirectBranch { target, .. } => [target, ValueRef::NONE],
        _ => [ValueRef::NONE; 2],
    };
    for value in values {
        count_reference(block, uses, value);
    }
}

fn core(node: &Armlet, block: &Block, gfni: bool) -> Option<u32> {
    let q = node.imm & 1 != 0;
    let clear = u32::from(!q);
    let fixed_transport = if crate::backend::regalloc::SPILL_XMMS.contains(&0) { 0 } else { 2 };
    Some(match node.op {
        Op::VecBitAffine => {
            let literal = block.code.get(node.args[1].as_usize())?;
            if literal.op != Op::ConstU64 { return None }
            affine_cost(literal.imm, (node.imm >> 8) as u8, q, gfni) + fixed_transport
        }
        Op::VecNot => 2 + clear,
        Op::VecAnd | Op::VecOrr | Op::VecEor | Op::VecBic | Op::VecOrn | Op::VecAdd8 => 1 + clear,
        Op::VecMul8 => 10 + clear,
        Op::VecShlImm8 | Op::VecUshrImm8 => if gfni { 4 + clear + fixed_transport } else { 5 + clear },
        Op::VecSshrImm8 => if gfni { 4 + clear + fixed_transport } else { (if q { 6 } else { 3 }) + clear },
        Op::VecUshl8 => (if q { 16 } else { 8 }) * 9 + 1,
        Op::VecSshl8 => (if q { 16 } else { 8 }) * 10 + 1,
        Op::GetV if node.ty == Ty::U128 => 1,
        _ => return None,
    })
}

fn dead_constant(value: ValueRef, block: &Block, uses: &[u8], vectors: &[Option<[u64; 2]>]) -> u32 {
    if value.is_none() || uses.get(value.as_usize()).copied() != Some(1) {
        return 0;
    }
    let node = &block.code[value.as_usize()];
    match node.op {
        Op::ConstU32 | Op::ConstU64 => 2,
        Op::VecBuildQ if vectors.get(value.as_usize()).copied().flatten().is_some() => {
            4 + dead_constant(node.args[0], block, uses, vectors)
                + dead_constant(node.args[1], block, uses, vectors)
        }
        Op::VecDupGpr8 if vectors.get(value.as_usize()).copied().flatten().is_some() => {
            4 + u32::from(node.imm & 1 == 0) + dead_constant(node.args[0], block, uses, vectors)
        }
        _ => 0,
    }
}

fn dead_operands(node: &Armlet, source: ValueRef, block: &Block, uses: &[u8], vectors: &[Option<[u64; 2]>]) -> u32 {
    let mut credit = 0;
    for (index, value) in node.args.into_iter().enumerate() {
        if value != source && !node.args[..index].contains(&value) {
            credit += dead_constant(value, block, uses, vectors);
        }
    }
    credit
}

pub(in crate::optimizer) fn worthwhile(
    node: &Armlet,
    block: &Block,
    uses: &[u8],
    vectors: &[Option<[u64; 2]>],
    original: ByteMap,
    replacement: ByteMap,
    inner: Option<ByteMap>,
) -> bool {
    [false, true].into_iter().all(|gfni| {
        let Some(mut old) = core(node, block, gfni) else { return false };
        old += dead_operands(node, original.source, block, uses, vectors);
        if let Some(inner) = inner {
            if uses.get(original.source.as_usize()).copied() == Some(1) {
                let predecessor = &block.code[original.source.as_usize()];
                let Some(removed) = core(predecessor, block, gfni) else { return false };
                old += removed + dead_operands(predecessor, inner.source, block, uses, vectors);
            }
        }
        affine_cost(replacement.matrix, replacement.addend, replacement.q, gfni) + 4 <= old
    })
}
