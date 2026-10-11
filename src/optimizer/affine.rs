use crate::ir::{Armlet, Block, Op, Ty, ValueRef};

pub(super) mod cost;

#[derive(Clone, Copy, Debug)]
pub(super) struct ByteMap {
    pub source: ValueRef,
    pub matrix: u64,
    pub addend: u8,
    pub q: bool,
}

fn matrix(rows: impl Fn(u32) -> u8) -> u64 {
    (0..8).fold(0, |value, bit| value | ((rows(bit) as u64) << ((7 - bit) * 8)))
}

fn diagonal(mask: u8) -> u64 {
    matrix(|bit| mask & (1 << bit))
}

fn linear(matrix: u64, value: u8) -> u8 {
    (0..8).fold(0, |result, bit| {
        let row = (matrix >> ((7 - bit) * 8)) as u8;
        result | (((row & value).count_ones() as u8 & 1) << bit)
    })
}

fn uniform_byte(value: ValueRef, q: bool, vectors: &[Option<[u64; 2]>]) -> Option<u8> {
    let halves = vectors.get(value.as_usize()).copied().flatten()?;
    let byte = halves[0] as u8;
    let repeated = u64::from_le_bytes([byte; 8]);
    (halves[0] == repeated && (!q || halves[1] == repeated)).then_some(byte)
}

pub(super) fn recognize(
    a: &Armlet,
    block: &Block,
    consts: &[Option<u64>],
    vectors: &[Option<[u64; 2]>],
) -> Option<ByteMap> {
    if a.ty != Ty::U128 {
        return None;
    }
    let q = a.imm & 1 != 0;
    let mut source = a.args[0];
    let (matrix, addend) = match a.op {
        Op::VecBitAffine if a.imm & !0xff01 == 0 => {
            if block.code.get(a.args[1].as_usize())?.op != Op::ConstU64 {
                return None;
            }
            (consts.get(a.args[1].as_usize()).copied().flatten()?, (a.imm >> 8) as u8)
        }
        Op::VecNot if a.imm <= 1 => (diagonal(0xff), 0xff),
        Op::VecShlImm8 if a.imm >> 1 < 8 => {
            let shift = (a.imm >> 1) as u32;
            (matrix(|bit| if bit >= shift { 1 << (bit - shift) } else { 0 }), 0)
        }
        Op::VecUshrImm8 if a.imm >> 1 <= 8 => {
            let shift = (a.imm >> 1) as u32;
            (matrix(|bit| if bit + shift < 8 { 1 << (bit + shift) } else { 0 }), 0)
        }
        Op::VecSshrImm8 if a.imm >> 1 <= 8 => {
            let shift = (a.imm >> 1) as u32;
            (matrix(|bit| 1 << (bit + shift).min(7)), 0)
        }
        Op::VecAdd8 if a.imm <= 1 && a.args[0] == a.args[1] => {
            (matrix(|bit| if bit > 0 { 1 << (bit - 1) } else { 0 }), 0)
        }
        Op::VecMul8 if a.imm <= 1 => {
            let multiplier = if let Some(value) = uniform_byte(a.args[1], q, vectors) {
                value
            } else {
                source = a.args[1];
                uniform_byte(a.args[0], q, vectors)?
            };
            if !multiplier.is_power_of_two() {
                return None;
            }
            let shift = multiplier.trailing_zeros();
            (matrix(|bit| if bit >= shift { 1 << (bit - shift) } else { 0 }), 0)
        }
        Op::VecUshl8 | Op::VecSshl8 if a.imm <= 1 => {
            let count = uniform_byte(a.args[1], q, vectors)? as i8 as i32;
            let rows = |bit: u32| {
                if count >= 0 {
                    if count < 8 && bit >= count as u32 { 1 << (bit - count as u32) } else { 0 }
                } else if a.op == Op::VecSshl8 {
                    1 << (bit + count.unsigned_abs()).min(7)
                } else if bit + count.unsigned_abs() < 8 {
                    1 << (bit + count.unsigned_abs())
                } else {
                    0
                }
            };
            (matrix(rows), 0)
        }
        Op::VecAnd | Op::VecOrr | Op::VecEor if a.imm <= 1 => {
            let mask = if let Some(mask) = uniform_byte(a.args[1], q, vectors) {
                mask
            } else {
                source = a.args[1];
                uniform_byte(a.args[0], q, vectors)?
            };
            match a.op {
                Op::VecAnd => (diagonal(mask), 0),
                Op::VecOrr => (diagonal(!mask), mask),
                Op::VecEor => (diagonal(0xff), mask),
                _ => unreachable!(),
            }
        }
        Op::VecBic | Op::VecOrn if a.imm <= 1 => {
            let mask = uniform_byte(a.args[1], q, vectors)?;
            if a.op == Op::VecBic { (diagonal(!mask), 0) } else { (diagonal(mask), !mask) }
        }
        _ => return None,
    };
    if source.is_none() || block.code.get(source.as_usize())?.ty != Ty::U128 {
        return None;
    }
    Some(ByteMap { source, matrix, addend, q })
}

pub(super) fn compose(outer: ByteMap, inner: ByteMap) -> Option<ByteMap> {
    if !inner.q && outer.q && outer.addend != 0 {
        return None;
    }
    let combined = matrix(|output| {
        let outer_row = (outer.matrix >> ((7 - output) * 8)) as u8;
        (0..8).fold(0, |row, input| {
            if outer_row & (1 << input) != 0 {
                row ^ (inner.matrix >> ((7 - input) * 8)) as u8
            } else {
                row
            }
        })
    });
    Some(ByteMap {
        source: inner.source,
        matrix: combined,
        addend: linear(outer.matrix, inner.addend) ^ outer.addend,
        q: outer.q && inner.q,
    })
}

pub(super) fn vector_constant(
    a: &Armlet,
    block: &Block,
    consts: &[Option<u64>],
    vectors: &[Option<[u64; 2]>],
    recognized: Option<ByteMap>,
) -> Option<[u64; 2]> {
    if a.ty != Ty::U128 {
        return None;
    }
    if a.op == Op::Identity {
        return vectors.get(a.args[0].as_usize()).copied().flatten();
    }
    if a.op == Op::VecBuildQ {
        let half = |value: ValueRef| {
            if block.code.get(value.as_usize())?.ty != Ty::U64 {
                return None;
            }
            consts.get(value.as_usize()).copied().flatten()
        };
        return Some([half(a.args[0])?, half(a.args[1])?]);
    }
    if a.op == Op::VecDupGpr8 && a.imm <= 1 {
        if !matches!(block.code.get(a.args[0].as_usize())?.ty, Ty::U32 | Ty::U64) {
            return None;
        }
        let byte = consts.get(a.args[0].as_usize()).copied().flatten()? as u8;
        let repeated = u64::from_le_bytes([byte; 8]);
        return Some([repeated, if a.imm & 1 != 0 { repeated } else { 0 }]);
    }
    let map = recognized?;
    let halves = vectors.get(map.source.as_usize()).copied().flatten()?;
    let convert = |value: u64| {
        u64::from_le_bytes(value.to_le_bytes().map(|byte| linear(map.matrix, byte) ^ map.addend))
    };
    Some([convert(halves[0]), if map.q { convert(halves[1]) } else { 0 }])
}

#[cfg(test)]
mod tests;
