use super::*;
use crate::optimizer::{Scratch, optimize, optimize_with_scratch};
use crate::ir::{IrEmitter, Terminal};

fn applied(value: u8, matrix: u64, addend: u8) -> u8 {
    let mut result = addend;
    for output in 0..8 {
        for input in 0..8 {
            if (matrix >> ((7 - output) * 8 + input)) & 1 != 0 {
                result ^= ((value >> input) & 1) << output;
            }
        }
    }
    result
}

fn maps(block: &Block) -> usize {
    let forward: Vec<_> = block.iter_live().map(|(value, _)| value).collect();
    let reverse: Vec<_> = block.iter_live_rev().map(|(value, _)| value).collect();
    assert_eq!(forward, reverse.into_iter().rev().collect::<Vec<_>>());
    for pair in forward.windows(2) {
        assert_eq!(block.next_of(pair[0]), Some(pair[1]));
        assert_eq!(block.prev_of(pair[1]), Some(pair[0]));
    }
    block.iter_live().filter(|(_, node)| node.op == Op::VecBitAffine).count()
}

fn stored(block: &Block, register: u64) -> &Armlet {
    let sink = block.iter_live().filter(|(_, node)| node.op == Op::SetV && node.imm == register).last().unwrap().1;
    &block.code[sink.args[0].as_usize()]
}

fn literal_map(block: &Block, register: u64) -> (ValueRef, u64, u8, bool) {
    let value = stored(block, register);
    assert_eq!(value.op, Op::VecBitAffine);
    let matrix = &block.code[value.args[1].as_usize()];
    assert_eq!(matrix.op, Op::ConstU64);
    (value.args[0], matrix.imm, (value.imm >> 8) as u8, value.imm & 1 != 0)
}

#[derive(Clone, Copy, Debug)]
enum Shift {
    Left(u32),
    Unsigned(u32),
    Signed(u32),
}

impl Shift {
    fn emit(self, em: &mut IrEmitter<'_>, value: ValueRef, q: bool) -> ValueRef {
        match self {
            Self::Left(amount) => em.vec_shl_imm(value, 0, amount, q),
            Self::Unsigned(amount) => em.vec_ushr_imm(value, 0, amount, q),
            Self::Signed(amount) => em.vec_sshr_imm(value, 0, amount, q),
        }
    }

    fn apply(self, value: u8) -> u8 {
        match self {
            Self::Left(amount) => value << amount,
            Self::Unsigned(amount) => (value as u16 >> amount) as u8,
            Self::Signed(amount) => (value as i8 as i16 >> amount) as u8,
        }
    }
}

#[test]
fn every_byte_shift_pair_is_exact_and_preserves_q0_zeroing() {
    let shifts: Vec<_> = (0..8).map(Shift::Left)
        .chain((1..=8).map(Shift::Unsigned))
        .chain((1..=8).map(Shift::Signed)).collect();
    for &first in &shifts {
        for &second in &shifts {
            for q0 in [false, true] {
                for q1 in [false, true] {
                    let mut block = Block::new(0x1000);
                    let mut em = IrEmitter::new(&mut block, 0x1000);
                    let input = em.get_v_q(0);
                    let middle = first.emit(&mut em, input, q0);
                    let result = second.emit(&mut em, middle, q1);
                    em.set_v_q(1, result);
                    let consts = vec![None; block.code.len()];
                    let vectors = vec![None; block.code.len()];
                    let first_map = recognize(block.get(middle), &block, &consts, &vectors).unwrap();
                    let second_map = recognize(block.get(result), &block, &consts, &vectors).unwrap();
                    let proposed = compose(second_map, first_map).unwrap();
                    let result_op = block.get(result).op;
                    optimize(&mut block);
                    assert!(maps(&block) <= 1);
                    if maps(&block) == 1 {
                        let (source, matrix, addend, q) = literal_map(&block, 1);
                        assert_eq!((source, matrix, addend, q), (input, proposed.matrix, proposed.addend, q0 && q1));
                    } else {
                        assert_eq!(stored(&block, 1).op, result_op);
                    }
                    for value in 0..=255u8 {
                        assert_eq!(applied(value, proposed.matrix, proposed.addend), second.apply(first.apply(value)), "{first:?} {second:?} {q0} {q1} {value}");
                    }
                }
            }
        }
    }
}

#[test]
fn every_uniform_byte_mask_and_boolean_kind_composes_exactly() {
    for mask in 0..=255u8 {
        for op in [Op::VecAnd, Op::VecOrr, Op::VecEor, Op::VecBic, Op::VecOrn] {
            for q in [false, true] {
                for constant_left in [false, true] {
                    let mut block = Block::new(0x1000);
                    let mut em = IrEmitter::new(&mut block, 0x1000);
                    let input = em.get_v_q(0);
                    let middle = em.vec_not(input, q);
                    let lo = em.const_u64(u64::from_le_bytes([mask; 8]));
                    let hi = em.const_u64(if q { u64::from_le_bytes([mask; 8]) } else { 0x1937_9bdf_2468_ace0 });
                    let constant = em.vec_build_q(lo, hi);
                    let args = if constant_left { [constant, middle] } else { [middle, constant] };
                    let result = em.push(Armlet::new(op, Ty::U128).with_args(&args).with_imm(u64::from(q)));
                    em.set_v_q(1, result);
                    let consts = vec![None; block.code.len()];
                    let mut vectors = vec![None; block.code.len()];
                    vectors[constant.as_usize()] = Some([block.get(lo).imm, block.get(hi).imm]);
                    let first_map = recognize(block.get(middle), &block, &consts, &vectors).unwrap();
                    let proposed = recognize(block.get(result), &block, &consts, &vectors).and_then(|outer| compose(outer, first_map));
                    optimize(&mut block);
                    if constant_left && matches!(op, Op::VecBic | Op::VecOrn) {
                        assert!(proposed.is_none());
                        assert_eq!(stored(&block, 1).op, op);
                        assert_eq!(maps(&block), 0);
                        continue;
                    }
                    let proposed = proposed.unwrap();
                    if maps(&block) == 1 {
                        let (source, matrix, addend, output_q) = literal_map(&block, 1);
                        assert_eq!((source, matrix, addend, output_q), (input, proposed.matrix, proposed.addend, q));
                    } else {
                        assert_eq!(maps(&block), 0);
                        assert_eq!(stored(&block, 1).op, op);
                    }
                    for value in 0..=255u8 {
                        let expected = match op {
                            Op::VecAnd => !value & mask,
                            Op::VecOrr => !value | mask,
                            Op::VecEor => !value ^ mask,
                            Op::VecBic => !value & !mask,
                            Op::VecOrn => !value | !mask,
                            _ => unreachable!(),
                        };
                        assert_eq!(applied(value, proposed.matrix, proposed.addend), expected, "{op:?} {mask} {q} {constant_left} {value}");
                    }
                }
            }
        }
    }
}

#[test]
fn arbitrary_affine_matrices_and_offsets_compose_exactly() {
    let mut random = 0x1937_9bdf_2468_ace0u64;
    for _ in 0..512 {
        let mut next = || {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            random
        };
        let first = ByteMap { source: ValueRef::new(1), matrix: next(), addend: next() as u8, q: true };
        let second = ByteMap { source: ValueRef::new(2), matrix: next(), addend: next() as u8, q: true };
        let combined = compose(second, first).unwrap();
        assert_eq!(combined.source, first.source);
        for value in 0..=255u8 {
            assert_eq!(applied(value, combined.matrix, combined.addend), applied(applied(value, first.matrix, first.addend), second.matrix, second.addend));
        }
    }
}

#[test]
fn q0_then_nonzero_q1_offset_keeps_the_width_barrier() {
    let mut block = Block::new(0x1000);
    let mut em = IrEmitter::new(&mut block, 0x1000);
    let input = em.get_v_q(0);
    let narrow = em.vec_sshr_imm(input, 0, 8, false);
    let result = em.vec_not(narrow, true);
    em.set_v_q(1, result);
    optimize(&mut block);
    assert_eq!(maps(&block), 0);
    assert_eq!(stored(&block, 1).op, Op::VecNot);
    assert_eq!(stored(&block, 1).args[0], narrow);
    let inner = ByteMap { source: input, matrix: matrix(|bit| 1 << (bit + 8).min(7)), addend: 0, q: false };
    let outer = ByteMap { source: narrow, matrix: diagonal(0xff), addend: 0xff, q: true };
    assert!(compose(outer, inner).is_none());
    let cleared = compose(ByteMap { addend: 0, ..outer }, inner).unwrap();
    assert_eq!((cleared.source, cleared.q), (input, false));
}

#[test]
fn forwarded_uniform_counts_stop_at_every_barrier() {
    for barrier in 0..5 {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let count = em.const_u64(3);
        let counts = em.vec_dup_gpr(count, 0, true);
        em.set_v_q(1, counts);
        match barrier {
            1 => {
                let scalar = em.const_u32(5);
                em.set_v_s(1, scalar);
            }
            2 => {
                let address = em.const_u64(0x2000);
                let value = em.const_u64(0);
                em.store(address, value, 8);
            }
            3 => {
                let scalar = em.const_u64(5);
                em.set_v_d(1, scalar);
            }
            4 => {
                let address = em.const_u64(0x2000);
                em.load(address, 8);
            }
            _ => {}
        }
        let reread = em.get_v_q(1);
        let result = em.vec_variable_shift(input, reread, 0, true, false);
        em.set_v_q(2, result);
        optimize(&mut block);
        if barrier == 0 {
            let (source, matrix, addend, q) = literal_map(&block, 2);
            assert_eq!((source, addend, q), (input, 0, true));
            for value in 0..=255u8 {
                assert_eq!(applied(value, matrix, addend), value << 3);
            }
        } else {
            assert_eq!(maps(&block), 0, "barrier {barrier}");
            assert_eq!(stored(&block, 2).op, Op::VecUshl8, "barrier {barrier}");
            assert_eq!(stored(&block, 2).args[1], reread, "barrier {barrier}");
        }
    }
}

#[test]
fn q1_uniform_counts_require_both_halves() {
    for (low, high, q, fused) in [
        (0x0303_0303_0303_0303u64, 0x0303_0303_0303_0303u64, true, true),
        (0x0303_0303_0303_0303, 0x0505_0505_0505_0505, true, false),
        (0x0303_0303_0303_0303, 0x0505_0505_0505_0505, false, true),
    ] {
        for signed in [false, true] {
            let mut block = Block::new(0x1000);
            let mut em = IrEmitter::new(&mut block, 0x1000);
            let input = em.get_v_q(0);
            let lo = em.const_u64(low);
            let hi = em.const_u64(high);
            let counts = em.vec_build_q(lo, hi);
            let result = em.vec_variable_shift(input, counts, 0, q, signed);
            em.set_v_q(1, result);
            optimize(&mut block);
            assert_eq!(maps(&block), usize::from(fused), "{low:#x} {high:#x} {q} {signed}");
            if !fused {
                assert_eq!(stored(&block, 1).op, if signed { Op::VecSshl8 } else { Op::VecUshl8 });
            }
        }
    }
}

#[test]
fn heterogeneous_active_constants_are_not_uniform_affine_maps() {
    for (lo, hi, q) in [
        (0x5555_5555_5555_55aa, 0x5555_5555_5555_5555, false),
        (0x5555_5555_5555_5555, 0xaaaa_aaaa_aaaa_aaaa, true),
    ] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let middle = em.vec_not(input, q);
        let low = em.const_u64(lo);
        let high = em.const_u64(hi);
        let mask = em.vec_build_q(low, high);
        let result = em.vec_and(middle, mask, q);
        em.set_v_q(1, result);
        optimize(&mut block);
        assert_eq!(maps(&block), 0);
        assert_eq!(stored(&block, 1).op, Op::VecAnd);
    }
}

#[test]
fn wide_variable_carry_and_float_operations_remain_barriers() {
    for op in [Op::VecShlImm16, Op::VecUshrImm32, Op::VecSshrImm64, Op::VecUshl8, Op::VecAdd8, Op::VecSub8, Op::VecMul8, Op::VecFAdd_S] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let other = em.get_v_q(2);
        let first = em.vec_not(input, true);
        let middle = em.push(Armlet::new(op, Ty::U128).with_args(&[first, other]).with_imm(1));
        let result = em.vec_not(middle, true);
        em.set_v_q(1, result);
        optimize(&mut block);
        assert_eq!(maps(&block), 0, "{op:?}");
        assert_eq!(block.code[middle.as_usize()].op, op);
    }
}

#[test]
fn isolated_byte_operations_are_not_rewritten() {
    let mut block = Block::new(0x1000);
    let mut em = IrEmitter::new(&mut block, 0x1000);
    let input = em.get_v_q(0);
    let result = em.vec_not(input, true);
    em.set_v_q(1, result);
    optimize(&mut block);
    assert_eq!(maps(&block), 0);
    assert_eq!(stored(&block, 1).op, Op::VecNot);
}

#[test]
fn full_q_domain_forwarding_preserves_all_architectural_writes_and_cycles() {
    let mut block = Block::new(0x1000);
    let mut em = IrEmitter::new(&mut block, 0x1000);
    let input = em.get_v_q(0);
    let first = em.vec_bit_affine(input, 0, 0, true);
    em.set_v_q(1, first);
    em.current_pc = 0x1004;
    let reread = em.get_v_q(1);
    let result = em.vec_bit_affine(reread, 0x1357_9bdf_2468_ace0, 0, true);
    em.set_v_q(2, result);
    block.cycles = 2;
    block.end_pc = 0x1008;
    block.use_fastmem = true;
    block.terminal = Terminal::LinkBlock { next_pc: 0x1008 };
    optimize(&mut block);
    assert_eq!(maps(&block), 2);
    assert_eq!(literal_map(&block, 2).0, input);
    assert_eq!(block.iter_live().filter(|(_, node)| node.op == Op::SetV).count(), 2);
    assert_eq!(block.code[reread.as_usize()].op, Op::Void);
    assert_eq!(block.code[result.as_usize()].pc, 0x1004);
    assert_eq!(block.cycles, 2);
    assert_eq!(block.end_pc, 0x1008);
    assert!(block.use_fastmem);
    assert!(matches!(block.terminal, Terminal::LinkBlock { next_pc: 0x1008 }));
}

#[test]
fn scalar_writes_and_observable_callbacks_break_domain_forwarding() {
    for scalar_write in [false, true] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let first = em.vec_not(input, true);
        em.set_v_q(1, first);
        if scalar_write {
            let scalar = em.const_u32(0x1937_9bdf);
            em.set_v_s(1, scalar);
        } else {
            let address = em.const_u64(0x2000);
            let value = em.const_u64(0x1937_9bdf);
            em.store(address, value, 8);
        }
        let reread = em.get_v_q(1);
        let result = em.vec_not(reread, true);
        em.set_v_q(2, result);
        optimize(&mut block);
        assert_eq!(maps(&block), 0);
        assert_eq!(stored(&block, 2).args[0], reread);
        assert_eq!(block.code[reread.as_usize()].op, Op::GetV);
    }
}

#[test]
fn scratch_reuse_does_not_retain_domains_from_another_block() {
    let mut scratch = Scratch::new();
    for composed in [true, false] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let value = if composed { em.vec_bit_affine(input, 0x1357_9bdf_2468_ace0, 0x5a, true) } else { input };
        let result = em.vec_bit_affine(value, 0x1357_9bdf_2468_ace0, 0x5a, true);
        em.set_v_q(1, result);
        optimize_with_scratch(&mut block, &mut scratch);
        assert_eq!(maps(&block), 1);
        assert_eq!(literal_map(&block, 1).0, input);
    }
}

#[test]
fn byte_duplicate_constants_truncate_to_byte_and_clear_q0_upper_half() {
    for value in [0, 0x55, 0xff, 0x100, 0x1937_9bdf_2468_ace0, u64::MAX] {
        for q in [false, true] {
            for word in [false, true] {
                let mut block = Block::new(0x1000);
                let mut em = IrEmitter::new(&mut block, 0x1000);
                let scalar = if word { em.const_u32(value as u32) } else { em.const_u64(value) };
                let duplicate = em.vec_dup_gpr(scalar, 0, q);
                let mut consts = vec![None; block.code.len()];
                consts[scalar.as_usize()] = Some(block.get(scalar).imm);
                let vectors = vec![None; block.code.len()];
                let repeated = u64::from_le_bytes([value as u8; 8]);
                assert_eq!(
                    vector_constant(block.get(duplicate), &block, &consts, &vectors, None),
                    Some([repeated, if q { repeated } else { 0 }]),
                );
            }
        }
    }
}

#[test]
fn byte_duplicate_facts_do_not_infer_unknown_wider_or_invalid_domains() {
    let mut block = Block::new(0x1000);
    let mut em = IrEmitter::new(&mut block, 0x1000);
    let unknown = em.get_x(0);
    let constant = em.const_u64(0x1937_9bdf_2468_ace0);
    let qvalue = em.get_v_q(1);
    let duplicates = [
        em.vec_dup_gpr(unknown, 0, true),
        em.vec_dup_gpr(constant, 1, true),
        em.push(Armlet::new(Op::VecDupGpr8, Ty::U128).with_args(&[qvalue]).with_imm(1)),
        em.push(Armlet::new(Op::VecDupGpr8, Ty::U128).with_args(&[constant]).with_imm(3)),
    ];
    let mut consts = vec![None; block.code.len()];
    consts[constant.as_usize()] = Some(block.get(constant).imm);
    let vectors = vec![None; block.code.len()];
    for duplicate in duplicates {
        assert!(vector_constant(block.get(duplicate), &block, &consts, &vectors, None).is_none());
    }
}

#[test]
fn cost_gate_keeps_cheap_retained_outer_operations_native() {
    for fanout in [false, true] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let source = em.get_v_q(0);
        let first = em.vec_sshr_imm(source, 0, 3, true);
        let input = if fanout {
            em.set_v_q(1, first);
            first
        } else {
            em.set_v_q(1, first);
            em.get_v_q(1)
        };
        let value = em.vec_not(input, true);
        em.set_v_q(2, value);
        optimize(&mut block);
        assert_eq!(maps(&block), 0);
        assert_eq!(stored(&block, 2).op, Op::VecNot);
        assert_eq!(stored(&block, 2).args[0], input);
        assert_eq!(block.iter_live().filter(|(_, node)| node.op == Op::SetV).count(), 2);
    }
}

#[test]
fn cnt_proves_dup_masks_and_fuses_only_affine_subexpressions() {
    for q in [false, true] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let value = em.vec_cnt(input, q);
        em.set_v_q(1, value);
        block.cycles = 1;
        optimize(&mut block);
        assert_eq!(maps(&block), 1);
        assert_eq!(block.cycles, 1);
        let affine = block.iter_live().find(|(_, node)| node.op == Op::VecBitAffine).unwrap().1;
        let matrix = block.get(affine.args[1]);
        for value in 0..=255u8 {
            assert_eq!(applied(value, matrix.imm, (affine.imm >> 8) as u8), (value >> 1) & 0x55);
        }
        assert_eq!(block.iter_live().filter(|(_, node)| node.op == Op::VecAdd8).count(), 2);
        assert_eq!(block.iter_live().filter(|(_, node)| node.op == Op::VecSub8).count(), 1);
    }
}

#[test]
fn uniform_variable_byte_shifts_cover_all_signed_counts_and_input_bytes() {
    for signed in [false, true] {
        for q in [false, true] {
            for byte_count in 0..=255u8 {
                let mut block = Block::new(0x1000);
                let mut em = IrEmitter::new(&mut block, 0x1000);
                let source = em.get_v_q(0);
                let scalar = em.const_u64(0x1937_9bdf_2468_ac00 | u64::from(byte_count));
                let count = em.vec_dup_gpr(scalar, 0, q);
                let op = if signed { Op::VecSshl8 } else { Op::VecUshl8 };
                let value = em.push(Armlet::new(op, Ty::U128).with_args(&[source, count]).with_imm(u64::from(q)));
                em.set_v_q(1, value);
                optimize(&mut block);
                assert_eq!(maps(&block), 1);
                let (input, matrix, addend, output_q) = literal_map(&block, 1);
                assert_eq!((input, addend, output_q), (source, 0, q));
                let shift = byte_count as i8 as i32;
                for byte in 0..=255u8 {
                    let expected = if shift >= 0 {
                        if shift >= 8 { 0 } else { byte << shift as u32 }
                    } else {
                        let amount = shift.unsigned_abs();
                        if signed {
                            if amount >= 8 { if byte & 0x80 != 0 { 0xff } else { 0 } }
                            else { (byte as i8 as i16 >> amount) as u8 }
                        } else if amount >= 8 { 0 } else { byte >> amount }
                    };
                    assert_eq!(applied(byte, matrix, addend), expected, "{signed}/{q}/{shift}/{byte}");
                }
            }
        }
    }
}

#[test]
fn variable_byte_shift_facts_reject_heterogeneous_and_unknown_counts() {
    for q in [false, true] {
        for known in [false, true] {
            let mut block = Block::new(0x1000);
            let mut em = IrEmitter::new(&mut block, 0x1000);
            let input = em.get_v_q(0);
            let count = if known {
                let lo = em.const_u64(0x0303_0303_0303_0304);
                let hi = em.const_u64(0x0303_0303_0303_0303);
                em.vec_build_q(lo, hi)
            } else {
                em.get_v_q(1)
            };
            let value = em.push(Armlet::new(Op::VecUshl8, Ty::U128).with_args(&[input, count]).with_imm(u64::from(q)));
            em.set_v_q(2, value);
            optimize(&mut block);
            assert_eq!(maps(&block), 0);
            assert_eq!(stored(&block, 2).op, Op::VecUshl8);
        }
    }
}

#[test]
fn byte_doubling_and_power_of_two_multiply_are_facts_not_isolated_rewrites() {
    for q in [false, true] {
        let mut block = Block::new(0x1000);
        let mut em = IrEmitter::new(&mut block, 0x1000);
        let input = em.get_v_q(0);
        let doubled = em.vec_add(input, input, 0, q);
        em.set_v_q(1, doubled);
        let consts = vec![None; block.code.len()];
        let vectors = vec![None; block.code.len()];
        let map = recognize(block.get(doubled), &block, &consts, &vectors).unwrap();
        for byte in 0..=255u8 {
            assert_eq!(applied(byte, map.matrix, map.addend), byte.wrapping_add(byte));
        }
        optimize(&mut block);
        assert_eq!(maps(&block), 0);
        assert_eq!(stored(&block, 1).op, Op::VecAdd8);
        for multiplier in 0..=255u8 {
            let mut block = Block::new(0x1000);
            let mut em = IrEmitter::new(&mut block, 0x1000);
            let source = em.get_v_q(0);
            let scalar = em.const_u64(u64::from(multiplier));
            let constant = em.vec_dup_gpr(scalar, 0, q);
            let result = em.vec_mul(source, constant, 0, q);
            em.set_v_q(1, result);
            let consts = vec![None; block.code.len()];
            let mut vectors = vec![None; block.code.len()];
            let repeated = u64::from_le_bytes([multiplier; 8]);
            vectors[constant.as_usize()] = Some([repeated, if q { repeated } else { 0 }]);
            let map = recognize(block.get(result), &block, &consts, &vectors);
            assert_eq!(map.is_some(), multiplier.is_power_of_two());
            if let Some(map) = map {
                for byte in 0..=255u8 {
                    assert_eq!(applied(byte, map.matrix, map.addend), byte.wrapping_mul(multiplier));
                }
            }
            optimize(&mut block);
            assert_eq!(maps(&block), 0);
            assert_eq!(stored(&block, 1).op, Op::VecMul8);
        }
    }
}
