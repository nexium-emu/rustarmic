use crate::backend::clobbers::{clobbers_for_op, GprMask};
use crate::ir::{Block, Op, Terminal, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    /// Lives in a host GPR. `u8` is the x86 register encoding (0=RAX, 3=RBX, …).
    Reg(u8),
    /// Lives in a host XMM register, used as a fast non-cache-touching spill
    /// slot per the Intel optimisation manual recommendation. `u8` is the XMM
    /// index (0..15). Loads/stores use `movq` / `movd`.
    Xmm(u8),
    /// Lives on the host stack at `[rbp - offset]`.
    Spill(i32),
    None,
}

/// SSA live range. Stored as `(start, count)` rather than `(start, end)` so
/// the struct is 4 bytes — half the cache footprint of a `Vec<LiveRange>`.
/// `count == 0` is the dead marker; otherwise the range covers
/// `[start, start + count - 1]` inclusive. `count` saturates at 65 535 (blocks
/// cap at 65 536 SSA nodes anyway).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRange {
    pub start: u16,
    pub count: u16,
}

impl LiveRange {
    pub const DEAD: LiveRange = LiveRange { start: 0, count: 0 };

    #[inline]
    pub const fn point(idx: u16) -> LiveRange {
        LiveRange { start: idx, count: 1 }
    }

    #[inline]
    pub const fn is_dead(self) -> bool {
        self.count == 0
    }

    #[inline]
    pub const fn end(self) -> u32 {
        self.start as u32 + self.count.saturating_sub(1) as u32
    }

    #[inline]
    pub const fn start_u32(self) -> u32 {
        self.start as u32
    }

    #[inline]
    pub fn extend_to(&mut self, new_end: u32) {
        debug_assert!(!self.is_dead());
        let span = (new_end - self.start as u32).saturating_add(1);
        if span > self.count as u32 {
            self.count = span.min(u16::MAX as u32) as u16;
        }
    }

    #[inline]
    pub const fn contains(self, idx: u32) -> bool {
        !self.is_dead() && idx >= self.start as u32 && idx <= self.end()
    }
}

pub struct Allocation {
    pub locs:        Vec<Loc>,
    pub spill_bytes: i32,
    /// Bitmask of XMM registers (0..15) used as spill slots within this
    /// block. The prologue must save these into the per-block frame.
    pub used_xmms:   u16,
}

impl Allocation {
    #[inline]
    pub fn loc(&self, v: crate::ir::ValueRef) -> Loc {
        self.locs[v.as_usize()]
    }

    /// XMM6..15 are saved by the shared thunk, not per-block; this is kept
    /// as a no-op so we don't have to ripple a constant deletion through the
    /// rest of the codebase.
    #[inline]
    pub fn xmm_save_bytes(&self) -> i32 { 0 }

    pub fn iter_used_xmms(&self) -> impl Iterator<Item = u8> + '_ {
        (0..16u8).filter(move |i| (self.used_xmms >> i) & 1 != 0)
    }

    #[inline]
    pub fn frame_bytes(&self) -> i32 { 0 }
}

pub fn compute_live_ranges(block: &Block) -> Vec<LiveRange> {
    let n = block.code.len();
    let mut ranges = vec![LiveRange::DEAD; n];

    // "Time" is the linked-list position, NOT the Vec index. The optimizer's
    // `insert_before` peepholes (mul-fold, const-combine) push new nodes to
    // the tail of `code` but splice them in linked-list order, so idx-as-time
    // gets uses appearing before defs. Walking the live list assigns ordinals
    // matching execution order.
    let mut pos: Vec<u32> = vec![0; n];
    let mut t: u32 = 0;
    for (vr, _) in block.iter_live() {
        pos[vr.as_usize()] = t;
        let i = vr.as_usize();
        ranges[i] = LiveRange::point(t as u16);
        t = t.saturating_add(1);
    }

    for (vr, armlet) in block.iter_live() {
        let user_t = pos[vr.as_usize()];
        for arg in armlet.args.iter() {
            if arg.is_some() {
                let arg_idx = arg.as_usize();
                if arg_idx < n && !ranges[arg_idx].is_dead() && ranges[arg_idx].end() < user_t {
                    ranges[arg_idx].extend_to(user_t);
                }
            }
        }
    }

    let last_live = block.tail_vr().map(|v| pos[v.as_usize()]).unwrap_or(0);
    let term_refs: [Option<crate::ir::ValueRef>; 2] = match block.terminal {
        Terminal::ConditionalBranch { cond_nzcv, .. } => [Some(cond_nzcv), None],
        Terminal::CompareBranchZero { value, .. } | Terminal::TestBranchBit { value, .. } => {
            [Some(value), None]
        }
        Terminal::IndirectBranch { target, .. } => [Some(target), None],
        _ => [None, None],
    };
    for v in term_refs.into_iter().flatten() {
        if v.is_some() {
            let i = v.as_usize();
            if i < n && !ranges[i].is_dead() && ranges[i].end() < last_live {
                ranges[i].extend_to(last_live);
            }
        }
    }

    ranges
}

pub const ALLOCATABLE_GPRS: &[u8] = &[3, 12, 13, 14];

/// XMM registers used as fast spill slots (movq/movd round-trip beats a
/// stack store). On Windows these are callee-saved (XMM6..XMM15), so the
/// prologue restores any that the block actually uses.
#[cfg(target_os = "windows")]
pub const SPILL_XMMS: &[u8] = &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// On SysV every XMM is caller-saved, so using them across memory callbacks
/// is unsafe without per-call save/restore — leave the XMM spill pool empty
/// and fall back to stack spills there for now.
#[cfg(not(target_os = "windows"))]
pub const SPILL_XMMS: &[u8] = &[];

pub fn op_clobbers(op: Op) -> GprMask {
    clobbers_for_op(op).gpr
}

pub fn op_prefers_two_address(op: Op) -> bool {
    matches!(op,
        Op::Add32 | Op::Add64 | Op::Sub32 | Op::Sub64
        | Op::And32 | Op::And64 | Op::Or32 | Op::Or64
        | Op::Eor32 | Op::Eor64 | Op::Mul32 | Op::Mul64
    )
}

const SAVED_SIZE: i32 = 40;

pub fn linear_scan(block: &Block, ranges: &[LiveRange], pool: &[u8]) -> Allocation {
    let n = ranges.len();
    let mut locs = vec![Loc::None; n];
    let mut spill_cursor: i32 = SAVED_SIZE;
    let mut free: Vec<u8> = pool.iter().copied().rev().collect();
    // Bitmask of XMM register IDs currently free in the spill pool. Bit `r`
    // set ⇔ XMM `r` is available. Built by OR-ing one bit per SPILL_XMMS
    // entry, so the bit index matches the actual register number.
    let mut xmm_free: u16 = SPILL_XMMS.iter().fold(0u16, |a, &r| a | (1u16 << r));
    let mut used_xmms: u16 = 0;
    let mut active: Vec<(u32, u8, usize)> = Vec::new();
    // 128-bit values that own an XMM. Recycled on end-of-life back to xmm_free.
    // Scalar values that were spilled to an XMM are NOT tracked here — they
    // remain there for the rest of the block (simpler, slightly wasteful).
    let mut xmm_active: Vec<(u32, u8, usize)> = Vec::new();

    // Lambda that allocates a spill slot, preferring an XMM register. Picks
    // the lowest-numbered free XMM (matching the old `Vec.pop` semantic).
    let take_spill = |spill_cursor: &mut i32, xmm_free: &mut u16, used_xmms: &mut u16| -> Loc {
        if *xmm_free != 0 {
            let x = xmm_free.trailing_zeros() as u8;
            *xmm_free  &= !(1u16 << x);
            *used_xmms |=   1u16 << x;
            Loc::Xmm(x)
        } else {
            Loc::Spill(alloc_spill_slot(spill_cursor))
        }
    };

    // Clobber masks indexed by linked-list position (matching the time axis
    // used by live ranges), not by Vec idx. Optimizer peepholes that insert
    // new nodes via `insert_before` give them high Vec idx but earlier
    // linked-list positions, so iterating `block.code` here would be wrong.
    let mut clobber_masks: Vec<GprMask> = Vec::with_capacity(n);
    for (_vr, armlet) in block.iter_live() {
        clobber_masks.push(if armlet.is_eliminated() {
            GprMask::empty()
        } else {
            clobbers_for_op(armlet.op).gpr
        });
    }

    let mut intervals: Vec<usize> = (0..n)
        .filter(|&i| !ranges[i].is_dead() && block.code[i].ty != Ty::Void)
        .collect();
    intervals.sort_by_key(|&i| ranges[i].start);

    for vr_idx in intervals {
        let range = ranges[vr_idx];
        let start = range.start_u32();
        let end = range.end();

        active.retain(|&(active_end, reg, _)| {
            if active_end < start {
                free.push(reg);
                false
            } else {
                true
            }
        });

        xmm_active.retain(|&(active_end, xmm_idx, _)| {
            if active_end < start {
                xmm_free |= 1 << xmm_idx;
                false
            } else {
                true
            }
        });

        // 128-bit values must live in an XMM (or a 16-byte aligned stack slot).
        // Route them through a dedicated XMM allocation that recycles on death.
        if block.code[vr_idx].ty == Ty::U128 {
            locs[vr_idx] = if xmm_free != 0 {
                let x = xmm_free.trailing_zeros() as u8;
                xmm_free  &= !(1u16 << x);
                used_xmms |=   1u16 << x;
                xmm_active.push((end, x, vr_idx));
                Loc::Xmm(x)
            } else {
                Loc::Spill(alloc_spill_slot_16(&mut spill_cursor))
            };
            continue;
        }

        let op = block.code[vr_idx].op;
        if op == Op::Identity || op_prefers_two_address(op) {
            let src = block.code[vr_idx].args[0];
            if src.is_some() {
                let src_idx = src.as_usize();
                if src_idx < n
                    && !ranges[src_idx].is_dead()
                    && ranges[src_idx].end() == start
                {
                    if let Loc::Reg(reg) = locs[src_idx] {
                        locs[vr_idx] = Loc::Reg(reg);
                        if let Some(slot) = active.iter_mut().find(|(_, r, vi)| *r == reg && *vi == src_idx) {
                            *slot = (end, reg, vr_idx);
                        }
                        continue;
                    }
                    if op == Op::Identity {
                        locs[vr_idx] = locs[src_idx];
                        continue;
                    }
                }
            }
        }

        let forbidden = interior_clobber_mask(&clobber_masks, range);

        let safe_pos = free.iter().rposition(|&r| !mask_contains_gpr(forbidden, r));
        if let Some(pos) = safe_pos {
            let reg = free.remove(pos);
            locs[vr_idx] = Loc::Reg(reg);
            active.push((end, reg, vr_idx));
        } else if active.is_empty() {
            locs[vr_idx] = take_spill(&mut spill_cursor, &mut xmm_free, &mut used_xmms);
        } else {
            let candidate = active
                .iter()
                .enumerate()
                .filter(|&(_, &(_, reg, _))| !mask_contains_gpr(forbidden, reg))
                .max_by_key(|(_, e)| e.0);

            if let Some((spill_pos, &(victim_end, victim_reg, victim_vr))) = candidate {
                if victim_end > end {
                    let spilled = take_spill(&mut spill_cursor, &mut xmm_free, &mut used_xmms);
                    locs[victim_vr] = spilled;
                    locs[vr_idx]    = Loc::Reg(victim_reg);
                    active[spill_pos] = (end, victim_reg, vr_idx);
                } else {
                    locs[vr_idx] = take_spill(&mut spill_cursor, &mut xmm_free, &mut used_xmms);
                }
            } else {
                locs[vr_idx] = take_spill(&mut spill_cursor, &mut xmm_free, &mut used_xmms);
            }
        }
    }

    // Per-block XMM save is gone — the shared thunk preserves XMM6..15 once
    // per host→JIT call. We still track `used_xmms` for allocation correctness
    // (so spill scalars don't collide with live u128 values), but the prologue
    // emits nothing for them and stack-spill offsets need no extra shift.
    Allocation {
        locs,
        spill_bytes: spill_cursor - SAVED_SIZE,
        used_xmms,
    }
}

#[inline]
fn mask_contains_gpr(mask: GprMask, gpr: u8) -> bool {
    mask.bits() & (1u16 << gpr) != 0
}

fn interior_clobber_mask(clobber_masks: &[GprMask], range: LiveRange) -> GprMask {
    let mut mask = GprMask::empty();
    if range.is_dead() { return mask; }
    let start = range.start as usize;
    let end = range.end() as usize;
    if end > start + 1 {
        for i in (start + 1)..end {
            mask |= clobber_masks[i];
        }
    }
    mask
}

fn alloc_spill_slot(cursor: &mut i32) -> i32 {
    let aligned = (*cursor + 7) & !7;
    *cursor = aligned + 8;
    *cursor
}

fn alloc_spill_slot_16(cursor: &mut i32) -> i32 {
    let aligned = (*cursor + 15) & !15;
    *cursor = aligned + 16;
    *cursor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Armlet, IrEmitter, Ty};

    fn fresh_block() -> Block {
        Block::new(0x1000)
    }

    #[test]
    fn linear_chain_propagates_end_to_last_use() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let c1 = em.const_u64(1);
        let c2 = em.const_u64(2);
        let add1 = em.add(c1, c2, crate::arch::RegSize::X);
        let add2 = em.add(add1, c2, crate::arch::RegSize::X);
        em.set_x(0, add2);

        let ranges = compute_live_ranges(&b);
        // Ranges are now indexed by linked-list position; on this freshly
        // built block (no peephole inserts) position == idx - 1 because the
        // Op::Void sentinel occupies idx 0 but isn't in the live list.
        let pos = |v: crate::ir::ValueRef| v.idx() - 1;
        assert!(!ranges[c1.as_usize()].is_dead());
        assert_eq!(ranges[c1.as_usize()].end(), pos(add1));
        assert_eq!(ranges[c2.as_usize()].end(), pos(add2));
        assert!(ranges[add1.as_usize()].end() >= pos(add2));
    }

    #[test]
    fn terminal_referenced_value_lives_to_last_armlet() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let val = em.const_u64(0);
        let _filler = em.const_u64(7);
        let _filler2 = em.const_u64(8);
        em.push(Armlet::new(Op::CbZ, Ty::Void).with_args(&[val]).with_imm(0x2000));
        b.terminal = Terminal::CompareBranchZero {
            value: val,
            inverse: false,
            taken_pc: 0x2000,
            not_taken_pc: 0x1004,
        };

        let ranges = compute_live_ranges(&b);
        let last_pos = b.tail_vr().unwrap().idx() - 1; // subtract sentinel
        assert_eq!(ranges[val.as_usize()].end(), last_pos);
    }

    #[test]
    fn unlinked_armlet_has_dead_range() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let c = em.const_u64(42);
        let _add = em.add(c, c, crate::arch::RegSize::X);
        let head = b.head_vr().unwrap();
        b.unlink(head);

        let ranges = compute_live_ranges(&b);
        assert!(ranges[head.as_usize()].is_dead());
    }

    #[test]
    fn div_clobbers_include_rax_rcx_rdx() {
        let mask = op_clobbers(Op::SDiv64);
        assert!(mask.contains(GprMask::RAX));
        assert!(mask.contains(GprMask::RCX));
        assert!(mask.contains(GprMask::RDX));
    }

    #[test]
    fn linear_scan_assigns_regs_up_to_pool_size() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let mut vrs = Vec::new();
        for _ in 0..ALLOCATABLE_GPRS.len() {
            vrs.push(em.const_u64(0));
        }
        for &vr in &vrs {
            em.set_x(0, vr);
        }

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, ALLOCATABLE_GPRS);

        let mut assigned_regs = std::collections::HashSet::new();
        for &vr in &vrs {
            match alloc.locs[vr.as_usize()] {
                Loc::Reg(r) => { assigned_regs.insert(r); }
                other => panic!("expected Reg, got {:?}", other),
            }
        }
        assert_eq!(assigned_regs.len(), ALLOCATABLE_GPRS.len());
        assert_eq!(alloc.spill_bytes, 0);
    }

    #[test]
    fn linear_scan_spills_when_pool_exhausted() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let pool = ALLOCATABLE_GPRS.len();
        let extra = 3;
        let mut vrs = Vec::new();
        for _ in 0..(pool + extra) {
            vrs.push(em.const_u64(0));
        }
        for &vr in &vrs {
            em.set_x(0, vr);
        }

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, ALLOCATABLE_GPRS);

        let mut spilled = 0;
        let mut in_reg = 0;
        for &vr in &vrs {
            match alloc.locs[vr.as_usize()] {
                Loc::Reg(_) => in_reg += 1,
                Loc::Spill(_) | Loc::Xmm(_) => spilled += 1,
                Loc::None => panic!("live value should not be Loc::None"),
            }
        }
        assert_eq!(spilled, extra);
        assert_eq!(in_reg, pool);
        assert!(alloc.spill_bytes > 0 || alloc.used_xmms != 0);
    }

    #[test]
    fn linear_scan_avoids_spills_when_values_die_in_time() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        for _ in 0..(ALLOCATABLE_GPRS.len() + 4) {
            let c = em.const_u64(0);
            em.set_x(0, c);
        }

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, ALLOCATABLE_GPRS);

        let any_spill = alloc.locs.iter().any(|l| matches!(l, Loc::Spill(_) | Loc::Xmm(_)));
        assert!(!any_spill);
        assert_eq!(alloc.spill_bytes, 0);
        assert_eq!(alloc.used_xmms, 0);
    }

    #[test]
    fn identity_shares_reg_with_src_when_src_dies_at_identity() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let src = em.const_u64(0xABCD);
        let id = em.push(Armlet::new(Op::Identity, Ty::U64).with_args(&[src]));
        em.set_x(0, id);

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, ALLOCATABLE_GPRS);

        assert_eq!(alloc.locs[src.as_usize()], alloc.locs[id.as_usize()]);
        assert!(matches!(alloc.locs[id.as_usize()], Loc::Reg(_)));
    }

    #[test]
    fn u128_values_get_xmm_and_recycle_on_death() {
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        // Two non-overlapping 128-bit lifetimes — both should land in the SAME
        // XMM if the allocator recycles on death.
        let q0 = em.get_v_q(0);
        em.set_v_q(1, q0);
        let q2 = em.get_v_q(2);
        em.set_v_q(3, q2);

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, ALLOCATABLE_GPRS);

        let loc_a = alloc.loc(q0);
        let loc_b = alloc.loc(q2);
        assert!(matches!(loc_a, Loc::Xmm(_)), "first u128 should be Xmm, got {loc_a:?}");
        assert!(matches!(loc_b, Loc::Xmm(_)), "second u128 should be Xmm, got {loc_b:?}");
        if !SPILL_XMMS.is_empty() {
            assert_eq!(loc_a, loc_b, "non-overlapping u128 ranges should reuse the same XMM");
        }
    }

    #[test]
    fn overlapping_u128_values_take_different_xmms() {
        if SPILL_XMMS.len() < 2 { return; } // platform without XMM pool
        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let a = em.get_v_q(0);
        let c = em.get_v_q(1);
        em.set_v_q(2, a);
        em.set_v_q(3, c);

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, ALLOCATABLE_GPRS);
        assert_ne!(alloc.loc(a), alloc.loc(c));
    }

    #[test]
    fn linear_scan_avoids_pool_reg_clobbered_across_live_range() {
        use crate::arch::RegSize;

        let mut b = fresh_block();
        let mut em = IrEmitter::new(&mut b, 0x1000);
        let amount = em.const_u64(1);
        let val = em.const_u64(0xDEAD);
        let shifted = em.push(Armlet::new(Op::Lsl64, Ty::U64).with_args(&[val, amount]));
        let _ = em.add(val, shifted, RegSize::X);

        let ranges = compute_live_ranges(&b);
        let alloc = linear_scan(&b, &ranges, &[1]);

        assert!(
            matches!(alloc.locs[val.as_usize()], Loc::Spill(_) | Loc::Xmm(_)),
            "val crosses a shift that clobbers RCX, so it cannot live in RCX (got {:?})",
            alloc.locs[val.as_usize()]
        );
    }
}
