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

    /// 16 bytes per saved XMM register.
    #[inline]
    pub fn xmm_save_bytes(&self) -> i32 {
        self.used_xmms.count_ones() as i32 * 16
    }

    /// Iterates the indices of XMM registers used as spill slots in the order
    /// they were assigned save slots (ascending XMM index).
    pub fn iter_used_xmms(&self) -> impl Iterator<Item = u8> + '_ {
        (0..16u8).filter(move |i| (self.used_xmms >> i) & 1 != 0)
    }

    /// Offset (positive, subtracted from `rbp`) of the save slot for a given
    /// XMM register. Only valid for indices in `iter_used_xmms`.
    pub fn xmm_save_offset(&self, xmm: u8) -> i32 {
        let position = self.iter_used_xmms().position(|x| x == xmm).expect("xmm not in saved set");
        SAVED_SIZE + 16 * (position as i32 + 1)
    }

    #[inline]
    pub fn frame_bytes(&self) -> i32 {
        (self.spill_bytes + self.xmm_save_bytes() + 15) & -16
    }
}

pub fn compute_live_ranges(block: &Block) -> Vec<LiveRange> {
    let n = block.code.len();
    let mut ranges = vec![LiveRange::DEAD; n];

    for (vr, _) in block.iter_live() {
        let i = vr.as_usize();
        ranges[i] = LiveRange::point(vr.idx() as u16);
    }

    for (vr, armlet) in block.iter_live() {
        let user_idx = vr.idx();
        for arg in armlet.args.iter() {
            if arg.is_some() {
                let arg_idx = arg.as_usize();
                if arg_idx < n && !ranges[arg_idx].is_dead() && ranges[arg_idx].end() < user_idx {
                    ranges[arg_idx].extend_to(user_idx);
                }
            }
        }
    }

    let last_live = block.tail_vr().map(|v| v.idx()).unwrap_or(0);
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
    let mut xmm_free: Vec<u8> = SPILL_XMMS.iter().copied().rev().collect();
    let mut used_xmms: u16 = 0;
    let mut active: Vec<(u32, u8, usize)> = Vec::new();

    // Lambda that allocates a spill slot, preferring an XMM register.
    let mut take_spill = |spill_cursor: &mut i32, xmm_free: &mut Vec<u8>, used_xmms: &mut u16| -> Loc {
        if let Some(x) = xmm_free.pop() {
            *used_xmms |= 1 << x;
            Loc::Xmm(x)
        } else {
            Loc::Spill(alloc_spill_slot(spill_cursor))
        }
    };

    let clobber_masks: Vec<GprMask> = block.code.iter()
        .map(|a| if a.is_eliminated() { GprMask::empty() } else { clobbers_for_op(a.op).gpr })
        .collect();

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

    let xmm_save_bytes = used_xmms.count_ones() as i32 * 16;
    if xmm_save_bytes > 0 {
        for loc in locs.iter_mut() {
            if let Loc::Spill(off) = loc {
                *off += xmm_save_bytes;
            }
        }
    }
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
        assert!(!ranges[c1.as_usize()].is_dead());
        assert_eq!(ranges[c1.as_usize()].end(), add1.idx());
        assert_eq!(ranges[c2.as_usize()].end(), add2.idx());
        assert!(ranges[add1.as_usize()].end() >= add2.idx());
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
        let last_idx = b.tail_vr().unwrap().idx();
        assert_eq!(ranges[val.as_usize()].end(), last_idx);
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
