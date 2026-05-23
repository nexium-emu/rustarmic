//! The 32-byte flat IR instruction record.

use crate::ir::{Op, Ty, ValueRef};

bitflags::bitflags! {
    /// Per-instruction flags maintained by the translator/optimizer.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct ArmletFlags: u8 {
        /// The optimizer has marked this armlet dead but kept the slot
        /// to preserve SSA indices. Backend skips it.
        const ELIMINATED   = 1 << 0;
        /// Produces NZCV that some later armlet consumes.
        const NZCV_LIVE    = 1 << 1;
        /// 32-bit GPR write — backend must zero-extend top half.
        const W_SIZED      = 1 << 2;
        /// Translator decoded this from a guest instruction (vs synthesized).
        const GUEST_BOUND  = 1 << 3;
        /// Used by the optimizer to mark armlets it has already const-folded.
        const FOLDED       = 1 << 4;
    }
}

/// A single SSA instruction. Exactly 32 bytes — half a cache line.
///
/// Layout is `#[repr(C)]` so we get a stable, predictable footprint.
/// Field order is tuned so the hottest fields (`op`, `args`) sit at the
/// front of the record. The implicit padding before `imm` makes this
/// land at exactly 32 bytes with 8-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Armlet {
    pub op:    Op,           // 2 bytes  (offset 0)
    pub ty:    Ty,           // 1 byte   (offset 2)
    pub flags: ArmletFlags,  // 1 byte   (offset 3)
    pub args:  [ValueRef; 4],// 16 bytes (offset 4..20)
    // implicit 4-byte alignment pad here (offset 20..24)
    pub imm:   u64,          // 8 bytes  (offset 24..32)
}

const _: () = {
    // Force the size; if anyone bloats Armlet, fail the build loudly.
    assert!(core::mem::size_of::<Armlet>() == 32);
    assert!(core::mem::align_of::<Armlet>() <= 8);
};

impl Armlet {
    #[inline]
    pub const fn new(op: Op, ty: Ty) -> Self {
        Self {
            op,
            ty,
            flags: ArmletFlags::empty(),
            args:  [ValueRef::NONE; 4],
            imm:   0,
        }
    }

    #[inline]
    pub fn with_args(mut self, args: &[ValueRef]) -> Self {
        debug_assert!(args.len() <= 4);
        for (slot, &v) in self.args.iter_mut().zip(args.iter()) {
            *slot = v;
        }
        self
    }

    #[inline]
    pub fn with_imm(mut self, imm: u64) -> Self {
        self.imm = imm;
        self
    }

    #[inline]
    pub fn with_flags(mut self, flags: ArmletFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Iterate over the populated argument slots.
    #[inline]
    pub fn arg_iter(&self) -> impl Iterator<Item = ValueRef> + '_ {
        self.args.iter().copied().filter(|v| v.is_some())
    }

    /// Number of populated arg slots (counting prefix, not gaps).
    #[inline]
    pub fn arity(&self) -> usize {
        self.args.iter().take_while(|v| v.is_some()).count()
    }

    #[inline]
    pub fn is_eliminated(&self) -> bool {
        self.flags.contains(ArmletFlags::ELIMINATED)
    }

    #[inline]
    pub fn mark_eliminated(&mut self) {
        self.op = Op::Void;
        self.ty = Ty::Void;
        self.flags.insert(ArmletFlags::ELIMINATED);
        self.args = [ValueRef::NONE; 4];
    }

    /// Replace this armlet with a copy of `src` (used by copy propagation).
    /// The slot index is preserved so existing references stay valid.
    #[inline]
    pub fn become_identity(&mut self, src: ValueRef) {
        let ty = self.ty;
        *self = Armlet::new(Op::Identity, ty);
        self.args[0] = src;
    }

    /// Replace this armlet with a 32-bit constant.
    #[inline]
    pub fn become_const_u32(&mut self, v: u32) {
        *self = Armlet::new(Op::ConstU32, Ty::U32).with_imm(v as u64);
        self.flags.insert(ArmletFlags::FOLDED);
    }

    /// Replace this armlet with a 64-bit constant.
    #[inline]
    pub fn become_const_u64(&mut self, v: u64) {
        *self = Armlet::new(Op::ConstU64, Ty::U64).with_imm(v);
        self.flags.insert(ArmletFlags::FOLDED);
    }
}
