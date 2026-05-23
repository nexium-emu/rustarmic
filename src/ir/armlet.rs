use crate::ir::{Op, Ty, ValueRef};

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
    pub struct ArmletFlags: u8 {
        const NZCV_LIVE    = 1 << 0;
        const W_SIZED      = 1 << 1;
        const GUEST_BOUND  = 1 << 2;
        const FOLDED       = 1 << 3;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Armlet {
    pub op:    Op,
    pub ty:    Ty,
    pub flags: ArmletFlags,
    pub prev:  u32,
    pub next:  u32,
    pub args:  [ValueRef; 4],
    pub imm:   u64,
}

pub const LINK_NONE: u32 = 0;

const _: () = {
    assert!(core::mem::size_of::<Armlet>() == 40);
    assert!(core::mem::align_of::<Armlet>() <= 8);
};

impl Armlet {
    #[inline]
    pub const fn new(op: Op, ty: Ty) -> Self {
        Self {
            op,
            ty,
            flags: ArmletFlags::empty(),
            prev:  LINK_NONE,
            next:  LINK_NONE,
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

    #[inline]
    pub fn arg_iter(&self) -> impl Iterator<Item = ValueRef> + '_ {
        self.args.iter().copied().filter(|v| v.is_some())
    }

    #[inline]
    pub fn arity(&self) -> usize {
        self.args.iter().take_while(|v| v.is_some()).count()
    }

    #[inline]
    pub fn is_eliminated(&self) -> bool {
        self.op == Op::Void
    }

    #[inline]
    pub fn become_identity(&mut self, src: ValueRef) {
        let ty = self.ty;
        let prev = self.prev;
        let next = self.next;
        *self = Armlet::new(Op::Identity, ty);
        self.args[0] = src;
        self.prev = prev;
        self.next = next;
    }

    #[inline]
    pub fn become_const_u32(&mut self, v: u32) {
        let prev = self.prev;
        let next = self.next;
        *self = Armlet::new(Op::ConstU32, Ty::U32).with_imm(v as u64);
        self.flags.insert(ArmletFlags::FOLDED);
        self.prev = prev;
        self.next = next;
    }

    #[inline]
    pub fn become_const_u64(&mut self, v: u64) {
        let prev = self.prev;
        let next = self.next;
        *self = Armlet::new(Op::ConstU64, Ty::U64).with_imm(v);
        self.flags.insert(ArmletFlags::FOLDED);
        self.prev = prev;
        self.next = next;
    }
}
