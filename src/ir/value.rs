#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueRef(pub u32);

impl ValueRef {
    pub const NONE: ValueRef = ValueRef(0);

    #[inline]
    pub const fn new(idx: u32) -> Self {
        Self(idx)
    }
    #[inline]
    pub const fn idx(self) -> u32 {
        self.0
    }
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != 0
    }
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl core::fmt::Debug for ValueRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_none() {
            f.write_str("_")
        } else {
            write!(f, "%{}", self.0)
        }
    }
}

impl Default for ValueRef {
    #[inline]
    fn default() -> Self {
        Self::NONE
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    Void = 0,
    U1 = 1,
    U8 = 2,
    U16 = 3,
    U32 = 4,
    U64 = 5,
    U128 = 6,
    Nzcv = 7,
}

impl Ty {
    #[inline]
    pub const fn bits(self) -> u32 {
        match self {
            Ty::Void => 0,
            Ty::U1 => 1,
            Ty::U8 => 8,
            Ty::U16 => 16,
            Ty::U32 => 32,
            Ty::U64 => 64,
            Ty::U128 => 128,
            Ty::Nzcv => 4,
        }
    }

    #[inline]
    pub const fn is_int(self) -> bool {
        matches!(
            self,
            Ty::U1 | Ty::U8 | Ty::U16 | Ty::U32 | Ty::U64 | Ty::U128
        )
    }
}
