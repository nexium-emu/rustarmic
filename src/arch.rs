//! AArch64 architectural definitions: guest registers, flags, condition codes.

/// AArch64 has 31 general-purpose 64-bit registers (X0..X30).
/// Encoding 31 means either ZR (zero register) or SP depending on instruction class;
/// we keep SP as a separate slot in the CPU context.
pub const NUM_GPRS: usize = 31;

/// 32 SIMD/FP 128-bit registers (V0..V31).
pub const NUM_VREGS: usize = 32;

/// Reserved encoding meaning "the zero register" in instructions that don't write SP.
pub const ZR_ENCODING: u8 = 31;

/// AArch64 condition codes (4-bit field in conditional instructions).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    EQ = 0b0000, // Z == 1
    NE = 0b0001, // Z == 0
    CS = 0b0010, // C == 1   (also HS)
    CC = 0b0011, // C == 0   (also LO)
    MI = 0b0100, // N == 1
    PL = 0b0101, // N == 0
    VS = 0b0110, // V == 1
    VC = 0b0111, // V == 0
    HI = 0b1000, // C == 1 && Z == 0
    LS = 0b1001, // !(C == 1 && Z == 0)
    GE = 0b1010, // N == V
    LT = 0b1011, // N != V
    GT = 0b1100, // Z == 0 && N == V
    LE = 0b1101, // !(Z == 0 && N == V)
    AL = 0b1110, // always
    NV = 0b1111, // always (deprecated)
}

impl Cond {
    #[inline]
    pub fn from_bits(b: u8) -> Self {
        // Safety: enum covers all 16 values exhaustively.
        unsafe { core::mem::transmute(b & 0xF) }
    }

    /// Logical inverse (flip bit 0). NV does not invert to NV exactly — it inverts
    /// to AL per ARM, but at the IR level we treat both as "always" so we never
    /// need to invert them.
    #[inline]
    pub fn invert(self) -> Self {
        Self::from_bits((self as u8) ^ 1)
    }
}

/// PSTATE.NZCV packed as `[N=bit3, Z=bit2, C=bit1, V=bit0]`.
/// In the IR we keep NZCV as its own value type rather than four U1s — that lets
/// us elide flag computation when the consumer needs the whole packed nibble.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Nzcv(pub u8);

impl Nzcv {
    pub const N_BIT: u8 = 1 << 3;
    pub const Z_BIT: u8 = 1 << 2;
    pub const C_BIT: u8 = 1 << 1;
    pub const V_BIT: u8 = 1 << 0;

    #[inline] pub fn n(self) -> bool { (self.0 & Self::N_BIT) != 0 }
    #[inline] pub fn z(self) -> bool { (self.0 & Self::Z_BIT) != 0 }
    #[inline] pub fn c(self) -> bool { (self.0 & Self::C_BIT) != 0 }
    #[inline] pub fn v(self) -> bool { (self.0 & Self::V_BIT) != 0 }

    /// Evaluate an AArch64 condition against these flags.
    pub fn check(self, cond: Cond) -> bool {
        let (n, z, c, v) = (self.n(), self.z(), self.c(), self.v());
        match cond {
            Cond::EQ => z,
            Cond::NE => !z,
            Cond::CS => c,
            Cond::CC => !c,
            Cond::MI => n,
            Cond::PL => !n,
            Cond::VS => v,
            Cond::VC => !v,
            Cond::HI => c && !z,
            Cond::LS => !(c && !z),
            Cond::GE => n == v,
            Cond::LT => n != v,
            Cond::GT => !z && (n == v),
            Cond::LE => !(!z && (n == v)),
            Cond::AL | Cond::NV => true,
        }
    }
}

/// Size of a register operand.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegSize {
    W = 32, // 32-bit (writes zero-extend in AArch64)
    X = 64, // 64-bit
}
