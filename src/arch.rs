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

/// Packed truth tables for each AArch64 condition code.
///
/// `COND_TRUTH[cond as usize]` is a `u16` where bit `n` is `1` iff the
/// condition holds for NZCV value `n` (0..=15). The backend bakes this
/// constant into emitted code and tests the bit at index `NZCV` to evaluate
/// a condition in three x86 instructions (`mov` / `bt` / `setc`), replacing
/// the ~12-instruction bit-extract-and-combine sequence.
///
/// Verified by `test_cond_truth_matches_check` in this module.
pub const COND_TRUTH: [u16; 16] = [
    0xF0F0, // EQ:  Z == 1
    0x0F0F, // NE:  Z == 0
    0xCCCC, // CS:  C == 1
    0x3333, // CC:  C == 0
    0xFF00, // MI:  N == 1
    0x00FF, // PL:  N == 0
    0xAAAA, // VS:  V == 1
    0x5555, // VC:  V == 0
    0x0C0C, // HI:  C && !Z
    0xF3F3, // LS:  !(C && !Z)
    0xAA55, // GE:  N == V
    0x55AA, // LT:  N != V
    0x0A05, // GT:  !Z && N==V
    0xF5FA, // LE:  !(!Z && N==V)
    0xFFFF, // AL:  always
    0xFFFF, // NV:  always (deprecated to AL)
];

/// System-register identifier packed as `op0:op1:CRn:CRm:op2` (15 bits).
pub mod sysreg {
    pub const fn pack(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u16 {
        ((op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2) as u16
    }

    pub const NZCV:        u16 = pack(3, 3, 4,  2, 0);
    pub const FPCR:        u16 = pack(3, 3, 4,  4, 0);
    pub const FPSR:        u16 = pack(3, 3, 4,  4, 1);
    pub const TPIDR_EL0:   u16 = pack(3, 3, 13, 0, 2);
    pub const TPIDRRO_EL0: u16 = pack(3, 3, 13, 0, 3);
    pub const CTR_EL0:     u16 = pack(3, 3, 0,  0, 1);
    pub const DCZID_EL0:   u16 = pack(3, 3, 0,  0, 7);
    pub const MIDR_EL1:    u16 = pack(3, 0, 0,  0, 0);
    pub const MPIDR_EL1:   u16 = pack(3, 0, 0,  0, 5);
    pub const CNTFRQ_EL0:  u16 = pack(3, 3, 14, 0, 0);
    pub const CNTVCT_EL0:  u16 = pack(3, 3, 14, 0, 2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cond_truth_matches_check() {
        for cond_bits in 0..16u8 {
            let cond = Cond::from_bits(cond_bits);
            let tt = COND_TRUTH[cond as usize];
            for nzcv in 0..16u8 {
                let expected = Nzcv(nzcv).check(cond);
                let actual = (tt >> nzcv) & 1 != 0;
                assert_eq!(
                    expected, actual,
                    "cond={:?} nzcv={:04b}: truth-table says {}, check says {}",
                    cond, nzcv, actual, expected
                );
            }
        }
    }
}
