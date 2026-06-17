pub const NUM_GPRS: usize = 31;

pub const NUM_VREGS: usize = 32;

pub const ZR_ENCODING: u8 = 31;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    EQ = 0b0000,
    NE = 0b0001,
    CS = 0b0010,
    CC = 0b0011,
    MI = 0b0100,
    PL = 0b0101,
    VS = 0b0110,
    VC = 0b0111,
    HI = 0b1000,
    LS = 0b1001,
    GE = 0b1010,
    LT = 0b1011,
    GT = 0b1100,
    LE = 0b1101,
    AL = 0b1110,
    NV = 0b1111,
}

impl Cond {
    #[inline]
    pub fn from_bits(b: u8) -> Self {
        unsafe { core::mem::transmute(b & 0xF) }
    }

    #[inline]
    pub fn invert(self) -> Self {
        Self::from_bits((self as u8) ^ 1)
    }
}

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

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegSize {
    W = 32,
    X = 64,
}

pub const COND_TRUTH: [u16; 16] = [
    0xF0F0,
    0x0F0F,
    0xCCCC,
    0x3333,
    0xFF00,
    0x00FF,
    0xAAAA,
    0x5555,
    0x0C0C,
    0xF3F3,
    0xAA55,
    0x55AA,
    0x0A05,
    0xF5FA,
    0xFFFF,
    0xFFFF,
];

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
    pub const CNTPCT_EL0:  u16 = pack(3, 3, 14, 0, 1);
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
