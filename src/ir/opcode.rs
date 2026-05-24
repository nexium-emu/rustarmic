#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Void     = 0x000,
    Identity = 0x004,

    ConstU32 = 0x012,
    ConstU64 = 0x013,
    ConstU128 = 0x014,

    GetX     = 0x020,
    SetX     = 0x024,
    GetW     = 0x028,
    SetW     = 0x02C,
    GetSp    = 0x030,
    SetSp    = 0x034,
    GetNzcv  = 0x038,
    SetNzcv  = 0x03C,
    GetPc    = 0x040,
    GetV     = 0x044,
    SetV     = 0x048,

    Add8  = 0x100, Add16 = 0x101, Add32 = 0x102, Add64 = 0x103,
    Sub8  = 0x104, Sub16 = 0x105, Sub32 = 0x106, Sub64 = 0x107,
    Adc8  = 0x108, Adc16 = 0x109, Adc32 = 0x10A, Adc64 = 0x10B,
    Sbc8  = 0x10C, Sbc16 = 0x10D, Sbc32 = 0x10E, Sbc64 = 0x10F,

    AddsFlags8  = 0x110, AddsFlags16 = 0x111, AddsFlags32 = 0x112, AddsFlags64 = 0x113,
    SubsFlags8  = 0x114, SubsFlags16 = 0x115, SubsFlags32 = 0x116, SubsFlags64 = 0x117,

    And8 = 0x118, And16 = 0x119, And32 = 0x11A, And64 = 0x11B,
    Or8  = 0x11C, Or16  = 0x11D, Or32  = 0x11E, Or64  = 0x11F,
    Eor8 = 0x120, Eor16 = 0x121, Eor32 = 0x122, Eor64 = 0x123,
    Bic8 = 0x124, Bic16 = 0x125, Bic32 = 0x126, Bic64 = 0x127,
    Orn8 = 0x128, Orn16 = 0x129, Orn32 = 0x12A, Orn64 = 0x12B,
    Eon8 = 0x12C, Eon16 = 0x12D, Eon32 = 0x12E, Eon64 = 0x12F,

    Not8 = 0x130, Not16 = 0x131, Not32 = 0x132, Not64 = 0x133,
    Neg8 = 0x134, Neg16 = 0x135, Neg32 = 0x136, Neg64 = 0x137,

    Lsl8 = 0x140, Lsl16 = 0x141, Lsl32 = 0x142, Lsl64 = 0x143,
    Lsr8 = 0x144, Lsr16 = 0x145, Lsr32 = 0x146, Lsr64 = 0x147,
    Asr8 = 0x148, Asr16 = 0x149, Asr32 = 0x14A, Asr64 = 0x14B,
    Ror8 = 0x14C, Ror16 = 0x14D, Ror32 = 0x14E, Ror64 = 0x14F,

    Ubfm8 = 0x150, Ubfm16 = 0x151, Ubfm32 = 0x152, Ubfm64 = 0x153,
    Sbfm8 = 0x154, Sbfm16 = 0x155, Sbfm32 = 0x156, Sbfm64 = 0x157,
    Bfm8  = 0x158, Bfm16  = 0x159, Bfm32  = 0x15A, Bfm64  = 0x15B,
    Extr8 = 0x15C, Extr16 = 0x15D, Extr32 = 0x15E, Extr64 = 0x15F,

    Mul8  = 0x160, Mul16  = 0x161, Mul32  = 0x162, Mul64  = 0x163,
    Madd8 = 0x164, Madd16 = 0x165, Madd32 = 0x166, Madd64 = 0x167,
    Msub8 = 0x168, Msub16 = 0x169, Msub32 = 0x16A, Msub64 = 0x16B,

    UMulH64 = 0x16F,
    SMulH64 = 0x173,
    UMull32 = 0x176,
    SMull32 = 0x17A,
    UMAddl  = 0x17C,
    SMAddl  = 0x180,
    UMSubl  = 0x184,
    SMSubl  = 0x188,

    UDiv8 = 0x190, UDiv16 = 0x191, UDiv32 = 0x192, UDiv64 = 0x193,
    SDiv8 = 0x194, SDiv16 = 0x195, SDiv32 = 0x196, SDiv64 = 0x197,
    // AArch64 div semantics — backend MUST guard:
    //   UDiv/SDiv: divisor == 0  -> result = 0       (no #DE trap)
    //   SDiv only: dividend == INT_MIN && divisor == -1 -> result = dividend (no overflow trap)
    // x86 IDIV/DIV raise #DE in both cases; emit branches/cmov to bypass.

    Zext = 0x1A0,
    Sext = 0x1A4,

    Clz8  = 0x1A8, Clz16  = 0x1A9, Clz32  = 0x1AA, Clz64  = 0x1AB,
    Cls8  = 0x1AC, Cls16  = 0x1AD, Cls32  = 0x1AE, Cls64  = 0x1AF,
    Rbit8 = 0x1B0, Rbit16 = 0x1B1, Rbit32 = 0x1B2, Rbit64 = 0x1B3,
    Rev16 = 0x1B5, Rev32 = 0x1B6, Rev64 = 0x1B7,

    Csel8   = 0x200, Csel16   = 0x201, Csel32   = 0x202, Csel64   = 0x203,
    Csinc8  = 0x204, Csinc16  = 0x205, Csinc32  = 0x206, Csinc64  = 0x207,
    Csinv8  = 0x208, Csinv16  = 0x209, Csinv32  = 0x20A, Csinv64  = 0x20B,
    Csneg8  = 0x20C, Csneg16  = 0x20D, Csneg32  = 0x20E, Csneg64  = 0x20F,

    CcmpReg8 = 0x210, CcmpReg16 = 0x211, CcmpReg32 = 0x212, CcmpReg64 = 0x213,
    CcmpImm8 = 0x214, CcmpImm16 = 0x215, CcmpImm32 = 0x216, CcmpImm64 = 0x217,
    CcmnReg8 = 0x218, CcmnReg16 = 0x219, CcmnReg32 = 0x21A, CcmnReg64 = 0x21B,
    CcmnImm8 = 0x21C, CcmnImm16 = 0x21D, CcmnImm32 = 0x21E, CcmnImm64 = 0x21F,

    Branch              = 0x300,
    BranchLink          = 0x304,
    BranchIndirect      = 0x308,
    BranchIndirectLink  = 0x30C,
    Ret                 = 0x310,
    BranchCond          = 0x314,
    CbZ                 = 0x318,
    CbNz                = 0x31C,
    TbZ                 = 0x320,
    TbNz                = 0x324,

    Load8  = 0x400, Load16  = 0x401, Load32  = 0x402, Load64  = 0x403,
    Load128 = 0x404,

    LoadS8 = 0x408, LoadS16 = 0x409, LoadS32 = 0x40A,

    Store8 = 0x40C, Store16 = 0x40D, Store32 = 0x40E, Store64 = 0x40F,
    Store128 = 0x410,

    LoadAcq8  = 0x418, LoadAcq16  = 0x419, LoadAcq32  = 0x41A, LoadAcq64  = 0x41B,
    StoreRel8 = 0x41C, StoreRel16 = 0x41D, StoreRel32 = 0x41E, StoreRel64 = 0x41F,

    LoadEx8  = 0x420, LoadEx16  = 0x421, LoadEx32  = 0x422, LoadEx64  = 0x423,
    StoreEx8 = 0x424, StoreEx16 = 0x425, StoreEx32 = 0x426, StoreEx64 = 0x427,

    LoadPair8  = 0x428, LoadPair16  = 0x429, LoadPair32  = 0x42A, LoadPair64  = 0x42B,
    StorePair8 = 0x42C, StorePair16 = 0x42D, StorePair32 = 0x42E, StorePair64 = 0x42F,

    Fmov32 = 0x502, Fmov64 = 0x503,
    Fadd32 = 0x506, Fadd64 = 0x507,
    Fsub32 = 0x50A, Fsub64 = 0x50B,
    Fmul32 = 0x50E, Fmul64 = 0x50F,
    Fdiv32 = 0x512, Fdiv64 = 0x513,
    Fneg32 = 0x516, Fneg64 = 0x517,
    Fabs32 = 0x51A, Fabs64 = 0x51B,
    Fsqrt32 = 0x51E, Fsqrt64 = 0x51F,
    Fcmp32 = 0x522, Fcmp64 = 0x523,
    Fmax32 = 0x526, Fmax64 = 0x527,
    Fmin32 = 0x52A, Fmin64 = 0x52B,

    // FP conversions (FP→signed-int truncate, signed-int→FP).
    FcvtZsSW = 0x540,  // single → i32 (truncate)
    FcvtZsSX = 0x541,  // single → i64 (truncate)
    FcvtZsDW = 0x542,  // double → i32 (truncate)
    FcvtZsDX = 0x543,  // double → i64 (truncate)
    ScvtfWS  = 0x544,  // i32 → single
    ScvtfXS  = 0x545,  // i64 → single
    ScvtfWD  = 0x546,  // i32 → double
    ScvtfXD  = 0x547,  // i64 → double
    FcvtSD   = 0x548,  // single → double
    FcvtDS   = 0x549,  // double → single

    // Per-lane vector ops. The lane element width is encoded in the low 2 bits
    // (B/H/S/D = 1/2/4/8 bytes), matching the scalar size_log2 convention.
    // The Q-flag (1 = 128-bit vector, 0 = 64-bit half-vector with upper bits
    // zeroed) is carried in `imm` bit 0.
    VecAdd8  = 0x600, VecAdd16  = 0x601, VecAdd32  = 0x602, VecAdd64  = 0x603,
    VecSub8  = 0x604, VecSub16  = 0x605, VecSub32  = 0x606, VecSub64  = 0x607,

    // Bitwise logicals — lane size is meaningless; only the Q-flag matters.
    VecAnd   = 0x608,
    VecOrr   = 0x60C,
    VecEor   = 0x610,
    VecBic   = 0x614,
    VecOrn   = 0x618,

    VecDup   = 0x61C,
    Ins      = 0x620,
    Umov     = 0x624,
    Smov     = 0x628,

    // Glue ops for 128-bit values without dedicated 128-bit memory callbacks:
    // BuildQ assembles two u64 halves into a u128; ExtractLo/Hi64 pull them
    // back out. These map to single x86 instructions (movq / pinsrq / pextrq)
    // and let the optimizer fold round-trips when both halves are visible.
    VecBuildQ      = 0x62C,
    VecExtractLo64 = 0x630,
    VecExtractHi64 = 0x634,

    // Lane-sized extracts. Lane index goes in `imm`; the byte lane size
    // matches the low 2 bits (log2 byte width) per the existing convention.
    // Result is zero-extended into a U32 — callers do sign-extension via
    // a separate Sext armlet for SMOV-style usage.
    VecExtract8  = 0x638,
    VecExtract16 = 0x639,
    VecExtract32 = 0x63A,

    Mrs            = 0x700,
    Msr            = 0x704,
    Hint           = 0x708,
    Brk            = 0x70C,
    Svc            = 0x710,
    Hvc            = 0x714,
    MemoryBarrier  = 0x718,
    Clrex          = 0x71C,
}

impl Op {
    #[inline] pub const fn raw(self) -> u16 { self as u16 }

    #[inline]
    pub const fn base(self) -> u16 {
        (self as u16) & !0b11
    }

    #[inline]
    pub const fn size_log2(self) -> u32 {
        ((self as u16) & 0b11) as u32
    }

    #[inline]
    pub const fn size_bytes(self) -> u32 {
        1u32 << self.size_log2()
    }

    #[inline]
    pub const fn size_bits(self) -> u32 {
        8u32 << self.size_log2()
    }

    pub const fn has_side_effects(self) -> bool {
        use Op::*;
        if matches!(self,
            SetX | SetW | SetSp | SetNzcv | SetV
            | AddsFlags32 | AddsFlags64 | SubsFlags32 | SubsFlags64
            | Fcmp32 | Fcmp64
            | Mrs | Msr | Brk | Svc | Hvc | Hint | MemoryBarrier | Clrex
        ) {
            return true;
        }
        match self.base() {
            b if b == Store8.base()     => true,
            b if b == Store128 as u16   => true,
            b if b == StoreRel8.base()  => true,
            b if b == StoreEx8.base()   => true,
            b if b == LoadEx8.base()    => true,
            b if b == StorePair8.base() => true,
            b if b == Branch.base()
              || b == BranchLink.base()
              || b == BranchIndirect.base()
              || b == BranchIndirectLink.base()
              || b == Ret.base()
              || b == BranchCond.base()
              || b == CbZ.base()
              || b == CbNz.base()
              || b == TbZ.base()
              || b == TbNz.base() => true,
            _ => false,
        }
    }

    pub const fn is_terminator(self) -> bool {
        use Op::*;
        matches!(self,
            Branch | BranchLink | BranchIndirect | BranchIndirectLink
            | Ret | BranchCond | CbZ | CbNz | TbZ | TbNz
            | Brk | Svc | Hvc
        )
    }

    pub const fn is_pure(self) -> bool {
        use Op::*;
        if self.has_side_effects() {
            return false;
        }
        if matches!(self,
            GetX | GetW | GetSp | GetNzcv | GetV | Mrs
        ) {
            return false;
        }
        match self.base() {
            b if b == Load8.base() => false,
            b if b == Load128 as u16 => false,
            b if b == LoadS8.base() => false,
            b if b == LoadAcq8.base() => false,
            b if b == LoadPair8.base() => false,
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sized_families_encode_size_in_low_bits() {
        let cases: &[(Op, u32)] = &[
            (Op::Add8,  1), (Op::Add16,  2), (Op::Add32,  4), (Op::Add64,  8),
            (Op::Sub8,  1), (Op::Sub64,  8),
            (Op::Lsl8,  1), (Op::Lsl16,  2), (Op::Lsl32,  4), (Op::Lsl64,  8),
            (Op::Lsr64, 8), (Op::Asr32,  4), (Op::Ror16,  2),
            (Op::Load8, 1), (Op::Load16, 2), (Op::Load32, 4), (Op::Load64, 8),
            (Op::Store8, 1), (Op::Store64, 8),
            (Op::Csel32, 4), (Op::Csel64, 8),
            (Op::AddsFlags32, 4), (Op::SubsFlags64, 8),
        ];
        for &(op, bytes) in cases {
            assert_eq!(op.size_bytes(), bytes,
                "{:?}: expected size_bytes={}, got {}", op, bytes, op.size_bytes());
            assert_eq!(op.size_bits(), bytes * 8);
        }
    }

    #[test]
    fn family_bases_are_4_aligned_and_match_across_sizes() {
        assert_eq!(Op::Add8.base(), Op::Add64.base());
        assert_eq!(Op::Sub8.base(), Op::Sub64.base());
        assert_eq!(Op::Lsl8.base(), Op::Lsl64.base());
        assert_eq!(Op::Load8.base(), Op::Load64.base());
        assert_eq!(Op::Store8.base(), Op::Store64.base());

        assert_ne!(Op::Add64.base(), Op::Sub64.base());
        assert_ne!(Op::Lsl32.base(), Op::Lsr32.base());
        assert_ne!(Op::Lsl32.base(), Op::Asr32.base());

        assert_eq!(Op::Add64 as u16 & 0b11, 3);
        assert_eq!(Op::Add32 as u16 & 0b11, 2);
        assert_eq!(Op::Add16 as u16 & 0b11, 1);
        assert_eq!(Op::Add8  as u16 & 0b11, 0);
    }
}
