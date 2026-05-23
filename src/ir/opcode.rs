//! Armlet opcodes.
//!
//! Kept as a single `#[repr(u16)]` enum so it sits inside the 32-byte Armlet
//! without bloating it. New opcodes go in the matching category — keep the
//! list dense and avoid holes so the compiler can build dense match tables.

#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    // ─── Pseudo / SSA bookkeeping ───────────────────────────────────────────
    /// No-op, produced when an instruction has been DCE'd in place.
    Void = 0,
    /// %dst = %src — used by copy propagation. Lowered to nothing by the backend.
    Identity,
    /// 32-bit immediate (value packed in `Armlet::imm`).
    ConstU32,
    /// 64-bit immediate (value packed in `Armlet::imm`).
    ConstU64,
    /// 128-bit immediate (low half in `imm`, high half in `args[3]` re-used as raw bits — see armlet.rs helpers).
    ConstU128,

    // ─── Guest CPU state I/O ────────────────────────────────────────────────
    /// Read 64-bit guest GPR. `imm` = register encoding (0..=30).
    GetX,
    /// Write 64-bit guest GPR. `args[0]` = value, `imm` = reg.
    SetX,
    /// Read 32-bit view of guest GPR (low 32 bits). `imm` = reg.
    GetW,
    /// Write 32-bit value into guest GPR, zero-extending the top half. `args[0]` = value, `imm` = reg.
    SetW,
    /// Read SP.
    GetSp,
    /// Write SP. `args[0]` = value.
    SetSp,
    /// Read NZCV as a 4-bit packed value.
    GetNzcv,
    /// Write NZCV. `args[0]` = packed nibble (U8 or NZCV-typed).
    SetNzcv,
    /// Read guest PC. `imm` = absolute guest PC value (PC is statically known per armlet).
    GetPc,
    /// Read 128-bit vector reg. `imm` = reg (0..=31).
    GetV,
    /// Write 128-bit vector reg. `args[0]` = value, `imm` = reg.
    SetV,

    // ─── Integer ALU ────────────────────────────────────────────────────────
    /// %dst = %a + %b (no flags).
    Add32, Add64,
    /// %dst = %a - %b (no flags).
    Sub32, Sub64,
    /// %dst = %a + %b + carry. `args[2]` = carry (U1).
    Adc32, Adc64,
    /// %dst = %a - %b - !carry. `args[2]` = carry (U1).
    Sbc32, Sbc64,
    /// Like Add/Sub but produces an NZCV-typed sibling via `GetNzcvFromOp` consumers.
    AddsFlags32, AddsFlags64,
    SubsFlags32, SubsFlags64,

    And32, And64,
    Or32, Or64,
    Eor32, Eor64,
    Bic32, Bic64,
    Orn32, Orn64,
    Eon32, Eon64,
    Not32, Not64,
    Neg32, Neg64,

    /// Logical shifts and rotates. `args[1]` = shift amount (U8 or U32/U64).
    Lsl32, Lsl64,
    Lsr32, Lsr64,
    Asr32, Asr64,
    Ror32, Ror64,

    /// Bitfield ops. `imm` = (immr << 8) | imms.
    Ubfm32, Ubfm64,
    Sbfm32, Sbfm64,
    Bfm32,  Bfm64,
    /// EXTR (also covers ROR-imm at translate time). `args[0]` = hi, `args[1]` = lo, `imm` = lsb.
    Extr32, Extr64,

    Mul32, Mul64,
    /// MADD: %a * %b + %c.
    Madd32, Madd64,
    /// MSUB: %c - %a * %b.
    Msub32, Msub64,
    /// 64×64 → upper 64 of 128-bit product.
    UMulH64, SMulH64,
    /// 32×32 → 64.
    UMull32, SMull32,
    /// 32×32 → 64 then add 64.
    UMAddl, SMAddl,
    /// 32×32 → 64 then subtract from 64.
    UMSubl, SMSubl,

    UDiv32, UDiv64,
    SDiv32, SDiv64,

    /// Zero/Sign extend. The destination type encodes the target width.
    Zext, Sext,

    Clz32, Clz64,
    Cls32, Cls64,
    Rbit32, Rbit64,
    Rev16, Rev32, Rev64,

    // ─── Compare / select ───────────────────────────────────────────────────
    /// Conditional select — `args[0]`=true val, `args[1]`=false val, `imm` low byte = Cond, `args[2]`=NZCV.
    Csel32, Csel64,
    Csinc32, Csinc64,
    Csinv32, Csinv64,
    Csneg32, Csneg64,

    /// CCMP / CCMN. `args[0]`=a, `args[1]`=b, `args[2]`=NZCV, `imm` low byte=Cond, next byte=nzcv to use on fail.
    CcmpReg32, CcmpReg64,
    CcmpImm32, CcmpImm64,
    CcmnReg32, CcmnReg64,
    CcmnImm32, CcmnImm64,

    // ─── Branches / terminators ─────────────────────────────────────────────
    /// Unconditional direct branch. `imm` = absolute target PC. Always terminator.
    Branch,
    /// BL — link register set then branch. `imm` = target PC. Stores return PC into X30.
    BranchLink,
    /// Indirect branch through register value. `args[0]` = target.
    BranchIndirect,
    /// BLR — indirect with link.
    BranchIndirectLink,
    /// RET — indirect, hints x86 return stack.
    Ret,
    /// Conditional branch. `args[0]` = NZCV. `imm` low byte = Cond, high 56 bits = target PC.
    BranchCond,
    /// CBZ / CBNZ. `args[0]` = test value, `imm` = target PC, flags bit indicates inverse.
    CbZ, CbNz,
    /// TBZ / TBNZ. `args[0]` = test value, `imm` = (target_pc << 8) | bit_index, flags indicate inverse.
    TbZ, TbNz,

    // ─── Memory ─────────────────────────────────────────────────────────────
    /// Load. `args[0]` = address (U64). Destination type encodes width.
    Load8, Load16, Load32, Load64, Load128,
    /// Sign-extending load. Destination width controlled by `Ty`.
    LoadS8, LoadS16, LoadS32,
    /// Store. `args[0]` = address (U64), `args[1]` = value.
    Store8, Store16, Store32, Store64, Store128,

    /// Acquire/release variants. Backend inserts host fences as needed.
    LoadAcq32, LoadAcq64,
    StoreRel32, StoreRel64,

    /// Exclusive load. `args[0]` = address. Latches address into context.
    LoadEx32, LoadEx64,
    /// Exclusive store. `args[0]` = address, `args[1]` = value. Returns U32 (0=success).
    StoreEx32, StoreEx64,

    /// Pair load/store. `args[0]` = address, `args[1]` = (store: value2 / load: unused).
    /// Two destination types implied: for load the result is U128 holding {hi,lo}; for store
    /// the values come from args[1] (lo) and args[2] (hi).
    LoadPair32, LoadPair64,
    StorePair32, StorePair64,

    // ─── FP / NEON (initial subset) ─────────────────────────────────────────
    Fmov32, Fmov64,
    Fadd32, Fadd64,
    Fsub32, Fsub64,
    Fmul32, Fmul64,
    Fdiv32, Fdiv64,
    Fneg32, Fneg64,
    Fabs32, Fabs64,
    Fsqrt32, Fsqrt64,
    Fcmp32, Fcmp64,

    /// Vector lane ops (initial — extend later).
    VecAdd, VecSub, VecMul, VecAnd, VecOr, VecEor,
    VecDup,
    /// INS / UMOV / SMOV.
    Ins, Umov, Smov,

    // ─── System ─────────────────────────────────────────────────────────────
    /// Read system register. `imm` = encoded sysreg id.
    Mrs,
    /// Write system register.
    Msr,
    /// Hint (NOP/YIELD/WFE/...). `imm` = hint code.
    Hint,
    /// BRK — software breakpoint. Terminator.
    Brk,
    /// SVC — supervisor call. Terminator (returns to host).
    Svc,
    /// HVC — hypervisor call. Terminator.
    Hvc,
    /// DMB/DSB/ISB. `imm` = barrier kind.
    MemoryBarrier,

    // ─── Sentinel ───────────────────────────────────────────────────────────
    /// One past the last opcode — used for dense table sizing. NEVER construct.
    __Count,
}

impl Op {
    /// True if the op may produce externally-observable side effects.
    /// DCE must not eliminate side-effecting armlets even when their result is unused.
    #[inline]
    pub const fn has_side_effects(self) -> bool {
        use Op::*;
        matches!(self,
            SetX | SetW | SetSp | SetNzcv | SetV
            | Store8 | Store16 | Store32 | Store64 | Store128
            | StoreRel32 | StoreRel64
            | StoreEx32 | StoreEx64
            | StorePair32 | StorePair64
            | LoadEx32 | LoadEx64
            | Msr | Brk | Svc | Hvc | Hint | MemoryBarrier
            | Branch | BranchLink | BranchIndirect | BranchIndirectLink
            | Ret | BranchCond | CbZ | CbNz | TbZ | TbNz
        )
    }

    /// True if the op is a block terminator.
    #[inline]
    pub const fn is_terminator(self) -> bool {
        use Op::*;
        matches!(self,
            Branch | BranchLink | BranchIndirect | BranchIndirectLink
            | Ret | BranchCond | CbZ | CbNz | TbZ | TbNz
            | Brk | Svc | Hvc
        )
    }

    /// True if the op is pure (no side effects, deterministic in its args).
    /// Used by constant-folding/value-numbering inside the optimizer.
    #[inline]
    pub const fn is_pure(self) -> bool {
        !self.has_side_effects() && !matches!(self,
            Op::GetX | Op::GetW | Op::GetSp | Op::GetNzcv | Op::GetV
            | Op::Load8 | Op::Load16 | Op::Load32 | Op::Load64 | Op::Load128
            | Op::LoadS8 | Op::LoadS16 | Op::LoadS32
            | Op::LoadAcq32 | Op::LoadAcq64
            | Op::LoadPair32 | Op::LoadPair64
            | Op::Mrs
        )
    }
}
