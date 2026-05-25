//! Convenience builder used by the AArch64 translator.
//!
//! Wraps a `&mut Block` and exposes a typed surface for each common opcode.
//! The intent is to keep `frontend/translate/*` files readable while still
//! producing the same flat `Vec<Armlet>` layout the optimizer expects.

use crate::arch::{Cond, RegSize, ZR_ENCODING};
use crate::ir::{Armlet, ArmletFlags, Block, Op, Terminal, Ty, ValueRef};

pub struct IrEmitter<'b> {
    pub block: &'b mut Block,
    /// PC of the guest instruction currently being translated.
    pub current_pc: u64,
}

impl<'b> IrEmitter<'b> {
    #[inline]
    pub fn new(block: &'b mut Block, current_pc: u64) -> Self {
        Self { block, current_pc }
    }

    #[inline]
    pub fn push(&mut self, armlet: Armlet) -> ValueRef {
        self.block.push(armlet)
    }

    // ─── Constants ──────────────────────────────────────────────────────────
    #[inline]
    pub fn const_u32(&mut self, v: u32) -> ValueRef {
        self.push(Armlet::new(Op::ConstU32, Ty::U32).with_imm(v as u64))
    }

    #[inline]
    pub fn const_u64(&mut self, v: u64) -> ValueRef {
        self.push(Armlet::new(Op::ConstU64, Ty::U64).with_imm(v))
    }

    // ─── GPR access ─────────────────────────────────────────────────────────
    /// Read a 64-bit GPR. Encoding 31 reads as the zero register.
    pub fn get_x(&mut self, reg: u8) -> ValueRef {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING {
            return self.const_u64(0);
        }
        self.push(Armlet::new(Op::GetX, Ty::U64).with_imm(reg as u64))
    }

    /// Read a 32-bit GPR view. Encoding 31 reads as zero.
    pub fn get_w(&mut self, reg: u8) -> ValueRef {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING {
            return self.const_u32(0);
        }
        self.push(Armlet::new(Op::GetW, Ty::U32).with_imm(reg as u64))
    }

    /// Generic read sized by RegSize.
    pub fn get_gpr(&mut self, reg: u8, size: RegSize) -> ValueRef {
        match size {
            RegSize::W => self.get_w(reg),
            RegSize::X => self.get_x(reg),
        }
    }

    /// Write 64-bit value to GPR. Encoding 31 silently discards (WZR/XZR).
    pub fn set_x(&mut self, reg: u8, value: ValueRef) {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING { return; }
        self.push(Armlet::new(Op::SetX, Ty::Void)
            .with_args(&[value])
            .with_imm(reg as u64));
    }

    /// Write 32-bit value to GPR — top half of X register zero-extends.
    pub fn set_w(&mut self, reg: u8, value: ValueRef) {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING { return; }
        self.push(Armlet::new(Op::SetW, Ty::Void)
            .with_args(&[value])
            .with_imm(reg as u64)
            .with_flags(ArmletFlags::W_SIZED));
    }

    pub fn set_gpr(&mut self, reg: u8, value: ValueRef, size: RegSize) {
        match size {
            RegSize::W => self.set_w(reg, value),
            RegSize::X => self.set_x(reg, value),
        }
    }

    /// Read SP.
    pub fn get_sp(&mut self) -> ValueRef {
        self.push(Armlet::new(Op::GetSp, Ty::U64))
    }

    /// Read either SP or zero register depending on whether SP-encoding is allowed.
    /// When `reg==31`, returns SP if `sp_form` is true, otherwise ZR (constant 0).
    pub fn get_x_or_sp(&mut self, reg: u8, sp_form: bool) -> ValueRef {
        if reg == ZR_ENCODING {
            if sp_form { self.get_sp() } else { self.const_u64(0) }
        } else {
            self.get_x(reg)
        }
    }

    pub fn set_sp(&mut self, value: ValueRef) {
        self.push(Armlet::new(Op::SetSp, Ty::Void).with_args(&[value]));
    }

    pub fn set_x_or_sp(&mut self, reg: u8, value: ValueRef, sp_form: bool) {
        if reg == ZR_ENCODING {
            if sp_form { self.set_sp(value); }
            // else discard (XZR)
        } else {
            self.set_x(reg, value);
        }
    }

    // ─── V (SIMD/FP) register lanes ─────────────────────────────────────────
    /// Read the low 32 bits of V[reg] (S-precision scalar view).
    pub fn get_v_s(&mut self, reg: u8) -> ValueRef {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::GetV, Ty::U32).with_imm(reg as u64))
    }

    /// Read the low 64 bits of V[reg] (D-precision scalar view).
    pub fn get_v_d(&mut self, reg: u8) -> ValueRef {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::GetV, Ty::U64).with_imm(reg as u64))
    }

    /// Write 32-bit lane to V[reg], zeroing the upper 96 bits.
    pub fn set_v_s(&mut self, reg: u8, value: ValueRef) {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::SetV, Ty::Void)
            .with_args(&[value])
            .with_imm(reg as u64)
            .with_flags(ArmletFlags::W_SIZED));
    }

    /// Write 64-bit lane to V[reg], zeroing the upper 64 bits.
    pub fn set_v_d(&mut self, reg: u8, value: ValueRef) {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::SetV, Ty::Void)
            .with_args(&[value])
            .with_imm(reg as u64));
    }

    /// Read the full 128 bits of V[reg] (Q-precision vector view).
    pub fn get_v_q(&mut self, reg: u8) -> ValueRef {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::GetV, Ty::U128).with_imm(reg as u64))
    }

    /// Write all 128 bits of V[reg]. `value` must be a Ty::U128 value.
    pub fn set_v_q(&mut self, reg: u8, value: ValueRef) {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::SetV, Ty::Void)
            .with_args(&[value])
            .with_imm(reg as u64));
    }

    /// Combine two u64 halves into a u128 (lo = bits 0..63, hi = bits 64..127).
    pub fn vec_build_q(&mut self, lo: ValueRef, hi: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecBuildQ, Ty::U128).with_args(&[lo, hi]))
    }

    /// Extract the low 64 bits of a u128.
    pub fn vec_extract_lo64(&mut self, q: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecExtractLo64, Ty::U64).with_args(&[q]))
    }

    /// Extract the high 64 bits of a u128.
    pub fn vec_extract_hi64(&mut self, q: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecExtractHi64, Ty::U64).with_args(&[q]))
    }

    /// Extract an 8-bit lane (lane 0..15) of a u128, zero-extended to U32.
    pub fn vec_extract_u8(&mut self, q: ValueRef, lane: u32) -> ValueRef {
        debug_assert!(lane < 16);
        self.push(Armlet::new(Op::VecExtract8, Ty::U32).with_args(&[q]).with_imm(lane as u64))
    }

    /// Extract a 16-bit lane (lane 0..7) of a u128, zero-extended to U32.
    pub fn vec_extract_u16(&mut self, q: ValueRef, lane: u32) -> ValueRef {
        debug_assert!(lane < 8);
        self.push(Armlet::new(Op::VecExtract16, Ty::U32).with_args(&[q]).with_imm(lane as u64))
    }

    /// Extract a 32-bit lane (lane 0..3) of a u128.
    pub fn vec_extract_u32(&mut self, q: ValueRef, lane: u32) -> ValueRef {
        debug_assert!(lane < 4);
        self.push(Armlet::new(Op::VecExtract32, Ty::U32).with_args(&[q]).with_imm(lane as u64))
    }

    // ─── Per-lane vector ALU ────────────────────────────────────────────────
    // `lane_log2` selects element width (0=B, 1=H, 2=S, 3=D). `q` selects the
    // full 128-bit form (true) vs the half-width form (false, upper 64 zeroed).

    pub fn vec_add(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecAdd8, 1 => Op::VecAdd16, 2 => Op::VecAdd32, 3 => Op::VecAdd64,
            _ => unreachable!("bad lane_log2"),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    pub fn vec_sub(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSub8, 1 => Op::VecSub16, 2 => Op::VecSub32, 3 => Op::VecSub64,
            _ => unreachable!("bad lane_log2"),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    pub fn vec_and(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecAnd, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_orr(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecOrr, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_eor(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecEor, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_bic(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecBic, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_orn(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecOrn, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    pub fn vec_neg(&mut self, vn: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecNeg8, 1 => Op::VecNeg16, 2 => Op::VecNeg32, 3 => Op::VecNeg64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(q as u64))
    }

    pub fn vec_abs(&mut self, vn: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecAbs8, 1 => Op::VecAbs16, 2 => Op::VecAbs32, 3 => Op::VecAbs64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(q as u64))
    }

    pub fn vec_not(&mut self, vn: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecNot, Ty::U128).with_args(&[vn]).with_imm(q as u64))
    }

    pub fn vec_mul(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecMul8, 1 => Op::VecMul16, 2 => Op::VecMul32, 3 => Op::VecMul64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    /// Immediate shift. `shift` is the shift amount (0..lane_bits); the
    /// Q-flag lives in imm bit 0 with the shift amount in bits 1..8.
    pub fn vec_shl_imm(&mut self, vn: ValueRef, lane_log2: u32, shift: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecShlImm8, 1 => Op::VecShlImm16, 2 => Op::VecShlImm32, 3 => Op::VecShlImm64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((shift as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }
    pub fn vec_ushr_imm(&mut self, vn: ValueRef, lane_log2: u32, shift: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecUshrImm8, 1 => Op::VecUshrImm16, 2 => Op::VecUshrImm32, 3 => Op::VecUshrImm64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((shift as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }
    pub fn vec_sshr_imm(&mut self, vn: ValueRef, lane_log2: u32, shift: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSshrImm8, 1 => Op::VecSshrImm16, 2 => Op::VecSshrImm32, 3 => Op::VecSshrImm64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((shift as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }

    // ─── Per-lane compares ──────────────────────────────────────────────────
    pub fn vec_cmeq(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmEq8, 1 => Op::VecCmEq16, 2 => Op::VecCmEq32, 3 => Op::VecCmEq64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_cmgt(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmGt8, 1 => Op::VecCmGt16, 2 => Op::VecCmGt32, 3 => Op::VecCmGt64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_cmge(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmGe8, 1 => Op::VecCmGe16, 2 => Op::VecCmGe32, 3 => Op::VecCmGe64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_cmhi(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmHi8, 1 => Op::VecCmHi16, 2 => Op::VecCmHi32, 3 => Op::VecCmHi64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_cmhs(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmHs8, 1 => Op::VecCmHs16, 2 => Op::VecCmHs32, 3 => Op::VecCmHs64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    /// Bit-select. ARM BIT/BIF/BSL all read Vd as one of the inputs, so the
    /// IR op takes three sources: (vd_prev, vn, vm).
    pub fn vec_bit(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecBit, Ty::U128).with_args(&[vd_prev, vn, vm]).with_imm(q as u64))
    }
    pub fn vec_bif(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecBif, Ty::U128).with_args(&[vd_prev, vn, vm]).with_imm(q as u64))
    }
    pub fn vec_bsl(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecBsl, Ty::U128).with_args(&[vd_prev, vn, vm]).with_imm(q as u64))
    }

    /// Broadcast a scalar GPR value to all lanes of the result vector.
    /// `gpr_val` must be a U32 (for 8/16/32-bit lanes) or U64 (for 64-bit).
    pub fn vec_dup_gpr(&mut self, gpr_val: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecDupGpr8, 1 => Op::VecDupGpr16, 2 => Op::VecDupGpr32, 3 => Op::VecDupGpr64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[gpr_val]).with_imm(q as u64))
    }

    /// Insert a scalar GPR value into a specific lane of `vd_prev`.
    /// The lane index lives in `imm` bits 1..8; the Q-flag is bit 0.
    pub fn vec_ins_gpr(&mut self, vd_prev: ValueRef, gpr_val: ValueRef, lane_log2: u32, lane: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecInsGpr8, 1 => Op::VecInsGpr16, 2 => Op::VecInsGpr32, 3 => Op::VecInsGpr64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((lane as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vd_prev, gpr_val]).with_imm(imm))
    }

    /// EXT: concat(vm, vn) shifted right by `byte_off` bytes; low 16 written.
    pub fn vec_ext(&mut self, vn: ValueRef, vm: ValueRef, byte_off: u32, q: bool) -> ValueRef {
        let imm = (q as u64) | ((byte_off as u64) << 1);
        self.push(Armlet::new(Op::VecExt, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }

    pub fn vec_zip1(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecZip1_8, 1 => Op::VecZip1_16, 2 => Op::VecZip1_32, 3 => Op::VecZip1_64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_zip2(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecZip2_8, 1 => Op::VecZip2_16, 2 => Op::VecZip2_32, 3 => Op::VecZip2_64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    pub fn vec_smin(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSmin8, 1 => Op::VecSmin16, 2 => Op::VecSmin32, 3 => Op::VecSmin64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_smax(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSmax8, 1 => Op::VecSmax16, 2 => Op::VecSmax32, 3 => Op::VecSmax64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_umin(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecUmin8, 1 => Op::VecUmin16, 2 => Op::VecUmin32, 3 => Op::VecUmin64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }
    pub fn vec_umax(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecUmax8, 1 => Op::VecUmax16, 2 => Op::VecUmax32, 3 => Op::VecUmax64,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    /// ADDV.4S: horizontal sum of 4 32-bit lanes; result is a U32 scalar.
    pub fn vec_addv32(&mut self, vn: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecAddv32, Ty::U32).with_args(&[vn]))
    }

    // ─── Per-lane FP ────────────────────────────────────────────────────────
    fn vec_fbin(&mut self, op_s: Op, op_d: Op, double: bool, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        let op = if double { op_d } else { op_s };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(q as u64))
    }

    pub fn vec_fadd(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFAdd_S, Op::VecFAdd_D, double, vn, vm, q)
    }
    pub fn vec_fsub(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFSub_S, Op::VecFSub_D, double, vn, vm, q)
    }
    pub fn vec_fmul(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFMul_S, Op::VecFMul_D, double, vn, vm, q)
    }
    pub fn vec_fdiv(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFDiv_S, Op::VecFDiv_D, double, vn, vm, q)
    }
    pub fn vec_fmax(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFMax_S, Op::VecFMax_D, double, vn, vm, q)
    }
    pub fn vec_fmin(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFMin_S, Op::VecFMin_D, double, vn, vm, q)
    }
    pub fn vec_fcmeq(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFCmEq_S, Op::VecFCmEq_D, double, vn, vm, q)
    }
    pub fn vec_fcmgt(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFCmGt_S, Op::VecFCmGt_D, double, vn, vm, q)
    }
    pub fn vec_fcmge(&mut self, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_fbin(Op::VecFCmGe_S, Op::VecFCmGe_D, double, vn, vm, q)
    }

    /// FMLA: vd_prev + vn * vm (composed mul-add, not fused).
    pub fn vec_fmla(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        let op = if double { Op::VecFmla_D } else { Op::VecFmla_S };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vd_prev, vn, vm]).with_imm(q as u64))
    }
    pub fn vec_fmls(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, double: bool, q: bool) -> ValueRef {
        let op = if double { Op::VecFmls_D } else { Op::VecFmls_S };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vd_prev, vn, vm]).with_imm(q as u64))
    }

    fn vec_funop(&mut self, op_s: Op, op_d: Op, double: bool, vn: ValueRef, q: bool) -> ValueRef {
        let op = if double { op_d } else { op_s };
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(q as u64))
    }
    pub fn vec_fneg(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_funop(Op::VecFNeg_S, Op::VecFNeg_D, double, vn, q)
    }
    pub fn vec_fabs(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_funop(Op::VecFAbs_S, Op::VecFAbs_D, double, vn, q)
    }
    pub fn vec_fsqrt(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_funop(Op::VecFSqrt_S, Op::VecFSqrt_D, double, vn, q)
    }

    /// Widening add. `src_lane_log2` is 0..=2 (B/H/S); result lane is one
    /// wider (H/S/D). `high_half=true` reads bytes 8..16 of each source.
    pub fn vec_saddl(&mut self, vn: ValueRef, vm: ValueRef, src_lane_log2: u32, high_half: bool) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(Op::VecSaddl, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }
    pub fn vec_uaddl(&mut self, vn: ValueRef, vm: ValueRef, src_lane_log2: u32, high_half: bool) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(Op::VecUaddl, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }
    pub fn vec_ssubl(&mut self, vn: ValueRef, vm: ValueRef, src_lane_log2: u32, high_half: bool) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(Op::VecSsubl, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }
    pub fn vec_usubl(&mut self, vn: ValueRef, vm: ValueRef, src_lane_log2: u32, high_half: bool) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(Op::VecUsubl, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }
    pub fn vec_smull(&mut self, vn: ValueRef, vm: ValueRef, src_lane_log2: u32, high_half: bool) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(Op::VecSmull, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }
    pub fn vec_umull(&mut self, vn: ValueRef, vm: ValueRef, src_lane_log2: u32, high_half: bool) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(Op::VecUmull, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }

    /// Narrowing truncate. `src_lane_log2` is 1..=3 (H/S/D); result lane is
    /// one narrower (B/H/S). Result is packed in the low 64 bits; upper 64
    /// is zeroed.
    pub fn vec_xtn(&mut self, vn: ValueRef, src_lane_log2: u32) -> ValueRef {
        let imm = (src_lane_log2 as u64) << 2;
        self.push(Armlet::new(Op::VecXtn, Ty::U128).with_args(&[vn]).with_imm(imm))
    }
    /// XTN2: same narrowing as XTN but writes result to the UPPER 64 bits
    /// and preserves the LOW 64 from `vd_prev`.
    pub fn vec_xtn2(&mut self, vd_prev: ValueRef, vn: ValueRef, src_lane_log2: u32) -> ValueRef {
        let imm = (src_lane_log2 as u64) << 2;
        self.push(Armlet::new(Op::VecXtn2, Ty::U128).with_args(&[vd_prev, vn]).with_imm(imm))
    }

    /// TBL with a single-register table. Each byte of `indices` selects a
    /// byte from `table`; values >= 16 produce zero.
    pub fn vec_tbl(&mut self, table: ValueRef, indices: ValueRef, q: bool) -> ValueRef {
        self.push(Armlet::new(Op::VecTbl, Ty::U128).with_args(&[table, indices]).with_imm(q as u64))
    }

    /// REV16/32/64 family. `src_lane_log2` is the byte-level reversal
    /// granularity (0=B, 1=H, 2=S); `container_log2` selects which Rev op
    /// to use (1=H container/Rev16, 2=S/Rev32, 3=D/Rev64).
    pub fn vec_rev(&mut self, vn: ValueRef, src_lane_log2: u32, container_log2: u32, q: bool) -> ValueRef {
        let op = match container_log2 {
            1 => Op::VecRev16,
            2 => Op::VecRev32,
            3 => Op::VecRev64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }

    fn vec_perm(&mut self, op: Op, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let imm = (q as u64) | ((lane_log2 as u64) << 2);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn, vm]).with_imm(imm))
    }
    pub fn vec_uzp1(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        self.vec_perm(Op::VecUzp1, vn, vm, lane_log2, q)
    }
    pub fn vec_uzp2(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        self.vec_perm(Op::VecUzp2, vn, vm, lane_log2, q)
    }
    pub fn vec_trn1(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        self.vec_perm(Op::VecTrn1, vn, vm, lane_log2, q)
    }
    pub fn vec_trn2(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        self.vec_perm(Op::VecTrn2, vn, vm, lane_log2, q)
    }

    // ─── NZCV ───────────────────────────────────────────────────────────────
    pub fn get_nzcv(&mut self) -> ValueRef {
        self.push(Armlet::new(Op::GetNzcv, Ty::Nzcv))
    }

    pub fn set_nzcv(&mut self, value: ValueRef) {
        self.push(Armlet::new(Op::SetNzcv, Ty::Void).with_args(&[value]));
    }

    // ─── Integer ALU ────────────────────────────────────────────────────────
    pub fn add(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Add32, Ty::U32),
            RegSize::X => (Op::Add64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, b]))
    }

    pub fn sub(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Sub32, Ty::U32),
            RegSize::X => (Op::Sub64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, b]))
    }

    pub fn adds(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::AddsFlags32, Ty::U32),
            RegSize::X => (Op::AddsFlags64, Ty::U64),
        };
        self.push(Armlet::new(op, ty)
            .with_args(&[a, b])
            .with_flags(ArmletFlags::NZCV_LIVE))
    }

    pub fn subs(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::SubsFlags32, Ty::U32),
            RegSize::X => (Op::SubsFlags64, Ty::U64),
        };
        self.push(Armlet::new(op, ty)
            .with_args(&[a, b])
            .with_flags(ArmletFlags::NZCV_LIVE))
    }

    pub fn and(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::And32, Ty::U32),
            RegSize::X => (Op::And64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, b]))
    }

    pub fn or(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Or32, Ty::U32),
            RegSize::X => (Op::Or64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, b]))
    }

    pub fn eor(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Eor32, Ty::U32),
            RegSize::X => (Op::Eor64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, b]))
    }

    pub fn lsl(&mut self, a: ValueRef, amt: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Lsl32, Ty::U32),
            RegSize::X => (Op::Lsl64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, amt]))
    }

    pub fn lsr(&mut self, a: ValueRef, amt: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Lsr32, Ty::U32),
            RegSize::X => (Op::Lsr64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, amt]))
    }

    pub fn asr(&mut self, a: ValueRef, amt: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Asr32, Ty::U32),
            RegSize::X => (Op::Asr64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, amt]))
    }

    pub fn ror(&mut self, a: ValueRef, amt: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::Ror32, Ty::U32),
            RegSize::X => (Op::Ror64, Ty::U64),
        };
        self.push(Armlet::new(op, ty).with_args(&[a, amt]))
    }

    // ─── Memory ─────────────────────────────────────────────────────────────
    pub fn load(&mut self, addr: ValueRef, size_bytes: u32) -> ValueRef {
        let (op, ty) = match size_bytes {
            1  => (Op::Load8,   Ty::U8),
            2  => (Op::Load16,  Ty::U16),
            4  => (Op::Load32,  Ty::U32),
            8  => (Op::Load64,  Ty::U64),
            16 => (Op::Load128, Ty::U128),
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, ty).with_args(&[addr]))
    }

    pub fn store(&mut self, addr: ValueRef, value: ValueRef, size_bytes: u32) {
        let op = match size_bytes {
            1  => Op::Store8,
            2  => Op::Store16,
            4  => Op::Store32,
            8  => Op::Store64,
            16 => Op::Store128,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::Void).with_args(&[addr, value]));
    }

    // ─── Branches ───────────────────────────────────────────────────────────
    /// Direct unconditional branch. Sets the block terminal.
    pub fn branch(&mut self, target_pc: u64, link: bool) {
        let op = if link { Op::BranchLink } else { Op::Branch };
        if link {
            // BL: stash return address into X30.
            let ret_addr = self.const_u64(self.current_pc.wrapping_add(4));
            self.set_x(30, ret_addr);
        }
        self.push(Armlet::new(op, Ty::Void).with_imm(target_pc));
        self.block.terminal = Terminal::DirectBranch { target_pc, link };
    }

    /// Conditional direct branch — falls through to `current_pc + 4` if cond fails.
    pub fn branch_cond(&mut self, cond: Cond, target_pc: u64) {
        let nzcv = self.get_nzcv();
        self.push(Armlet::new(Op::BranchCond, Ty::Void)
            .with_args(&[nzcv])
            .with_imm(((target_pc as u64) << 8) | (cond as u64)));
        self.block.terminal = Terminal::ConditionalBranch {
            cond_nzcv: nzcv,
            cond_code: cond as u8,
            taken_pc: target_pc,
            not_taken_pc: self.current_pc.wrapping_add(4),
        };
    }

    pub fn branch_indirect(&mut self, target: ValueRef, link: bool, is_ret: bool) {
        let op = if is_ret {
            Op::Ret
        } else if link {
            Op::BranchIndirectLink
        } else {
            Op::BranchIndirect
        };
        if link {
            let ret_addr = self.const_u64(self.current_pc.wrapping_add(4));
            self.set_x(30, ret_addr);
        }
        self.push(Armlet::new(op, Ty::Void).with_args(&[target]));
        self.block.terminal = Terminal::IndirectBranch { target, link, is_ret };
    }
}
