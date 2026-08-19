use crate::arch::{Cond, RegSize, ZR_ENCODING};
use crate::ir::{Armlet, ArmletFlags, Block, Op, Terminal, Ty, ValueRef};

pub struct IrEmitter<'b> {
    pub block: &'b mut Block,
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

    #[inline]
    pub fn const_u32(&mut self, v: u32) -> ValueRef {
        self.push(Armlet::new(Op::ConstU32, Ty::U32).with_imm(v as u64))
    }

    #[inline]
    pub fn const_u64(&mut self, v: u64) -> ValueRef {
        self.push(Armlet::new(Op::ConstU64, Ty::U64).with_imm(v))
    }

    pub fn get_x(&mut self, reg: u8) -> ValueRef {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING {
            return self.const_u64(0);
        }
        self.push(Armlet::new(Op::GetX, Ty::U64).with_imm(reg as u64))
    }

    pub fn get_w(&mut self, reg: u8) -> ValueRef {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING {
            return self.const_u32(0);
        }
        self.push(Armlet::new(Op::GetW, Ty::U32).with_imm(reg as u64))
    }

    pub fn get_gpr(&mut self, reg: u8, size: RegSize) -> ValueRef {
        match size {
            RegSize::W => self.get_w(reg),
            RegSize::X => self.get_x(reg),
        }
    }

    pub fn set_x(&mut self, reg: u8, value: ValueRef) {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING {
            return;
        }
        self.push(
            Armlet::new(Op::SetX, Ty::Void)
                .with_args(&[value])
                .with_imm(reg as u64),
        );
    }

    pub fn set_w(&mut self, reg: u8, value: ValueRef) {
        debug_assert!(reg < 32);
        if reg == ZR_ENCODING {
            return;
        }
        self.push(
            Armlet::new(Op::SetW, Ty::Void)
                .with_args(&[value])
                .with_imm(reg as u64)
                .with_flags(ArmletFlags::W_SIZED),
        );
    }

    pub fn set_gpr(&mut self, reg: u8, value: ValueRef, size: RegSize) {
        match size {
            RegSize::W => self.set_w(reg, value),
            RegSize::X => self.set_x(reg, value),
        }
    }

    pub fn get_sp(&mut self) -> ValueRef {
        self.push(Armlet::new(Op::GetSp, Ty::U64))
    }

    pub fn get_x_or_sp(&mut self, reg: u8, sp_form: bool) -> ValueRef {
        if reg == ZR_ENCODING {
            if sp_form {
                self.get_sp()
            } else {
                self.const_u64(0)
            }
        } else {
            self.get_x(reg)
        }
    }

    pub fn set_sp(&mut self, value: ValueRef) {
        self.push(Armlet::new(Op::SetSp, Ty::Void).with_args(&[value]));
    }

    pub fn set_x_or_sp(&mut self, reg: u8, value: ValueRef, sp_form: bool) {
        if reg == ZR_ENCODING {
            if sp_form {
                self.set_sp(value);
            }
        } else {
            self.set_x(reg, value);
        }
    }

    /// Write an integer result, preserving the architectural W-register
    /// zero-extension while still supporting SP when encoded as register 31.
    pub fn set_gpr_or_sp(&mut self, reg: u8, value: ValueRef, size: RegSize, sp_form: bool) {
        if reg == ZR_ENCODING {
            if sp_form {
                self.set_sp(value);
            }
        } else {
            self.set_gpr(reg, value, size);
        }
    }

    pub fn get_v_s(&mut self, reg: u8) -> ValueRef {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::GetV, Ty::U32).with_imm(reg as u64))
    }

    pub fn get_v_d(&mut self, reg: u8) -> ValueRef {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::GetV, Ty::U64).with_imm(reg as u64))
    }

    pub fn set_v_s(&mut self, reg: u8, value: ValueRef) {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(
            Armlet::new(Op::SetV, Ty::Void)
                .with_args(&[value])
                .with_imm(reg as u64)
                .with_flags(ArmletFlags::W_SIZED),
        );
    }

    pub fn set_v_d(&mut self, reg: u8, value: ValueRef) {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(
            Armlet::new(Op::SetV, Ty::Void)
                .with_args(&[value])
                .with_imm(reg as u64),
        );
    }

    pub fn get_v_q(&mut self, reg: u8) -> ValueRef {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(Armlet::new(Op::GetV, Ty::U128).with_imm(reg as u64))
    }

    pub fn set_v_q(&mut self, reg: u8, value: ValueRef) {
        debug_assert!((reg as usize) < crate::arch::NUM_VREGS);
        self.push(
            Armlet::new(Op::SetV, Ty::Void)
                .with_args(&[value])
                .with_imm(reg as u64),
        );
    }

    pub fn vec_build_q(&mut self, lo: ValueRef, hi: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecBuildQ, Ty::U128).with_args(&[lo, hi]))
    }

    pub fn vec_extract_lo64(&mut self, q: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecExtractLo64, Ty::U64).with_args(&[q]))
    }

    pub fn vec_extract_hi64(&mut self, q: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecExtractHi64, Ty::U64).with_args(&[q]))
    }

    pub fn vec_extract_u8(&mut self, q: ValueRef, lane: u32) -> ValueRef {
        debug_assert!(lane < 16);
        self.push(
            Armlet::new(Op::VecExtract8, Ty::U32)
                .with_args(&[q])
                .with_imm(lane as u64),
        )
    }

    pub fn vec_extract_u16(&mut self, q: ValueRef, lane: u32) -> ValueRef {
        debug_assert!(lane < 8);
        self.push(
            Armlet::new(Op::VecExtract16, Ty::U32)
                .with_args(&[q])
                .with_imm(lane as u64),
        )
    }

    pub fn vec_extract_u32(&mut self, q: ValueRef, lane: u32) -> ValueRef {
        debug_assert!(lane < 4);
        self.push(
            Armlet::new(Op::VecExtract32, Ty::U32)
                .with_args(&[q])
                .with_imm(lane as u64),
        )
    }

    pub fn vec_add(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecAdd8,
            1 => Op::VecAdd16,
            2 => Op::VecAdd32,
            3 => Op::VecAdd64,
            _ => unreachable!("bad lane_log2"),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_sub(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSub8,
            1 => Op::VecSub16,
            2 => Op::VecSub32,
            3 => Op::VecSub64,
            _ => unreachable!("bad lane_log2"),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_and(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecAnd, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_orr(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecOrr, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_eor(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecEor, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_bic(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecBic, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_orn(&mut self, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecOrn, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_neg(&mut self, vn: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecNeg8,
            1 => Op::VecNeg16,
            2 => Op::VecNeg32,
            3 => Op::VecNeg64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn])
                .with_imm(q as u64),
        )
    }

    pub fn vec_abs(&mut self, vn: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecAbs8,
            1 => Op::VecAbs16,
            2 => Op::VecAbs32,
            3 => Op::VecAbs64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn])
                .with_imm(q as u64),
        )
    }

    pub fn vec_not(&mut self, vn: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecNot, Ty::U128)
                .with_args(&[vn])
                .with_imm(q as u64),
        )
    }

    pub fn vec_mul(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecMul8,
            1 => Op::VecMul16,
            2 => Op::VecMul32,
            3 => Op::VecMul64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_shl_imm(&mut self, vn: ValueRef, lane_log2: u32, shift: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecShlImm8,
            1 => Op::VecShlImm16,
            2 => Op::VecShlImm32,
            3 => Op::VecShlImm64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((shift as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }
    pub fn vec_ushr_imm(&mut self, vn: ValueRef, lane_log2: u32, shift: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecUshrImm8,
            1 => Op::VecUshrImm16,
            2 => Op::VecUshrImm32,
            3 => Op::VecUshrImm64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((shift as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }
    pub fn vec_sshr_imm(&mut self, vn: ValueRef, lane_log2: u32, shift: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSshrImm8,
            1 => Op::VecSshrImm16,
            2 => Op::VecSshrImm32,
            3 => Op::VecSshrImm64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((shift as u64) << 1);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }

    pub fn vec_cmeq(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmEq8,
            1 => Op::VecCmEq16,
            2 => Op::VecCmEq32,
            3 => Op::VecCmEq64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_cmgt(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmGt8,
            1 => Op::VecCmGt16,
            2 => Op::VecCmGt32,
            3 => Op::VecCmGt64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_cmge(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmGe8,
            1 => Op::VecCmGe16,
            2 => Op::VecCmGe32,
            3 => Op::VecCmGe64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_cmhi(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmHi8,
            1 => Op::VecCmHi16,
            2 => Op::VecCmHi32,
            3 => Op::VecCmHi64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_cmhs(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecCmHs8,
            1 => Op::VecCmHs16,
            2 => Op::VecCmHs32,
            3 => Op::VecCmHs64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_bit(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecBit, Ty::U128)
                .with_args(&[vd_prev, vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_bif(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecBif, Ty::U128)
                .with_args(&[vd_prev, vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_bsl(&mut self, vd_prev: ValueRef, vn: ValueRef, vm: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecBsl, Ty::U128)
                .with_args(&[vd_prev, vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_dup_gpr(&mut self, gpr_val: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecDupGpr8,
            1 => Op::VecDupGpr16,
            2 => Op::VecDupGpr32,
            3 => Op::VecDupGpr64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[gpr_val])
                .with_imm(q as u64),
        )
    }

    pub fn vec_ins_gpr(
        &mut self,
        vd_prev: ValueRef,
        gpr_val: ValueRef,
        lane_log2: u32,
        lane: u32,
        q: bool,
    ) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecInsGpr8,
            1 => Op::VecInsGpr16,
            2 => Op::VecInsGpr32,
            3 => Op::VecInsGpr64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((lane as u64) << 1);
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vd_prev, gpr_val])
                .with_imm(imm),
        )
    }

    pub fn vec_ext(&mut self, vn: ValueRef, vm: ValueRef, byte_off: u32, q: bool) -> ValueRef {
        let imm = (q as u64) | ((byte_off as u64) << 1);
        self.push(
            Armlet::new(Op::VecExt, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }

    pub fn vec_zip1(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecZip1_8,
            1 => Op::VecZip1_16,
            2 => Op::VecZip1_32,
            3 => Op::VecZip1_64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_zip2(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecZip2_8,
            1 => Op::VecZip2_16,
            2 => Op::VecZip2_32,
            3 => Op::VecZip2_64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_smin(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSmin8,
            1 => Op::VecSmin16,
            2 => Op::VecSmin32,
            3 => Op::VecSmin64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_smax(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecSmax8,
            1 => Op::VecSmax16,
            2 => Op::VecSmax32,
            3 => Op::VecSmax64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_umin(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecUmin8,
            1 => Op::VecUmin16,
            2 => Op::VecUmin32,
            3 => Op::VecUmin64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_umax(&mut self, vn: ValueRef, vm: ValueRef, lane_log2: u32, q: bool) -> ValueRef {
        let op = match lane_log2 {
            0 => Op::VecUmax8,
            1 => Op::VecUmax16,
            2 => Op::VecUmax32,
            3 => Op::VecUmax64,
            _ => unreachable!(),
        };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
    }

    pub fn vec_addv32(&mut self, vn: ValueRef) -> ValueRef {
        self.push(Armlet::new(Op::VecAddv32, Ty::U32).with_args(&[vn]))
    }

    fn vec_fbin(
        &mut self,
        op_s: Op,
        op_d: Op,
        double: bool,
        vn: ValueRef,
        vm: ValueRef,
        q: bool,
    ) -> ValueRef {
        let op = if double { op_d } else { op_s };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(q as u64),
        )
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

    pub fn vec_fmla(
        &mut self,
        vd_prev: ValueRef,
        vn: ValueRef,
        vm: ValueRef,
        double: bool,
        q: bool,
    ) -> ValueRef {
        let op = if double { Op::VecFmla_D } else { Op::VecFmla_S };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vd_prev, vn, vm])
                .with_imm(q as u64),
        )
    }
    pub fn vec_fmls(
        &mut self,
        vd_prev: ValueRef,
        vn: ValueRef,
        vm: ValueRef,
        double: bool,
        q: bool,
    ) -> ValueRef {
        let op = if double { Op::VecFmls_D } else { Op::VecFmls_S };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vd_prev, vn, vm])
                .with_imm(q as u64),
        )
    }

    fn vec_frint(&mut self, op_s: Op, op_d: Op, double: bool, vn: ValueRef, q: bool) -> ValueRef {
        let op = if double { op_d } else { op_s };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn])
                .with_imm(q as u64),
        )
    }
    pub fn vec_frintn(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_frint(Op::VecFRintN_S, Op::VecFRintN_D, double, vn, q)
    }
    pub fn vec_frintm(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_frint(Op::VecFRintM_S, Op::VecFRintM_D, double, vn, q)
    }
    pub fn vec_frintp(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_frint(Op::VecFRintP_S, Op::VecFRintP_D, double, vn, q)
    }
    pub fn vec_frintz(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_frint(Op::VecFRintZ_S, Op::VecFRintZ_D, double, vn, q)
    }
    pub fn vec_frinta(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_frint(Op::VecFRintA_S, Op::VecFRintA_D, double, vn, q)
    }
    pub fn vec_frintx(&mut self, vn: ValueRef, double: bool, q: bool) -> ValueRef {
        self.vec_frint(Op::VecFRintX_S, Op::VecFRintX_D, double, vn, q)
    }

    fn vec_funop(&mut self, op_s: Op, op_d: Op, double: bool, vn: ValueRef, q: bool) -> ValueRef {
        let op = if double { op_d } else { op_s };
        self.push(
            Armlet::new(op, Ty::U128)
                .with_args(&[vn])
                .with_imm(q as u64),
        )
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

    pub fn vec_saddl(
        &mut self,
        vn: ValueRef,
        vm: ValueRef,
        src_lane_log2: u32,
        high_half: bool,
    ) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(
            Armlet::new(Op::VecSaddl, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }
    pub fn vec_uaddl(
        &mut self,
        vn: ValueRef,
        vm: ValueRef,
        src_lane_log2: u32,
        high_half: bool,
    ) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(
            Armlet::new(Op::VecUaddl, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }
    pub fn vec_ssubl(
        &mut self,
        vn: ValueRef,
        vm: ValueRef,
        src_lane_log2: u32,
        high_half: bool,
    ) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(
            Armlet::new(Op::VecSsubl, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }
    pub fn vec_usubl(
        &mut self,
        vn: ValueRef,
        vm: ValueRef,
        src_lane_log2: u32,
        high_half: bool,
    ) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(
            Armlet::new(Op::VecUsubl, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }
    pub fn vec_smull(
        &mut self,
        vn: ValueRef,
        vm: ValueRef,
        src_lane_log2: u32,
        high_half: bool,
    ) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(
            Armlet::new(Op::VecSmull, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }
    pub fn vec_umull(
        &mut self,
        vn: ValueRef,
        vm: ValueRef,
        src_lane_log2: u32,
        high_half: bool,
    ) -> ValueRef {
        let imm = ((high_half as u64) << 1) | ((src_lane_log2 as u64) << 2);
        self.push(
            Armlet::new(Op::VecUmull, Ty::U128)
                .with_args(&[vn, vm])
                .with_imm(imm),
        )
    }

    pub fn vec_xtn(&mut self, vn: ValueRef, src_lane_log2: u32) -> ValueRef {
        let imm = (src_lane_log2 as u64) << 2;
        self.push(
            Armlet::new(Op::VecXtn, Ty::U128)
                .with_args(&[vn])
                .with_imm(imm),
        )
    }
    pub fn vec_xtn2(&mut self, vd_prev: ValueRef, vn: ValueRef, src_lane_log2: u32) -> ValueRef {
        let imm = (src_lane_log2 as u64) << 2;
        self.push(
            Armlet::new(Op::VecXtn2, Ty::U128)
                .with_args(&[vd_prev, vn])
                .with_imm(imm),
        )
    }

    pub fn vec_tbl(&mut self, table: ValueRef, indices: ValueRef, q: bool) -> ValueRef {
        self.push(
            Armlet::new(Op::VecTbl, Ty::U128)
                .with_args(&[table, indices])
                .with_imm(q as u64),
        )
    }
    pub fn vec_tbl2(
        &mut self,
        table0: ValueRef,
        table1: ValueRef,
        indices: ValueRef,
        q: bool,
    ) -> ValueRef {
        self.push(
            Armlet::new(Op::VecTbl2, Ty::U128)
                .with_args(&[table0, table1, indices])
                .with_imm(q as u64),
        )
    }
    pub fn vec_tbl3(
        &mut self,
        table0: ValueRef,
        table1: ValueRef,
        table2: ValueRef,
        indices: ValueRef,
        q: bool,
    ) -> ValueRef {
        self.push(
            Armlet::new(Op::VecTbl3, Ty::U128)
                .with_args(&[table0, table1, table2, indices])
                .with_imm(q as u64),
        )
    }

    pub fn vec_rev(
        &mut self,
        vn: ValueRef,
        src_lane_log2: u32,
        container_log2: u32,
        q: bool,
    ) -> ValueRef {
        let op = match container_log2 {
            1 => Op::VecRev16,
            2 => Op::VecRev32,
            3 => Op::VecRev64,
            _ => unreachable!(),
        };
        let imm = (q as u64) | ((src_lane_log2 as u64) << 2);
        self.push(Armlet::new(op, Ty::U128).with_args(&[vn]).with_imm(imm))
    }

    fn vec_perm(
        &mut self,
        op: Op,
        vn: ValueRef,
        vm: ValueRef,
        lane_log2: u32,
        q: bool,
    ) -> ValueRef {
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

    pub fn get_nzcv(&mut self) -> ValueRef {
        self.push(Armlet::new(Op::GetNzcv, Ty::Nzcv))
    }

    pub fn set_nzcv(&mut self, value: ValueRef) {
        self.push(Armlet::new(Op::SetNzcv, Ty::Void).with_args(&[value]));
    }

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
        self.push(
            Armlet::new(op, ty)
                .with_args(&[a, b])
                .with_flags(ArmletFlags::NZCV_LIVE),
        )
    }

    pub fn subs(&mut self, a: ValueRef, b: ValueRef, size: RegSize) -> ValueRef {
        let (op, ty) = match size {
            RegSize::W => (Op::SubsFlags32, Ty::U32),
            RegSize::X => (Op::SubsFlags64, Ty::U64),
        };
        self.push(
            Armlet::new(op, ty)
                .with_args(&[a, b])
                .with_flags(ArmletFlags::NZCV_LIVE),
        )
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

    pub fn load(&mut self, addr: ValueRef, size_bytes: u32) -> ValueRef {
        let (op, ty) = match size_bytes {
            1 => (Op::Load8, Ty::U8),
            2 => (Op::Load16, Ty::U16),
            4 => (Op::Load32, Ty::U32),
            8 => (Op::Load64, Ty::U64),
            16 => (Op::Load128, Ty::U128),
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, ty).with_args(&[addr]))
    }

    pub fn store(&mut self, addr: ValueRef, value: ValueRef, size_bytes: u32) {
        let op = match size_bytes {
            1 => Op::Store8,
            2 => Op::Store16,
            4 => Op::Store32,
            8 => Op::Store64,
            16 => Op::Store128,
            _ => unreachable!(),
        };
        self.push(Armlet::new(op, Ty::Void).with_args(&[addr, value]));
    }

    pub fn branch(&mut self, target_pc: u64, link: bool) {
        let op = if link { Op::BranchLink } else { Op::Branch };
        if link {
            let ret_addr = self.const_u64(self.current_pc.wrapping_add(4));
            self.set_x(30, ret_addr);
        }
        self.push(Armlet::new(op, Ty::Void).with_imm(target_pc));
        self.block.terminal = Terminal::DirectBranch { target_pc, link };
    }

    pub fn branch_cond(&mut self, cond: Cond, target_pc: u64) {
        let nzcv = self.get_nzcv();
        self.push(
            Armlet::new(Op::BranchCond, Ty::Void)
                .with_args(&[nzcv])
                .with_imm((target_pc << 8) | (cond as u64)),
        );
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
        self.block.terminal = Terminal::IndirectBranch {
            target,
            link,
            is_ret,
        };
    }
}
