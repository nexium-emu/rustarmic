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
