use iced_x86::code_asm::*;

use crate::arch::{Cond, NUM_GPRS, ZR_ENCODING};
use crate::backend::abi::{
    ARG3_REG, CALL_PRECALL_SUB, CTX_REG, SCRATCH0, SCRATCH1, SCRATCH2, SCRATCH3,
};
use crate::backend::operand::{
    gpr32, gpr64, load32, load64, load_xmm_d, load_xmm_s, store32, store64,
    store_xmm_d, store_xmm_s,
};
use crate::backend::regalloc::{Allocation, Loc};
use crate::error::{Error, Result};
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};
use crate::jit::context::cpu_offsets;
use crate::jit::memory::{
    addr_mem_read8, addr_mem_read16, addr_mem_read32, addr_mem_read64,
    addr_mem_write8, addr_mem_write16, addr_mem_write32, addr_mem_write64,
};

pub fn emit_armlet(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    if a.is_eliminated() { return Ok(()); }

    let dst_vr: Option<ValueRef> = if a.ty != Ty::Void {
        Some(ValueRef::new(idx as u32))
    } else { None };

    match a.op {
        Op::Void => {}
        Op::Identity => {
            if let Some(d) = dst_vr {
                if alloc.loc(a.args[0]) != alloc.loc(d) {
                    if a.ty.bits() <= 32 {
                        load32(asm, alloc, a.args[0], eax)?;
                        store32(asm, alloc, d, eax)?;
                    } else {
                        load64(asm, alloc, a.args[0], SCRATCH0)?;
                        store64(asm, alloc, d, SCRATCH0)?;
                    }
                }
            }
        }

        Op::ConstU32 => {
            let d = dst_vr.unwrap();
            asm.mov(eax, (a.imm as u32) as i32)?;
            store32(asm, alloc, d, eax)?;
        }
        Op::ConstU64 => {
            let d = dst_vr.unwrap();
            asm.mov(SCRATCH0, a.imm as i64)?;
            store64(asm, alloc, d, SCRATCH0)?;
        }

        Op::GetX => {
            let d = dst_vr.unwrap();
            let reg = a.imm as usize;
            load_guest_x(asm, SCRATCH0, reg)?;
            store64(asm, alloc, d, SCRATCH0)?;
        }
        Op::GetW => {
            let d = dst_vr.unwrap();
            let reg = a.imm as usize;
            load_guest_x(asm, SCRATCH0, reg)?;
            store32(asm, alloc, d, eax)?;
        }
        Op::SetX => {
            let reg = a.imm as usize;
            load64(asm, alloc, a.args[0], SCRATCH0)?;
            store_guest_x(asm, reg, SCRATCH0)?;
        }
        Op::SetW => {
            let reg = a.imm as usize;
            load32(asm, alloc, a.args[0], eax)?;
            store_guest_x(asm, reg, SCRATCH0)?;
        }
        Op::GetSp => {
            let d = dst_vr.unwrap();
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::sp() as i32))?;
            store64(asm, alloc, d, SCRATCH0)?;
        }
        Op::SetSp => {
            load64(asm, alloc, a.args[0], SCRATCH0)?;
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::sp() as i32), SCRATCH0)?;
        }
        Op::GetNzcv => {
            let d = dst_vr.unwrap();
            asm.movzx(eax, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
            store32(asm, alloc, d, eax)?;
        }
        Op::SetNzcv => {
            load32(asm, alloc, a.args[0], eax)?;
            asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
        }
        Op::GetPc => {
            let d = dst_vr.unwrap();
            asm.mov(SCRATCH0, a.imm as i64)?;
            store64(asm, alloc, d, SCRATCH0)?;
        }

        Op::GetV => {
            let d = dst_vr.unwrap();
            let reg = a.imm as usize;
            let off = cpu_offsets::vreg(reg) as i32;
            if a.ty.bits() <= 32 {
                asm.mov(eax, dword_ptr(CTX_REG + off))?;
                store32(asm, alloc, d, eax)?;
            } else {
                asm.mov(SCRATCH0, qword_ptr(CTX_REG + off))?;
                store64(asm, alloc, d, SCRATCH0)?;
            }
        }
        Op::SetV => {
            let reg = a.imm as usize;
            let off = cpu_offsets::vreg(reg) as i32;
            if a.flags.contains(crate::ir::ArmletFlags::W_SIZED) {
                load32(asm, alloc, a.args[0], eax)?;
                asm.mov(dword_ptr(CTX_REG + off), eax)?;
                asm.mov(dword_ptr(CTX_REG + off + 4), 0i32)?;
            } else {
                load64(asm, alloc, a.args[0], SCRATCH0)?;
                asm.mov(qword_ptr(CTX_REG + off), SCRATCH0)?;
            }
            asm.mov(qword_ptr(CTX_REG + off + 8), 0i32)?;
        }

        Op::Add32 | Op::Add64 => emit_binop(asm, alloc, a, dst_vr, BinKind::Add,  a.op.size_bits())?,
        Op::Sub32 | Op::Sub64 => emit_binop(asm, alloc, a, dst_vr, BinKind::Sub,  a.op.size_bits())?,
        Op::And32 | Op::And64 => emit_binop(asm, alloc, a, dst_vr, BinKind::And,  a.op.size_bits())?,
        Op::Or32  | Op::Or64  => emit_binop(asm, alloc, a, dst_vr, BinKind::Or,   a.op.size_bits())?,
        Op::Eor32 | Op::Eor64 => emit_binop(asm, alloc, a, dst_vr, BinKind::Xor,  a.op.size_bits())?,
        Op::Mul32 | Op::Mul64 => emit_binop(asm, alloc, a, dst_vr, BinKind::Imul, a.op.size_bits())?,

        Op::Adc32 | Op::Adc64 => emit_adc_sbc(asm, alloc, a, dst_vr, false, a.op.size_bits())?,
        Op::Sbc32 | Op::Sbc64 => emit_adc_sbc(asm, alloc, a, dst_vr, true,  a.op.size_bits())?,

        Op::UDiv32 | Op::UDiv64 => emit_div(asm, alloc, a, dst_vr, false, a.op.size_bits())?,
        Op::SDiv32 | Op::SDiv64 => emit_div(asm, alloc, a, dst_vr, true,  a.op.size_bits())?,

        Op::Clz32  | Op::Clz64  => emit_clz(asm, alloc, a, dst_vr, a.op.size_bits())?,
        Op::Cls32  | Op::Cls64  => emit_cls(asm, alloc, a, dst_vr, a.op.size_bits())?,
        Op::Rbit32 | Op::Rbit64 => emit_rbit(asm, alloc, a, dst_vr, a.op.size_bits())?,
        Op::Rev16  => emit_rev16(asm, alloc, a, dst_vr, if a.ty == Ty::U64 { 64 } else { 32 })?,
        Op::Rev32  => emit_rev32_within64(asm, alloc, a, dst_vr)?,
        Op::Rev64  => emit_bswap(asm, alloc, a, dst_vr, if a.ty == Ty::U64 { 64 } else { 32 })?,

        Op::Lsl32 | Op::Lsl64 => emit_shift(asm, alloc, a, dst_vr, ShiftKind::Lsl, a.op.size_bits())?,
        Op::Lsr32 | Op::Lsr64 => emit_shift(asm, alloc, a, dst_vr, ShiftKind::Lsr, a.op.size_bits())?,
        Op::Asr32 | Op::Asr64 => emit_shift(asm, alloc, a, dst_vr, ShiftKind::Asr, a.op.size_bits())?,
        Op::Ror32 | Op::Ror64 => emit_shift(asm, alloc, a, dst_vr, ShiftKind::Ror, a.op.size_bits())?,

        Op::Not32 | Op::Not64 => emit_unop(asm, alloc, a, dst_vr, UnopKind::Not, a.op.size_bits())?,
        Op::Neg32 | Op::Neg64 => emit_unop(asm, alloc, a, dst_vr, UnopKind::Neg, a.op.size_bits())?,

        Op::AddsFlags32 | Op::AddsFlags64 | Op::SubsFlags32 | Op::SubsFlags64 => {
            emit_flagged_addsub(asm, alloc, a, dst_vr)?;
        }

        Op::Load8 | Op::Load16 | Op::Load32 | Op::Load64 =>
            emit_load(asm, alloc, a, dst_vr, a.op.size_bytes())?,
        Op::Store8 | Op::Store16 | Op::Store32 | Op::Store64 =>
            emit_store(asm, alloc, a, a.op.size_bytes())?,

        Op::LoadEx8 | Op::LoadEx16 | Op::LoadEx32 | Op::LoadEx64 =>
            emit_load_ex(asm, alloc, a, dst_vr, a.op.size_bytes())?,
        Op::StoreEx8 | Op::StoreEx16 | Op::StoreEx32 | Op::StoreEx64 =>
            emit_store_ex(asm, alloc, a, dst_vr, a.op.size_bytes())?,

        Op::Csel32 | Op::Csel64 => emit_csel(asm, alloc, a, dst_vr)?,

        Op::Fadd32 | Op::Fadd64 => emit_fbinop(asm, alloc, a, dst_vr, FpBinKind::Add, a.op.size_bits())?,
        Op::Fsub32 | Op::Fsub64 => emit_fbinop(asm, alloc, a, dst_vr, FpBinKind::Sub, a.op.size_bits())?,
        Op::Fmul32 | Op::Fmul64 => emit_fbinop(asm, alloc, a, dst_vr, FpBinKind::Mul, a.op.size_bits())?,
        Op::Fdiv32 | Op::Fdiv64 => emit_fbinop(asm, alloc, a, dst_vr, FpBinKind::Div, a.op.size_bits())?,
        Op::Fmax32 | Op::Fmax64 => emit_fbinop(asm, alloc, a, dst_vr, FpBinKind::Max, a.op.size_bits())?,
        Op::Fmin32 | Op::Fmin64 => emit_fbinop(asm, alloc, a, dst_vr, FpBinKind::Min, a.op.size_bits())?,
        Op::Fcmp32 | Op::Fcmp64 => emit_fcmp(asm, alloc, a, a.op.size_bits())?,
        Op::Fsqrt32 | Op::Fsqrt64 => emit_fsqrt(asm, alloc, a, dst_vr, a.op.size_bits())?,

        Op::FcvtZsSW => emit_fcvt_zs(asm, alloc, a, dst_vr, false, false)?,
        Op::FcvtZsSX => emit_fcvt_zs(asm, alloc, a, dst_vr, false, true)?,
        Op::FcvtZsDW => emit_fcvt_zs(asm, alloc, a, dst_vr, true,  false)?,
        Op::FcvtZsDX => emit_fcvt_zs(asm, alloc, a, dst_vr, true,  true)?,
        Op::ScvtfWS  => emit_scvtf(asm, alloc, a, dst_vr, false, false)?,
        Op::ScvtfXS  => emit_scvtf(asm, alloc, a, dst_vr, false, true)?,
        Op::ScvtfWD  => emit_scvtf(asm, alloc, a, dst_vr, true,  false)?,
        Op::ScvtfXD  => emit_scvtf(asm, alloc, a, dst_vr, true,  true)?,
        Op::FcvtSD   => emit_fcvt_precision(asm, alloc, a, dst_vr, false)?,
        Op::FcvtDS   => emit_fcvt_precision(asm, alloc, a, dst_vr, true)?,

        op if op.is_terminator() => {}

        Op::Hint | Op::MemoryBarrier => {}

        Op::Clrex => {
            asm.mov(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), 0i32)?;
        }

        Op::Mrs => emit_mrs(asm, alloc, a, dst_vr)?,
        Op::Msr => emit_msr(asm, alloc, a)?,

        other => return Err(Error::Unsupported {
            pc: block.start_pc,
            opcode: other as u32,
        }),
    }

    Ok(())
}

fn load_guest_x(asm: &mut CodeAssembler, dst: AsmRegister64, reg: usize) -> Result<()> {
    if reg == ZR_ENCODING as usize {
        asm.xor(dst, dst)?;
        return Ok(());
    }
    debug_assert!(reg < NUM_GPRS);
    let off = cpu_offsets::xreg(reg) as i32;
    asm.mov(dst, qword_ptr(CTX_REG + off))?;
    Ok(())
}

fn store_guest_x(asm: &mut CodeAssembler, reg: usize, src: AsmRegister64) -> Result<()> {
    if reg == ZR_ENCODING as usize {
        return Ok(());
    }
    debug_assert!(reg < NUM_GPRS);
    let off = cpu_offsets::xreg(reg) as i32;
    asm.mov(qword_ptr(CTX_REG + off), src)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum BinKind { Add, Sub, And, Or, Xor, Imul }

fn apply_bin_32(asm: &mut CodeAssembler, k: BinKind, l: AsmRegister32, r: AsmRegister32) -> Result<()> {
    match k {
        BinKind::Add  => asm.add(l, r)?,
        BinKind::Sub  => asm.sub(l, r)?,
        BinKind::And  => asm.and(l, r)?,
        BinKind::Or   => asm.or (l, r)?,
        BinKind::Xor  => asm.xor(l, r)?,
        BinKind::Imul => asm.imul_2(l, r)?,
    }
    Ok(())
}

fn apply_bin_64(asm: &mut CodeAssembler, k: BinKind, l: AsmRegister64, r: AsmRegister64) -> Result<()> {
    match k {
        BinKind::Add  => asm.add(l, r)?,
        BinKind::Sub  => asm.sub(l, r)?,
        BinKind::And  => asm.and(l, r)?,
        BinKind::Or   => asm.or (l, r)?,
        BinKind::Xor  => asm.xor(l, r)?,
        BinKind::Imul => asm.imul_2(l, r)?,
    }
    Ok(())
}

fn emit_binop(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, k: BinKind, bits: u32) -> Result<()> {
    if bits == 64 {
        if let Some(d) = dst {
            if let Loc::Reg(r) = alloc.loc(d) {
                if alloc.loc(a.args[0]) == Loc::Reg(r) {
                    load64(asm, alloc, a.args[1], SCRATCH1)?;
                    apply_bin_64(asm, k, gpr64(r), SCRATCH1)?;
                    return Ok(());
                }
            }
        }
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        load64(asm, alloc, a.args[1], SCRATCH1)?;
        apply_bin_64(asm, k, SCRATCH0, SCRATCH1)?;
        if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
    } else {
        if let Some(d) = dst {
            if let Loc::Reg(r) = alloc.loc(d) {
                if alloc.loc(a.args[0]) == Loc::Reg(r) {
                    load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
                    apply_bin_32(asm, k, gpr32(r), gpr32(scratch1_id()))?;
                    return Ok(());
                }
            }
        }
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
        apply_bin_32(asm, k, eax, gpr32(scratch1_id()))?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum UnopKind { Not, Neg }

fn emit_unop(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, k: UnopKind, bits: u32) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        match k {
            UnopKind::Not => asm.not(SCRATCH0)?,
            UnopKind::Neg => asm.neg(SCRATCH0)?,
        }
        if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        match k {
            UnopKind::Not => asm.not(eax)?,
            UnopKind::Neg => asm.neg(eax)?,
        }
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ShiftKind { Lsl, Lsr, Asr, Ror }

fn emit_shift(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, kind: ShiftKind, bits: u32) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        load64(asm, alloc, a.args[1], rcx)?;
        match kind {
            ShiftKind::Lsl => asm.shl(SCRATCH0, cl)?,
            ShiftKind::Lsr => asm.shr(SCRATCH0, cl)?,
            ShiftKind::Asr => asm.sar(SCRATCH0, cl)?,
            ShiftKind::Ror => asm.ror(SCRATCH0, cl)?,
        }
        if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], ecx)?;
        match kind {
            ShiftKind::Lsl => asm.shl(eax, cl)?,
            ShiftKind::Lsr => asm.shr(eax, cl)?,
            ShiftKind::Asr => asm.sar(eax, cl)?,
            ShiftKind::Ror => asm.ror(eax, cl)?,
        }
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_flagged_addsub(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>) -> Result<()> {
    let is_64 = matches!(a.op, Op::AddsFlags64 | Op::SubsFlags64);
    let is_sub = matches!(a.op, Op::SubsFlags32 | Op::SubsFlags64);

    if is_64 {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        load64(asm, alloc, a.args[1], SCRATCH1)?;
        if is_sub { asm.sub(SCRATCH0, SCRATCH1)?; }
        else      { asm.add(SCRATCH0, SCRATCH1)?; }
        if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
        if is_sub { asm.sub(eax, gpr32(scratch1_id()))?; }
        else      { asm.add(eax, gpr32(scratch1_id()))?; }
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }

    asm.sets(r8b)?;
    asm.sete(r9b)?;
    asm.setc(r10b)?;
    asm.seto(r11b)?;
    if is_sub { asm.xor(r10b, 1i32)?; }
    asm.movzx(eax, r8b)?;
    asm.shl(eax, 3i32)?;
    asm.movzx(ecx, r9b)?;
    asm.shl(ecx, 2i32)?;
    asm.or(eax, ecx)?;
    asm.movzx(ecx, r10b)?;
    asm.shl(ecx, 1i32)?;
    asm.or(eax, ecx)?;
    asm.movzx(ecx, r11b)?;
    asm.or(eax, ecx)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
    Ok(())
}

fn emit_adc_sbc(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    is_sub: bool,
    bits: u32,
) -> Result<()> {
    asm.bt(dword_ptr(CTX_REG + cpu_offsets::nzcv() as i32), 1i32)?;
    if is_sub { asm.cmc()?; }
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        load64(asm, alloc, a.args[1], rcx)?;
        if is_sub { asm.sbb(rax, rcx)?; } else { asm.adc(rax, rcx)?; }
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], ecx)?;
        if is_sub { asm.sbb(eax, ecx)?; } else { asm.adc(eax, ecx)?; }
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_div(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    signed: bool,
    bits: u32,
) -> Result<()> {
    let mut lbl_zero = asm.create_label();
    let mut lbl_done = asm.create_label();

    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        load64(asm, alloc, a.args[1], rcx)?;
        asm.test(rcx, rcx)?;
        asm.jz(lbl_zero)?;
        if signed {
            let mut lbl_do_div = asm.create_label();
            asm.cmp(rcx, -1i32)?;
            asm.jne(lbl_do_div)?;
            asm.mov(rdx, i64::MIN)?;
            asm.cmp(rax, rdx)?;
            asm.je(lbl_done)?;
            asm.set_label(&mut lbl_do_div)?;
            asm.cqo()?;
            asm.idiv(rcx)?;
        } else {
            asm.xor(rdx, rdx)?;
            asm.div(rcx)?;
        }
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_zero)?;
        asm.xor(rax, rax)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        load32(asm, alloc, a.args[1], ecx)?;
        asm.test(ecx, ecx)?;
        asm.jz(lbl_zero)?;
        if signed {
            let mut lbl_do_div = asm.create_label();
            asm.cmp(ecx, -1i32)?;
            asm.jne(lbl_do_div)?;
            asm.cmp(eax, i32::MIN)?;
            asm.je(lbl_done)?;
            asm.set_label(&mut lbl_do_div)?;
            asm.cdq()?;
            asm.idiv(ecx)?;
        } else {
            asm.xor(edx, edx)?;
            asm.div(ecx)?;
        }
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_zero)?;
        asm.xor(eax, eax)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_clz(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bits: u32) -> Result<()> {
    let mut lbl_zero = asm.create_label();
    let mut lbl_done = asm.create_label();
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.test(rax, rax)?;
        asm.jz(lbl_zero)?;
        asm.bsr(rcx, rax)?;
        asm.mov(rax, 63i64)?;
        asm.sub(rax, rcx)?;
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_zero)?;
        asm.mov(rax, 64i64)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.test(eax, eax)?;
        asm.jz(lbl_zero)?;
        asm.bsr(ecx, eax)?;
        asm.mov(eax, 31i32)?;
        asm.sub(eax, ecx)?;
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_zero)?;
        asm.mov(eax, 32i32)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_cls(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bits: u32) -> Result<()> {
    let mut lbl_all_same = asm.create_label();
    let mut lbl_done = asm.create_label();
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, rax)?;
        asm.shl(rax, 1i32)?;
        asm.xor(rax, rcx)?;
        asm.test(rax, rax)?;
        asm.jz(lbl_all_same)?;
        asm.bsr(rcx, rax)?;
        asm.mov(rax, 63i64)?;
        asm.sub(rax, rcx)?;
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_all_same)?;
        asm.mov(rax, 63i64)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.mov(ecx, eax)?;
        asm.shl(eax, 1i32)?;
        asm.xor(eax, ecx)?;
        asm.test(eax, eax)?;
        asm.jz(lbl_all_same)?;
        asm.bsr(ecx, eax)?;
        asm.mov(eax, 31i32)?;
        asm.sub(eax, ecx)?;
        asm.jmp(lbl_done)?;
        asm.set_label(&mut lbl_all_same)?;
        asm.mov(eax, 31i32)?;
        asm.set_label(&mut lbl_done)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_rbit(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bits: u32) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        rbit64_inplace(asm)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        rbit32_inplace(asm)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn rbit64_inplace(asm: &mut CodeAssembler) -> Result<()> {
    asm.mov(rcx, rax)?;
    asm.shr(rcx, 1i32)?;
    asm.mov(rdx, 0x5555_5555_5555_5555i64)?;
    asm.and(rcx, rdx)?;
    asm.and(rax, rdx)?;
    asm.shl(rax, 1i32)?;
    asm.or(rax, rcx)?;
    asm.mov(rcx, rax)?;
    asm.shr(rcx, 2i32)?;
    asm.mov(rdx, 0x3333_3333_3333_3333i64)?;
    asm.and(rcx, rdx)?;
    asm.and(rax, rdx)?;
    asm.shl(rax, 2i32)?;
    asm.or(rax, rcx)?;
    asm.mov(rcx, rax)?;
    asm.shr(rcx, 4i32)?;
    asm.mov(rdx, 0x0F0F_0F0F_0F0F_0F0Fi64)?;
    asm.and(rcx, rdx)?;
    asm.and(rax, rdx)?;
    asm.shl(rax, 4i32)?;
    asm.or(rax, rcx)?;
    asm.bswap(rax)?;
    Ok(())
}

fn rbit32_inplace(asm: &mut CodeAssembler) -> Result<()> {
    asm.mov(ecx, eax)?;
    asm.shr(ecx, 1i32)?;
    asm.and(ecx, 0x5555_5555_u32 as i32)?;
    asm.and(eax, 0x5555_5555_u32 as i32)?;
    asm.shl(eax, 1i32)?;
    asm.or(eax, ecx)?;
    asm.mov(ecx, eax)?;
    asm.shr(ecx, 2i32)?;
    asm.and(ecx, 0x3333_3333_u32 as i32)?;
    asm.and(eax, 0x3333_3333_u32 as i32)?;
    asm.shl(eax, 2i32)?;
    asm.or(eax, ecx)?;
    asm.mov(ecx, eax)?;
    asm.shr(ecx, 4i32)?;
    asm.and(ecx, 0x0F0F_0F0F_u32 as i32)?;
    asm.and(eax, 0x0F0F_0F0F_u32 as i32)?;
    asm.shl(eax, 4i32)?;
    asm.or(eax, ecx)?;
    asm.bswap(eax)?;
    Ok(())
}

fn emit_rev16(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bits: u32) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, rax)?;
        asm.shr(rcx, 8i32)?;
        asm.mov(rdx, 0x00FF_00FF_00FF_00FFi64)?;
        asm.and(rcx, rdx)?;
        asm.and(rax, rdx)?;
        asm.shl(rax, 8i32)?;
        asm.or(rax, rcx)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.mov(ecx, eax)?;
        asm.shr(ecx, 8i32)?;
        asm.and(ecx, 0x00FF_00FF_u32 as i32)?;
        asm.and(eax, 0x00FF_00FF_u32 as i32)?;
        asm.shl(eax, 8i32)?;
        asm.or(eax, ecx)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_rev32_within64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>) -> Result<()> {
    load64(asm, alloc, a.args[0], rax)?;
    asm.bswap(rax)?;
    asm.rol(rax, 32i32)?;
    if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    Ok(())
}

fn emit_bswap(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bits: u32) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.bswap(rax)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.bswap(eax)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

fn emit_load(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bytes: u32) -> Result<()> {
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.mov(ARG3_REG, CTX_REG)?;
    let fn_addr = match bytes {
        1 => addr_mem_read8(),
        2 => addr_mem_read16(),
        4 => addr_mem_read32(),
        8 => addr_mem_read64(),
        _ => return Err(Error::Backend("unsupported load width".into())),
    };
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.mov(SCRATCH0, fn_addr as i64)?;
    asm.call(SCRATCH0)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(d) = dst {
        match bytes {
            1 | 2 | 4 => store32(asm, alloc, d, eax)?,
            8         => store64(asm, alloc, d, rax)?,
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn emit_store(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, bytes: u32) -> Result<()> {
    if bytes == 8 {
        load64(asm, alloc, a.args[1], SCRATCH3)?;
    } else {
        load32(asm, alloc, a.args[1], gpr32(scratch3_id()))?;
    }
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.mov(ARG3_REG, CTX_REG)?;
    let fn_addr = match bytes {
        1 => addr_mem_write8(),
        2 => addr_mem_write16(),
        4 => addr_mem_write32(),
        8 => addr_mem_write64(),
        _ => return Err(Error::Backend("unsupported store width".into())),
    };
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.mov(SCRATCH0, fn_addr as i64)?;
    asm.call(SCRATCH0)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    Ok(())
}

fn emit_load_ex(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, bytes: u32) -> Result<()> {
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.mov(ARG3_REG, CTX_REG)?;
    let fn_addr = match bytes {
        1 => addr_mem_read8(),
        2 => addr_mem_read16(),
        4 => addr_mem_read32(),
        8 => addr_mem_read64(),
        _ => return Err(Error::Backend("unsupported ldex width".into())),
    };
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.mov(SCRATCH0, fn_addr as i64)?;
    asm.call(SCRATCH0)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    if let Some(d) = dst {
        match bytes {
            1 | 2 | 4 => store32(asm, alloc, d, eax)?,
            8         => store64(asm, alloc, d, rax)?,
            _ => unreachable!(),
        }
    }
    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::exclusive_addr() as i32), SCRATCH1)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), bytes as i32)?;
    Ok(())
}

fn emit_store_ex(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bytes: u32,
) -> Result<()> {
    let mut lbl_fail = asm.create_label();
    let mut lbl_done = asm.create_label();

    load64(asm, alloc, a.args[0], SCRATCH1)?;
    asm.cmp(qword_ptr(CTX_REG + cpu_offsets::exclusive_addr() as i32), SCRATCH1)?;
    asm.jne(lbl_fail)?;
    asm.cmp(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), bytes as i32)?;
    asm.jne(lbl_fail)?;

    if bytes == 8 {
        load64(asm, alloc, a.args[1], SCRATCH3)?;
    } else {
        load32(asm, alloc, a.args[1], gpr32(scratch3_id()))?;
    }
    asm.mov(ARG3_REG, CTX_REG)?;
    let fn_addr = match bytes {
        1 => addr_mem_write8(),
        2 => addr_mem_write16(),
        4 => addr_mem_write32(),
        8 => addr_mem_write64(),
        _ => return Err(Error::Backend("unsupported stex width".into())),
    };
    asm.sub(rsp, CALL_PRECALL_SUB)?;
    asm.mov(SCRATCH0, fn_addr as i64)?;
    asm.call(SCRATCH0)?;
    asm.add(rsp, CALL_PRECALL_SUB)?;
    asm.xor(eax, eax)?;
    asm.jmp(lbl_done)?;

    asm.set_label(&mut lbl_fail)?;
    asm.mov(eax, 1i32)?;

    asm.set_label(&mut lbl_done)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), 0i32)?;
    if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    Ok(())
}

fn emit_csel(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>) -> Result<()> {
    let cond = Cond::from_bits(a.imm as u8);
    let is_64 = matches!(a.op, Op::Csel64);

    load32(asm, alloc, a.args[2], edx)?;
    emit_cond_check_byte(asm, cond)?;
    asm.test(al, al)?;
    if is_64 {
        load64(asm, alloc, a.args[1], SCRATCH1)?;
        load64(asm, alloc, a.args[0], SCRATCH2)?;
        asm.cmovne(SCRATCH1, SCRATCH2)?;
        if let Some(d) = dst { store64(asm, alloc, d, SCRATCH1)?; }
    } else {
        load32(asm, alloc, a.args[1], eax)?;
        load32(asm, alloc, a.args[0], gpr32(scratch1_id()))?;
        asm.cmovne(eax, gpr32(scratch1_id()))?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

pub fn emit_cond_check_byte(asm: &mut CodeAssembler, cond: Cond) -> Result<()> {
    let tt = crate::arch::COND_TRUTH[cond as usize] as i32;
    asm.mov(eax, tt)?;
    asm.bt(eax, edx)?;
    asm.setc(al)?;
    Ok(())
}

#[inline]
fn scratch1_id() -> u8 {
    #[cfg(target_os = "windows")]
    { 2 }
    #[cfg(not(target_os = "windows"))]
    { 6 }
}

#[inline]
fn scratch3_id() -> u8 {
    #[cfg(target_os = "windows")]
    { 8 }
    #[cfg(not(target_os = "windows"))]
    { 2 }
}

#[derive(Clone, Copy)]
enum FpBinKind { Add, Sub, Mul, Div, Max, Min }

fn emit_fbinop(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    k: FpBinKind,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        load_xmm_d(asm, alloc, a.args[1], xmm1)?;
        match k {
            FpBinKind::Add => asm.addsd(xmm0, xmm1)?,
            FpBinKind::Sub => asm.subsd(xmm0, xmm1)?,
            FpBinKind::Mul => asm.mulsd(xmm0, xmm1)?,
            FpBinKind::Div => asm.divsd(xmm0, xmm1)?,
            FpBinKind::Max => asm.maxsd(xmm0, xmm1)?,
            FpBinKind::Min => asm.minsd(xmm0, xmm1)?,
        }
        if let Some(d) = dst { store_xmm_d(asm, alloc, d, xmm0)?; }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        load_xmm_s(asm, alloc, a.args[1], xmm1)?;
        match k {
            FpBinKind::Add => asm.addss(xmm0, xmm1)?,
            FpBinKind::Sub => asm.subss(xmm0, xmm1)?,
            FpBinKind::Mul => asm.mulss(xmm0, xmm1)?,
            FpBinKind::Div => asm.divss(xmm0, xmm1)?,
            FpBinKind::Max => asm.maxss(xmm0, xmm1)?,
            FpBinKind::Min => asm.minss(xmm0, xmm1)?,
        }
        if let Some(d) = dst { store_xmm_s(asm, alloc, d, xmm0)?; }
    }
    Ok(())
}

/// `FCVT` between single and double precision.
/// `src_is_double = false` → single → double (CVTSS2SD);
/// `src_is_double = true`  → double → single (CVTSD2SS).
fn emit_fcvt_precision(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    src_is_double: bool,
) -> Result<()> {
    if src_is_double {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        asm.cvtsd2ss(xmm0, xmm0)?;
        if let Some(d) = dst { store_xmm_s(asm, alloc, d, xmm0)?; }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        asm.cvtss2sd(xmm0, xmm0)?;
        if let Some(d) = dst { store_xmm_d(asm, alloc, d, xmm0)?; }
    }
    Ok(())
}

/// `FCVTZS` — FP → signed int with round-toward-zero (truncate).
/// `src_is_double` selects between SS/SD; `dst_is_x` selects 32 vs 64-bit
/// destination width.
fn emit_fcvt_zs(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    src_is_double: bool,
    dst_is_x: bool,
) -> Result<()> {
    if src_is_double {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        if dst_is_x { asm.cvttsd2si(rax, xmm0)?; }
        else        { asm.cvttsd2si(eax, xmm0)?; }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        if dst_is_x { asm.cvttss2si(rax, xmm0)?; }
        else        { asm.cvttss2si(eax, xmm0)?; }
    }
    if let Some(d) = dst {
        if dst_is_x { store64(asm, alloc, d, rax)?; }
        else        { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

/// `SCVTF` — signed int → FP. `src_is_x` selects 32 vs 64-bit source;
/// `dst_is_double` selects single vs double precision result.
fn emit_scvtf(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    dst_is_double: bool,
    src_is_x: bool,
) -> Result<()> {
    if src_is_x {
        load64(asm, alloc, a.args[0], rax)?;
        if dst_is_double { asm.cvtsi2sd(xmm0, rax)?; }
        else             { asm.cvtsi2ss(xmm0, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        if dst_is_double { asm.cvtsi2sd(xmm0, eax)?; }
        else             { asm.cvtsi2ss(xmm0, eax)?; }
    }
    if let Some(d) = dst {
        if dst_is_double { store_xmm_d(asm, alloc, d, xmm0)?; }
        else             { store_xmm_s(asm, alloc, d, xmm0)?; }
    }
    Ok(())
}

fn emit_fsqrt(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        asm.sqrtsd(xmm0, xmm0)?;
        if let Some(d) = dst { store_xmm_d(asm, alloc, d, xmm0)?; }
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        asm.sqrtss(xmm0, xmm0)?;
        if let Some(d) = dst { store_xmm_s(asm, alloc, d, xmm0)?; }
    }
    Ok(())
}

/// FCMP: maps x86 EFLAGS (after UCOMISS/UCOMISD) to ARM NZCV nibble.
///
/// ARM FCMP sets NZCV as:
///   GT  → 0010, LT  → 1000, EQ  → 0110, Unord → 0011.
/// x86 UCOMIS* leaves:
///   GT  → ZF=0,PF=0,CF=0; LT → CF=1; EQ → ZF=1; Unord → PF=ZF=CF=1.
/// We mask CF/ZF with `!PF` so LT/EQ collapse to 0 in the unordered case,
/// then synthesise C = !N and pack the nibble.
fn emit_fcmp(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, bits: u32) -> Result<()> {
    if bits == 64 {
        load_xmm_d(asm, alloc, a.args[0], xmm0)?;
        load_xmm_d(asm, alloc, a.args[1], xmm1)?;
        asm.ucomisd(xmm0, xmm1)?;
    } else {
        load_xmm_s(asm, alloc, a.args[0], xmm0)?;
        load_xmm_s(asm, alloc, a.args[1], xmm1)?;
        asm.ucomiss(xmm0, xmm1)?;
    }
    // Capture EFLAGS bits BEFORE any arithmetic — AND/OR/XOR all clear CF/OF.
    asm.setp(r8b)?;
    asm.setnp(cl)?;
    asm.setz(r9b)?;
    asm.setc(r10b)?;
    // Now compute. PF set ⇒ unordered, so mask ZF and CF with !PF.
    asm.and(r9b, cl)?;       // Z = ZF & !PF
    asm.and(r10b, cl)?;      // N = CF & !PF
    asm.mov(r11b, r10b)?;
    asm.xor(r11b, 1i32)?;    // C = !N
    asm.shl(r10b, 3i32)?;
    asm.shl(r9b, 2i32)?;
    asm.shl(r11b, 1i32)?;
    asm.or(r10b, r9b)?;
    asm.or(r10b, r11b)?;
    asm.or(r10b, r8b)?;      // V = PF
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), r10b)?;
    Ok(())
}

fn emit_mrs(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
) -> Result<()> {
    use crate::arch::sysreg;
    let d = match dst { Some(d) => d, None => return Ok(()) };
    let id = a.imm as u16;
    match id {
        sysreg::TPIDR_EL0 => {
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::tpidr_el0() as i32))?;
        }
        sysreg::TPIDRRO_EL0 => {
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::tpidrro_el0() as i32))?;
        }
        sysreg::NZCV => {
            asm.movzx(eax, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
            asm.shl(rax, 28i32)?;
        }
        sysreg::FPCR => {
            asm.mov(eax, dword_ptr(CTX_REG + cpu_offsets::fpcr() as i32))?;
        }
        sysreg::FPSR => {
            asm.mov(eax, dword_ptr(CTX_REG + cpu_offsets::fpsr() as i32))?;
        }
        sysreg::CTR_EL0 => {
            asm.mov(SCRATCH0, 0x8444_8004u64 as i64)?;
        }
        sysreg::DCZID_EL0 => {
            asm.mov(SCRATCH0, 0x4i64)?;
        }
        sysreg::MIDR_EL1 => {
            asm.mov(SCRATCH0, 0x412F_D050u64 as i64)?;
        }
        sysreg::MPIDR_EL1 => {
            asm.mov(SCRATCH0, 0x8000_0000u64 as i64)?;
        }
        sysreg::CNTFRQ_EL0 => {
            asm.mov(SCRATCH0, 1_000_000_000i64)?;
        }
        sysreg::CNTVCT_EL0 => {
            asm.xor(rax, rax)?;
        }
        _ => return Err(Error::Unsupported {
            pc: 0,
            opcode: ((Op::Mrs as u32) << 16) | id as u32,
        }),
    }
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}

fn emit_msr(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
) -> Result<()> {
    use crate::arch::sysreg;
    let id = a.imm as u16;
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    match id {
        sysreg::TPIDR_EL0 => {
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::tpidr_el0() as i32), SCRATCH0)?;
        }
        sysreg::TPIDRRO_EL0 => {
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::tpidrro_el0() as i32), SCRATCH0)?;
        }
        sysreg::NZCV => {
            asm.shr(rax, 28i32)?;
            asm.and(eax, 0xFi32)?;
            asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
        }
        sysreg::FPCR => {
            asm.mov(dword_ptr(CTX_REG + cpu_offsets::fpcr() as i32), eax)?;
        }
        sysreg::FPSR => {
            asm.mov(dword_ptr(CTX_REG + cpu_offsets::fpsr() as i32), eax)?;
        }
        _ => {}
    }
    Ok(())
}
