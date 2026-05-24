use iced_x86::code_asm::*;

use crate::arch::{Cond, NUM_GPRS, ZR_ENCODING};
use crate::backend::abi::{
    ARG3_REG, CALL_PRECALL_SUB, CTX_REG, SCRATCH0, SCRATCH1, SCRATCH2, SCRATCH3,
};
use crate::backend::operand::{gpr32, load32, load64, store32, store64};
use crate::backend::regalloc::Allocation;
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

        Op::Add32 => emit_binop_32(asm, alloc, a, dst_vr, BinKind::Add)?,
        Op::Add64 => emit_binop_64(asm, alloc, a, dst_vr, BinKind::Add)?,
        Op::Sub32 => emit_binop_32(asm, alloc, a, dst_vr, BinKind::Sub)?,
        Op::Sub64 => emit_binop_64(asm, alloc, a, dst_vr, BinKind::Sub)?,
        Op::And32 => emit_binop_32(asm, alloc, a, dst_vr, BinKind::And)?,
        Op::And64 => emit_binop_64(asm, alloc, a, dst_vr, BinKind::And)?,
        Op::Or32  => emit_binop_32(asm, alloc, a, dst_vr, BinKind::Or)?,
        Op::Or64  => emit_binop_64(asm, alloc, a, dst_vr, BinKind::Or)?,
        Op::Eor32 => emit_binop_32(asm, alloc, a, dst_vr, BinKind::Xor)?,
        Op::Eor64 => emit_binop_64(asm, alloc, a, dst_vr, BinKind::Xor)?,
        Op::Mul32 => emit_binop_32(asm, alloc, a, dst_vr, BinKind::Imul)?,
        Op::Mul64 => emit_binop_64(asm, alloc, a, dst_vr, BinKind::Imul)?,

        Op::Adc32 => emit_adc_sbc(asm, alloc, a, dst_vr, false, false)?,
        Op::Adc64 => emit_adc_sbc(asm, alloc, a, dst_vr, false, true)?,
        Op::Sbc32 => emit_adc_sbc(asm, alloc, a, dst_vr, true,  false)?,
        Op::Sbc64 => emit_adc_sbc(asm, alloc, a, dst_vr, true,  true)?,

        Op::UDiv32 => emit_div(asm, alloc, a, dst_vr, false, false)?,
        Op::UDiv64 => emit_div(asm, alloc, a, dst_vr, false, true)?,
        Op::SDiv32 => emit_div(asm, alloc, a, dst_vr, true,  false)?,
        Op::SDiv64 => emit_div(asm, alloc, a, dst_vr, true,  true)?,

        Op::Clz32 => emit_clz(asm, alloc, a, dst_vr, false)?,
        Op::Clz64 => emit_clz(asm, alloc, a, dst_vr, true)?,
        Op::Cls32 => emit_cls(asm, alloc, a, dst_vr, false)?,
        Op::Cls64 => emit_cls(asm, alloc, a, dst_vr, true)?,
        Op::Rbit32 => emit_rbit(asm, alloc, a, dst_vr, false)?,
        Op::Rbit64 => emit_rbit(asm, alloc, a, dst_vr, true)?,
        Op::Rev16  => emit_rev16(asm, alloc, a, dst_vr, a.ty == Ty::U64)?,
        Op::Rev32  => emit_rev32_within64(asm, alloc, a, dst_vr)?,
        Op::Rev64  => emit_bswap(asm, alloc, a, dst_vr, a.ty == Ty::U64)?,

        Op::Lsl32 => emit_shift_32(asm, alloc, a, dst_vr, ShiftKind::Lsl)?,
        Op::Lsl64 => emit_shift_64(asm, alloc, a, dst_vr, ShiftKind::Lsl)?,
        Op::Lsr32 => emit_shift_32(asm, alloc, a, dst_vr, ShiftKind::Lsr)?,
        Op::Lsr64 => emit_shift_64(asm, alloc, a, dst_vr, ShiftKind::Lsr)?,
        Op::Asr32 => emit_shift_32(asm, alloc, a, dst_vr, ShiftKind::Asr)?,
        Op::Asr64 => emit_shift_64(asm, alloc, a, dst_vr, ShiftKind::Asr)?,
        Op::Ror32 => emit_shift_32(asm, alloc, a, dst_vr, ShiftKind::Ror)?,
        Op::Ror64 => emit_shift_64(asm, alloc, a, dst_vr, ShiftKind::Ror)?,

        Op::Not32 => emit_unop_32(asm, alloc, a, dst_vr, UnopKind::Not)?,
        Op::Not64 => emit_unop_64(asm, alloc, a, dst_vr, UnopKind::Not)?,
        Op::Neg32 => emit_unop_32(asm, alloc, a, dst_vr, UnopKind::Neg)?,
        Op::Neg64 => emit_unop_64(asm, alloc, a, dst_vr, UnopKind::Neg)?,

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

        op if op.is_terminator() => {}

        Op::Hint | Op::MemoryBarrier => {}

        Op::Clrex => {
            asm.mov(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), 0i32)?;
        }

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

fn emit_binop_32(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, k: BinKind) -> Result<()> {
    load32(asm, alloc, a.args[0], eax)?;
    load32(asm, alloc, a.args[1], gpr32(scratch1_id()))?;
    apply_bin_32(asm, k, eax, gpr32(scratch1_id()))?;
    if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    Ok(())
}

fn emit_binop_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, k: BinKind) -> Result<()> {
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    load64(asm, alloc, a.args[1], SCRATCH1)?;
    apply_bin_64(asm, k, SCRATCH0, SCRATCH1)?;
    if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
    Ok(())
}

#[derive(Clone, Copy)]
enum UnopKind { Not, Neg }

fn emit_unop_32(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, k: UnopKind) -> Result<()> {
    load32(asm, alloc, a.args[0], eax)?;
    match k {
        UnopKind::Not => asm.not(eax)?,
        UnopKind::Neg => asm.neg(eax)?,
    }
    if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    Ok(())
}

fn emit_unop_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, k: UnopKind) -> Result<()> {
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    match k {
        UnopKind::Not => asm.not(SCRATCH0)?,
        UnopKind::Neg => asm.neg(SCRATCH0)?,
    }
    if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
    Ok(())
}

#[derive(Clone, Copy)]
enum ShiftKind { Lsl, Lsr, Asr, Ror }

fn emit_shift_32(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, kind: ShiftKind) -> Result<()> {
    load32(asm, alloc, a.args[0], eax)?;
    load32(asm, alloc, a.args[1], ecx)?;
    match kind {
        ShiftKind::Lsl => asm.shl(eax, cl)?,
        ShiftKind::Lsr => asm.shr(eax, cl)?,
        ShiftKind::Asr => asm.sar(eax, cl)?,
        ShiftKind::Ror => asm.ror(eax, cl)?,
    }
    if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    Ok(())
}

fn emit_shift_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, kind: ShiftKind) -> Result<()> {
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    load64(asm, alloc, a.args[1], rcx)?;
    match kind {
        ShiftKind::Lsl => asm.shl(SCRATCH0, cl)?,
        ShiftKind::Lsr => asm.shr(SCRATCH0, cl)?,
        ShiftKind::Asr => asm.sar(SCRATCH0, cl)?,
        ShiftKind::Ror => asm.ror(SCRATCH0, cl)?,
    }
    if let Some(d) = dst { store64(asm, alloc, d, SCRATCH0)?; }
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
    is_64: bool,
) -> Result<()> {
    asm.bt(dword_ptr(CTX_REG + cpu_offsets::nzcv() as i32), 1i32)?;
    if is_sub { asm.cmc()?; }
    if is_64 {
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
    is_64: bool,
) -> Result<()> {
    let mut lbl_zero = asm.create_label();
    let mut lbl_done = asm.create_label();

    if is_64 {
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

fn emit_clz(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, is_64: bool) -> Result<()> {
    let mut lbl_zero = asm.create_label();
    let mut lbl_done = asm.create_label();
    if is_64 {
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

fn emit_cls(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, is_64: bool) -> Result<()> {
    let mut lbl_all_same = asm.create_label();
    let mut lbl_done = asm.create_label();
    if is_64 {
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

fn emit_rbit(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, is_64: bool) -> Result<()> {
    if is_64 {
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

fn emit_rev16(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, is_64: bool) -> Result<()> {
    if is_64 {
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

fn emit_bswap(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueRef>, is_64: bool) -> Result<()> {
    if is_64 {
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
