use iced_x86::code_asm::*;

use crate::arch::{Cond, NUM_GPRS, ZR_ENCODING};
use crate::backend::abi::{
    ARG3_REG, CALL_PRECALL_SUB, CTX_REG, SCRATCH0, SCRATCH1, SCRATCH2, SCRATCH3,
};
use crate::jit::memory::{
    addr_mem_read8, addr_mem_read16, addr_mem_read32, addr_mem_read64,
    addr_mem_write8, addr_mem_write16, addr_mem_write32, addr_mem_write64,
};
use crate::backend::reg_alloc::{Allocation, ValueLoc};
use crate::error::{Error, Result};
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};
use crate::jit::context::cpu_offsets;

pub fn emit_armlet(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    if a.is_eliminated() { return Ok(()); }

    let dst = if a.ty != Ty::Void {
        Some(alloc.loc(ValueRef::new(idx as u32)))
    } else { None };

    match a.op {
        Op::Void => {}
        Op::Identity => {
            if let Some(d) = dst {
                let src_loc = alloc.loc(a.args[0]);
                load_int(asm, SCRATCH0, src_loc)?;
                store_int(asm, d, SCRATCH0)?;
            }
        }

        Op::ConstU32 => {
            let d = dst.unwrap();
            asm.mov(eax, (a.imm as u32) as i32)?;
            asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
        }
        Op::ConstU64 => {
            let d = dst.unwrap();
            asm.mov(SCRATCH0, a.imm as i64)?;
            asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH0)?;
        }

        Op::GetX => {
            let d = dst.unwrap();
            let reg = a.imm as usize;
            load_guest_x(asm, SCRATCH0, reg)?;
            store_int(asm, d, SCRATCH0)?;
        }
        Op::GetW => {
            let d = dst.unwrap();
            let reg = a.imm as usize;
            load_guest_x(asm, SCRATCH0, reg)?;
            asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
        }
        Op::SetX => {
            let reg = a.imm as usize;
            let src_loc = alloc.loc(a.args[0]);
            load_int(asm, SCRATCH0, src_loc)?;
            store_guest_x(asm, reg, SCRATCH0)?;
        }
        Op::SetW => {
            let reg = a.imm as usize;
            let src_loc = alloc.loc(a.args[0]);
            asm.mov(eax, dword_ptr(rbp - src_loc.stack_offset))?;
            store_guest_x(asm, reg, SCRATCH0)?;
        }
        Op::GetSp => {
            let d = dst.unwrap();
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::sp() as i32))?;
            store_int(asm, d, SCRATCH0)?;
        }
        Op::SetSp => {
            let src_loc = alloc.loc(a.args[0]);
            load_int(asm, SCRATCH0, src_loc)?;
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::sp() as i32), SCRATCH0)?;
        }
        Op::GetNzcv => {
            let d = dst.unwrap();
            asm.movzx(eax, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
            asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
        }
        Op::SetNzcv => {
            let src_loc = alloc.loc(a.args[0]);
            asm.mov(eax, dword_ptr(rbp - src_loc.stack_offset))?;
            asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
        }
        Op::GetPc => {
            let d = dst.unwrap();
            asm.mov(SCRATCH0, a.imm as i64)?;
            store_int(asm, d, SCRATCH0)?;
        }

        Op::Add32 => emit_binop_32(asm, alloc, a, dst, BinKind::Add)?,
        Op::Add64 => emit_binop_64(asm, alloc, a, dst, BinKind::Add)?,
        Op::Sub32 => emit_binop_32(asm, alloc, a, dst, BinKind::Sub)?,
        Op::Sub64 => emit_binop_64(asm, alloc, a, dst, BinKind::Sub)?,
        Op::And32 => emit_binop_32(asm, alloc, a, dst, BinKind::And)?,
        Op::And64 => emit_binop_64(asm, alloc, a, dst, BinKind::And)?,
        Op::Or32  => emit_binop_32(asm, alloc, a, dst, BinKind::Or)?,
        Op::Or64  => emit_binop_64(asm, alloc, a, dst, BinKind::Or)?,
        Op::Eor32 => emit_binop_32(asm, alloc, a, dst, BinKind::Xor)?,
        Op::Eor64 => emit_binop_64(asm, alloc, a, dst, BinKind::Xor)?,
        Op::Mul32 => emit_binop_32(asm, alloc, a, dst, BinKind::Imul)?,
        Op::Mul64 => emit_binop_64(asm, alloc, a, dst, BinKind::Imul)?,

        Op::Lsl32 => emit_shift_32(asm, alloc, a, dst, ShiftKind::Lsl)?,
        Op::Lsl64 => emit_shift_64(asm, alloc, a, dst, ShiftKind::Lsl)?,
        Op::Lsr32 => emit_shift_32(asm, alloc, a, dst, ShiftKind::Lsr)?,
        Op::Lsr64 => emit_shift_64(asm, alloc, a, dst, ShiftKind::Lsr)?,
        Op::Asr32 => emit_shift_32(asm, alloc, a, dst, ShiftKind::Asr)?,
        Op::Asr64 => emit_shift_64(asm, alloc, a, dst, ShiftKind::Asr)?,
        Op::Ror32 => emit_shift_32(asm, alloc, a, dst, ShiftKind::Ror)?,
        Op::Ror64 => emit_shift_64(asm, alloc, a, dst, ShiftKind::Ror)?,

        Op::Not32 => emit_unop_32(asm, alloc, a, dst, UnopKind::Not)?,
        Op::Not64 => emit_unop_64(asm, alloc, a, dst, UnopKind::Not)?,
        Op::Neg32 => emit_unop_32(asm, alloc, a, dst, UnopKind::Neg)?,
        Op::Neg64 => emit_unop_64(asm, alloc, a, dst, UnopKind::Neg)?,

        Op::AddsFlags32 | Op::AddsFlags64 | Op::SubsFlags32 | Op::SubsFlags64 => {
            emit_flagged_addsub(asm, alloc, a, dst)?;
        }

        Op::Load8  => emit_load(asm, alloc, a, dst, 1)?,
        Op::Load16 => emit_load(asm, alloc, a, dst, 2)?,
        Op::Load32 => emit_load(asm, alloc, a, dst, 4)?,
        Op::Load64 => emit_load(asm, alloc, a, dst, 8)?,
        Op::Store8  => emit_store(asm, alloc, a, 1)?,
        Op::Store16 => emit_store(asm, alloc, a, 2)?,
        Op::Store32 => emit_store(asm, alloc, a, 4)?,
        Op::Store64 => emit_store(asm, alloc, a, 8)?,

        Op::Csel32 | Op::Csel64 => emit_csel(asm, alloc, a, dst)?,

        op if op.is_terminator() => {}

        Op::Hint | Op::MemoryBarrier => {}

        other => return Err(Error::Unsupported {
            pc: block.start_pc,
            opcode: other as u32,
        }),
    }

    Ok(())
}

fn load_int(asm: &mut CodeAssembler, reg: AsmRegister64, loc: ValueLoc) -> Result<()> {
    match loc.width {
        4 => { asm.mov(eax_from(reg), dword_ptr(rbp - loc.stack_offset))?; }
        _ => { asm.mov(reg, qword_ptr(rbp - loc.stack_offset))?; }
    }
    Ok(())
}

fn store_int(asm: &mut CodeAssembler, loc: ValueLoc, reg: AsmRegister64) -> Result<()> {
    match loc.width {
        4 => { asm.mov(dword_ptr(rbp - loc.stack_offset), eax_from(reg))?; }
        _ => { asm.mov(qword_ptr(rbp - loc.stack_offset), reg)?; }
    }
    Ok(())
}

#[inline]
fn eax_from(r: AsmRegister64) -> AsmRegister32 {
    if r == rax { eax }
    else if r == rcx { ecx }
    else if r == rdx { edx }
    else if r == rsi { esi }
    else if r == rdi { edi }
    else if r == r8  { r8d }
    else if r == r9  { r9d }
    else { panic!("no 32-bit alias mapped for register"); }
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

fn emit_binop_32(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, k: BinKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();
    asm.mov(eax,                 dword_ptr(rbp - l.stack_offset))?;
    asm.mov(eax_from(SCRATCH1),  dword_ptr(rbp - r.stack_offset))?;
    apply_bin_32(asm, k, eax, eax_from(SCRATCH1))?;
    asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    Ok(())
}

fn emit_binop_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, k: BinKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();
    asm.mov(SCRATCH0, qword_ptr(rbp - l.stack_offset))?;
    asm.mov(SCRATCH1, qword_ptr(rbp - r.stack_offset))?;
    apply_bin_64(asm, k, SCRATCH0, SCRATCH1)?;
    asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH0)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum UnopKind { Not, Neg }

fn emit_unop_32(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, k: UnopKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let d = dst.unwrap();
    asm.mov(eax, dword_ptr(rbp - l.stack_offset))?;
    match k {
        UnopKind::Not => asm.not(eax)?,
        UnopKind::Neg => asm.neg(eax)?,
    }
    asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    Ok(())
}

fn emit_unop_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, k: UnopKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let d = dst.unwrap();
    asm.mov(SCRATCH0, qword_ptr(rbp - l.stack_offset))?;
    match k {
        UnopKind::Not => asm.not(SCRATCH0)?,
        UnopKind::Neg => asm.neg(SCRATCH0)?,
    }
    asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH0)?;
    Ok(())
}

#[derive(Clone, Copy)]
enum ShiftKind { Lsl, Lsr, Asr, Ror }

fn emit_shift_32(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, kind: ShiftKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();
    asm.mov(eax, dword_ptr(rbp - l.stack_offset))?;
    asm.mov(ecx, dword_ptr(rbp - r.stack_offset))?;
    match kind {
        ShiftKind::Lsl => asm.shl(eax, cl)?,
        ShiftKind::Lsr => asm.shr(eax, cl)?,
        ShiftKind::Asr => asm.sar(eax, cl)?,
        ShiftKind::Ror => asm.ror(eax, cl)?,
    }
    asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    Ok(())
}

fn emit_shift_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, kind: ShiftKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();
    asm.mov(SCRATCH0, qword_ptr(rbp - l.stack_offset))?;
    asm.mov(rcx, qword_ptr(rbp - r.stack_offset))?;
    match kind {
        ShiftKind::Lsl => asm.shl(SCRATCH0, cl)?,
        ShiftKind::Lsr => asm.shr(SCRATCH0, cl)?,
        ShiftKind::Asr => asm.sar(SCRATCH0, cl)?,
        ShiftKind::Ror => asm.ror(SCRATCH0, cl)?,
    }
    asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH0)?;
    Ok(())
}

fn emit_flagged_addsub(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();

    let is_64 = matches!(a.op, Op::AddsFlags64 | Op::SubsFlags64);
    let is_sub = matches!(a.op, Op::SubsFlags32 | Op::SubsFlags64);

    if is_64 {
        asm.mov(SCRATCH0, qword_ptr(rbp - l.stack_offset))?;
        asm.mov(SCRATCH1, qword_ptr(rbp - r.stack_offset))?;
        if is_sub { asm.sub(SCRATCH0, SCRATCH1)?; }
        else      { asm.add(SCRATCH0, SCRATCH1)?; }
        asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH0)?;
    } else {
        asm.mov(eax, dword_ptr(rbp - l.stack_offset))?;
        asm.mov(eax_from(SCRATCH1), dword_ptr(rbp - r.stack_offset))?;
        if is_sub { asm.sub(eax, eax_from(SCRATCH1))?; }
        else      { asm.add(eax, eax_from(SCRATCH1))?; }
        asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    }

    asm.sets(r8b)?;
    asm.sete(r9b)?;
    asm.setc(r10b)?;
    asm.seto(r11b)?;

    if is_sub {
        asm.xor(r10b, 1i32)?;
    }

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

fn emit_load(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, bytes: u32) -> Result<()> {
    let addr_loc = alloc.loc(a.args[0]);
    let d = dst.unwrap();
    asm.mov(SCRATCH1, qword_ptr(rbp - addr_loc.stack_offset))?;
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
    match bytes {
        1 | 2 | 4 => { asm.mov(dword_ptr(rbp - d.stack_offset), eax)?; }
        8         => { asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH0)?; }
        _ => unreachable!(),
    }
    Ok(())
}

fn emit_store(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, bytes: u32) -> Result<()> {
    let addr_loc = alloc.loc(a.args[0]);
    let val_loc  = alloc.loc(a.args[1]);
    if bytes == 8 {
        asm.mov(SCRATCH3, qword_ptr(rbp - val_loc.stack_offset))?;
    } else {
        asm.mov(eax_from(SCRATCH3), dword_ptr(rbp - val_loc.stack_offset))?;
    }
    asm.mov(SCRATCH1, qword_ptr(rbp - addr_loc.stack_offset))?;
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

fn emit_csel(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let nz = alloc.loc(a.args[2]);
    let d  = dst.unwrap();
    let cond = Cond::from_bits(a.imm as u8);

    asm.mov(edx, dword_ptr(rbp - nz.stack_offset))?;
    let is_64 = matches!(a.op, Op::Csel64);

    emit_cond_check_byte(asm, cond)?;
    asm.test(al, al)?;
    if is_64 {
        asm.mov(SCRATCH1, qword_ptr(rbp - r.stack_offset))?;
        asm.mov(SCRATCH2, qword_ptr(rbp - l.stack_offset))?;
        asm.cmovne(SCRATCH1, SCRATCH2)?;
        asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH1)?;
    } else {
        asm.mov(eax, dword_ptr(rbp - r.stack_offset))?;
        asm.mov(eax_from(SCRATCH1), dword_ptr(rbp - l.stack_offset))?;
        asm.cmovne(eax, eax_from(SCRATCH1))?;
        asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    }
    Ok(())
}

pub fn emit_cond_check_byte(asm: &mut CodeAssembler, cond: Cond) -> Result<()> {
    asm.mov(eax, edx)?;
    asm.shr(eax, 3i32)?;
    asm.and(eax, 1i32)?;

    asm.mov(ecx, edx)?;
    asm.shr(ecx, 2i32)?;
    asm.and(ecx, 1i32)?;

    asm.mov(r8d, edx)?;
    asm.and(r8d, 1i32)?;

    asm.shr(edx, 1i32)?;
    asm.and(edx, 1i32)?;

    match cond {
        Cond::EQ => { asm.mov(eax, ecx)?; }
        Cond::NE => { asm.mov(eax, ecx)?; asm.xor(eax, 1i32)?; }
        Cond::CS => { asm.mov(eax, edx)?; }
        Cond::CC => { asm.mov(eax, edx)?; asm.xor(eax, 1i32)?; }
        Cond::MI => {}
        Cond::PL => { asm.xor(eax, 1i32)?; }
        Cond::VS => { asm.mov(eax, r8d)?; }
        Cond::VC => { asm.mov(eax, r8d)?; asm.xor(eax, 1i32)?; }
        Cond::HI => {
            asm.mov(eax, ecx)?;
            asm.xor(eax, 1i32)?;
            asm.and(eax, edx)?;
        }
        Cond::LS => {
            asm.mov(eax, ecx)?;
            asm.xor(eax, 1i32)?;
            asm.and(eax, edx)?;
            asm.xor(eax, 1i32)?;
        }
        Cond::GE => {
            asm.xor(eax, r8d)?;
            asm.xor(eax, 1i32)?;
        }
        Cond::LT => {
            asm.xor(eax, r8d)?;
        }
        Cond::GT => {
            asm.xor(eax, r8d)?;
            asm.xor(eax, 1i32)?;
            asm.mov(esi, ecx)?;
            asm.xor(esi, 1i32)?;
            asm.and(eax, esi)?;
        }
        Cond::LE => {
            asm.xor(eax, r8d)?;
            asm.xor(eax, 1i32)?;
            asm.mov(esi, ecx)?;
            asm.xor(esi, 1i32)?;
            asm.and(eax, esi)?;
            asm.xor(eax, 1i32)?;
        }
        Cond::AL | Cond::NV => { asm.mov(eax, 1i32)?; }
    }
    Ok(())
}
