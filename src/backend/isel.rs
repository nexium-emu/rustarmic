//! Per-opcode instruction selection. Each opcode translates to a tiny x86
//! sequence that:
//!
//! 1. Loads operands from their stack slots into scratch registers.
//! 2. Performs the work.
//! 3. Stores the result back to the destination slot.
//!
//! Stack offsets are negative from `rbp`; the helper `slot(off)` builds the
//! correctly-sized memory operand based on the slot width.

use iced_x86::code_asm::*;

use crate::arch::{Cond, NUM_GPRS, ZR_ENCODING};
use crate::backend::abi::{CTX_REG, SCRATCH1, SCRATCH2, SCRATCH3};
use crate::backend::reg_alloc::{Allocation, ValueLoc};
use crate::error::{Error, Result};
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};
use crate::jit::context::cpu_offsets;

/// Emit the body of a single armlet.
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
                load_int(asm, SCRATCH1, src_loc)?;
                store_int(asm, d, SCRATCH1)?;
            }
        }

        Op::ConstU32 => {
            let d = dst.unwrap();
            asm.mov(eax, (a.imm as u32) as i32)?;
            asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
        }
        Op::ConstU64 => {
            let d = dst.unwrap();
            asm.mov(SCRATCH1, a.imm as i64)?;
            asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH1)?;
        }

        // ── GPR I/O ─────────────────────────────────────────────────────────
        Op::GetX => {
            let d = dst.unwrap();
            let reg = a.imm as usize;
            load_guest_x(asm, SCRATCH1, reg)?;
            store_int(asm, d, SCRATCH1)?;
        }
        Op::GetW => {
            let d = dst.unwrap();
            let reg = a.imm as usize;
            load_guest_x(asm, SCRATCH1, reg)?;
            asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
        }
        Op::SetX => {
            let reg = a.imm as usize;
            let src_loc = alloc.loc(a.args[0]);
            load_int(asm, SCRATCH1, src_loc)?;
            store_guest_x(asm, reg, SCRATCH1)?;
        }
        Op::SetW => {
            let reg = a.imm as usize;
            let src_loc = alloc.loc(a.args[0]);
            // 32-bit load into EAX zero-extends into RAX; store as 64-bit.
            asm.mov(eax, dword_ptr(rbp - src_loc.stack_offset))?;
            store_guest_x(asm, reg, SCRATCH1)?;
        }
        Op::GetSp => {
            let d = dst.unwrap();
            asm.mov(SCRATCH1, qword_ptr(CTX_REG + cpu_offsets::sp() as i32))?;
            store_int(asm, d, SCRATCH1)?;
        }
        Op::SetSp => {
            let src_loc = alloc.loc(a.args[0]);
            load_int(asm, SCRATCH1, src_loc)?;
            asm.mov(qword_ptr(CTX_REG + cpu_offsets::sp() as i32), SCRATCH1)?;
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
            asm.mov(SCRATCH1, a.imm as i64)?;
            store_int(asm, d, SCRATCH1)?;
        }

        // ── Integer ALU ────────────────────────────────────────────────────
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

        // ── ADDS/SUBS — compute result and store NZCV ───────────────────────
        Op::AddsFlags32 | Op::AddsFlags64 | Op::SubsFlags32 | Op::SubsFlags64 => {
            emit_flagged_addsub(asm, alloc, a, dst)?;
        }

        // ── Memory ─────────────────────────────────────────────────────────
        Op::Load8  => emit_load(asm, alloc, a, dst, 1)?,
        Op::Load16 => emit_load(asm, alloc, a, dst, 2)?,
        Op::Load32 => emit_load(asm, alloc, a, dst, 4)?,
        Op::Load64 => emit_load(asm, alloc, a, dst, 8)?,
        Op::Store8  => emit_store(asm, alloc, a, 1)?,
        Op::Store16 => emit_store(asm, alloc, a, 2)?,
        Op::Store32 => emit_store(asm, alloc, a, 4)?,
        Op::Store64 => emit_store(asm, alloc, a, 8)?,

        // ── Csel ────────────────────────────────────────────────────────────
        Op::Csel32 | Op::Csel64 => emit_csel(asm, alloc, a, dst)?,

        // ── Terminators are handled separately by emit_block ────────────────
        op if op.is_terminator() => {}

        // ── Hints / barriers: no-op ─────────────────────────────────────────
        Op::Hint | Op::MemoryBarrier => {}

        other => return Err(Error::Unsupported {
            pc: block.start_pc,
            opcode: other as u32,
        }),
    }

    Ok(())
}

// ─── Helpers ────────────────────────────────────────────────────────────────

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
    if r == rax { eax } else if r == r10 { r10d } else if r == r11 { r11d }
    else if r == rcx { ecx } else if r == rdx { edx } else if r == r8 { r8d } else if r == r9 { r9d }
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
    asm.mov(eax, dword_ptr(rbp - l.stack_offset))?;
    asm.mov(r10d, dword_ptr(rbp - r.stack_offset))?;
    apply_bin_32(asm, k, eax, r10d)?;
    asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    Ok(())
}

fn emit_binop_64(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, k: BinKind) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();
    asm.mov(SCRATCH1, qword_ptr(rbp - l.stack_offset))?;
    asm.mov(SCRATCH2, qword_ptr(rbp - r.stack_offset))?;
    apply_bin_64(asm, k, SCRATCH1, SCRATCH2)?;
    asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH1)?;
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
    asm.mov(SCRATCH1, qword_ptr(rbp - l.stack_offset))?;
    match k {
        UnopKind::Not => asm.not(SCRATCH1)?,
        UnopKind::Neg => asm.neg(SCRATCH1)?,
    }
    asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH1)?;
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
    asm.mov(SCRATCH1, qword_ptr(rbp - l.stack_offset))?;
    asm.mov(rcx, qword_ptr(rbp - r.stack_offset))?;
    match kind {
        ShiftKind::Lsl => asm.shl(SCRATCH1, cl)?,
        ShiftKind::Lsr => asm.shr(SCRATCH1, cl)?,
        ShiftKind::Asr => asm.sar(SCRATCH1, cl)?,
        ShiftKind::Ror => asm.ror(SCRATCH1, cl)?,
    }
    asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH1)?;
    Ok(())
}

fn emit_flagged_addsub(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let d = dst.unwrap();

    let is_64 = matches!(a.op, Op::AddsFlags64 | Op::SubsFlags64);
    let is_sub = matches!(a.op, Op::SubsFlags32 | Op::SubsFlags64);

    if is_64 {
        asm.mov(SCRATCH1, qword_ptr(rbp - l.stack_offset))?;
        asm.mov(SCRATCH2, qword_ptr(rbp - r.stack_offset))?;
        if is_sub { asm.sub(SCRATCH1, SCRATCH2)?; }
        else      { asm.add(SCRATCH1, SCRATCH2)?; }
        asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH1)?;
    } else {
        asm.mov(eax, dword_ptr(rbp - l.stack_offset))?;
        asm.mov(r10d, dword_ptr(rbp - r.stack_offset))?;
        if is_sub { asm.sub(eax, r10d)?; }
        else      { asm.add(eax, r10d)?; }
        asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    }

    // AArch64 NZCV from EFLAGS:
    //   N = SF, Z = ZF, V = OF
    //   C: ADD → CF; SUB → !CF.
    asm.sets(dl)?;
    asm.sete(dh)?;
    asm.setc(cl)?;
    asm.seto(ch)?;

    if is_sub {
        asm.xor(cl, 1i32)?;
    }

    asm.movzx(eax, dl)?;
    asm.shl(eax, 3i32)?;
    asm.movzx(r10d, dh)?;
    asm.shl(r10d, 2i32)?;
    asm.or(eax, r10d)?;
    asm.movzx(r10d, cl)?;
    asm.shl(r10d, 1i32)?;
    asm.or(eax, r10d)?;
    asm.movzx(r10d, ch)?;
    asm.or(eax, r10d)?;

    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
    Ok(())
}

fn emit_load(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>, bytes: u32) -> Result<()> {
    let addr_loc = alloc.loc(a.args[0]);
    let d = dst.unwrap();
    asm.mov(SCRATCH3, qword_ptr(rbp - addr_loc.stack_offset))?;
    asm.mov(SCRATCH1, qword_ptr(CTX_REG + cpu_offsets::mem_base() as i32))?;
    asm.add(SCRATCH1, SCRATCH3)?;
    match bytes {
        1 => { asm.movzx(eax, byte_ptr(SCRATCH1))?;
               asm.mov(dword_ptr(rbp - d.stack_offset), eax)?; }
        2 => { asm.movzx(eax, word_ptr(SCRATCH1))?;
               asm.mov(dword_ptr(rbp - d.stack_offset), eax)?; }
        4 => { asm.mov(eax, dword_ptr(SCRATCH1))?;
               asm.mov(dword_ptr(rbp - d.stack_offset), eax)?; }
        8 => { asm.mov(SCRATCH2, qword_ptr(SCRATCH1))?;
               asm.mov(qword_ptr(rbp - d.stack_offset), SCRATCH2)?; }
        _ => return Err(Error::Backend("unsupported load width".into())),
    }
    Ok(())
}

fn emit_store(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, bytes: u32) -> Result<()> {
    let addr_loc = alloc.loc(a.args[0]);
    let val_loc  = alloc.loc(a.args[1]);
    asm.mov(SCRATCH3, qword_ptr(rbp - addr_loc.stack_offset))?;
    asm.mov(SCRATCH1, qword_ptr(CTX_REG + cpu_offsets::mem_base() as i32))?;
    asm.add(SCRATCH1, SCRATCH3)?;
    match bytes {
        1 => { asm.mov(eax, dword_ptr(rbp - val_loc.stack_offset))?;
               asm.mov(byte_ptr(SCRATCH1), al)?; }
        2 => { asm.mov(eax, dword_ptr(rbp - val_loc.stack_offset))?;
               asm.mov(word_ptr(SCRATCH1), ax)?; }
        4 => { asm.mov(eax, dword_ptr(rbp - val_loc.stack_offset))?;
               asm.mov(dword_ptr(SCRATCH1), eax)?; }
        8 => { asm.mov(SCRATCH2, qword_ptr(rbp - val_loc.stack_offset))?;
               asm.mov(qword_ptr(SCRATCH1), SCRATCH2)?; }
        _ => return Err(Error::Backend("unsupported store width".into())),
    }
    Ok(())
}

fn emit_csel(asm: &mut CodeAssembler, alloc: &Allocation, a: Armlet, dst: Option<ValueLoc>) -> Result<()> {
    let l = alloc.loc(a.args[0]);
    let r = alloc.loc(a.args[1]);
    let nz = alloc.loc(a.args[2]);
    let d  = dst.unwrap();
    let cond = Cond::from_bits(a.imm as u8);

    asm.mov(r10d, dword_ptr(rbp - nz.stack_offset))?;
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
        asm.mov(r10d, dword_ptr(rbp - l.stack_offset))?;
        asm.cmovne(eax, r10d)?;
        asm.mov(dword_ptr(rbp - d.stack_offset), eax)?;
    }
    Ok(())
}

/// Materialize the boolean result of an AArch64 condition into AL.
/// Caller must have loaded the NZCV nibble into the low 4 bits of r10d.
pub fn emit_cond_check_byte(asm: &mut CodeAssembler, cond: Cond) -> Result<()> {
    asm.mov(eax, r10d)?;
    asm.shr(eax, 3i32)?;
    asm.and(eax, 1i32)?;     // eax = N

    asm.mov(ecx, r10d)?;
    asm.shr(ecx, 2i32)?;
    asm.and(ecx, 1i32)?;     // ecx = Z

    asm.mov(edx, r10d)?;
    asm.shr(edx, 1i32)?;
    asm.and(edx, 1i32)?;     // edx = C

    asm.mov(r8d, r10d)?;
    asm.and(r8d, 1i32)?;     // r8d = V

    match cond {
        Cond::EQ => { asm.mov(eax, ecx)?; }
        Cond::NE => { asm.mov(eax, ecx)?; asm.xor(eax, 1i32)?; }
        Cond::CS => { asm.mov(eax, edx)?; }
        Cond::CC => { asm.mov(eax, edx)?; asm.xor(eax, 1i32)?; }
        Cond::MI => { /* eax already = N */ }
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
            asm.mov(r10d, ecx)?;
            asm.xor(r10d, 1i32)?;
            asm.and(eax, r10d)?;
        }
        Cond::LE => {
            asm.xor(eax, r8d)?;
            asm.xor(eax, 1i32)?;
            asm.mov(r10d, ecx)?;
            asm.xor(r10d, 1i32)?;
            asm.and(eax, r10d)?;
            asm.xor(eax, 1i32)?;
        }
        Cond::AL | Cond::NV => { asm.mov(eax, 1i32)?; }
    }
    Ok(())
}
