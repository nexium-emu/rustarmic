use iced_x86::code_asm::*;

use crate::arch::{Cond, NUM_GPRS, ZR_ENCODING};
use crate::backend::abi::{
    ARG3_REG, CALL_PRECALL_SUB, CTX_REG, SCRATCH0, SCRATCH1, SCRATCH2, SCRATCH3,
};
use crate::backend::operand::{
    get_xmm_q, gpr32, gpr64, into_xmm_q, load32, load64, load_xmm_d, load_xmm_s, store32,
    store64, store_xmm_d, store_xmm_q, store_xmm_s, working_xmm_for,
};
use crate::backend::regalloc::{Allocation, Loc};
use crate::error::{Error, Result};
use crate::ir::{Armlet, Block, Op, Ty, ValueRef};
use crate::jit::context::cpu_offsets;
use crate::jit::memory::{
    addr_mem_read8, addr_mem_read16, addr_mem_read32, addr_mem_read64,
    addr_mem_write8, addr_mem_write16, addr_mem_write32, addr_mem_write64,
};

/// Function-pointer shape every per-op emitter conforms to. The block + idx
/// give the emitter the armlet, its destination value (derived from idx +
/// `ty != Void`), and full SSA context if needed.
pub type EmitFn = fn(&mut CodeAssembler, &Block, &Allocation, usize) -> Result<()>;

#[inline]
fn dst_of(a: &Armlet, idx: usize) -> Option<ValueRef> {
    if a.ty != Ty::Void { Some(ValueRef::new(idx as u32)) } else { None }
}

pub fn emit_armlet(
    asm: &mut CodeAssembler,
    block: &Block,
    alloc: &Allocation,
    idx: usize,
) -> Result<()> {
    let a = block.code[idx];
    if a.is_eliminated() { return Ok(()); }
    if a.op.is_terminator() { return Ok(()); }

    let f = dispatch_op(a.op).ok_or(Error::Unsupported {
        pc: block.start_pc,
        opcode: a.op as u32,
    })?;
    f(asm, block, alloc, idx)
}

/// Map each opcode to its emit function. LLVM compiles this match into a
/// jump table indexed by the discriminant — exactly the "dispatch table"
/// shape we want, just expressed as a `match` so each entry is type-checked
/// at compile time rather than maintained as a parallel const array.
fn dispatch_op(op: Op) -> Option<EmitFn> {
    use Op::*;
    Some(match op {
        Void => emit_nop,
        Identity => emit_op_identity,

        ConstU32 => emit_op_const_u32,
        ConstU64 => emit_op_const_u64,

        GetX => emit_op_get_x,
        GetW => emit_op_get_w,
        SetX => emit_op_set_x,
        SetW => emit_op_set_w,
        GetSp => emit_op_get_sp,
        SetSp => emit_op_set_sp,
        GetNzcv => emit_op_get_nzcv,
        SetNzcv => emit_op_set_nzcv,
        GetPc => emit_op_get_pc,
        GetV => emit_op_get_v,
        SetV => emit_op_set_v,

        Add32 | Add64 => emit_op_add,
        Sub32 | Sub64 => emit_op_sub,
        And32 | And64 => emit_op_and,
        Or32  | Or64  => emit_op_or,
        Eor32 | Eor64 => emit_op_xor,
        Mul32 | Mul64 => emit_op_mul,

        Adc32 | Adc64 => emit_op_adc,
        Sbc32 | Sbc64 => emit_op_sbc,

        UDiv32 | UDiv64 => emit_op_udiv,
        SDiv32 | SDiv64 => emit_op_sdiv,

        Clz32  | Clz64  => emit_op_clz,
        Cls32  | Cls64  => emit_op_cls,
        Rbit32 | Rbit64 => emit_op_rbit,
        Rev16  => emit_op_rev16,
        Rev32  => emit_op_rev32,
        Rev64  => emit_op_rev64,

        Lsl32 | Lsl64 => emit_op_lsl,
        Lsr32 | Lsr64 => emit_op_lsr,
        Asr32 | Asr64 => emit_op_asr,
        Ror32 | Ror64 => emit_op_ror,

        Not32 | Not64 => emit_op_not,
        Neg32 | Neg64 => emit_op_neg,

        AddsFlags32 | AddsFlags64 | SubsFlags32 | SubsFlags64 => emit_op_flagged_addsub,

        Load8 | Load16 | Load32 | Load64 => emit_op_load,
        Store8 | Store16 | Store32 | Store64 => emit_op_store,

        LoadEx8 | LoadEx16 | LoadEx32 | LoadEx64 => emit_op_load_ex,
        StoreEx8 | StoreEx16 | StoreEx32 | StoreEx64 => emit_op_store_ex,

        Csel32 | Csel64 => emit_op_csel,

        Fadd32 | Fadd64 => emit_op_fadd,
        Fsub32 | Fsub64 => emit_op_fsub,
        Fmul32 | Fmul64 => emit_op_fmul,
        Fdiv32 | Fdiv64 => emit_op_fdiv,
        Fmax32 | Fmax64 => emit_op_fmax,
        Fmin32 | Fmin64 => emit_op_fmin,
        Fcmp32 | Fcmp64 => emit_op_fcmp,
        Fsqrt32 | Fsqrt64 => emit_op_fsqrt_,
        Fneg32 | Fneg64 => emit_op_fneg_,
        Fabs32 | Fabs64 => emit_op_fabs_,

        FcvtZsSW => emit_op_fcvt_zs_sw,
        FcvtZsSX => emit_op_fcvt_zs_sx,
        FcvtZsDW => emit_op_fcvt_zs_dw,
        FcvtZsDX => emit_op_fcvt_zs_dx,
        ScvtfWS  => emit_op_scvtf_ws,
        ScvtfXS  => emit_op_scvtf_xs,
        ScvtfWD  => emit_op_scvtf_wd,
        ScvtfXD  => emit_op_scvtf_xd,
        FcvtSD   => emit_op_fcvt_sd,
        FcvtDS   => emit_op_fcvt_ds,

        VecBuildQ      => emit_op_vec_build_q,
        VecExtractLo64 => emit_op_vec_extract_lo64,
        VecExtractHi64 => emit_op_vec_extract_hi64,
        VecExtract8    => emit_op_vec_extract8,
        VecExtract16   => emit_op_vec_extract16,
        VecExtract32   => emit_op_vec_extract32,

        VecAdd8 | VecAdd16 | VecAdd32 | VecAdd64 => emit_op_vec_add,
        VecSub8 | VecSub16 | VecSub32 | VecSub64 => emit_op_vec_sub,
        VecAnd => emit_op_vec_and,
        VecOrr => emit_op_vec_orr,
        VecEor => emit_op_vec_eor,
        VecBic => emit_op_vec_bic,
        VecOrn => emit_op_vec_orn,

        VecNeg8 | VecNeg16 | VecNeg32 | VecNeg64 => emit_op_vec_neg,
        VecAbs8 | VecAbs16 | VecAbs32           => emit_op_vec_abs,
        VecNot => emit_op_vec_not,

        VecMul16 | VecMul32 => emit_op_vec_mul,

        VecShlImm16 | VecShlImm32 | VecShlImm64   => emit_op_vec_shl_imm,
        VecUshrImm16 | VecUshrImm32 | VecUshrImm64 => emit_op_vec_ushr_imm,
        VecSshrImm16 | VecSshrImm32                => emit_op_vec_sshr_imm,

        VecCmEq8 | VecCmEq16 | VecCmEq32 | VecCmEq64 => emit_op_vec_cmeq,
        VecCmGt8 | VecCmGt16 | VecCmGt32 | VecCmGt64 => emit_op_vec_cmgt,
        VecCmGe8 | VecCmGe16 | VecCmGe32 | VecCmGe64 => emit_op_vec_cmge,
        VecCmHi16 | VecCmHi32 | VecCmHi64            => emit_op_vec_cmhi,
        VecCmHs16 | VecCmHs32 | VecCmHs64            => emit_op_vec_cmhs,

        VecBit => emit_op_vec_bit,
        VecBif => emit_op_vec_bif,
        VecBsl => emit_op_vec_bsl,

        VecDupGpr8 | VecDupGpr16 | VecDupGpr32 | VecDupGpr64 => emit_op_vec_dup_gpr,
        VecInsGpr8 | VecInsGpr16 | VecInsGpr32 | VecInsGpr64 => emit_op_vec_ins_gpr,

        VecExt => emit_op_vec_ext,
        VecZip1_8 | VecZip1_16 | VecZip1_32 | VecZip1_64 => emit_op_vec_zip1,
        VecZip2_8 | VecZip2_16 | VecZip2_32 | VecZip2_64 => emit_op_vec_zip2,

        VecSmin8 | VecSmin16 | VecSmin32 => emit_op_vec_smin,
        VecSmax8 | VecSmax16 | VecSmax32 => emit_op_vec_smax,
        VecUmin8 | VecUmin16 | VecUmin32 => emit_op_vec_umin,
        VecUmax8 | VecUmax16 | VecUmax32 => emit_op_vec_umax,

        VecAddv32 => emit_op_vec_addv32,

        VecFAdd_S  | VecFAdd_D  => emit_op_vec_fadd,
        VecFSub_S  | VecFSub_D  => emit_op_vec_fsub,
        VecFMul_S  | VecFMul_D  => emit_op_vec_fmul,
        VecFDiv_S  | VecFDiv_D  => emit_op_vec_fdiv,
        VecFMax_S  | VecFMax_D  => emit_op_vec_fmax,
        VecFMin_S  | VecFMin_D  => emit_op_vec_fmin,
        VecFNeg_S  | VecFNeg_D  => emit_op_vec_fneg,
        VecFAbs_S  | VecFAbs_D  => emit_op_vec_fabs,
        VecFSqrt_S | VecFSqrt_D => emit_op_vec_fsqrt,

        VecSaddl => emit_op_vec_addl_signed,
        VecUaddl => emit_op_vec_addl_unsigned,
        VecXtn   => emit_op_vec_xtn,
        VecTbl   => emit_op_vec_tbl,

        Hint | MemoryBarrier => emit_nop,
        Clrex => emit_op_clrex,

        Mrs => emit_op_mrs,
        Msr => emit_op_msr,

        _ => return None,
    })
}

// ─── Per-op adapter functions ────────────────────────────────────────────

fn emit_nop(_: &mut CodeAssembler, _: &Block, _: &Allocation, _: usize) -> Result<()> { Ok(()) }

fn emit_op_identity(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = match dst_of(&a, idx) { Some(d) => d, None => return Ok(()) };
    if alloc.loc(a.args[0]) == alloc.loc(d) { return Ok(()); }
    if a.ty.bits() <= 32 {
        load32(asm, alloc, a.args[0], eax)?;
        store32(asm, alloc, d, eax)?;
    } else {
        load64(asm, alloc, a.args[0], SCRATCH0)?;
        store64(asm, alloc, d, SCRATCH0)?;
    }
    Ok(())
}

fn emit_op_const_u32(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(eax, (a.imm as u32) as i32)?;
    store32(asm, alloc, d, eax)?;
    Ok(())
}
fn emit_op_const_u64(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(SCRATCH0, a.imm as i64)?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}

fn emit_op_get_x(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    load_guest_x(asm, SCRATCH0, a.imm as usize)?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}
fn emit_op_get_w(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    load_guest_x(asm, SCRATCH0, a.imm as usize)?;
    store32(asm, alloc, d, eax)?;
    Ok(())
}
fn emit_op_set_x(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    store_guest_x(asm, a.imm as usize, SCRATCH0)?;
    Ok(())
}
fn emit_op_set_w(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    load32(asm, alloc, a.args[0], eax)?;
    store_guest_x(asm, a.imm as usize, SCRATCH0)?;
    Ok(())
}
fn emit_op_get_sp(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(SCRATCH0, qword_ptr(CTX_REG + cpu_offsets::sp() as i32))?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}
fn emit_op_set_sp(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, _idx: usize) -> Result<()> {
    let a = block.code[_idx];
    load64(asm, alloc, a.args[0], SCRATCH0)?;
    asm.mov(qword_ptr(CTX_REG + cpu_offsets::sp() as i32), SCRATCH0)?;
    Ok(())
}
fn emit_op_get_nzcv(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.movzx(eax, byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32))?;
    store32(asm, alloc, d, eax)?;
    Ok(())
}
fn emit_op_set_nzcv(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    load32(asm, alloc, a.args[0], eax)?;
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::nzcv() as i32), al)?;
    Ok(())
}
fn emit_op_get_pc(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    asm.mov(SCRATCH0, a.imm as i64)?;
    store64(asm, alloc, d, SCRATCH0)?;
    Ok(())
}

fn emit_op_get_v(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let off = cpu_offsets::vreg(a.imm as usize) as i32;
    match a.ty {
        Ty::U32 => {
            asm.mov(eax, dword_ptr(CTX_REG + off))?;
            store32(asm, alloc, d, eax)?;
        }
        Ty::U64 => {
            asm.mov(SCRATCH0, qword_ptr(CTX_REG + off))?;
            store64(asm, alloc, d, SCRATCH0)?;
        }
        Ty::U128 => {
            asm.movdqu(xmm0, xmmword_ptr(CTX_REG + off))?;
            store_xmm_q(asm, alloc, d, xmm0)?;
        }
        other => return Err(Error::Backend(format!("GetV with unsupported ty {:?}", other))),
    }
    Ok(())
}
fn emit_op_set_v(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let off = cpu_offsets::vreg(a.imm as usize) as i32;
    let src_ty = block.code[a.args[0].as_usize()].ty;
    match src_ty {
        Ty::U32 => {
            load32(asm, alloc, a.args[0], eax)?;
            asm.mov(dword_ptr(CTX_REG + off), eax)?;
            asm.mov(dword_ptr(CTX_REG + off + 4), 0i32)?;
            asm.mov(qword_ptr(CTX_REG + off + 8), 0i32)?;
        }
        Ty::U64 => {
            load64(asm, alloc, a.args[0], SCRATCH0)?;
            asm.mov(qword_ptr(CTX_REG + off), SCRATCH0)?;
            asm.mov(qword_ptr(CTX_REG + off + 8), 0i32)?;
        }
        Ty::U128 => {
            let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
            asm.movdqu(xmmword_ptr(CTX_REG + off), src)?;
        }
        other => return Err(Error::Backend(format!("SetV with unsupported src ty {:?}", other))),
    }
    Ok(())
}

// ── Integer ALU adapters (binops, unops, shifts) ─────────────────────────
macro_rules! adapt_binop {
    ($name:ident, $kind:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_binop(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_binop!(emit_op_add, BinKind::Add);
adapt_binop!(emit_op_sub, BinKind::Sub);
adapt_binop!(emit_op_and, BinKind::And);
adapt_binop!(emit_op_or,  BinKind::Or);
adapt_binop!(emit_op_xor, BinKind::Xor);
adapt_binop!(emit_op_mul, BinKind::Imul);

macro_rules! adapt_adc {
    ($name:ident, $subtract:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_adc_sbc(asm, alloc, a, dst_of(&a, idx), $subtract, a.op.size_bits())
        }
    };
}
adapt_adc!(emit_op_adc, false);
adapt_adc!(emit_op_sbc, true);

macro_rules! adapt_div {
    ($name:ident, $signed:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_div(asm, alloc, a, dst_of(&a, idx), $signed, a.op.size_bits())
        }
    };
}
adapt_div!(emit_op_udiv, false);
adapt_div!(emit_op_sdiv, true);

macro_rules! adapt_unop_count {
    ($name:ident, $emit:ident) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            $emit(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
        }
    };
}
adapt_unop_count!(emit_op_clz,  emit_clz);
adapt_unop_count!(emit_op_cls,  emit_cls);
adapt_unop_count!(emit_op_rbit, emit_rbit);

fn emit_op_rev16(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let bits = if a.ty == Ty::U64 { 64 } else { 32 };
    emit_rev16(asm, alloc, a, dst_of(&a, idx), bits)
}
fn emit_op_rev32(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_rev32_within64(asm, alloc, a, dst_of(&a, idx))
}
fn emit_op_rev64(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let bits = if a.ty == Ty::U64 { 64 } else { 32 };
    emit_bswap(asm, alloc, a, dst_of(&a, idx), bits)
}

macro_rules! adapt_shift {
    ($name:ident, $kind:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_shift(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_shift!(emit_op_lsl, ShiftKind::Lsl);
adapt_shift!(emit_op_lsr, ShiftKind::Lsr);
adapt_shift!(emit_op_asr, ShiftKind::Asr);
adapt_shift!(emit_op_ror, ShiftKind::Ror);

macro_rules! adapt_unop_simple {
    ($name:ident, $kind:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_unop(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_unop_simple!(emit_op_not, UnopKind::Not);
adapt_unop_simple!(emit_op_neg, UnopKind::Neg);

fn emit_op_flagged_addsub(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_flagged_addsub(asm, alloc, a, dst_of(&a, idx))
}

// ── Memory adapters ──────────────────────────────────────────────────────
fn emit_op_load(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_load(asm, alloc, a, dst_of(&a, idx), a.op.size_bytes())
}
fn emit_op_store(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_store(asm, alloc, a, a.op.size_bytes())
}
fn emit_op_load_ex(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_load_ex(asm, alloc, a, dst_of(&a, idx), a.op.size_bytes())
}
fn emit_op_store_ex(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_store_ex(asm, alloc, a, dst_of(&a, idx), a.op.size_bytes())
}

fn emit_op_csel(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_csel(asm, alloc, a, dst_of(&a, idx))
}

// ── FP scalar adapters ───────────────────────────────────────────────────
macro_rules! adapt_fbinop {
    ($name:ident, $kind:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_fbinop(asm, alloc, a, dst_of(&a, idx), $kind, a.op.size_bits())
        }
    };
}
adapt_fbinop!(emit_op_fadd, FpBinKind::Add);
adapt_fbinop!(emit_op_fsub, FpBinKind::Sub);
adapt_fbinop!(emit_op_fmul, FpBinKind::Mul);
adapt_fbinop!(emit_op_fdiv, FpBinKind::Div);
adapt_fbinop!(emit_op_fmax, FpBinKind::Max);
adapt_fbinop!(emit_op_fmin, FpBinKind::Min);

fn emit_op_fcmp(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_fcmp(asm, alloc, a, a.op.size_bits())
}
fn emit_op_fsqrt_(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_fsqrt(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
}
fn emit_op_fneg_(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_fneg(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
}
fn emit_op_fabs_(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_fabs(asm, alloc, a, dst_of(&a, idx), a.op.size_bits())
}

macro_rules! adapt_fcvt_zs {
    ($name:ident, $double:expr, $to_x:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_fcvt_zs(asm, alloc, a, dst_of(&a, idx), $double, $to_x)
        }
    };
}
adapt_fcvt_zs!(emit_op_fcvt_zs_sw, false, false);
adapt_fcvt_zs!(emit_op_fcvt_zs_sx, false, true);
adapt_fcvt_zs!(emit_op_fcvt_zs_dw, true,  false);
adapt_fcvt_zs!(emit_op_fcvt_zs_dx, true,  true);

macro_rules! adapt_scvtf {
    ($name:ident, $double:expr, $from_x:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_scvtf(asm, alloc, a, dst_of(&a, idx), $double, $from_x)
        }
    };
}
adapt_scvtf!(emit_op_scvtf_ws, false, false);
adapt_scvtf!(emit_op_scvtf_xs, false, true);
adapt_scvtf!(emit_op_scvtf_wd, true,  false);
adapt_scvtf!(emit_op_scvtf_xd, true,  true);

fn emit_op_fcvt_sd(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_fcvt_precision(asm, alloc, a, dst_of(&a, idx), false)
}
fn emit_op_fcvt_ds(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_fcvt_precision(asm, alloc, a, dst_of(&a, idx), true)
}

// ── Vector glue + per-lane ops ───────────────────────────────────────────
fn emit_op_vec_build_q(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let working = working_xmm_for(alloc, d, xmm0);
    load64(asm, alloc, a.args[0], rax)?;
    asm.movq(working, rax)?;
    load64(asm, alloc, a.args[1], rax)?;
    asm.pinsrq(working, rax, 1)?;
    store_xmm_q(asm, alloc, d, working)?;
    Ok(())
}
fn emit_op_vec_extract_lo64(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.movq(rax, src)?;
    store64(asm, alloc, d, rax)
}
fn emit_op_vec_extract_hi64(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrq(rax, src, 1)?;
    store64(asm, alloc, d, rax)
}
fn emit_op_vec_extract8(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrb(eax, src, a.imm as i32)?;
    store32(asm, alloc, d, eax)
}
fn emit_op_vec_extract16(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrw(eax, src, a.imm as i32)?;
    store32(asm, alloc, d, eax)
}
fn emit_op_vec_extract32(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src = get_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.pextrd(eax, src, a.imm as i32)?;
    store32(asm, alloc, d, eax)
}

fn emit_op_vec_add(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_vec_binop(asm, alloc, a, dst_of(&a, idx), VecBinKind::Add(a.op.size_log2()))
}
fn emit_op_vec_sub(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_vec_binop(asm, alloc, a, dst_of(&a, idx), VecBinKind::Sub(a.op.size_log2()))
}
macro_rules! adapt_vec_logic {
    ($name:ident, $kind:expr) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            emit_vec_binop(asm, alloc, a, dst_of(&a, idx), $kind)
        }
    };
}
adapt_vec_logic!(emit_op_vec_and, VecBinKind::And);
adapt_vec_logic!(emit_op_vec_orr, VecBinKind::Orr);
adapt_vec_logic!(emit_op_vec_eor, VecBinKind::Eor);
adapt_vec_logic!(emit_op_vec_bic, VecBinKind::Bic);
adapt_vec_logic!(emit_op_vec_orn, VecBinKind::Orn);

fn emit_op_vec_neg(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    // working = 0 - vn (per lane). pxor zeros working; psubX subtracts vn.
    asm.pxor(working, working)?;
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    match a.op.size_log2() {
        0 => asm.psubb(working, src)?,
        1 => asm.psubw(working, src)?,
        2 => asm.psubd(working, src)?,
        3 => asm.psubq(working, src)?,
        _ => unreachable!(),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_abs(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pabsb(working, src)?,
        1 => asm.pabsw(working, src)?,
        2 => asm.pabsd(working, src)?,
        _ => return Err(Error::Backend(format!("VecAbs lane {} not supported", a.op.size_log2()))),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_not(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    // working = vn ^ all-ones
    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.pcmpeqd(xmm1, xmm1)?;
    asm.pxor(working, xmm1)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_mul(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        1 => asm.pmullw(working, other)?,
        2 => asm.pmulld(working, other)?,
        _ => return Err(Error::Backend(format!("VecMul lane {} not supported", a.op.size_log2()))),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_shl_imm(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let shift = (a.imm >> 1) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match a.op.size_log2() {
        1 => asm.psllw(working, shift as i32)?,
        2 => asm.pslld(working, shift as i32)?,
        3 => asm.psllq(working, shift as i32)?,
        _ => return Err(Error::Backend(format!("VecShlImm lane {} not supported", a.op.size_log2()))),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_ushr_imm(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let shift = (a.imm >> 1) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match a.op.size_log2() {
        1 => asm.psrlw(working, shift as i32)?,
        2 => asm.psrld(working, shift as i32)?,
        3 => asm.psrlq(working, shift as i32)?,
        _ => return Err(Error::Backend(format!("VecUshrImm lane {} not supported", a.op.size_log2()))),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_sshr_imm(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let shift = (a.imm >> 1) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match a.op.size_log2() {
        1 => asm.psraw(working, shift as i32)?,
        2 => asm.psrad(working, shift as i32)?,
        _ => return Err(Error::Backend(format!("VecSshrImm lane {} not supported (no PSRAQ pre-AVX-512)", a.op.size_log2()))),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

// ── Compare ops ──────────────────────────────────────────────────────────
fn emit_op_vec_cmeq(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pcmpeqb(working, other)?,
        1 => asm.pcmpeqw(working, other)?,
        2 => asm.pcmpeqd(working, other)?,
        3 => asm.pcmpeqq(working, other)?,
        _ => unreachable!(),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_cmgt(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pcmpgtb(working, other)?,
        1 => asm.pcmpgtw(working, other)?,
        2 => asm.pcmpgtd(working, other)?,
        3 => asm.pcmpgtq(working, other)?,
        _ => unreachable!(),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

/// CMGE Vd, Vn, Vm  ⇒  Vd = ~(Vm >s Vn). Compute pcmpgt(Vm, Vn), then
/// invert with all-ones XOR.
fn emit_op_vec_cmge(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    // Load Vm into working (it gets the >-vs-Vn applied).
    into_xmm_q(asm, alloc, a.args[1], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    match a.op.size_log2() {
        0 => asm.pcmpgtb(working, vn)?,
        1 => asm.pcmpgtw(working, vn)?,
        2 => asm.pcmpgtd(working, vn)?,
        3 => asm.pcmpgtq(working, vn)?,
        _ => unreachable!(),
    }
    asm.pcmpeqd(xmm2, xmm2)?;
    asm.pxor(working, xmm2)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

/// Unsigned a > b  ⇔  signed (a ^ sign) > (b ^ sign). We materialize the
/// sign mask per lane via `pcmpeqd; psll<X>` and apply pxor before the
/// signed compare.
fn emit_unsigned_cmp(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    invert: bool, // true for >=, false for >
) -> Result<()> {
    let q_form = (a.imm & 1) != 0;
    let lane = a.op.size_log2();
    let working = working_xmm_for(alloc, d, xmm0);

    // Build the per-lane sign-bit mask in xmm2.
    asm.pcmpeqd(xmm2, xmm2)?;
    let lane_bits_minus_1: i32 = ((8 << lane) - 1) as i32;
    match lane {
        1 => asm.psllw(xmm2, lane_bits_minus_1)?,
        2 => asm.pslld(xmm2, lane_bits_minus_1)?,
        3 => asm.psllq(xmm2, lane_bits_minus_1)?,
        _ => return Err(Error::Backend(format!("unsigned cmp lane {} not supported", lane))),
    }

    // working = (a ^ sign), other = (b ^ sign). For invert=true (CMHS) we
    // compute (b > a) and flip, so we swap which arg gets loaded into working.
    let (work_src, other_src) = if invert { (a.args[1], a.args[0]) } else { (a.args[0], a.args[1]) };
    into_xmm_q(asm, alloc, work_src, working)?;
    asm.pxor(working, xmm2)?;

    // Load other into xmm1, apply sign flip.
    let other_src_xmm = get_xmm_q(asm, alloc, other_src, xmm1)?;
    if other_src_xmm != xmm1 {
        asm.movdqa(xmm1, other_src_xmm)?;
    }
    asm.pxor(xmm1, xmm2)?;

    match lane {
        1 => asm.pcmpgtw(working, xmm1)?,
        2 => asm.pcmpgtd(working, xmm1)?,
        3 => asm.pcmpgtq(working, xmm1)?,
        _ => unreachable!(),
    }
    if invert {
        asm.pcmpeqd(xmm2, xmm2)?;
        asm.pxor(working, xmm2)?;
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_cmhi(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_unsigned_cmp(asm, alloc, a, d, false)
}
fn emit_op_vec_cmhs(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_unsigned_cmp(asm, alloc, a, d, true)
}

// ── Bit-select ───────────────────────────────────────────────────────────
// args[0] = vd_prev, args[1] = vn, args[2] = vm.
fn emit_op_vec_bit(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    // result = (vd & ~vm) | (vn & vm) = pandn(vm, vd) | (vn & vm)
    // Plan: working = vm; xmm1 = vn AND vm; then working = pandn(working, vd) = ~vm & vd; then por.
    into_xmm_q(asm, alloc, a.args[2], working)?;             // working = vm
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.movdqa(xmm2, working)?;                              // xmm2 = vm
    asm.pand(xmm2, vn)?;                                     // xmm2 = vn & vm
    let vd = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    asm.pandn(working, vd)?;                                 // working = ~vm & vd
    asm.por(working, xmm2)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_bif(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    // result = (vd & vm) | (vn & ~vm) = (vd & vm) | pandn(vm, vn)
    into_xmm_q(asm, alloc, a.args[2], working)?;             // working = vm
    let vd = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    asm.movdqa(xmm2, working)?;                              // xmm2 = vm
    asm.pand(xmm2, vd)?;                                     // xmm2 = vd & vm
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.pandn(working, vn)?;                                 // working = ~vm & vn
    asm.por(working, xmm2)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_bsl(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    // result = (vn & vd) | (vm & ~vd) = (vn & vd) | pandn(vd, vm)
    into_xmm_q(asm, alloc, a.args[0], working)?;             // working = vd
    let vn = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.movdqa(xmm2, working)?;                              // xmm2 = vd
    asm.pand(xmm2, vn)?;                                     // xmm2 = vn & vd
    let vm = get_xmm_q(asm, alloc, a.args[2], xmm1)?;
    asm.pandn(working, vm)?;                                 // working = ~vd & vm
    asm.por(working, xmm2)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

// ── DUP from GPR (broadcast scalar to all lanes) ─────────────────────────
fn emit_op_vec_dup_gpr(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let lane = a.op.size_log2();
    let working = working_xmm_for(alloc, d, xmm0);

    match lane {
        0 => {
            // 8-bit broadcast: movd + pshufb with zero mask.
            load32(asm, alloc, a.args[0], eax)?;
            asm.movd(working, eax)?;
            asm.pxor(xmm1, xmm1)?;
            asm.pshufb(working, xmm1)?;
        }
        1 => {
            // 16-bit broadcast: movd + pshuflw + pshufd.
            load32(asm, alloc, a.args[0], eax)?;
            asm.movd(working, eax)?;
            asm.pshuflw(working, working, 0)?;
            asm.pshufd(working, working, 0)?;
        }
        2 => {
            // 32-bit broadcast: movd + pshufd.
            load32(asm, alloc, a.args[0], eax)?;
            asm.movd(working, eax)?;
            asm.pshufd(working, working, 0)?;
        }
        3 => {
            // 64-bit broadcast: movq + punpcklqdq.
            load64(asm, alloc, a.args[0], rax)?;
            asm.movq(working, rax)?;
            asm.punpcklqdq(working, working)?;
        }
        _ => unreachable!(),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

// ── EXT (palignr) ────────────────────────────────────────────────────────
fn emit_op_vec_ext(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let byte_off = (a.imm >> 1) as i32;
    let working = working_xmm_for(alloc, d, xmm0);
    // ARM EXT result = bytes [byte_off .. byte_off+16) of concat(Vm, Vn) where
    // Vm is the HIGH 128. x86 PALIGNR dst, src, imm shifts {dst:src} right by
    // `imm` bytes and keeps the low 128. So we load Vm into working (the
    // "dst" of palignr → high 128) and Vn becomes the src (low 128).
    into_xmm_q(asm, alloc, a.args[1], working)?;
    let vn = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    asm.palignr(working, vn, byte_off)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

// ── ZIP1 / ZIP2 (punpcklXX / punpckhXX) ──────────────────────────────────
fn emit_op_vec_zip1(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    match a.op.size_log2() {
        0 => asm.punpcklbw (working, other)?,
        1 => asm.punpcklwd (working, other)?,
        2 => asm.punpckldq (working, other)?,
        3 => asm.punpcklqdq(working, other)?,
        _ => unreachable!(),
    }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_zip2(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    if q_form {
        // ZIP2 = interleave the HIGH halves. PUNPCKH on 128-bit registers does
        // exactly that.
        match a.op.size_log2() {
            0 => asm.punpckhbw (working, other)?,
            1 => asm.punpckhwd (working, other)?,
            2 => asm.punpckhdq (working, other)?,
            3 => asm.punpckhqdq(working, other)?,
            _ => unreachable!(),
        }
    } else {
        // For the 64-bit form, the "halves" we want to interleave are the
        // upper 32 bits of each source — i.e. bytes 4..8. The standard fix
        // is to shift each source right by 4 bytes (psrldq 4), then PUNPCKL
        // at the requested lane size.
        asm.psrldq(working, 4)?;
        asm.movdqa(xmm2, other)?;
        asm.psrldq(xmm2, 4)?;
        match a.op.size_log2() {
            0 => asm.punpcklbw(working, xmm2)?,
            1 => asm.punpcklwd(working, xmm2)?,
            2 => asm.punpckldq(working, xmm2)?,
            _ => return Err(Error::Backend("ZIP2 64-bit lane requires Q form".into())),
        }
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)
}

// ── SMIN/SMAX/UMIN/UMAX ──────────────────────────────────────────────────
macro_rules! emit_vec_minmax {
    ($fn_name:ident, $b:ident, $w:ident, $d:ident) => {
        fn $fn_name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            let d = dst_of(&a, idx).unwrap();
            let q_form = (a.imm & 1) != 0;
            let working = working_xmm_for(alloc, d, xmm0);
            into_xmm_q(asm, alloc, a.args[0], working)?;
            let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
            match a.op.size_log2() {
                0 => asm.$b(working, other)?,
                1 => asm.$w(working, other)?,
                2 => asm.$d(working, other)?,
                _ => return Err(Error::Backend(format!("{} lane {} unsupported", stringify!($fn_name), a.op.size_log2()))),
            }
            if !q_form { asm.movq(working, working)?; }
            store_xmm_q(asm, alloc, d, working)
        }
    };
}
emit_vec_minmax!(emit_op_vec_smin, pminsb, pminsw, pminsd);
emit_vec_minmax!(emit_op_vec_smax, pmaxsb, pmaxsw, pmaxsd);
emit_vec_minmax!(emit_op_vec_umin, pminub, pminuw, pminud);
emit_vec_minmax!(emit_op_vec_umax, pmaxub, pmaxuw, pmaxud);

// ── Per-lane FP ──────────────────────────────────────────────────────────
#[inline]
fn vec_fp_is_double(op: Op) -> bool { (op as u16 & 1) != 0 }

macro_rules! emit_vec_fbin {
    ($name:ident, $ps:ident, $pd:ident) => {
        fn $name(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
            let a = block.code[idx];
            let d = dst_of(&a, idx).unwrap();
            let q_form = (a.imm & 1) != 0;
            let double = vec_fp_is_double(a.op);
            let working = working_xmm_for(alloc, d, xmm0);
            into_xmm_q(asm, alloc, a.args[0], working)?;
            let other = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
            if double { asm.$pd(working, other)?; } else { asm.$ps(working, other)?; }
            if !q_form { asm.movq(working, working)?; }
            store_xmm_q(asm, alloc, d, working)
        }
    };
}
emit_vec_fbin!(emit_op_vec_fadd, addps, addpd);
emit_vec_fbin!(emit_op_vec_fsub, subps, subpd);
emit_vec_fbin!(emit_op_vec_fmul, mulps, mulpd);
emit_vec_fbin!(emit_op_vec_fdiv, divps, divpd);
emit_vec_fbin!(emit_op_vec_fmax, maxps, maxpd);
emit_vec_fbin!(emit_op_vec_fmin, minps, minpd);

fn emit_op_vec_fneg(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    // Build sign-bit mask per lane in xmm1, then XOR.
    asm.pcmpeqd(xmm1, xmm1)?;
    if double { asm.psllq(xmm1, 63)?; } else { asm.pslld(xmm1, 31)?; }
    asm.pxor(working, xmm1)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fabs(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;
    // Build abs mask per lane in xmm1 (clear sign bit), then AND.
    asm.pcmpeqd(xmm1, xmm1)?;
    if double { asm.psrlq(xmm1, 1)?; } else { asm.psrld(xmm1, 1)?; }
    asm.pand(working, xmm1)?;
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_fsqrt(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let double = vec_fp_is_double(a.op);
    let working = working_xmm_for(alloc, d, xmm0);
    let src = get_xmm_q(asm, alloc, a.args[0], xmm1)?;
    if double { asm.sqrtpd(working, src)?; } else { asm.sqrtps(working, src)?; }
    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

// ── Widening add ──────────────────────────────────────────────────────────
fn emit_widening_addl(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    d: ValueRef,
    signed: bool,
) -> Result<()> {
    let high_half = ((a.imm >> 1) & 1) != 0;
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let working = working_xmm_for(alloc, d, xmm0);

    // Get both sources into scratch XMMs; if high_half, shift each right by
    // 8 bytes so the high half is now in the low half (ready for pmovsxXX).
    into_xmm_q(asm, alloc, a.args[0], working)?;
    let other_src = get_xmm_q(asm, alloc, a.args[1], xmm1)?;
    if other_src != xmm1 { asm.movdqa(xmm1, other_src)?; }

    if high_half {
        asm.psrldq(working, 8)?;
        asm.psrldq(xmm1, 8)?;
    }

    // Widen each source to the result lane in place.
    if signed {
        match src_lane {
            0 => { asm.pmovsxbw(working, working)?; asm.pmovsxbw(xmm1, xmm1)?; asm.paddw(working, xmm1)?; }
            1 => { asm.pmovsxwd(working, working)?; asm.pmovsxwd(xmm1, xmm1)?; asm.paddd(working, xmm1)?; }
            2 => { asm.pmovsxdq(working, working)?; asm.pmovsxdq(xmm1, xmm1)?; asm.paddq(working, xmm1)?; }
            _ => return Err(Error::Backend(format!("VecSaddl lane {} not supported", src_lane))),
        }
    } else {
        match src_lane {
            0 => { asm.pmovzxbw(working, working)?; asm.pmovzxbw(xmm1, xmm1)?; asm.paddw(working, xmm1)?; }
            1 => { asm.pmovzxwd(working, working)?; asm.pmovzxwd(xmm1, xmm1)?; asm.paddd(working, xmm1)?; }
            2 => { asm.pmovzxdq(working, working)?; asm.pmovzxdq(xmm1, xmm1)?; asm.paddq(working, xmm1)?; }
            _ => return Err(Error::Backend(format!("VecUaddl lane {} not supported", src_lane))),
        }
    }
    store_xmm_q(asm, alloc, d, working)
}

fn emit_op_vec_addl_signed(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_addl(asm, alloc, a, d, true)
}
fn emit_op_vec_addl_unsigned(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    emit_widening_addl(asm, alloc, a, d, false)
}

// ── TBL (single-register table permute) ──────────────────────────────────
//
// x86 PSHUFB zeroes destination bytes only when the index byte's bit 7 is
// set; for indices 16..127 it does (index & 0x0F) instead of zeroing, which
// disagrees with ARM TBL ("any index >= 16 → 0"). We fix this by saturating-
// adding 0x70 to each index byte first: now any vm >= 16 has bit 7 set
// (16+0x70 = 0x80) and PSHUFB zeroes those lanes.
fn emit_op_vec_tbl(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let q_form = (a.imm & 1) != 0;
    let working = working_xmm_for(alloc, d, xmm0);

    // Build per-byte 0x70 mask in xmm2.
    asm.mov(rax, 0x7070_7070_7070_7070u64 as i64)?;
    asm.movq(xmm2, rax)?;
    asm.punpcklqdq(xmm2, xmm2)?;

    // Modified indices in xmm1 = vm + 0x70 saturating.
    into_xmm_q(asm, alloc, a.args[1], xmm1)?;
    asm.paddusb(xmm1, xmm2)?;

    // Table in working; shuffle.
    into_xmm_q(asm, alloc, a.args[0], working)?;
    asm.pshufb(working, xmm1)?;

    if !q_form { asm.movq(working, working)?; }
    store_xmm_q(asm, alloc, d, working)
}

// ── Narrowing truncate (XTN) ─────────────────────────────────────────────
fn emit_op_vec_xtn(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let src_lane = ((a.imm >> 2) & 0x3) as u32;
    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, a.args[0], working)?;

    match src_lane {
        1 => {
            // H -> B: mask low byte of each H lane, then packuswb (no
            // saturation since values are <= 0xFF).
            asm.pcmpeqd(xmm1, xmm1)?;
            asm.psrlw(xmm1, 8)?;           // each H lane = 0x00FF
            asm.pand(working, xmm1)?;
            asm.packuswb(working, working)?;
        }
        2 => {
            // S -> H: mask low halfword, then packusdw (SSE4.1).
            asm.pcmpeqd(xmm1, xmm1)?;
            asm.psrld(xmm1, 16)?;
            asm.pand(working, xmm1)?;
            asm.packusdw(working, working)?;
        }
        3 => {
            // D -> S: pshufd to pick low 32 of each D lane.
            // imm 0b00_00_10_00 = 0x08 selects dwords [0, 2, 0, 0].
            asm.pshufd(working, working, 0x08)?;
        }
        _ => return Err(Error::Backend(format!("VecXtn src lane {} not supported", src_lane))),
    }
    asm.movq(working, working)?; // zero upper 64
    store_xmm_q(asm, alloc, d, working)
}

// ── ADDV.4S (horizontal sum) ─────────────────────────────────────────────
fn emit_op_vec_addv32(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    // working = Vn; two phaddd reduce 4 lanes to one in lane 0.
    into_xmm_q(asm, alloc, a.args[0], xmm0)?;
    asm.phaddd(xmm0, xmm0)?;
    asm.phaddd(xmm0, xmm0)?;
    asm.movd(eax, xmm0)?;
    store32(asm, alloc, d, eax)
}

// ── INS from GPR (write one lane) ───────────────────────────────────────
fn emit_op_vec_ins_gpr(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    let d = dst_of(&a, idx).unwrap();
    let lane_idx = (a.imm >> 1) as i32;
    let lane = a.op.size_log2();
    let working = working_xmm_for(alloc, d, xmm0);
    // Start with vd_prev in working; insert the new lane value.
    into_xmm_q(asm, alloc, a.args[0], working)?;
    match lane {
        0 => {
            load32(asm, alloc, a.args[1], eax)?;
            asm.pinsrb(working, eax, lane_idx)?;
        }
        1 => {
            load32(asm, alloc, a.args[1], eax)?;
            asm.pinsrw(working, eax, lane_idx)?;
        }
        2 => {
            load32(asm, alloc, a.args[1], eax)?;
            asm.pinsrd(working, eax, lane_idx)?;
        }
        3 => {
            load64(asm, alloc, a.args[1], rax)?;
            asm.pinsrq(working, rax, lane_idx)?;
        }
        _ => unreachable!(),
    }
    store_xmm_q(asm, alloc, d, working)
}

// ── System / misc adapters ───────────────────────────────────────────────
fn emit_op_clrex(asm: &mut CodeAssembler, _block: &Block, _alloc: &Allocation, _idx: usize) -> Result<()> {
    asm.mov(byte_ptr(CTX_REG + cpu_offsets::exclusive_size() as i32), 0i32)?;
    Ok(())
}
fn emit_op_mrs(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_mrs(asm, alloc, a, dst_of(&a, idx))
}
fn emit_op_msr(asm: &mut CodeAssembler, block: &Block, alloc: &Allocation, idx: usize) -> Result<()> {
    let a = block.code[idx];
    emit_msr(asm, alloc, a)
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

/// FNEG via GPR `xor` against the sign bit. One instruction for 32-bit
/// (imm32 fits); 64-bit needs a `mov rcx, imm64; xor rax, rcx` pair since
/// `xor r64, imm32` would sign-extend. No XMM roundtrip — the value
/// usually lives in a GPR already.
fn emit_fneg(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, 0x8000_0000_0000_0000_u64 as i64)?;
        asm.xor(rax, rcx)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.xor(eax, 0x8000_0000_u32 as i32)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
    }
    Ok(())
}

/// FABS via GPR `and` clearing the sign bit. Same shape as FNEG.
fn emit_fabs(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    bits: u32,
) -> Result<()> {
    if bits == 64 {
        load64(asm, alloc, a.args[0], rax)?;
        asm.mov(rcx, 0x7FFF_FFFF_FFFF_FFFF_u64 as i64)?;
        asm.and(rax, rcx)?;
        if let Some(d) = dst { store64(asm, alloc, d, rax)?; }
    } else {
        load32(asm, alloc, a.args[0], eax)?;
        asm.and(eax, 0x7FFF_FFFF_u32 as i32)?;
        if let Some(d) = dst { store32(asm, alloc, d, eax)?; }
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

#[derive(Clone, Copy)]
enum VecBinKind {
    Add(u32),    // lane log2 byte size
    Sub(u32),
    And, Orr, Eor, Bic, Orn,
}

/// Apply a 128-bit XMM binop, then mask off the high 64 bits when `q=0`.
///
/// Tries to compute in-place in dst's XMM (saving a movdqa) and to consume
/// each source straight from its allocator-chosen XMM (saving a load). xmm0
/// only gets used as a fallback when dst spills or a source spills.
fn emit_vec_binop(
    asm: &mut CodeAssembler,
    alloc: &Allocation,
    a: Armlet,
    dst: Option<ValueRef>,
    kind: VecBinKind,
) -> Result<()> {
    let d = dst.unwrap();
    let q_form = (a.imm & 1) != 0;

    // BIC and ORN are non-commutative against the regular pandn/por-not pattern,
    // so the operand the working register needs to hold is different:
    //   - BIC:  working = Vm   (so PANDN working, Vn → ~Vm & Vn = Vn & ~Vm) ✓
    //   - ORN:  working = Vn   (we'll NOT a scratch copy of Vm and OR into working)
    //   - rest: working = Vn   (so OP working, Vm)
    let (working_src, other_src) = match kind {
        VecBinKind::Bic => (a.args[1], a.args[0]),
        _               => (a.args[0], a.args[1]),
    };

    let working = working_xmm_for(alloc, d, xmm0);
    into_xmm_q(asm, alloc, working_src, working)?;
    let other = get_xmm_q(asm, alloc, other_src, xmm1)?;

    match kind {
        VecBinKind::Add(sz) => match sz {
            0 => asm.paddb(working, other)?,
            1 => asm.paddw(working, other)?,
            2 => asm.paddd(working, other)?,
            3 => asm.paddq(working, other)?,
            _ => unreachable!(),
        },
        VecBinKind::Sub(sz) => match sz {
            0 => asm.psubb(working, other)?,
            1 => asm.psubw(working, other)?,
            2 => asm.psubd(working, other)?,
            3 => asm.psubq(working, other)?,
            _ => unreachable!(),
        },
        VecBinKind::And => asm.pand (working, other)?,
        VecBinKind::Orr => asm.por  (working, other)?,
        VecBinKind::Eor => asm.pxor (working, other)?,
        VecBinKind::Bic => asm.pandn(working, other)?, // working=~Vm, other=Vn → Vn & ~Vm
        VecBinKind::Orn => {
            // working = Vn; we need Vn | ~Vm. Invert Vm into xmm2 then por.
            asm.pcmpeqd(xmm2, xmm2)?;        // all-ones
            // xmm2 ^= other  (= ~Vm). xor with a register source is fine.
            asm.pxor(xmm2, other)?;
            asm.por (working, xmm2)?;
        }
    }

    if !q_form {
        // 64-bit form: zero the upper 64 bits. movq xmm, xmm copies the low
        // 64 and clears the high 64.
        asm.movq(working, working)?;
    }
    store_xmm_q(asm, alloc, d, working)?;
    Ok(())
}
